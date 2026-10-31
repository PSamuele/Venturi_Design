"""
grid_study.py
=============
Grid refinement study: how much does the answer still change with the grid?

The same case is solved on several grids, each finer than the last by a
roughly constant factor, with the SAME shape of spacing (same wall
clustering, same throat refinement, same ramp width). Then, for every
quantity of interest (QUANTITIES):

    observed order p   how fast the result changes as cells shrink
    extrapolated value what the result would be with infinitely small cells
    estimated error    distance of the finest result from the extrapolated one
    GCI                an error band on the finest-grid result

The formulas follow Celik et al., "Procedure for estimation and reporting of
uncertainty due to discretization in CFD applications", J. Fluids Eng. 130
(2008) 078001. Index 1 is the finest grid, 3 the coarsest of a triplet.

    h_i   = N_i^(-1/2)                      (representative cell size, 2D)
    r21   = h2 / h1,   r32 = h3 / h2
    e21   = phi2 - phi1,   e32 = phi3 - phi2,   s = sign(e32 / e21)
    p     = | ln|e32 / e21| + q(p) | / ln r21
    q(p)  = ln( (r21^p - s) / (r32^p - s) )      (solved by fixed point)
    phi_ext = (r21^p phi1 - phi2) / (r21^p - 1)
    GCI21 = 1.25 * |(phi1 - phi2) / phi1| / (r21^p - 1)

The bars in p hide a case the paper does not discuss: changes that GROW with
refinement also give a positive p. So the sign is checked before taking the
absolute value, and only a converging triplet gets an error estimate:

    converging    e21 and e32 agree in sign and the changes shrink
    oscillatory   s < 0: the result goes up and down
    diverging     the changes grow as cells shrink
    flat          no change at all: already grid independent to round-off
    unresolved    p cannot be found (one change is exactly zero, or the
                  iteration on p does not settle)

Absolute pressures are not in QUANTITIES: a percentage of 100 kPa makes any
error look small. Pressure DIFFERENCES are used instead.

Usage
-----
    python main.py --grid-study                          # after a normal run
    python main.py --grid-study --grid-levels 90x36,128x51,180x72
    python -m venturi.grid_study                         # only the study
    python -m venturi.grid_study --beta 0.6 --jobs 2     # any main.py option
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

DEFAULT_TOL = 1e-4   # tighter than a normal run, so the iteration error is
                     # small next to the grid error being measured
REFINEMENT = math.sqrt(2.0)   # between the default levels
MIN_RATIO = 1.3               # Celik et al.: smaller steps drown in other errors
FORMAL_ORDER = 2.0            # order of the scheme on smooth flow


class Quantity(NamedTuple):
    key: str
    label: str
    unit: str
    fmt: str


QUANTITIES = (
    Quantity("Cd", "discharge coefficient C_d", "-", ".5f"),
    Quantity("dp", "dp between the wall taps", "Pa", ".1f"),
    Quantity("dp_section", "dp of section averages", "Pa", ".1f"),
    Quantity("recovery_pct", "pressure recovered", "%", ".2f"),
    Quantity("dp_loss", "permanent pressure loss", "Pa", ".1f"),
    Quantity("loss_coef", "loss coefficient K", "-", ".3f"),
    Quantity("dp_max", "inlet minus lowest pressure", "Pa", ".1f"),
    Quantity("u_max", "highest speed", "m/s", ".3f"),
)

BEHAVIOUR = {"converging": "converging", "flat": "no change",
             "oscillatory": "up and down", "diverging": "moving away",
             "unresolved": "no estimate"}


def values_of(rep) -> Dict[str, float]:
    """The QUANTITIES of a ValidationReport."""
    return dict(Cd=rep.Cd, dp=rep.dp_measured, dp_section=rep.dp_section_mean,
                recovery_pct=100.0 * rep.pressure_recovery, dp_loss=rep.dp_loss,
                loss_coef=rep.loss_coef, dp_max=rep.dp_max, u_max=rep.u_max)


# ---------------------------------------------------------------------------
# Richardson / GCI
# ---------------------------------------------------------------------------
@dataclass
class GCIResult:
    p: float                 # observed order
    phi_ext: float           # extrapolated value
    e_a21: float             # relative change finest -> middle grid
    e_ext21: float           # relative distance finest -> extrapolated
    gci21: float             # error band on the finest result (relative)
    status: str              # see the module docstring

    @property
    def oscillatory(self) -> bool:
        return self.status == "oscillatory"

    @property
    def usable(self) -> bool:
        """True when phi_ext and gci21 can be trusted as an estimate."""
        return self.status in ("converging", "flat")


def gci(h: Sequence[float], phi: Sequence[float], safety: float = 1.25,
        max_iter: int = 200) -> GCIResult:
    """Richardson extrapolation and GCI for three grids.

    Args:
        h:   cell sizes (h1 < h2 < h3), finest first.
        phi: results on those grids, finest first.
    """
    h1, h2, h3 = h
    f1, f2, f3 = phi
    if not (h1 < h2 < h3):
        raise ValueError("cell sizes must be given finest first, strictly increasing")
    nan = float("nan")
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    if not all(math.isfinite(v) for v in phi) or f1 == 0.0:
        return GCIResult(nan, nan, nan, nan, nan, "unresolved")
    e_a21 = abs(e21 / f1)
    if e21 == 0.0 and e32 == 0.0:
        return GCIResult(float("inf"), f1, 0.0, 0.0, 0.0, "flat")
    if e21 == 0.0 or e32 == 0.0:
        # one change vanished by chance: it says nothing about the other one
        return GCIResult(nan, nan, e_a21, nan, nan, "unresolved")
    s = math.copysign(1.0, e32 / e21)
    log_ratio = math.log(abs(e32 / e21))
    p = abs(log_ratio) / math.log(r21)                    # first guess (q = 0)
    try:
        for _ in range(max_iter):
            num = r21 ** p - s
            den = r32 ** p - s
            q = math.log(num / den) if num > 0.0 and den > 0.0 else 0.0
            signed = log_ratio + q
            p_new = abs(signed) / math.log(r21)
            if abs(p_new - p) < 1e-12:
                p = p_new
                break
            p = p_new
        else:
            return GCIResult(nan, nan, e_a21, nan, nan, "unresolved")
        rp = r21 ** p
    except OverflowError:
        return GCIResult(nan, nan, e_a21, nan, nan, "unresolved")
    if rp <= 1.0:
        return GCIResult(p, nan, e_a21, nan, nan, "unresolved")
    phi_ext = (rp * f1 - f2) / (rp - 1.0)
    e_ext21 = abs((phi_ext - f1) / phi_ext) if phi_ext != 0.0 else nan
    if s < 0.0:
        status = "oscillatory"
    elif signed < 0.0:
        status = "diverging"
    else:
        status = "converging"
    return GCIResult(p=p, phi_ext=phi_ext, e_a21=e_a21, e_ext21=e_ext21,
                     gci21=safety * e_a21 / (rp - 1.0), status=status)


def order_note(p: float) -> str:
    """What the observed order says about a converging quantity.

    Returns a short warning for the behaviour column, "" when p is believable.
    """
    # TODO(human): compare p with FORMAL_ORDER and warn when it is suspicious.
    return ""


# ---------------------------------------------------------------------------
# Choosing the grids
# ---------------------------------------------------------------------------
def default_levels(Nz: int, Nr: int) -> List[Tuple[int, int]]:
    """The given grid, plus one about sqrt(2) coarser and one finer per direction."""
    return [(math.ceil(Nz / REFINEMENT), math.ceil(Nr / REFINEMENT)), (Nz, Nr),
            (math.ceil(Nz * REFINEMENT), math.ceil(Nr * REFINEMENT))]


def parse_levels(text: str) -> List[Tuple[int, int]]:
    """'64x26,90x36' -> [(64, 26), (90, 36)], coarsest first."""
    levels = []
    for item in text.split(","):
        try:
            nz, nr = item.lower().split("x")
            levels.append((int(nz), int(nr)))
        except ValueError:
            raise ValueError(f"grid '{item}' is not in the form NzxNr, for example 90x36.")
    levels.sort(key=lambda t: t[0] * t[1])
    return levels


def study_levels(text: Optional[str], Nz: int, Nr: int) -> List[Tuple[int, int]]:
    """The grids of a study: the ones given, or the default ones around Nz x Nr."""
    levels = parse_levels(text) if text else default_levels(Nz, Nr)
    if len(levels) < 3:
        raise ValueError("a grid study needs at least three grids.")
    cells = [nz * nr for nz, nr in levels]
    if len(set(cells)) < len(cells):
        raise ValueError("every grid of a study needs a different number of cells.")
    return levels


# ---------------------------------------------------------------------------
# Running the grids
# ---------------------------------------------------------------------------
def run_level(args_dict: Dict, Nz: int, Nr: int, ramp: float) -> Dict:
    """Solve one grid level and return the numbers of interest."""
    from .cli import config_from_args
    from .fvmesh import build_fvmesh
    from .geometry import create_venturi_geometry
    from .solver import solve_fv
    from .validation import validate

    args = argparse.Namespace(**args_dict)
    args.Nz, args.Nr = Nz, Nr
    config = config_from_args(args)
    geom = create_venturi_geometry(config)
    mesh = build_fvmesh(geom, Nz, Nr, config.wall_clustering,
                        config.throat_refine, throat_ramp=ramp)
    t0 = time.time()
    res = solve_fv(mesh, config, verbose=False)
    rep = validate(res, mesh, geom, config)
    return dict(Nz=Nz, Nr=Nr, cells=Nz * Nr, iters=res.n_iter,
                converged=bool(res.converged), residual=float(res.final_residual),
                time_s=time.time() - t0, y_plus=float(res.y_plus_max),
                p_min=rep.p_min, **values_of(rep))


def _notes(triplet: List[Dict]) -> str:
    """What makes a triplet's estimate less reliable."""
    out = []
    for r in triplet:
        if not r["converged"]:
            out.append(f"{r['Nz']}x{r['Nr']} not converged")
        if r["y_plus"] >= 5.0:
            out.append(f"{r['Nz']}x{r['Nr']} y+ = {r['y_plus']:.1f} (> 5)")
    h = [r["cells"] ** -0.5 for r in triplet]
    for a, b in zip(h, h[1:]):
        if b / a < MIN_RATIO:
            out.append(f"refinement ratio {b / a:.2f} (< {MIN_RATIO})")
    return "; ".join(out) if out else "ok"


def _grid_name(r: Dict) -> str:
    return f"{r['Nz']}x{r['Nr']}"


def _triplet_rows(triplet: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """Markdown rows and CSV records of one triplet, one per quantity."""
    h = [r["cells"] ** -0.5 for r in triplet]
    name = ", ".join(_grid_name(r) for r in triplet)
    lines, records = [], []
    for q in QUANTITIES:
        phi = [r[q.key] for r in triplet]
        g = gci(h, phi)
        behaviour = BEHAVIOUR[g.status]
        record = dict(grids=name.replace(", ", " "), quantity=q.key, unit=q.unit,
                      finest=phi[0], extrapolated=g.phi_ext, change_pct=100 * g.e_a21,
                      error_pct=100 * g.e_ext21, gci_pct=100 * g.gci21,
                      band=g.gci21 * abs(phi[0]), order=g.p, behaviour=g.status)
        if g.usable:
            note = order_note(g.p) if g.status == "converging" else ""
            if note:
                behaviour += f", {note}"
            lines.append(f"| {q.label} [{q.unit}] | {phi[0]:{q.fmt}} | {g.phi_ext:{q.fmt}} | "
                         f"{100 * g.e_a21:.3f} | {100 * g.e_ext21:.3f} | {100 * g.gci21:.3f} | "
                         f"+/- {g.gci21 * abs(phi[0]):{q.fmt}} | {g.p:.2f} | {behaviour} |")
        else:
            lines.append(f"| {q.label} [{q.unit}] | {phi[0]:{q.fmt}} | - | "
                         f"{100 * g.e_a21:.3f} | - | - | - | - | {behaviour} |")
            for key in ("extrapolated", "error_pct", "gci_pct", "band", "order"):
                record[key] = ""
        records.append(record)
    return lines, records


def _this_run_lines(rows: List[Dict], base: Dict[str, float], base_name: str) -> List[str]:
    """The run the user asked for, against the finest triplet's extrapolation."""
    h = [r["cells"] ** -0.5 for r in rows[:3]]
    lines = ["## This run", "",
             f"The {base_name} run above, against the extrapolated values of the "
             f"finest three grids:", "",
             "| quantity | this run | extrapolated | estimated error of this run [%] |",
             "|---|---|---|---|"]
    for q in QUANTITIES:
        g = gci(h, [r[q.key] for r in rows[:3]])
        if g.usable and g.phi_ext != 0.0:
            err = 100 * abs((base[q.key] - g.phi_ext) / g.phi_ext)
            lines.append(f"| {q.label} [{q.unit}] | {base[q.key]:{q.fmt}} | "
                         f"{g.phi_ext:{q.fmt}} | {err:.3f} |")
        else:
            lines.append(f"| {q.label} [{q.unit}] | {base[q.key]:{q.fmt}} | - | - |")
    lines += ["", "A normal run picks its own throat ramp and stops at a looser residual, "
              "so its grid can differ a little from the study grid of the same size.", ""]
    return lines


def build_report(rows: List[Dict], case: str, time_step: str, tol: float, refine: float,
                 ramp: float, base: Optional[Dict[str, float]] = None,
                 base_name: str = "") -> Tuple[str, List[Dict]]:
    """Markdown report and CSV records of a study. rows are finest first."""
    h = [r["cells"] ** -0.5 for r in rows]
    lines = ["# Grid study", "",
             f"Case: `{case}`, time step `{time_step}`, residual target {tol:g}, "
             f"throat refinement {refine:.2f}, ramp {ramp:.3f} throat lengths.", "",
             "| grid | cells | h / h_finest | iterations | converged | residual | y+ | time [s] |",
             "|---|---|---|---|---|---|---|---|"]
    for r, hi in zip(rows, h):
        lines.append(f"| {r['Nz']} x {r['Nr']} | {r['cells']} | {hi / h[0]:.3f} | {r['iters']} | "
                     f"{'yes' if r['converged'] else 'NO'} | {r['residual']:.2e} | "
                     f"{r['y_plus']:.2f} | {r['time_s']:.0f} |")
    lines += ["", "## Values on each grid", "",
              "| quantity | " + " | ".join(_grid_name(r) for r in rows) + " |",
              "|---" * (len(rows) + 1) + "|"]
    for q in QUANTITIES:
        lines.append(f"| {q.label} [{q.unit}] | "
                     + " | ".join(f"{r[q.key]:{q.fmt}}" for r in rows) + " |")
    lines += [f"| lowest pressure [Pa] | " + " | ".join(f"{r['p_min']:.1f}" for r in rows) + " |",
              "", "## Error estimate", "",
              "Richardson extrapolation and GCI (Celik et al. 2008), on the finest "
              "grid of each group of three consecutive grids.", ""]
    records: List[Dict] = []
    for k in range(len(rows) - 2):
        triplet = rows[k:k + 3]
        table, recs = _triplet_rows(triplet)
        records += recs
        lines += [f"**{', '.join(_grid_name(r) for r in triplet)}** (notes: {_notes(triplet)})", "",
                  "| quantity | finest grid | extrapolated | last change [%] | "
                  "estimated error [%] | GCI [%] | band | order p | behaviour |",
                  "|---|---|---|---|---|---|---|---|---|", *table, ""]
    if base is not None:
        lines += _this_run_lines(rows, base, base_name)
    lines += ["Estimated error: distance of the finest grid from the extrapolated value. "
              "GCI: the grid-independent value is expected within +/- GCI of the finest "
              "grid, the same band in units in the band column. A row is only as reliable "
              "as its notes say: a grid that did not converge carries iteration error too, "
              "and y+ above 5 is outside what the wall models need. With four grids or "
              "more, compare p between the groups: if it moves, the grids are not yet in "
              "the range where the estimate holds."]
    return "\n".join(lines) + "\n", records


def _write_csv(path: str, records: List[Dict]) -> None:
    keys = list(records[0].keys())
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for r in records:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")


def run_study(args: argparse.Namespace, levels: List[Tuple[int, int]], jobs: int, tol: float,
              base: Optional[Dict[str, float]] = None, base_name: str = "",
              write_files: bool = True) -> str:
    """Solve the case on every grid, print the report and return it.

    Args:
        args: the options of the case, as parsed by cli.build_parser.
        levels: grids as (Nz, Nr), coarsest first.
        jobs: grids solved at the same time.
        tol: residual target of the study.
        base: values_of() a normal run, to report that run's own error.
        base_name: name of that run's grid.
        write_files: write grid_study.md and the two CSV files.
    """
    # Same ramp width on every grid: taken from the coarsest one, which
    # needs the widest ramp to keep its cells within 20 % of each other.
    from .cli import config_from_args
    from .fvmesh import throat_ramp_width
    from .geometry import create_venturi_geometry
    a0 = argparse.Namespace(**vars(args))
    a0.Nz, a0.Nr = levels[0]
    a0.tol = tol
    cfg0 = config_from_args(a0)
    geom0 = create_venturi_geometry(cfg0)
    zs = geom0.z_stations
    ramp = (throat_ramp_width(geom0.L_total, levels[0][0], zs[2], zs[3], cfg0.throat_refine)
            if cfg0.throat_refine > 1.0 else 0.5)

    print(f"Grid study: {', '.join(f'{nz}x{nr}' for nz, nr in levels)}; "
          f"tol {tol:g}, throat refinement {cfg0.throat_refine:.2f}, "
          f"ramp {ramp:.3f} throat lengths, {jobs} at a time.", flush=True)
    arg_dict = vars(a0)
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = [pool.submit(run_level, arg_dict, nz, nr, ramp) for nz, nr in levels]
        rows = []
        for f in as_completed(futures):
            r = f.result()
            rows.append(r)
            print(f"  {r['Nz']:4d} x {r['Nr']:3d}: {r['iters']:7d} iterations, "
                  f"{'converged' if r['converged'] else 'NOT converged'}, "
                  f"{r['time_s']:6.1f} s, residual {r['residual']:.2e}, "
                  f"C_d {r['Cd']:.5f}, y+ {r['y_plus']:.2f}", flush=True)

    rows.sort(key=lambda r: -r["cells"])          # finest first
    case = os.path.basename(cfg0.output_dir)
    text, records = build_report(rows, case, cfg0.time_step, tol, cfg0.throat_refine,
                                 ramp, base, base_name)
    print("\n" + text)
    if write_files:
        out_dir = os.path.join("results", "grid_study_" + case)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "grid_study.md"), "w", encoding="utf-8") as fh:
            fh.write(text)
        _write_csv(os.path.join(out_dir, "grid_study.csv"), rows)
        _write_csv(os.path.join(out_dir, "grid_study_gci.csv"), records)
        print(f"Written to {out_dir}{os.sep}")
    return text


def build_parser() -> argparse.ArgumentParser:
    from .cli import build_parser as case_parser
    p = case_parser()
    p.prog = "python -m venturi.grid_study"
    p.description = ("Solve one case on several grids and estimate how far the "
                     "finest result still is from the grid-independent value.")
    p.add_argument("--levels", dest="grid_levels", default=argparse.SUPPRESS,
                   help="same as --grid-levels.")
    p.add_argument("--jobs", dest="grid_jobs", type=int, default=argparse.SUPPRESS,
                   help="same as --grid-jobs.")
    p.set_defaults(tol=DEFAULT_TOL)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        levels = study_levels(args.grid_levels, args.Nz, args.Nr)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    run_study(args, levels, args.grid_jobs, args.tol, write_files=not args.no_export)
    return 0


if __name__ == "__main__":
    sys.exit(main())
