"""
grid_study.py
=============
Grid refinement study: how much does the answer still change with the grid?

The same case is solved on three or more grids with the same shape of spacing
(same wall clustering, same throat refinement, same ramp width). The finest
three then give, for every quantity in QUANTITIES, an extrapolated value, an
estimated error and a GCI (an error band on the finest result).

The formulas follow Celik et al., "Procedure for estimation and reporting of
uncertainty due to discretization in CFD applications", J. Fluids Eng. 130
(2008) 078001. Index 1 is the finest grid, 3 the coarsest.

    h_i   = N_i^(-1/2)                      (representative cell size, 2D)
    r21   = h2 / h1,   r32 = h3 / h2
    e21   = phi2 - phi1,   e32 = phi3 - phi2
    p     = | ln(e32 / e21) + q(p) | / ln r21
    q(p)  = ln( (r21^p - 1) / (r32^p - 1) )      (solved by fixed point)
    phi_ext = (r21^p phi1 - phi2) / (r21^p - 1)
    GCI21 = 1.25 * |(phi1 - phi2) / phi1| / (r21^p - 1)

p enters the error at no more than MAX_ORDER: a scheme can't really do better
than its own order, so a higher p only means errors cancelling by chance.

When the changes don't shrink steadily (the result goes up and down, or moves
away), Richardson does not apply and a rough estimate is given instead: the
spread of the three grids is the GCI and half of it the estimated error.

Usage
-----
    python main.py --grid-study
    python main.py --grid-study --grid-levels 90x36,128x51,180x72
"""

from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

GRID_TOL = 1e-4          # tighter than a normal run, so the iteration error is
                         # small next to the grid error being measured
GRID_MAX_ITER = 200_000  # C_d has settled long before, even when the residual hasn't
MAX_ORDER = 2.0          # order of the scheme
REFINEMENT = math.sqrt(2.0)   # between the default grids


class Quantity(NamedTuple):
    key: str
    label: str
    unit: str
    fmt: str


# Pressure differences, not absolute pressures: a percentage of 100 kPa makes
# any error look small.
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
    p: float                 # observed order (nan for a rough estimate)
    phi_ext: float           # extrapolated value (the finest one if rough)
    e_ext21: float           # estimated relative error of the finest result
    gci21: float             # error band on the finest result (relative)
    rough: bool              # True: Richardson didn't apply, see the docstring


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
    spread = (max(phi) - min(phi)) / (abs(f1) or 1.0)
    rough = GCIResult(float("nan"), f1, 0.5 * spread, spread, True)

    e21, e32 = f2 - f1, f3 - f2
    if not e21 * e32 > 0.0:                        # up and down, or no change
        return rough
    r21, r32 = h2 / h1, h3 / h2
    log_ratio = math.log(e32 / e21)
    p = abs(log_ratio) / math.log(r21)             # first guess (q = 0)
    try:
        for _ in range(max_iter):
            signed = log_ratio + math.log((r21 ** p - 1.0) / (r32 ** p - 1.0))
            p_new = abs(signed) / math.log(r21)
            if abs(p_new - p) < 1e-12:
                break
            p = p_new
        else:
            return rough
    except (ArithmeticError, ValueError):
        return rough
    p = p_new
    if signed <= 0.0:                              # changes grow with refinement
        return rough
    rp = r21 ** min(p, MAX_ORDER)
    phi_ext = (rp * f1 - f2) / (rp - 1.0)
    return GCIResult(p=p, phi_ext=phi_ext, e_ext21=abs((phi_ext - f1) / phi_ext),
                     gci21=safety * abs(e21 / f1) / (rp - 1.0), rough=False)


# ---------------------------------------------------------------------------
# Choosing and running the grids
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
    if len({nz * nr for nz, nr in levels}) < 3:
        raise ValueError("a grid study needs at least three grids of different size.")
    return levels


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


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(rows: List[Dict], case: str, ramp: float,
                 base: Optional[Dict[str, float]] = None,
                 base_name: str = "") -> Tuple[str, List[Dict]]:
    """Markdown report and CSV records of a study. rows are finest first."""
    finest = rows[:3]
    h = [r["cells"] ** -0.5 for r in finest]
    names = [f"{r['Nz']}x{r['Nr']}" for r in rows]
    lines = ["# Grid study", "", f"Case: `{case}`, residual target {GRID_TOL:g}, "
             f"ramp {ramp:.3f} throat lengths.", "",
             "| grid | cells | iterations | converged | residual | y+ | time [s] |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['Nz']} x {r['Nr']} | {r['cells']} | {r['iters']} | "
                     f"{'yes' if r['converged'] else 'no'} | {r['residual']:.2e} | "
                     f"{r['y_plus']:.2f} | {r['time_s']:.0f} |")
    lines += ["", "| quantity | " + " | ".join(names) + " |", "|---" * (len(rows) + 1) + "|"]
    for q in QUANTITIES:
        lines.append(f"| {q.label} [{q.unit}] | "
                     + " | ".join(f"{r[q.key]:{q.fmt}}" for r in rows) + " |")
    lines.append("| lowest pressure [Pa] | "
                 + " | ".join(f"{r['p_min']:.1f}" for r in rows) + " |")

    extra = f" {base_name} run | its error [%] |" if base else ""
    lines += ["", f"Error estimate on {names[0]} (from {', '.join(names[:3])}):", "",
              f"| quantity | value | extrapolated | estimated error [%] | GCI [%] | band | order p |{extra}",
              "|---" * (9 if base else 7) + "|"]
    records = []
    for q in QUANTITIES:
        phi = [r[q.key] for r in finest]
        g = gci(h, phi)
        band = g.gci21 * abs(phi[0])
        order = "rough" if g.rough else f"{g.p:.2f}"
        row = (f"| {q.label} [{q.unit}] | {phi[0]:{q.fmt}} | {g.phi_ext:{q.fmt}} | "
               f"{100 * g.e_ext21:.3f} | {100 * g.gci21:.3f} | +/- {band:{q.fmt}} | {order} |")
        if base:
            err = abs(base[q.key] - g.phi_ext) / (abs(g.phi_ext) or 1.0)
            row += f" {base[q.key]:{q.fmt}} | {100 * err:.3f} |"
        lines.append(row)
        records.append(dict(quantity=q.key, unit=q.unit, finest=phi[0], extrapolated=g.phi_ext,
                            error_pct=100 * g.e_ext21, gci_pct=100 * g.gci21, band=band,
                            order=g.p, rough=g.rough))
    lines += ["", "Estimated error: distance of the finest grid from the extrapolated value. "
              "GCI: the grid-independent value is expected within +/- GCI of the finest "
              "grid. `rough` means the values don't change steadily with the grid: the "
              "numbers then come from the spread of the three grids. These are grid errors "
              "only, and less reliable if a grid isn't converged or has y+ above 5."]
    return "\n".join(lines) + "\n", records


def _write_csv(path: str, records: List[Dict]) -> None:
    keys = list(records[0].keys())
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for r in records:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")


def run_study(args: argparse.Namespace, levels: List[Tuple[int, int]],
              base: Optional[Dict[str, float]] = None, base_name: str = "",
              write_files: bool = True) -> str:
    """Solve the case on every grid, print the report and return it.

    Args:
        args: the options of the case, as parsed by cli.build_parser.
        levels: grids as (Nz, Nr), coarsest first.
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
    a0.tol, a0.max_iter = GRID_TOL, min(args.max_iter, GRID_MAX_ITER)
    cfg0 = config_from_args(a0)
    geom0 = create_venturi_geometry(cfg0)
    zs = geom0.z_stations
    ramp = (throat_ramp_width(geom0.L_total, levels[0][0], zs[2], zs[3], cfg0.throat_refine)
            if cfg0.throat_refine > 1.0 else 0.5)

    print(f"Grid study: {', '.join(f'{nz}x{nr}' for nz, nr in levels)}", flush=True)
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        futures = [pool.submit(run_level, vars(a0), nz, nr, ramp) for nz, nr in levels]
        rows = []
        for f in as_completed(futures):
            r = f.result()
            rows.append(r)
            print(f"  {r['Nz']:4d} x {r['Nr']:3d}: {r['iters']:7d} iterations, "
                  f"{r['time_s']:6.1f} s, C_d {r['Cd']:.5f}", flush=True)

    rows.sort(key=lambda r: -r["cells"])          # finest first
    case = os.path.basename(cfg0.output_dir)
    text, records = build_report(rows, case, ramp, base, base_name)
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
