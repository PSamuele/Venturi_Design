"""
cli.py
======
Command line, interactive wizard and the run pipeline:

    inputs -> sizing -> wall profile -> grid -> flow solution -> checks -> files
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable, Optional

from .config import VenturiConfig
from .export import export_all
from .fluids import FLUIDS, list_fluids
from .fvmesh import build_fvmesh
from .geometry import create_venturi_geometry
from .optimizer import run_optimization
from .solver import solve_fv
from .turbulence import SUPPORTED as TURBULENCE_MODELS
from .validation import print_report, to_dict, validate

RESULTS_DIR = "results"

EPILOG = """\
examples:
  python main.py --D 0.1 --beta 0.5 --fluid water_20C --p-inlet 101325 --p-throat 95000
  python main.py --fluid custom --rho 1.2 --mu 1.8e-5 --p-inlet 200000 --p-throat 180000
  python main.py --list-fluids

run time (single CPU core, measured):
  90 x 36 grid   about 1 minute (default)
  finer grids take much longer: see "Speed" in the README
"""


def _refine(text: str):
    """--throat-refine accepts 'auto' or a number."""
    if text.strip().lower() == "auto":
        return "auto"
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'auto' or a number, got '{text}'")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="venturi", description="Design a Venturi tube and check it with a flow simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG)
    a = p.add_argument

    a("--list-fluids", action="store_true", help="Print the preset fluids and exit.")
    a("--fluid", default="water_20C", choices=list(FLUIDS) + ["custom"],
      help="Preset fluid (default: water_20C).")
    a("--rho", type=float, default=None, help="Density [kg/m3], needed with --fluid custom.")
    a("--mu", type=float, default=None, help="Dynamic viscosity [Pa s], needed with --fluid custom.")
    a("--p-vap", type=float, default=None,
      help="Vapour pressure [Pa] for --fluid custom (default 0: no cavitation check).")

    a("--D", type=float, default=0.100, help="Pipe diameter [m] (default 0.1).")
    a("--beta", type=float, default=0.50, help="Throat diameter / pipe diameter (default 0.5).")
    a("--alpha-conv", type=float, default=21.0, help="Total angle of the converging cone [deg].")
    a("--alpha-div", type=float, default=8.0, help="Total angle of the diverging cone [deg].")
    a("--no-blend", action="store_true", help="Sharp corners instead of rounded ones.")

    a("--p-inlet", type=float, default=101325.0, help="Upstream pressure [Pa].")
    a("--p-throat", type=float, default=95000.0, help="Wanted throat pressure [Pa].")
    a("--cd-design", type=float, default=1.0,
      help="Discharge coefficient assumed when sizing (1 = no friction).")

    a("--Nz", type=int, default=90, help="Cells along the axis (default 90).")
    a("--Nr", type=int, default=36, help="Cells along the radius (default 36).")
    a("--clustering", type=float, default=2.7,
      help="Wall clustering: 0 = even cells, larger = thinner cells at the wall.")
    a("--throat-refine", type=_refine, default="auto",
      help="How many times shorter the throat cells are: 'auto' (default) "
           "= 1/beta^2, capped at 6, or a number >= 1 (1 = even spacing).")
    a("--turbulence", default="baldwin_lomax", choices=list(TURBULENCE_MODELS),
      help="Turbulence model (default baldwin_lomax).")
    a("--max-iter", type=int, default=400000, help="Iteration limit.")
    a("--tol", type=float, default=1e-3, help="Steady-state target for the residual.")
    a("--cfl", type=float, default=0.6, help="CFL number, the time step safety factor.")
    a("--time-step", default="local", choices=["local", "global"],
      help="local: each cell takes its own largest stable step (default); "
           "global: all cells take the smallest one (slower, same result).")

    a("--cd-ref", type=float, default=None,
      help="Reference C_d to compare with (for example from a datasheet).")
    a("--cd-tol", type=float, default=3.0, help="Allowed deviation from --cd-ref [%%].")

    a("--output-dir", default=None,
      help=f"Output folder (default: {RESULTS_DIR}/<name built from the inputs>).")
    a("--no-export", action="store_true", help="Compute only, write no files.")
    return p


# ---------------------------------------------------------------------------
# Interactive wizard
# ---------------------------------------------------------------------------
def _ask(prompt: str, default, cast: Callable = float, check: Optional[Callable] = None):
    """Ask until the answer can be converted and passes the check."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = cast(raw)
        except ValueError:
            print("  Not a valid value, try again.")
            continue
        if check is not None and not check(value):
            print("  Value out of range, try again.")
            continue
        return value


def interactive_wizard() -> argparse.Namespace:
    args = build_parser().parse_args([])
    print("=" * 66)
    print("  VENTURI DESIGN - INTERACTIVE MODE")
    print("=" * 66)
    print("\n  [1] I choose D and beta")
    print("  [2] choose them for me from a flow rate")
    mode = _ask("Mode", 1, int, lambda v: v in (1, 2))

    print("\nFluids: " + ", ".join(FLUIDS))
    args.fluid = _ask("Fluid", "water_20C", str, lambda v: v in FLUIDS)
    args.p_inlet = _ask("Upstream pressure [Pa]", 101325.0, float, lambda v: v > 0)
    args.p_throat = _ask("Throat pressure [Pa]", 95000.0, float,
                         lambda v: 0 < v < args.p_inlet)

    if mode == 2:
        Q = _ask("Flow rate [m3/s]", 0.05, float, lambda v: v > 0)
        print("\n  [1] smallest pressure loss\n  [2] shortest tube\n  [3] balanced")
        obj = _ask("Goal", 3, int, lambda v: v in (1, 2, 3))
        rho = FLUIDS[args.fluid].density_at(args.p_inlet)
        res = run_optimization(Q, args.p_inlet, args.p_throat, rho, obj)
        if not res["success"]:
            print(f"  Warning: the optimiser did not fully converge ({res['message']}).")
        print(f"  Chosen: D = {res['D_inlet']*1000:.1f} mm, beta = {res['beta']:.3f}, "
              f"alpha_div = {res['alpha_div_deg']:.1f} deg")
        args.D, args.beta, args.alpha_div = res["D_inlet"], res["beta"], res["alpha_div_deg"]
    else:
        args.D = _ask("Pipe diameter D [m]", 0.1, float, lambda v: v > 0)
        args.beta = _ask("beta = d/D", 0.5, float, lambda v: 0 < v < 1)
    return args


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def default_output_dir(args) -> str:
    name = (f"{args.fluid}_D{args.D*1000:.0f}_beta{args.beta:.2f}"
            f"_Pin{int(args.p_inlet)}_Pth{int(args.p_throat)}")
    return os.path.join(RESULTS_DIR, name)


def config_from_args(args) -> VenturiConfig:
    return VenturiConfig(
        D=args.D, beta=args.beta,
        alpha_conv_deg=args.alpha_conv, alpha_div_deg=args.alpha_div,
        use_blend_radii=not args.no_blend,
        p_inlet=args.p_inlet, p_throat_target=args.p_throat, cd_design=args.cd_design,
        fluid=args.fluid, rho=args.rho, mu=args.mu, p_vap=args.p_vap,
        Nz=args.Nz, Nr=args.Nr, wall_clustering=args.clustering,
        throat_refine=args.throat_refine,
        turbulence_model=args.turbulence, cfl=args.cfl, time_step=args.time_step,
        max_iter=args.max_iter, tol=args.tol,
        output_dir=args.output_dir or default_output_dir(args))


def run(args) -> int:
    if args.list_fluids:
        print(list_fluids())
        print("\nOwn fluid: --fluid custom --rho <value> --mu <value> [--p-vap <value>]")
        return 0

    print("\n[1/6] Sizing from the inputs")
    try:
        config = config_from_args(args)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    print(config.summary())
    for w in config.warnings():
        print(f"  [WARNING] {w}")

    print("\n[2/6] Wall profile")
    geom = create_venturi_geometry(config)
    print(f"      total length {geom.L_total*1000:.1f} mm  "
          f"(inlet {geom.L_inlet*1000:.1f}, converging {geom.L_conv*1000:.1f}, "
          f"throat {geom.L_throat*1000:.1f}, diverging {geom.L_div*1000:.1f}, "
          f"outlet {geom.L_outlet*1000:.1f})")

    print(f"\n[3/6] Grid {config.Nz} x {config.Nr}")
    mesh = build_fvmesh(geom, config.Nz, config.Nr, config.wall_clustering,
                        config.throat_refine)
    print(f"      {mesh.n_cells} cells, volume {mesh.vol.sum()*1e6:.2f} cm3, "
          f"smallest cell {mesh.vol.min()*1e9:.3f} mm3")

    print(f"\n[4/6] Flow solution, turbulence model '{config.turbulence_model}'")
    t0 = time.time()
    result = solve_fv(mesh, config, verbose=True)
    print(f"      {'CONVERGED' if result.converged else 'NOT CONVERGED'} - "
          f"{result.n_iter} iterations in {time.time()-t0:.1f} s, "
          f"residual {result.final_residual:.2e}")

    print("\n[5/6] Checks")
    rep = validate(result, mesh, geom, config, cd_ref=args.cd_ref, cd_tol_pct=args.cd_tol)
    print_report(rep)

    if args.no_export:
        print("\n[6/6] Files: skipped (--no-export)")
    else:
        print("\n[6/6] Files")
        files = export_all(mesh, result.uz, result.ur, result.p, geom, config, to_dict(rep))
        print(f"      {len(files)} files in {config.output_dir}{os.sep}")

    print("\n" + "=" * 66)
    print("RESULT: " + ("ALL CHECKS PASSED" if rep.all_passed else "SOME CHECKS FAILED"))
    print("=" * 66 + "\n")
    return 0 if rep.all_passed else 2


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = interactive_wizard() if not argv else build_parser().parse_args(argv)
    return run(args)

