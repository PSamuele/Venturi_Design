"""
Venturi Design & CFD
====================
Parametric design of a Venturi tube plus CFD validation.

Workflow
--------
    imposed values  ->  inverse sizing  ->  ISO 5167-4 geometry
    ->  finite-volume grid  ->  axisymmetric Navier-Stokes with turbulence
    ->  validation against C_d  ->  export

Usage
-----
    python main.py --D 0.1 --beta 0.5 --p-inlet 101325 --p-throat 95000
    python main.py --list-fluids
    python main.py                       (interactive wizard)
"""

from __future__ import annotations

import argparse
import sys
import time

from venturi.config import VenturiConfig, FLUID_PRESETS, list_fluids
from venturi.geometry import create_venturi_geometry
from venturi.fvmesh import build_fvmesh
from venturi.fvsolver import solve_fv
from venturi.validation_fv import validate, print_report, to_dict, ISO_TYPES
from venturi.compat import mesh_view
from venturi.export import export_all
from venturi.optimizer import run_optimization


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="venturi",
        description="Parametric design and CFD validation of a Venturi tube.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py --D 0.1 --beta 0.5 --fluid water_20C --p-inlet 101325 --p-throat 95000
  python main.py --D 0.08 --beta 0.6 --fluid custom --rho 1.2 --mu 1.8e-5 \\
                 --p-inlet 200000 --p-throat 180000
  python main.py --list-fluids

Computational cost (measured, single CPU):
   90 x 36  ->  110,000 iterations,  ~1 minute   (default)
  128 x 51  ->  272,000 iterations,  ~4.5 minutes
Finer grids grow quickly: see the README note on cost.
""")

    p.add_argument("--list-fluids", action="store_true",
                   help="Show the preset fluids and exit.")
    p.add_argument("--fluid", type=str, default="water_20C",
                   choices=list(FLUID_PRESETS.keys()) + ["custom"],
                   help="Preset fluid (default: water_20C).")
    p.add_argument("--rho", type=float, default=None,
                   help="Density [kg/m3], only with --fluid custom.")
    p.add_argument("--mu", type=float, default=None,
                   help="Dynamic viscosity [Pa s], only with --fluid custom.")

    p.add_argument("--D", type=float, default=0.100, help="Inlet diameter [m].")
    p.add_argument("--beta", type=float, default=0.50, help="Ratio d/D.")
    p.add_argument("--alpha-conv", type=float, default=21.0,
                   help="Total convergent angle [deg]. ISO: 21 +/- 1.")
    p.add_argument("--alpha-div", type=float, default=8.0,
                   help="Total divergent angle [deg]. ISO: 7 - 15.")
    p.add_argument("--no-blend", action="store_true",
                   help="Disable the ISO blend radii.")
    p.add_argument("--venturi-type", type=str, default="machined",
                   choices=sorted(ISO_TYPES.keys()),
                   help="Construction type; sets the ISO C_d to compare against.")

    p.add_argument("--p-inlet", type=float, default=101325.0,
                   help="Upstream pressure [Pa].")
    p.add_argument("--p-throat", type=float, default=95000.0,
                   help="Target throat pressure [Pa].")

    p.add_argument("--Nz", type=int, default=90,
                   help="Axial cells (default: 90, about 1 minute of compute).")
    p.add_argument("--Nr", type=int, default=36,
                   help="Radial cells (default: 36).")
    p.add_argument("--clustering", type=float, default=2.7,
                   help="Cell clustering towards the wall.")

    p.add_argument("--turbulence", type=str, default="baldwin_lomax",
                   choices=["laminar", "mixing_length", "baldwin_lomax"],
                   help="Turbulence model.")
    p.add_argument("--max-iter", type=int, default=400000,
                   help="Maximum iterations. Hundreds of thousands of steps "
                        "are normal: see the cost note in the README.")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Tolerance on the dimensionless residual |du/dt|*L/U^2.")
    p.add_argument("--cfl", type=float, default=0.6, help="CFL number.")

    p.add_argument("--output-dir", type=str, default=None,
                   help="Output folder (default: derived from the parameters).")
    p.add_argument("--no-export", action="store_true",
                   help="Run the computation without writing any file.")
    return p


# ---------------------------------------------------------------------------
# Interactive wizard
# ---------------------------------------------------------------------------
def interactive_wizard() -> dict:
    print("=" * 66)
    print("  VENTURI DESIGN - INTERACTIVE WIZARD")
    print("=" * 66)
    print("\nMode:")
    print("  [1] fixed-parameter design")
    print("  [2] automatic shape optimisation")
    mode = input("Choice [1/2]: ").strip() or "1"

    print("\nFluids: " + ", ".join(FLUID_PRESETS.keys()))
    fluid = input("Fluid [water_20C]: ").strip()
    if fluid not in FLUID_PRESETS:
        fluid = "water_20C"

    p_in = float(input("Upstream pressure [Pa] (101325): ").strip() or "101325")
    p_th = float(input("Throat pressure [Pa] (95000): ").strip() or "95000")

    args = dict(fluid=fluid, rho=None, mu=None, p_inlet=p_in, p_throat=p_th,
                no_blend=False, venturi_type="machined",
                Nz=90, Nr=36, clustering=2.7, turbulence="baldwin_lomax",
                max_iter=400000, tol=1e-3, cfl=0.6,
                output_dir=None, no_export=False, list_fluids=False)

    if mode == "2":
        Q = float(input("Target volumetric flow rate [m3/s] (0.05): ").strip() or "0.05")
        print("\nObjective:")
        print("  [1] minimum pressure loss")
        print("  [2] minimum axial length")
        print("  [3] balanced")
        obj = int(input("Choice [1/2/3]: ").strip() or "3")
        res = run_optimization(Q, p_in, p_th, FLUID_PRESETS[fluid]["rho"], obj)
        if not res["success"]:
            print("  Warning: the optimiser did not fully converge.")
        print(f"  Optimum: D = {res['D_inlet']*1000:.1f} mm, beta = {res['beta']:.3f}, "
              f"alpha_conv = {res['alpha_conv_deg']:.1f} deg, "
              f"alpha_div = {res['alpha_div_deg']:.1f} deg")
        args.update(D=res["D_inlet"], beta=res["beta"],
                    alpha_conv=res["alpha_conv_deg"],
                    alpha_div=res["alpha_div_deg"])
    else:
        args.update(D=float(input("Inlet diameter [m] (0.1): ").strip() or "0.1"),
                    beta=float(input("Beta ratio (0.5): ").strip() or "0.5"),
                    alpha_conv=21.0, alpha_div=8.0)
    return args


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run(args) -> int:
    if getattr(args, "list_fluids", False):
        print("\nPreset fluids:\n")
        print(list_fluids())
        print("\nCustom fluid: --fluid custom --rho <val> --mu <val>")
        return 0

    if args.fluid == "custom" and (args.rho is None or args.mu is None):
        print("Error: --fluid custom requires --rho and --mu.")
        return 1

    if not args.output_dir:
        args.output_dir = (f"output_{args.fluid}_D{args.D*1000:.0f}"
                           f"_beta{args.beta:.2f}_Pin{int(args.p_inlet)}"
                           f"_Pth{int(args.p_throat)}")

    # ---------------------------------------------------- [1/5] inverse sizing
    print("\n[1/5] Inverse sizing from the imposed values")
    kw = dict(D_inlet=args.D, beta=args.beta,
              alpha_conv_deg=args.alpha_conv, alpha_div_deg=args.alpha_div,
              L_throat_factor=1.0, L_inlet_factor=1.5, L_outlet_factor=3.0,
              use_blend_radii=not args.no_blend,
              p_inlet=args.p_inlet, p_throat_target=args.p_throat,
              fluid_name=args.fluid, Nz=args.Nz, Nr=args.Nr,
              max_iter=args.max_iter, tol_steady=args.tol, cfl_max=args.cfl,
              turbulence_model=args.turbulence, output_dir=args.output_dir)
    if args.fluid == "custom":
        kw["rho"] = args.rho
        kw["mu"] = args.mu

    config = VenturiConfig(**kw)
    print(config.summary())
    for w in config.validate():
        print(f"  {w}")

    # -------------------------------------------------------- [2/5] geometry
    print("\n[2/5] ISO 5167-4 geometry")
    geom = create_venturi_geometry(config)
    print(f"      total length {geom.L_total*1000:.1f} mm  "
          f"(inlet {geom.L_inlet*1000:.1f}, convergent {geom.L_conv*1000:.1f}, "
          f"throat {geom.L_throat*1000:.1f}, divergent {geom.L_div*1000:.1f}, "
          f"outlet {geom.L_outlet*1000:.1f})")

    # ------------------------------------------------------------ [3/5] grid
    print(f"\n[3/5] Finite-volume grid {args.Nz} x {args.Nr}")
    mesh = build_fvmesh(geom, args.Nz, args.Nr, args.clustering)
    print(f"      {mesh.n_cells} cells, volume {mesh.vol.sum()*1e6:.2f} cm3, "
          f"smallest cell {mesh.vol.min()*1e9:.3f} mm3")

    # ---------------------------------------------------------- [4/5] solver
    print(f"\n[4/5] Axisymmetric Navier-Stokes, turbulence '{args.turbulence}'")
    t0 = time.time()
    result = solve_fv(mesh, config, verbose=True)
    print(f"      {'CONVERGED' if result.converged else 'NOT CONVERGED'} - "
          f"{result.n_iter} iterations in {time.time()-t0:.1f} s, "
          f"residual {result.final_residual:.2e}")

    # ------------------------------------------------------ [5/5] validation
    print("\n[5/5] Validation")
    rep = validate(result, mesh, geom, config, venturi_type=args.venturi_type)
    print_report(rep)

    # ---------------------------------------------------------------- export
    if not args.no_export:
        print()
        files = export_all(mesh_view(mesh), result.uz, result.ur, result.p,
                           geom, config, to_dict(rep))
        print(f"\n{len(files)} files written to {config.output_dir}/")

    print("\n" + "=" * 66)
    print("RESULT: " + ("ALL CHECKS PASSED" if rep.all_passed
                        else "SOME CHECKS FAILED"))
    print("=" * 66 + "\n")
    return 0 if rep.all_passed else 2


def main() -> int:
    parser = build_parser()
    if len(sys.argv) == 1:
        d = interactive_wizard()
        args = argparse.Namespace(**d)
    else:
        args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
