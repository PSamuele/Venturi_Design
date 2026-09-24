"""
Venturi design and flow check.

Main pieces, in the order a run uses them:

    VenturiConfig            inputs and sizing
    create_venturi_geometry  wall profile R(z)
    build_fvmesh             grid of ring-shaped cells
    solve_fv                 flow solution
    validate                 checks and discharge coefficient
    export_all               output files
"""

from .config import VenturiConfig
from .fluids import FLUIDS, Fluid, list_fluids
from .geometry import VenturiGeometry, create_venturi_geometry
from .fvmesh import FVMesh, build_fvmesh, divergence
from .projection import Projector
from .solver import FVResult, solve_fv, inlet_profile
from .turbulence import compute_nu_t, wall_y_plus, SUPPORTED as TURBULENCE_MODELS
from .validation import ValidationReport, validate, print_report, to_dict
from .export import export_all
from .optimizer import run_optimization

__version__ = "3.0.0"

__all__ = [
    "VenturiConfig", "FLUIDS", "Fluid", "list_fluids",
    "VenturiGeometry", "create_venturi_geometry",
    "FVMesh", "build_fvmesh", "divergence", "Projector",
    "FVResult", "solve_fv", "inlet_profile",
    "compute_nu_t", "wall_y_plus", "TURBULENCE_MODELS",
    "ValidationReport", "validate", "print_report", "to_dict",
    "export_all", "run_optimization",
]
