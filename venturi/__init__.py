"""
Venturi Design & CFD
====================
Parametric design of a Venturi tube to ISO 5167-4, validated with an
axisymmetric finite-volume Navier-Stokes solver.

Main API
--------
    VenturiConfig            inverse sizing from the imposed values
    create_venturi_geometry  ISO 5167-4 wall profile
    build_fvmesh             body-fitted finite-volume grid
    solve_fv                 steady solver with algebraic turbulence
    validate                 validation against the standard C_d
    export_all               write every output format
"""

from .config import VenturiConfig, FLUID_PRESETS, list_fluids
from .geometry import VenturiGeometry, create_venturi_geometry
from .fvmesh import FVMesh, build_fvmesh, divergence
from .projection import Projector, compute_face_coefficients
from .fvsolver import FVResult, solve_fv, inlet_profile
from .turbulence import compute_nu_t, wall_y_plus, SUPPORTED as TURBULENCE_MODELS
from .validation_fv import ValidationReport, validate, print_report, to_dict, ISO_TYPES
from .compat import MeshView, mesh_view
from .export import export_all
from .optimizer import run_optimization

__version__ = "2.0.0"

__all__ = [
    "VenturiConfig", "FLUID_PRESETS", "list_fluids",
    "VenturiGeometry", "create_venturi_geometry",
    "FVMesh", "build_fvmesh", "divergence",
    "Projector", "compute_face_coefficients",
    "FVResult", "solve_fv", "inlet_profile",
    "compute_nu_t", "wall_y_plus", "TURBULENCE_MODELS",
    "ValidationReport", "validate", "print_report", "to_dict", "ISO_TYPES",
    "MeshView", "mesh_view", "export_all", "run_optimization",
]
