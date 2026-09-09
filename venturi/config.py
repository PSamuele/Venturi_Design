"""
Venturi Tube Design Tool — Configuration Module
================================================
Comprehensive configuration dataclasses, fluid thermodynamic property database,
1D Bernoulli analytical design, Reynolds number regimes, cavitation assessment,
and ISO 5167-4 validation checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List


# ---------------------------------------------------------------------------
# Fluid Properties Model & Database
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FluidProperties:
    """
    Immutable physical and thermodynamic properties of a working fluid.

    Attributes:
        name: Unique string identifier.
        rho: Mass density [kg/m^3].
        mu: Dynamic shear viscosity [Pa*s].
        nu: Kinematic viscosity [m^2/s].
        p_vap: Vapor pressure at operating temperature [Pa].
        temp_C: Fluid temperature [°C].
        description: Informative human-readable description.
    """
    name: str
    rho: float
    mu: float
    nu: float
    p_vap: float = 0.0
    temp_C: float = 20.0
    description: str = ""

    def __getitem__(self, item: str) -> Any:
        """Dictionary-like access for backward compatibility."""
        if item == "desc":
            return self.description
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f"Property {item} not found in FluidProperties")


# Extensive 15+ Fluid Database covering water at multiple temperatures,
# gases, organic liquids, oils, liquid metals, and fuels.
FLUID_DATABASE: Dict[str, FluidProperties] = {
    "water_10C": FluidProperties(
        name="water_10C",
        rho=999.7,
        mu=1.307e-3,
        nu=1.3074e-6,
        p_vap=1228.0,
        temp_C=10.0,
        description="Liquid water at 10°C, 1 atm",
    ),
    "water_20C": FluidProperties(
        name="water_20C",
        rho=998.2,
        mu=1.002e-3,
        nu=1.0038e-6,
        p_vap=2338.8,
        temp_C=20.0,
        description="Standard liquid water at 20°C, 1 atm",
    ),
    "water_40C": FluidProperties(
        name="water_40C",
        rho=992.2,
        mu=0.653e-3,
        nu=0.6581e-6,
        p_vap=7384.0,
        temp_C=40.0,
        description="Liquid water at 40°C, 1 atm",
    ),
    "water_50C": FluidProperties(
        name="water_50C",
        rho=988.0,
        mu=0.547e-3,
        nu=0.5536e-6,
        p_vap=12350.0,
        temp_C=50.0,
        description="Liquid water at 50°C, 1 atm",
    ),
    "water_60C": FluidProperties(
        name="water_60C",
        rho=983.2,
        mu=0.467e-3,
        nu=0.4750e-6,
        p_vap=19940.0,
        temp_C=60.0,
        description="Liquid water at 60°C, 1 atm",
    ),
    "water_80C": FluidProperties(
        name="water_80C",
        rho=971.8,
        mu=0.355e-3,
        nu=0.3653e-6,
        p_vap=47390.0,
        temp_C=80.0,
        description="Liquid water at 80°C, 1 atm",
    ),
    "air_0C": FluidProperties(
        name="air_0C",
        rho=1.293,
        mu=1.716e-5,
        nu=1.3271e-5,
        p_vap=0.0,
        temp_C=0.0,
        description="Dry air at 0°C, 1 atm",
    ),
    "air_20C": FluidProperties(
        name="air_20C",
        rho=1.204,
        mu=1.825e-5,
        nu=1.5158e-5,
        p_vap=0.0,
        temp_C=20.0,
        description="Dry air at 20°C, 1 atm",
    ),
    "air_50C": FluidProperties(
        name="air_50C",
        rho=1.092,
        mu=1.963e-5,
        nu=1.7976e-5,
        p_vap=0.0,
        temp_C=50.0,
        description="Dry air at 50°C, 1 atm",
    ),
    "air_100C": FluidProperties(
        name="air_100C",
        rho=0.946,
        mu=2.181e-5,
        nu=2.3055e-5,
        p_vap=0.0,
        temp_C=100.0,
        description="Dry air at 100°C, 1 atm",
    ),
    "glycerin": FluidProperties(
        name="glycerin",
        rho=1261.0,
        mu=1.412,
        nu=1.1197e-3,
        p_vap=0.01,
        temp_C=20.0,
        description="Pure Glycerin (100%) at 20°C",
    ),
    "glycerin_50": FluidProperties(
        name="glycerin_50",
        rho=1130.0,
        mu=6.000e-3,
        nu=5.3097e-6,
        p_vap=1800.0,
        temp_C=20.0,
        description="Aqueous glycerin solution (50% wt) at 20°C",
    ),
    "ethylene_glycol": FluidProperties(
        name="ethylene_glycol",
        rho=1113.0,
        mu=1.610e-2,
        nu=1.4465e-5,
        p_vap=8.0,
        temp_C=20.0,
        description="Pure ethylene glycol at 20°C",
    ),
    "glycol_50": FluidProperties(
        name="glycol_50",
        rho=1082.0,
        mu=3.340e-3,
        nu=3.0869e-6,
        p_vap=1500.0,
        temp_C=20.0,
        description="Ethylene glycol 50% aqueous solution at 20°C",
    ),
    "oil_SAE30": FluidProperties(
        name="oil_SAE30",
        rho=891.0,
        mu=0.2386,
        nu=2.6779e-4,
        p_vap=0.0,
        temp_C=20.0,
        description="Motor Oil SAE 30 at 20°C",
    ),
    "oil_ISO_VG46": FluidProperties(
        name="oil_ISO_VG46",
        rho=875.0,
        mu=0.0402,
        nu=4.5943e-5,
        p_vap=0.0,
        temp_C=40.0,
        description="Hydraulic Oil ISO VG 46 at 40°C",
    ),
    "mercury": FluidProperties(
        name="mercury",
        rho=13534.0,
        mu=1.530e-3,
        nu=1.1305e-7,
        p_vap=0.16,
        temp_C=20.0,
        description="Liquid elemental mercury at 20°C",
    ),
    "ethanol": FluidProperties(
        name="ethanol",
        rho=789.0,
        mu=1.200e-3,
        nu=1.5209e-6,
        p_vap=5870.0,
        temp_C=20.0,
        description="Pure ethanol at 20°C",
    ),
    "ethanol_20C": FluidProperties(
        name="ethanol_20C",
        rho=789.0,
        mu=1.200e-3,
        nu=1.5209e-6,
        p_vap=5870.0,
        temp_C=20.0,
        description="Pure ethanol at 20°C",
    ),
    "kerosene": FluidProperties(
        name="kerosene",
        rho=810.0,
        mu=1.640e-3,
        nu=2.0247e-6,
        p_vap=500.0,
        temp_C=20.0,
        description="Kerosene / Jet-A fuel at 20°C",
    ),
    "diesel": FluidProperties(
        name="diesel",
        rho=832.0,
        mu=2.600e-3,
        nu=3.1250e-6,
        p_vap=1000.0,
        temp_C=20.0,
        description="Diesel fuel (automotive) at 20°C",
    ),
    "seawater": FluidProperties(
        name="seawater",
        rho=1025.0,
        mu=1.070e-3,
        nu=1.0439e-6,
        p_vap=2300.0,
        temp_C=20.0,
        description="Standard ocean seawater (35 PSU salinity) at 20°C",
    ),
}

# Backward compatibility alias dict
FLUID_PRESETS: Dict[str, Dict[str, Any]] = {
    name: {
        "rho": prop.rho,
        "mu": prop.mu,
        "nu": prop.nu,
        "p_vap": prop.p_vap,
        "temp_C": prop.temp_C,
        "desc": prop.description,
        "description": prop.description,
    }
    for name, prop in FLUID_DATABASE.items()
}


def get_fluid(name: str) -> FluidProperties:
    """
    Retrieve fluid properties by name from FLUID_DATABASE.

    Args:
        name: Name key of the fluid (e.g., 'water_20C', 'air_20C').

    Returns:
        FluidProperties instance.

    Raises:
        KeyError: If fluid name is not found in the database.
    """
    if name in FLUID_DATABASE:
        return FLUID_DATABASE[name]
    raise KeyError(f"Fluid '{name}' not found in database. Available fluids: {list(FLUID_DATABASE.keys())}")


def register_custom_fluid(
    name: str,
    rho: float,
    mu: float,
    nu: Optional[float] = None,
    p_vap: float = 0.0,
    temp_C: float = 20.0,
    description: str = "",
) -> FluidProperties:
    """
    Register a custom fluid into the global database.

    Args:
        name: Unique identifier name for the custom fluid.
        rho: Density [kg/m^3].
        mu: Dynamic viscosity [Pa*s].
        nu: Optional kinematic viscosity [m^2/s] (defaults to mu / rho).
        p_vap: Vapor pressure [Pa].
        temp_C: Operating temperature [°C].
        description: Description of the fluid.

    Returns:
        The newly created FluidProperties object.
    """
    if rho <= 0:
        raise ValueError(f"Density rho must be strictly positive, got {rho}")
    if mu <= 0:
        raise ValueError(f"Dynamic viscosity mu must be strictly positive, got {mu}")
    if nu is None:
        nu = mu / rho

    fluid_prop = FluidProperties(
        name=name,
        rho=float(rho),
        mu=float(mu),
        nu=float(nu),
        p_vap=float(p_vap),
        temp_C=float(temp_C),
        description=description or f"Custom fluid: {name}",
    )
    FLUID_DATABASE[name] = fluid_prop
    FLUID_PRESETS[name] = {
        "rho": fluid_prop.rho,
        "mu": fluid_prop.mu,
        "nu": fluid_prop.nu,
        "p_vap": fluid_prop.p_vap,
        "temp_C": fluid_prop.temp_C,
        "desc": fluid_prop.description,
        "description": fluid_prop.description,
    }
    return fluid_prop


def list_fluids() -> str:
    """
    Returns a formatted tabular string of all registered fluids.
    """
    lines = [
        f"{'Name':<16} {'rho [kg/m3]':>12} {'mu [Pa*s]':>14} {'nu [m2/s]':>14} {'p_vap [Pa]':>12}   {'Description'}",
        "-" * 95,
    ]
    for name, prop in FLUID_DATABASE.items():
        lines.append(
            f"{name:<16} {prop.rho:>12.1f} {prop.mu:>14.4e} {prop.nu:>14.4e} {prop.p_vap:>12.1f}   {prop.description}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# VenturiConfig Dataclass
# ---------------------------------------------------------------------------
@dataclass
class VenturiConfig:
    """
    Master configuration dataclass for Venturi tube geometry, fluid dynamics,
    discretization mesh, and numerical CFD solver settings.
    """

    # --- Geometric Parameters ---
    D: float = 0.10                     # Inlet & outlet pipe diameter [m] (ISO: 0.05 <= D <= 1.20)
    beta: float = 0.50                  # Throat diameter ratio d/D (ISO: 0.30 <= beta <= 0.75)
    alpha_conv_deg: float = 21.0        # Convergent cone angle [deg] (ISO: 21 ± 1 deg)
    alpha_div_deg: float = 8.0          # Divergent cone angle [deg] (ISO: 7 – 15 deg)
    L_inlet_ratio: float = 1.5          # L_inlet / D (straight inlet pipe length factor)
    L_throat_ratio: float = 1.0         # L_throat / d (ISO: 1.0)
    L_outlet_ratio: float = 3.0         # L_outlet / D (straight outlet pipe length factor)
    use_blend_radii: bool = True        # Enable ISO 5167-4 circular blending arcs (R1, R2, R3, R4)
    blend_R1_ratio: float = 0.2         # R1 / D (Inlet -> Conv blend arc)
    blend_R2_ratio: float = 0.2         # R2 / d (Conv -> Throat blend arc)
    blend_R3_ratio: float = 0.2         # R3 / d (Throat -> Div blend arc)
    blend_R4_ratio: float = 0.2         # R4 / D (Div -> Outlet blend arc)
    geometry_type: str = "iso5167"      # "iso5167", "bezier", "spline"

    # --- Flow / Operating Conditions ---
    v_inlet: float = 1.0                # Mean bulk inlet velocity [m/s]
    p_inlet: float = 101325.0           # Static pressure at inlet [Pa]
    p_outlet: float = 101325.0          # Static reference pressure at outlet [Pa]
    p_throat_target: Optional[float] = None  # Target throat pressure [Pa] for 1D inverse design

    # --- Fluid Selection & Physical Properties ---
    fluid: str = "water_20C"            # Fluid lookup key in FLUID_DATABASE
    rho: float = 998.2                  # Density [kg/m^3]
    mu: float = 1.002e-3                # Dynamic viscosity [Pa*s]
    nu: float = 1.004e-6                # Kinematic viscosity [m^2/s]
    p_vap: float = 2338.8               # Vapor pressure [Pa]

    # --- Mesh Discretization ---
    Nz: int = 120                       # Number of axial grid nodes/cells
    Nr: int = 40                        # Number of radial grid nodes/cells
    r_clustering: float = 1.5           # Wall clustering stretching factor (tanh)
    radial_distribution: str = "tanh"   # "tanh", "geometric", "uniform"
    axial_distribution: str = "uniform" # "uniform", "clustered_throat"

    # --- Numerical Solver Settings ---
    cfl: float = 0.5                    # Maximum Courant-Friedrichs-Lewy number for adaptive dt
    max_iter: int = 5000                # Maximum CFD time-stepping iterations
    tol: float = 1e-5                   # Steady-state residual convergence tolerance
    poisson_tol: float = 1e-6           # Pressure Poisson equation tolerance
    poisson_max_iter: int = 1000        # Max iterations for Poisson linear solver
    poisson_solver: str = "sor"         # "sor" (Numba RB-SOR), "splu" (Sparse LU), "cg"
    sor_omega: float = 1.85             # Successive Over-Relaxation factor
    advection_scheme: str = "upwind"    # "upwind", "central", "tvd_minmod"
    turbulence_model: str = "baldwin_lomax" # "laminar", "mixing_length", "baldwin_lomax", "spalart_allmaras"
    output_dir: str = "output"          # Default export directory

    # --- Derived Physical & Geometric Quantities ---
    d: float = field(init=False)
    R_inlet: float = field(init=False)
    R_throat: float = field(init=False)
    alpha_conv_rad: float = field(init=False)
    alpha_div_rad: float = field(init=False)
    area_inlet: float = field(init=False)
    area_throat: float = field(init=False)
    v_throat: float = field(init=False)
    mass_flow_rate: float = field(init=False)
    volumetric_flow_rate: float = field(init=False)
    Re_D: float = field(init=False)
    Re_throat: float = field(init=False)
    bernoulli_dp: float = field(init=False)
    cavitation_number: float = field(init=False)
    is_cavitation_risk: bool = field(init=False)
    flow_regime: str = field(init=False)

    def __init__(
        self,
        # Geometric parameters
        D: Optional[float] = None,
        D_inlet: Optional[float] = None,
        beta: float = 0.50,
        alpha_conv_deg: float = 21.0,
        alpha_div_deg: float = 8.0,
        L_inlet_ratio: Optional[float] = None,
        L_inlet_factor: Optional[float] = None,
        L_throat_ratio: Optional[float] = None,
        L_throat_factor: Optional[float] = None,
        L_outlet_ratio: Optional[float] = None,
        L_outlet_factor: Optional[float] = None,
        use_blend_radii: bool = True,
        blend_R1_ratio: float = 0.2,
        blend_R2_ratio: float = 0.2,
        blend_R3_ratio: float = 0.2,
        blend_R4_ratio: float = 0.2,
        geometry_type: str = "iso5167",
        # Flow / Operating conditions
        v_inlet: float = 1.0,
        p_inlet: float = 101325.0,
        p_outlet: float = 101325.0,
        p_throat_target: Optional[float] = None,
        # Fluid properties
        fluid: Optional[str] = None,
        fluid_name: Optional[str] = None,
        rho: Optional[float] = None,
        mu: Optional[float] = None,
        nu: Optional[float] = None,
        p_vap: Optional[float] = None,
        # Mesh discretization
        Nz: int = 120,
        Nr: int = 40,
        r_clustering: Optional[float] = None,
        wall_clustering: Optional[float] = None,
        radial_distribution: str = "tanh",
        axial_distribution: str = "uniform",
        # Solver parameters
        cfl: Optional[float] = None,
        cfl_max: Optional[float] = None,
        max_iter: int = 5000,
        tol: Optional[float] = None,
        tol_steady: Optional[float] = None,
        poisson_tol: float = 1e-6,
        poisson_max_iter: int = 1000,
        poisson_solver: str = "sor",
        sor_omega: float = 1.85,
        advection_scheme: str = "upwind",
        turbulence_model: str = "baldwin_lomax",
        output_dir: str = "output",
        **kwargs: Any,
    ):
        """
        Initialize VenturiConfig supporting both standard and legacy alias arguments.
        """
        # Diameter aliases
        if D is not None:
            self.D = float(D)
        elif D_inlet is not None:
            self.D = float(D_inlet)
        else:
            self.D = 0.10

        self.beta = float(beta)
        if self.D <= 0:
            raise ValueError(f"Pipe diameter D must be strictly positive, got {self.D}")
        if self.beta <= 0 or self.beta > 1.0:
            raise ValueError(f"Throat diameter ratio beta must be in (0.0, 1.0], got {self.beta}")

        self.alpha_conv_deg = float(alpha_conv_deg)
        self.alpha_div_deg = float(alpha_div_deg)

        # Length ratio aliases
        if L_inlet_ratio is not None:
            self.L_inlet_ratio = float(L_inlet_ratio)
        elif L_inlet_factor is not None:
            self.L_inlet_ratio = float(L_inlet_factor)
        else:
            self.L_inlet_ratio = 1.5

        if L_throat_ratio is not None:
            self.L_throat_ratio = float(L_throat_ratio)
        elif L_throat_factor is not None:
            self.L_throat_ratio = float(L_throat_factor)
        else:
            self.L_throat_ratio = 1.0

        if L_outlet_ratio is not None:
            self.L_outlet_ratio = float(L_outlet_ratio)
        elif L_outlet_factor is not None:
            self.L_outlet_ratio = float(L_outlet_factor)
        else:
            self.L_outlet_ratio = 3.0

        self.use_blend_radii = bool(use_blend_radii)
        self.blend_R1_ratio = float(blend_R1_ratio)
        self.blend_R2_ratio = float(blend_R2_ratio)
        self.blend_R3_ratio = float(blend_R3_ratio)
        self.blend_R4_ratio = float(blend_R4_ratio)
        self.geometry_type = str(geometry_type)

        # Flow conditions
        self.v_inlet = float(v_inlet)
        self.p_inlet = float(p_inlet)
        self.p_outlet = float(p_outlet)
        self.p_throat_target = float(p_throat_target) if p_throat_target is not None else None

        # Fluid selection
        chosen_fluid = fluid or fluid_name or "water_20C"
        self.fluid = chosen_fluid

        # Fluid property lookup or custom assignment
        if chosen_fluid != "custom" and chosen_fluid in FLUID_DATABASE:
            db_fluid = FLUID_DATABASE[chosen_fluid]
            self.rho = float(rho if rho is not None else db_fluid.rho)
            self.mu = float(mu if mu is not None else db_fluid.mu)
            self.nu = float(nu if nu is not None else (self.mu / self.rho))
            self.p_vap = float(p_vap if p_vap is not None else db_fluid.p_vap)
        else:
            self.rho = float(rho if rho is not None else 998.2)
            self.mu = float(mu if mu is not None else 1.002e-3)
            self.nu = float(nu if nu is not None else (self.mu / self.rho))
            self.p_vap = float(p_vap if p_vap is not None else 0.0)

        # Mesh discretization
        self.Nz = int(Nz)
        self.Nr = int(Nr)
        if r_clustering is not None:
            self.r_clustering = float(r_clustering)
        elif wall_clustering is not None:
            self.r_clustering = float(wall_clustering)
        else:
            self.r_clustering = 1.5

        self.radial_distribution = str(radial_distribution)
        self.axial_distribution = str(axial_distribution)

        # Numerical solver
        if cfl is not None:
            self.cfl = float(cfl)
        elif cfl_max is not None:
            self.cfl = float(cfl_max)
        else:
            self.cfl = 0.5

        self.max_iter = int(max_iter)
        if tol is not None:
            self.tol = float(tol)
        elif tol_steady is not None:
            self.tol = float(tol_steady)
        else:
            self.tol = 1e-5

        self.poisson_tol = float(poisson_tol)
        self.poisson_max_iter = int(poisson_max_iter)
        self.poisson_solver = str(poisson_solver)
        self.sor_omega = float(sor_omega)
        self.advection_scheme = str(advection_scheme)
        self.turbulence_model = str(turbulence_model)
        self.output_dir = str(output_dir)

        # Compute derived values
        self._compute_derived()

    # -----------------------------------------------------------------
    # Backward Compatibility Properties
    # -----------------------------------------------------------------
    @property
    def D_inlet(self) -> float:
        return self.D

    @D_inlet.setter
    def D_inlet(self, value: float) -> None:
        self.D = float(value)
        self._compute_derived()

    @property
    def d_throat(self) -> float:
        return self.d

    @property
    def wall_clustering(self) -> float:
        return self.r_clustering

    @wall_clustering.setter
    def wall_clustering(self, value: float) -> None:
        self.r_clustering = float(value)

    @property
    def L_inlet_factor(self) -> float:
        return self.L_inlet_ratio

    @L_inlet_factor.setter
    def L_inlet_factor(self, value: float) -> None:
        self.L_inlet_ratio = float(value)

    @property
    def L_throat_factor(self) -> float:
        return self.L_throat_ratio

    @L_throat_factor.setter
    def L_throat_factor(self, value: float) -> None:
        self.L_throat_ratio = float(value)

    @property
    def L_outlet_factor(self) -> float:
        return self.L_outlet_ratio

    @L_outlet_factor.setter
    def L_outlet_factor(self, value: float) -> None:
        self.L_outlet_ratio = float(value)

    @property
    def cfl_max(self) -> float:
        return self.cfl

    @cfl_max.setter
    def cfl_max(self, value: float) -> None:
        self.cfl = float(value)

    @property
    def tol_steady(self) -> float:
        return self.tol

    @tol_steady.setter
    def tol_steady(self, value: float) -> None:
        self.tol = float(value)

    @property
    def fluid_name(self) -> str:
        return self.fluid

    @fluid_name.setter
    def fluid_name(self, value: str) -> None:
        self.fluid = str(value)
        if value in FLUID_DATABASE:
            db_fluid = FLUID_DATABASE[value]
            self.rho = db_fluid.rho
            self.mu = db_fluid.mu
            self.nu = db_fluid.nu
            self.p_vap = db_fluid.p_vap
        self._compute_derived()

    @property
    def Q(self) -> float:
        return self.volumetric_flow_rate

    # -----------------------------------------------------------------
    # Analytical Calculations & Derived Properties
    # -----------------------------------------------------------------
    def _compute_derived(self) -> None:
        """
        Compute geometric, 1D Bernoulli kinematic, Reynolds, and cavitation derived quantities.
        """
        # Geometric dimensions
        self.d = self.beta * self.D
        self.R_inlet = self.D / 2.0
        self.R_throat = self.d / 2.0
        self.alpha_conv_rad = math.radians(self.alpha_conv_deg)
        self.alpha_div_rad = math.radians(self.alpha_div_deg)
        self.area_inlet = math.pi * (self.R_inlet ** 2)
        self.area_throat = math.pi * (self.R_throat ** 2)

        # Kinematic viscosity
        if self.rho > 0:
            self.nu = self.mu / self.rho

        # 1D Bernoulli & Continuity Velocity Calculations
        beta4 = self.beta ** 4
        if self.p_throat_target is not None:
            dp = self.p_inlet - self.p_throat_target
            if dp > 0:
                self.v_throat = math.sqrt(2.0 * dp / (self.rho * (1.0 - beta4)))
                self.v_inlet = self.v_throat * (self.beta ** 2)
                self.bernoulli_dp = dp
            else:
                # Fallback for non-positive target pressure difference
                self.v_inlet = 0.1
                self.v_throat = self.v_inlet / (self.beta ** 2)
                self.bernoulli_dp = 0.5 * self.rho * (self.v_inlet ** 2) * (1.0 / beta4 - 1.0)
        else:
            # Velocity-driven flow
            self.v_throat = self.v_inlet / (self.beta ** 2)
            self.bernoulli_dp = 0.5 * self.rho * (self.v_inlet ** 2) * (1.0 / beta4 - 1.0)

        # Flow rates
        self.volumetric_flow_rate = self.area_inlet * self.v_inlet
        self.mass_flow_rate = self.rho * self.volumetric_flow_rate

        # Reynolds numbers
        self.Re_D = (self.rho * self.v_inlet * self.D) / (self.mu + 1e-15)
        self.Re_throat = (self.rho * self.v_throat * self.d) / (self.mu + 1e-15)

        # Flow regime classification
        if self.Re_D <= 2300:
            self.flow_regime = "laminar"
        elif self.Re_D < 4000:
            self.flow_regime = "transitional"
        else:
            self.flow_regime = "turbulent"

        # Cavitation assessment
        p_throat_static = self.p_inlet - self.bernoulli_dp
        q_throat = 0.5 * self.rho * (self.v_throat ** 2)
        if q_throat > 0:
            self.cavitation_number = (p_throat_static - self.p_vap) / q_throat
        else:
            self.cavitation_number = float("inf")

        self.is_cavitation_risk = (self.cavitation_number < 0.2) or (p_throat_static <= self.p_vap)

    # -----------------------------------------------------------------
    # Verification & Reporting Methods
    # -----------------------------------------------------------------
    def summary(self) -> str:
        """
        Generate a comprehensive, human-readable summary of the configuration.
        """
        fluid_desc = FLUID_DATABASE.get(self.fluid, FluidProperties(
            name=self.fluid, rho=self.rho, mu=self.mu, nu=self.nu, description="Custom Fluid"
        )).description

        lines = [
            "=" * 70,
            "VENTURI TUBE SYSTEM CONFIGURATION",
            "=" * 70,
            f"  Fluid Medium:          {self.fluid} ({fluid_desc})",
            f"  Density (rho):         {self.rho:.2f} kg/m^3",
            f"  Dynamic Visc (mu):     {self.mu:.4e} Pa*s",
            f"  Kinematic Visc (nu):   {self.nu:.4e} m^2/s",
            f"  Vapor Pressure:        {self.p_vap:.1f} Pa",
            "-" * 70,
            f"  Inlet Diameter (D):    {self.D * 1000:.2f} mm",
            f"  Throat Diameter (d):   {self.d * 1000:.2f} mm (beta = {self.beta:.3f})",
            f"  Conv Angle (alpha_1):  {self.alpha_conv_deg:.1f}° (total cone)",
            f"  Div Angle (alpha_2):   {self.alpha_div_deg:.1f}° (total cone)",
            f"  Blending Arcs (R1-R4): {'Enabled (ISO 5167-4)' if self.use_blend_radii else 'Sharp Corners'}",
            "-" * 70,
            f"  Inlet Velocity:        {self.v_inlet:.4f} m/s",
            f"  Throat Velocity:       {self.v_throat:.4f} m/s",
            f"  Inlet Pressure:        {self.p_inlet:.1f} Pa",
            f"  Bernoulli dp (1D):     {self.bernoulli_dp:.1f} Pa",
            f"  Volumetric Flow (Q):   {self.volumetric_flow_rate * 1e3:.4f} L/s ({self.volumetric_flow_rate * 3600:.3f} m^3/h)",
            f"  Mass Flow Rate (m_dot):{self.mass_flow_rate:.4f} kg/s",
            f"  Inlet Reynolds (Re_D): {self.Re_D:.0f} [{self.flow_regime.upper()}]",
            f"  Throat Reynolds (Re_d):{self.Re_throat:.0f}",
            f"  Cavitation Index (sig):{self.cavitation_number:.3f} (Risk: {'YES - CAVITATION LIKELY' if self.is_cavitation_risk else 'No'})",
            "-" * 70,
            f"  Mesh Resolution:       Nz = {self.Nz}, Nr = {self.Nr} ({self.Nz * self.Nr} nodes)",
            f"  Wall Clustering (tanh):{self.r_clustering:.2f}",
            f"  Turbulence Model:      {self.turbulence_model}",
            "=" * 70,
        ]
        return "\n".join(lines)

    def validate(self) -> List[str]:
        """
        Validate configuration against ISO 5167-4 standards and physical limits.
        Returns a list of warning/diagnostic messages.
        """
        warnings: List[str] = []

        # Diameter ratio beta range: 0.30 <= beta <= 0.75
        if not (0.30 <= self.beta <= 0.75):
            warnings.append(
                f"[!] beta = {self.beta:.3f} is outside the ISO 5167-4 standard range [0.30, 0.75]."
            )

        # Diameter D range: 0.05 m <= D <= 1.20 m
        if not (0.05 <= self.D <= 1.20):
            warnings.append(
                f"[!] D = {self.D * 1000:.1f} mm is outside the ISO 5167-4 standard range [50 mm, 1200 mm]."
            )

        # Convergent total cone angle: 21 ± 1 deg (or 19 - 23 deg)
        if not (19.0 <= self.alpha_conv_deg <= 23.0):
            warnings.append(
                f"[!] alpha_conv = {self.alpha_conv_deg:.1f}° is outside standard ISO 5167 range (21° ± 1°)."
            )

        # Divergent total cone angle: 7 – 15 deg
        if not (7.0 <= self.alpha_div_deg <= 15.0):
            warnings.append(
                f"[!] alpha_div = {self.alpha_div_deg:.1f}° is outside standard ISO 5167 range (7° – 15°)."
            )

        # Reynolds number regime check
        if self.Re_D > 2300:
            warnings.append(
                f"[!] Re_D = {self.Re_D:.0f} > 2300: Flow is in transitional/turbulent regime. "
                "Ensure turbulence modeling is enabled."
            )

        # ISO 5167 discharge coefficient validity: 2e5 <= Re_D <= 1e6
        if self.Re_D < 2.0e5:
            warnings.append(
                f"[!] Re_D = {self.Re_D:.0f} < 2e5: Standard constant ISO 5167 discharge coefficient Cd ~ 0.995 "
                "may have higher uncertainty."
            )

        # Target throat pressure validity
        if self.p_throat_target is not None and self.p_throat_target >= self.p_inlet:
            warnings.append(
                f"[!] Target throat pressure p_throat ({self.p_throat_target:.0f} Pa) >= p_inlet "
                f"({self.p_inlet:.0f} Pa) -- non-physical negative or zero pressure drop."
            )

        # Cavitation warning
        if self.is_cavitation_risk:
            warnings.append(
                f"[!] High cavitation risk detected: cavitation index sigma = {self.cavitation_number:.3f} < 0.2 "
                f"or throat static pressure ({self.p_inlet - self.bernoulli_dp:.0f} Pa) <= p_vap ({self.p_vap:.0f} Pa)."
            )

        # Discretization resolution check
        if self.Nz < 20:
            warnings.append(f"[!] Axial mesh nodes Nz = {self.Nz} < 20 may produce under-resolved gradients.")
        if self.Nr < 10:
            warnings.append(f"[!] Radial mesh nodes Nr = {self.Nr} < 10 may under-resolve boundary layers.")

        return warnings
