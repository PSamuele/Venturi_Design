"""
config.py
=========
All the inputs of a run, plus the quantities derived from them.

Sizing
------
The user gives two pressures, p_inlet and p_throat_target. The flow rate
that produces that pressure drop follows from Bernoulli plus continuity:

    v_throat = C_design * sqrt( 2 * (p_inlet - p_throat) / (rho * (1 - beta^4)) )
    v_inlet  = v_throat * beta^2

C_design = 1 is the ideal, frictionless flow. A real tube loses a little
energy, so at that flow rate the computed pressure drop comes out a few
percent larger than requested. Setting C_design to the discharge
coefficient you expect (for example 0.98) compensates for that.

If p_throat_target is None, v_inlet is used directly instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from .fluids import FLUIDS

# Air as an ideal gas, used only for the Mach number warning.
# Speed of sound a = sqrt(GAMMA * R * T), T in kelvin.
GAMMA_AIR = 1.4
R_AIR = 287.05          # [J/(kg K)]


@dataclass
class VenturiConfig:
    # --- geometry -----------------------------------------------------------
    D: float = 0.10                  # inlet and outlet pipe diameter [m]
    beta: float = 0.50               # throat diameter / pipe diameter [-]
    alpha_conv_deg: float = 21.0     # total angle of the converging cone [deg]
    alpha_div_deg: float = 8.0       # total angle of the diverging cone [deg]
    L_inlet_ratio: float = 1.5       # straight inlet length / D
    L_throat_ratio: float = 1.0      # throat length / d
    L_outlet_ratio: float = 3.0      # straight outlet length / D
    use_blend_radii: bool = True     # round the four corners of the profile
    blend_R1_ratio: float = 0.2      # corner radius inlet -> cone, / D
    blend_R2_ratio: float = 0.2      # corner radius cone -> throat, / d
    blend_R3_ratio: float = 0.2      # corner radius throat -> cone, / d
    blend_R4_ratio: float = 0.2      # corner radius cone -> outlet, / D

    # --- operating point ----------------------------------------------------
    p_inlet: float = 101325.0                  # upstream pressure [Pa]
    p_throat_target: Optional[float] = 95000.0 # wanted throat pressure [Pa]
    v_inlet: float = 1.0                       # used only if p_throat_target is None
    cd_design: float = 1.0                     # C_design in the sizing formula

    # --- fluid --------------------------------------------------------------
    fluid: str = "water_20C"         # key of FLUIDS, or "custom"
    rho: Optional[float] = None      # overrides the table (required for custom)
    mu: Optional[float] = None
    p_vap: Optional[float] = None

    # --- grid and solver ----------------------------------------------------
    Nz: int = 90                     # cells along the axis
    Nr: int = 36                     # cells along the radius
    wall_clustering: float = 2.7     # 0 = uniform, larger = thinner wall cells
    throat_refine: float = 4.0       # throat cells this many times shorter than the rest
    turbulence_model: str = "baldwin_lomax"
    cfl: float = 0.6
    max_iter: int = 400000
    tol: float = 1e-3                # steady-state residual target
    output_dir: str = "results"

    # --- derived (filled by __post_init__) ----------------------------------
    d: float = field(init=False)
    R_inlet: float = field(init=False)
    R_throat: float = field(init=False)
    alpha_conv_rad: float = field(init=False)
    alpha_div_rad: float = field(init=False)
    area_inlet: float = field(init=False)
    area_throat: float = field(init=False)
    is_gas: bool = field(init=False)
    temp_C: float = field(init=False)
    nu: float = field(init=False)
    v_throat: float = field(init=False)
    Q: float = field(init=False)
    mass_flow_rate: float = field(init=False)
    Re_D: float = field(init=False)
    Re_throat: float = field(init=False)
    dp_ideal: float = field(init=False)
    p_throat_ideal: float = field(init=False)
    cavitation_number: float = field(init=False)
    mach_throat: float = field(init=False)
    flow_regime: str = field(init=False)

    def __post_init__(self) -> None:
        self._check_inputs()
        self._set_fluid()
        self._compute_derived()

    # ------------------------------------------------------------------------
    def _check_inputs(self) -> None:
        if self.D <= 0.0:
            raise ValueError(f"D must be positive, got {self.D}.")
        if not 0.0 < self.beta < 1.0:
            raise ValueError(f"beta must be between 0 and 1 (excluded), got {self.beta}.")
        for name in ("alpha_conv_deg", "alpha_div_deg"):
            a = getattr(self, name)
            if not 0.0 < a < 180.0:
                raise ValueError(f"{name} must be between 0 and 180 deg, got {a}.")
        if self.p_throat_target is not None and self.p_throat_target >= self.p_inlet:
            raise ValueError(
                f"The throat pressure ({self.p_throat_target:.0f} Pa) must be lower "
                f"than the inlet pressure ({self.p_inlet:.0f} Pa): the flow speeds "
                f"up in the throat, so its pressure drops.")
        if not 0.0 < self.cd_design <= 1.0:
            raise ValueError(f"cd_design must be in (0, 1], got {self.cd_design}.")
        if self.throat_refine < 1.0:
            raise ValueError(f"throat_refine must be >= 1, got {self.throat_refine}.")
        if self.Nz < 2 or self.Nr < 3:
            raise ValueError(f"The grid needs Nz >= 2 and Nr >= 3 (got {self.Nz} x {self.Nr}).")

    def _set_fluid(self) -> None:
        if self.fluid == "custom":
            if self.rho is None or self.mu is None:
                raise ValueError("fluid 'custom' needs both rho and mu.")
            self.is_gas = False
            self.temp_C = 20.0
            if self.p_vap is None:
                self.p_vap = 0.0
            return
        if self.fluid not in FLUIDS:
            raise ValueError(f"Unknown fluid '{self.fluid}'. "
                             f"Known: {', '.join(FLUIDS)} or 'custom'.")
        f = FLUIDS[self.fluid]
        self.is_gas = f.is_gas
        self.temp_C = f.temp_C
        if self.rho is None:
            # Gases: density at the inlet pressure, not at the 1 atm of the
            # table (ideal gas, same temperature). The solver keeps it
            # constant along the tube; see the Mach warning below.
            self.rho = f.density_at(self.p_inlet)
        if self.mu is None:
            self.mu = f.mu
        if self.p_vap is None:
            self.p_vap = f.p_vap

    def _compute_derived(self) -> None:
        self.rho = float(self.rho)
        self.mu = float(self.mu)
        self.nu = self.mu / self.rho

        self.d = self.beta * self.D
        self.R_inlet = 0.5 * self.D
        self.R_throat = 0.5 * self.d
        self.alpha_conv_rad = math.radians(self.alpha_conv_deg)
        self.alpha_div_rad = math.radians(self.alpha_div_deg)
        self.area_inlet = math.pi * self.R_inlet ** 2
        self.area_throat = math.pi * self.R_throat ** 2

        k = 1.0 - self.beta ** 4
        if self.p_throat_target is not None:
            dp = self.p_inlet - self.p_throat_target
            self.v_throat = self.cd_design * math.sqrt(2.0 * dp / (self.rho * k))
            self.v_inlet = self.v_throat * self.beta ** 2
        else:
            self.v_throat = self.v_inlet / self.beta ** 2

        # Pressure drop that frictionless flow would have at this flow rate
        self.dp_ideal = 0.5 * self.rho * self.v_throat ** 2 * k
        self.p_throat_ideal = self.p_inlet - self.dp_ideal

        self.Q = self.area_inlet * self.v_inlet
        self.mass_flow_rate = self.rho * self.Q
        self.Re_D = self.rho * self.v_inlet * self.D / self.mu
        self.Re_throat = self.rho * self.v_throat * self.d / self.mu

        if self.Re_D <= 2300.0:
            self.flow_regime = "laminar"
        elif self.Re_D < 4000.0:
            self.flow_regime = "transitional"
        else:
            self.flow_regime = "turbulent"

        q_th = 0.5 * self.rho * self.v_throat ** 2
        self.cavitation_number = (self.p_throat_ideal - self.p_vap) / q_th
        if self.is_gas:
            a = math.sqrt(GAMMA_AIR * R_AIR * (self.temp_C + 273.15))
            self.mach_throat = self.v_throat / a
        else:
            self.mach_throat = 0.0

    # ------------------------------------------------------------------------
    def summary(self) -> str:
        """Human-readable summary of inputs and derived values."""
        lines = [
            "=" * 66,
            f"  fluid             {self.fluid}",
            f"  rho               {self.rho:.4g} kg/m3",
            f"  mu                {self.mu:.4e} Pa s",
            f"  nu                {self.nu:.4e} m2/s",
            f"  p_vap             {self.p_vap:.1f} Pa",
            "-" * 66,
            f"  D                 {self.D*1000:.2f} mm",
            f"  d                 {self.d*1000:.2f} mm   (beta = {self.beta:.3f})",
            f"  alpha_conv        {self.alpha_conv_deg:.1f} deg (total cone angle)",
            f"  alpha_div         {self.alpha_div_deg:.1f} deg (total cone angle)",
            f"  rounded corners   {'yes' if self.use_blend_radii else 'no'}",
            "-" * 66,
            f"  v_inlet           {self.v_inlet:.4f} m/s",
            f"  v_throat          {self.v_throat:.4f} m/s",
            f"  Q                 {self.Q*1e3:.4f} L/s  ({self.Q*3600:.3f} m3/h)",
            f"  mass flow         {self.mass_flow_rate:.4f} kg/s",
            f"  p_inlet           {self.p_inlet:.1f} Pa",
            f"  ideal dp          {self.dp_ideal:.1f} Pa   (C_design = {self.cd_design:.3f})",
            f"  Re_D              {self.Re_D:.0f}  ({self.flow_regime})",
            f"  Re_d (throat)     {self.Re_throat:.0f}",
            f"  sigma             {self.cavitation_number:.3f}   (cavitation number)",
        ]
        if self.is_gas:
            lines.append(f"  Mach (throat)     {self.mach_throat:.3f}")
        lines += [
            "-" * 66,
            f"  grid              {self.Nz} x {self.Nr} = {self.Nz*self.Nr} cells, "
            f"wall clustering {self.wall_clustering:.2f}, "
            f"throat refinement {self.throat_refine:.1f}",
            f"  turbulence model  {self.turbulence_model}",
            "=" * 66,
        ]
        return "\n".join(lines)

    def warnings(self) -> List[str]:
        """Physical and numerical warnings. None of them stops the run."""
        w: List[str] = []
        if self.turbulence_model == "laminar" and self.Re_D > 2300.0:
            w.append(f"Re_D = {self.Re_D:.0f} > 2300 but the turbulence model is "
                     f"'laminar': the flow is almost certainly turbulent.")
        if self.turbulence_model != "laminar" and self.Re_D <= 2300.0:
            w.append(f"Re_D = {self.Re_D:.0f} <= 2300: the flow is laminar, "
                     f"consider --turbulence laminar.")
        if self.p_throat_ideal <= self.p_vap:
            w.append(f"Throat pressure from Bernoulli ({self.p_throat_ideal:.0f} Pa) is "
                     f"at or below the vapour pressure ({self.p_vap:.0f} Pa): the "
                     f"liquid will boil (cavitate) and this solver cannot model it.")
        if self.is_gas and self.mach_throat > 0.3:
            w.append(f"Throat Mach number {self.mach_throat:.2f} > 0.3: the gas "
                     f"density changes by about "
                     f"{100*self.dp_ideal/self.p_inlet:.1f} % between inlet and throat, "
                     f"but this solver assumes constant density.")
        if self.Nz < 20:
            w.append(f"Nz = {self.Nz} < 20: the axial gradients are poorly resolved.")
        if self.Nr < 10:
            w.append(f"Nr = {self.Nr} < 10: the boundary layer is poorly resolved.")
        return w
