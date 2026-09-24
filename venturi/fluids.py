"""
fluids.py
=========
Table of preset fluids.

Every entry holds the properties at one temperature and at the reference
pressure P_REF (1 atm). For liquids those values are used as they are.
For gases the density is rescaled with the inlet pressure (see
`density_at`), because a gas at 2 bar is twice as dense as at 1 bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

P_REF = 101325.0    # reference pressure of the table [Pa]


@dataclass(frozen=True)
class Fluid:
    """Physical properties of one fluid.

    Attributes:
        rho:    density at P_REF [kg/m^3]
        mu:     dynamic viscosity [Pa s]
        p_vap:  vapour pressure [Pa] (0 for gases: they cannot cavitate)
        temp_C: temperature the values refer to [deg C]
        is_gas: True for gases (density follows pressure)
        description: one line for the fluid list
    """
    rho: float
    mu: float
    p_vap: float
    temp_C: float
    is_gas: bool
    description: str

    @property
    def nu(self) -> float:
        """Kinematic viscosity at P_REF [m^2/s]."""
        return self.mu / self.rho

    def density_at(self, p: float) -> float:
        """Density at absolute pressure p [Pa].

        Liquids: constant. Gases: ideal gas at the table temperature, so
        rho = rho_ref * p / P_REF. The viscosity of a gas depends almost only
        on temperature, so mu is not rescaled.
        """
        if self.is_gas:
            return self.rho * p / P_REF
        return self.rho


FLUIDS: Dict[str, Fluid] = {
    "water_10C":       Fluid(999.7,   1.307e-3, 1228.0,  10.0, False, "Water at 10 C"),
    "water_20C":       Fluid(998.2,   1.002e-3, 2338.8,  20.0, False, "Water at 20 C"),
    "water_40C":       Fluid(992.2,   0.653e-3, 7384.0,  40.0, False, "Water at 40 C"),
    "water_50C":       Fluid(988.0,   0.547e-3, 12350.0, 50.0, False, "Water at 50 C"),
    "water_60C":       Fluid(983.2,   0.467e-3, 19940.0, 60.0, False, "Water at 60 C"),
    "water_80C":       Fluid(971.8,   0.355e-3, 47390.0, 80.0, False, "Water at 80 C"),
    "seawater":        Fluid(1025.0,  1.070e-3, 2300.0,  20.0, False, "Sea water (35 g/kg salt) at 20 C"),
    "air_0C":          Fluid(1.293,   1.716e-5, 0.0,     0.0,  True,  "Dry air at 0 C"),
    "air_20C":         Fluid(1.204,   1.825e-5, 0.0,     20.0, True,  "Dry air at 20 C"),
    "air_50C":         Fluid(1.092,   1.963e-5, 0.0,     50.0, True,  "Dry air at 50 C"),
    "air_100C":        Fluid(0.946,   2.181e-5, 0.0,     100.0, True, "Dry air at 100 C"),
    "glycerin":        Fluid(1261.0,  1.412,    0.01,    20.0, False, "Pure glycerin at 20 C"),
    "glycerin_50":     Fluid(1130.0,  6.000e-3, 1800.0,  20.0, False, "Glycerin 50 % in water at 20 C"),
    "ethylene_glycol": Fluid(1113.0,  1.610e-2, 8.0,     20.0, False, "Pure ethylene glycol at 20 C"),
    "glycol_50":       Fluid(1082.0,  3.340e-3, 1500.0,  20.0, False, "Ethylene glycol 50 % in water at 20 C"),
    "oil_SAE30":       Fluid(891.0,   0.2386,   0.0,     20.0, False, "Motor oil SAE 30 at 20 C"),
    "oil_VG46":        Fluid(875.0,   0.0402,   0.0,     40.0, False, "Hydraulic oil VG 46 at 40 C"),
    "mercury":         Fluid(13534.0, 1.530e-3, 0.16,    20.0, False, "Mercury at 20 C"),
    "ethanol":         Fluid(789.0,   1.200e-3, 5870.0,  20.0, False, "Pure ethanol at 20 C"),
    "kerosene":        Fluid(810.0,   1.640e-3, 500.0,   20.0, False, "Kerosene / jet fuel at 20 C"),
    "diesel":          Fluid(832.0,   2.600e-3, 1000.0,  20.0, False, "Diesel fuel at 20 C"),
}


def list_fluids() -> str:
    """The fluid table as text, one line per fluid."""
    lines = [f"{'name':<16} {'rho [kg/m3]':>12} {'mu [Pa s]':>11} "
             f"{'p_vap [Pa]':>11} {'gas':>4}   description",
             "-" * 90]
    for name, f in FLUIDS.items():
        lines.append(f"{name:<16} {f.rho:>12.3f} {f.mu:>11.3e} {f.p_vap:>11.1f} "
                     f"{'yes' if f.is_gas else 'no':>4}   {f.description}")
    return "\n".join(lines)
