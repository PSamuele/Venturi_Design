"""
validation_fv.py
================
Automatic validation of the finite-volume solver results.

Primary criterion: the DISCHARGE COEFFICIENT C_d.

Why not a comparison against ideal Bernoulli
--------------------------------------------
Ideal Bernoulli neglects friction, flow separation and velocity-profile
non-uniformity. A correct computation therefore CANNOT match it: the real
pressure drop is systematically larger by a few percent. Demanding agreement
within 5% of ideal Bernoulli declares a right answer "failed" and a
wrong-in-the-other-direction answer "passed".

The discharge coefficient, by contrast, is defined precisely as the ratio of
real to ideal flow rate:

    C_d = Q_real / [ A_throat * sqrt( 2*dp / (rho*(1-beta^4)) ) ]

and it is tabulated by ISO 5167-4 per construction type. Comparing the
computed C_d against the standard is a genuine physical test, because the
reference value is measured experimentally rather than derived from the same
theory being checked.

Pressure taps
-------------
The standard places them upstream of the convergent and at the midpoint of
the throat. Using the domain inlet section instead would fold the pressure
loss of the upstream pipe run into the measurement, artificially lowering
C_d.

ISO 5167-4:2003 reference values
--------------------------------
    "as cast"     C = 0.984   100 <= D <= 800 mm, 0.30 <= beta <= 0.75
    "machined"    C = 0.995    50 <= D <= 350 mm, 0.40 <= beta <= 0.75
    "fabricated"  C = 0.985   (rough-welded sheet-iron)
in all cases for 2e5 <= Re_D <= 2e6. For the machined type the standard gives
C = 1.000 above Re_D = 1e6.

Note: ISO 5167-4:2022 reportedly moved the machined value from 0.995 to
0.990. That comes from a secondary source and has not been verified against
the text of the standard; this module uses the 2003 values and states them
explicitly in the report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# Reference values. Re_max = None means "no upper limit stated for this type
# in the clause cited".
ISO_TYPES: Dict[str, Dict[str, Any]] = {
    "as_cast": dict(Cd=0.984, D_min=0.100, D_max=0.800,
                    beta_min=0.30, beta_max=0.75, Re_min=2.0e5, Re_max=2.0e6,
                    label="as-cast convergent"),
    "machined": dict(Cd=0.995, D_min=0.050, D_max=0.350,
                     beta_min=0.40, beta_max=0.75, Re_min=2.0e5, Re_max=1.0e6,
                     label="machined convergent"),
    "fabricated": dict(Cd=0.985, D_min=None, D_max=None,
                       beta_min=None, beta_max=None, Re_min=2.0e5, Re_max=2.0e6,
                       label="rough-welded sheet iron"),
}

# Acceptance thresholds for the computation
TOL_MASS_PCT = 0.1        # allowed mass-balance error [%]
TOL_DIV_NONDIM = 1.0e-9   # relative residual divergence
TOL_CD_PCT = 3.0          # allowed deviation of C_d from the ISO value [%]
TOL_YPLUS = 5.0           # max y+ for a wall-integrated model


@dataclass
class Check:
    name: str
    value: float
    target: str
    passed: bool
    note: str = ""


@dataclass
class ValidationReport:
    checks: List[Check] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    Cd: float = 0.0
    Cd_iso: Optional[float] = None
    dp_measured: float = 0.0
    dp_ideal: float = 0.0
    pressure_recovery: float = 0.0
    Re_D: float = 0.0
    iso_applicable: bool = False

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _area_weighted(field_2d: np.ndarray, area: np.ndarray, i: int) -> float:
    """Area-weighted mean over section i."""
    return float(np.sum(field_2d[i, :] * area[i, :]) / np.sum(area[i, :]))


def validate(result, mesh, geom, config,
             venturi_type: str = "machined") -> ValidationReport:
    """Validate a solver result.

    Args:
        result: FVResult.
        mesh: FVMesh.
        geom: VenturiGeometry (z_stations is needed to locate the taps).
        config: VenturiConfig.
        venturi_type: one of the ISO_TYPES keys.
    """
    if venturi_type not in ISO_TYPES:
        raise ValueError(f"Unknown construction type '{venturi_type}'. "
                         f"Allowed: {', '.join(ISO_TYPES)}.")
    ref = ISO_TYPES[venturi_type]
    rep = ValidationReport()

    D = float(config.D_inlet)
    beta = float(config.beta)
    rho = float(config.rho)
    Re = float(config.Re_D)
    rep.Re_D = Re

    # ------------------------------------------------------- pressure taps
    zs = np.asarray(geom.z_stations, dtype=float)
    z_tap_up = zs[1] - 0.5 * D            # 0.5 D upstream of the convergent
    z_tap_th = 0.5 * (zs[2] + zs[3])      # throat midpoint
    i_up = int(np.argmin(np.abs(mesh.z_c - z_tap_up)))
    i_th = int(np.argmin(np.abs(mesh.z_c - z_tap_th)))
    i_out = mesh.Nz - 1

    p_up = _area_weighted(result.p, mesh.A_ax[:-1, :], i_up)
    p_th = _area_weighted(result.p, mesh.A_ax[:-1, :], i_th)
    p_out = _area_weighted(result.p, mesh.A_ax[:-1, :], i_out)
    dp = p_up - p_th
    rep.dp_measured = dp
    rep.dp_ideal = float(config.bernoulli_dp)
    rep.pressure_recovery = (p_out - p_th) / dp if dp != 0.0 else 0.0

    # ------------------------------------------------------- mass balance
    Q_in = float(result.F_ax[0, :].sum())
    Q_out = float(result.F_ax[mesh.Nz, :].sum())
    mass_err = abs(Q_in - Q_out) / abs(Q_in) * 100.0
    rep.checks.append(Check(
        "Inlet/outlet mass balance", mass_err,
        f"< {TOL_MASS_PCT} %", mass_err < TOL_MASS_PCT, "relative error [%]"))

    # -------------------------------------------------- residual divergence
    # Relative normalisation: a cell's imbalance is compared against the
    # fluxes crossing it. That is the right measure, because a residual of
    # 1e-17 m3/s means different things in a thin wall cell and in a core one.
    from .fvmesh import divergence as _div
    div_cell = np.abs(_div(result.F_ax, result.F_rad, mesh))
    flux_cell = (np.abs(result.F_ax[:-1, :]) + np.abs(result.F_ax[1:, :])
                 + np.abs(result.F_rad[:, :-1]) + np.abs(result.F_rad[:, 1:]))
    div_nd = float(np.max(div_cell / (flux_cell + 1e-300)))
    rep.checks.append(Check(
        "Relative residual divergence", div_nd,
        f"< {TOL_DIV_NONDIM:.0e}", div_nd < TOL_DIV_NONDIM,
        "cell imbalance divided by the fluxes crossing it"))

    # ------------------------------------------------------------ convergence
    rep.checks.append(Check(
        "Convergence to steady state", float(result.final_residual),
        f"< {float(config.tol_steady):.0e}", bool(result.converged),
        "residual |du/dt|*L/U^2"))

    # -------------------------------------------------------- wall resolution
    rep.checks.append(Check(
        "Maximum wall y+", float(result.y_plus_max),
        f"< {TOL_YPLUS}", result.y_plus_max < TOL_YPLUS,
        "required by a wall-integrated algebraic model"))

    # ---------------------------------------------------- discharge coefficient
    A_th = float(mesh.A_ax[i_th, :].sum())
    if dp > 0.0:
        Cd = Q_in / (A_th * math.sqrt(2.0 * dp / (rho * (1.0 - beta ** 4))))
    else:
        Cd = float("nan")
    rep.Cd = Cd

    # Validity range of the standard
    ok_Re = ref["Re_min"] <= Re <= (ref["Re_max"] or math.inf)
    ok_D = (ref["D_min"] is None) or (ref["D_min"] <= D <= ref["D_max"])
    ok_b = (ref["beta_min"] is None) or (ref["beta_min"] <= beta <= ref["beta_max"])
    rep.iso_applicable = ok_Re and ok_D and ok_b

    if not ok_Re:
        rep.warnings.append(
            f"Re_D = {Re:.0f} is outside the ISO 5167-4 range for type "
            f"'{venturi_type}' ({ref['Re_min']:.0e} - {ref['Re_max']:.0e}). "
            f"The standard C_d is undefined here: the comparison is "
            f"indicative, not certifying.")
    if not ok_D:
        rep.warnings.append(
            f"D = {D*1000:.0f} mm is outside the ISO range "
            f"({ref['D_min']*1000:.0f} - {ref['D_max']*1000:.0f} mm).")
    if not ok_b:
        rep.warnings.append(
            f"beta = {beta:.3f} is outside the ISO range "
            f"({ref['beta_min']:.2f} - {ref['beta_max']:.2f}).")

    Cd_iso = float(ref["Cd"])
    if venturi_type == "machined" and Re > 1.0e6:
        Cd_iso = 1.000
    rep.Cd_iso = Cd_iso
    dev = abs(Cd - Cd_iso) / Cd_iso * 100.0 if Cd == Cd else float("inf")
    rep.checks.append(Check(
        f"C_d vs ISO 5167-4 ({ref['label']}, C={Cd_iso:.3f})", Cd,
        f"deviation < {TOL_CD_PCT} %", dev < TOL_CD_PCT,
        f"measured deviation {dev:.2f} %"
        + ("" if rep.iso_applicable else "  [OUTSIDE VALIDITY RANGE]")))

    return rep


def print_report(rep: ValidationReport) -> None:
    """Print the validation to the console."""
    print("      " + "-" * 66)
    for c in rep.checks:
        flag = "PASS" if c.passed else "FAIL"
        print(f"      [{flag}] {c.name}")
        print(f"             value {c.value:.6g}   target {c.target}")
        if c.note:
            print(f"             {c.note}")
    print("      " + "-" * 66)
    print(f"      dp measured at ISO taps : {rep.dp_measured:.1f} Pa")
    print(f"      ideal Bernoulli dp      : {rep.dp_ideal:.1f} Pa "
          f"(reference only, not a criterion)")
    print(f"      pressure recovery       : {rep.pressure_recovery*100:.1f} %")
    for w in rep.warnings:
        print(f"      [WARNING] {w}")


def to_dict(rep: ValidationReport) -> Dict[str, Any]:
    """Serialise for the Markdown report written by export.py."""
    d: Dict[str, Any] = {}
    for c in rep.checks:
        d[c.name] = {"value": c.value, "passed": c.passed, "target": c.target}
    d["dp measured [Pa]"] = round(rep.dp_measured, 1)
    d["ideal Bernoulli dp [Pa]"] = round(rep.dp_ideal, 1)
    d["pressure recovery [%]"] = round(rep.pressure_recovery * 100.0, 1)
    d["Re_D"] = round(rep.Re_D, 0)
    d["C_d computed"] = round(rep.Cd, 4)
    d["C_d ISO 5167-4"] = rep.Cd_iso
    d["within ISO validity range"] = rep.iso_applicable
    for i, w in enumerate(rep.warnings, 1):
        d[f"warning {i}"] = w
    return d
