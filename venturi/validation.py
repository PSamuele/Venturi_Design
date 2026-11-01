"""
validation.py
=============
Checks on a finished run, and the discharge coefficient C_d.

Discharge coefficient
---------------------
C_d is the real flow rate divided by the flow rate that frictionless flow
would give for the same measured pressure drop:

    C_d = Q / [ A_throat * sqrt( 2*dp / (rho*(1-beta^4)) ) ]

It is always a bit below 1, because friction makes the real pressure drop
larger than the ideal one. dp is measured between two pressure taps:

    upstream tap : 0.5 D before the start of the converging cone
    throat tap   : middle of the throat

Both taps read the pressure at the wall, like the small holes of a real
tube. The difference of the section-averaged pressures is also reported.

The upstream tap sits before the cone rather than at the domain inlet, so
the friction loss of the inlet pipe run is not counted.

C_d is always computed and reported. It becomes a pass/fail check only if
you give a reference value (cd_ref) to compare against, for example a
value from a datasheet or from your own measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .fvmesh import divergence

# Acceptance thresholds
TOL_MASS_PCT = 0.1        # inlet/outlet flow rate mismatch [%]
TOL_DIV_NONDIM = 1.0e-9   # leftover divergence, relative to the cell fluxes
TOL_YPLUS = 5.0           # the wall models need the first cell at y+ below this


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
    Cd_ref: Optional[float] = None
    dp_measured: float = 0.0
    dp_section_mean: float = 0.0
    dp_ideal: float = 0.0
    pressure_recovery: float = 0.0
    dp_loss: float = 0.0          # pressure not recovered, upstream tap -> outlet
    loss_coef: float = 0.0        # dp_loss / (0.5 * rho * v_inlet^2)
    p_up: float = 0.0             # upstream tap
    p_throat: float = 0.0         # throat tap
    p_out: float = 0.0            # last cell at the wall
    p_min: float = 0.0
    z_p_min: float = 0.0          # where p_min is, from the inlet [m]
    dp_max: float = 0.0           # p_inlet - p_min
    cavitation_margin: float = 0.0  # p_min - p_vap
    u_max: float = 0.0            # highest speed in the field
    Q: float = 0.0
    v_inlet: float = 0.0
    v_throat: float = 0.0
    Re_D: float = 0.0
    Re_throat: float = 0.0
    sigma: float = 0.0            # cavitation number
    y_plus_max: float = 0.0

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)


def section_mean(p: np.ndarray, mesh, z: float) -> float:
    """Pressure averaged over the cross-section (volume weighted) at position z.

    Interpolated along z like the taps: the nearest row would move the
    section by up to half a cell, a different amount on every grid.
    """
    means = np.sum(p * mesh.vol, axis=1) / np.sum(mesh.vol, axis=1)
    return float(np.interp(z, mesh.z_c, means))


def tap_pressure(p: np.ndarray, mesh, z: float) -> float:
    """Pressure a tap drilled in the wall at axial position z would read.

    A real tap is a small hole in the wall, so it reads the pressure AT the
    wall. The pressure hardly changes across the thin wall cell (the flow
    there is parallel to the wall), so the wall-cell values are used,
    interpolated linearly along z to the exact tap position. Taking the
    nearest cell instead would move the tap by up to half a cell, and in the
    throat the wall pressure changes quickly near the corners.
    """
    return float(np.interp(z, mesh.z_c, p[:, -1]))


def validate(result, mesh, geom, config, cd_ref: Optional[float] = None,
             cd_tol_pct: float = 3.0) -> ValidationReport:
    """Run all checks on a solver result.

    Args:
        result: FVResult from solve_fv.
        mesh: FVMesh.
        geom: VenturiGeometry (for the tap positions).
        config: VenturiConfig.
        cd_ref: reference C_d. If given, C_d must be within cd_tol_pct of it.
        cd_tol_pct: allowed deviation from cd_ref [%].
    """
    rep = ValidationReport(Re_D=float(config.Re_D), Cd_ref=cd_ref,
                           Re_throat=float(config.Re_throat), Q=float(config.Q),
                           v_inlet=float(config.v_inlet), v_throat=float(config.v_throat),
                           sigma=float(config.cavitation_number),
                           y_plus_max=float(result.y_plus_max))
    D, beta, rho = float(config.D), float(config.beta), float(config.rho)

    # ------------------------------------------------------- pressure taps
    zs = geom.z_stations
    z_up = zs[1] - 0.5 * D
    z_th = 0.5 * (zs[2] + zs[3])
    p_up = tap_pressure(result.p, mesh, z_up)
    p_th = tap_pressure(result.p, mesh, z_th)
    p_out = tap_pressure(result.p, mesh, mesh.z_c[-1])
    dp = p_up - p_th
    rep.dp_measured = dp
    rep.dp_section_mean = (section_mean(result.p, mesh, z_up)
                           - section_mean(result.p, mesh, z_th))
    rep.dp_ideal = float(config.dp_ideal)
    rep.pressure_recovery = (p_out - p_th) / dp if dp != 0.0 else 0.0
    rep.p_up, rep.p_throat, rep.p_out = p_up, p_th, p_out
    rep.dp_loss = p_up - p_out
    rep.loss_coef = rep.dp_loss / (0.5 * rho * float(config.v_inlet) ** 2)
    rep.u_max = float(np.hypot(result.uz, result.ur).max())

    # ------------------------------------------------------- mass balance
    Q_in = float(result.F_ax[0, :].sum())
    Q_out = float(result.F_ax[mesh.Nz, :].sum())
    mass_err = abs(Q_in - Q_out) / abs(Q_in) * 100.0
    rep.checks.append(Check(
        "Inlet/outlet mass balance", mass_err, f"< {TOL_MASS_PCT} %",
        mass_err < TOL_MASS_PCT, "flow rate mismatch [%]"))

    # -------------------------------------------------- leftover divergence
    # Each cell's imbalance is compared with the fluxes crossing it, so thin
    # wall cells and big core cells are judged on the same scale.
    div_cell = np.abs(divergence(result.F_ax, result.F_rad, mesh))
    flux_cell = (np.abs(result.F_ax[:-1, :]) + np.abs(result.F_ax[1:, :])
                 + np.abs(result.F_rad[:, :-1]) + np.abs(result.F_rad[:, 1:]))
    div_nd = float(np.max(div_cell / (flux_cell + 1e-300)))
    rep.checks.append(Check(
        "Leftover divergence", div_nd, f"< {TOL_DIV_NONDIM:.0e}",
        div_nd < TOL_DIV_NONDIM, "cell imbalance / fluxes through the cell"))

    # ------------------------------------------------------------ convergence
    rep.checks.append(Check(
        "Steady state reached", float(result.final_residual),
        f"< {float(config.tol):.0e}", bool(result.converged),
        "residual |du/dt| * L / U^2"))

    # -------------------------------------------------------- wall resolution
    if config.turbulence_model != "laminar":
        rep.checks.append(Check(
            "First cell y+", float(result.y_plus_max), f"< {TOL_YPLUS}",
            result.y_plus_max < TOL_YPLUS,
            "wall cells thin enough for the turbulence model"))

    # --------------------------------------------------------- cavitation
    rep.p_min = float(result.p.min())
    rep.z_p_min = float(mesh.z_c[np.unravel_index(np.argmin(result.p), result.p.shape)[0]])
    rep.dp_max = float(config.p_inlet) - rep.p_min
    rep.cavitation_margin = rep.p_min - float(config.p_vap)
    if config.p_vap > 0.0:
        rep.checks.append(Check(
            "No cavitation", rep.p_min, f"> p_vap = {config.p_vap:.0f} Pa",
            rep.p_min > config.p_vap, "lowest pressure in the whole field [Pa]"))

    # --------------------------------------------------- discharge coefficient
    A_th = math.pi * geom.R_throat ** 2
    if dp > 0.0:
        rep.Cd = Q_in / (A_th * math.sqrt(2.0 * dp / (rho * (1.0 - beta ** 4))))
    else:
        rep.Cd = float("nan")
        rep.warnings.append("The measured pressure drop is not positive: C_d is undefined.")
    if cd_ref is not None:
        dev = abs(rep.Cd - cd_ref) / cd_ref * 100.0 if rep.Cd == rep.Cd else float("inf")
        rep.checks.append(Check(
            f"C_d vs reference {cd_ref:.4f}", rep.Cd, f"within {cd_tol_pct} %",
            dev < cd_tol_pct, f"deviation {dev:.2f} %"))
    return rep


def print_report(rep: ValidationReport) -> None:
    print("      " + "-" * 60)
    for c in rep.checks:
        print(f"      [{'PASS' if c.passed else 'FAIL'}] {c.name}")
        print(f"             value {c.value:.6g}   target {c.target}")
        if c.note:
            print(f"             {c.note}")
    print("      " + "-" * 60)
    print(f"      C_d (computed)          : {rep.Cd:.4f}")
    print(f"      flow rate               : {rep.Q*1e3:.3f} L/s")
    print(f"      mean speed pipe / throat: {rep.v_inlet:.3f} / {rep.v_throat:.3f} m/s")
    print(f"      highest speed           : {rep.u_max:.3f} m/s")
    print(f"      Re pipe / throat        : {rep.Re_D:.0f} / {rep.Re_throat:.0f}")
    print(f"      upstream tap            : {rep.p_up:.1f} Pa")
    print(f"      throat tap              : {rep.p_throat:.1f} Pa")
    print(f"      outlet                  : {rep.p_out:.1f} Pa")
    print(f"      dp between the wall taps: {rep.dp_measured:.1f} Pa")
    print(f"      dp of section averages  : {rep.dp_section_mean:.1f} Pa (for comparison)")
    print(f"      dp without friction     : {rep.dp_ideal:.1f} Pa (for comparison)")
    print(f"      pressure recovered      : {rep.pressure_recovery*100:.1f} % of dp")
    print(f"      permanent pressure loss : {rep.dp_loss:.1f} Pa (K = {rep.loss_coef:.3f})")
    print(f"      lowest pressure         : {rep.p_min:.1f} Pa, {rep.z_p_min*1e3:.1f} mm from the inlet")
    print(f"      margin to boiling       : {rep.cavitation_margin:.1f} Pa (sigma = {rep.sigma:.3f})")
    for w in rep.warnings:
        print(f"      [WARNING] {w}")


def to_dict(rep: ValidationReport) -> Dict[str, Any]:
    """Flat dictionary for the Markdown report."""
    d: Dict[str, Any] = {}
    for c in rep.checks:
        d[c.name] = {"value": c.value, "passed": c.passed, "target": c.target}
    d["C_d computed"] = round(rep.Cd, 4)
    if rep.Cd_ref is not None:
        d["C_d reference"] = rep.Cd_ref
    d["dp between wall taps [Pa]"] = round(rep.dp_measured, 1)
    d["dp of section averages [Pa]"] = round(rep.dp_section_mean, 1)
    d["dp without friction [Pa]"] = round(rep.dp_ideal, 1)
    d["pressure recovered [%]"] = round(rep.pressure_recovery * 100.0, 1)
    d["permanent pressure loss [Pa]"] = round(rep.dp_loss, 1)
    d["loss coefficient K (on pipe speed)"] = round(rep.loss_coef, 3)
    d["upstream tap pressure [Pa]"] = round(rep.p_up, 1)
    d["throat tap pressure [Pa]"] = round(rep.p_throat, 1)
    d["outlet pressure [Pa]"] = round(rep.p_out, 1)
    d["lowest pressure [Pa]"] = round(rep.p_min, 1)
    d["lowest pressure, from the inlet [mm]"] = round(rep.z_p_min * 1e3, 1)
    d["margin to boiling [Pa]"] = round(rep.cavitation_margin, 1)
    d["cavitation number sigma"] = round(rep.sigma, 3)
    d["mean speed, pipe [m/s]"] = round(rep.v_inlet, 3)
    d["mean speed, throat [m/s]"] = round(rep.v_throat, 3)
    d["highest speed [m/s]"] = round(rep.u_max, 3)
    d["flow rate [L/s]"] = round(rep.Q * 1e3, 3)
    d["Re_D"] = round(rep.Re_D, 0)
    d["Re_throat"] = round(rep.Re_throat, 0)
    for i, w in enumerate(rep.warnings, 1):
        d[f"warning {i}"] = w
    return d
