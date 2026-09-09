"""
turbulence.py
=============
Algebraic eddy-viscosity models for the axisymmetric solver.

Why they are needed
-------------------
At the Reynolds number of a typical industrial Venturi (Re_D ~ 1e5), momentum
transport near the wall is dominated by turbulent eddies, not by molecular
viscosity: the ratio between the two reaches two or three orders of
magnitude. A laminar computation at that Reynolds predicts a boundary layer
that is too thick and a wall shear that is simply wrong, so it gets both the
pressure recovery in the diffuser and the discharge coefficient wrong too.

Available models
----------------
    "laminar"        nu_t = 0
    "mixing_length"  Prandtl with van Driest damping
    "baldwin_lomax"  two-layer algebraic (Baldwin & Lomax, AIAA 78-257)

The model is selected by config.turbulence_model. An unrecognised string
raises an exception: it is never silently ignored.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Standard Baldwin-Lomax constants
KAPPA = 0.40        # von Karman constant
A_PLUS = 26.0       # van Driest damping constant
C_CP = 1.6
C_KLEB = 0.3
C_WK = 0.25
K_OUTER = 0.0168    # Clauser constant

SUPPORTED = ("laminar", "mixing_length", "baldwin_lomax")


@njit(cache=True)
def _vorticity(uz, ur, r_c, R_c, dRdz_c, z_c, eta_c):
    """Magnitude of the azimuthal vorticity |omega_theta| = |dur/dz - duz/dr|.

    Cell centres in a given column j sit at different radii as i varies (the
    grid follows the wall), so the derivative along a grid line is NOT the
    derivative at constant radius. The chain rule with eta = r/R(z) gives

        df/dz |_r  =  df/dxi |_eta  -  (r * R'/R) * df/dr

    The radial derivative, by contrast, is direct: along a column i every
    centre shares the same z.
    """
    Nz, Nr = uz.shape
    omega = np.zeros((Nz, Nr))

    for i in range(Nz):
        im = max(i - 1, 0)
        ip = min(i + 1, Nz - 1)
        dz = z_c[ip] - z_c[im]
        for j in range(Nr):
            jm = max(j - 1, 0)
            jp = min(j + 1, Nr - 1)
            dr = r_c[i, jp] - r_c[i, jm]

            if dr > 1e-30:
                duz_dr = (uz[i, jp] - uz[i, jm]) / dr
                dur_dr = (ur[i, jp] - ur[i, jm]) / dr
            else:
                duz_dr = 0.0
                dur_dr = 0.0

            if dz > 1e-30:
                dur_dxi = (ur[ip, j] - ur[im, j]) / dz
            else:
                dur_dxi = 0.0

            # metric correction back to constant z
            metric = (r_c[i, j] / R_c[i]) * dRdz_c[i]
            dur_dz = dur_dxi - metric * dur_dr

            omega[i, j] = abs(dur_dz - duz_dr)

    return omega


@njit(cache=True)
def _mixing_length(omega, y, u_tau, nu, delta):
    """Prandtl + van Driest: nu_t = (kappa*y*D)^2 * |omega|, capped at 0.09*delta."""
    Nz, Nr = omega.shape
    nu_t = np.zeros((Nz, Nr))
    for i in range(Nz):
        for j in range(Nr):
            y_plus = y[i, j] * u_tau[i] / nu
            damp = 1.0 - np.exp(-y_plus / A_PLUS)
            length = KAPPA * y[i, j] * damp
            cap = 0.09 * delta[i]
            if length > cap:
                length = cap
            nu_t[i, j] = length * length * omega[i, j]
    return nu_t


@njit(cache=True)
def _baldwin_lomax(omega, y, u_mag, u_tau, nu):
    """Two-layer algebraic model.

    Inner layer (near wall):
        nu_t = (kappa * y * D)^2 * |omega|,   D = 1 - exp(-y+/A+)

    Outer layer:
        F(y)   = y * |omega| * D
        F_max  = maximum of F along the normal, attained at y = y_max
        F_wake = min( y_max*F_max , C_wk*y_max*u_dif^2/F_max )
        F_kleb = 1 / (1 + 5.5*(C_kleb*y/y_max)^6)   Klebanoff intermittency
        nu_t   = K * C_cp * F_wake * F_kleb

    The switch from inner to outer happens at the FIRST height, starting from
    the wall, where the inner value exceeds the outer one. That is the point
    at which the wall formula stops being valid.

    Index convention: j = Nr-1 is the cell adjacent to the wall and j = 0 the
    one adjacent to the axis, so the sweep runs from high j downwards.
    """
    Nz, Nr = omega.shape
    nu_t = np.zeros((Nz, Nr))

    for i in range(Nz):
        # --- the F function and its maximum along the normal ---------------
        F_max = 0.0
        y_max = 0.0
        for j in range(Nr):
            y_plus = y[i, j] * u_tau[i] / nu
            damp = 1.0 - np.exp(-y_plus / A_PLUS)
            F = y[i, j] * omega[i, j] * damp
            if F > F_max:
                F_max = F
                y_max = y[i, j]

        if F_max <= 1e-30 or y_max <= 1e-30:
            continue

        # --- velocity difference along the normal --------------------------
        u_hi = u_mag[i, 0]
        u_lo = u_mag[i, 0]
        for j in range(Nr):
            if u_mag[i, j] > u_hi:
                u_hi = u_mag[i, j]
            if u_mag[i, j] < u_lo:
                u_lo = u_mag[i, j]
        u_dif = u_hi - u_lo

        f1 = y_max * F_max
        f2 = C_WK * y_max * u_dif * u_dif / F_max
        F_wake = f1 if f1 < f2 else f2

        # --- blending of the two layers ------------------------------------
        switched = False
        for j in range(Nr - 1, -1, -1):
            y_plus = y[i, j] * u_tau[i] / nu
            damp = 1.0 - np.exp(-y_plus / A_PLUS)
            length = KAPPA * y[i, j] * damp
            inner = length * length * omega[i, j]

            ratio = C_KLEB * y[i, j] / y_max
            F_kleb = 1.0 / (1.0 + 5.5 * ratio ** 6)
            outer = K_OUTER * C_CP * F_wake * F_kleb

            if not switched and inner <= outer:
                nu_t[i, j] = inner
            else:
                switched = True
                nu_t[i, j] = outer

    return nu_t


def compute_nu_t(uz, ur, mesh, nu: float, model: str) -> np.ndarray:
    """Kinematic eddy viscosity nu_t [m^2/s] per cell.

    Args:
        uz, ur: cell-centre velocity fields, shape (Nz, Nr).
        mesh: FVMesh.
        nu: molecular kinematic viscosity [m^2/s].
        model: one of SUPPORTED.

    Raises:
        NotImplementedError: if the model is not implemented.
    """
    model = str(model).strip().lower()
    if model not in SUPPORTED:
        raise NotImplementedError(
            f"Turbulence model '{model}' is not implemented. "
            f"Available: {', '.join(SUPPORTED)}."
        )

    if model == "laminar":
        return np.zeros_like(uz)

    y = mesh.wall_distance
    u_mag = np.sqrt(uz ** 2 + ur ** 2)

    # Friction velocity from no-slip: tau_w/rho = nu * |du/dn| at the wall.
    # Cell j = Nr-1 is the one adjacent to the wall, where velocity is zero.
    du_dn_wall = u_mag[:, -1] / y[:, -1]
    u_tau = np.sqrt(np.maximum(nu * du_dn_wall, 1e-30))

    omega = _vorticity(uz, ur, mesh.r_c, mesh.R_c, mesh.dRdz_c,
                       mesh.z_c, mesh.eta_c)

    if model == "mixing_length":
        return _mixing_length(omega, y, u_tau, nu, mesh.R_c)

    return _baldwin_lomax(omega, y, u_mag, u_tau, nu)


def wall_y_plus(uz, ur, mesh, nu: float) -> np.ndarray:
    """y+ of the first wall cell, as a grid-adequacy diagnostic."""
    u_mag = np.sqrt(uz ** 2 + ur ** 2)
    y1 = mesh.wall_distance[:, -1]
    u_tau = np.sqrt(np.maximum(nu * u_mag[:, -1] / y1, 1e-30))
    return y1 * u_tau / nu
