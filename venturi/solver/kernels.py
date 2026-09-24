"""
kernels.py
==========
Numba-compiled loops of the solver. Each function works on whole arrays
in place, so the Python driver in solver.py only calls them in sequence.

Index convention (same as fvmesh.py): i = axial cell 0..Nz-1, j = radial
cell 0..Nr-1, j = 0 next to the axis, j = Nr-1 next to the wall.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def reconstruct(F_ax, F_rad, A_ax, Cr, Cz, uz, ur):
    """Reconstruct cell-centre velocities from the face fluxes."""
    Nz, Nr = uz.shape

    for i in range(Nz):
        for j in range(Nr):
            uz[i, j] = (F_ax[i, j] + F_ax[i + 1, j]) / (A_ax[i, j] + A_ax[i + 1, j])

    for i in range(Nz):
        for j in range(Nr):
            # radial velocity on the cell's two conical faces
            num = 0.0
            den = 0.0
            for jf in (j, j + 1):
                cr = Cr[i, jf]
                if cr <= 1e-30:
                    continue
                if jf == 0:
                    uzf = uz[i, 0]
                elif jf == Nr:
                    uzf = 0.0            # wall: no-slip
                else:
                    uzf = 0.5 * (uz[i, jf - 1] + uz[i, jf])
                urf = (F_rad[i, jf] + Cz[i, jf] * uzf) / cr
                num += cr * urf
                den += cr
            ur[i, j] = num / den if den > 1e-30 else 0.0


@njit(inline="always")
def _tvd(phi_far, phi_up, phi_down):
    """Face value from the van Leer limiter."""
    d = phi_down - phi_up
    if abs(d) < 1e-30:
        return phi_up
    rr = (phi_up - phi_far) / d
    psi = (rr + abs(rr)) / (1.0 + abs(rr))
    return phi_up + 0.5 * psi * d


@njit(cache=True, fastmath=True)
def explicit_rhs(uz, ur, F_ax, F_rad, nu_eff,
                  c_ax, c_rad, c_in, vol, uz_in, az, ar):
    """Explicit part of the accelerations: convection + axial diffusion.

    Radial diffusion, the wall term and the axisymmetric source are handled
    separately (see radial_explicit), because those are
    the terms that constrain the time step: the wall cells are thin and
    nu_eff peaks there.
    """
    Nz, Nr = uz.shape

    for i in range(Nz):
        for j in range(Nr):
            az[i, j] = 0.0
            ar[i, j] = 0.0

    # ---------------- axial faces ----------------------------------------
    for j in range(Nr):
        # inlet: imposed value, incoming flux
        F = F_ax[0, j]
        az[0, j] += F * uz_in[j]
        ar[0, j] += 0.0
        # Diffusion through the inlet face. The eddy viscosity of the
        # incoming flow is taken equal to that of the first cell (zero
        # gradient); using the molecular value here would switch turbulent
        # mixing off right at the inlet.
        az[0, j] += nu_eff[0, j] * c_in[j] * (uz_in[j] - uz[0, j])
        ar[0, j] += nu_eff[0, j] * c_in[j] * (0.0 - ur[0, j])

        for i in range(1, Nz):
            F = F_ax[i, j]
            if F > 0.0:
                fz = _tvd(uz[max(i - 2, 0), j], uz[i - 1, j], uz[i, j])
                fr = _tvd(ur[max(i - 2, 0), j], ur[i - 1, j], ur[i, j])
            else:
                fz = _tvd(uz[min(i + 1, Nz - 1), j], uz[i, j], uz[i - 1, j])
                fr = _tvd(ur[min(i + 1, Nz - 1), j], ur[i, j], ur[i - 1, j])
            # convection: leaving cell i-1, entering cell i
            az[i - 1, j] -= F * fz
            ar[i - 1, j] -= F * fr
            az[i, j] += F * fz
            ar[i, j] += F * fr

            # diffusion
            nu_f = 0.5 * (nu_eff[i - 1, j] + nu_eff[i, j])
            gz = nu_f * c_ax[i, j] * (uz[i, j] - uz[i - 1, j])
            gr = nu_f * c_ax[i, j] * (ur[i, j] - ur[i - 1, j])
            az[i - 1, j] += gz
            ar[i - 1, j] += gr
            az[i, j] -= gz
            ar[i, j] -= gr

        # outlet: pure upwind, zero gradient (no diffusion)
        F = F_ax[Nz, j]
        az[Nz - 1, j] -= F * uz[Nz - 1, j]
        ar[Nz - 1, j] -= F * ur[Nz - 1, j]

    # ---------------- conical faces --------------------------------------
    for i in range(Nz):
        for j in range(1, Nr):
            F = F_rad[i, j]
            if F > 0.0:
                fz = _tvd(uz[i, max(j - 2, 0)], uz[i, j - 1], uz[i, j])
                fr = _tvd(ur[i, max(j - 2, 0)], ur[i, j - 1], ur[i, j])
            else:
                fz = _tvd(uz[i, min(j + 1, Nr - 1)], uz[i, j], uz[i, j - 1])
                fr = _tvd(ur[i, min(j + 1, Nr - 1)], ur[i, j], ur[i, j - 1])
            az[i, j - 1] -= F * fz
            ar[i, j - 1] -= F * fr
            az[i, j] += F * fz
            ar[i, j] += F * fr

        # axis (j = 0): zero face area, no contribution

    # ---------------- normalisation --------------------------------------
    for i in range(Nz):
        for j in range(Nr):
            inv_v = 1.0 / vol[i, j]
            az[i, j] *= inv_v
            ar[i, j] *= inv_v


@njit(cache=True, fastmath=True)
def radial_explicit(uz, ur, az_e, ar_e, nu_eff, nu_lam, c_rad, cw1, cw2,
                     vol, r_c, az, ar):
    """Explicit radial diffusion, wall term and 1/r^2 source.

    Everything is explicit: no operator sits between the acceleration and
    the pressure gradient, so the steady state does not depend on the time
    step. (An earlier implicit variant of this step did make it depend on
    dt and was removed.)

    Wall gradient: quadratic reconstruction through the two nearest cell
    centres and the zero value at the wall. The two-point difference
    (u_wall - u_P)/d approximates the derivative HALFWAY to the wall rather
    than AT the wall, and is therefore only first-order accurate. Since it is
    precisely the wall shear that sets the pressure loss, that error drags
    the whole solution down to first order - verified on Hagen-Poiseuille,
    where the observed order goes from 1.0 to 2.0 once it is replaced.
    """
    Nz, Nr = uz.shape
    for i in range(Nz):
        for j in range(Nr):
            inv_v = 1.0 / vol[i, j]
            gz = 0.0
            gr = 0.0
            if j > 0:
                a_lo = 0.5 * (nu_eff[i, j - 1] + nu_eff[i, j]) * c_rad[i, j]
                gz -= a_lo * (uz[i, j] - uz[i, j - 1])
                gr -= a_lo * (ur[i, j] - ur[i, j - 1])
            if j < Nr - 1:
                a_up = 0.5 * (nu_eff[i, j] + nu_eff[i, j + 1]) * c_rad[i, j + 1]
                gz += a_up * (uz[i, j + 1] - uz[i, j])
                gr += a_up * (ur[i, j + 1] - ur[i, j])
            else:
                # At the wall nu_t = 0, so only molecular viscosity acts:
                # that is where transport goes back to being laminar.
                gz -= nu_lam * (cw1[i] * uz[i, j] + cw2[i] * uz[i, j - 1])
                gr -= nu_lam * (cw1[i] * ur[i, j] + cw2[i] * ur[i, j - 1])
            az[i, j] = az_e[i, j] + gz * inv_v
            ar[i, j] = (ar_e[i, j] + gr * inv_v
                        - nu_eff[i, j] * ur[i, j] / (r_c[i, j] * r_c[i, j]))


@njit(cache=True, fastmath=True)
def cell_timesteps(F_ax, F_rad, nu_eff, nu_lam, c_ax, c_in, c_rad, cw1,
                   vol, r_c, cfl, combine, dt):
    """Largest stable time step of every cell, written into dt (Nz, Nr).

    Each cell takes the smallest of three limits:

      convection        cfl * V / (sum of |flux| through its faces)
      axial diffusion   0.5 * V / (sum of nu_f * c_f over its axial faces)
      radial diffusion  0.5 / (sum of nu_f * c_f over its radial faces / V
                               + 0.5 * nu_eff / r^2)

    The last one includes the decay term -nu_eff * u_r / r^2 of the u_r
    equation. The minimum of dt over all cells is the global time step.

    combine = False: the cell takes the smallest of the three limits.
    combine = True:  1/dt = 1/dt_conv + 1/dt_axial + 1/dt_radial. The three
    processes act together, so their rates add up. This is needed when every
    cell runs at its own limit (local time step): with the plain minimum, a
    cell where two limits are close would run past its true stability bound.
    With a global step almost every cell runs far below its limit, and the
    minimum (the original rule) is kept there.
    """
    Nz, Nr = vol.shape
    for i in range(Nz):
        for j in range(Nr):
            v = vol[i, j]
            nu_w = nu_eff[i, j]
            inv_conv = 0.0
            inv_ax = 0.0
            inv_rad = 0.0

            sum_F = (abs(F_ax[i, j]) + abs(F_ax[i + 1, j])
                     + abs(F_rad[i, j]) + abs(F_rad[i, j + 1]))
            if sum_F > 1e-30:
                inv_conv = sum_F / (cfl * v)

            s_ax = (nu_w * c_in[j]) if i == 0 else \
                (0.5 * (nu_eff[i - 1, j] + nu_w) * c_ax[i, j])
            if i < Nz - 1:
                s_ax += 0.5 * (nu_eff[i + 1, j] + nu_w) * c_ax[i + 1, j]
            if s_ax > 1e-30:
                inv_ax = s_ax / (0.5 * v)

            s_r = 0.0
            if j > 0:
                s_r += 0.5 * (nu_eff[i, j - 1] + nu_w) * c_rad[i, j]
            if j < Nr - 1:
                s_r += 0.5 * (nu_w + nu_eff[i, j + 1]) * c_rad[i, j + 1]
            else:
                s_r += nu_lam * cw1[i]
            rate = s_r / v + 0.5 * nu_w / (r_c[i, j] * r_c[i, j])
            if rate > 1e-30:
                inv_rad = rate / 0.5

            if combine:
                total = inv_conv + inv_ax + inv_rad
            else:
                total = max(inv_conv, max(inv_ax, inv_rad))
            best = 1.0 / total if total > 0.0 else 1.0e30
            dt[i, j] = best


@njit(cache=True)
def face_timesteps(dt, dt_ax, dt_rad):
    """Time step of every face: the smaller one of its two cells.

    Taking the smaller value keeps each cell within its own stability
    limit whatever the neighbour does. The inlet face and the axis and wall
    faces carry no flux update, their value is only filled for tidiness.
    """
    Nz, Nr = dt.shape
    for j in range(Nr):
        dt_ax[0, j] = dt[0, j]
        for i in range(1, Nz):
            dt_ax[i, j] = min(dt[i - 1, j], dt[i, j])
        dt_ax[Nz, j] = dt[Nz - 1, j]
    for i in range(Nz):
        dt_rad[i, 0] = dt[i, 0]
        for j in range(1, Nr):
            dt_rad[i, j] = min(dt[i, j - 1], dt[i, j])
        dt_rad[i, Nr] = dt[i, Nr - 1]


@njit(cache=True, fastmath=True)
def predict_fluxes(F_ax, F_rad, az, ar, A_ax, Cr, Cz, dt_ax, dt_rad):
    """F* = F + dt_f * (acceleration interpolated to the face) * area.

    dt_ax (Nz+1, Nr) and dt_rad (Nz, Nr+1) hold the time step of each face;
    with a global time step every entry is the same number.
    """
    Nz, Nr = az.shape

    for j in range(Nr):
        for i in range(1, Nz):
            a = 0.5 * (az[i - 1, j] + az[i, j])
            F_ax[i, j] += dt_ax[i, j] * A_ax[i, j] * a
        F_ax[Nz, j] += dt_ax[Nz, j] * A_ax[Nz, j] * az[Nz - 1, j]
        # F_ax[0, j] stays imposed (inlet flow rate)

    for i in range(Nz):
        for j in range(1, Nr):
            afz = 0.5 * (az[i, j - 1] + az[i, j])
            afr = 0.5 * (ar[i, j - 1] + ar[i, j])
            F_rad[i, j] += dt_rad[i, j] * (Cr[i, j] * afr - Cz[i, j] * afz)
        # j = 0 (axis) and j = Nr (wall): flux imposed to zero


@njit(cache=True)
def max_change(a_new, a_old, b_new, b_old):
    """max(|a_new - a_old|, |b_new - b_old|) over all cells, without temporaries."""
    m = 0.0
    Nz, Nr = a_new.shape
    for i in range(Nz):
        for j in range(Nr):
            d = abs(a_new[i, j] - a_old[i, j])
            if d > m:
                m = d
            d = abs(b_new[i, j] - b_old[i, j])
            if d > m:
                m = d
    return m


@njit(cache=True)
def max_rate(a_new, a_old, b_new, b_old, dt):
    """max over cells of |change| / dt_cell, for a time step that varies per cell."""
    m = 0.0
    Nz, Nr = a_new.shape
    for i in range(Nz):
        for j in range(Nr):
            d = max(abs(a_new[i, j] - a_old[i, j]), abs(b_new[i, j] - b_old[i, j])) / dt[i, j]
            if d > m:
                m = d
    return m


@njit(cache=True)
def any_above(a, b):
    """True if a > b anywhere (no temporary boolean array)."""
    Nz, Nr = a.shape
    for i in range(Nz):
        for j in range(Nr):
            if a[i, j] > b[i, j]:
                return True
    return False
