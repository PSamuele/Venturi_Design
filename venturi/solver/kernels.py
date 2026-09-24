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
def explicit_rhs(uz, ur, F_ax, F_rad, nu_eff, nu_lam,
                  c_ax, c_rad, c_in, vol, uz_in, az, ar):
    """Explicit part of the accelerations: convection + axial diffusion.

    Radial diffusion, the wall term and the axisymmetric source are handled
    separately (see radial_explicit / implicit_radial), because those are
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
        az[0, j] += nu_lam * c_in[j] * (uz_in[j] - uz[0, j])
        ar[0, j] += nu_lam * c_in[j] * (0.0 - ur[0, j])

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

    This is the reference path of the scheme. It inserts no operator between
    the acceleration and the pressure gradient, so the steady state does not
    depend on the time step. It costs more iterations than the implicit
    variant, but it is the one whose accuracy has been verified.

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
def implicit_radial(uz, ur, az_e, ar_e, nu_eff, nu_lam, c_rad, cw1, cw2,
                     vol, r_c, dt, az, ar):
    """Implicit radial diffusion, solved with the Thomas algorithm.

    Why implicit in this direction only: explicit stability requires
    dt < 0.5*V/(nu_eff*c). Near the wall the cells are thin (to resolve the
    boundary layer at y+ ~ 1) and nu_eff peaks there, so that limit is
    several times tighter than the convective one and dominates the cost
    (the README section "Speed" gives the measured numbers). Made implicit along the radial column alone,
    the system is tridiagonal and solves in O(Nr) per column, with no sparse
    matrices.

    It returns the EFFECTIVE acceleration a = (u* - u)/dt, so the rest of the
    scheme (flux prediction, projection) is unchanged.

    CAVEAT: this path makes the steady state depend on the time step, because
    the preconditioner acts on the acceleration but not on the pressure
    gradient supplied afterwards by the projection. It is disabled by
    default. See the README section "Speed".
    """
    Nz, Nr = uz.shape
    lo = np.empty(Nr)
    di = np.empty(Nr)
    up = np.empty(Nr)
    rz = np.empty(Nr)
    rr = np.empty(Nr)
    cp = np.empty(Nr)
    dz_ = np.empty(Nr)
    dr_ = np.empty(Nr)

    for i in range(Nz):
        for j in range(Nr):
            inv_v = 1.0 / vol[i, j]

            # inner face towards the axis
            if j > 0:
                a_lo = 0.5 * (nu_eff[i, j - 1] + nu_eff[i, j]) * c_rad[i, j]
            else:
                a_lo = 0.0
            # inner face towards the wall
            if j < Nr - 1:
                a_up = 0.5 * (nu_eff[i, j] + nu_eff[i, j + 1]) * c_rad[i, j + 1]
            else:
                a_up = 0.0

            lo[j] = -dt * inv_v * a_lo
            up[j] = -dt * inv_v * a_up
            d = 1.0 + dt * inv_v * (a_lo + a_up)

            # wall: no-slip with u = 0, three-point quadratic gradient
            if j == Nr - 1:
                d += dt * inv_v * nu_lam * cw1[i]
                lo[j] += dt * inv_v * nu_lam * cw2[i]

            di[j] = d
            rz[j] = uz[i, j] + dt * az_e[i, j]
            rr[j] = ur[i, j] + dt * ar_e[i, j]

        # Thomas sweep for uz
        b = di[0]
        cp[0] = up[0] / b
        dz_[0] = rz[0] / b
        for j in range(1, Nr):
            b = di[j] - lo[j] * cp[j - 1]
            cp[j] = up[j] / b
            dz_[j] = (rz[j] - lo[j] * dz_[j - 1]) / b
        for j in range(Nr - 2, -1, -1):
            dz_[j] -= cp[j] * dz_[j + 1]

        # Thomas sweep for ur, with the axisymmetric 1/r^2 source added to
        # the diagonal: it is diagonal and dissipative, hence unconditionally
        # stable when treated implicitly.
        for j in range(Nr):
            di[j] += dt * nu_eff[i, j] / (r_c[i, j] * r_c[i, j])
        b = di[0]
        cp[0] = up[0] / b
        dr_[0] = rr[0] / b
        for j in range(1, Nr):
            b = di[j] - lo[j] * cp[j - 1]
            cp[j] = up[j] / b
            dr_[j] = (rr[j] - lo[j] * dr_[j - 1]) / b
        for j in range(Nr - 2, -1, -1):
            dr_[j] -= cp[j] * dr_[j + 1]

        for j in range(Nr):
            az[i, j] = (dz_[j] - uz[i, j]) / dt
            ar[i, j] = (dr_[j] - ur[i, j]) / dt


@njit(cache=True, fastmath=True)
def timestep_radial(nu_eff, nu_lam, c_rad, cw1, vol):
    """Radial diffusive limit, needed only on the explicit path."""
    Nz, Nr = vol.shape
    dt = 1.0e30
    for i in range(Nz):
        for j in range(Nr):
            s = 0.0
            if j > 0:
                s += 0.5 * (nu_eff[i, j - 1] + nu_eff[i, j]) * c_rad[i, j]
            if j < Nr - 1:
                s += 0.5 * (nu_eff[i, j] + nu_eff[i, j + 1]) * c_rad[i, j + 1]
            else:
                s += nu_lam * cw1[i]
            if s > 1e-30:
                d = 0.5 * vol[i, j] / s
                if d < dt:
                    dt = d
    return dt


@njit(cache=True, fastmath=True)
def timestep(F_ax, F_rad, nu_eff, nu_lam, c_ax, c_in, vol, cfl):
    """Global time step from the convective (CFL) and axial diffusive limits."""
    Nz, Nr = vol.shape
    dt = 1.0e30
    for i in range(Nz):
        for j in range(Nr):
            sum_F = (abs(F_ax[i, j]) + abs(F_ax[i + 1, j])
                     + abs(F_rad[i, j]) + abs(F_rad[i, j + 1]))

            nu_w = nu_eff[i, j]
            sum_nc = 0.0
            sum_nc += (nu_lam * c_in[j]) if i == 0 else \
                (0.5 * (nu_eff[i - 1, j] + nu_w) * c_ax[i, j])
            if i < Nz - 1:
                sum_nc += 0.5 * (nu_eff[i + 1, j] + nu_w) * c_ax[i + 1, j]

            v = vol[i, j]
            if sum_F > 1e-30:
                d = cfl * v / sum_F
                if d < dt:
                    dt = d
            if sum_nc > 1e-30:
                d = 0.5 * v / sum_nc
                if d < dt:
                    dt = d
    return dt


@njit(cache=True, fastmath=True)
def predict_fluxes(F_ax, F_rad, az, ar, A_ax, Cr, Cz, dt):
    """F* = F + dt * (acceleration interpolated to the face) * area."""
    Nz, Nr = az.shape

    for j in range(Nr):
        for i in range(1, Nz):
            a = 0.5 * (az[i - 1, j] + az[i, j])
            F_ax[i, j] += dt * A_ax[i, j] * a
        F_ax[Nz, j] += dt * A_ax[Nz, j] * az[Nz - 1, j]
        # F_ax[0, j] stays imposed (inlet flow rate)

    for i in range(Nz):
        for j in range(1, Nr):
            afz = 0.5 * (az[i, j - 1] + az[i, j])
            afr = 0.5 * (ar[i, j - 1] + ar[i, j])
            F_rad[i, j] += dt * (Cr[i, j] * afr - Cz[i, j] * afz)
        # j = 0 (axis) and j = Nr (wall): flux imposed to zero
