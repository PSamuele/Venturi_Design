"""
fvsolver.py
===========
Incompressible axisymmetric Navier-Stokes solver on a finite-volume grid.

Scheme
------
The primary variables are the FACE FLUXES (F_ax, F_rad). Cell-centre
velocities are reconstructed from them. This is the MAC arrangement: with
pressure at the cell centre and flux on the face, the pressure gradient that
corrects a flux is compact (it only involves the two cells sharing that
face). There is therefore no odd-even "checkerboard" decoupling of the kind
that plagues collocated grids, and no Rhie-Chow stabilisation is needed.

Time marching (pseudo-transient towards steady state):
    1. reconstruct cell velocities from the fluxes
    2. eddy viscosity (algebraic model)
    3. cell accelerations: -convection + diffusion + axisymmetric source
    4. predicted fluxes  F* = F + dt * (acceleration projected onto the face)
    5. divergence-free projection -> F and potential phi
    6. physical pressure p = rho * phi / dt

Step 5 is exact to machine precision (see projection.py), so mass is
conserved at every single iteration, not merely at convergence.

Convection
----------
TVD interpolation with the van Leer limiter. First-order upwind would inject
a numerical viscosity of order u*dz/2, which at Re ~ 1e5 dwarfs both the
molecular and the eddy viscosity, smearing the boundary layer and corrupting
the discharge coefficient. Pure second order, on the other hand, oscillates
across the steep gradients at the throat. The limiter switches between the
two while preserving monotonicity.

Known approximation
-------------------
The viscous term implemented is div(nu_eff * grad(u)). The transpose term
div(nu_eff * grad(u)^T), which is non-zero wherever nu_eff varies in space,
is omitted. This is the standard approximation for solvers with an algebraic
eddy viscosity: its contribution scales as O(dnu_t/dr * dur/dz), which is
small both in the core (nu_t nearly uniform) and at the wall (dur/dz nearly
zero).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
from numba import njit

from .fvmesh import FVMesh, divergence
from .projection import Projector
from .turbulence import compute_nu_t, wall_y_plus


@dataclass
class FVResult:
    """Outcome of a simulation."""
    uz: np.ndarray              # (Nz, Nr) axial velocity at cell centres
    ur: np.ndarray              # (Nz, Nr) radial velocity at cell centres
    p: np.ndarray               # (Nz, Nr) absolute pressure [Pa]
    nu_t: np.ndarray            # (Nz, Nr) eddy viscosity [m^2/s]
    F_ax: np.ndarray            # (Nz+1, Nr) axial fluxes [m^3/s]
    F_rad: np.ndarray           # (Nz, Nr+1) conical-face fluxes [m^3/s]
    converged: bool
    n_iter: int
    final_residual: float
    residual_history: List[float] = field(default_factory=list)
    max_divergence: float = 0.0
    mass_error_pct: float = 0.0
    y_plus_max: float = 0.0
    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Inlet profile
# ---------------------------------------------------------------------------
def inlet_profile(mesh: FVMesh, Re_D: float, Q_target: float) -> np.ndarray:
    """Axial inlet velocity profile, normalised to an exact flow rate.

    Below Re = 2300 the flow is laminar and the developed profile is
    parabolic. Above it, the turbulent profile is far flatter and follows
    Nikuradse's power law u/u_max = (1 - r/R)^(1/n), with n increasing with
    Reynolds. Imposing a parabola at Re ~ 1e5 overstates the centreline
    velocity by 64% (2*V_mean instead of ~1.22*V_mean) - that was the error
    that made it impossible to reconcile the result with Bernoulli, which
    works in MEAN velocity.

    The profile is finally rescaled so that the sum of the face fluxes equals
    Q_target exactly: the imposed flow rate is then a hard input rather than
    an outcome of the discretisation.
    """
    r = mesh.r_c[0, :]
    R = mesh.R_f[0]

    if Re_D < 2300.0:
        prof = 1.0 - (r / R) ** 2
    else:
        n = -1.7 + 1.8 * math.log10(max(Re_D, 1.0e4))
        n = min(max(n, 6.0), 10.0)
        prof = np.maximum(1.0 - r / R, 0.0) ** (1.0 / n)

    Q_raw = float(np.sum(mesh.A_ax[0, :] * prof))
    return prof * (Q_target / Q_raw)


# ---------------------------------------------------------------------------
# Compiled kernels
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _reconstruct(F_ax, F_rad, A_ax, Cr, Cz, uz, ur):
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
def _explicit_rhs(uz, ur, F_ax, F_rad, nu_eff, nu_lam,
                  c_ax, c_rad, c_in, vol, uz_in, az, ar):
    """Explicit part of the accelerations: convection + axial diffusion.

    Radial diffusion, the wall term and the axisymmetric source are handled
    separately (see _radial_explicit / _implicit_radial), because those are
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
def _radial_explicit(uz, ur, az_e, ar_e, nu_eff, nu_lam, c_rad, cw1, cw2,
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
def _implicit_radial(uz, ur, az_e, ar_e, nu_eff, nu_lam, c_rad, cw1, cw2,
                     vol, r_c, dt, az, ar):
    """Implicit radial diffusion, solved with the Thomas algorithm.

    Why implicit in this direction only: explicit stability requires
    dt < 0.5*V/(nu_eff*c). Near the wall the cells are thin (to resolve the
    boundary layer at y+ ~ 1) and nu_eff reaches ~200 times the molecular
    value, so that limit is about 30 times more severe than the convective
    one and dominates the cost. Made implicit along the radial column alone,
    the system is tridiagonal and solves in O(Nr) per column, with no sparse
    matrices.

    It returns the EFFECTIVE acceleration a = (u* - u)/dt, so the rest of the
    scheme (flux prediction, projection) is unchanged.

    CAVEAT: this path makes the steady state depend on the time step, because
    the preconditioner acts on the acceleration but not on the pressure
    gradient supplied afterwards by the projection. It is disabled by
    default. See the README section "Known limitation".
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
def _timestep_radial(nu_eff, nu_lam, c_rad, cw1, vol):
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
def _timestep(F_ax, F_rad, nu_eff, nu_lam, c_ax, c_in, vol, cfl):
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
def _predict_fluxes(F_ax, F_rad, az, ar, A_ax, Cr, Cz, dt):
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


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def solve_fv(mesh: FVMesh, config, verbose: bool = True,
             turb_every: int = 5, turb_relax: float = 0.3,
             p_relax: float = 0.3,
             radial_implicit: bool = False) -> FVResult:
    """Solve the steady field on the given domain.

    Args:
        mesh: finite-volume grid.
        config: VenturiConfig (uses rho, nu, v_inlet, Re_D, max_iter,
            tol_steady, cfl_max, turbulence_model).
        verbose: print progress.
        turb_every: how often (in iterations) to refresh nu_t.
        turb_relax: under-relaxation factor on nu_t (0 < x <= 1).
        p_relax: under-relaxation on the pressure accumulation (0 < x <= 1).
        radial_implicit: radial implicit accelerator. Roughly 7x fewer
            iterations, but it makes the steady state depend on the time
            step; off by default. See the README.
    """
    t0 = time.time()

    Nz, Nr = mesh.Nz, mesh.Nr
    rho = float(config.rho)
    nu_lam = float(config.nu)
    cfl = float(config.cfl_max)

    # --- geometric coefficients --------------------------------------------
    proj = Projector(mesh)
    c_ax = proj.coef.c_ax.copy()
    c_rad = proj.coef.c_rad.copy()

    # Dirichlet velocity-boundary coefficients (not used by the pressure, so
    # computed separately).
    c_in = mesh.A_ax[0, :] / (mesh.z_c[0] - mesh.z_f[0])

    # Three-point wall gradient: normal distances of the two cell centres
    # nearest the wall, and the Lagrange coefficients of the derivative
    # evaluated on the wall itself (where the velocity is zero).
    A_w = mesh.A_rad[:, Nr]
    d1 = (mesh.R_c - mesh.r_c[:, Nr - 1]) * mesh.nr_rad[:, Nr]
    d2 = (mesh.R_c - mesh.r_c[:, Nr - 2]) * mesh.nr_rad[:, Nr]
    cw1 = A_w * d2 / (d1 * (d2 - d1))
    cw2 = -A_w * d1 / (d2 * (d2 - d1))

    # c_ax[Nz] is needed by the pressure (Dirichlet outlet) but NOT by
    # diffusion, where the outlet has zero gradient: a dedicated copy zeroes
    # it for the momentum equations.
    c_ax_diff = c_ax.copy()
    c_ax_diff[Nz, :] = 0.0

    # --- inlet condition ----------------------------------------------------
    Q_target = float(config.v_inlet * math.pi * config.R_inlet ** 2)
    uz_in = inlet_profile(mesh, float(config.Re_D), Q_target)
    F_in = mesh.A_ax[0, :] * uz_in

    # --- initialisation -----------------------------------------------------
    F_ax = np.zeros((Nz + 1, Nr))
    F_rad = np.zeros((Nz, Nr + 1))
    for i in range(Nz + 1):
        F_ax[i, :] = mesh.A_ax[i, :] * (Q_target / mesh.A_ax[i, :].sum())
    F_ax[0, :] = F_in
    F_ax, F_rad, _ = proj.project(F_ax, F_rad)
    F_ax[0, :] = F_in

    uz = np.zeros((Nz, Nr))
    ur = np.zeros((Nz, Nr))
    az = np.zeros((Nz, Nr))
    ar = np.zeros((Nz, Nr))
    az_e = np.zeros((Nz, Nr))
    ar_e = np.zeros((Nz, Nr))
    az_tot = np.zeros((Nz, Nr))
    ar_tot = np.zeros((Nz, Nr))
    gz = np.zeros((Nz, Nr))      # accumulated pressure acceleration
    gr = np.zeros((Nz, Nr))
    dgz = np.zeros((Nz, Nr))
    dgr = np.zeros((Nz, Nr))
    p_acc = np.zeros((Nz, Nr))
    nu_t = np.zeros((Nz, Nr))
    _reconstruct(F_ax, F_rad, mesh.A_ax, mesh.Cr_rad, mesh.Cz_rad, uz, ur)

    U_ref = float(config.v_throat)
    L_ref = float(mesh.z_f[-1] - mesh.z_f[0])
    res_scale = L_ref / (U_ref ** 2)

    history: List[float] = []
    converged = False
    residual = float("inf")
    phi = np.zeros((Nz, Nr))
    dt = 0.0
    n = 0

    for n in range(1, int(config.max_iter) + 1):
        uz_prev = uz.copy()
        ur_prev = ur.copy()

        if (n - 1) % turb_every == 0:
            nu_t_new = compute_nu_t(uz, ur, mesh, nu_lam, config.turbulence_model)
            # Under-relaxation: the algebraic model derives nu_t from a
            # maximum searched along the wall normal, and that maximum can
            # jump from one cell to the next between iterations. Applying the
            # new field at once passes the jump straight into the residual,
            # which then stops falling monotonically even though the solution
            # is perfectly stable.
            nu_t = (1.0 - turb_relax) * nu_t + turb_relax * nu_t_new
        nu_eff = nu_lam + nu_t

        _explicit_rhs(uz, ur, F_ax, F_rad, nu_eff, nu_lam,
                      c_ax_diff, c_rad, c_in, mesh.vol, uz_in, az_e, ar_e)

        dt = _timestep(F_ax, F_rad, nu_eff, nu_lam, c_ax_diff, c_in,
                       mesh.vol, cfl)

        if radial_implicit:
            # Incremental form: the already-known pressure acceleration goes
            # into the predictor, before the implicit solve, so that pressure
            # and every other term pass through the same operator.
            np.add(az_e, gz, out=az_tot)
            np.add(ar_e, gr, out=ar_tot)
            _implicit_radial(uz, ur, az_tot, ar_tot, nu_eff, nu_lam, c_rad,
                             cw1, cw2, mesh.vol, mesh.r_c, dt, az, ar)
        else:
            dt = min(dt, _timestep_radial(nu_eff, nu_lam, c_rad, cw1, mesh.vol))
            _radial_explicit(uz, ur, az_e, ar_e, nu_eff, nu_lam, c_rad,
                             cw1, cw2, mesh.vol, mesh.r_c, az, ar)

        _predict_fluxes(F_ax, F_rad, az, ar, mesh.A_ax,
                        mesh.Cr_rad, mesh.Cz_rad, dt)
        F_ax_star = F_ax.copy()
        F_rad_star = F_rad.copy()

        F_ax, F_rad, phi = proj.project(F_ax, F_rad)

        if radial_implicit:
            # The flux correction is the effect of the pressure correction:
            # reconstructed at cell centres and divided by dt, it is the
            # increment of the accumulated pressure acceleration. The flux
            # correction itself is applied IN FULL (otherwise the field would
            # stop being divergence-free); only a fraction feeds the stored
            # acceleration, because the predictor-projection feedback
            # otherwise has gain above one.
            _reconstruct(F_ax - F_ax_star, F_rad - F_rad_star, mesh.A_ax,
                         mesh.Cr_rad, mesh.Cz_rad, dgz, dgr)
            gz += p_relax * dgz / dt
            gr += p_relax * dgr / dt
            p_acc += p_relax * (rho / dt) * phi
        else:
            # Explicit path: the projection supplies the entire pressure
            # gradient at every step, so p = rho*phi/dt with no accumulation
            # and no under-relaxation.
            p_acc = (rho / dt) * phi

        _reconstruct(F_ax, F_rad, mesh.A_ax, mesh.Cr_rad, mesh.Cz_rad, uz, ur)

        # Dimensionless steady residual: |du/dt| * L / U^2.
        # Dividing by dt is essential: without it the residual falls simply
        # by shrinking the time step, with the solution no closer to steady.
        d = max(float(np.abs(uz - uz_prev).max()), float(np.abs(ur - ur_prev).max()))
        residual = d / dt * res_scale
        history.append(residual)

        if verbose and (n == 1 or n % 5000 == 0):
            print(f"  iter {n:7d} | residual {residual:.3e} | dt {dt:.3e} s "
                  f"| max nu_t/nu {float((nu_t / nu_lam).max()):.1f}")

        if residual < float(config.tol_steady) and n > 20:
            converged = True
            if verbose:
                print(f"  converged at iteration {n}, residual {residual:.3e}")
            break

    # --- pressure -----------------------------------------------------------
    # Zero on the outlet face by construction. The field is then shifted so
    # that the mean inlet pressure matches the value imposed by the user.
    p = p_acc
    p_in_mean = float(np.sum(p[0, :] * mesh.A_ax[0, :]) / mesh.A_ax[0, :].sum())
    p = p + (float(config.p_inlet) - p_in_mean)

    # --- diagnostics --------------------------------------------------------
    div = divergence(F_ax, F_rad, mesh)
    Q_in = float(F_ax[0, :].sum())
    Q_out = float(F_ax[Nz, :].sum())
    mass_err = abs(Q_in - Q_out) / abs(Q_in) * 100.0
    yp = wall_y_plus(uz, ur, mesh, nu_lam)

    return FVResult(
        uz=uz, ur=ur, p=p, nu_t=nu_t, F_ax=F_ax, F_rad=F_rad,
        converged=converged, n_iter=n, final_residual=residual,
        residual_history=history,
        max_divergence=float(np.abs(div).max()),
        mass_error_pct=mass_err,
        y_plus_max=float(yp.max()),
        wall_time_s=time.time() - t0,
    )
