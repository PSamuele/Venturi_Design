"""
solver.py
=========
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

from ..fvmesh import FVMesh, divergence
from ..projection import Projector
from ..turbulence import compute_nu_t, wall_y_plus
from .kernels import (reconstruct, explicit_rhs, radial_explicit,
                      implicit_radial, timestep, timestep_radial,
                      predict_fluxes, max_change)


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
            tol, cfl, turbulence_model).
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
    cfl = float(config.cfl)

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
    reconstruct(F_ax, F_rad, mesh.A_ax, mesh.Cr_rad, mesh.Cz_rad, uz, ur)

    U_ref = float(config.v_throat)
    L_ref = float(mesh.z_f[-1] - mesh.z_f[0])
    res_scale = L_ref / (U_ref ** 2)

    history: List[float] = []
    converged = False
    residual = float("inf")
    phi = np.zeros((Nz, Nr))
    dt = 0.0
    n = 0

    # Work buffers, allocated once: the loop runs ~1e5 times, so any array
    # created inside it costs more than the arithmetic done on it.
    uz_new = np.empty((Nz, Nr))
    ur_new = np.empty((Nz, Nr))
    nu_eff = np.empty((Nz, Nr))
    if radial_implicit:
        F_ax_star = np.empty_like(F_ax)
        F_rad_star = np.empty_like(F_rad)

    for n in range(1, int(config.max_iter) + 1):
        if (n - 1) % turb_every == 0:
            nu_t_new = compute_nu_t(uz, ur, mesh, nu_lam, config.turbulence_model)
            # Under-relaxation: the algebraic model derives nu_t from a
            # maximum searched along the wall normal, and that maximum can
            # jump from one cell to the next between iterations. Applying the
            # new field at once passes the jump straight into the residual,
            # which then stops falling monotonically even though the solution
            # is perfectly stable.
            nu_t = (1.0 - turb_relax) * nu_t + turb_relax * nu_t_new
        np.add(nu_t, nu_lam, out=nu_eff)

        explicit_rhs(uz, ur, F_ax, F_rad, nu_eff, nu_lam,
                      c_ax_diff, c_rad, c_in, mesh.vol, uz_in, az_e, ar_e)

        dt = timestep(F_ax, F_rad, nu_eff, nu_lam, c_ax_diff, c_in,
                       mesh.vol, cfl)

        if radial_implicit:
            # Incremental form: the already-known pressure acceleration goes
            # into the predictor, before the implicit solve, so that pressure
            # and every other term pass through the same operator.
            np.add(az_e, gz, out=az_tot)
            np.add(ar_e, gr, out=ar_tot)
            implicit_radial(uz, ur, az_tot, ar_tot, nu_eff, nu_lam, c_rad,
                             cw1, cw2, mesh.vol, mesh.r_c, dt, az, ar)
        else:
            dt = min(dt, timestep_radial(nu_eff, nu_lam, c_rad, cw1, mesh.vol))
            radial_explicit(uz, ur, az_e, ar_e, nu_eff, nu_lam, c_rad,
                             cw1, cw2, mesh.vol, mesh.r_c, az, ar)

        predict_fluxes(F_ax, F_rad, az, ar, mesh.A_ax,
                        mesh.Cr_rad, mesh.Cz_rad, dt)
        if radial_implicit:
            F_ax_star[...] = F_ax
            F_rad_star[...] = F_rad

        phi = proj.project_inplace(F_ax, F_rad)

        if radial_implicit:
            # The flux correction is the effect of the pressure correction:
            # reconstructed at cell centres and divided by dt, it is the
            # increment of the accumulated pressure acceleration. The flux
            # correction itself is applied IN FULL (otherwise the field would
            # stop being divergence-free); only a fraction feeds the stored
            # acceleration, because the predictor-projection feedback
            # otherwise has gain above one.
            reconstruct(F_ax - F_ax_star, F_rad - F_rad_star, mesh.A_ax,
                         mesh.Cr_rad, mesh.Cz_rad, dgz, dgr)
            gz += p_relax * dgz / dt
            gr += p_relax * dgr / dt
            p_acc += p_relax * (rho / dt) * phi
        # Explicit path: the projection supplies the entire pressure gradient
        # at every step, so p = rho*phi/dt with no accumulation. It is only
        # needed at the end, so it is computed after the loop.

        reconstruct(F_ax, F_rad, mesh.A_ax, mesh.Cr_rad, mesh.Cz_rad, uz_new, ur_new)

        # Dimensionless steady residual: |du/dt| * L / U^2.
        # Dividing by dt is essential: without it the residual falls simply
        # by shrinking the time step, with the solution no closer to steady.
        residual = max_change(uz_new, uz, ur_new, ur) / dt * res_scale
        uz, uz_new = uz_new, uz
        ur, ur_new = ur_new, ur
        history.append(residual)

        if verbose and (n == 1 or n % 5000 == 0):
            print(f"  iter {n:7d} | residual {residual:.3e} | dt {dt:.3e} s "
                  f"| max nu_t/nu {float((nu_t / nu_lam).max()):.1f}")

        if residual < float(config.tol) and n > 20:
            converged = True
            if verbose:
                print(f"  converged at iteration {n}, residual {residual:.3e}")
            break

    # --- pressure -----------------------------------------------------------
    # Zero on the outlet face by construction. The field is then shifted so
    # that the mean inlet pressure matches the value imposed by the user.
    if radial_implicit:
        p = p_acc
    else:
        p = (rho / dt) * phi if dt > 0.0 else np.zeros((Nz, Nr))
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
