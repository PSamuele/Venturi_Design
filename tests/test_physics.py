"""Physical validation against known solutions."""
import numpy as np
import pytest
from types import SimpleNamespace

from venturi.fvmesh import build_fvmesh
from venturi.fvsolver import solve_fv, inlet_profile

R_PIPE, L_PIPE = 0.01, 0.5
RHO, MU = 1000.0, 1.0e-3
NU = MU / RHO
V_MEAN = 0.005


class StraightPipe:
    """Straight pipe: the minimal geometry for a Poiseuille comparison."""
    L_total = L_PIPE
    z_stations = [0.0, 0.1, 0.2, 0.3, 0.4, L_PIPE]

    def radius(self, z):
        return np.full_like(np.atleast_1d(z), R_PIPE, dtype=float)

    def radius_derivative(self, z):
        return np.zeros_like(np.atleast_1d(z), dtype=float)


def _pipe_config(**kw):
    base = dict(rho=RHO, nu=NU, v_inlet=V_MEAN, v_throat=V_MEAN, R_inlet=R_PIPE,
                p_inlet=0.0, Re_D=RHO * V_MEAN * 2 * R_PIPE / MU,
                max_iter=60000, tol_steady=1e-7, cfl_max=0.5,
                turbulence_model="laminar")
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture(scope="module")
def poiseuille():
    mesh = build_fvmesh(StraightPipe(), 60, 30, 0.0)
    return mesh, solve_fv(mesh, _pipe_config(), verbose=False)


def test_poiseuille_pressure_gradient(poiseuille):
    """dp/dz must equal 32*mu*V/D^2 (Hagen-Poiseuille)."""
    mesh, r = poiseuille
    assert r.converged
    pm = lambda i: float(np.sum(r.p[i] * mesh.A_ax[i]) / mesh.A_ax[i].sum())
    i1, i2 = 30, 56
    dpdz = (pm(i1) - pm(i2)) / (mesh.z_c[i2] - mesh.z_c[i1])
    exact = 32.0 * MU * V_MEAN / (2 * R_PIPE) ** 2
    assert abs(dpdz - exact) / exact < 0.01


def test_poiseuille_velocity_profile(poiseuille):
    """The developed profile must be the exact parabola."""
    mesh, r = poiseuille
    k = 50
    exact = 2.0 * V_MEAN * (1.0 - (mesh.r_c[k, :] / R_PIPE) ** 2)
    assert np.abs(r.uz[k, :] - exact).max() / (2 * V_MEAN) < 0.01


def test_steady_state_independent_of_timestep():
    """The steady state must not depend on the time step.

    If it does, some operator is acting on the acceleration but not on the
    pressure gradient, and the result is an artefact of the scheme rather
    than the solution of the equations.
    """
    mesh = build_fvmesh(StraightPipe(), 40, 20, 0.0)
    pm = lambda r, i: float(np.sum(r.p[i] * mesh.A_ax[i]) / mesh.A_ax[i].sum())
    vals = []
    for cfl in (0.6, 0.15):
        r = solve_fv(mesh, _pipe_config(cfl_max=cfl), verbose=False)
        vals.append((pm(r, 20) - pm(r, 37)) / (mesh.z_c[37] - mesh.z_c[20]))
    assert abs(vals[0] - vals[1]) / abs(vals[0]) < 1e-6


def test_mass_conserved_by_solver(poiseuille):
    _, r = poiseuille
    assert r.mass_error_pct < 1e-8


def test_inlet_profile_delivers_exact_flow_rate():
    mesh = build_fvmesh(StraightPipe(), 40, 20, 0.0)
    Q = 1.234e-3
    prof = inlet_profile(mesh, 1.0e5, Q)
    assert float(np.sum(mesh.A_ax[0, :] * prof)) == pytest.approx(Q, rel=1e-13)


def test_inlet_profile_shape_depends_on_regime():
    """Parabolic below Re 2300, power law (far flatter) above it."""
    mesh = build_fvmesh(StraightPipe(), 40, 40, 0.0)
    Q = np.pi * R_PIPE ** 2 * V_MEAN
    lam = inlet_profile(mesh, 500.0, Q)
    turb = inlet_profile(mesh, 1.0e5, Q)
    assert lam[0] / V_MEAN == pytest.approx(2.0, rel=0.02)
    assert 1.1 < turb[0] / V_MEAN < 1.35
