"""Turbulence models: behaviour and error handling."""
import numpy as np
import pytest

from venturi.config import VenturiConfig
from venturi.geometry import create_venturi_geometry
from venturi.fvmesh import build_fvmesh
from venturi.turbulence import compute_nu_t, wall_y_plus, SUPPORTED


@pytest.fixture(scope="module")
def setup():
    cfg = VenturiConfig(D_inlet=0.1, beta=0.5, Nz=40, Nr=24)
    mesh = build_fvmesh(create_venturi_geometry(cfg), 40, 24, 2.5)
    uz = np.tile(3.0 * (1.0 - (mesh.r_c / mesh.R_c[:, None]) ** 2) ** (1 / 7),
                 (1, 1))
    ur = np.zeros_like(uz)
    return mesh, uz, ur, 1.0e-6


def test_laminar_gives_zero(setup):
    mesh, uz, ur, nu = setup
    assert np.all(compute_nu_t(uz, ur, mesh, nu, "laminar") == 0.0)


@pytest.mark.parametrize("model", [m for m in SUPPORTED if m != "laminar"])
def test_nu_t_is_non_negative(setup, model):
    mesh, uz, ur, nu = setup
    assert np.all(compute_nu_t(uz, ur, mesh, nu, model) >= 0.0)


@pytest.mark.parametrize("model", [m for m in SUPPORTED if m != "laminar"])
def test_nu_t_exceeds_molecular_at_high_reynolds(setup, model):
    """At high Reynolds the eddy viscosity must dominate the molecular one."""
    mesh, uz, ur, nu = setup
    assert compute_nu_t(uz, ur, mesh, nu, model).max() > 10.0 * nu


def test_unknown_model_is_rejected(setup):
    """An unimplemented model must fail loudly, not be silently ignored."""
    mesh, uz, ur, nu = setup
    with pytest.raises(NotImplementedError):
        compute_nu_t(uz, ur, mesh, nu, "spalart_allmaras")


def test_y_plus_is_positive(setup):
    mesh, uz, ur, nu = setup
    yp = wall_y_plus(uz, ur, mesh, nu)
    assert np.all(yp > 0.0) and np.all(np.isfinite(yp))
