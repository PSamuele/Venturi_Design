"""Conservation properties that must hold to machine precision."""
import numpy as np
import pytest

from venturi.config import VenturiConfig
from venturi.geometry import create_venturi_geometry
from venturi.fvmesh import build_fvmesh, divergence
from venturi.projection import Projector


@pytest.fixture(scope="module")
def mesh():
    cfg = VenturiConfig(D=0.1, beta=0.5, Nz=60, Nr=30)
    return build_fvmesh(create_venturi_geometry(cfg), 60, 30, 2.5)


def test_volume_matches_analytic_integral(mesh):
    """The sum of cell volumes must equal the volume of the duct.

    The tolerance is not machine precision because the Gauss-Legendre
    quadrature is exact on each smooth run of the profile but not on cells
    that straddle a junction between runs (cone/blend arc), where R' is
    discontinuous. The error is O(1e-6) at 60 cells and drops to O(1e-8) at
    200. It does not affect conservation, which is exact by telescoping of
    the fluxes regardless of how accurate the volumes are.
    """
    from scipy.integrate import quad
    geom = create_venturi_geometry(VenturiConfig(D=0.1, beta=0.5))
    V, _ = quad(lambda z: np.pi * float(geom.radius(np.array([z]))[0]) ** 2,
                0.0, geom.L_total, limit=400)
    assert abs(mesh.vol.sum() - V) / V < 1e-5


def test_inlet_area_is_exact(mesh):
    cfg = VenturiConfig(D=0.1, beta=0.5)
    assert mesh.A_ax[0].sum() == pytest.approx(np.pi * cfg.R_inlet ** 2, rel=1e-14)


def test_geometric_conservation_law(mesh):
    """A uniform flow must have exactly zero discrete divergence.

    If it does not, the grid creates or destroys mass wherever the section
    varies, and the projection mistakes the error for something physical.
    """
    F_ax = mesh.A_ax * 1.0
    F_rad = -mesh.Cz_rad * 1.0
    div = divergence(F_ax, F_rad, mesh)
    assert np.abs(div).max() / np.abs(F_ax).max() < 1e-13


def test_projection_removes_divergence(mesh):
    rng = np.random.default_rng(0)
    F_ax = np.zeros((mesh.Nz + 1, mesh.Nr))
    F_rad = np.zeros((mesh.Nz, mesh.Nr + 1))
    F_ax[0, :] = mesh.A_ax[0, :] * 2.0
    F_ax[1:, :] = rng.normal(size=(mesh.Nz, mesh.Nr)) * mesh.A_ax[1:, :]
    F_rad[:, 1:mesh.Nr] = (rng.normal(size=(mesh.Nz, mesh.Nr - 1))
                           * mesh.A_rad[:, 1:mesh.Nr])

    before = np.abs(divergence(F_ax, F_rad, mesh)).max()
    F_ax_c, F_rad_c, _ = Projector(mesh).project(F_ax, F_rad)
    after = np.abs(divergence(F_ax_c, F_rad_c, mesh)).max()
    assert after / before < 1e-10


def test_projection_preserves_boundary_fluxes(mesh):
    """The imposed inlet flow rate and the impermeable wall must be untouched."""
    rng = np.random.default_rng(1)
    F_ax = rng.normal(size=(mesh.Nz + 1, mesh.Nr)) * mesh.A_ax
    F_rad = np.zeros((mesh.Nz, mesh.Nr + 1))
    F_in = F_ax[0].copy()
    F_ax_c, F_rad_c, _ = Projector(mesh).project(F_ax, F_rad)
    assert np.array_equal(F_ax_c[0], F_in)
    assert np.all(F_rad_c[:, mesh.Nr] == 0.0)
    assert np.all(F_rad_c[:, 0] == 0.0)


def test_global_mass_balance(mesh):
    """Q_out must equal Q_in to machine precision."""
    rng = np.random.default_rng(2)
    F_ax = np.zeros((mesh.Nz + 1, mesh.Nr))
    F_rad = np.zeros((mesh.Nz, mesh.Nr + 1))
    F_ax[0, :] = mesh.A_ax[0, :] * 1.5
    F_ax[1:, :] = rng.normal(size=(mesh.Nz, mesh.Nr)) * mesh.A_ax[1:, :]
    F_ax_c, _, _ = Projector(mesh).project(F_ax, F_rad)
    Q_in, Q_out = F_ax_c[0].sum(), F_ax_c[mesh.Nz].sum()
    # The limit here is LU round-off, not the scheme.
    assert abs(Q_in - Q_out) / abs(Q_in) < 1e-9


def test_poisson_matrix_is_symmetric(mesh):
    L = Projector(mesh).L
    assert abs(L - L.T).max() < 1e-9 * abs(L).max()


# ---------------------------------------------------------------------------
# Grid with shorter cells in the throat
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def refined():
    geom = create_venturi_geometry(VenturiConfig(D=0.1, beta=0.5))
    return geom, build_fvmesh(geom, 60, 20, 2.5, throat_refine=3.0)


def test_refine_1_gives_uniform_axial_faces():
    geom = create_venturi_geometry(VenturiConfig(D=0.1, beta=0.5))
    m = build_fvmesh(geom, 40, 10, 2.5, throat_refine=1.0)
    np.testing.assert_array_equal(m.z_f, np.linspace(0.0, geom.L_total, 41))


def test_refined_grid_is_denser_in_throat_and_smooth(refined):
    geom, m = refined
    dz = np.diff(m.z_f)
    assert np.all(dz > 0.0)
    assert m.z_f[0] == 0.0 and m.z_f[-1] == geom.L_total
    zs = geom.z_stations
    in_throat = (m.z_c > zs[2]) & (m.z_c < zs[3])
    assert dz[in_throat].mean() < dz.max() / 2.0
    # neighbouring cells never differ by more than 20 % (the ramps widen until they do not)
    assert np.max(np.maximum(dz[1:] / dz[:-1], dz[:-1] / dz[1:])) <= 1.2 + 1e-12


def test_refined_grid_keeps_conservation(refined):
    """GCL and exact projection must not depend on the axial spacing."""
    _, m = refined
    div = divergence(m.A_ax * 1.0, -m.Cz_rad * 1.0, m)
    assert np.abs(div).max() / m.A_ax.max() < 1e-13
    rng = np.random.default_rng(3)
    F_ax = rng.normal(size=(m.Nz + 1, m.Nr)) * m.A_ax
    F_rad = np.zeros((m.Nz, m.Nr + 1))
    F_rad[:, 1:m.Nr] = rng.normal(size=(m.Nz, m.Nr - 1)) * m.A_rad[:, 1:m.Nr]
    before = np.abs(divergence(F_ax, F_rad, m)).max()
    F_ax_c, F_rad_c, _ = Projector(m).project(F_ax, F_rad)
    assert np.abs(divergence(F_ax_c, F_rad_c, m)).max() / before < 1e-10


def test_weighted_projection_is_exact_and_symmetric(mesh):
    """With a time step per face the projection must stay exact."""
    rng = np.random.default_rng(4)
    P = Projector(mesh)
    P.set_weights(rng.uniform(0.1, 10.0, size=(mesh.Nz + 1, mesh.Nr)),
                  rng.uniform(0.1, 10.0, size=(mesh.Nz, mesh.Nr + 1)))
    assert abs(P.L - P.L.T).max() < 1e-9 * abs(P.L).max()
    F_ax = rng.normal(size=(mesh.Nz + 1, mesh.Nr)) * mesh.A_ax
    F_rad = np.zeros((mesh.Nz, mesh.Nr + 1))
    F_rad[:, 1:mesh.Nr] = rng.normal(size=(mesh.Nz, mesh.Nr - 1)) * mesh.A_rad[:, 1:mesh.Nr]
    F_in = F_ax[0].copy()
    before = np.abs(divergence(F_ax, F_rad, mesh)).max()
    F_ax_c, F_rad_c, _ = P.project(F_ax, F_rad)
    assert np.abs(divergence(F_ax_c, F_rad_c, mesh)).max() / before < 1e-10
    assert np.array_equal(F_ax_c[0], F_in)
    assert np.all(F_rad_c[:, 0] == 0.0) and np.all(F_rad_c[:, mesh.Nr] == 0.0)
