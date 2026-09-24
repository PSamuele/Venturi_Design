"""Wall profile: rounded corners must join the straight pieces without kinks."""
import numpy as np
import pytest

from venturi.config import VenturiConfig
from venturi.geometry import create_venturi_geometry


@pytest.fixture(scope="module")
def geom():
    return create_venturi_geometry(VenturiConfig(D=0.1, beta=0.5))


def test_end_radii(geom):
    assert geom.radius(0.0) == pytest.approx(0.05)
    assert geom.radius(geom.L_total) == pytest.approx(0.05)
    z3 = 0.5 * (geom.z_stations[2] + geom.z_stations[3])
    assert geom.radius(z3) == pytest.approx(0.025)


def test_radius_and_slope_are_continuous_at_every_arc_end(geom):
    """R and dR/dz must match on both sides of each arc end (no step, no kink)."""
    eps = 1e-9
    for z_s, z_e, *_ in geom.arcs:
        for z in (z_s, z_e):
            assert geom.radius(z - eps) == pytest.approx(geom.radius(z + eps), abs=1e-8)
            assert geom.radius_derivative(z - eps) == pytest.approx(
                geom.radius_derivative(z + eps), abs=1e-6)


def test_slope_matches_finite_difference(geom):
    z = np.linspace(1e-4, geom.L_total - 1e-4, 4001)
    h = 1e-7
    fd = (geom.radius(z + h) - geom.radius(z - h)) / (2 * h)
    # skip the few points that straddle an arc end, where R' is only C0 in R''
    ok = np.ones_like(z, dtype=bool)
    for z_s, z_e, *_ in geom.arcs:
        ok &= (np.abs(z - z_s) > 1e-6) & (np.abs(z - z_e) > 1e-6)
    np.testing.assert_allclose(geom.radius_derivative(z)[ok], fd[ok], atol=1e-6)


def test_sharp_corners_option(geom):
    g = create_venturi_geometry(VenturiConfig(D=0.1, beta=0.5, use_blend_radii=False))
    z1, z2 = g.z_stations[1:3]
    assert g.arcs == []
    assert g.radius(0.5 * (z1 + z2)) == pytest.approx(0.5 * (0.05 + 0.025))
