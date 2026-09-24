"""Richardson extrapolation / GCI on sequences with a known answer."""
import math

import pytest

from venturi.grid_study import gci, parse_levels


@pytest.mark.parametrize("p_true", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("h", [(1.0, 1.4, 2.0), (1.0, 1.3, 1.9)])
def test_power_law_is_recovered_exactly(p_true, h):
    """phi = phi0 + C h^p: the observed order must be p and phi_ext = phi0,
    also when the refinement ratio is not the same between the grids."""
    phi0, C = 0.975, 0.004
    phi = [phi0 + C * hi ** p_true for hi in h]
    g = gci(h, phi)
    assert not g.oscillatory
    assert g.p == pytest.approx(p_true, rel=1e-8)
    assert g.phi_ext == pytest.approx(phi0, rel=1e-10)
    # GCI = 1.25 * relative change / (r^p - 1)
    r = h[1] / h[0]
    assert g.gci21 == pytest.approx(1.25 * abs((phi[0] - phi[1]) / phi[0]) / (r ** p_true - 1))


def test_oscillatory_convergence_is_flagged():
    g = gci((1.0, 1.4, 2.0), [0.975, 0.976, 0.9745])
    assert g.oscillatory


def test_identical_results_mean_no_grid_error():
    g = gci((1.0, 1.4, 2.0), [0.975, 0.975, 0.975])
    assert g.gci21 == 0.0 and g.phi_ext == 0.975


def test_cell_sizes_must_be_finest_first():
    with pytest.raises(ValueError):
        gci((2.0, 1.4, 1.0), [1.0, 1.1, 1.2])


def test_levels_are_sorted_coarsest_first():
    assert parse_levels("128x51,64x26,90x36") == [(64, 26), (90, 36), (128, 51)]
