"""Richardson extrapolation / GCI on sequences with a known answer."""
import pytest

from venturi.cli import build_parser
from venturi.grid_study import (QUANTITIES, build_report, default_levels, gci,
                                parse_levels, study_levels)


@pytest.mark.parametrize("p_true", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("h", [(1.0, 1.4, 2.0), (1.0, 1.3, 1.9)])
def test_power_law_is_recovered_exactly(p_true, h):
    """phi = phi0 + C h^p: the observed order must be p and phi_ext = phi0,
    also when the refinement ratio is not the same between the grids."""
    phi0, C = 0.975, 0.004
    phi = [phi0 + C * hi ** p_true for hi in h]
    g = gci(h, phi)
    assert not g.rough
    assert g.p == pytest.approx(p_true, rel=1e-8)
    assert g.phi_ext == pytest.approx(phi0, rel=1e-10)
    # GCI = 1.25 * relative change / (r^p - 1)
    r = h[1] / h[0]
    assert g.gci21 == pytest.approx(1.25 * abs((phi[0] - phi[1]) / phi[0]) / (r ** p_true - 1))


def test_celik_example():
    h = [n ** -0.5 for n in (18000, 8000, 4500)]
    g = gci(h, (6.063, 5.972, 5.863))
    assert g.p == pytest.approx(1.53, abs=0.01)
    assert g.phi_ext == pytest.approx(6.1685, abs=1e-3)
    assert 100 * g.gci21 == pytest.approx(2.2, abs=0.05)


def test_order_above_the_scheme_does_not_shrink_the_error():
    h = (1.0, 1.4, 2.0)
    phi = [0.975 + 0.004 * hi ** 4 for hi in h]
    g = gci(h, phi)
    assert g.p == pytest.approx(4.0, rel=1e-8)
    assert g.gci21 == pytest.approx(1.25 * abs((phi[0] - phi[1]) / phi[0]) / (1.4 ** 2 - 1))


@pytest.mark.parametrize("phi", [[0.975, 0.976, 0.9745],     # up and down
                                 [1.03, 1.01, 1.00],         # changes grow
                                 [0.97, 0.975, 0.975],       # one change is zero
                                 [0.975, 0.975, 0.97]])
def test_unsteady_results_still_get_an_error_from_their_spread(phi):
    g = gci((1.0, 1.4, 2.0), phi)
    spread = (max(phi) - min(phi)) / abs(phi[0])
    assert g.rough
    assert g.gci21 == pytest.approx(spread)
    assert g.e_ext21 == pytest.approx(0.5 * spread)
    assert g.phi_ext == phi[0]


def test_unequal_ratios_can_shrink_the_change_and_still_converge():
    """A smaller step r32 gives a smaller e32 on a perfectly good sequence."""
    h = (1.0, 1.8, 2.2)
    phi = [0.975 + 0.004 * hi ** 2 for hi in h]
    assert abs(phi[2] - phi[1]) < abs(phi[1] - phi[0])
    assert not gci(h, phi).rough


def test_identical_results_mean_no_grid_error():
    g = gci((1.0, 1.4, 2.0), [0.975, 0.975, 0.975])
    assert g.gci21 == 0.0 and g.phi_ext == 0.975


def test_a_failed_case_does_not_crash():
    assert gci((1.0, 1.4, 2.0), [float("nan"), 0.97, 0.96]).rough


def test_steep_triplet_does_not_crash():
    gci((1.0, 1.311, 1.311 * 2.16), [1.0, 0.99624, 0.99180])


def test_cell_sizes_must_be_finest_first():
    with pytest.raises(ValueError):
        gci((2.0, 1.4, 1.0), [1.0, 1.1, 1.2])


def test_levels_are_sorted_coarsest_first():
    assert parse_levels("128x51,64x26,90x36") == [(64, 26), (90, 36), (128, 51)]


def test_default_levels_surround_the_grid():
    assert default_levels(90, 36) == [(64, 26), (90, 36), (128, 51)]


@pytest.mark.parametrize("text", ["64x26,90x36", "64x26,90", "64x26,26x64,90x36"])
def test_bad_levels_are_refused(text):
    with pytest.raises(ValueError):
        study_levels(text, 90, 36)


def test_grid_study_only_runs_when_asked():
    assert build_parser().parse_args([]).grid_study is False
    assert build_parser().parse_args(["--grid-study"]).grid_study is True


def _row(nz, nr, value):
    row = dict(Nz=nz, Nr=nr, cells=nz * nr, iters=100, converged=True, residual=1e-4,
               time_s=1.0, y_plus=3.0, p_min=95000.0)
    row.update({q.key: value for q in QUANTITIES})
    return row


def test_report_gives_error_and_gci_of_every_quantity():
    grids = ((128, 51), (90, 36), (64, 26))
    rows = [_row(nz, nr, 1.0 + 500.0 / (nz * nr)) for nz, nr in grids]
    base = {q.key: 1.1 for q in QUANTITIES}
    text, records = build_report(rows, "case", 0.78, base, "90x36")
    assert len(records) == len(QUANTITIES)
    for q in QUANTITIES:
        assert q.label in text
    assert all(not r["rough"] and r["gci_pct"] > 0.0 for r in records)
    assert all(r["order"] == pytest.approx(2.0, rel=1e-6) for r in records)
    assert "90x36 run" in text


def test_report_without_a_run_has_no_run_columns():
    grids = ((128, 51), (90, 36), (64, 26))
    text, _ = build_report([_row(nz, nr, 1.0) for nz, nr in grids], "case", 0.78)
    assert " run |" not in text
