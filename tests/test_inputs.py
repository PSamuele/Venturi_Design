"""Input checks, sizing, fluids, optimiser and the checks done after a run."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from venturi.cli import build_parser, config_from_args, default_output_dir
from venturi.config import VenturiConfig
from venturi.fluids import FLUIDS, P_REF
from venturi.fvmesh import build_fvmesh
from venturi.geometry import create_venturi_geometry
from venturi.optimizer import run_optimization
from venturi.validation import validate


# ------------------------------------------------------------------ inputs
@pytest.mark.parametrize("beta", [0.0, 1.0, 1.2, -0.3])
def test_beta_outside_0_1_is_rejected(beta):
    with pytest.raises(ValueError):
        VenturiConfig(beta=beta)


def test_throat_pressure_not_below_inlet_is_rejected():
    with pytest.raises(ValueError):
        VenturiConfig(p_inlet=100000.0, p_throat_target=100000.0)


def test_custom_fluid_needs_rho_and_mu():
    with pytest.raises(ValueError):
        VenturiConfig(fluid="custom", rho=1000.0)


def test_sizing_gives_the_requested_ideal_pressure_drop():
    """Bernoulli + continuity, inverted: the ideal dp must be the one asked for."""
    c = VenturiConfig(p_inlet=101325.0, p_throat_target=95000.0)
    assert c.dp_ideal == pytest.approx(6325.0, rel=1e-12)
    assert c.v_inlet * c.area_inlet == pytest.approx(c.v_throat * c.area_throat, rel=1e-12)


def test_cd_design_scales_the_flow_rate():
    ideal = VenturiConfig(cd_design=1.0)
    real = VenturiConfig(cd_design=0.98)
    assert real.Q / ideal.Q == pytest.approx(0.98, rel=1e-12)


# ------------------------------------------------------------------ fluids
def test_gas_density_follows_inlet_pressure():
    c = VenturiConfig(fluid="air_20C", p_inlet=2 * P_REF, p_throat_target=2 * P_REF - 5000.0)
    assert c.rho == pytest.approx(2 * FLUIDS["air_20C"].rho, rel=1e-12)


def test_liquid_density_ignores_pressure():
    c = VenturiConfig(fluid="water_20C", p_inlet=5e5, p_throat_target=4.9e5)
    assert c.rho == FLUIDS["water_20C"].rho


def test_mach_warning_for_fast_gas():
    fast = VenturiConfig(fluid="air_20C", p_inlet=101325.0, p_throat_target=95000.0)
    slow = VenturiConfig(fluid="air_20C", p_inlet=101325.0, p_throat_target=100800.0)
    assert fast.mach_throat > 0.3 and any("Mach" in w for w in fast.warnings())
    assert slow.mach_throat < 0.3 and not any("Mach" in w for w in slow.warnings())


def test_cavitation_warning_when_throat_below_vapour_pressure():
    c = VenturiConfig(fluid="water_80C", p_inlet=101325.0, p_throat_target=40000.0)
    assert any("vapour pressure" in w for w in c.warnings())


# ------------------------------------------------------------------ command line
def test_command_line_options_reach_the_config():
    args = build_parser().parse_args(["--clustering", "1.9", "--Nz", "40", "--Nr", "20",
                                      "--cfl", "0.3", "--tol", "1e-4"])
    c = config_from_args(args)
    assert (c.wall_clustering, c.Nz, c.Nr, c.cfl, c.tol) == (1.9, 40, 20, 0.3, 1e-4)


def test_results_go_under_the_results_folder():
    args = build_parser().parse_args(["--D", "0.1"])
    assert default_output_dir(args).split("/")[0].split("\\")[0] == "results"
    assert config_from_args(args).output_dir == default_output_dir(args)


# ------------------------------------------------------------------ optimiser
def test_optimiser_rejects_non_positive_pressure_drop():
    with pytest.raises(ValueError):
        run_optimization(0.01, 1e5, 1e5, 998.2, 3)


def test_optimiser_result_reproduces_requested_pressure_drop():
    Q, p_in, p_th, rho = 0.01, 101325.0, 95000.0, 998.2
    res = run_optimization(Q, p_in, p_th, rho, 3)
    c = VenturiConfig(D=res["D_inlet"], beta=res["beta"], p_inlet=p_in, p_throat_target=p_th, rho=rho)
    assert c.Q == pytest.approx(Q, rel=1e-9)


# ------------------------------------------------------------------ checks after a run
def _fake_result(cfg, mesh, p):
    """A result whose fluxes carry exactly the design flow rate."""
    F_ax = np.tile(mesh.A_ax[0] / mesh.A_ax[0].sum() * cfg.Q, (mesh.Nz + 1, 1))
    return SimpleNamespace(p=p, F_ax=F_ax, F_rad=np.zeros((mesh.Nz, mesh.Nr + 1)),
                           final_residual=0.0, converged=True, y_plus_max=1.0)


def test_taps_read_the_wall_at_the_exact_position():
    """dp must come from the wall cells, interpolated to the tap positions."""
    cfg = VenturiConfig(Nz=60, Nr=12)
    geom = create_venturi_geometry(cfg)
    mesh = build_fvmesh(geom, 60, 12, 2.0)
    zs = geom.z_stations
    z_up, z_th = zs[1] - 0.5 * cfg.D, 0.5 * (zs[2] + zs[3])
    # wall pressure falls linearly along z; the core is 1000 Pa lower still
    p = np.empty((60, 12))
    p[:, -1] = 1.0e5 - 20000.0 * mesh.z_c
    p[:, :-1] = p[:, -1:] - 1000.0
    rep = validate(_fake_result(cfg, mesh, p), mesh, geom, cfg)
    dp = 20000.0 * (z_th - z_up)
    assert rep.dp_measured == pytest.approx(dp, rel=1e-12)
    expected = cfg.Q / (cfg.area_throat * math.sqrt(2 * dp / (cfg.rho * (1 - cfg.beta**4))))
    assert rep.Cd == pytest.approx(expected, rel=1e-12)


def test_cavitation_check_uses_lowest_pressure_of_the_field():
    cfg = VenturiConfig(Nz=30, Nr=10)
    geom = create_venturi_geometry(cfg)
    mesh = build_fvmesh(geom, 30, 10, 2.0)
    p = np.full((30, 10), 1.0e5)
    p[15, 3] = cfg.p_vap - 1.0
    rep = validate(_fake_result(cfg, mesh, p), mesh, geom, cfg)
    assert not next(c for c in rep.checks if c.name == "No cavitation").passed


def test_cd_reference_is_optional():
    cfg = VenturiConfig(Nz=30, Nr=10)
    geom = create_venturi_geometry(cfg)
    mesh = build_fvmesh(geom, 30, 10, 2.0)
    p = np.full((30, 10), 1.0e5)
    res = _fake_result(cfg, mesh, p)
    assert not any(c.name.startswith("C_d") for c in validate(res, mesh, geom, cfg).checks)
    rep = validate(res, mesh, geom, cfg, cd_ref=0.98)
    assert any(c.name.startswith("C_d") for c in rep.checks)


def test_throat_refine_below_1_is_rejected():
    with pytest.raises(ValueError):
        VenturiConfig(throat_refine=0.5)


def test_throat_refine_option_reaches_the_config():
    c = config_from_args(build_parser().parse_args(["--throat-refine", "2.5"]))
    assert c.throat_refine == 2.5


@pytest.mark.parametrize("beta", [0.45, 0.5, 0.6, 0.75])
def test_auto_throat_refine_is_one_over_beta_squared(beta):
    """Same crossing time per cell in pipe and throat: dz_t / v_t = dz_p / v_p."""
    c = VenturiConfig(beta=beta)
    assert c.throat_refine == pytest.approx(1.0 / beta**2)
    assert c.throat_refine == pytest.approx(c.v_throat / c.v_inlet)
    assert not c.throat_refine_capped


def test_auto_throat_refine_is_capped_with_a_warning():
    c = VenturiConfig(beta=0.3)
    assert c.throat_refine == 6.0 and c.throat_refine_capped
    assert any("capped" in w for w in c.warnings())


def test_time_step_option():
    assert config_from_args(build_parser().parse_args([])).time_step == "local"
    c = config_from_args(build_parser().parse_args(["--time-step", "global"]))
    assert c.time_step == "global"
    with pytest.raises(ValueError):
        VenturiConfig(time_step="implicit")
