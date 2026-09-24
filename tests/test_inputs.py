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


def test_taps_read_the_wall_cell():
    """dp must come from the wall cells, not from the section average."""
    cfg = VenturiConfig(Nz=60, Nr=12)
    geom = create_venturi_geometry(cfg)
    mesh = build_fvmesh(geom, 60, 12, 2.0)
    p = np.full((60, 12), 1.0e5)
    zs = geom.z_stations
    i_th = int(np.argmin(np.abs(mesh.z_c - 0.5 * (zs[2] + zs[3]))))
    p[i_th, :] = 1.0e5 - 7000.0      # the whole throat row is lower...
    p[i_th, -1] = 1.0e5 - 6000.0     # ...but the wall reads 6000 Pa less
    rep = validate(_fake_result(cfg, mesh, p), mesh, geom, cfg)
    assert rep.dp_measured == pytest.approx(6000.0)
    expected = cfg.Q / (cfg.area_throat * math.sqrt(2 * 6000.0 / (cfg.rho * (1 - cfg.beta**4))))
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
