"""Export: every file written must carry the right data on the right point.

The field tests put a known function of position into the arrays (pressure
equal to the axial coordinate, axial velocity equal to the radius). After
writing and reading back, each value must match the coordinates of its own
point. A C/Fortran ordering mix-up, or a revolution about the wrong axis,
breaks that match immediately.

The tests that need pyvista or ezdxf are skipped when the library is missing.
"""
import os

import numpy as np
import pytest

import venturi.export as ex
from venturi.export import cad, paraview
from venturi.config import VenturiConfig
from venturi.fvmesh import build_fvmesh
from venturi.geometry import create_venturi_geometry

NZ, NR = 30, 12


def _case(tmp_path):
    cfg = VenturiConfig(D=0.1, beta=0.5, output_dir=str(tmp_path))
    geom = create_venturi_geometry(cfg)
    mesh = build_fvmesh(geom, NZ, NR, 2.7)
    p = mesh.z_cc.copy()              # pressure  = axial coordinate
    uz = mesh.r_c.copy()              # u_axial   = radius
    ur = np.ones_like(mesh.z_cc)         # u_radial  = 1 everywhere
    return cfg, geom, mesh, uz, ur, p


def test_missing_libraries_are_not_reported_as_written(tmp_path, monkeypatch):
    monkeypatch.setattr(paraview, "pv", None)
    monkeypatch.setattr(cad, "ezdxf", None)
    monkeypatch.setattr(cad, "cq", None)
    cfg, geom, mesh, uz, ur, p = _case(tmp_path)
    files = ex.export_all(mesh, uz, ur, p, geom, cfg, {})
    assert files, "at least the matplotlib/CSV/Markdown outputs must exist"
    assert all(os.path.isfile(f) for f in files)
    assert not any(f.endswith((".vts", ".dxf", ".stl", ".step")) for f in files)


def test_vts_2d_fields_sit_on_their_points(tmp_path):
    pv = pytest.importorskip("pyvista")
    cfg, geom, mesh, uz, ur, p = _case(tmp_path)
    g = pv.read(paraview.export_paraview_2d(mesh, uz, ur, p, cfg.output_dir))
    x, y = g.points[:, 0], g.points[:, 1]
    np.testing.assert_allclose(g.point_data["Pressure"], x, rtol=0, atol=1e-12)
    np.testing.assert_allclose(g.point_data["Axial_Velocity"], y, rtol=0, atol=1e-12)


def test_vts_3d_is_revolved_about_x_with_consistent_fields(tmp_path):
    pv = pytest.importorskip("pyvista")
    cfg, geom, mesh, uz, ur, p = _case(tmp_path)
    g = pv.read(paraview.export_paraview_3d(mesh, uz, ur, p, cfg.output_dir))
    x, y, z = g.points.T
    rad = np.hypot(y, z)
    vel = g.point_data["Velocity"]
    np.testing.assert_allclose(g.point_data["Pressure"], x, rtol=0, atol=1e-12)
    np.testing.assert_allclose(vel[:, 0], rad, rtol=0, atol=1e-12)
    # u_radial = 1 -> (u_y, u_z) is the unit vector (y, z)/r
    np.testing.assert_allclose(vel[:, 1], y / rad, rtol=0, atol=1e-9)
    np.testing.assert_allclose(vel[:, 2], z / rad, rtol=0, atol=1e-9)


def test_stl_is_a_tube_around_the_x_axis(tmp_path):
    pv = pytest.importorskip("pyvista")
    cfg, geom, mesh, uz, ur, p = _case(tmp_path)
    s = pv.read(paraview.export_stl(geom, cfg.output_dir))
    x, y, z = s.points.T.astype(float)
    assert x.min() == pytest.approx(0.0, abs=1e-6)
    assert x.max() == pytest.approx(geom.L_total, rel=1e-6)
    # every vertex must lie on the wall: sqrt(y^2 + z^2) = R(x). STL is float32.
    np.testing.assert_allclose(np.hypot(y, z), geom.radius(x), rtol=0, atol=1e-6)
    assert np.ptp(z) == pytest.approx(2 * geom.R_inlet, rel=1e-3)


def test_dxf_profile_and_centerline(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    cfg, geom, mesh, uz, ur, p = _case(tmp_path)
    doc = ezdxf.readfile(cad.export_dxf(geom, cfg.output_dir))
    assert doc.layers.get("CENTERLINE").dxf.linetype == "CENTER"
    upper = doc.modelspace().query('LWPOLYLINE[layer=="PROFILE_UPPER"]')[0]
    ys = np.array([pt[1] for pt in upper.get_points("xy")])
    assert ys.max() == pytest.approx(geom.R_inlet, rel=1e-12)
    assert ys.min() == pytest.approx(geom.R_throat, rel=1e-2)
