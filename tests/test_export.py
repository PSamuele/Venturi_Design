"""Export: every file written must carry the right data on the right point.

The field tests put a known function of position into the arrays (pressure
equal to the axial coordinate, axial velocity equal to the radius). After
writing and reading back, each value must match the coordinates of its own
point. A C/Fortran ordering mix-up, or a revolution about the wrong axis,
breaks that match immediately.

The tests that need pyvista or ezdxf are skipped when the library is missing.
"""
import math
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
    assert not any(f.endswith((".vts", ".dxf", ".step")) for f in files)
    assert sum(f.endswith(".stl") for f in files) == 2   # STL needs no library


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


def _read_stl(path):
    """Triangles (n, 3, 3) of a binary STL, with shared vertices merged."""
    data = np.fromfile(path, dtype=[("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")], offset=84)
    return data["v"].reshape(-1, 3, 3).astype(float)


def _open_edges(tris):
    _, index = np.unique(tris.reshape(-1, 3).round(4), axis=0, return_inverse=True)
    index = index.ravel().reshape(-1, 3)
    edges = np.sort(np.vstack((index[:, [0, 1]], index[:, [1, 2]], index[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.sum(counts != 2))


def _stl_volume(tris):
    return np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0


def _exact_volume(loop):
    """Volume under a closed (z, r) profile by Pappus, from a dense sampling [mm3]."""
    pts = cad._sample(loop, 0.02, 0.002)
    z, r = pts[:, 0], pts[:, 1]
    return abs(np.pi * np.sum((r[1:]**2 + r[1:] * r[:-1] + r[:-1]**2) * np.diff(z)) / 3.0)


GEOMETRIES = {
    "default": dict(),
    "sharp corners": dict(use_blend_radii=False),
    "wall thicker than the throat arcs": dict(D=0.03, beta=0.75, wall_thickness=0.012),
    "steep diffuser": dict(D=0.03153, beta=0.75, alpha_div_deg=15.0),
}


def _geom(tmp_path, **kw):
    kw = {"D": 0.1, "beta": 0.5, "output_dir": str(tmp_path), **kw}
    return create_venturi_geometry(VenturiConfig(**kw))


def test_wall_segments_follow_the_profile():
    for kw in GEOMETRIES.values():
        geom = _geom("", **kw)
        segs = geom.wall_segments()
        for a, b in zip(segs, segs[1:]):
            assert a.end == pytest.approx(b.start, abs=1e-12)
        points = [s.start for s in segs] + [s.end for s in segs]
        points += [s.mid for s in segs if s.mid]
        z, r = np.array(points).T
        np.testing.assert_allclose(r, geom.radius(z), rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", GEOMETRIES)
def test_tube_stl_is_closed_hollow_and_in_mm(tmp_path, name):
    geom = _geom(tmp_path, **GEOMETRIES[name])
    tris = _read_stl(cad.export_stl(geom, str(tmp_path)))
    assert _open_edges(tris) == 0
    x, y, z = tris.reshape(-1, 3).T
    rad = np.hypot(y, z)
    assert x.min() < 0.0                                     # bell mouth upstream of x = 0
    assert x.max() > geom.L_total * 1e3                      # outlet lip, in mm
    assert rad.min() == pytest.approx(geom.R_throat * 1e3, rel=1e-4)    # the bore is open
    t = geom.config.wall_thickness * 1e3
    assert _stl_volume(tris) == pytest.approx(_exact_volume(cad.tube_loop(geom)), rel=2e-3)
    assert _stl_volume(tris) > 0.0                           # normals point out
    bore = (x >= 0.0) & (x < 0.5 * geom.z_stations[1] * 1e3)   # straight inlet pipe
    assert bore.any()
    wall = np.minimum(np.abs(rad[bore] - geom.R_inlet * 1e3),
                      np.abs(rad[bore] - geom.R_inlet * 1e3 - t))
    assert wall.max() < 1e-3


def test_tube_wall_keeps_its_thickness(tmp_path):
    geom = _geom(tmp_path, D=0.03153, beta=0.75, alpha_div_deg=15.0)
    inner = cad.wetted_path(geom)
    outer = cad.outer_path(inner, 3.0)
    assert len(outer) == len(inner)
    for a, b in zip(inner, outer):
        assert math.dist(a.start, b.start) == pytest.approx(3.0)
        assert math.dist(a.end, b.end) == pytest.approx(3.0)


@pytest.mark.parametrize("name", GEOMETRIES)
def test_fluid_stl_is_closed_and_fills_the_bore(tmp_path, name):
    geom = _geom(tmp_path, **GEOMETRIES[name])
    loop = cad.fluid_loop(geom)
    tris = _read_stl(cad.export_fluid_stl(geom, str(tmp_path)))
    assert _open_edges(tris) == 0
    assert _stl_volume(tris) == pytest.approx(_exact_volume(loop), rel=2e-3)
    assert tris[..., 0].max() == pytest.approx(geom.L_total * 1e3, rel=1e-6)
    # no point of the fluid sticks out of the bore
    outside = tris.reshape(-1, 3)
    rad = np.hypot(outside[:, 1], outside[:, 2])
    inside_tube = (outside[:, 0] >= 0.0)
    assert np.all(rad[inside_tube] <= geom.radius(outside[inside_tube, 0] / 1e3) * 1e3 + 1e-3)


@pytest.mark.parametrize("name", GEOMETRIES)
def test_step_files_are_valid_solids_in_mm(tmp_path, name):
    cq = pytest.importorskip("cadquery")
    geom = _geom(tmp_path, **GEOMETRIES[name])
    tube = cq.importers.importStep(cad.export_step(geom, str(tmp_path))).val()
    fluid = cq.importers.importStep(cad.export_fluid_step(geom, str(tmp_path))).val()
    assert tube.isValid() and fluid.isValid()
    assert tube.Volume() == pytest.approx(_exact_volume(cad.tube_loop(geom)), rel=1e-3)
    assert fluid.Volume() == pytest.approx(_exact_volume(cad.fluid_loop(geom)), rel=1e-3)
    assert tube.BoundingBox().xlen == pytest.approx(cad.tube_envelope(geom)[0], rel=1e-2)
    assert len(tube.Solids()) == len(fluid.Solids()) == 1


def test_dxf_profile_and_centerline(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    cfg, geom, mesh, uz, ur, p = _case(tmp_path)
    doc = ezdxf.readfile(cad.export_dxf(geom, cfg.output_dir))
    assert doc.layers.get("CENTERLINE").dxf.linetype == "CENTER"
    upper = doc.modelspace().query('LWPOLYLINE[layer=="PROFILE_UPPER"]')[0]
    ys = np.array([pt[1] for pt in upper.get_points("xy")])
    assert ys.max() == pytest.approx(geom.R_inlet, rel=1e-12)
    assert ys.min() == pytest.approx(geom.R_throat, rel=1e-2)
