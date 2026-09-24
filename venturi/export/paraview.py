"""ParaView files (.vts) and the wall surface (.stl). Need pyvista."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None


def _point_array(a: np.ndarray) -> np.ndarray:
    """Flatten a field in the order PyVista uses for the points.

    pv.StructuredGrid(x, y, z) flattens the coordinates in Fortran order
    (first index fastest). Point data must be flattened the same way,
    otherwise every value lands on the wrong point.
    """
    return np.asarray(a).ravel(order="F")


def _attach_fields(grid, uz, u_y, u_z, p) -> None:
    """Velocity (x = axial), speed and pressure on a structured grid."""
    grid.point_data["Velocity"] = np.column_stack(
        (_point_array(uz), _point_array(u_y), _point_array(u_z)))
    grid.point_data["Velocity_Magnitude"] = _point_array(np.sqrt(uz**2 + u_y**2 + u_z**2))
    grid.point_data["Pressure"] = _point_array(p)


def export_paraview_2d(mesh, uz, ur, p, out_dir: str) -> Optional[str]:
    """Meridional plane (x = axial, y = radial) as venturi_cfd_2d.vts."""
    if pv is None:
        print("      Warning: pyvista is not installed; skipping venturi_cfd_2d.vts.")
        return None
    path = os.path.join(out_dir, "venturi_cfd_2d.vts")
    grid = pv.StructuredGrid(mesh.z_cc, mesh.r_c, np.zeros_like(mesh.z_cc))
    _attach_fields(grid, uz, ur, np.zeros_like(ur), p)
    grid.point_data["Axial_Velocity"] = _point_array(uz)
    grid.point_data["Radial_Velocity"] = _point_array(ur)
    grid.save(path)
    return path


def export_paraview_3d(mesh, uz, ur, p, out_dir: str, n_theta: int = 72) -> Optional[str]:
    """Full 3D field as venturi_cfd_3d.vts, made by spinning the plane about x.

    The (Nz, Nr) plane is rotated in n_theta steps, giving an
    (Nz, Nr, n_theta+1) block. The radial velocity is split into its y and z
    parts. The points are the cell centres, so the volume stops half a cell
    short of the wall and of the axis.
    """
    if pv is None:
        print("      Warning: pyvista is not installed; skipping venturi_cfd_3d.vts.")
        return None
    path = os.path.join(out_dir, "venturi_cfd_3d.vts")
    th = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)[None, None, :]
    z3 = np.broadcast_to(mesh.z_cc[:, :, None], mesh.z_cc.shape + (th.shape[2],))
    r3 = mesh.r_c[:, :, None]
    grid = pv.StructuredGrid(np.ascontiguousarray(z3), r3 * np.cos(th), r3 * np.sin(th))
    rep = lambda a: np.broadcast_to(a[:, :, None], a.shape + (th.shape[2],))
    ur3 = ur[:, :, None]
    _attach_fields(grid, rep(uz), ur3 * np.cos(th), ur3 * np.sin(th), rep(p))
    grid.save(path)
    return path


def export_stl(geom, out_dir: str) -> Optional[str]:
    """Inner wall as a triangle surface (venturi_3d.stl), axis along x."""
    if pv is None:
        print("      Warning: pyvista is not installed; skipping venturi_3d.stl.")
        return None
    path = os.path.join(out_dir, "venturi_3d.stl")
    z, r = geom.profile_points(200)
    points = np.column_stack((z, r, np.zeros_like(z)))
    n = len(points)
    # VTK line list: [2, i0, i1, 2, i1, i2, ...]
    lines = np.column_stack((np.full(n - 1, 2), np.arange(n - 1), np.arange(1, n))).ravel()
    surf = pv.PolyData(points, lines=lines).extrude_rotate(
        resolution=72, angle=360.0, capping=False, rotation_axis=(1.0, 0.0, 0.0))
    surf.triangulate().save(path)
    return path
