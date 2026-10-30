"""CAD files, all in millimetres with the axis along x.

    venturi_profile.dxf     2D wall line (needs ezdxf)
    venturi_3d.step/.stl    the hollow tube, ready to print (STEP needs cadquery)
    venturi_fluid.step/.stl the air or water inside it, ready to mesh for CFD

The STEP and STL files come from the same profile of exact lines and arcs:

    bell mouth -> wall of VenturiGeometry -> rounded lip at the outlet

The bell mouth lowers the entry loss: a rounded inlet loses about 0.04 of the
dynamic pressure, a sharp one about 0.5. The bore stays straight up to the
outlet, where the flow leaves cleanly, and both ends of the wall are closed by
a half circle. x = 0 is still the inlet plane of the flow solution, so the
bell mouth sits at negative x.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np

from ..geometry import Segment, arc_segment

try:
    import ezdxf
except ImportError:
    ezdxf = None

try:
    import cadquery as cq
except ImportError:
    cq = None

BELL_SWEEP_DEG = 45.0    # how far the bell mouth turns away from the axis
CHORD_TOL_MM = 0.01      # largest distance between an STL facet and the true wall
MAX_STATIONS = 400       # upper limit of the STL rings along the profile


def export_dxf(geom, out_dir: str) -> Optional[str]:
    """Upper and lower wall lines plus the axis (venturi_profile.dxf)."""
    if ezdxf is None:
        print("      Warning: ezdxf is not installed; skipping venturi_profile.dxf.")
        return None
    path = os.path.join(out_dir, "venturi_profile.dxf")
    doc = ezdxf.new("R2010", setup=True)   # setup=True loads the CENTER linetype
    msp = doc.modelspace()
    doc.layers.add("PROFILE_UPPER", color=1)
    doc.layers.add("PROFILE_LOWER", color=2)
    doc.layers.add("CENTERLINE", color=3, linetype="CENTER")
    z, r = geom.profile_points(200)
    msp.add_lwpolyline(list(zip(z, r)), dxfattribs={"layer": "PROFILE_UPPER"})
    msp.add_lwpolyline(list(zip(z, -r)), dxfattribs={"layer": "PROFILE_LOWER"})
    msp.add_line((0, 0), (geom.L_total, 0), dxfattribs={"layer": "CENTERLINE"})
    doc.saveas(path)
    return path


# ---------------------------------------------------------------------------
# Profiles, in mm
# ---------------------------------------------------------------------------
def bell_radius(D_mm: float, wall_mm: float) -> float:
    """Radius of the bell mouth at the inlet [mm]."""
    # TODO(human): choose the sizing rule; it must stay larger than wall_mm
    return max(0.2 * D_mm, 1.5 * wall_mm)


def wetted_path(geom) -> List[Segment]:
    """Surface the fluid touches, from the bell mouth to the outlet plane."""
    wall = [Segment(*(p and (p[0] * 1e3, p[1] * 1e3) for p in s))
            for s in geom.wall_segments()]
    R = geom.R_inlet * 1e3
    Rb = bell_radius(geom.config.D * 1e3, geom.config.wall_thickness * 1e3)
    a = math.radians(BELL_SWEEP_DEG)
    centre = (0.0, R + Rb)
    start = (-Rb * math.sin(a), centre[1] - Rb * math.cos(a))
    return [arc_segment(start, (0.0, R), centre)] + wall


def _offset_segment(s: Segment, t: float) -> Optional[Segment]:
    """Move a segment by t to the left of its direction; None if it vanishes."""
    if s.centre is None:
        dz, dr = s.end[0] - s.start[0], s.end[1] - s.start[1]
        k = t / math.hypot(dz, dr)
        move = lambda p: (p[0] - dr * k, p[1] + dz * k)
        return Segment(move(s.start), move(s.end))
    # The centre lies to the left of the direction when the wall curves that way
    cz, cr = s.centre
    left = ((s.end[0] - s.start[0]) * (cr - s.start[1])
            - (s.end[1] - s.start[1]) * (cz - s.start[0])) > 0.0
    Ra = math.hypot(s.start[0] - cz, s.start[1] - cr)
    Ra_new = Ra - t if left else Ra + t
    if Ra_new <= 1e-6:
        return None
    move = lambda p: (cz + (p[0] - cz) * Ra_new / Ra, cr + (p[1] - cr) * Ra_new / Ra)
    return Segment(move(s.start), move(s.end), move(s.mid), s.centre)


def _intersect(a: Segment, b: Segment) -> Tuple[float, float]:
    """Point where the lines through two straight segments cross."""
    x1, y1 = a.start
    dx1, dy1 = a.end[0] - x1, a.end[1] - y1
    x2, y2 = b.start
    dx2, dy2 = b.end[0] - x2, b.end[1] - y2
    u = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / (dx1 * dy2 - dy1 * dx2)
    return (x1 + u * dx1, y1 + u * dy1)


def outer_path(path: List[Segment], t: float) -> List[Segment]:
    """Outside of a wall of thickness t.

    Sharp corners, and concave arcs tighter than t, leave the straight
    neighbours apart or crossed: they are extended to meet at a corner.
    """
    out: List[Segment] = []
    for s in path:
        o = _offset_segment(s, t)
        if o is None:
            continue
        if out and math.dist(out[-1].end, o.start) > 1e-9:
            p = _intersect(out[-1], o)
            out[-1] = out[-1]._replace(end=p)
            o = o._replace(start=p)
        out.append(o)
    return out


def _lip(start, end, sign: int) -> Segment:
    """Half circle from start to end, bulging to the right of it (sign = +1)."""
    cz, cr = 0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1])
    mid = (cz + sign * 0.5 * (end[1] - start[1]), cr - sign * 0.5 * (end[0] - start[0]))
    return Segment(start, end, mid, (cz, cr))


def _reversed(s: Segment) -> Segment:
    return Segment(s.end, s.start, s.mid, s.centre)


def tube_loop(geom) -> List[Segment]:
    """Closed cross-section of the tube wall."""
    inner = wetted_path(geom)
    outer = outer_path(inner, geom.config.wall_thickness * 1e3)
    return (inner
            + [_lip(inner[-1].end, outer[-1].end, 1)]
            + [_reversed(s) for s in reversed(outer)]
            + [_lip(outer[0].start, inner[0].start, 1)])


def fluid_loop(geom) -> List[Segment]:
    """Cross-section of the fluid, open along the axis."""
    path = wetted_path(geom)
    z0, z1 = path[0].start[0], path[-1].end[0]
    return ([Segment((z0, 0.0), path[0].start)] + path
            + [Segment(path[-1].end, (z1, 0.0))])


def tube_envelope(geom) -> Tuple[float, float]:
    """Overall length and largest diameter of the tube [mm]."""
    pts = _sample(tube_loop(geom), 1.0, 0.05)
    return float(np.ptp(pts[:, 0])), float(2.0 * pts[:, 1].max())


# ---------------------------------------------------------------------------
# STEP
# ---------------------------------------------------------------------------
def _write_step(loop: List[Segment], path: str) -> Optional[str]:
    """Revolve a profile about x into a solid and save it."""
    name = os.path.basename(path)
    if cq is None:
        print(f"      Warning: cadquery is not installed; skipping {name}.")
        return None
    try:
        wp = cq.Workplane("XY").moveTo(*loop[0].start)
        for s in loop:
            wp = wp.lineTo(*s.end) if s.mid is None else wp.threePointArc(s.mid, s.end)
        solid = wp.close().revolve(360, (0, 0, 0), (1, 0, 0)).val()
        if not solid.isValid():
            print(f"      Warning: {name} not written, the solid is not valid.")
            return None
        solid.exportStep(path)
        return path
    except Exception as e:     # cadquery raises many different types
        print(f"      Warning: STEP export failed: {e}")
        return None


def export_step(geom, out_dir: str) -> Optional[str]:
    """Hollow tube as a solid (venturi_3d.step)."""
    return _write_step(tube_loop(geom), os.path.join(out_dir, "venturi_3d.step"))


def export_fluid_step(geom, out_dir: str) -> Optional[str]:
    """Fluid volume as a solid (venturi_fluid.step)."""
    return _write_step(fluid_loop(geom), os.path.join(out_dir, "venturi_fluid.step"))


# ---------------------------------------------------------------------------
# STL
# ---------------------------------------------------------------------------
def _sample(loop: List[Segment], ds: float, dtheta: float) -> np.ndarray:
    """Points along a profile, every joint between segments included."""
    pts = [loop[0].start]
    for s in loop:
        if s.centre is None:
            n = max(1, math.ceil(math.dist(s.start, s.end) / ds))
            pts += [(s.start[0] + (s.end[0] - s.start[0]) * i / n,
                     s.start[1] + (s.end[1] - s.start[1]) * i / n) for i in range(1, n + 1)]
            continue
        cz, cr = s.centre
        a0 = math.atan2(s.start[1] - cr, s.start[0] - cz)
        half = math.atan2(s.mid[1] - cr, s.mid[0] - cz) - a0
        half = (half + math.pi) % (2 * math.pi) - math.pi
        Ra = math.dist(s.start, s.centre)
        n = max(1, math.ceil(abs(2 * half) * Ra / ds), math.ceil(abs(2 * half) / dtheta))
        pts += [(cz + Ra * math.cos(a0 + 2 * half * i / n),
                 cr + Ra * math.sin(a0 + 2 * half * i / n)) for i in range(1, n)]
        pts.append(s.end)
    return np.array(pts)


def _revolve_mesh(profile: np.ndarray, n_theta: int) -> np.ndarray:
    """Triangles (n, 3, 3) of a profile spun about x, normals pointing out.

    Each point becomes a ring of n_theta vertices, or a single vertex on the
    axis. Neighbouring rings share their vertices, so a closed profile, or one
    that starts and ends on the axis, gives a surface with no open edge.
    """
    closed = np.allclose(profile[0], profile[-1])
    if closed:
        profile = profile[:-1]
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    rings, verts, count = [], [], 0
    for z, r in profile:
        if r < 1e-9:
            verts.append([[z, 0.0, 0.0]])
            rings.append(np.full(n_theta, count))
            count += 1
        else:
            verts.append(np.column_stack((np.full(n_theta, z), r * np.cos(theta), r * np.sin(theta))))
            rings.append(np.arange(count, count + n_theta))
            count += n_theta
    verts = np.vstack(verts)

    faces = []
    for a, b in zip(rings, rings[1:] + rings[:1] if closed else rings[1:]):
        a1, b1 = np.roll(a, -1), np.roll(b, -1)
        if b[0] != b1[0]:
            faces.append(np.column_stack((a, b, b1)))
        if a[0] != a1[0]:
            faces.append(np.column_stack((a, b1, a1)))
    tris = verts[np.vstack(faces)]
    signed = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum()
    return tris[:, ::-1] if signed < 0.0 else tris


def _write_stl(loop: List[Segment], path: str) -> str:
    """Mesh a profile with nearly square cells and save it as binary STL."""
    r_max = max(p[1] for s in loop for p in (s.start, s.end)) + CHORD_TOL_MM
    dtheta = 2.0 * math.acos(1.0 - CHORD_TOL_MM / r_max)
    n_theta = max(72, math.ceil(2.0 * math.pi / dtheta))
    length = sum(math.dist(s.start, s.end) for s in loop)
    ds = max(2.0 * math.pi * r_max / n_theta, length / MAX_STATIONS)
    tris = _revolve_mesh(_sample(loop, ds, dtheta), n_theta)

    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    data = np.zeros(len(tris), dtype=[("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")])
    data["n"] = normals
    data["v"] = tris.reshape(-1, 9)
    with open(path, "wb") as f:
        f.write(b"Venturi tube, millimetres".ljust(80, b" "))
        f.write(np.uint32(len(tris)).tobytes())
        data.tofile(f)
    return path


def export_stl(geom, out_dir: str) -> str:
    """Hollow tube as a closed mesh (venturi_3d.stl)."""
    return _write_stl(tube_loop(geom), os.path.join(out_dir, "venturi_3d.stl"))


def export_fluid_stl(geom, out_dir: str) -> str:
    """Fluid volume as a closed mesh (venturi_fluid.stl)."""
    return _write_stl(fluid_loop(geom), os.path.join(out_dir, "venturi_fluid.stl"))
