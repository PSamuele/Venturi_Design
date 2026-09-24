"""CAD files: 2D profile (.dxf, needs ezdxf) and solid (.step, needs cadquery)."""

from __future__ import annotations

import os
from typing import Optional

try:
    import ezdxf
except ImportError:
    ezdxf = None

try:
    import cadquery as cq
except ImportError:
    cq = None


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


def export_step(geom, out_dir: str) -> Optional[str]:
    """Solid of the fluid volume (venturi_3d.step), axis along x."""
    if cq is None:
        print("      Warning: cadquery is not installed; skipping venturi_3d.step.")
        return None
    path = os.path.join(out_dir, "venturi_3d.step")
    z, r = geom.profile_points(200)
    pts = [(0.0, 0.0)] + list(zip(z, r)) + [(geom.L_total, 0.0)]
    try:
        wp = cq.Workplane("XY").polyline(pts).close().revolve(360, (0, 0, 0), (1, 0, 0))
        wp.val().exportStep(path)
        return path
    except Exception as e:     # cadquery raises many different types
        print(f"      Warning: STEP export failed: {e}")
        return None
