"""Writes every output file of a run into one folder."""

from __future__ import annotations

import os
from typing import List, Optional

from .cad import export_dxf, export_step
from .paraview import export_paraview_2d, export_paraview_3d, export_stl
from .plots import export_svg_geometry, export_svg_results
from .tables import export_pointcloud, export_report

__all__ = ["export_all"]


def export_all(mesh, uz, ur, p, geom, config, validation: Optional[dict] = None) -> List[str]:
    """Write all outputs into config.output_dir and return the files written.

    A file whose library is missing is skipped with a warning and is not
    in the returned list.
    """
    out = config.output_dir
    os.makedirs(out, exist_ok=True)
    written = [
        export_paraview_2d(mesh, uz, ur, p, out),
        export_paraview_3d(mesh, uz, ur, p, out),
        export_dxf(geom, out),
        export_stl(geom, out),
        export_step(geom, out),
        export_svg_geometry(geom, config, out),
        export_svg_results(mesh, uz, ur, p, geom, config, out),
        *export_pointcloud(mesh, uz, ur, p, out),
        export_report(config, geom, out, validation),
    ]
    files = [f for f in written if f and os.path.isfile(f)]
    for f in files:
        print(f"      written {f}")
    return files
