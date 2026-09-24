"""Point cloud (.csv, .ply) and the Markdown report."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from . import paraview


def export_pointcloud(mesh, uz, ur, p, out_dir: str, n_angles: int = 8) -> Tuple[str, str]:
    """Cell centres copied at n_angles angles around the axis.

    Columns: x (axial), y, z, pressure, speed. Returns (ply_path, csv_path);
    ply_path is "" when pyvista is missing or the write fails.
    """
    ply_path = os.path.join(out_dir, "venturi_pointcloud.ply")
    csv_path = os.path.join(out_dir, "venturi_pointcloud.csv")

    theta = np.radians(np.arange(n_angles) * 360.0 / n_angles)[:, None]   # (n, 1)
    z = mesh.z_cc.ravel()[None, :]
    r = mesh.r_c.ravel()[None, :]
    x_all = np.broadcast_to(z, (n_angles, z.size)).ravel()
    y_all = (r * np.cos(theta)).ravel()
    z_all = (r * np.sin(theta)).ravel()
    p_all = np.tile(p.ravel(), n_angles)
    v_all = np.tile(np.hypot(uz, ur).ravel(), n_angles)

    data = np.column_stack((x_all, y_all, z_all, p_all, v_all))
    np.savetxt(csv_path, data, delimiter=",", fmt="%.10g",
               header="x,y,z,pressure,velocity_magnitude", comments="")

    if paraview.pv is None:
        return "", csv_path
    cloud = paraview.pv.PolyData(data[:, :3])
    cloud.point_data["Pressure"] = p_all
    cloud.point_data["Velocity_Magnitude"] = v_all
    try:
        cloud.save(ply_path)
    except Exception as e:     # the PLY writer can fail on some VTK builds
        print(f"      Warning: PLY export failed: {e}")
        ply_path = ""
    return ply_path, csv_path


def export_report(config, geom, out_dir: str, validation: Optional[dict] = None) -> str:
    """Short Markdown summary of the run (venturi_report.md)."""
    path = os.path.join(out_dir, "venturi_report.md")
    lines = [
        "# Venturi run report", "",
        "## Inputs", "",
        f"- fluid: `{config.fluid}`, rho = {config.rho:.4g} kg/m3, mu = {config.mu:.3e} Pa s",
        f"- p_inlet = {config.p_inlet:.1f} Pa, target throat pressure = "
        f"{config.p_throat_target if config.p_throat_target is not None else '-'} Pa",
        f"- flow rate Q = {config.Q:.6f} m3/s ({config.Q*1e3:.3f} L/s), Re_D = {config.Re_D:.0f}",
        "",
        "## Geometry", "",
        f"- D = {config.D*1e3:.2f} mm, d = {config.d*1e3:.2f} mm, beta = {config.beta:.3f}",
        f"- alpha_conv = {config.alpha_conv_deg:.1f} deg, alpha_div = {config.alpha_div_deg:.1f} deg",
        f"- total length = {geom.L_total*1e3:.2f} mm",
        "",
    ]
    if validation:
        lines += ["## Results", "", "| item | value | status |", "|---|---|---|"]
        for k, v in validation.items():
            if isinstance(v, dict) and "passed" in v:
                val = v["value"]
                val = f"{val:.6g}" if isinstance(val, float) else val
                lines.append(f"| {k} | {val} | {'PASS' if v['passed'] else 'FAIL'} |")
            else:
                lines.append(f"| {k} | {v} | - |")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
