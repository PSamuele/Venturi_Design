"""Figures as .svg (matplotlib, no window needed)."""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def export_svg_geometry(geom, config, out_dir: str) -> str:
    """Side view of the wall profile (venturi_geometry.svg)."""
    path = os.path.join(out_dir, "venturi_geometry.svg")
    z, r = geom.profile_points(500)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(z, r, "b-", linewidth=2, label="wall")
    ax.plot(z, -r, "b-", linewidth=2)
    ax.plot([0, geom.L_total], [0, 0], "k--", alpha=0.5, label="axis")
    ax.set_title(f"Venturi profile - D = {config.D*1e3:.1f} mm, d = {config.d*1e3:.1f} mm, "
                 f"alpha_conv = {config.alpha_conv_deg:.1f} deg, "
                 f"alpha_div = {config.alpha_div_deg:.1f} deg")
    ax.set_xlabel("z, distance along the axis [m]")
    ax.set_ylabel("r, radius [m]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def export_svg_results(mesh, uz, ur, p, geom, config, out_dir: str) -> str:
    """Speed map, pressure map and pressure along the axis (venturi_results.svg)."""
    path = os.path.join(out_dir, "venturi_results.svg")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    z, r = mesh.z_cc, mesh.r_c

    c1 = ax1.contourf(z, r, np.hypot(uz, ur), levels=50, cmap="viridis")
    fig.colorbar(c1, ax=ax1, label="speed [m/s]")
    sz, sr = max(1, z.shape[0] // 30), max(1, z.shape[1] // 15)
    ax1.quiver(z[::sz, ::sr], r[::sz, ::sr], uz[::sz, ::sr], ur[::sz, ::sr],
               color="white", alpha=0.7)
    ax1.set_title("Speed (arrows show the flow direction)")
    ax1.set_ylabel("r [m]")

    c2 = ax2.contourf(z, r, p, levels=50, cmap="inferno")
    fig.colorbar(c2, ax=ax2, label="pressure [Pa]")
    ax2.set_title("Pressure")
    ax2.set_ylabel("r [m]")

    # Frictionless reference: same flow rate, speed = Q / local area
    z_ax = z[:, 0]
    v_ideal = config.v_inlet * (config.R_inlet / np.maximum(geom.radius(z_ax), 1e-10)) ** 2
    p_ideal = config.p_inlet + 0.5 * config.rho * (config.v_inlet**2 - v_ideal**2)
    ax3.plot(z_ax, p[:, 0], "r-", linewidth=2, label="computed (cells next to the axis)")
    ax3.plot(z_ax, p_ideal, "k--", alpha=0.7, label="Bernoulli, no friction")
    ax3.set_title("Pressure along the axis")
    ax3.set_xlabel("z [m]")
    ax3.set_ylabel("pressure [Pa]")
    ax3.legend()
    ax3.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path
