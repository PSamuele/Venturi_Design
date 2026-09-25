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


# Blue = low, red = high, 20 bands with thin lines between them: easier to
# read values off than a smooth gradient.
CMAP = "RdYlBu_r"
LEVELS = 20


def _arrow_sample(mesh, n_axial: int = 36, n_radial: int = 8):
    """Cell indices for the arrows: evenly spread along z and across the
    section (in eta = r/R), so they don't crowd the thin wall cells."""
    z = mesh.z_c
    i_idx = np.unique(np.searchsorted(z, np.linspace(z[0], z[-1], n_axial)).clip(0, len(z) - 1))
    eta_t = np.linspace(0.08, 0.92, n_radial)
    j_idx = np.unique(np.abs(mesh.eta_c[None, :] - eta_t[:, None]).argmin(axis=1))
    return np.ix_(i_idx, j_idx)


def _field_panel(fig, ax, z, r, values, label, title):
    c = ax.contourf(z, r, values, levels=LEVELS, cmap=CMAP)
    ax.contour(z, r, values, levels=c.levels, colors="k", linewidths=0.2, alpha=0.4)
    fig.colorbar(c, ax=ax, label=label)
    ax.set_title(title)
    ax.set_ylabel("r [m]")
    return c


def export_svg_results(mesh, uz, ur, p, geom, config, out_dir: str) -> str:
    """Speed map, pressure map and pressure along the axis (venturi_results.svg)."""
    path = os.path.join(out_dir, "venturi_results.svg")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    z, r = mesh.z_cc, mesh.r_c
    speed = np.hypot(uz, ur)

    _field_panel(fig, ax1, z, r, speed, "speed [m/s]",
                 "Speed, arrows show the flow direction (radial scale stretched)")
    # The radial axis is stretched many times compared with the axial one,
    # so the cones look much steeper than they are. angles="xy" draws each
    # arrow in the same stretched coordinates, so it follows the drawn wall;
    # with the default the arrows keep their true (small) angle and look
    # horizontal. Arrows have equal length: the colour already gives speed.
    k = _arrow_sample(mesh)
    s = np.maximum(speed[k], 1e-12)
    ax1.quiver(z[k], r[k], uz[k] / s, ur[k] / s, angles="xy",
               color="k", width=0.0016, headwidth=4, scale=45, pivot="mid")

    _field_panel(fig, ax2, z, r, p / 1000.0, "pressure [kPa]", "Pressure")

    # Frictionless reference: same flow rate, speed = Q / local area
    z_ax = z[:, 0]
    v_ideal = config.v_inlet * (config.R_inlet / np.maximum(geom.radius(z_ax), 1e-10)) ** 2
    p_ideal = config.p_inlet + 0.5 * config.rho * (config.v_inlet**2 - v_ideal**2)
    ax3.plot(z_ax, p[:, 0] / 1000.0, "r-", linewidth=2, label="computed (cells next to the axis)")
    ax3.plot(z_ax, p_ideal / 1000.0, "k--", alpha=0.7, label="Bernoulli, no friction")
    ax3.set_title("Pressure along the axis")
    ax3.set_xlabel("z [m]")
    ax3.set_ylabel("pressure [kPa]")
    ax3.legend()
    ax3.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path
