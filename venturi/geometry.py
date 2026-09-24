"""
geometry.py
===========
Wall profile R(z) of the Venturi tube.

The profile is made of five pieces, from left to right:

    straight inlet pipe  ->  converging cone  ->  straight throat
    ->  diverging cone  ->  straight outlet pipe

Each of the four corners between pieces can be rounded with a circular arc
that touches both neighbouring pieces tangentially (no kink in the wall).
For an arc of radius Ra between two lines meeting at an angle theta, the
arc starts and ends at a distance T = Ra * tan(theta/2) from the corner.
T is capped so that an arc never eats more than part of a piece; Ra is then
recomputed from the capped T.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .config import VenturiConfig

# One arc: (z_start, z_end, z_centre, r_centre, radius, sign)
# sign = +1: the arc bulges outwards (centre below the wall)
# sign = -1: the arc bulges inwards  (centre above the wall)
Arc = Tuple[float, float, float, float, float, int]


class VenturiGeometry:
    """Wall profile built from a VenturiConfig.

    Attributes:
        z_stations: nominal corner positions [z0 .. z5] along the axis [m]
        L_inlet, L_conv, L_throat, L_div, L_outlet, L_total: lengths [m]
        R_inlet, R_throat: pipe and throat radius [m]
        blend_R1 .. blend_R4: actual radius of the four corner arcs [m]
    """

    def __init__(self, config: VenturiConfig):
        self.config = config
        D, d = config.D, config.d
        R_in, R_th = config.R_inlet, config.R_throat
        th1 = 0.5 * config.alpha_conv_rad      # half angle of the converging cone
        th2 = 0.5 * config.alpha_div_rad       # half angle of the diverging cone
        self._tan1 = math.tan(th1)
        self._tan2 = math.tan(th2)

        self.L_inlet = config.L_inlet_ratio * D
        self.L_conv = (R_in - R_th) / self._tan1
        self.L_throat = config.L_throat_ratio * d
        self.L_div = (R_in - R_th) / self._tan2
        self.L_outlet = config.L_outlet_ratio * D
        self.L_total = (self.L_inlet + self.L_conv + self.L_throat
                        + self.L_div + self.L_outlet)
        self.z_stations = np.cumsum([0.0, self.L_inlet, self.L_conv,
                                     self.L_throat, self.L_div, self.L_outlet])
        self.R_inlet = R_in
        self.R_throat = R_th

        self.arcs: List[Arc] = []
        self.blend_R1 = self.blend_R2 = self.blend_R3 = self.blend_R4 = 0.0
        if not config.use_blend_radii:
            return

        z1, z2, z3, z4 = self.z_stations[1:5]

        # Arc 1: inlet pipe -> converging cone (bulges outwards)
        T = min(config.blend_R1_ratio * D * math.tan(th1 / 2),
                0.80 * self.L_inlet, 0.45 * self.L_conv)
        self.blend_R1 = Ra = T / math.tan(th1 / 2)
        self.arcs.append((z1 - T, z1 + T * math.cos(th1), z1 - T, R_in - Ra, Ra, 1))

        # Arc 2: converging cone -> throat (bulges inwards)
        T = min(config.blend_R2_ratio * d * math.tan(th1 / 2),
                0.45 * self.L_conv, 0.45 * self.L_throat)
        self.blend_R2 = Ra = T / math.tan(th1 / 2)
        self.arcs.append((z2 - T * math.cos(th1), z2 + T, z2 + T, R_th + Ra, Ra, -1))

        # Arc 3: throat -> diverging cone (bulges inwards)
        T = min(config.blend_R3_ratio * d * math.tan(th2 / 2),
                0.45 * self.L_throat, 0.45 * self.L_div)
        self.blend_R3 = Ra = T / math.tan(th2 / 2)
        self.arcs.append((z3 - T, z3 + T * math.cos(th2), z3 - T, R_th + Ra, Ra, -1))

        # Arc 4: diverging cone -> outlet pipe (bulges outwards)
        T = min(config.blend_R4_ratio * D * math.tan(th2 / 2),
                0.45 * self.L_div, 0.80 * self.L_outlet)
        self.blend_R4 = Ra = T / math.tan(th2 / 2)
        self.arcs.append((z4 - T * math.cos(th2), z4 + T, z4 + T, R_in - Ra, Ra, 1))

    # ------------------------------------------------------------------------
    def radius(self, z):
        """Wall radius R(z) [m]. Accepts a number or an array."""
        z = np.asarray(z, dtype=float)
        z1, z2, z3, z4 = self.z_stations[1:5]
        r = np.select(
            [z <= z1, z <= z2, z <= z3, z <= z4],
            [np.full_like(z, self.R_inlet),
             self.R_inlet - (z - z1) * self._tan1,
             np.full_like(z, self.R_throat),
             self.R_throat + (z - z3) * self._tan2],
            default=self.R_inlet)
        for z_s, z_e, z_c, r_c, Ra, sign in self.arcs:
            inside = (z >= z_s) & (z <= z_e)
            arc = r_c + sign * np.sqrt(np.maximum(Ra**2 - (z - z_c)**2, 0.0))
            r = np.where(inside, arc, r)
        return float(r) if r.ndim == 0 else r

    def radius_derivative(self, z):
        """Slope of the wall dR/dz [-]. Accepts a number or an array."""
        z = np.asarray(z, dtype=float)
        z1, z2, z3, z4 = self.z_stations[1:5]
        dr = np.select([(z > z1) & (z <= z2), (z > z3) & (z <= z4)],
                       [np.full_like(z, -self._tan1), np.full_like(z, self._tan2)],
                       default=0.0)
        for z_s, z_e, z_c, r_c, Ra, sign in self.arcs:
            inside = (z >= z_s) & (z <= z_e)
            slope = -sign * (z - z_c) / np.sqrt(np.maximum(Ra**2 - (z - z_c)**2, 1e-14))
            dr = np.where(inside, slope, dr)
        return float(dr) if dr.ndim == 0 else dr

    def profile_points(self, n_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """n_points evenly spaced (z, R(z)) pairs from inlet to outlet."""
        z = np.linspace(0.0, self.L_total, n_points)
        return z, self.radius(z)


def create_venturi_geometry(config: VenturiConfig) -> VenturiGeometry:
    return VenturiGeometry(config)
