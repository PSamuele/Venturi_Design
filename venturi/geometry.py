"""
Venturi Tube Design Tool — Geometry Module
===========================================
Analytical geometry generation adhering strictly to ISO 5167-4 specifications
with all 4 circular blending arcs (R1, R2, R3, R4) ensuring exact C1 continuity,
analytical first and second spatial derivatives R'(z) and R''(z),
and C2-continuous parametric Bézier/B-spline profiles for CFD shape optimization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np

from .config import VenturiConfig


# ---------------------------------------------------------------------------
# VenturiGeometry Class
# ---------------------------------------------------------------------------
@dataclass
class VenturiGeometry:
    """
    ISO 5167-4 compliant Venturi tube geometry with exact circular blending arcs.

    Attributes:
        z_stations: 1D array of nominal junction stations [z0, z1, z2, z3, z4, z5].
        L_inlet: Length of straight inlet section [m].
        L_conv: Length of nominal convergent cone [m].
        L_throat: Length of straight throat section [m].
        L_div: Length of nominal divergent cone [m].
        L_outlet: Length of straight outlet section [m].
        L_total: Total axial length of the Venturi tube [m].
        R_inlet: Pipe inlet / outlet radius [m].
        R_throat: Throat radius [m].
        blend_R1: Actual radius of Inlet -> Convergent fillet arc [m].
        blend_R2: Actual radius of Convergent -> Throat fillet arc [m].
        blend_R3: Actual radius of Throat -> Divergent fillet arc [m].
        blend_R4: Actual radius of Divergent -> Outlet fillet arc [m].
        config: Master VenturiConfig reference.
    """
    z_stations: np.ndarray
    L_inlet: float
    L_conv: float
    L_throat: float
    L_div: float
    L_outlet: float
    L_total: float
    R_inlet: float
    R_throat: float
    blend_R1: float
    blend_R2: float
    blend_R3: float
    blend_R4: float
    config: VenturiConfig

    # Pre-computed arc parameters: (z_start, z_end, z_center, r_center, R_arc, sign)
    _arc1: Optional[Tuple[float, float, float, float, float, int]] = None
    _arc2: Optional[Tuple[float, float, float, float, float, int]] = None
    _arc3: Optional[Tuple[float, float, float, float, float, int]] = None
    _arc4: Optional[Tuple[float, float, float, float, float, int]] = None

    def __init__(self, config: Optional[VenturiConfig] = None, **kwargs: Any):
        """
        Initialize VenturiGeometry from VenturiConfig or direct keyword arguments.
        """
        if config is None:
            if "config" in kwargs and isinstance(kwargs["config"], VenturiConfig):
                config = kwargs["config"]
            else:
                config = VenturiConfig(**kwargs)

        self.config = config
        D = config.D
        d = config.d
        R_in = config.R_inlet
        R_th = config.R_throat

        theta1 = config.alpha_conv_rad / 2.0
        theta2 = config.alpha_div_rad / 2.0

        # Section lengths
        L_inlet = config.L_inlet_ratio * D
        L_conv = (R_in - R_th) / math.tan(theta1)
        L_throat = config.L_throat_ratio * d
        L_div = (R_in - R_th) / math.tan(theta2)
        L_outlet = config.L_outlet_ratio * D
        L_total = L_inlet + L_conv + L_throat + L_div + L_outlet

        # Nominal junction stations
        z0 = 0.0
        z1 = z0 + L_inlet
        z2 = z1 + L_conv
        z3 = z2 + L_throat
        z4 = z3 + L_div
        z5 = z4 + L_outlet
        stations = np.array([z0, z1, z2, z3, z4, z5], dtype=float)

        self.z_stations = stations
        self.L_inlet = L_inlet
        self.L_conv = L_conv
        self.L_throat = L_throat
        self.L_div = L_div
        self.L_outlet = L_outlet
        self.L_total = L_total
        self.R_inlet = R_in
        self.R_throat = R_th

        # Circular blending arcs computation
        R1 = 0.0
        R2 = 0.0
        R3 = 0.0
        R4 = 0.0
        arc1 = None
        arc2 = None
        arc3 = None
        arc4 = None

        if config.use_blend_radii:
            # Arc 1: Inlet -> Convergent Transition (convex fillet)
            R1_target = config.blend_R1_ratio * D
            T1 = R1_target * math.tan(theta1 / 2.0)
            T1 = min(T1, 0.80 * L_inlet, 0.45 * L_conv)
            R1 = T1 / math.tan(theta1 / 2.0)

            z_t1_in = z1 - T1
            z_t1_co = z1 + T1 * math.cos(theta1)
            z_c1 = z_t1_in
            r_c1 = R_in - R1
            arc1 = (z_t1_in, z_t1_co, z_c1, r_c1, R1, 1)

            # Arc 2: Convergent -> Throat Transition (concave fillet)
            R2_target = config.blend_R2_ratio * d
            T2 = R2_target * math.tan(theta1 / 2.0)
            T2 = min(T2, 0.45 * L_conv, 0.45 * L_throat)
            R2 = T2 / math.tan(theta1 / 2.0)

            z_t2_co = z2 - T2 * math.cos(theta1)
            z_t2_th = z2 + T2
            z_c2 = z_t2_th
            r_c2 = R_th + R2
            arc2 = (z_t2_co, z_t2_th, z_c2, r_c2, R2, -1)

            # Arc 3: Throat -> Divergent Transition (concave fillet)
            R3_target = config.blend_R3_ratio * d
            T3 = R3_target * math.tan(theta2 / 2.0)
            T3 = min(T3, 0.45 * L_throat, 0.45 * L_div)
            R3 = T3 / math.tan(theta2 / 2.0)

            z_t3_th = z3 - T3
            z_t3_di = z3 + T3 * math.cos(theta2)
            z_c3 = z_t3_th
            r_c3 = R_th + R3
            arc3 = (z_t3_th, z_t3_di, z_c3, r_c3, R3, -1)

            # Arc 4: Divergent -> Outlet Transition (convex fillet)
            R4_target = config.blend_R4_ratio * D
            T4 = R4_target * math.tan(theta2 / 2.0)
            T4 = min(T4, 0.45 * L_div, 0.80 * L_outlet)
            R4 = T4 / math.tan(theta2 / 2.0)

            z_t4_di = z4 - T4 * math.cos(theta2)
            z_t4_out = z4 + T4
            z_c4 = z_t4_out
            r_c4 = R_in - R4
            arc4 = (z_t4_di, z_t4_out, z_c4, r_c4, R4, 1)

        self.blend_R1 = R1
        self.blend_R2 = R2
        self.blend_R3 = R3
        self.blend_R4 = R4
        self._arc1 = arc1
        self._arc2 = arc2
        self._arc3 = arc3
        self._arc4 = arc4

    # -----------------------------------------------------------------
    # Primary Geometry Evaluation Methods
    # -----------------------------------------------------------------
    def radius(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate wall radius R(z) [m] at axial position(s) z.
        """
        z_arr = np.asarray(z, dtype=float)
        is_scalar = z_arr.ndim == 0
        z_1d = np.atleast_1d(z_arr)
        r = np.zeros_like(z_1d, dtype=float)

        z1, z2, z3, z4 = self.z_stations[1:5]
        tan_theta1 = math.tan(self.config.alpha_conv_rad / 2.0)
        tan_theta2 = math.tan(self.config.alpha_div_rad / 2.0)

        # 1. Base piecewise linear profile
        mask_inlet = z_1d <= z1
        mask_conv = (z_1d > z1) & (z_1d <= z2)
        mask_throat = (z_1d > z2) & (z_1d <= z3)
        mask_div = (z_1d > z3) & (z_1d <= z4)
        mask_outlet = z_1d > z4

        r[mask_inlet] = self.R_inlet
        r[mask_conv] = self.R_inlet - (z_1d[mask_conv] - z1) * tan_theta1
        r[mask_throat] = self.R_throat
        r[mask_div] = self.R_throat + (z_1d[mask_div] - z3) * tan_theta2
        r[mask_outlet] = self.R_inlet

        # 2. Overwrite blending arc intervals if enabled
        if self.config.use_blend_radii:
            for arc in (self._arc1, self._arc2, self._arc3, self._arc4):
                if arc is not None:
                    z_s, z_e, z_c, r_c, R, sign = arc
                    mask_arc = (z_1d >= z_s) & (z_1d <= z_e)
                    if np.any(mask_arc):
                        arg = np.maximum(R**2 - (z_1d[mask_arc] - z_c)**2, 0.0)
                        r[mask_arc] = r_c + sign * np.sqrt(arg)

        if is_scalar:
            return float(r[0])
        return r.reshape(z_arr.shape)

    def radius_derivative(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate first spatial derivative dR/dz at axial position(s) z.
        """
        z_arr = np.asarray(z, dtype=float)
        is_scalar = z_arr.ndim == 0
        z_1d = np.atleast_1d(z_arr)
        dr = np.zeros_like(z_1d, dtype=float)

        z1, z2, z3, z4 = self.z_stations[1:5]
        tan_theta1 = math.tan(self.config.alpha_conv_rad / 2.0)
        tan_theta2 = math.tan(self.config.alpha_div_rad / 2.0)

        # Base piecewise slopes
        mask_conv = (z_1d > z1) & (z_1d <= z2)
        mask_div = (z_1d > z3) & (z_1d <= z4)
        dr[mask_conv] = -tan_theta1
        dr[mask_div] = tan_theta2

        # Overwrite with exact circular arc derivatives
        if self.config.use_blend_radii:
            for arc in (self._arc1, self._arc2, self._arc3, self._arc4):
                if arc is not None:
                    z_s, z_e, z_c, r_c, R, sign = arc
                    mask_arc = (z_1d >= z_s) & (z_1d <= z_e)
                    if np.any(mask_arc):
                        arg = np.maximum(R**2 - (z_1d[mask_arc] - z_c)**2, 1e-14)
                        dr[mask_arc] = -sign * (z_1d[mask_arc] - z_c) / np.sqrt(arg)

        if is_scalar:
            return float(dr[0])
        return dr.reshape(z_arr.shape)

    def radius_second_derivative(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate second spatial derivative d^2R/dz^2 at axial position(s) z.
        """
        z_arr = np.asarray(z, dtype=float)
        is_scalar = z_arr.ndim == 0
        z_1d = np.atleast_1d(z_arr)
        d2r = np.zeros_like(z_1d, dtype=float)

        # In straight and conical sections, d2R/dz2 = 0
        # In circular blending arcs:
        if self.config.use_blend_radii:
            for arc in (self._arc1, self._arc2, self._arc3, self._arc4):
                if arc is not None:
                    z_s, z_e, z_c, r_c, R, sign = arc
                    mask_arc = (z_1d >= z_s) & (z_1d <= z_e)
                    if np.any(mask_arc):
                        arg = np.maximum(R**2 - (z_1d[mask_arc] - z_c)**2, 1e-14)
                        # d/dz [-sign * (z - z_c) / sqrt(R^2 - (z-z_c)^2)]
                        # = -sign * R^2 / (R^2 - (z - z_c)^2)^(3/2)
                        d2r[mask_arc] = -sign * (R**2) / (arg ** 1.5)

        if is_scalar:
            return float(d2r[0])
        return d2r.reshape(z_arr.shape)

    def get_stations(self) -> Dict[str, float]:
        """
        Return named dictionary of nominal junction station coordinates.
        """
        return {
            "z0_inlet_start": float(self.z_stations[0]),
            "z1_conv_start": float(self.z_stations[1]),
            "z2_throat_start": float(self.z_stations[2]),
            "z3_div_start": float(self.z_stations[3]),
            "z4_outlet_start": float(self.z_stations[4]),
            "z5_outlet_end": float(self.z_stations[5]),
        }

    def get_blend_arcs(self) -> Dict[str, Dict[str, float]]:
        """
        Return geometric parameters of all 4 blending arcs.
        """
        arcs = {}
        for name, arc in [("Arc1_Inlet_Conv", self._arc1),
                          ("Arc2_Conv_Throat", self._arc2),
                          ("Arc3_Throat_Div", self._arc3),
                          ("Arc4_Div_Outlet", self._arc4)]:
            if arc is not None:
                arcs[name] = {
                    "z_start": arc[0],
                    "z_end": arc[1],
                    "z_center": arc[2],
                    "r_center": arc[3],
                    "radius": arc[4],
                    "sign": arc[5],
                }
            else:
                arcs[name] = {"enabled": 0.0}
        return arcs

    def profile_points(self, n_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate dense coordinate arrays (z, R(z)) for CAD, visualization, and export.
        """
        z_profile = np.linspace(0.0, self.L_total, n_points)
        r_profile = self.radius(z_profile)
        return z_profile, r_profile


# ---------------------------------------------------------------------------
# Smooth Parametric Bézier / B-Spline Geometry for Shape Optimization
# ---------------------------------------------------------------------------
class BezierVenturiGeometry(VenturiGeometry):
    """
    Arbitrary C2-smooth parametric Venturi geometry using quintic Hermite / Bézier
    polynomials with optional interior shape control deformation parameters.

    Ideal for gradient-based or surrogate aerodynamic shape optimization.
    """

    def __init__(
        self,
        config: Optional[VenturiConfig] = None,
        conv_weights: Optional[np.ndarray] = None,
        div_weights: Optional[np.ndarray] = None,
        **kwargs: Any,
    ):
        super().__init__(config=config, **kwargs)
        self.conv_weights = np.array(conv_weights, dtype=float) if conv_weights is not None else np.zeros(3)
        self.div_weights = np.array(div_weights, dtype=float) if div_weights is not None else np.zeros(3)

    @staticmethod
    def _quintic_blend(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Quintic polynomial with zero 1st and 2nd derivatives at endpoints s=0 and s=1:
        H5(s) = 1 - 10*s^3 + 15*s^4 - 6*s^5.
        Returns H5(s), H5'(s), H5''(s).
        """
        s_c = np.clip(s, 0.0, 1.0)
        h = 1.0 - 10.0 * (s_c**3) + 15.0 * (s_c**4) - 6.0 * (s_c**5)
        dh_ds = -30.0 * (s_c**2) + 60.0 * (s_c**3) - 30.0 * (s_c**4)
        d2h_ds2 = -60.0 * s_c + 180.0 * (s_c**2) - 120.0 * (s_c**3)
        return h, dh_ds, d2h_ds2

    @staticmethod
    def _shape_modes(s: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Internal C2-smooth deformation bubble modes with zero value, 1st, and 2nd derivatives at s=0, 1.
        Modes:
          phi_1(s) = 64 * s^3 * (1-s)^3
          phi_2(s) = 128 * s^3 * (1-s)^3 * (2s - 1)
          phi_3(s) = 256 * s^3 * (1-s)^3 * (6s^2 - 6s + 1)
        """
        s_c = np.clip(s, 0.0, 1.0)
        p = (s_c**3) * ((1.0 - s_c)**3)
        dp_ds = 3.0 * (s_c**2) * ((1.0 - s_c)**3) - 3.0 * (s_c**3) * ((1.0 - s_c)**2)
        d2p_ds2 = (
            6.0 * s_c * ((1.0 - s_c)**3)
            - 18.0 * (s_c**2) * ((1.0 - s_c)**2)
            + 6.0 * (s_c**3) * (1.0 - s_c)
        )

        modes = [64.0 * p]
        dmodes = [64.0 * dp_ds]
        d2modes = [64.0 * d2p_ds2]

        if len(weights) > 1:
            q1 = 2.0 * s_c - 1.0
            dq1 = 2.0
            modes.append(128.0 * p * q1)
            dmodes.append(128.0 * (dp_ds * q1 + p * dq1))
            d2modes.append(128.0 * (d2p_ds2 * q1 + 2.0 * dp_ds * dq1))

        if len(weights) > 2:
            q2 = 6.0 * (s_c**2) - 6.0 * s_c + 1.0
            dq2 = 12.0 * s_c - 6.0
            d2q2 = 12.0
            modes.append(256.0 * p * q2)
            dmodes.append(256.0 * (dp_ds * q2 + p * dq2))
            d2modes.append(256.0 * (d2p_ds2 * q2 + 2.0 * dp_ds * dq2 + p * d2q2))

        val = np.zeros_like(s_c)
        dval = np.zeros_like(s_c)
        d2val = np.zeros_like(s_c)

        for w, m, dm, d2m in zip(weights, modes, dmodes, d2modes):
            val += w * m
            dval += w * dm
            d2val += w * d2m

        return val, dval, d2val

    def radius(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        z_arr = np.asarray(z, dtype=float)
        is_scalar = z_arr.ndim == 0
        z_1d = np.atleast_1d(z_arr)
        r = np.zeros_like(z_1d, dtype=float)

        z0, z1, z2, z3, z4, z5 = self.z_stations
        delta_R = self.R_inlet - self.R_throat

        # Inlet pipe
        m_in = z_1d <= z1
        r[m_in] = self.R_inlet

        # Convergent Bézier curve
        m_conv = (z_1d > z1) & (z_1d <= z2)
        if np.any(m_conv):
            s_conv = (z_1d[m_conv] - z1) / (z2 - z1)
            h, _, _ = self._quintic_blend(s_conv)
            def_val, _, _ = self._shape_modes(s_conv, self.conv_weights)
            r[m_conv] = self.R_throat + delta_R * (h + def_val)

        # Throat pipe
        m_th = (z_1d > z2) & (z_1d <= z3)
        r[m_th] = self.R_throat

        # Divergent Bézier curve
        m_div = (z_1d > z3) & (z_1d <= z4)
        if np.any(m_div):
            s_div = (z_1d[m_div] - z3) / (z4 - z3)
            h, _, _ = self._quintic_blend(s_div)
            def_val, _, _ = self._shape_modes(s_div, self.div_weights)
            r[m_div] = self.R_throat + delta_R * ((1.0 - h) + def_val)

        # Outlet pipe
        m_out = z_1d > z4
        r[m_out] = self.R_inlet

        # Lower bound clamping to guarantee strictly positive physical wall radius
        r_min = max(0.001, 0.05 * self.config.D)
        r = np.maximum(r, r_min)

        if is_scalar:
            return float(r[0])
        return r.reshape(z_arr.shape)

    def radius_derivative(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        z_arr = np.asarray(z, dtype=float)
        is_scalar = z_arr.ndim == 0
        z_1d = np.atleast_1d(z_arr)
        dr = np.zeros_like(z_1d, dtype=float)

        z0, z1, z2, z3, z4, z5 = self.z_stations
        delta_R = self.R_inlet - self.R_throat

        m_conv = (z_1d > z1) & (z_1d <= z2)
        if np.any(m_conv):
            dz = z2 - z1
            s_conv = (z_1d[m_conv] - z1) / dz
            _, dh_ds, _ = self._quintic_blend(s_conv)
            _, ddef_ds, _ = self._shape_modes(s_conv, self.conv_weights)
            dr[m_conv] = (delta_R / dz) * (dh_ds + ddef_ds)

        m_div = (z_1d > z3) & (z_1d <= z4)
        if np.any(m_div):
            dz = z4 - z3
            s_div = (z_1d[m_div] - z3) / dz
            _, dh_ds, _ = self._quintic_blend(s_div)
            _, ddef_ds, _ = self._shape_modes(s_div, self.div_weights)
            dr[m_div] = (delta_R / dz) * (-dh_ds + ddef_ds)

        if is_scalar:
            return float(dr[0])
        return dr.reshape(z_arr.shape)

    def radius_second_derivative(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        z_arr = np.asarray(z, dtype=float)
        is_scalar = z_arr.ndim == 0
        z_1d = np.atleast_1d(z_arr)
        d2r = np.zeros_like(z_1d, dtype=float)

        z0, z1, z2, z3, z4, z5 = self.z_stations
        delta_R = self.R_inlet - self.R_throat

        m_conv = (z_1d > z1) & (z_1d <= z2)
        if np.any(m_conv):
            dz = z2 - z1
            s_conv = (z_1d[m_conv] - z1) / dz
            _, _, d2h_ds2 = self._quintic_blend(s_conv)
            _, _, d2def_ds2 = self._shape_modes(s_conv, self.conv_weights)
            d2r[m_conv] = (delta_R / (dz**2)) * (d2h_ds2 + d2def_ds2)

        m_div = (z_1d > z3) & (z_1d <= z4)
        if np.any(m_div):
            dz = z4 - z3
            s_div = (z_1d[m_div] - z3) / dz
            _, _, d2h_ds2 = self._quintic_blend(s_div)
            _, _, d2def_ds2 = self._shape_modes(s_div, self.div_weights)
            d2r[m_div] = (delta_R / (dz**2)) * (-d2h_ds2 + d2def_ds2)

        if is_scalar:
            return float(d2r[0])
        return d2r.reshape(z_arr.shape)


# ---------------------------------------------------------------------------
# Module-Level Compatibility Functions
# ---------------------------------------------------------------------------
def create_venturi_geometry(config: VenturiConfig) -> VenturiGeometry:
    """
    Factory function creating a VenturiGeometry instance from a VenturiConfig.
    """
    return VenturiGeometry(config=config)


def radius_at(z: Union[float, np.ndarray], geom: VenturiGeometry) -> Union[float, np.ndarray]:
    """
    Return the Venturi radius R(z) at axial coordinate(s) z.
    """
    return geom.radius(z)


def dradius_dz(z: Union[float, np.ndarray], geom: VenturiGeometry) -> Union[float, np.ndarray]:
    """
    Return the first spatial derivative dR/dz at axial coordinate(s) z.
    """
    return geom.radius_derivative(z)


def d2radius_dz2(z: Union[float, np.ndarray], geom: VenturiGeometry) -> Union[float, np.ndarray]:
    """
    Return the second spatial derivative d^2R/dz^2 at axial coordinate(s) z.
    """
    return geom.radius_second_derivative(z)


def profile_points(geom: VenturiGeometry, n_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dense arrays of coordinates (z, R) for plotting or CAD export.
    """
    return geom.profile_points(n_points=n_points)
