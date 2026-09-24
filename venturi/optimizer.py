"""
optimizer.py
============
Quick choice of beta and of the diverging angle for a wanted flow rate and
pressure drop, BEFORE running the CFD.

It uses only simple formulas, not the solver:

    D from Bernoulli:  dp = 8 rho Q^2 / (pi^2 D^4) * (1/beta^4 - 1)
    lasting loss:      K = 3.2 * tan(alpha_div/2)^1.25 * (1 - beta^2)^2
                       dp_loss = K * rho * v_throat^2 / 2
                       (empirical estimate for a conical diffuser)
    length:            sum of the five pieces of the profile

The converging angle is NOT optimised: it has no effect on the loss formula
above, so an optimiser would just push it to its bound to shorten the tube.
It is an input instead (default 21 deg).
"""

from __future__ import annotations

import math

from scipy.optimize import minimize

BOUNDS_BETA = (0.30, 0.75)
BOUNDS_ALPHA_DIV = (5.0, 15.0)   # [deg] wider diffusers tend to separate


def calc_D_from_beta(beta: float, Q: float, dp: float, rho: float) -> float:
    """Pipe diameter [m] that gives pressure drop dp at flow rate Q."""
    return (8.0 * rho * Q**2 / (math.pi**2 * dp) * (1.0 / beta**4 - 1.0)) ** 0.25


def estimate_pressure_loss(beta: float, alpha_div_deg: float, v_throat: float, rho: float) -> float:
    """Pressure lost for good in the diverging cone [Pa] (empirical)."""
    K = 3.2 * math.tan(math.radians(alpha_div_deg) / 2.0) ** 1.25 * (1.0 - beta**2) ** 2
    return K * 0.5 * rho * v_throat**2


def estimate_length(D: float, beta: float, alpha_conv_deg: float, alpha_div_deg: float) -> float:
    """Total length [m] with the default straight-run factors (1.5 D, d, 3 D)."""
    dR = 0.5 * D * (1.0 - beta)
    return (1.5 * D + beta * D + 3.0 * D
            + dR / math.tan(math.radians(alpha_conv_deg) / 2.0)
            + dR / math.tan(math.radians(alpha_div_deg) / 2.0))


def run_optimization(Q: float, p_inlet: float, p_throat: float, rho: float,
                     objective_type: int, alpha_conv_deg: float = 21.0) -> dict:
    """Pick beta and alpha_div.

    objective_type: 1 = smallest lasting loss, 2 = shortest tube,
                    3 = half and half.
    """
    if Q <= 0.0:
        raise ValueError(f"The flow rate must be positive, got {Q}.")
    dp = p_inlet - p_throat
    if dp <= 0.0:
        raise ValueError(f"The throat pressure ({p_throat:.0f} Pa) must be lower "
                         f"than the inlet pressure ({p_inlet:.0f} Pa).")
    if objective_type not in (1, 2, 3):
        raise ValueError(f"objective_type must be 1, 2 or 3, got {objective_type}.")

    def objective(x):
        beta, a_div = x
        D = calc_D_from_beta(beta, Q, dp, rho)
        v_th = Q / (math.pi * (0.5 * beta * D) ** 2)
        loss = estimate_pressure_loss(beta, a_div, v_th, rho) / (0.1 * dp)
        length = estimate_length(D, beta, alpha_conv_deg, a_div) / 2.0
        penalty = 1000.0 * (D - 5.0) if D > 5.0 else 0.0
        if objective_type == 1:
            return loss + penalty
        if objective_type == 2:
            return length + penalty
        return 0.5 * loss + 0.5 * length + penalty

    res = minimize(objective, [0.5, 8.0], bounds=[BOUNDS_BETA, BOUNDS_ALPHA_DIV],
                   method="SLSQP")
    beta, a_div = (float(v) for v in res.x)
    return {
        "D_inlet": calc_D_from_beta(beta, Q, dp, rho),
        "beta": beta,
        "alpha_conv_deg": float(alpha_conv_deg),
        "alpha_div_deg": a_div,
        "success": bool(res.success),
        "message": str(res.message),
    }
