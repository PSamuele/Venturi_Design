import math
import numpy as np
from scipy.optimize import minimize

def calc_D_from_beta(beta: float, Q: float, dp: float, rho: float) -> float:
    """Recover the inlet diameter (m) from inverse Bernoulli."""
    # dp = (8 * rho * Q^2 / (pi^2 * D^4)) * (1/beta^4 - 1)
    # D^4 = (8 * rho * Q^2 / (pi^2 * dp)) * (1/beta^4 - 1)
    term1 = (8.0 * rho * Q**2) / ((math.pi**2) * dp)
    term2 = (1.0 / (beta**4)) - 1.0
    return (term1 * term2)**0.25

def estimate_pressure_loss(beta: float, alpha_div_deg: float, v_throat: float, rho: float) -> float:
    """
    Estimate the permanent pressure loss (Pa).
    Based on empirical loss coefficients for the diffuser and the throat.
    """
    alpha_div_rad = math.radians(alpha_div_deg)
    # Diffuser loss coefficient (approximate)
    K_div = 3.2 * (math.tan(alpha_div_rad/2))**1.25 * (1 - beta**2)**2
    # Estimated total loss = K * kinetic energy at the throat
    dp_loss = K_div * 0.5 * rho * v_throat**2
    return dp_loss

def estimate_length(D_inlet: float, beta: float, alpha_conv_deg: float, alpha_div_deg: float) -> float:
    """Estimate the total length (cylindrical runs approximated)."""
    d_th = beta * D_inlet
    R_in = D_inlet / 2
    R_th = d_th / 2
    L_in = 1.5 * D_inlet
    L_out = 3.0 * D_inlet
    L_th = 1.0 * d_th
    L_co = (R_in - R_th) / math.tan(math.radians(alpha_conv_deg)/2)
    L_di = (R_in - R_th) / math.tan(math.radians(alpha_div_deg)/2)
    return L_in + L_co + L_th + L_di + L_out

def run_optimization(Q: float, p_inlet: float, p_throat: float, rho: float, objective_type: int) -> dict:
    """
    Optimise the Venturi geometry.
    objective_type: 1 = minimise loss, 2 = minimise length, 3 = balanced
    """
    dp_target = p_inlet - p_throat
    
    # x = [beta, alpha_conv, alpha_div]
    # Standard Venturi bounds (to avoid separation and unphysical shapes)
    bounds = [
        (0.30, 0.75),   # beta
        (15.0, 25.0),   # alpha_conv
        (5.0, 15.0)     # alpha_div
    ]
    
    # Initial guess (centre of the domain)
    x0 = [0.5, 21.0, 8.0]
    
    def objective_func(x):
        beta, a_co, a_di = x
        D = calc_D_from_beta(beta, Q, dp_target, rho)
        d_th = beta * D
        
        # Velocities
        v_th = Q / (math.pi * (d_th/2)**2)
        
        # Physical metrics
        L = estimate_length(D, beta, a_co, a_di)
        loss = estimate_pressure_loss(beta, a_di, v_th, rho)
        
        # Penalty if D grows unrealistically large
        penalty = 0.0
        if D > 5.0: penalty += (D - 5.0) * 1000
        
        # Normalisations for the optimiser
        # Typical length ~ 1-5 m; typical loss ~ hundreds to thousands of Pa
        L_norm = L / 2.0
        loss_norm = loss / (dp_target * 0.1) # Normalizzato sul 10% del dp
        
        if objective_type == 1:
            return loss_norm + penalty
        elif objective_type == 2:
            return L_norm + penalty
        else: # 3 = Bilanciato
            return 0.5 * loss_norm + 0.5 * L_norm + penalty

    res = minimize(objective_func, x0, bounds=bounds, method='SLSQP')
    
    beta_opt, a_co_opt, a_di_opt = res.x
    D_opt = calc_D_from_beta(beta_opt, Q, dp_target, rho)
    
    return {
        "D_inlet": float(D_opt),
        "beta": float(beta_opt),
        "alpha_conv_deg": float(a_co_opt),
        "alpha_div_deg": float(a_di_opt),
        "success": res.success,
        "message": res.message
    }
