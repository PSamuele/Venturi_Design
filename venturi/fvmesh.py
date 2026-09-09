"""
fvmesh.py
=========
Axisymmetric body-fitted finite-volume grid for the Venturi tube.

Why finite volumes rather than finite differences: the unknowns live at cell
CENTRES and the conserved quantities are the FLUXES through the FACES. Every
interior face is shared by two cells and enters their two balances with
opposite signs, so summing over all cells cancels the interior contributions
and leaves only the inlet/outlet balance. That is where exact mass
conservation comes from.

Coordinate map
--------------
    z   = physical axial coordinate  [m]
    eta = r / R_wall(z)   in [0, 1]  (dimensionless)

Lines of constant eta are CONICAL surfaces, not cylindrical ones, because R
varies with z. Their normals and areas are computed accordingly.

Index conventions
-----------------
    i = 0 .. Nz-1   cells in the axial direction
    j = 0 .. Nr-1   cells in the radial direction
    j = 0           cell adjacent to the axis
    j = Nr-1        cell adjacent to the wall
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Four-point Gauss-Legendre quadrature on [-1, 1].
# Used to integrate R(z)^2 exactly over the cell volumes: taking the value at
# the cell centre instead would introduce an O(h^2) volume error, which would
# then show up in the mass balance.
_GL_X = np.array([-0.8611363115940526, -0.3399810435848563,
                  0.3399810435848563, 0.8611363115940526])
_GL_W = np.array([0.3478548451374538, 0.6521451548625461,
                  0.6521451548625461, 0.3478548451374538])


@dataclass
class FVMesh:
    """Axisymmetric finite-volume grid.

    Geometric attributes
    --------------------
    Nz, Nr : number of CELLS (not nodes)
    z_f    : (Nz+1,)      axial face positions
    z_c    : (Nz,)        axial cell-centre positions
    eta_f  : (Nr+1,)      radial (dimensionless) face positions
    eta_c  : (Nr,)        eta of the cell centre
    R_f    : (Nz+1,)      R_wall(z_f)
    R_c    : (Nz,)        R_wall(z_c)
    dRdz_c : (Nz,)        R'_wall(z_c)
    r_c    : (Nz, Nr)     cell-centre radius = eta_c * R_c
    z_cc   : (Nz, Nr)     cell-centre axial coordinate, broadcast
    vol    : (Nz, Nr)     annular cell volume [m^3]

    Axial faces (normal along +z), index i = 0..Nz
    ----------------------------------------------
    A_ax   : (Nz+1, Nr)   annulus area [m^2]

    Conical faces (normal towards +eta), index j = 0..Nr
    ----------------------------------------------------
    A_rad  : (Nz, Nr+1)   conical surface area [m^2]
    nz_rad : (Nz, Nr+1)   z component of the outward normal
    nr_rad : (Nz, Nr+1)   r component of the outward normal
    Cr_rad : (Nz, Nr+1)   flux coefficient multiplying ur (see below)
    Cz_rad : (Nz, Nr+1)   flux coefficient multiplying uz (see below)
    """

    Nz: int
    Nr: int
    z_f: np.ndarray
    z_c: np.ndarray
    eta_f: np.ndarray
    eta_c: np.ndarray
    R_f: np.ndarray
    R_c: np.ndarray
    dRdz_c: np.ndarray
    r_c: np.ndarray
    z_cc: np.ndarray
    vol: np.ndarray
    A_ax: np.ndarray
    A_rad: np.ndarray
    nz_rad: np.ndarray
    nr_rad: np.ndarray
    Cr_rad: np.ndarray
    Cz_rad: np.ndarray
    wall_distance: np.ndarray

    @property
    def n_cells(self) -> int:
        return self.Nz * self.Nr


def _radial_face_distribution(Nr: int, gamma: float) -> np.ndarray:
    """Face distribution in eta, clustered towards the wall.

    Uses the tanh map s -> tanh(gamma*s)/tanh(gamma). Its derivative equals
    gamma/tanh(gamma) > 1 at s=0 and decreases monotonically, so spacing is
    wide near the axis and tight near the wall, where the boundary layer
    needs to be resolved. gamma <= 0 disables clustering.
    """
    s = np.linspace(0.0, 1.0, Nr + 1)
    if gamma <= 0.0:
        eta_f = s.copy()
    else:
        eta_f = np.tanh(gamma * s) / np.tanh(gamma)
    eta_f[0] = 0.0
    eta_f[-1] = 1.0
    return eta_f


def build_fvmesh(geom, Nz: int, Nr: int, wall_clustering: float = 2.0,
                 centroid: str = "midpoint") -> FVMesh:
    """Build the finite-volume grid from the wall geometry.

    Args:
        geom: object exposing .L_total, .radius(z), .radius_derivative(z).
        Nz: number of axial cells.
        Nr: number of radial cells.
        wall_clustering: gamma of the wall-clustering map.
        centroid: "midpoint" (default) or "volume". With a uniform radial
            distribution, "midpoint" places every face exactly halfway
            between the two neighbouring cell centres, which is the condition
            for the two-point flux approximation to be second-order accurate.
    """
    if Nz < 2 or Nr < 2:
        raise ValueError(f"At least 2 cells per direction are needed (Nz={Nz}, Nr={Nr}).")

    L = float(geom.L_total)

    # --- axial grid: uniform faces, centres at mid-cell ---------------------
    z_f = np.linspace(0.0, L, Nz + 1)
    z_c = 0.5 * (z_f[:-1] + z_f[1:])
    dz = z_f[1:] - z_f[:-1]                       # (Nz,)

    # --- radial grid in eta -------------------------------------------------
    eta_f = _radial_face_distribution(Nr, wall_clustering)

    e1, e2 = eta_f[:-1], eta_f[1:]
    if centroid == "volume":
        # Volume centroid of the annulus in eta.
        eta_c = (2.0 / 3.0) * (e2 ** 3 - e1 ** 3) / (e2 ** 2 - e1 ** 2)
    else:
        eta_c = 0.5 * (e1 + e2)

    # --- wall profile -------------------------------------------------------
    R_f = np.asarray(geom.radius(z_f), dtype=np.float64)
    R_c = np.asarray(geom.radius(z_c), dtype=np.float64)
    dRdz_c = np.asarray(geom.radius_derivative(z_c), dtype=np.float64)

    # --- cell volumes -------------------------------------------------------
    # V = integral over z of [ pi * (eta2^2 - eta1^2) * R(z)^2 ] dz.
    # The z integral uses 4-point Gauss-Legendre, exact for R(z)^2 up to
    # degree 7, so the conical runs and the ISO blend arcs are captured with
    # no appreciable quadrature error.
    zq = 0.5 * (z_f[:-1, None] + z_f[1:, None]) + 0.5 * dz[:, None] * _GL_X[None, :]
    wq = 0.5 * dz[:, None] * _GL_W[None, :]                      # (Nz, 4)
    Rq = np.asarray(geom.radius(zq.ravel()), dtype=np.float64).reshape(zq.shape)
    dRq = np.asarray(geom.radius_derivative(zq.ravel()), dtype=np.float64).reshape(zq.shape)

    int_R2_dz = np.sum(wq * Rq ** 2, axis=1)                     # (Nz,)
    ring = np.pi * (e2 ** 2 - e1 ** 2)                           # (Nr,)
    vol = int_R2_dz[:, None] * ring[None, :]                     # (Nz, Nr)

    # --- centroid coordinates -----------------------------------------------
    r_c = eta_c[None, :] * R_c[:, None]                          # (Nz, Nr)
    z_cc = np.repeat(z_c[:, None], Nr, axis=1)                   # (Nz, Nr)

    # --- axial face areas (z = const, circular annulus) ---------------------
    # ring already carries the factor pi: A = pi*(eta2^2-eta1^2)*R^2 = ring*R^2
    A_ax = (R_f[:, None] ** 2) * ring[None, :]                   # (Nz+1, Nr)

    # --- conical face areas and normals (eta = const) -----------------------
    # The surface eta = eta_j has equation r = eta_j * R(z).
    # Outward normal (towards increasing eta), unnormalised: (-eta_j R', 1).
    # Area element:  dA = 2*pi*r * sqrt(1 + (eta_j R')^2) dz
    slope_q = eta_f[None, :, None] * dRq[:, None, :]             # (Nz, Nr+1, 4)
    integrand = (2.0 * np.pi * eta_f[None, :, None] * Rq[:, None, :]
                 * np.sqrt(1.0 + slope_q ** 2))
    A_rad = np.sum(wq[:, None, :] * integrand, axis=2)           # (Nz, Nr+1)

    slope_c = eta_f[None, :] * dRdz_c[:, None]                   # (Nz, Nr+1)
    norm = np.sqrt(1.0 + slope_c ** 2)
    nz_rad = -slope_c / norm
    nr_rad = 1.0 / norm

    # --- conical-face flux coefficients -------------------------------------
    # The volumetric flux through r = eta_j * R(z) is exactly
    #     F = 2*pi*eta_j   * integral( R * ur ) dz
    #       - 2*pi*eta_j^2 * integral( R * R' * uz ) dz
    # Factoring uz and ur out (evaluated at the face centre) leaves two purely
    # geometric coefficients:
    #     Cr = 2*pi*eta_j   * integral(R dz)
    #     Cz = 2*pi*eta_j^2 * integral(R R' dz)
    # This split is not cosmetic: it enforces the geometric conservation law
    # (GCL). With a uniform field uz = const, ur = 0 the discrete divergence
    # comes out exactly zero, because
    #     integral(R R' dz) = 0.5 * (R_right^2 - R_left^2)
    # is the very quantity appearing in the difference of the axial face
    # areas. Using A_rad * (u . n) evaluated at the cell centre instead, a
    # uniform flow would generate a spurious O(h^2) divergence and hence a
    # parasitic pressure field right in the converging and diverging cones.
    int_R_dz = np.sum(wq * Rq, axis=1)                           # (Nz,)
    int_RRp_dz = 0.5 * (R_f[1:] ** 2 - R_f[:-1] ** 2)            # (Nz,) exact
    Cr_rad = 2.0 * np.pi * eta_f[None, :] * int_R_dz[:, None]
    Cz_rad = 2.0 * np.pi * (eta_f[None, :] ** 2) * int_RRp_dz[:, None]

    # --- wall distance (for the turbulence models) --------------------------
    slope_wall = dRdz_c[:, None]
    wall_distance = (R_c[:, None] - r_c) / np.sqrt(1.0 + slope_wall ** 2)
    wall_distance = np.maximum(wall_distance, 1e-12)

    return FVMesh(
        Nz=Nz, Nr=Nr,
        z_f=z_f, z_c=z_c, eta_f=eta_f, eta_c=eta_c,
        R_f=R_f, R_c=R_c, dRdz_c=dRdz_c,
        r_c=r_c, z_cc=z_cc, vol=vol,
        A_ax=A_ax, A_rad=A_rad, nz_rad=nz_rad, nr_rad=nr_rad,
        Cr_rad=Cr_rad, Cz_rad=Cz_rad,
        wall_distance=wall_distance,
    )


# ---------------------------------------------------------------------------
# Discrete divergence operator
# ---------------------------------------------------------------------------
def divergence(F_ax: np.ndarray, F_rad: np.ndarray, mesh: FVMesh) -> np.ndarray:
    """Discrete divergence: sum of the fluxes LEAVING each cell [m^3/s].

    Args:
        F_ax:  (Nz+1, Nr) volumetric flux through the axial faces, positive
               towards +z.
        F_rad: (Nz, Nr+1) volumetric flux through the conical faces, positive
               towards +eta (towards the wall).

    Returns:
        (Nz, Nr) mass imbalance per cell [m^3/s]. If it is zero everywhere,
        the flux field is divergence-free.
    """
    return (F_ax[1:, :] - F_ax[:-1, :]) + (F_rad[:, 1:] - F_rad[:, :-1])
