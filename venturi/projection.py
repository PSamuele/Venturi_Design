"""
projection.py
=============
Projection onto the divergence-free subspace for the finite-volume grid.

Principle
---------
A projection is exact if and only if the matrix used to solve for pressure is
EXACTLY the product D @ G of the discrete divergence and gradient operators.
Here that is not an assumption to be checked: the matrix is ASSEMBLED from
the very same face coefficients `c_f` that later correct the fluxes. There
are not two discretisations to keep in step - there is only one.

For each interior face between cell P and cell N:

    flux correction leaving P :  dF = c_f * (phi_P - phi_N)
    matrix contribution       :  L[P,P] += c_f ,  L[P,N] -= c_f
                                 L[N,N] += c_f ,  L[N,P] -= c_f

The face coefficient is

    c_f = A_f / (d_PN . n_f)

with A_f the face area, n_f its unit normal and d_PN the vector joining the
two centroids. Dividing by the PROJECTION of d_PN onto the normal (rather
than by its length) is the "over-relaxed" non-orthogonality correction: on
the Venturi's conical faces the centre-to-centre line and the normal make an
angle of up to ~10 degrees, and ignoring it would bias the pressure gradient.

Boundary conditions for the correction potential phi
----------------------------------------------------
    inlet  : flow rate imposed  -> flux must not be corrected -> no coeff.
    wall   : zero flux          -> flux must not be corrected -> no coeff.
    axis   : zero area          -> no coeff. (automatic)
    outlet : pressure imposed   -> phi = 0 on the face -> diagonal only

The outlet face is the only correctable boundary: that is where the outgoing
flow rate automatically rebalances against the incoming one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .fvmesh import FVMesh, divergence


@dataclass
class FaceCoefficients:
    """Coefficients c_f = A_f / (d_PN . n_f) for every correctable face."""
    c_ax: np.ndarray    # (Nz+1, Nr) - valid for i = 1..Nz (0 at the inlet)
    c_rad: np.ndarray   # (Nz, Nr+1) - valid for j = 1..Nr-1 (0 at axis/wall)


def compute_face_coefficients(mesh: FVMesh) -> FaceCoefficients:
    """Compute the geometric face coefficients."""
    Nz, Nr = mesh.Nz, mesh.Nr

    # --- interior axial faces: i = 1..Nz-1, between cells (i-1,j) and (i,j) --
    # n = (1, 0);  d_PN = (z_c[i] - z_c[i-1],  r_c[i,j] - r_c[i-1,j])
    # d_PN . n = z_c[i] - z_c[i-1]
    c_ax = np.zeros((Nz + 1, Nr), dtype=np.float64)
    dz_cc = mesh.z_c[1:] - mesh.z_c[:-1]                      # (Nz-1,)
    c_ax[1:Nz, :] = mesh.A_ax[1:Nz, :] / dz_cc[:, None]

    # --- outlet axial face i = Nz: cell (Nz-1,j) towards the boundary --------
    d_out = mesh.z_f[Nz] - mesh.z_c[Nz - 1]
    c_ax[Nz, :] = mesh.A_ax[Nz, :] / d_out

    # --- interior conical faces: j = 1..Nr-1, between (i,j-1) and (i,j) -----
    # The two centroids share the same z, so d_PN = (0, dr) and
    # d_PN . n = dr * nr_rad  (nr_rad < 1: this is where non-orthogonality
    # enters).
    c_rad = np.zeros((Nz, Nr + 1), dtype=np.float64)
    dr_cc = mesh.r_c[:, 1:] - mesh.r_c[:, :-1]                # (Nz, Nr-1)
    proj = dr_cc * mesh.nr_rad[:, 1:Nr]
    c_rad[:, 1:Nr] = mesh.A_rad[:, 1:Nr] / proj

    return FaceCoefficients(c_ax=c_ax, c_rad=c_rad)


class Projector:
    """Divergence-free projector with a reusable LU factorisation.

    The operator does NOT depend on the time step: it solves for a potential
    phi, and the physical pressure follows as p' = rho/dt * phi. The LU
    factorisation is therefore computed once and reused for every iteration.
    """

    def __init__(self, mesh: FVMesh):
        self.mesh = mesh
        self.coef = compute_face_coefficients(mesh)
        self.L = self._assemble()
        self.lu = spla.splu(self.L.tocsc())

    # -- assembly ------------------------------------------------------------
    def _assemble(self) -> sp.csr_matrix:
        mesh = self.mesh
        Nz, Nr = mesh.Nz, mesh.Nr
        c_ax, c_rad = self.coef.c_ax, self.coef.c_rad

        def idx(i, j):
            return i * Nr + j

        rows, cols, vals = [], [], []

        # interior axial faces
        I, J = np.meshgrid(np.arange(1, Nz), np.arange(Nr), indexing="ij")
        P = idx(I - 1, J).ravel()
        N = idx(I, J).ravel()
        c = c_ax[1:Nz, :].ravel()
        rows.extend([P, P, N, N])
        cols.extend([P, N, N, P])
        vals.extend([c, -c, c, -c])

        # interior conical faces
        I, J = np.meshgrid(np.arange(Nz), np.arange(1, Nr), indexing="ij")
        P = idx(I, J - 1).ravel()
        N = idx(I, J).ravel()
        c = c_rad[:, 1:Nr].ravel()
        rows.extend([P, P, N, N])
        cols.extend([P, N, N, P])
        vals.extend([c, -c, c, -c])

        # outlet face: Dirichlet phi = 0, diagonal only
        P = idx(Nz - 1, np.arange(Nr))
        c = c_ax[Nz, :]
        rows.append(P)
        cols.append(P)
        vals.append(c)

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)

        n = Nz * Nr
        return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    # -- projection ----------------------------------------------------------
    def project(self, F_ax: np.ndarray, F_rad: np.ndarray):
        """Make the flux field divergence-free.

        Args:
            F_ax:  (Nz+1, Nr) predicted axial fluxes. Row 0 (inlet) is imposed
                   and left untouched.
            F_rad: (Nz, Nr+1) predicted conical fluxes. Columns 0 (axis) and
                   Nr (wall) are imposed to zero and left untouched.

        Returns:
            (F_ax_corrected, F_rad_corrected, phi) with phi of shape (Nz, Nr).
        """
        mesh = self.mesh
        Nz, Nr = mesh.Nz, mesh.Nr
        c_ax, c_rad = self.coef.c_ax, self.coef.c_rad

        div = divergence(F_ax, F_rad, mesh)
        phi = self.lu.solve(-div.ravel()).reshape(Nz, Nr)

        F_ax = F_ax.copy()
        F_rad = F_rad.copy()

        # dF on face i (leaving P = cell i-1) = c * (phi_P - phi_N)
        F_ax[1:Nz, :] += c_ax[1:Nz, :] * (phi[:-1, :] - phi[1:, :])
        # outlet: phi = 0 on the boundary
        F_ax[Nz, :] += c_ax[Nz, :] * phi[Nz - 1, :]
        # interior conical faces
        F_rad[:, 1:Nr] += c_rad[:, 1:Nr] * (phi[:, :-1] - phi[:, 1:])

        return F_ax, F_rad, phi
