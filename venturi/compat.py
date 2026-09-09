"""
compat.py
=========
Adapter between the finite-volume grid and the export module.

export.py was written for the old node-based grid and accesses `mesh.z`,
`mesh.r`, `mesh.Nz`, `mesh.Nr` as (Nz, Nr) matrices. The finite-volume grid
exposes the same information under different names, because values live at
cell centroids rather than at vertices.

Rather than duplicating export.py, a thin view is provided here. The exported
fields (velocity, pressure) are cell averages: for contour plots, point
clouds and ParaView files that is the correct representation, and it is
exactly what finite-volume solvers write out.
"""

from __future__ import annotations

import numpy as np

from .fvmesh import FVMesh


class MeshView:
    """View compatible with the old node-based mesh interface."""

    def __init__(self, fv: FVMesh):
        self._fv = fv
        self.z = fv.z_cc                    # (Nz, Nr)
        self.r = fv.r_c                     # (Nz, Nr)
        self.Nz = fv.Nz
        self.Nr = fv.Nr
        self.R_wall = fv.R_c                # (Nz,)
        self.dR_dz = fv.dRdz_c
        self.wall_distance = fv.wall_distance
        # Nominal spacings, used only by textual diagnostics
        self.dz = np.diff(fv.z_f)
        self.dr = np.gradient(fv.r_c, axis=1)

    def __getattr__(self, item):
        # Anything not redefined here is looked up on the FV grid, so the
        # adapter never hides information.
        return getattr(self._fv, item)


def mesh_view(fv: FVMesh) -> MeshView:
    return MeshView(fv)
