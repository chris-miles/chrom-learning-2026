"""PCA-based trajectory projection for 3D -> 2D visualization.

Computes a PCA basis from a cell's combined pole + chromosome trajectories,
then projects arbitrary 3D trajectory data into that basis.  Useful for
comparing real vs simulated trajectories in a common coordinate system.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chromlearn.io.trajectory import TrimmedCell


@dataclass
class PCABasis:
    """A 2-component PCA basis fitted from 3D trajectory data."""

    origin: np.ndarray       # (3,) — centroid subtracted before projection
    components: np.ndarray   # (3, 2) — columns are PC1, PC2

    def project(self, points: np.ndarray) -> np.ndarray:
        """Project (…, 3) points into the PCA plane.  Returns (…, 2)."""
        return (points - self.origin) @ self.components


def fit_pca_basis(cell: TrimmedCell, sign_ref: str = "pole_axis") -> PCABasis:
    """Fit a 2-component PCA basis from a cell's poles + chromosomes.

    All valid (non-NaN) timepoint/particle positions are concatenated into a
    single point cloud, centered, and decomposed via SVD.

    Args:
        cell: A TrimmedCell with ``.centrioles`` (T, 3, 2) and
            ``.chromosomes`` (T, 3, N).
        sign_ref: Sign convention for the PCs.  ``"pole_axis"`` aligns PC1
            so that pole 1 → pole 2 has a positive PC1 component at t=0.

    Returns:
        A :class:`PCABasis` that can project any 3D data into the same plane.
    """
    poles = cell.centrioles                   # (T, 3, 2)
    chroms = cell.chromosomes                 # (T, 3, N)

    # Collect all valid 3D positions into (M, 3)
    parts = []
    for p in range(poles.shape[2]):
        pts = poles[:, :, p]                  # (T, 3)
        valid = np.all(np.isfinite(pts), axis=1)
        parts.append(pts[valid])
    for j in range(chroms.shape[2]):
        pts = chroms[:, :, j]
        valid = np.all(np.isfinite(pts), axis=1)
        parts.append(pts[valid])

    all_pts = np.concatenate(parts, axis=0)   # (M, 3)
    origin = all_pts.mean(axis=0)
    centered = all_pts - origin

    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:2].T                     # (3, 2)

    # Stable sign convention: align PC1 with pole1->pole2 at t=0
    if sign_ref == "pole_axis":
        pole_vec = poles[0, :, 1] - poles[0, :, 0]
        if np.isfinite(pole_vec).all():
            if pole_vec @ components[:, 0] < 0:
                components[:, 0] *= -1
            # Align PC2 so it forms a right-handed system with PC1 in 3D
            cross = np.cross(components[:, 0], components[:, 1])
            if cross @ Vt[2] < 0:
                components[:, 1] *= -1

    return PCABasis(origin=origin, components=components)
