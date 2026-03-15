from __future__ import annotations

import numpy as np

from chromlearn.io.trajectory import TrimmedCell


def _pairwise_feature_sum(
    source: np.ndarray,
    neighbors: np.ndarray,
    basis,
) -> np.ndarray:
    n_basis = basis.n_basis
    feature = np.zeros((3, n_basis), dtype=np.float64)
    for neighbor in neighbors:
        if np.any(np.isnan(neighbor)):
            continue
        delta = neighbor - source
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-12:
            continue
        direction = delta / distance
        phi = basis.evaluate(np.array([distance]))[0]
        feature += direction[:, np.newaxis] * phi[np.newaxis, :]
    return feature


def build_design_matrix(
    cells: list[TrimmedCell],
    basis_xx,
    basis_xy,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the stacked regression design matrix and velocity response."""

    rows: list[np.ndarray] = []
    responses: list[np.ndarray] = []

    for cell in cells:
        chromosomes = cell.chromosomes
        centrioles = cell.centrioles
        n_timepoints = chromosomes.shape[0]
        n_chromosomes = chromosomes.shape[2]

        for time_index in range(n_timepoints - 1):
            positions = chromosomes[time_index].T
            next_positions = chromosomes[time_index + 1].T
            pole_positions = centrioles[time_index].T

            for chrom_index in range(n_chromosomes):
                position = positions[chrom_index]
                next_position = next_positions[chrom_index]
                if np.any(np.isnan(position)) or np.any(np.isnan(next_position)):
                    continue

                velocity = (next_position - position) / cell.dt
                other_positions = np.delete(positions, chrom_index, axis=0)
                g_xx = _pairwise_feature_sum(position, other_positions, basis_xx)
                g_xy = _pairwise_feature_sum(position, pole_positions, basis_xy)
                block = np.hstack([g_xx, g_xy])
                rows.append(block)
                responses.append(velocity)

    n_cols = basis_xx.n_basis + basis_xy.n_basis
    if not rows:
        return np.zeros((0, n_cols), dtype=np.float64), np.zeros(0, dtype=np.float64)

    return np.vstack(rows), np.concatenate(responses)
