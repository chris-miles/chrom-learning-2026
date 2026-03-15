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
    basis_eval_mode: str = "ito",
) -> tuple[np.ndarray, np.ndarray]:
    """Build the stacked regression design matrix and velocity response.

    Parameters
    ----------
    cells : list[TrimmedCell]
        Trimmed trajectory data for each cell.
    basis_xx, basis_xy : Basis
        Basis function objects for chromosome-chromosome and
        chromosome-centrosome interactions.
    basis_eval_mode : str, default ``"ito"``
        Controls which positions are used to evaluate pairwise distances
        and basis functions.  The velocity is always ``(X(t+1) - X(t)) / dt``.

        * ``"ito"`` -- Evaluate basis at current positions ``X(t)``.
          Loop starts at ``time_index = 0``.
        * ``"ito_shift"`` -- Evaluate basis at *previous* positions
          ``X(t-1)``.  Loop starts at ``time_index = 1`` so that the
          previous frame is always available.  Useful for avoiding
          same-timestep correlation between noise in the velocity and
          noise in the basis evaluation positions.
        * ``"strato"`` -- Evaluate basis at the midpoint
          ``(X(t) + X(t+1)) / 2`` (Stratonovich-style discretisation).
          Loop starts at ``time_index = 0``.
    """

    if basis_eval_mode not in ("ito", "ito_shift", "strato"):
        raise ValueError(
            f"Unknown basis_eval_mode={basis_eval_mode!r}. "
            "Expected 'ito', 'ito_shift', or 'strato'."
        )

    rows: list[np.ndarray] = []
    responses: list[np.ndarray] = []

    for cell in cells:
        chromosomes = cell.chromosomes
        centrioles = cell.centrioles
        n_timepoints = chromosomes.shape[0]
        n_chromosomes = chromosomes.shape[2]

        t_start = 1 if basis_eval_mode == "ito_shift" else 0

        for time_index in range(t_start, n_timepoints - 1):
            # --- velocity positions (always current and next) ---
            positions = chromosomes[time_index].T
            next_positions = chromosomes[time_index + 1].T

            # --- evaluation positions for basis/distances ---
            if basis_eval_mode == "ito":
                eval_chroms = chromosomes[time_index].T
                eval_poles = centrioles[time_index].T
            elif basis_eval_mode == "ito_shift":
                eval_chroms = chromosomes[time_index - 1].T
                eval_poles = centrioles[time_index - 1].T
            else:  # strato
                eval_chroms = (0.5 * (chromosomes[time_index] + chromosomes[time_index + 1])).T
                eval_poles = (0.5 * (centrioles[time_index] + centrioles[time_index + 1])).T

            for chrom_index in range(n_chromosomes):
                position = positions[chrom_index]
                next_position = next_positions[chrom_index]
                eval_position = eval_chroms[chrom_index]
                if (
                    np.any(np.isnan(position))
                    or np.any(np.isnan(next_position))
                    or np.any(np.isnan(eval_position))
                ):
                    continue

                velocity = (next_position - position) / cell.dt
                other_eval_positions = np.delete(eval_chroms, chrom_index, axis=0)
                g_xx = _pairwise_feature_sum(eval_position, other_eval_positions, basis_xx)
                g_xy = _pairwise_feature_sum(eval_position, eval_poles, basis_xy)
                block = np.hstack([g_xx, g_xy])
                rows.append(block)
                responses.append(velocity)

    n_cols = basis_xx.n_basis + basis_xy.n_basis
    if not rows:
        return np.zeros((0, n_cols), dtype=np.float64), np.zeros(0, dtype=np.float64)

    return np.vstack(rows), np.concatenate(responses)
