from __future__ import annotations

import numpy as np

from chromlearn.io.trajectory import TrimmedCell


def build_design_matrix(
    cells: list[TrimmedCell],
    basis_xx,
    basis_xy,
    basis_eval_mode: str = "ito",
    topology: str = "poles",
    r_cutoff_xx: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the stacked regression design matrix and velocity response.

    Parameters
    ----------
    cells : list[TrimmedCell]
        Trimmed trajectory data for each cell.
    basis_xx : Basis or None
        Basis function object for chromosome-chromosome interactions.
        ``None`` to skip chromosome-chromosome block entirely.
    basis_xy : Basis
        Basis function object for chromosome-partner interactions.
    basis_eval_mode : str, default ``"ito"``
        Controls which positions are used to evaluate pairwise distances
        and basis functions.  The velocity is always ``(X(t+1) - X(t)) / dt``.

        * ``"ito"`` -- Evaluate basis at current positions ``X(t)``.
        * ``"ito_shift"`` -- Evaluate basis at *previous* positions ``X(t-1)``.
        * ``"strato"`` -- Evaluate basis at the midpoint ``(X(t) + X(t+1)) / 2``.
    topology : str, default ``"poles"``
        Interaction topology — controls which partner positions are used.
    r_cutoff_xx : float or None
        If not None, zero out chromosome-chromosome interactions for
        distances above this cutoff.
    """
    from chromlearn.io.trajectory import get_partners

    if basis_eval_mode not in ("ito", "ito_shift", "strato"):
        raise ValueError(
            f"Unknown basis_eval_mode={basis_eval_mode!r}. "
            "Expected 'ito', 'ito_shift', or 'strato'."
        )

    all_G: list[np.ndarray] = []
    all_V: list[np.ndarray] = []

    for cell in cells:
        partners = get_partners(cell, topology)
        G_cell, V_cell = _build_cell_design_matrix(
            cell, partners, basis_xx, basis_xy, basis_eval_mode,
            r_cutoff_xx=r_cutoff_xx,
        )
        if G_cell.size > 0:
            all_G.append(G_cell)
            all_V.append(V_cell)

    n_bxx = basis_xx.n_basis if basis_xx is not None else 0
    n_cols = n_bxx + basis_xy.n_basis
    if not all_G:
        return np.zeros((0, n_cols), dtype=np.float64), np.zeros(0, dtype=np.float64)

    return np.vstack(all_G), np.concatenate(all_V)


def _build_cell_design_matrix(
    cell: TrimmedCell,
    partners: np.ndarray,
    basis_xx,
    basis_xy,
    basis_eval_mode: str,
    r_cutoff_xx: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized design matrix construction for a single cell.

    Parameters
    ----------
    partners : np.ndarray
        Shape ``(n_partners, T, 3)`` — external interaction partner trajectories.
    basis_xx : Basis or None
        ``None`` to skip chromosome-chromosome interactions.
    r_cutoff_xx : float or None
        If not None, zero out xx interactions for distances above this cutoff.
    """
    chromosomes = cell.chromosomes
    T = chromosomes.shape[0]
    N = chromosomes.shape[2]
    n_partners = partners.shape[0]
    n_bxx = basis_xx.n_basis if basis_xx is not None else 0
    n_bxy = basis_xy.n_basis
    dt = cell.dt

    t_start = 1 if basis_eval_mode == "ito_shift" else 0

    rows_G: list[np.ndarray] = []
    rows_V: list[np.ndarray] = []

    for t in range(t_start, T - 1):
        # Positions for velocity: (N, 3)
        pos_cur = chromosomes[t].T      # (N, 3)
        pos_next = chromosomes[t + 1].T  # (N, 3)

        # Evaluation positions for basis
        if basis_eval_mode == "ito":
            eval_chroms = pos_cur            # (N, 3)
            eval_partners = partners[:, t, :]       # (n_partners, 3)
        elif basis_eval_mode == "ito_shift":
            eval_chroms = chromosomes[t - 1].T
            eval_partners = partners[:, t - 1, :]
        else:  # strato
            eval_chroms = 0.5 * (pos_cur + pos_next)
            eval_partners = 0.5 * (partners[:, t, :] + partners[:, t + 1, :])

        # Valid mask: chromosome i is valid if pos_cur[i], pos_next[i], and
        # eval_chroms[i] all have no NaN.
        valid = (
            ~np.any(np.isnan(pos_cur), axis=1)
            & ~np.any(np.isnan(pos_next), axis=1)
            & ~np.any(np.isnan(eval_chroms), axis=1)
        )  # (N,)
        n_valid = valid.sum()
        if n_valid == 0:
            continue

        valid_idx = np.flatnonzero(valid)
        eval_valid = eval_chroms[valid_idx]  # (n_valid, 3)

        # --- Chromosome-chromosome interactions ---
        if basis_xx is not None:
            # Deltas: (n_valid, N, 3)
            delta_xx = eval_chroms[np.newaxis, :, :] - eval_valid[:, np.newaxis, :]
            dist_xx = np.linalg.norm(delta_xx, axis=2)  # (n_valid, N)

            # Mask: skip self-pairs, NaN neighbors, zero distances,
            # and distances beyond the xx cutoff (if set).
            neighbor_valid = ~np.any(np.isnan(eval_chroms), axis=1)  # (N,)
            pair_mask = np.ones((n_valid, N), dtype=bool)
            pair_mask[np.arange(n_valid), valid_idx] = False  # exclude self
            pair_mask[:, ~neighbor_valid] = False
            pair_mask &= dist_xx > 1e-12
            if r_cutoff_xx is not None:
                pair_mask &= dist_xx <= r_cutoff_xx

            # Unit directions where valid, zero elsewhere
            safe_dist = np.where(pair_mask, dist_xx, 1.0)
            direction_xx = delta_xx / safe_dist[:, :, np.newaxis]  # (n_valid, N, 3)
            direction_xx[~pair_mask] = 0.0

            # Evaluate basis on all distances at once
            flat_dist_xx = dist_xx[pair_mask]
            phi_xx_flat = basis_xx.evaluate(flat_dist_xx)  # (n_pairs, n_bxx)

            # Scatter back to (n_valid, N, n_bxx)
            phi_xx = np.zeros((n_valid, N, n_bxx), dtype=np.float64)
            phi_xx[pair_mask] = phi_xx_flat

            # Contract: g_xx[i, d, b] = sum_j direction_xx[i, j, d] * phi_xx[i, j, b]
            g_xx = np.einsum("ijd,ijb->idb", direction_xx, phi_xx)  # (n_valid, 3, n_bxx)
        else:
            g_xx = np.zeros((n_valid, 3, 0), dtype=np.float64)

        # --- Chromosome-partner interactions ---
        # delta_xy[i, p, :] = eval_partners[p] - eval_valid[i]
        delta_xy = eval_partners[np.newaxis, :, :] - eval_valid[:, np.newaxis, :]  # (n_valid, n_partners, 3)
        dist_xy = np.linalg.norm(delta_xy, axis=2)  # (n_valid, n_partners)

        pole_mask = dist_xy > 1e-12
        safe_dist_xy = np.where(pole_mask, dist_xy, 1.0)
        direction_xy = delta_xy / safe_dist_xy[:, :, np.newaxis]
        direction_xy[~pole_mask] = 0.0

        flat_dist_xy = dist_xy[pole_mask]
        phi_xy_flat = basis_xy.evaluate(flat_dist_xy)
        phi_xy = np.zeros((n_valid, n_partners, n_bxy), dtype=np.float64)
        phi_xy[pole_mask] = phi_xy_flat

        g_xy = np.einsum("ipd,ipb->idb", direction_xy, phi_xy)  # (n_valid, 3, n_bxy)

        # --- Assemble row block ---
        G_block = np.concatenate([g_xx, g_xy], axis=2)  # (n_valid, 3, n_bxx + n_bxy)

        # Velocity: (n_valid, 3)
        velocity = (pos_next[valid_idx] - pos_cur[valid_idx]) / dt

        # Flatten: interleave spatial dims for each chromosome
        rows_G.append(G_block.reshape(-1, n_bxx + n_bxy))
        rows_V.append(velocity.reshape(-1))

    n_cols = n_bxx + n_bxy
    if not rows_G:
        return np.zeros((0, n_cols), dtype=np.float64), np.zeros(0, dtype=np.float64)

    return np.vstack(rows_G), np.concatenate(rows_V)
