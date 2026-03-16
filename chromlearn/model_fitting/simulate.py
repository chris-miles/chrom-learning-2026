from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class SyntheticDataset:
    """Synthetic benchmark with known ground-truth kernels and trajectories."""

    chromosomes: np.ndarray  # (T, 3, N)
    centrosomes: np.ndarray  # (T, 3, 2) for backward compat
    kernel_xx: Callable[[np.ndarray], np.ndarray] | None
    kernel_xy: Callable[[np.ndarray], np.ndarray]
    D_x: float
    dt: float


def simulate_trajectories(
    kernel_xx: Callable[[np.ndarray], np.ndarray] | None,
    kernel_xy: Callable[[np.ndarray], np.ndarray],
    partner_positions: np.ndarray,
    x0: np.ndarray,
    n_steps: int,
    dt: float,
    D_x: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Euler-Maruyama forward simulation of overdamped Langevin dynamics.

    At each step: ``X(t+dt) = X(t) + F(X(t)) * dt + sqrt(2 D dt) * xi``
    where ``xi ~ N(0, I)``.

    Args:
        kernel_xx: ``f_xx(r)`` -- chromosome-chromosome kernel.  Takes a 1-D
            array of distances, returns force magnitudes.  ``None`` to skip
            chromosome-chromosome forces.
        kernel_xy: ``f_xy(r)`` -- chromosome-partner kernel.
        partner_positions: Shape ``(n_partners, n_steps+1, 3)``.
        x0: Initial chromosome positions, shape ``(N, 3)``.
        n_steps: Number of time steps to simulate.
        dt: Time step size in seconds.
        D_x: Isotropic diffusion coefficient.
        rng: Random number generator.

    Returns:
        Trajectory array of shape ``(n_steps+1, 3, N)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_chromosomes = x0.shape[0]
    trajectory = np.zeros((n_steps + 1, 3, n_chromosomes), dtype=np.float64)
    trajectory[0] = x0.T
    noise_scale = np.sqrt(2.0 * D_x * dt)

    for step in range(n_steps):
        # Work in (N, 3) layout for vectorized pairwise arithmetic.
        positions = trajectory[step].T
        forces = np.zeros_like(positions)

        if kernel_xx is not None:
            delta_xx = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
            dist_xx = np.linalg.norm(delta_xx, axis=2)
            pair_mask_xx = dist_xx > 1e-12
            safe_dist_xx = np.where(pair_mask_xx, dist_xx, 1.0)
            direction_xx = delta_xx / safe_dist_xx[:, :, np.newaxis]
            direction_xx[~pair_mask_xx] = 0.0

            f_xx = np.zeros_like(dist_xx)
            if pair_mask_xx.any():
                f_xx[pair_mask_xx] = kernel_xx(dist_xx[pair_mask_xx])
            forces += np.sum(direction_xx * f_xx[:, :, np.newaxis], axis=1)

        partners_t = partner_positions[:, step, :]
        delta_xy = partners_t[np.newaxis, :, :] - positions[:, np.newaxis, :]
        dist_xy = np.linalg.norm(delta_xy, axis=2)
        pair_mask_xy = dist_xy > 1e-12
        safe_dist_xy = np.where(pair_mask_xy, dist_xy, 1.0)
        direction_xy = delta_xy / safe_dist_xy[:, :, np.newaxis]
        direction_xy[~pair_mask_xy] = 0.0

        f_xy = np.zeros_like(dist_xy)
        if pair_mask_xy.any():
            f_xy[pair_mask_xy] = kernel_xy(dist_xy[pair_mask_xy])
        forces += np.sum(direction_xy * f_xy[:, :, np.newaxis], axis=1)

        # Preserve the historical RNG draw order from the original
        # implementation, which sampled in (3, N) layout.
        noise = noise_scale * rng.standard_normal(size=(3, n_chromosomes)).T
        trajectory[step + 1] = (positions + forces * dt + noise).T

    return trajectory


def generate_synthetic_data(
    kernel_xx: Callable[[np.ndarray], np.ndarray] | None,
    kernel_xy: Callable[[np.ndarray], np.ndarray],
    n_chromosomes: int = 20,
    n_steps: int = 100,
    dt: float = 5.0,
    D_x: float = 0.1,
    pole_separation: float = 10.0,
    rng: np.random.Generator | None = None,
) -> SyntheticDataset:
    """Generate a synthetic benchmark dataset with known ground-truth kernels.

    Creates centrosomes at fixed symmetric positions on the x-axis, draws
    random initial chromosome positions, and runs
    :func:`simulate_trajectories` forward.
    """
    if rng is None:
        rng = np.random.default_rng()

    partners = np.zeros((2, n_steps + 1, 3), dtype=np.float64)
    partners[0, :, 0] = -0.5 * pole_separation
    partners[1, :, 0] = 0.5 * pole_separation
    x0 = rng.normal(0.0, 2.0, size=(n_chromosomes, 3))
    chromosomes = simulate_trajectories(
        kernel_xx=kernel_xx,
        kernel_xy=kernel_xy,
        partner_positions=partners,
        x0=x0,
        n_steps=n_steps,
        dt=dt,
        D_x=D_x,
        rng=rng,
    )
    return SyntheticDataset(
        chromosomes=chromosomes,
        centrosomes=partners.transpose(1, 2, 0),  # (T, 3, 2) for backward compat
        kernel_xx=kernel_xx,
        kernel_xy=kernel_xy,
        D_x=D_x,
        dt=dt,
    )


def add_localization_noise(
    trajectories: np.ndarray,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add isotropic Gaussian localisation noise to trajectory positions.

    Args:
        trajectories: Position array of shape ``(T, 3, N)``.
        sigma: Standard deviation of noise per coordinate (microns).
        rng: Random number generator.

    Returns:
        Noisy trajectory of the same shape.
    """
    if rng is None:
        rng = np.random.default_rng()
    return trajectories + sigma * rng.standard_normal(size=trajectories.shape)
