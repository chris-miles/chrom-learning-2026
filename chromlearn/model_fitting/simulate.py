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
    n_partners = partner_positions.shape[0]
    trajectory = np.zeros((n_steps + 1, 3, n_chromosomes), dtype=np.float64)
    trajectory[0] = x0.T
    noise_scale = np.sqrt(2.0 * D_x * dt)

    for step in range(n_steps):
        positions = trajectory[step]
        forces = np.zeros_like(positions)

        for chrom_index in range(n_chromosomes):
            current = positions[:, chrom_index]

            if kernel_xx is not None:
                for neighbor_index in range(n_chromosomes):
                    if neighbor_index == chrom_index:
                        continue
                    delta = positions[:, neighbor_index] - current
                    distance = float(np.linalg.norm(delta))
                    if distance <= 1e-12:
                        continue
                    direction = delta / distance
                    forces[:, chrom_index] += kernel_xx(np.array([distance]))[0] * direction

            for partner_index in range(n_partners):
                delta = partner_positions[partner_index, step] - current
                distance = float(np.linalg.norm(delta))
                if distance <= 1e-12:
                    continue
                direction = delta / distance
                forces[:, chrom_index] += kernel_xy(np.array([distance]))[0] * direction

        noise = noise_scale * rng.standard_normal(size=(3, n_chromosomes))
        trajectory[step + 1] = positions + forces * dt + noise

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
