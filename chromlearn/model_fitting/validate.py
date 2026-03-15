from __future__ import annotations

import numpy as np
from scipy import stats


def one_step_prediction_error(V: np.ndarray, G: np.ndarray, theta: np.ndarray) -> float:
    """Mean squared one-step prediction error."""
    predictions = G @ theta
    return float(np.mean((V - predictions) ** 2))


def residual_diagnostics(residuals: np.ndarray) -> dict[str, float]:
    """Compute summary statistics on regression residuals.

    Returns a dict with ``mean``, ``std``, ``skewness``, ``kurtosis``, and
    (when n >= 3) ``normality_p_value`` from the Shapiro-Wilk test.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    diagnostics = {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "skewness": float(stats.skew(residuals, bias=False)),
        "kurtosis": float(stats.kurtosis(residuals, fisher=True, bias=False)),
    }
    if residuals.size >= 3:
        sample = residuals
        if residuals.size > 5000:
            rng = np.random.default_rng(0)
            sample = rng.choice(residuals, size=5000, replace=False)
        diagnostics["normality_p_value"] = float(stats.shapiro(sample)[1])
    return diagnostics


def kernel_recovery_error(
    r: np.ndarray,
    true_values: np.ndarray,
    fitted_values: np.ndarray,
) -> float:
    """Root-mean-square error between true and fitted kernel values."""
    _ = r
    return float(np.sqrt(np.mean((true_values - fitted_values) ** 2)))


def summary_statistics(chromosomes: np.ndarray, centrosomes: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for comparing real and simulated trajectories.

    Returns a dict with mean/std distance from pole center, mean/std
    pairwise chromosome distance, and lag-1 MSD.
    """
    n_timepoints, _, n_chromosomes = chromosomes.shape
    center = 0.5 * (centrosomes[:, :, 0] + centrosomes[:, :, 1])
    distances_from_center = np.linalg.norm(
        np.moveaxis(chromosomes, 2, 1) - center[:, np.newaxis, :],
        axis=2,
    )

    pairwise_distances: list[float] = []
    sample_stride = max(1, n_timepoints // 10)
    for time_index in range(0, n_timepoints, sample_stride):
        positions = chromosomes[time_index].T
        for first in range(n_chromosomes):
            for second in range(first + 1, n_chromosomes):
                pairwise_distances.append(
                    float(np.linalg.norm(positions[first] - positions[second]))
                )

    displacements = chromosomes[1:] - chromosomes[:-1]
    msd_lag1 = float(np.mean(np.sum(displacements**2, axis=1)))
    return {
        "mean_dist_from_center": float(np.mean(distances_from_center)),
        "std_dist_from_center": float(np.std(distances_from_center)),
        "mean_pairwise_dist": float(np.mean(pairwise_distances)),
        "std_pairwise_dist": float(np.std(pairwise_distances)),
        "msd_lag1": msd_lag1,
    }
