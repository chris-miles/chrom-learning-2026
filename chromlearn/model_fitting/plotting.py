from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from chromlearn.model_fitting.fit import BootstrapResult, CVResult
from chromlearn.model_fitting.model import FittedModel


def plot_kernels(
    model: FittedModel,
    bootstrap: BootstrapResult | None = None,
    n_points: int = 200,
    ci_levels: list[float] | None = None,
) -> plt.Figure:
    if ci_levels is None:
        ci_levels = [0.05]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    specs = [
        ("xx", model.basis_xx, model.theta_xx, "Chromosome-chromosome"),
        ("xy", model.basis_xy, model.theta_xy, "Centrosome on chromosome"),
    ]

    for axis, (name, basis, theta, title) in zip(axes, specs):
        radius = np.linspace(basis.r_min, basis.r_max, n_points)
        phi = basis.evaluate(radius)
        axis.plot(radius, phi @ theta, color="C0", linewidth=2)
        if bootstrap is not None:
            if name == "xx":
                samples = bootstrap.theta_samples[:, : model.n_basis_xx]
            else:
                samples = bootstrap.theta_samples[:, model.n_basis_xx :]
            curves = phi @ samples.T
            for level in ci_levels:
                lo = np.quantile(curves, level, axis=1)
                hi = np.quantile(curves, 1.0 - level, axis=1)
                axis.fill_between(radius, lo, hi, color="C0", alpha=0.2)
        axis.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
        axis.set_xlabel("Distance (um)")
        axis.set_ylabel("Force")
        axis.set_title(title)

    figure.tight_layout()
    return figure


def plot_cv_curve(cv_results: dict[str, CVResult]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6, 4))
    labels = list(cv_results)
    x = np.arange(len(labels))
    means = [cv_results[label].mean_error for label in labels]
    stds = [cv_results[label].std_error for label in labels]
    axis.errorbar(x, means, yerr=stds, fmt="o-", capsize=4)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_ylabel("Mean squared error")
    axis.set_title("Leave-one-cell-out CV")
    figure.tight_layout()
    return figure


def plot_residuals(residuals: np.ndarray) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(residuals, bins=40, density=True, alpha=0.75)
    axes[0].set_title("Residual distribution")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Density")
    stats.probplot(residuals, plot=axes[1])
    axes[1].set_title("Residual Q-Q plot")
    figure.tight_layout()
    return figure


def plot_diffusion(diffusion_result, n_points: int = 200) -> plt.Figure:
    """Plot the fitted D(coordinate) curve."""
    coords = np.linspace(diffusion_result.basis_D.r_min, diffusion_result.basis_D.r_max, n_points)
    D_values = diffusion_result.evaluate(coords)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(coords, D_values, color="C0", linewidth=2)
    axis.axhline(diffusion_result.D_scalar, color="0.5", linestyle="--", linewidth=0.8, label="Scalar average")
    axis.set_xlabel(f"Position coordinate ({diffusion_result.coord_name})")
    axis.set_ylabel("D (um\u00b2/s)")
    axis.set_title("Diffusion coefficient")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_recovery(
    r: np.ndarray,
    true_values: np.ndarray,
    fitted_values: np.ndarray,
    kernel_name: str = "",
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(r, true_values, "k--", linewidth=2, label="True")
    axis.plot(r, fitted_values, color="C0", linewidth=2, label="Fitted")
    axis.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Distance (um)")
    axis.set_ylabel("Force")
    axis.set_title(f"Kernel recovery {kernel_name}".strip())
    axis.legend()
    figure.tight_layout()
    return figure
