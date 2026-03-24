from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.linalg import block_diag

from chromlearn.io.trajectory import TrimmedCell
from chromlearn.model_fitting import FitConfig


@dataclass
class FitResult:
    """Result of a single penalised least-squares kernel fit."""

    theta: np.ndarray
    residuals: np.ndarray
    D_x: float


@dataclass
class BootstrapResult:
    """Bootstrap distribution of kernel coefficients (cell-level resampling)."""

    theta_samples: np.ndarray
    theta_mean: np.ndarray
    theta_std: np.ndarray


@dataclass
class CVResult:
    """Leave-one-cell-out cross-validation result (MSE on held-out cells)."""

    held_out_errors: np.ndarray
    mean_error: float
    fold_sd: float

    @property
    def fold_se(self) -> float:
        """Standard error of the mean (fold_sd / sqrt(n_valid_folds))."""
        n = int(np.sum(np.isfinite(self.held_out_errors)))
        return self.fold_sd / np.sqrt(n) if n > 0 else np.inf

    @property
    def std_error(self) -> float:
        """Deprecated alias for fold_sd (kept for backward compatibility)."""
        return self.fold_sd


@dataclass
class RolloutCVResult:
    """Leave-one-cell-out rollout validation summary.

    Per-cell arrays have shape ``(n_cells,)``; ``horizon_errors`` has shape
    ``(n_cells, n_horizons)``.  Callers can derive means via
    ``np.nanmean(result.axial_mse)`` etc.
    """

    horizons: np.ndarray
    axial_mse: np.ndarray
    radial_mse: np.ndarray
    endpoint_mean_error: np.ndarray
    final_axial_wasserstein: np.ndarray
    final_radial_wasserstein: np.ndarray
    horizon_errors: np.ndarray


def _block_roughness(R_xx: np.ndarray | None, R_xy: np.ndarray) -> np.ndarray:
    if R_xx is None:
        return R_xy
    return block_diag(R_xx, R_xy)


def _topology_has_chroms(topology: str) -> bool:
    return topology in ("poles_and_chroms", "center_and_chroms")


def fit_kernels(
    G: np.ndarray,
    V: np.ndarray,
    lambda_ridge: float,
    lambda_rough: float,
    R: np.ndarray,
) -> FitResult:
    """Solve penalized least squares for kernel coefficients.

    Minimises ``||V - G @ theta||^2 + lambda_ridge * ||theta||^2
    + lambda_rough * theta^T R theta`` via the normal equations.

    Args:
        G: Design matrix, shape ``(n_obs, n_basis)``.
        V: Response vector (displacement velocities), shape ``(n_obs,)``.
        lambda_ridge: Ridge (L2) regularisation strength.
        lambda_rough: Roughness penalty strength.
        R: Roughness penalty matrix, shape ``(n_basis, n_basis)``.

    Returns:
        FitResult with ``theta``, ``residuals``, and ``D_x`` (set to 0;
        call :func:`estimate_diffusion` separately).
    """
    if G.ndim != 2:
        raise ValueError("G must be a 2D array.")
    if V.ndim != 1:
        raise ValueError("V must be a 1D array.")
    if G.shape[0] != V.shape[0]:
        raise ValueError("G and V must have the same number of rows.")

    n_basis = G.shape[1]
    normal_matrix = G.T @ G + lambda_ridge * np.eye(n_basis) + lambda_rough * R
    rhs = G.T @ V
    theta = np.linalg.solve(normal_matrix, rhs)
    residuals = V - G @ theta
    return FitResult(theta=theta, residuals=residuals, D_x=0.0)


def estimate_diffusion(
    V: np.ndarray,
    G: np.ndarray,
    theta: np.ndarray,
    dt: float,
    d: int = 3,
    diffusion_mode: str = "msd",
) -> float:
    """Estimate the scalar diffusion coefficient from regression residuals.

    For overdamped Langevin dynamics the velocity residuals have per-component
    variance ``2 D / dt``, so ``D = mean(residual^2) * dt / 2``.

    The *d* parameter is accepted for interface compatibility but is not used
    in the calculation because the flattened residual vector already includes
    all spatial components and the per-component mean is dimension-independent.

    Args:
        V: Response vector (velocity units).
        G: Design matrix.
        theta: Fitted coefficients.
        dt: Time step in seconds.
        d: Spatial dimension (unused; kept for API symmetry).
        diffusion_mode: Estimation mode. ``"msd"`` uses residual variance
            (default). For position-dependent diffusion (e.g. ``"local"``),
            use ``diffusion.local_diffusion_estimates`` directly on the raw
            displacement data instead of this convenience function.

    Returns:
        Estimated isotropic diffusion coefficient ``D_x``.
    """
    if d <= 0:
        raise ValueError("d must be positive.")
    if diffusion_mode != "msd":
        raise ValueError(
            f"estimate_diffusion only supports diffusion_mode='msd'. "
            f"For mode '{diffusion_mode}', use "
            f"diffusion.local_diffusion_estimates directly."
        )
    residuals = V - G @ theta
    mean_square_residual = float(np.mean(residuals**2))
    return 0.5 * mean_square_residual * dt


def bootstrap_kernels(
    cells: list[TrimmedCell],
    config: FitConfig,
    n_boot: int = 250,
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """Bootstrap kernel fits by resampling cells with replacement.

    Each bootstrap iteration draws ``len(cells)`` cells with replacement,
    rebuilds the design matrix, and refits.  The returned
    :class:`BootstrapResult` contains the full distribution of ``theta``
    samples for confidence-interval construction.
    """
    from chromlearn.model_fitting.basis import BSplineBasis, HatBasis
    from chromlearn.model_fitting.features import build_design_matrix

    if rng is None:
        rng = np.random.default_rng()
    if not cells:
        raise ValueError("bootstrap_kernels requires at least one cell.")

    BasisClass = BSplineBasis if config.basis_type == "bspline" else HatBasis
    if _topology_has_chroms(config.topology):
        basis_xx = BasisClass(config.r_min_xx, config.r_max_xx, config.n_basis_xx)
    else:
        basis_xx = None
    basis_xy = BasisClass(config.r_min_xy, config.r_max_xy, config.n_basis_xy)

    R_xx = basis_xx.roughness_matrix() if basis_xx is not None else None
    roughness = _block_roughness(R_xx, basis_xy.roughness_matrix())
    n_bxx = basis_xx.n_basis if basis_xx is not None else 0
    theta_samples = np.zeros((n_boot, n_bxx + basis_xy.n_basis))

    for boot_index in range(n_boot):
        sampled_indices = rng.choice(len(cells), size=len(cells), replace=True)
        sampled_cells = [cells[i] for i in sampled_indices]
        G, V = build_design_matrix(
            sampled_cells, basis_xx, basis_xy,
            basis_eval_mode=config.basis_eval_mode,
            topology=config.topology,
        )
        theta_samples[boot_index] = fit_kernels(
            G, V,
            lambda_ridge=config.lambda_ridge,
            lambda_rough=config.lambda_rough,
            R=roughness,
        ).theta

    return BootstrapResult(
        theta_samples=theta_samples,
        theta_mean=np.mean(theta_samples, axis=0),
        theta_std=np.std(theta_samples, axis=0),
    )


def cross_validate(
    cells: list[TrimmedCell],
    config: FitConfig,
) -> CVResult:
    """Leave-one-cell-out cross-validation.

    For each fold one cell is held out, the model is fit on the remaining
    cells, and the mean squared prediction error on the held-out cell is
    recorded.

    Returns:
        CVResult with per-fold errors and summary statistics.
    """
    from chromlearn.model_fitting.basis import BSplineBasis, HatBasis
    from chromlearn.model_fitting.features import build_design_matrix

    if not cells:
        raise ValueError("cross_validate requires at least one cell.")

    BasisClass = BSplineBasis if config.basis_type == "bspline" else HatBasis
    if _topology_has_chroms(config.topology):
        basis_xx = BasisClass(config.r_min_xx, config.r_max_xx, config.n_basis_xx)
    else:
        basis_xx = None
    basis_xy = BasisClass(config.r_min_xy, config.r_max_xy, config.n_basis_xy)

    R_xx = basis_xx.roughness_matrix() if basis_xx is not None else None
    roughness = _block_roughness(R_xx, basis_xy.roughness_matrix())
    errors = np.full(len(cells), np.nan, dtype=np.float64)

    for held_out_index in range(len(cells)):
        train_cells = [
            cell for index, cell in enumerate(cells) if index != held_out_index
        ]
        test_cells = [cells[held_out_index]]

        G_train, V_train = build_design_matrix(
            train_cells, basis_xx, basis_xy,
            basis_eval_mode=config.basis_eval_mode,
            topology=config.topology,
        )
        if G_train.size == 0:
            continue

        fit_result = fit_kernels(
            G_train,
            V_train,
            lambda_ridge=config.lambda_ridge,
            lambda_rough=config.lambda_rough,
            R=roughness,
        )
        G_test, V_test = build_design_matrix(
            test_cells, basis_xx, basis_xy,
            basis_eval_mode=config.basis_eval_mode,
            topology=config.topology,
        )
        if G_test.size == 0:
            continue
        predictions = G_test @ fit_result.theta
        errors[held_out_index] = np.mean((V_test - predictions) ** 2)

    return CVResult(
        held_out_errors=errors,
        mean_error=float(np.nanmean(errors)),
        fold_sd=float(np.nanstd(errors)),
    )


def rollout_cross_validate(
    cells: list[TrimmedCell],
    config: FitConfig,
    n_reps: int = 8,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    rng: np.random.Generator | None = None,
) -> RolloutCVResult:
    """Leave-one-cell-out rollout validation on spindle-frame summaries.

    Each fold fits the model on ``len(cells) - 1`` cells, simulates the held-out
    cell forward from its real initial conditions while using its real partner
    trajectories, and compares the resulting spindle-frame summaries.

    Metrics per held-out cell:

    - ``axial_mse`` / ``radial_mse``: full-trajectory MSE of mean position.
    - ``endpoint_mean_error``: squared error of the mean position at the final frame.
    - ``final_axial_wasserstein`` / ``final_radial_wasserstein``: 1-D Wasserstein
      distance between real and simulated chromosome distributions at the final frame.
    - ``horizon_errors``: combined axial+radial squared error at selected time horizons.
    """
    from chromlearn.io.trajectory import spindle_frame
    from chromlearn.model_fitting.simulate import simulate_cell

    if not cells:
        raise ValueError("rollout_cross_validate requires at least one cell.")
    if rng is None:
        rng = np.random.default_rng()

    horizon_values = np.array(sorted({int(h) for h in horizons if int(h) > 0}), dtype=int)
    if horizon_values.size == 0:
        raise ValueError("horizons must contain at least one positive integer.")

    n_cells = len(cells)
    axial_mse = np.full(n_cells, np.nan, dtype=np.float64)
    radial_mse = np.full(n_cells, np.nan, dtype=np.float64)
    endpoint_mean_error = np.full(n_cells, np.nan, dtype=np.float64)
    final_axial_wasserstein = np.full(n_cells, np.nan, dtype=np.float64)
    final_radial_wasserstein = np.full(n_cells, np.nan, dtype=np.float64)
    horizon_errors = np.full((n_cells, horizon_values.size), np.nan, dtype=np.float64)

    for held_out_index in range(n_cells):
        train_cells = [
            cell for index, cell in enumerate(cells) if index != held_out_index
        ]
        test_cell = cells[held_out_index]
        model = fit_model(train_cells, config)

        real_sf = spindle_frame(test_cell)
        real_axial_mean = np.nanmean(real_sf.axial, axis=1)
        real_radial_mean = np.nanmean(real_sf.radial, axis=1)

        # Simulate n_reps rollouts and average their spindle-frame summaries
        sim_rollouts = []
        for _ in range(n_reps):
            rep_rng = np.random.default_rng(
                int(rng.integers(0, np.iinfo(np.int64).max))
            )
            _, sim_cell = simulate_cell(test_cell, model, rng=rep_rng)
            sim_rollouts.append(spindle_frame(sim_cell))

        sim_axial_mean = np.nanmean(
            np.stack([np.nanmean(sf.axial, axis=1) for sf in sim_rollouts], axis=0),
            axis=0,
        )
        sim_radial_mean = np.nanmean(
            np.stack([np.nanmean(sf.radial, axis=1) for sf in sim_rollouts], axis=0),
            axis=0,
        )

        axial_mse[held_out_index] = float(
            np.nanmean((real_axial_mean - sim_axial_mean) ** 2)
        )
        radial_mse[held_out_index] = float(
            np.nanmean((real_radial_mean - sim_radial_mean) ** 2)
        )
        endpoint_mean_error[held_out_index] = float(
            (real_axial_mean[-1] - sim_axial_mean[-1]) ** 2
            + (real_radial_mean[-1] - sim_radial_mean[-1]) ** 2
        )

        # Final-frame distributional comparison (pool all rollouts)
        real_ax_valid = real_sf.axial[-1]
        real_ax_valid = real_ax_valid[np.isfinite(real_ax_valid)]
        real_rad_valid = real_sf.radial[-1]
        real_rad_valid = real_rad_valid[np.isfinite(real_rad_valid)]
        sim_ax_valid = np.concatenate(
            [sf.axial[-1][np.isfinite(sf.axial[-1])] for sf in sim_rollouts]
        )
        sim_rad_valid = np.concatenate(
            [sf.radial[-1][np.isfinite(sf.radial[-1])] for sf in sim_rollouts]
        )

        if real_ax_valid.size > 0 and sim_ax_valid.size > 0:
            final_axial_wasserstein[held_out_index] = float(
                stats.wasserstein_distance(real_ax_valid, sim_ax_valid)
            )
        if real_rad_valid.size > 0 and sim_rad_valid.size > 0:
            final_radial_wasserstein[held_out_index] = float(
                stats.wasserstein_distance(real_rad_valid, sim_rad_valid)
            )

        T = real_axial_mean.size
        for horizon_index, horizon in enumerate(horizon_values):
            if horizon >= T:
                continue
            horizon_errors[held_out_index, horizon_index] = float(
                (real_axial_mean[horizon] - sim_axial_mean[horizon]) ** 2
                + (real_radial_mean[horizon] - sim_radial_mean[horizon]) ** 2
            )

    return RolloutCVResult(
        horizons=horizon_values,
        axial_mse=axial_mse,
        radial_mse=radial_mse,
        endpoint_mean_error=endpoint_mean_error,
        final_axial_wasserstein=final_axial_wasserstein,
        final_radial_wasserstein=final_radial_wasserstein,
        horizon_errors=horizon_errors,
    )


def fit_model(
    cells: list[TrimmedCell],
    config: FitConfig | None = None,
) -> "FittedModel":
    """Fit pairwise interaction kernels from trimmed trajectories.

    Constructs bases from *config*, builds the design matrix, solves the
    penalized regression, and estimates the scalar diffusion coefficient.

    Args:
        cells: Trimmed cell trajectories.
        config: Fitting configuration.  Uses ``FitConfig()`` defaults when
            *None*.

    Returns:
        A :class:`FittedModel` ready for evaluation and plotting.
    """
    from chromlearn.model_fitting.basis import BSplineBasis, HatBasis
    from chromlearn.model_fitting.features import build_design_matrix
    from chromlearn.model_fitting.model import FittedModel

    if config is None:
        config = FitConfig()

    BasisClass = BSplineBasis if config.basis_type == "bspline" else HatBasis
    if _topology_has_chroms(config.topology):
        basis_xx = BasisClass(config.r_min_xx, config.r_max_xx, config.n_basis_xx)
    else:
        basis_xx = None
    basis_xy = BasisClass(config.r_min_xy, config.r_max_xy, config.n_basis_xy)

    G, V = build_design_matrix(
        cells, basis_xx, basis_xy,
        basis_eval_mode=config.basis_eval_mode,
        topology=config.topology,
    )
    R_xx = basis_xx.roughness_matrix() if basis_xx is not None else None
    roughness = _block_roughness(R_xx, basis_xy.roughness_matrix())
    result = fit_kernels(
        G, V,
        lambda_ridge=config.lambda_ridge,
        lambda_rough=config.lambda_rough,
        R=roughness,
    )
    # --- Diffusion estimation ---
    from chromlearn.model_fitting.diffusion import (
        estimate_diffusion_variable,
        local_diffusion_estimates,
    )

    if config.diffusion_mode == "msd":
        D_x = estimate_diffusion(V, G, result.theta, dt=config.dt)
    else:
        # Non-MSD modes operate on raw displacements, not regression residuals
        D_locals = local_diffusion_estimates(
            cells, dt=config.dt, mode=config.diffusion_mode,
            fit_result=result if config.diffusion_mode == "f_corrected" else None,
            basis_xx=basis_xx if config.diffusion_mode == "f_corrected" else None,
            basis_xy=basis_xy if config.diffusion_mode == "f_corrected" else None,
            topology=config.topology,
        )
        all_D = np.concatenate([d.ravel() for d in D_locals])
        valid_D = all_D[np.isfinite(all_D)]
        D_x = float(np.mean(valid_D)) if valid_D.size > 0 else 0.0

    diffusion_model = None
    if config.D_variable:
        BasisClass_D = BSplineBasis if config.basis_type == "bspline" else HatBasis
        basis_D = BasisClass_D(config.r_min_D, config.r_max_D, config.n_basis_D)
        diffusion_model = estimate_diffusion_variable(
            cells, basis_D,
            coord_name=config.D_coordinate,
            dt=config.dt,
            mode=config.diffusion_mode,
            fit_result=result if config.diffusion_mode == "f_corrected" else None,
            basis_xx=basis_xx if config.diffusion_mode == "f_corrected" else None,
            basis_xy=basis_xy if config.diffusion_mode == "f_corrected" else None,
            topology=config.topology,
        )

    return FittedModel(
        theta=result.theta,
        n_basis_xx=basis_xx.n_basis if basis_xx is not None else 0,
        n_basis_xy=basis_xy.n_basis,
        basis_xx=basis_xx,
        basis_xy=basis_xy,
        D_x=D_x,
        dt=config.dt,
        metadata={"n_cells": len(cells)},
        diffusion_model=diffusion_model,
        topology=config.topology,
    )


def paired_cv_differences(
    cv_results: dict[str, CVResult],
    reference: str,
) -> dict[str, tuple[float, float]]:
    """Paired fold-by-fold CV loss differences relative to a reference topology.

    Because every topology is evaluated on the same held-out cells, the correct
    uncertainty for comparing two topologies is the SE of the **paired**
    foldwise difference.

    Args:
        cv_results: Mapping from topology name to CVResult.
        reference: The topology to subtract (typically the best-scoring one).

    Returns:
        Dict mapping each topology to ``(mean_diff, se_diff)`` where
        ``mean_diff = mean(errors_topo - errors_ref)`` across folds.
    """
    ref_errors = cv_results[reference].held_out_errors
    out: dict[str, tuple[float, float]] = {}
    for topo, cv in cv_results.items():
        diff = cv.held_out_errors - ref_errors
        valid = np.isfinite(diff)
        n = int(valid.sum())
        mean_diff = float(np.mean(diff[valid])) if n > 0 else np.inf
        sd_diff = float(np.std(diff[valid], ddof=1)) if n > 1 else np.inf
        se_diff = sd_diff / np.sqrt(n) if n > 0 else np.inf
        out[topo] = (mean_diff, se_diff)
    return out
