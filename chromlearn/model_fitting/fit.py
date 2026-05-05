from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from scipy.linalg import block_diag

from chromlearn.io.trajectory import TrimmedCell
from chromlearn.model_fitting import FitConfig

_DEFAULT_N_JOBS: int = -1  # -1 → all cores (joblib convention)


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
    ``(n_cells, n_horizons)``.

    In **deterministic** mode (``deterministic=True``), a single noise-free
    ODE rollout is used and ``path_mse == ensemble_mse`` (both are the
    drift-trajectory MSE).

    In **stochastic** mode (the default, ``deterministic=False``), ``path_mse`` is the
    expected per-chromosome 3D squared error of a single rollout (averaged
    across replicates, includes drift bias + stochastic variance), and
    ``ensemble_mse`` averages positions across replicates before comparing
    to reality (cancels stochastic variance, isolates drift bias).

    ``horizon_path_mse`` and ``horizon_ensemble_mse`` are the horizon-resolved
    versions, both shape ``(n_cells, n_horizons)``.
    """

    horizons: np.ndarray
    path_mse: np.ndarray
    ensemble_mse: np.ndarray
    axial_mse: np.ndarray
    radial_mse: np.ndarray
    endpoint_mean_error: np.ndarray
    final_axial_wasserstein: np.ndarray
    final_radial_wasserstein: np.ndarray
    horizon_errors: np.ndarray
    horizon_path_mse: np.ndarray
    horizon_ensemble_mse: np.ndarray


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


def _boot_one(cells, sampled_indices, basis_xx, basis_xy, config, roughness):
    """Run a single bootstrap iteration."""
    from chromlearn.model_fitting.features import build_design_matrix

    sampled_cells = [cells[i] for i in sampled_indices]
    G, V = build_design_matrix(
        sampled_cells, basis_xx, basis_xy,
        basis_eval_mode=config.basis_eval_mode,
        topology=config.topology,
        r_cutoff_xx=config.r_cutoff_xx,
    )
    return fit_kernels(
        G, V,
        lambda_ridge=config.lambda_ridge,
        lambda_rough=config.lambda_rough,
        R=roughness,
    ).theta


def bootstrap_kernels(
    cells: list[TrimmedCell],
    config: FitConfig,
    n_boot: int = 250,
    rng: np.random.Generator | None = None,
    n_jobs: int = _DEFAULT_N_JOBS,
) -> BootstrapResult:
    """Bootstrap kernel fits by resampling cells with replacement.

    Each bootstrap iteration draws ``len(cells)`` cells with replacement,
    rebuilds the design matrix, and refits.  The returned
    :class:`BootstrapResult` contains the full distribution of ``theta``
    samples for confidence-interval construction.

    *n_jobs* controls parallelism (joblib convention: ``-1`` = all cores,
    ``1`` = serial).
    """
    from chromlearn.model_fitting.basis import BSplineBasis, HatBasis

    if rng is None:
        rng = np.random.default_rng()
    if not cells:
        raise ValueError("bootstrap_kernels requires at least one cell.")

    from chromlearn.model_fitting.basis import make_basis_with_envelope

    BasisClass = BSplineBasis if config.basis_type == "bspline" else HatBasis
    if _topology_has_chroms(config.topology):
        basis_xx = make_basis_with_envelope(
            BasisClass, config.r_min_xx, config.r_max_xx, config.n_basis_xx,
            envelope_r0=config.envelope_r0_xx, envelope_w=config.envelope_w_xx,
        )
    else:
        basis_xx = None
    basis_xy = BasisClass(config.r_min_xy, config.r_max_xy, config.n_basis_xy)

    R_xx = basis_xx.roughness_matrix() if basis_xx is not None else None
    roughness = _block_roughness(R_xx, basis_xy.roughness_matrix())

    # Pre-generate all resample indices (deterministic from rng)
    all_indices = [
        rng.choice(len(cells), size=len(cells), replace=True)
        for _ in range(n_boot)
    ]

    theta_list = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_boot_one)(cells, idx, basis_xx, basis_xy, config, roughness)
        for idx in all_indices
    )

    theta_samples = np.array(theta_list)
    return BootstrapResult(
        theta_samples=theta_samples,
        theta_mean=np.mean(theta_samples, axis=0),
        theta_std=np.std(theta_samples, axis=0),
    )


def cross_validate(
    cells: list[TrimmedCell],
    config: FitConfig,
    k_folds: int | None = None,
) -> CVResult:
    """Cross-validation (LOO or k-fold).

    For each fold the held-out cells are removed, the model is fit on the
    remaining cells, and the mean squared prediction error on each held-out
    cell is recorded.  Per-cell error array always has shape ``(n_cells,)``.

    When *k_folds* is ``None`` or ``>= len(cells)`` this is leave-one-out.

    Returns:
        CVResult with per-cell errors and summary statistics.
    """
    from chromlearn.model_fitting.basis import BSplineBasis, HatBasis
    from chromlearn.model_fitting.features import build_design_matrix

    if not cells:
        raise ValueError("cross_validate requires at least one cell.")

    n_cells = len(cells)

    if k_folds is None or k_folds >= n_cells:
        folds = [[i] for i in range(n_cells)]
    else:
        if k_folds < 2:
            raise ValueError("k_folds must be >= 2 (or None for LOO).")
        folds = [group.tolist() for group in np.array_split(range(n_cells), k_folds)]

    from chromlearn.model_fitting.basis import make_basis_with_envelope

    BasisClass = BSplineBasis if config.basis_type == "bspline" else HatBasis
    if _topology_has_chroms(config.topology):
        basis_xx = make_basis_with_envelope(
            BasisClass, config.r_min_xx, config.r_max_xx, config.n_basis_xx,
            envelope_r0=config.envelope_r0_xx, envelope_w=config.envelope_w_xx,
        )
    else:
        basis_xx = None
    basis_xy = BasisClass(config.r_min_xy, config.r_max_xy, config.n_basis_xy)

    R_xx = basis_xx.roughness_matrix() if basis_xx is not None else None
    roughness = _block_roughness(R_xx, basis_xy.roughness_matrix())
    errors = np.full(n_cells, np.nan, dtype=np.float64)

    for fold_indices in folds:
        fold_set = set(fold_indices)
        train_cells = [cell for i, cell in enumerate(cells) if i not in fold_set]

        G_train, V_train = build_design_matrix(
            train_cells, basis_xx, basis_xy,
            basis_eval_mode=config.basis_eval_mode,
            topology=config.topology,
            r_cutoff_xx=config.r_cutoff_xx,
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

        for held_out_index in fold_indices:
            test_cells = [cells[held_out_index]]
            G_test, V_test = build_design_matrix(
                test_cells, basis_xx, basis_xy,
                basis_eval_mode=config.basis_eval_mode,
                topology=config.topology,
                r_cutoff_xx=config.r_cutoff_xx,
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


def _rollout_one(test_cell, model, seed, deterministic=False):
    """Run one simulate_cell rep, return chromosomes + spindle-frame arrays."""
    from chromlearn.io.trajectory import spindle_frame
    from chromlearn.model_fitting.simulate import simulate_cell

    rep_rng = None if deterministic else np.random.default_rng(seed)
    _, sim_cell = simulate_cell(test_cell, model, rng=rep_rng,
                                deterministic=deterministic)
    sf = spindle_frame(sim_cell)
    return sim_cell.chromosomes, sf.axial, sf.radial


def rollout_cross_validate(
    cells: list[TrimmedCell],
    config: FitConfig,
    n_reps: int = 8,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    rng: np.random.Generator | None = None,
    k_folds: int | None = None,
    n_jobs: int = _DEFAULT_N_JOBS,
    deterministic: bool = False,
) -> RolloutCVResult:
    """Rollout cross-validation (LOO or k-fold).

    Each fold fits the model on training cells, simulates each held-out cell
    forward from its real initial conditions while using its real partner
    trajectories, and compares the resulting spindle-frame summaries.

    When *deterministic* is True, a single noise-free ODE rollout replaces
    the stochastic replicate ensemble.  This scores the drift trajectory
    directly and is appropriate when the goal is topology/drift selection and
    the noise model is uncertain.  ``n_reps`` and ``rng`` are ignored.

    When *k_folds* is ``None`` or ``>= len(cells)`` this is leave-one-out.
    Otherwise cells are split into *k_folds* groups (using
    ``np.array_split``, so non-divisible sizes are handled automatically).
    Per-cell metric arrays always have shape ``(n_cells,)`` regardless of
    fold structure.

    Metrics per held-out cell:

    - ``axial_mse`` / ``radial_mse``: full-trajectory MSE of mean position.
    - ``endpoint_mean_error``: squared error of the mean position at the final frame.
    - ``final_axial_wasserstein`` / ``final_radial_wasserstein``: 1-D Wasserstein
      distance between real and simulated chromosome distributions at the final frame.
    - ``horizon_errors``: combined axial+radial squared error at selected time horizons.
    """
    from chromlearn.io.trajectory import spindle_frame

    if not cells:
        raise ValueError("rollout_cross_validate requires at least one cell.")
    if rng is None:
        rng = np.random.default_rng()

    horizon_values = np.array(sorted({int(h) for h in horizons if int(h) > 0}), dtype=int)
    if horizon_values.size == 0:
        raise ValueError("horizons must contain at least one positive integer.")

    n_cells = len(cells)

    # Build fold assignments
    if k_folds is None or k_folds >= n_cells:
        folds = [[i] for i in range(n_cells)]
    else:
        if k_folds < 2:
            raise ValueError("k_folds must be >= 2 (or None for LOO).")
        folds = [group.tolist() for group in np.array_split(range(n_cells), k_folds)]

    path_mse = np.full(n_cells, np.nan, dtype=np.float64)
    ensemble_mse = np.full(n_cells, np.nan, dtype=np.float64)
    axial_mse = np.full(n_cells, np.nan, dtype=np.float64)
    radial_mse = np.full(n_cells, np.nan, dtype=np.float64)
    endpoint_mean_error = np.full(n_cells, np.nan, dtype=np.float64)
    final_axial_wasserstein = np.full(n_cells, np.nan, dtype=np.float64)
    final_radial_wasserstein = np.full(n_cells, np.nan, dtype=np.float64)
    horizon_errors = np.full((n_cells, horizon_values.size), np.nan, dtype=np.float64)
    horizon_path_mse = np.full((n_cells, horizon_values.size), np.nan, dtype=np.float64)
    horizon_ensemble_mse = np.full((n_cells, horizon_values.size), np.nan, dtype=np.float64)

    fold_set: set[int]
    for fold_indices in folds:
        fold_set = set(fold_indices)
        train_cells = [cell for i, cell in enumerate(cells) if i not in fold_set]
        model = fit_model(train_cells, config)

        for held_out_index in fold_indices:
            test_cell = cells[held_out_index]

            real_sf = spindle_frame(test_cell)
            real_axial_mean = np.nanmean(real_sf.axial, axis=1)
            real_radial_mean = np.nanmean(real_sf.radial, axis=1)

            if deterministic:
                # Single noise-free ODE rollout
                sim_chrom, sim_axial, sim_radial = _rollout_one(
                    test_cell, model, seed=0, deterministic=True,
                )
                sim_chroms = [sim_chrom]
                sim_axials = [sim_axial]
                sim_radials = [sim_radial]
            else:
                # Pre-generate seeds for reproducibility
                rep_seeds = [
                    int(rng.integers(0, np.iinfo(np.int64).max))
                    for _ in range(n_reps)
                ]

                # Simulate n_reps rollouts (parallel via joblib)
                results = Parallel(n_jobs=n_jobs, verbose=1)(
                    delayed(_rollout_one)(test_cell, model, s)
                    for s in rep_seeds
                )

                sim_chroms = [r[0] for r in results]
                sim_axials = [r[1] for r in results]
                sim_radials = [r[2] for r in results]

            # Per-rep path MSE (includes both drift bias and stochastic
            # variance; identical to ensemble_mse when deterministic)
            real_chroms = test_cell.chromosomes  # (T, 3, N)
            rep_mses = []
            for sc in sim_chroms:
                diff_3d = real_chroms - sc  # (T, 3, N)
                any_nan = np.any(np.isnan(diff_3d), axis=1)  # (T, N)
                sq_err = np.sum(diff_3d ** 2, axis=1)  # (T, N)
                sq_err[any_nan] = np.nan
                rep_mses.append(float(np.nanmean(sq_err)))
            path_mse[held_out_index] = float(np.mean(rep_mses))

            # Ensemble-mean metric: average simulated positions across reps,
            # then compare to reality.  Cancels model-side stochastic variance.
            # In deterministic mode this is identical to path_mse (one rep).
            stacked = np.stack(sim_chroms, axis=0)  # (n_reps, T, 3, N)
            ens_sum = np.nansum(stacked, axis=0)
            ens_count = np.sum(np.isfinite(stacked), axis=0)
            safe_count = np.where(ens_count > 0, ens_count, 1.0)
            ens_mean_chroms = np.where(
                ens_count > 0, ens_sum / safe_count, np.nan,
            )  # (T, 3, N)
            ens_diff = real_chroms - ens_mean_chroms
            ens_sq = np.sum(ens_diff ** 2, axis=1)
            ens_sq[np.any(np.isnan(ens_diff), axis=1)] = np.nan
            ensemble_mse[held_out_index] = float(np.nanmean(ens_sq))

            # Spindle-frame diagnostics
            sim_axial_mean = np.nanmean(
                np.stack([np.nanmean(ax, axis=1) for ax in sim_axials], axis=0),
                axis=0,
            )
            sim_radial_mean = np.nanmean(
                np.stack([np.nanmean(rad, axis=1) for rad in sim_radials], axis=0),
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
                [ax[-1][np.isfinite(ax[-1])] for ax in sim_axials]
            )
            sim_rad_valid = np.concatenate(
                [rad[-1][np.isfinite(rad[-1])] for rad in sim_radials]
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

                # Horizon-resolved ensemble MSE (3D position space)
                ens_diff_h = real_chroms[horizon] - ens_mean_chroms[horizon]  # (3, N)
                ens_sq_h = np.sum(ens_diff_h ** 2, axis=0)  # (N,)
                ens_sq_h[np.any(np.isnan(ens_diff_h), axis=0)] = np.nan
                horizon_ensemble_mse[held_out_index, horizon_index] = float(
                    np.nanmean(ens_sq_h)
                )

                # Horizon-resolved path MSE (average across reps)
                rep_h_mses = []
                for sc in sim_chroms:
                    diff_h = real_chroms[horizon] - sc[horizon]  # (3, N)
                    sq_h = np.sum(diff_h ** 2, axis=0)  # (N,)
                    sq_h[np.any(np.isnan(diff_h), axis=0)] = np.nan
                    rep_h_mses.append(float(np.nanmean(sq_h)))
                horizon_path_mse[held_out_index, horizon_index] = float(
                    np.mean(rep_h_mses)
                )

    return RolloutCVResult(
        horizons=horizon_values,
        path_mse=path_mse,
        ensemble_mse=ensemble_mse,
        axial_mse=axial_mse,
        radial_mse=radial_mse,
        endpoint_mean_error=endpoint_mean_error,
        final_axial_wasserstein=final_axial_wasserstein,
        final_radial_wasserstein=final_radial_wasserstein,
        horizon_errors=horizon_errors,
        horizon_path_mse=horizon_path_mse,
        horizon_ensemble_mse=horizon_ensemble_mse,
    )


@dataclass
class ForecastHorizonResult:
    """Rolling-window forecast horizon validation.

    For each horizon h, the model is re-initialized from the real positions at
    each frame t0, simulated forward h steps, and compared to reality at t0+h.
    This answers "over what timescales is the model accurate?" in contrast to
    the from-NEB rollout which answers "when does the model break down?"

    ``ensemble_mse``: shape ``(n_cells, n_horizons)`` -- MSE of the ensemble-mean
    forecast (averaged across reps before measuring error, cancelling stochastic
    variance).

    ``path_mse``: shape ``(n_cells, n_horizons)`` -- MSE of individual rollouts
    (includes stochastic variance).
    """

    horizons: np.ndarray
    ensemble_mse: np.ndarray
    path_mse: np.ndarray


def _forecast_window_one(test_cell, model, t0, h, seed, deterministic=False):
    """Simulate h steps from real positions at t0, return simulated chroms at t0+h."""
    from chromlearn.io.trajectory import get_partners
    from chromlearn.model_fitting.simulate import kernel_callables, simulate_trajectories

    rep_rng = None if deterministic else np.random.default_rng(seed)
    kernel_xx, kernel_xy = kernel_callables(model)
    partners = get_partners(test_cell, model.topology)

    # Slice partner positions for this window
    partner_window = partners[:, t0:t0 + h + 1, :]
    x0 = test_cell.chromosomes[t0].T  # (N, 3)

    traj = simulate_trajectories(
        kernel_xx=kernel_xx,
        kernel_xy=kernel_xy,
        partner_positions=partner_window,
        x0=x0,
        n_steps=h,
        dt=test_cell.dt,
        D_x=model.D_x,
        rng=rep_rng,
        deterministic=deterministic,
    )
    return traj[h]  # (3, N) at the forecast endpoint


def forecast_horizon_cross_validate(
    cells: list[TrimmedCell],
    config: FitConfig,
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20),
    n_reps: int = 8,
    rng: np.random.Generator | None = None,
    n_jobs: int = _DEFAULT_N_JOBS,
    deterministic: bool = False,
) -> ForecastHorizonResult:
    """Rolling-window forecast cross-validation.

    For each held-out cell and each horizon h, re-initialize from the real
    positions at every valid starting frame t0, simulate h steps forward
    (using real partner trajectories), and compare to reality at t0+h.

    This gives a clean forecast-horizon curve showing "over what timescales
    is the model accurate?" independent of trajectory position.

    When *deterministic* is True, a single noise-free ODE forecast replaces
    the stochastic replicate ensemble.  ``n_reps`` and ``rng`` are ignored.

    Uses common random numbers (seeded per fold) across topologies when
    called with the same rng seed.  Cell-level weighting: each cell
    contributes equally regardless of trajectory length.
    """
    if not cells:
        raise ValueError("forecast_horizon_cross_validate requires at least one cell.")
    if rng is None:
        rng = np.random.default_rng()

    horizon_values = np.array(sorted({int(h) for h in horizons if int(h) > 0}), dtype=int)
    n_cells = len(cells)
    max_h = int(horizon_values.max())

    ens_mse = np.full((n_cells, horizon_values.size), np.nan, dtype=np.float64)
    path_mse_arr = np.full((n_cells, horizon_values.size), np.nan, dtype=np.float64)

    for i in range(n_cells):
        train_cells = [c for j, c in enumerate(cells) if j != i]
        model = fit_model(train_cells, config)
        test_cell = cells[i]
        T = test_cell.chromosomes.shape[0]

        for hi, h in enumerate(horizon_values):
            if h >= T:
                continue

            # Valid starting frames: 0 to T-h-1
            n_windows = T - h
            eff_reps = 1 if deterministic else n_reps

            if deterministic:
                # Single ODE forecast per window -- no joblib overhead
                results = [
                    _forecast_window_one(test_cell, model, t0, h, seed=0,
                                         deterministic=True)
                    for t0 in range(n_windows)
                ]
            else:
                seeds = [int(rng.integers(0, np.iinfo(np.int64).max))
                         for _ in range(n_windows * eff_reps)]

                # Run all (window, rep) combinations in parallel
                jobs = []
                for t0 in range(n_windows):
                    for rep in range(eff_reps):
                        seed_idx = t0 * eff_reps + rep
                        jobs.append(delayed(_forecast_window_one)(
                            test_cell, model, t0, h, seeds[seed_idx]
                        ))

                results = Parallel(n_jobs=n_jobs, verbose=1)(jobs)

            # Reshape: (n_windows, eff_reps) each element is (3, N)
            window_path_mses = []
            window_ens_mses = []

            for t0 in range(n_windows):
                real_chroms_h = test_cell.chromosomes[t0 + h]  # (3, N)
                rep_sims = []
                rep_mses = []
                for rep in range(eff_reps):
                    sim_h = results[t0 * eff_reps + rep]  # (3, N)
                    rep_sims.append(sim_h)
                    diff = real_chroms_h - sim_h
                    sq = np.sum(diff ** 2, axis=0)
                    sq[np.any(np.isnan(diff), axis=0)] = np.nan
                    rep_mses.append(float(np.nanmean(sq)))

                window_path_mses.append(float(np.mean(rep_mses)))

                # Ensemble mean across reps (identical to single rep when deterministic)
                stacked = np.stack(rep_sims, axis=0)  # (eff_reps, 3, N)
                ens_count = np.sum(np.isfinite(stacked), axis=0)
                safe_count = np.where(ens_count > 0, ens_count, 1.0)
                ens_mean = np.where(ens_count > 0,
                                    np.nansum(stacked, axis=0) / safe_count,
                                    np.nan)
                ens_diff = real_chroms_h - ens_mean
                ens_sq = np.sum(ens_diff ** 2, axis=0)
                ens_sq[np.any(np.isnan(ens_diff), axis=0)] = np.nan
                window_ens_mses.append(float(np.nanmean(ens_sq)))

            # Average across windows (equal window weighting within each cell)
            path_mse_arr[i, hi] = float(np.nanmean(window_path_mses))
            ens_mse[i, hi] = float(np.nanmean(window_ens_mses))

    return ForecastHorizonResult(
        horizons=horizon_values,
        ensemble_mse=ens_mse,
        path_mse=path_mse_arr,
    )


def evaluate_all_loocv(
    cells: list[TrimmedCell],
    config: FitConfig,
    rollout_horizons: tuple[int, ...] = (1, 5, 10, 20),
    forecast_horizons: tuple[int, ...] = (1, 3, 5, 10, 20),
    compute_one_step: bool = True,
) -> tuple["CVResult | None", RolloutCVResult, "ForecastHorizonResult"]:
    """Single-pass LOOCV across 1-step CV, from-NEB rollout, and rolling-window
    forecast metrics.

    Fits the model ONCE per held-out cell (instead of three times across the
    three separate CV functions), then evaluates all three metrics against
    the same FittedModel.  Numerically equivalent to calling::

        cross_validate(cells, config)
        rollout_cross_validate(cells, config, horizons=rollout_horizons,
                               deterministic=True)
        forecast_horizon_cross_validate(cells, config, horizons=forecast_horizons,
                                        deterministic=True)

    independently.  Speedup ~3x on the fitting cost; lower wall-clock when
    rollout/forecast simulation dominates.  Deterministic mode only.

    Args:
        cells: Trimmed cell trajectories (LOOCV; one held-out cell per fold).
        config: Fitting configuration.
        rollout_horizons: Horizons (frames) for from-NEB rollout metrics.
        forecast_horizons: Horizons (frames) for rolling-window forecast.
        compute_one_step: If False, skip the 1-step CV residual block (returns
            None for the CVResult slot).

    Returns:
        Tuple ``(CVResult | None, RolloutCVResult, ForecastHorizonResult)``.
    """
    from chromlearn.io.trajectory import spindle_frame
    from chromlearn.model_fitting.basis import (
        BSplineBasis, HatBasis, make_basis_with_envelope,
    )
    from chromlearn.model_fitting.features import build_design_matrix
    from chromlearn.model_fitting.simulate import simulate_cell

    if not cells:
        raise ValueError("evaluate_all_loocv requires at least one cell.")
    n_cells = len(cells)

    rollout_horizon_values = np.array(
        sorted({int(h) for h in rollout_horizons if int(h) > 0}), dtype=int)
    forecast_horizon_values = np.array(
        sorted({int(h) for h in forecast_horizons if int(h) > 0}), dtype=int)
    if rollout_horizon_values.size == 0:
        raise ValueError("rollout_horizons must contain at least one positive integer.")
    if forecast_horizon_values.size == 0:
        raise ValueError("forecast_horizons must contain at least one positive integer.")

    # Build basis ONCE (shared across folds for the 1-step CV test design matrix)
    BasisClass = BSplineBasis if config.basis_type == "bspline" else HatBasis
    if _topology_has_chroms(config.topology):
        basis_xx = make_basis_with_envelope(
            BasisClass, config.r_min_xx, config.r_max_xx, config.n_basis_xx,
            envelope_r0=config.envelope_r0_xx, envelope_w=config.envelope_w_xx,
        )
    else:
        basis_xx = None
    basis_xy = BasisClass(config.r_min_xy, config.r_max_xy, config.n_basis_xy)

    # 1-step CV outputs
    cv_errors = np.full(n_cells, np.nan, dtype=np.float64)

    # Rollout outputs (per-cell)
    path_mse = np.full(n_cells, np.nan, dtype=np.float64)
    ensemble_mse_arr = np.full(n_cells, np.nan, dtype=np.float64)
    axial_mse = np.full(n_cells, np.nan, dtype=np.float64)
    radial_mse = np.full(n_cells, np.nan, dtype=np.float64)
    endpoint_mean_error = np.full(n_cells, np.nan, dtype=np.float64)
    final_axial_w = np.full(n_cells, np.nan, dtype=np.float64)
    final_radial_w = np.full(n_cells, np.nan, dtype=np.float64)
    horizon_errors = np.full((n_cells, rollout_horizon_values.size), np.nan, dtype=np.float64)
    horizon_path_mse = np.full((n_cells, rollout_horizon_values.size), np.nan, dtype=np.float64)
    horizon_ensemble_mse = np.full((n_cells, rollout_horizon_values.size), np.nan, dtype=np.float64)

    # Forecast outputs (per-cell, per-horizon)
    fc_ens_mse = np.full((n_cells, forecast_horizon_values.size), np.nan, dtype=np.float64)
    fc_path_mse = np.full((n_cells, forecast_horizon_values.size), np.nan, dtype=np.float64)

    for held_out_index in range(n_cells):
        train_cells = [c for j, c in enumerate(cells) if j != held_out_index]
        # SINGLE fit per fold — the dedup win.
        model = fit_model(train_cells, config)
        test_cell = cells[held_out_index]

        # ---- 1. one-step CV residual (matches cross_validate)
        if compute_one_step:
            G_test, V_test = build_design_matrix(
                [test_cell], basis_xx, basis_xy,
                basis_eval_mode=config.basis_eval_mode,
                topology=config.topology,
                r_cutoff_xx=config.r_cutoff_xx,
            )
            if G_test.size > 0:
                predictions = G_test @ model.theta
                cv_errors[held_out_index] = float(np.mean((V_test - predictions) ** 2))

        # ---- 2. From-NEB deterministic rollout (matches rollout_cross_validate, deterministic=True)
        _, sim_cell = simulate_cell(test_cell, model, rng=None, deterministic=True)
        sim_sf = spindle_frame(sim_cell)
        sim_chrom = sim_cell.chromosomes
        sim_axial = sim_sf.axial
        sim_radial = sim_sf.radial

        real_sf = spindle_frame(test_cell)
        real_axial_mean = np.nanmean(real_sf.axial, axis=1)
        real_radial_mean = np.nanmean(real_sf.radial, axis=1)
        real_chroms = test_cell.chromosomes

        # path MSE = ensemble MSE in deterministic mode (single rep)
        diff_3d = real_chroms - sim_chrom
        any_nan = np.any(np.isnan(diff_3d), axis=1)
        sq_err = np.sum(diff_3d ** 2, axis=1)
        sq_err[any_nan] = np.nan
        path_mse[held_out_index] = float(np.nanmean(sq_err))
        ensemble_mse_arr[held_out_index] = path_mse[held_out_index]

        sim_axial_mean = np.nanmean(sim_axial, axis=1)
        sim_radial_mean = np.nanmean(sim_radial, axis=1)
        axial_mse[held_out_index] = float(np.nanmean((real_axial_mean - sim_axial_mean) ** 2))
        radial_mse[held_out_index] = float(np.nanmean((real_radial_mean - sim_radial_mean) ** 2))
        endpoint_mean_error[held_out_index] = float(
            (real_axial_mean[-1] - sim_axial_mean[-1]) ** 2
            + (real_radial_mean[-1] - sim_radial_mean[-1]) ** 2
        )

        real_ax_valid = real_sf.axial[-1][np.isfinite(real_sf.axial[-1])]
        real_rad_valid = real_sf.radial[-1][np.isfinite(real_sf.radial[-1])]
        sim_ax_valid = sim_axial[-1][np.isfinite(sim_axial[-1])]
        sim_rad_valid = sim_radial[-1][np.isfinite(sim_radial[-1])]
        if real_ax_valid.size > 0 and sim_ax_valid.size > 0:
            final_axial_w[held_out_index] = float(stats.wasserstein_distance(real_ax_valid, sim_ax_valid))
        if real_rad_valid.size > 0 and sim_rad_valid.size > 0:
            final_radial_w[held_out_index] = float(stats.wasserstein_distance(real_rad_valid, sim_rad_valid))

        T_traj = real_axial_mean.size
        for hi, h in enumerate(rollout_horizon_values):
            if h >= T_traj:
                continue
            horizon_errors[held_out_index, hi] = float(
                (real_axial_mean[h] - sim_axial_mean[h]) ** 2
                + (real_radial_mean[h] - sim_radial_mean[h]) ** 2
            )
            ens_diff_h = real_chroms[h] - sim_chrom[h]
            ens_sq_h = np.sum(ens_diff_h ** 2, axis=0)
            ens_sq_h[np.any(np.isnan(ens_diff_h), axis=0)] = np.nan
            horizon_ensemble_mse[held_out_index, hi] = float(np.nanmean(ens_sq_h))
            horizon_path_mse[held_out_index, hi] = horizon_ensemble_mse[held_out_index, hi]

        # ---- 3. Rolling-window forecast (matches forecast_horizon_cross_validate, deterministic=True)
        T_cell = test_cell.chromosomes.shape[0]
        for hi, h in enumerate(forecast_horizon_values):
            if h >= T_cell:
                continue
            n_windows = T_cell - h
            window_mses = []
            for t0 in range(n_windows):
                sim_h = _forecast_window_one(
                    test_cell, model, t0, h, seed=0, deterministic=True,
                )
                real_h = test_cell.chromosomes[t0 + h]
                diff = real_h - sim_h
                sq = np.sum(diff ** 2, axis=0)
                sq[np.any(np.isnan(diff), axis=0)] = np.nan
                window_mses.append(float(np.nanmean(sq)))
            mean_mse = float(np.nanmean(window_mses))
            fc_path_mse[held_out_index, hi] = mean_mse
            fc_ens_mse[held_out_index, hi] = mean_mse  # det. mode == path

    cv_result = None
    if compute_one_step:
        cv_result = CVResult(
            held_out_errors=cv_errors,
            mean_error=float(np.nanmean(cv_errors)),
            fold_sd=float(np.nanstd(cv_errors)),
        )

    rollout_result = RolloutCVResult(
        horizons=rollout_horizon_values,
        path_mse=path_mse,
        ensemble_mse=ensemble_mse_arr,
        axial_mse=axial_mse,
        radial_mse=radial_mse,
        endpoint_mean_error=endpoint_mean_error,
        final_axial_wasserstein=final_axial_w,
        final_radial_wasserstein=final_radial_w,
        horizon_errors=horizon_errors,
        horizon_path_mse=horizon_path_mse,
        horizon_ensemble_mse=horizon_ensemble_mse,
    )

    forecast_result = ForecastHorizonResult(
        horizons=forecast_horizon_values,
        ensemble_mse=fc_ens_mse,
        path_mse=fc_path_mse,
    )

    return cv_result, rollout_result, forecast_result


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

    from chromlearn.model_fitting.basis import make_basis_with_envelope

    BasisClass = BSplineBasis if config.basis_type == "bspline" else HatBasis
    if _topology_has_chroms(config.topology):
        basis_xx = make_basis_with_envelope(
            BasisClass, config.r_min_xx, config.r_max_xx, config.n_basis_xx,
            envelope_r0=config.envelope_r0_xx, envelope_w=config.envelope_w_xx,
        )
    else:
        basis_xx = None
    basis_xy = BasisClass(config.r_min_xy, config.r_max_xy, config.n_basis_xy)

    G, V = build_design_matrix(
        cells, basis_xx, basis_xy,
        basis_eval_mode=config.basis_eval_mode,
        topology=config.topology,
        r_cutoff_xx=config.r_cutoff_xx,
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
            r_cutoff_xx=config.r_cutoff_xx,
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
            r_cutoff_xx=config.r_cutoff_xx,
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
        r_cutoff_xx=config.r_cutoff_xx,
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
