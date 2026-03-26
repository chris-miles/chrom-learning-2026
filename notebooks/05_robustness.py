# %% [markdown]
# # 05 -- Robustness & Hyperparameter Sensitivity
#
# Sensitivity of the winning topology from Notebook 04 to: basis size,
# regularisation strengths, estimator mode, endpoint method, and diffusion
# estimation mode.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chromlearn import find_repo_root

ROOT = find_repo_root(Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chromlearn.io.catalog import load_condition
from chromlearn.io.trajectory import trim_trajectory
from chromlearn.model_fitting import FitConfig
from chromlearn.model_fitting.basis import BSplineBasis
from chromlearn.model_fitting.diffusion import local_diffusion_estimates
from chromlearn.model_fitting.features import build_design_matrix
from chromlearn.model_fitting.fit import (
    CVResult,
    RolloutCVResult,
    cross_validate,
    fit_kernels,
    fit_model,
    rollout_cross_validate,
)
from chromlearn.model_fitting.model import FittedModel
from chromlearn.model_fitting.plotting import plot_cv_curve, plot_kernels

plt.rcParams["figure.dpi"] = 110

# %% [markdown]
# ## Setup

# %%
# Update this after consulting Notebook 04 results.
WINNING_TOPOLOGY = "poles"
BASE_FRAC = 1.0 / 3.0

CONDITION = "rpe18_ctr"
cells_raw = load_condition(CONDITION)
cells = [trim_trajectory(c, method="neb_ao_frac", frac=BASE_FRAC) for c in cells_raw]
print(f"Loaded {len(cells)} rpe18_ctr cells (trimmed to neb_ao_frac={BASE_FRAC:.3f} window).")

BASE_CONFIG = FitConfig(
    topology=WINNING_TOPOLOGY,
    n_basis_xx=10,
    n_basis_xy=10,
    lambda_ridge=1e-3,
    lambda_rough=1.0,
    basis_eval_mode="ito",
    endpoint_method="neb_ao_frac",
    endpoint_frac=BASE_FRAC,
    diffusion_mode="msd",
    dt=5.0,
)

ROLLOUT_REPS = 32
ROLLOUT_HORIZONS = (1, 5, 10, 20)
ROLLOUT_BASE_SEED = 20260325


def rollout_path_score(result: RolloutCVResult) -> float:
    """Primary rollout score: per-chromosome 3D path MSE on held-out cells."""
    return float(np.nanmean(result.path_mse))


def rollout_path_se(result: RolloutCVResult) -> float:
    """SE across held-out cells for the path MSE."""
    valid = result.path_mse[np.isfinite(result.path_mse)]
    if valid.size == 0:
        return np.inf
    return float(np.nanstd(valid) / np.sqrt(valid.size))


def rollout_ensemble_score(result: RolloutCVResult) -> float:
    """Ensemble-mean MSE: averages sim positions across reps before
    comparing to reality, cancelling model-side stochastic variance."""
    return float(np.nanmean(result.ensemble_mse))


def rollout_ensemble_se(result: RolloutCVResult) -> float:
    """SE across held-out cells for the ensemble MSE."""
    valid = result.ensemble_mse[np.isfinite(result.ensemble_mse)]
    if valid.size == 0:
        return np.inf
    return float(np.nanstd(valid) / np.sqrt(valid.size))


def rollout_w1_score(result: RolloutCVResult) -> float:
    """Final-frame distribution mismatch: axial W1 + radial W1."""
    return float(
        np.nanmean(result.final_axial_wasserstein)
        + np.nanmean(result.final_radial_wasserstein)
    )


def run_rollout_cv(
    cells_in: list,
    cfg: FitConfig,
    seed: int,
    k_folds: int | None = None,
) -> RolloutCVResult:
    """Deterministic rollout CV helper for sweep comparisons."""
    return rollout_cross_validate(
        cells_in,
        cfg,
        n_reps=ROLLOUT_REPS,
        horizons=ROLLOUT_HORIZONS,
        rng=np.random.default_rng(seed),
        k_folds=k_folds,
    )

# %% [markdown]
# ## Sweep 0: Rollout replicate convergence
#
# Before trusting any rollout-based comparison, check how many replicates are
# needed for the path MSE and W1 scores to stabilize.  We fix the baseline
# config, run a large number of rollout reps, and plot the cumulative mean
# as a function of reps used.

# %%
CONVERGENCE_MAX_REPS = 100
CONVERGENCE_SEED = 20260325

print(f"Running convergence check: {CONVERGENCE_MAX_REPS} rollout reps on baseline config...")
print(f"  ({len(cells)} folds x {CONVERGENCE_MAX_REPS} reps = {len(cells) * CONVERGENCE_MAX_REPS} simulations)")

# We need per-fold, per-rep scores, so we run rollout_cross_validate once
# with many reps and then compute cumulative statistics from the raw arrays.
# But rollout_cross_validate only returns fold-level aggregates, so we need
# to go one level deeper here.

from scipy import stats  # noqa: E402

from chromlearn.io.trajectory import spindle_frame  # noqa: E402
from chromlearn.model_fitting.simulate import simulate_cell  # noqa: E402

conv_rng = np.random.default_rng(CONVERGENCE_SEED)

# Per-fold, per-rep raw scores
n_conv_cells = len(cells)
conv_path_mse = np.full((n_conv_cells, CONVERGENCE_MAX_REPS), np.nan)
conv_w1_axial = np.full((n_conv_cells, CONVERGENCE_MAX_REPS), np.nan)
conv_w1_radial = np.full((n_conv_cells, CONVERGENCE_MAX_REPS), np.nan)
# Ensemble-mean MSE as a function of N reps (computed from stored trajectories)
conv_ensemble_mse = np.full((n_conv_cells, CONVERGENCE_MAX_REPS), np.nan)

for fold_idx in range(n_conv_cells):
    train_cells = [c for i, c in enumerate(cells) if i != fold_idx]
    test_cell = cells[fold_idx]
    model_conv = fit_model(train_cells, BASE_CONFIG)

    real_chroms = test_cell.chromosomes  # (T, 3, N)
    real_sf = spindle_frame(test_cell)
    real_ax_final = real_sf.axial[-1]
    real_ax_final = real_ax_final[np.isfinite(real_ax_final)]
    real_rad_final = real_sf.radial[-1]
    real_rad_final = real_rad_final[np.isfinite(real_rad_final)]

    print(f"  Fold {fold_idx+1}/{n_conv_cells} ({test_cell.cell_id})...", flush=True)

    # Running sum for ensemble mean (avoids storing all trajectories).
    # Use nan_to_num so a single NaN rep doesn't poison all subsequent means.
    running_sum = np.zeros_like(real_chroms, dtype=np.float64)
    running_count = np.zeros_like(real_chroms, dtype=np.float64)

    for rep in range(CONVERGENCE_MAX_REPS):
        rep_rng = np.random.default_rng(
            int(conv_rng.integers(0, np.iinfo(np.int64).max))
        )
        _, sim_cell = simulate_cell(test_cell, model_conv, rng=rep_rng)

        # Per-rep path MSE
        diff_3d = real_chroms - sim_cell.chromosomes
        any_nan = np.any(np.isnan(diff_3d), axis=1)
        sq_err = np.sum(diff_3d ** 2, axis=1)
        sq_err[any_nan] = np.nan
        conv_path_mse[fold_idx, rep] = float(np.nanmean(sq_err))

        # Ensemble-mean MSE: update running sum, compute MSE of
        # ensemble mean using first (rep+1) replicates.
        # NaN-safe: only accumulate finite values.
        valid_mask = np.isfinite(sim_cell.chromosomes)
        running_sum += np.where(valid_mask, sim_cell.chromosomes, 0.0)
        running_count += valid_mask.astype(np.float64)
        safe_count = np.where(running_count > 0, running_count, 1.0)
        ens_mean = np.where(running_count > 0, running_sum / safe_count, np.nan)
        ens_diff = real_chroms - ens_mean
        ens_sq = np.sum(ens_diff ** 2, axis=1)
        ens_sq[np.any(np.isnan(ens_diff), axis=1)] = np.nan
        conv_ensemble_mse[fold_idx, rep] = float(np.nanmean(ens_sq))

        # Per-rep W1
        sim_sf = spindle_frame(sim_cell)
        sim_ax = sim_sf.axial[-1]
        sim_ax = sim_ax[np.isfinite(sim_ax)]
        sim_rad = sim_sf.radial[-1]
        sim_rad = sim_rad[np.isfinite(sim_rad)]
        if real_ax_final.size > 0 and sim_ax.size > 0:
            conv_w1_axial[fold_idx, rep] = float(
                stats.wasserstein_distance(real_ax_final, sim_ax)
            )
        if real_rad_final.size > 0 and sim_rad.size > 0:
            conv_w1_radial[fold_idx, rep] = float(
                stats.wasserstein_distance(real_rad_final, sim_rad)
            )

print("Convergence simulations complete.")

# %%
# Cumulative mean across reps (averaged over folds)
rep_counts = np.arange(1, CONVERGENCE_MAX_REPS + 1)

cum_path_mse = np.nancumsum(conv_path_mse, axis=1) / rep_counts[np.newaxis, :]
cum_path_mse_mean = np.nanmean(cum_path_mse, axis=0)

cum_w1_total = np.nancumsum(conv_w1_axial + conv_w1_radial, axis=1) / rep_counts[np.newaxis, :]
cum_w1_mean = np.nanmean(cum_w1_total, axis=0)

cell_ids = [c.cell_id for c in cells]

# Identify outlier: cell with highest converged path MSE
final_path_per_cell = cum_path_mse[:, -1]
outlier_idx = int(np.nanargmax(final_path_per_cell))
outlier_id = cell_ids[outlier_idx]
print(f"\nPer-cell converged path MSE (at {CONVERGENCE_MAX_REPS} reps):")
for i, cid in enumerate(cell_ids):
    flag = " <-- outlier" if i == outlier_idx else ""
    print(f"  {cid:20s}: {final_path_per_cell[i]:.4e}{flag}")

fig_conv, axes_conv = plt.subplots(1, 3, figsize=(18, 4.5))

# Path MSE convergence (cumulative mean of per-rep scores)
for fold_idx in range(n_conv_cells):
    is_outlier = fold_idx == outlier_idx
    axes_conv[0].plot(
        rep_counts, cum_path_mse[fold_idx],
        color="C3" if is_outlier else "0.7",
        linewidth=1.4 if is_outlier else 0.6,
        alpha=1.0 if is_outlier else 0.5,
        label=cell_ids[fold_idx] if is_outlier else None,
    )
axes_conv[0].plot(rep_counts, cum_path_mse_mean, color="C0", linewidth=2, label="mean (all cells)")
cum_path_mse_excl = np.nanmean(
    np.delete(cum_path_mse, outlier_idx, axis=0), axis=0,
)
axes_conv[0].plot(
    rep_counts, cum_path_mse_excl, color="C0", linewidth=2, linestyle="--",
    label=f"mean (excl. {outlier_id})",
)
axes_conv[0].set_xlabel("Number of rollout replicates")
axes_conv[0].set_ylabel("Cumulative mean path MSE")
axes_conv[0].set_title("Path MSE (per-rep avg)\n~96% noise floor, flat by N=5")
axes_conv[0].legend(fontsize=7)

# Ensemble-mean MSE convergence (average positions, then compare)
ens_mean_across_folds = np.nanmean(conv_ensemble_mse, axis=0)
for fold_idx in range(n_conv_cells):
    is_outlier = fold_idx == outlier_idx
    axes_conv[1].plot(
        rep_counts, conv_ensemble_mse[fold_idx],
        color="C3" if is_outlier else "0.7",
        linewidth=1.4 if is_outlier else 0.6,
        alpha=1.0 if is_outlier else 0.5,
        label=cell_ids[fold_idx] if is_outlier else None,
    )
axes_conv[1].plot(rep_counts, ens_mean_across_folds, color="C2", linewidth=2, label="mean (all cells)")
ens_excl = np.nanmean(
    np.delete(conv_ensemble_mse, outlier_idx, axis=0), axis=0,
)
axes_conv[1].plot(
    rep_counts, ens_excl, color="C2", linewidth=2, linestyle="--",
    label=f"mean (excl. {outlier_id})",
)
axes_conv[1].set_xlabel("Number of rollout replicates")
axes_conv[1].set_ylabel("Ensemble-mean MSE")
axes_conv[1].set_title("Ensemble MSE (avg positions, then compare)\ndrops as noise cancels, stabilizes ~16-32 reps")
axes_conv[1].legend(fontsize=7)

# W1 convergence
cum_w1_total = np.nancumsum(conv_w1_axial + conv_w1_radial, axis=1) / rep_counts[np.newaxis, :]
cum_w1_mean = np.nanmean(cum_w1_total, axis=0)
for fold_idx in range(n_conv_cells):
    is_outlier = fold_idx == outlier_idx
    axes_conv[2].plot(
        rep_counts, cum_w1_total[fold_idx],
        color="C3" if is_outlier else "0.7",
        linewidth=1.4 if is_outlier else 0.6,
        alpha=1.0 if is_outlier else 0.5,
        label=cell_ids[fold_idx] if is_outlier else None,
    )
axes_conv[2].plot(rep_counts, cum_w1_mean, color="C1", linewidth=2, label="mean (all cells)")
cum_w1_excl = np.nanmean(
    np.delete(cum_w1_total, outlier_idx, axis=0), axis=0,
)
axes_conv[2].plot(
    rep_counts, cum_w1_excl, color="C1", linewidth=2, linestyle="--",
    label=f"mean (excl. {outlier_id})",
)
axes_conv[2].set_xlabel("Number of rollout replicates")
axes_conv[2].set_ylabel("Cumulative mean W1 (axial + radial)")
axes_conv[2].set_title("W1 convergence\neach line = one held-out cell")
axes_conv[2].legend(fontsize=7)

fig_conv.suptitle("Sweep 0 — Rollout replicate convergence (baseline config)", fontsize=12)
fig_conv.tight_layout()
plt.show()

# Tail variation diagnostic
tail_start = int(0.8 * CONVERGENCE_MAX_REPS)
tail_variation_path = (
    np.ptp(cum_path_mse_mean[tail_start:]) / cum_path_mse_mean[-1] * 100
)
tail_variation_ens = (
    np.ptp(ens_mean_across_folds[tail_start:]) / ens_mean_across_folds[-1] * 100
)
tail_variation_w1 = (
    np.ptp(cum_w1_mean[tail_start:]) / cum_w1_mean[-1] * 100
)
print(f"\nTail variation (last 20% of reps):")
print(f"  path MSE:     {tail_variation_path:.1f}% of final value")
print(f"  ensemble MSE: {tail_variation_ens:.1f}% of final value")
print(f"  W1:           {tail_variation_w1:.1f}% of final value")
print(f"\nConverged values (at {CONVERGENCE_MAX_REPS} reps):")
print(f"  path MSE     = {cum_path_mse_mean[-1]:.4e}")
print(f"  ensemble MSE = {ens_mean_across_folds[-1]:.4e}")
print(f"  W1           = {cum_w1_mean[-1]:.4e}")

# %% [markdown]
# ## Sweep 1: Joint (n\_basis, lambda\_ridge, lambda\_rough) grid
#
# The old one-at-a-time sweeps can miss interactions (e.g., more basis functions
# may need stronger regularization). Instead we do a full 3D grid and visualize
# 2D slices through the best value of the third parameter.
#
# We use the ensemble-mean MSE as the primary selection target (averages
# simulated positions across reps before comparing to reality, cancelling
# stochastic noise).  Path MSE, W1, and one-step CV MSE are recorded
# alongside for comparison.

# %%
from itertools import combinations as _combinations, product as _product  # noqa: E402

N_BASIS_GRID = [4, 8, 16, 32, 64]
LAMBDA_RIDGE_GRID = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
LAMBDA_ROUGH_GRID = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])

grid_configs = list(_product(N_BASIS_GRID, LAMBDA_RIDGE_GRID, LAMBDA_ROUGH_GRID))
n_grid = len(grid_configs)
print(f"Joint grid: {len(N_BASIS_GRID)} x {len(LAMBDA_RIDGE_GRID)} x {len(LAMBDA_ROUGH_GRID)} = {n_grid} configs")


def _run_one_grid_point(
    cells_in,
    base_cfg,
    rollout_base_seed,
    rollout_reps,
    rollout_horizons,
    k_folds,
    compute_one_step_cv,
    idx,
    nb,
    lr,
    lro,
):
    """Worker function for parallel grid sweep (all args explicit for joblib)."""
    cfg = FitConfig(
        topology=base_cfg.topology,
        n_basis_xx=nb,
        n_basis_xy=nb,
        r_min_xx=base_cfg.r_min_xx,
        r_max_xx=base_cfg.r_max_xx,
        r_min_xy=base_cfg.r_min_xy,
        r_max_xy=base_cfg.r_max_xy,
        basis_type=base_cfg.basis_type,
        lambda_ridge=lr,
        lambda_rough=lro,
        basis_eval_mode=base_cfg.basis_eval_mode,
        endpoint_method=base_cfg.endpoint_method,
        endpoint_frac=base_cfg.endpoint_frac,
        diffusion_mode=base_cfg.diffusion_mode,
        dt=base_cfg.dt,
    )
    cv_result = (
        cross_validate(cells_in, cfg, k_folds=k_folds)
        if compute_one_step_cv else None
    )
    rollout_result = rollout_cross_validate(
        cells_in, cfg,
        n_reps=rollout_reps,
        horizons=rollout_horizons,
        rng=np.random.default_rng(rollout_base_seed + idx),
        k_folds=k_folds,
    )
    return (nb, lr, lro), cv_result, rollout_result


import os  # noqa: E402

from joblib import Parallel, delayed  # noqa: E402

N_WORKERS = min(os.cpu_count() or 1, n_grid)


def _run_grid_sweep(
    cells_in,
    base_cfg,
    grid_configs_in,
    rollout_base_seed,
    rollout_reps,
    rollout_horizons,
    *,
    k_folds: int | None,
    label: str,
    compute_one_step_cv: bool = True,
    print_config_scores: bool = True,
    joblib_verbose: int = 10,
    announce: bool = True,
) -> tuple[dict[tuple, CVResult], dict[tuple, RolloutCVResult]]:
    """Run the full 3D grid under a chosen CV design."""
    if announce:
        print(f"Running {label} grid sweep with {N_WORKERS} parallel workers (joblib/loky)...")
    grid_cv_local: dict[tuple, CVResult] = {}
    grid_rollout_local: dict[tuple, RolloutCVResult] = {}

    results = Parallel(n_jobs=N_WORKERS, verbose=joblib_verbose)(
        delayed(_run_one_grid_point)(
            cells_in, base_cfg, rollout_base_seed, rollout_reps, rollout_horizons,
            k_folds, compute_one_step_cv, idx, nb, lr, lro,
        )
        for idx, (nb, lr, lro) in enumerate(grid_configs_in, start=1)
    )

    for key, cv_result, rollout_result in results:
        if cv_result is not None:
            grid_cv_local[key] = cv_result
        grid_rollout_local[key] = rollout_result
        if print_config_scores:
            nb, lr, lro = key
            score_line = (
                f"  n_basis={nb}, ridge={lr:.0e}, rough={lro:.0e}"
                f"  ens_MSE={rollout_ensemble_score(rollout_result):.4e}"
                f"  path_MSE={rollout_path_score(rollout_result):.4e}"
                f"  W1={rollout_w1_score(rollout_result):.4e}"
            )
            if cv_result is not None:
                score_line = (
                    f"  n_basis={nb}, ridge={lr:.0e}, rough={lro:.0e}"
                    f"  1-step={cv_result.mean_error:.4e}"
                    f"  ens_MSE={rollout_ensemble_score(rollout_result):.4e}"
                    f"  path_MSE={rollout_path_score(rollout_result):.4e}"
                    f"  W1={rollout_w1_score(rollout_result):.4e}"
                )
            print(score_line)

    if announce:
        print(f"{label} grid sweep complete: {len(grid_rollout_local)} configs evaluated.")
    return grid_cv_local, grid_rollout_local


grid_cv, grid_rollout = _run_grid_sweep(
    cells,
    BASE_CONFIG,
    grid_configs,
    ROLLOUT_BASE_SEED,
    ROLLOUT_REPS,
    ROLLOUT_HORIZONS,
    k_folds=None,
    label="LOO",
)

# %%
# Primary selection: ensemble-mean MSE
best_key = min(grid_configs, key=lambda k: rollout_ensemble_score(grid_rollout[k]))
best_n_basis_rollout, best_ridge_rollout, best_rough_rollout = best_key
best_key_cv = min(grid_configs, key=lambda k: grid_cv[k].mean_error)
best_key_path = min(grid_configs, key=lambda k: rollout_path_score(grid_rollout[k]))

print(f"\nBest by ensemble MSE:     n_basis={best_key[0]}, "
      f"lambda_ridge={best_key[1]:.2e}, lambda_rough={best_key[2]:.2e}  "
      f"(ens MSE = {rollout_ensemble_score(grid_rollout[best_key]):.4e})")
print(f"Best by rollout path MSE: n_basis={best_key_path[0]}, "
      f"lambda_ridge={best_key_path[1]:.2e}, lambda_rough={best_key_path[2]:.2e}  "
      f"(path MSE = {rollout_path_score(grid_rollout[best_key_path]):.4e})")
print(f"Best by 1-step CV:        n_basis={best_key_cv[0]}, "
      f"lambda_ridge={best_key_cv[1]:.2e}, lambda_rough={best_key_cv[2]:.2e}  "
      f"(CV MSE = {grid_cv[best_key_cv].mean_error:.4e})")

# %%
# 2D slice heatmaps through the best value of the third parameter

def _grid_slice_2d(
    fixed_param: str,
    fixed_val,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    results: dict,
    score_fn,
) -> np.ndarray:
    """Extract a 2D score matrix from the grid results."""
    mat = np.full((len(y_vals), len(x_vals)), np.nan)
    for xi, xv in enumerate(x_vals):
        for yi, yv in enumerate(y_vals):
            if fixed_param == "n_basis":
                key = (fixed_val, xv, yv)
            elif fixed_param == "lambda_ridge":
                key = (xv, fixed_val, yv)
            else:
                key = (xv, yv, fixed_val)
            if key in results:
                mat[yi, xi] = score_fn(results[key])
    return mat


fig_slices, axes_slices = plt.subplots(1, 3, figsize=(18, 5))

# Slice 1: ridge vs rough at best n_basis
mat_rr = _grid_slice_2d(
    "n_basis", best_n_basis_rollout,
    LAMBDA_RIDGE_GRID, LAMBDA_ROUGH_GRID, grid_rollout, rollout_ensemble_score,
)
im0 = axes_slices[0].imshow(
    mat_rr, origin="lower", aspect="auto",
    extent=[
        np.log10(LAMBDA_RIDGE_GRID[0]), np.log10(LAMBDA_RIDGE_GRID[-1]),
        np.log10(LAMBDA_ROUGH_GRID[0]), np.log10(LAMBDA_ROUGH_GRID[-1]),
    ],
)
axes_slices[0].set_xlabel("log10(lambda_ridge)")
axes_slices[0].set_ylabel("log10(lambda_rough)")
axes_slices[0].set_title(f"Ensemble MSE at n_basis={best_n_basis_rollout}")
fig_slices.colorbar(im0, ax=axes_slices[0], shrink=0.8)

# Slice 2: n_basis vs ridge at best rough
mat_nr = _grid_slice_2d(
    "lambda_rough", best_rough_rollout,
    np.array(N_BASIS_GRID, dtype=float), LAMBDA_RIDGE_GRID, grid_rollout, rollout_ensemble_score,
)
im1 = axes_slices[1].imshow(
    mat_nr, origin="lower", aspect="auto",
    extent=[
        N_BASIS_GRID[0], N_BASIS_GRID[-1],
        np.log10(LAMBDA_RIDGE_GRID[0]), np.log10(LAMBDA_RIDGE_GRID[-1]),
    ],
)
axes_slices[1].set_xlabel("n_basis")
axes_slices[1].set_ylabel("log10(lambda_ridge)")
axes_slices[1].set_title(f"Ensemble MSE at lambda_rough={best_rough_rollout:.0e}")
fig_slices.colorbar(im1, ax=axes_slices[1], shrink=0.8)

# Slice 3: n_basis vs rough at best ridge
mat_nro = _grid_slice_2d(
    "lambda_ridge", best_ridge_rollout,
    np.array(N_BASIS_GRID, dtype=float), LAMBDA_ROUGH_GRID, grid_rollout, rollout_ensemble_score,
)
im2 = axes_slices[2].imshow(
    mat_nro, origin="lower", aspect="auto",
    extent=[
        N_BASIS_GRID[0], N_BASIS_GRID[-1],
        np.log10(LAMBDA_ROUGH_GRID[0]), np.log10(LAMBDA_ROUGH_GRID[-1]),
    ],
)
axes_slices[2].set_xlabel("n_basis")
axes_slices[2].set_ylabel("log10(lambda_rough)")
axes_slices[2].set_title(f"Ensemble MSE at lambda_ridge={best_ridge_rollout:.0e}")
fig_slices.colorbar(im2, ax=axes_slices[2], shrink=0.8)

fig_slices.suptitle(
    f"Sweep 1 -- Joint hyperparameter grid: 2D slices through best 3rd param\n"
    f"({ROLLOUT_REPS} rollout reps/fold)",
    fontsize=12,
)
fig_slices.tight_layout()
plt.show()

# %% [markdown]
# ### 1D marginal views
#
# For each hyperparameter, show the best ensemble MSE achieved at each
# value (minimizing over the other two parameters). This gives the envelope
# of the 3D grid, analogous to the old 1D sweeps but without fixing the
# other parameters at possibly suboptimal values.

# %%
def _marginal_stats(scores, configs, param_idx, param_vals):
    """Min, mean, max of scores grouped by one hyperparameter axis."""
    lo, mid, hi = [], [], []
    for v in param_vals:
        group = [scores[k] for k in configs if k[param_idx] == v]
        lo.append(min(group))
        mid.append(np.mean(group))
        hi.append(max(group))
    return np.array(lo), np.array(mid), np.array(hi)


def _plot_marginal(ax, x, lo, mid, hi, color, xlabel, ylabel, title, xlog=False):
    ax.fill_between(x, lo, hi, alpha=0.12, color=color)
    ax.plot(x, hi, "^-", color=color, linewidth=0.8, markersize=4, alpha=0.4, label="max")
    ax.plot(x, mid, "s-", color=color, linewidth=1.2, markersize=5, alpha=0.6, label="mean")
    ax.plot(x, lo, "o-", color=color, linewidth=2.2, markersize=6, label="min (envelope)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)
    if xlog:
        ax.set_xscale("log")


grid_scores_loo = {k: rollout_ensemble_score(grid_rollout[k]) for k in grid_configs}

nb_lo, nb_mid, nb_hi = _marginal_stats(grid_scores_loo, grid_configs, 0, N_BASIS_GRID)
lr_lo, lr_mid, lr_hi = _marginal_stats(grid_scores_loo, grid_configs, 1, LAMBDA_RIDGE_GRID)
lro_lo, lro_mid, lro_hi = _marginal_stats(grid_scores_loo, grid_configs, 2, LAMBDA_ROUGH_GRID)

# Keep nb_best etc. for re-use in 1b comparison
nb_best, lr_best, lro_best = nb_lo, lr_lo, lro_lo

fig_marginal, axes_m = plt.subplots(1, 3, figsize=(16, 4))
_plot_marginal(axes_m[0], N_BASIS_GRID, nb_lo, nb_mid, nb_hi, "C0",
               "n_basis", "Ensemble MSE", "n_basis marginal")
axes_m[0].set_xticks(N_BASIS_GRID)
_plot_marginal(axes_m[1], LAMBDA_RIDGE_GRID, lr_lo, lr_mid, lr_hi, "C1",
               "lambda_ridge", "Ensemble MSE", "lambda_ridge marginal", xlog=True)
_plot_marginal(axes_m[2], LAMBDA_ROUGH_GRID, lro_lo, lro_mid, lro_hi, "C2",
               "lambda_rough", "Ensemble MSE", "lambda_rough marginal", xlog=True)

fig_marginal.suptitle("1D marginals from the joint grid (mean with min-max band)", fontsize=12)
fig_marginal.tight_layout()
plt.show()

# %% [markdown]
# ### Kernel shape stability across the grid
#
# Fit each of the 144 grid configs on all cells and plot the resulting
# chrom-pole kernel.  The spaghetti cloud shows the full variation; the
# marginal slices color by the swept parameter.

# %%
from matplotlib.colors import LogNorm  # noqa: E402

print("Fitting models for all 144 grid configs (all cells, no CV)...")
grid_models: dict[tuple, FittedModel] = {}
for nb, lr, lro in grid_configs:
    cfg = FitConfig(
        topology=BASE_CONFIG.topology,
        n_basis_xx=nb,
        n_basis_xy=nb,
        r_min_xx=BASE_CONFIG.r_min_xx,
        r_max_xx=BASE_CONFIG.r_max_xx,
        r_min_xy=BASE_CONFIG.r_min_xy,
        r_max_xy=BASE_CONFIG.r_max_xy,
        basis_type=BASE_CONFIG.basis_type,
        lambda_ridge=lr,
        lambda_rough=lro,
        basis_eval_mode=BASE_CONFIG.basis_eval_mode,
        endpoint_method=BASE_CONFIG.endpoint_method,
        endpoint_frac=BASE_CONFIG.endpoint_frac,
        diffusion_mode=BASE_CONFIG.diffusion_mode,
        dt=BASE_CONFIG.dt,
    )
    grid_models[(nb, lr, lro)] = fit_model(cells, cfg)
print(f"  Done: {len(grid_models)} models fitted.")

# %%
# Spaghetti cloud: all 144 kernels, best highlighted
_topology_has_chroms_grid = WINNING_TOPOLOGY in ("poles_and_chroms", "center_and_chroms")
_n_kernel_panels = 2 if _topology_has_chroms_grid else 1
_N_EVAL = 200

fig_spaghetti, ax_sp = plt.subplots(
    1, _n_kernel_panels, figsize=(6 * _n_kernel_panels, 4.5), squeeze=False,
)

for key in grid_configs:
    model = grid_models[key]
    if _topology_has_chroms_grid:
        r_xx = np.linspace(model.basis_xx.r_min, model.basis_xx.r_max, _N_EVAL)
        f_xx = model.basis_xx.evaluate(r_xx) @ model.theta_xx
        ax_sp[0, 0].plot(r_xx, f_xx, color="0.75", linewidth=0.4, alpha=0.5)
    r_xy = np.linspace(model.basis_xy.r_min, model.basis_xy.r_max, _N_EVAL)
    f_xy = model.basis_xy.evaluate(r_xy) @ model.theta_xy
    ax_sp[0, _n_kernel_panels - 1].plot(
        r_xy, f_xy, color="0.75", linewidth=0.4, alpha=0.5,
    )

# Best config in bold
best_model = grid_models[best_key]
if _topology_has_chroms_grid:
    r_xx = np.linspace(best_model.basis_xx.r_min, best_model.basis_xx.r_max, _N_EVAL)
    f_xx = best_model.basis_xx.evaluate(r_xx) @ best_model.theta_xx
    ax_sp[0, 0].plot(r_xx, f_xx, color="C3", linewidth=2.5, label="best config")
    ax_sp[0, 0].axhline(0, color="0.5", linestyle="--", linewidth=0.6)
    ax_sp[0, 0].set_xlabel("Distance (um)")
    ax_sp[0, 0].set_ylabel("Force")
    ax_sp[0, 0].set_title("Chrom-chrom kernel")
    ax_sp[0, 0].legend(fontsize=8)

r_xy = np.linspace(best_model.basis_xy.r_min, best_model.basis_xy.r_max, _N_EVAL)
f_xy = best_model.basis_xy.evaluate(r_xy) @ best_model.theta_xy
ax_xy = ax_sp[0, _n_kernel_panels - 1]
ax_xy.plot(r_xy, f_xy, color="C3", linewidth=2.5, label="best config")
ax_xy.axhline(0, color="0.5", linestyle="--", linewidth=0.6)
ax_xy.set_xlabel("Distance (um)")
ax_xy.set_ylabel("Force")
ax_xy.set_title("Chrom-pole kernel")
ax_xy.legend(fontsize=8)

fig_spaghetti.suptitle(
    "Sweep 1 — Kernel shape across all 144 grid configs (gray) vs best (red)",
    fontsize=12,
)
fig_spaghetti.tight_layout()
plt.show()

# %%
# 1D marginal slices: sweep one parameter, fix the other two at best
_param_slices = [
    ("n_basis", 0, N_BASIS_GRID, best_ridge_rollout, best_rough_rollout, "viridis", False),
    ("lambda_ridge", 1, LAMBDA_RIDGE_GRID, best_n_basis_rollout, best_rough_rollout, "plasma", True),
    ("lambda_rough", 2, LAMBDA_ROUGH_GRID, best_n_basis_rollout, best_ridge_rollout, "cividis", True),
]

fig_kslice, axes_ks = plt.subplots(
    1, len(_param_slices), figsize=(6 * len(_param_slices), 4.5),
)

for ax, (pname, pidx, pvals, fix1, fix2, cmap_name, is_log) in zip(axes_ks, _param_slices):
    cmap = plt.get_cmap(cmap_name)
    if is_log:
        norm = LogNorm(vmin=pvals[0], vmax=pvals[-1])
    else:
        norm = plt.Normalize(vmin=pvals[0], vmax=pvals[-1])

    for pv in pvals:
        if pidx == 0:
            key = (pv, fix1, fix2)
        elif pidx == 1:
            key = (fix1, pv, fix2)
        else:
            key = (fix1, fix2, pv)
        model = grid_models[key]
        r_xy = np.linspace(model.basis_xy.r_min, model.basis_xy.r_max, _N_EVAL)
        f_xy = model.basis_xy.evaluate(r_xy) @ model.theta_xy
        color = cmap(norm(pv))
        label = f"{pv:.0e}" if is_log else f"{int(pv)}"
        ax.plot(r_xy, f_xy, color=color, linewidth=1.8, label=label)

    ax.axhline(0, color="0.5", linestyle="--", linewidth=0.6)
    ax.set_xlabel("Distance (um)")
    ax.set_ylabel("Force")
    ax.set_title(f"Chrom-pole kernel\nsweeping {pname}")
    ax.legend(fontsize=7, title=pname, title_fontsize=7)

fig_kslice.suptitle(
    "Sweep 1 — Kernel shape: 1D marginal slices (other params at best)",
    fontsize=12,
)
fig_kslice.tight_layout()
plt.show()

# %% [markdown]
# ## Sweep 1b: Repeated random 2-fold stress test on the grid surface
#
# This is not used to choose *k*.  It is a deliberate "other extreme" check:
# rerun the same grid under many random 2-fold CV partitions and ask whether
# the *average* ensemble-MSE surface becomes materially sharper, or whether
# the hyperparameter landscape remains broadly flat.

# %%
from scipy.stats import spearmanr  # noqa: E402

SWEEP1B_K = 2
SWEEP1B_N_REPEATS = 24
SWEEP1B_SEED = 20260327
SWEEP1B_JOBLIB_VERBOSE = 10


def _sample_twofold_partitions(
    n_cells: int,
    n_samples: int,
    seed: int,
) -> tuple[list[tuple[int, ...]], int]:
    """Sample unique unordered 2-fold partitions without replacement."""
    if n_cells % 2 != 0:
        raise ValueError("Repeated 2-fold sweep requires an even number of cells.")
    half = n_cells // 2
    all_first_halves = [combo for combo in _combinations(range(n_cells), half) if 0 in combo]
    n_available = len(all_first_halves)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_available, size=min(n_samples, n_available), replace=False)
    return [all_first_halves[i] for i in chosen], n_available


sampled_twofold_partitions, n_twofold_total = _sample_twofold_partitions(
    len(cells), SWEEP1B_N_REPEATS, SWEEP1B_SEED,
)


def _run_one_grid_point_repeated_twofold(
    cells_in,
    base_cfg,
    sampled_partitions,
    rollout_base_seed,
    rollout_reps,
    rollout_horizons,
    idx,
    nb,
    lr,
    lro,
):
    """One hyperparameter config, averaged over many sampled 2-fold splits."""
    cfg = FitConfig(
        topology=base_cfg.topology,
        n_basis_xx=nb,
        n_basis_xy=nb,
        r_min_xx=base_cfg.r_min_xx,
        r_max_xx=base_cfg.r_max_xx,
        r_min_xy=base_cfg.r_min_xy,
        r_max_xy=base_cfg.r_max_xy,
        basis_type=base_cfg.basis_type,
        lambda_ridge=lr,
        lambda_rough=lro,
        basis_eval_mode=base_cfg.basis_eval_mode,
        endpoint_method=base_cfg.endpoint_method,
        endpoint_frac=base_cfg.endpoint_frac,
        diffusion_mode=base_cfg.diffusion_mode,
        dt=base_cfg.dt,
    )

    cv_scores = np.full(len(sampled_partitions), np.nan, dtype=np.float64)
    ensemble_scores = np.full(len(sampled_partitions), np.nan, dtype=np.float64)

    for rep_idx, first_half in enumerate(sampled_partitions):
        first_half_set = set(first_half)
        second_half = [i for i in range(len(cells_in)) if i not in first_half_set]
        perm = list(first_half) + second_half
        shuffled_cells = [cells_in[i] for i in perm]

        cv_result = cross_validate(shuffled_cells, cfg, k_folds=SWEEP1B_K)
        rollout_result = rollout_cross_validate(
            shuffled_cells,
            cfg,
            n_reps=rollout_reps,
            horizons=rollout_horizons,
            rng=np.random.default_rng(rollout_base_seed + 10_000 * idx + rep_idx),
            k_folds=SWEEP1B_K,
        )
        cv_scores[rep_idx] = cv_result.mean_error
        ensemble_scores[rep_idx] = rollout_ensemble_score(rollout_result)

    return (nb, lr, lro), cv_scores, ensemble_scores


print(f"Sweep 1b: repeated random {SWEEP1B_K}-fold stress test "
      f"({len(sampled_twofold_partitions)} sampled partitions out of {n_twofold_total}, "
      f"{ROLLOUT_REPS} rollout reps/fold)")
print(f"  Parallelizing over the {len(grid_configs)} hyperparameter settings "
      f"with Loky ({N_WORKERS} workers); each worker averages over all sampled partitions.")

results_k2 = Parallel(n_jobs=N_WORKERS, verbose=SWEEP1B_JOBLIB_VERBOSE)(
    delayed(_run_one_grid_point_repeated_twofold)(
        cells,
        BASE_CONFIG,
        sampled_twofold_partitions,
        ROLLOUT_BASE_SEED,
        ROLLOUT_REPS,
        ROLLOUT_HORIZONS,
        idx,
        nb,
        lr,
        lro,
    )
    for idx, (nb, lr, lro) in enumerate(grid_configs, start=1)
)

grid_cv_k2_repeats = np.full((len(sampled_twofold_partitions), len(grid_configs)), np.nan)
grid_ensemble_k2_repeats = np.full((len(sampled_twofold_partitions), len(grid_configs)), np.nan)
for key_idx, (key, cv_scores, ensemble_scores) in enumerate(results_k2):
    grid_cv_k2_repeats[:, key_idx] = cv_scores
    grid_ensemble_k2_repeats[:, key_idx] = ensemble_scores

grid_1step_loo = np.array([grid_cv[k].mean_error for k in grid_configs])
grid_1step_k2 = np.mean(grid_cv_k2_repeats, axis=0)
grid_ensemble_loo = np.array([rollout_ensemble_score(grid_rollout[k]) for k in grid_configs])
grid_ensemble_k2 = np.mean(grid_ensemble_k2_repeats, axis=0)
grid_ensemble_k2_sd = np.std(grid_ensemble_k2_repeats, axis=0)
grid_ensemble_k2_map = {
    key: grid_ensemble_k2[i] for i, key in enumerate(grid_configs)
}
best_key_cv_k2 = grid_configs[int(np.argmin(grid_1step_k2))]
best_key_k2 = grid_configs[int(np.argmin(grid_ensemble_k2))]
rank_rho_cv_k2, rank_p_cv_k2 = spearmanr(grid_1step_loo, grid_1step_k2)
rank_rho_k2, rank_p_k2 = spearmanr(grid_ensemble_loo, grid_ensemble_k2)
spread_loo_pct = 100.0 * (grid_ensemble_loo.max() - grid_ensemble_loo.min()) / grid_ensemble_loo.mean()
spread_k2_pct = 100.0 * (grid_ensemble_k2.max() - grid_ensemble_k2.min()) / grid_ensemble_k2.mean()
k2_cv_median_pct = 100.0 * np.median(grid_ensemble_k2_sd / grid_ensemble_k2)

top5_loo = set(sorted(grid_configs, key=lambda k: rollout_ensemble_score(grid_rollout[k]))[:5])
top5_k2 = {grid_configs[i] for i in np.argsort(grid_ensemble_k2)[:5]}

print(f"\nBest by ensemble MSE (LOO):    n_basis={best_key[0]}, "
      f"lambda_ridge={best_key[1]:.2e}, lambda_rough={best_key[2]:.2e}  "
      f"(ens MSE = {rollout_ensemble_score(grid_rollout[best_key]):.4e})")
print(f"Best by ensemble MSE (2-fold mean over repeats): n_basis={best_key_k2[0]}, "
      f"lambda_ridge={best_key_k2[1]:.2e}, lambda_rough={best_key_k2[2]:.2e}  "
      f"(mean ens MSE = {grid_ensemble_k2[np.argmin(grid_ensemble_k2)]:.4e})")
print(f"Best by 1-step CV (LOO):    n_basis={best_key_cv[0]}, "
      f"lambda_ridge={best_key_cv[1]:.2e}, lambda_rough={best_key_cv[2]:.2e}  "
      f"(CV MSE = {grid_cv[best_key_cv].mean_error:.4e})")
print(f"Best by 1-step CV (2-fold mean over repeats): n_basis={best_key_cv_k2[0]}, "
      f"lambda_ridge={best_key_cv_k2[1]:.2e}, lambda_rough={best_key_cv_k2[2]:.2e}  "
      f"(mean CV MSE = {grid_1step_k2[np.argmin(grid_1step_k2)]:.4e})")
print(f"Rank correlation for 1-step CV (LOO vs 2-fold): rho={rank_rho_cv_k2:.3f}, p={rank_p_cv_k2:.1e}")
print(f"Rank correlation (LOO vs 2-fold): rho={rank_rho_k2:.3f}, p={rank_p_k2:.1e}")
print(f"Relative score spread across grid: LOO={spread_loo_pct:.2f}%, 2-fold={spread_k2_pct:.2f}%")
print(f"Median CV across repeated 2-fold scores: {k2_cv_median_pct:.2f}%")
print(f"Top-5 overlap (LOO vs 2-fold): {len(top5_loo & top5_k2)}/5")

# %%
fig_1b, axes_1b = plt.subplots(1, 2, figsize=(13, 4.5))

axes_1b[0].scatter(grid_ensemble_loo, grid_ensemble_k2, s=22, alpha=0.7, edgecolors="none")
lo_min = min(grid_ensemble_loo.min(), grid_ensemble_k2.min())
lo_max = max(grid_ensemble_loo.max(), grid_ensemble_k2.max())
axes_1b[0].plot([lo_min, lo_max], [lo_min, lo_max], color="0.5", linestyle="--", linewidth=1.0)
axes_1b[0].set_xlabel("LOO ensemble MSE")
axes_1b[0].set_ylabel("2-fold ensemble MSE")
axes_1b[0].set_title(f"Grid scores by config\nSpearman rho={rank_rho_k2:.2f}")

axes_1b[1].plot(np.sort(grid_ensemble_loo), label="LOO", linewidth=2)
axes_1b[1].plot(np.sort(grid_ensemble_k2), label="2-fold", linewidth=2)
axes_1b[1].set_xlabel("Grid config rank (sorted by ensemble MSE)")
axes_1b[1].set_ylabel("Ensemble MSE")
axes_1b[1].set_title("Score spread across the grid")
axes_1b[1].legend()

fig_1b.suptitle(
    f"Sweep 1b — Hyperparameter surface stress test (LOO vs repeated {SWEEP1B_K}-fold)",
    fontsize=12,
)
fig_1b.tight_layout()
plt.show()

# %% [markdown]
# ### Surface comparison plots
#
# Compare the LOO and repeated 2-fold ensemble-MSE landscapes directly: 1D
# marginal envelopes for each hyperparameter, plus the ridge-vs-rough slice at
# `n_basis = 8` for both CV designs.

# %%
grid_scores_k2 = {k: grid_ensemble_k2[i] for i, k in enumerate(grid_configs)}
nb_lo_k2, nb_mid_k2, nb_hi_k2 = _marginal_stats(grid_scores_k2, grid_configs, 0, N_BASIS_GRID)
lr_lo_k2, lr_mid_k2, lr_hi_k2 = _marginal_stats(grid_scores_k2, grid_configs, 1, LAMBDA_RIDGE_GRID)
lro_lo_k2, lro_mid_k2, lro_hi_k2 = _marginal_stats(grid_scores_k2, grid_configs, 2, LAMBDA_ROUGH_GRID)

# Keep nb_best_k2 etc. for backward compat
nb_best_k2, lr_best_k2, lro_best_k2 = nb_lo_k2, lr_lo_k2, lro_lo_k2

fig_1b_marg, axes_1b_marg = plt.subplots(2, 3, figsize=(16, 7), squeeze=False)

# Row 1: LOO marginals
_plot_marginal(axes_1b_marg[0, 0], N_BASIS_GRID, nb_lo, nb_mid, nb_hi, "C0",
               "", "Ensemble MSE", "LOO: n_basis marginal")
axes_1b_marg[0, 0].set_xticks(N_BASIS_GRID)
_plot_marginal(axes_1b_marg[0, 1], LAMBDA_RIDGE_GRID, lr_lo, lr_mid, lr_hi, "C1",
               "", "Ensemble MSE", "LOO: lambda_ridge marginal", xlog=True)
_plot_marginal(axes_1b_marg[0, 2], LAMBDA_ROUGH_GRID, lro_lo, lro_mid, lro_hi, "C2",
               "", "Ensemble MSE", "LOO: lambda_rough marginal", xlog=True)

# Row 2: repeated 2-fold marginals
_plot_marginal(axes_1b_marg[1, 0], N_BASIS_GRID, nb_lo_k2, nb_mid_k2, nb_hi_k2, "C0",
               "n_basis", "Ensemble MSE", "Repeated 2-fold: n_basis marginal")
axes_1b_marg[1, 0].set_xticks(N_BASIS_GRID)
_plot_marginal(axes_1b_marg[1, 1], LAMBDA_RIDGE_GRID, lr_lo_k2, lr_mid_k2, lr_hi_k2, "C1",
               "lambda_ridge", "Ensemble MSE", "Repeated 2-fold: lambda_ridge marginal", xlog=True)
_plot_marginal(axes_1b_marg[1, 2], LAMBDA_ROUGH_GRID, lro_lo_k2, lro_mid_k2, lro_hi_k2, "C2",
               "lambda_rough", "Ensemble MSE", "Repeated 2-fold: lambda_rough marginal", xlog=True)

fig_1b_marg.suptitle("Sweep 1b — 1D marginal comparison: LOO vs repeated 2-fold", fontsize=12)
fig_1b_marg.tight_layout()
plt.show()

# %%
HEATMAP_N_BASIS = 8
mat_rr_loo_8 = _grid_slice_2d(
    "n_basis", HEATMAP_N_BASIS,
    LAMBDA_RIDGE_GRID, LAMBDA_ROUGH_GRID, grid_rollout, rollout_ensemble_score,
)
mat_rr_k2_8 = _grid_slice_2d(
    "n_basis", HEATMAP_N_BASIS,
    LAMBDA_RIDGE_GRID, LAMBDA_ROUGH_GRID, grid_ensemble_k2_map, lambda x: x,
)

vmin_rr = np.nanmin([np.nanmin(mat_rr_loo_8), np.nanmin(mat_rr_k2_8)])
vmax_rr = np.nanmax([np.nanmax(mat_rr_loo_8), np.nanmax(mat_rr_k2_8)])

fig_1b_heat, axes_1b_heat = plt.subplots(1, 2, figsize=(12, 5))

im_loo = axes_1b_heat[0].imshow(
    mat_rr_loo_8,
    origin="lower",
    aspect="auto",
    extent=[
        np.log10(LAMBDA_RIDGE_GRID[0]), np.log10(LAMBDA_RIDGE_GRID[-1]),
        np.log10(LAMBDA_ROUGH_GRID[0]), np.log10(LAMBDA_ROUGH_GRID[-1]),
    ],
    vmin=vmin_rr,
    vmax=vmax_rr,
)
axes_1b_heat[0].set_xlabel("log10(lambda_ridge)")
axes_1b_heat[0].set_ylabel("log10(lambda_rough)")
axes_1b_heat[0].set_title(f"LOO ensemble MSE at n_basis={HEATMAP_N_BASIS}")
fig_1b_heat.colorbar(im_loo, ax=axes_1b_heat[0], shrink=0.8)

im_k2 = axes_1b_heat[1].imshow(
    mat_rr_k2_8,
    origin="lower",
    aspect="auto",
    extent=[
        np.log10(LAMBDA_RIDGE_GRID[0]), np.log10(LAMBDA_RIDGE_GRID[-1]),
        np.log10(LAMBDA_ROUGH_GRID[0]), np.log10(LAMBDA_ROUGH_GRID[-1]),
    ],
    vmin=vmin_rr,
    vmax=vmax_rr,
)
axes_1b_heat[1].set_xlabel("log10(lambda_ridge)")
axes_1b_heat[1].set_ylabel("log10(lambda_rough)")
axes_1b_heat[1].set_title(f"Repeated 2-fold ensemble MSE at n_basis={HEATMAP_N_BASIS}")
fig_1b_heat.colorbar(im_k2, ax=axes_1b_heat[1], shrink=0.8)

fig_1b_heat.suptitle("Sweep 1b — Ridge vs rough slice comparison", fontsize=12)
fig_1b_heat.tight_layout()
plt.show()

# %%
# Kernel comparison: LOO-best vs 2-fold-best (full-data fits already available)
fig_1b_kern, ax_1b_kern = plt.subplots(
    1, _n_kernel_panels, figsize=(6 * _n_kernel_panels, 4.5), squeeze=False,
)

for key, label, color, lw in [
    (best_key, f"LOO best {best_key}", "C0", 2.2),
    (best_key_k2, f"2-fold best {best_key_k2}", "C1", 2.2),
]:
    model = grid_models[key]
    if _topology_has_chroms_grid:
        r_xx = np.linspace(model.basis_xx.r_min, model.basis_xx.r_max, _N_EVAL)
        f_xx = model.basis_xx.evaluate(r_xx) @ model.theta_xx
        ax_1b_kern[0, 0].plot(r_xx, f_xx, color=color, linewidth=lw, label=label)
    r_xy = np.linspace(model.basis_xy.r_min, model.basis_xy.r_max, _N_EVAL)
    f_xy = model.basis_xy.evaluate(r_xy) @ model.theta_xy
    ax_1b_kern[0, _n_kernel_panels - 1].plot(
        r_xy, f_xy, color=color, linewidth=lw, label=label,
    )

if _topology_has_chroms_grid:
    ax_1b_kern[0, 0].axhline(0, color="0.5", linestyle="--", linewidth=0.6)
    ax_1b_kern[0, 0].set_xlabel("Distance (um)")
    ax_1b_kern[0, 0].set_ylabel("Force")
    ax_1b_kern[0, 0].set_title("Chrom-chrom kernel")
    ax_1b_kern[0, 0].legend(fontsize=7)

ax_1b_xy = ax_1b_kern[0, _n_kernel_panels - 1]
ax_1b_xy.axhline(0, color="0.5", linestyle="--", linewidth=0.6)
ax_1b_xy.set_xlabel("Distance (um)")
ax_1b_xy.set_ylabel("Force")
ax_1b_xy.set_title("Chrom-pole kernel")
ax_1b_xy.legend(fontsize=7)

fig_1b_kern.suptitle(
    "Sweep 1b — Kernel shape: LOO-best vs 2-fold-best (both fit on all cells)",
    fontsize=12,
)
fig_1b_kern.tight_layout()
plt.show()

# %% [markdown]
# ## Sweep 2: Estimator mode (Ito / Ito-shift / Stratonovich)
#
# "ito" = current positions, "ito_shift" = previous positions (decorrelates
# localisation noise), "strato" = midpoint (Stratonovich convention).
# With constant D(x) there is no spurious-drift correction, so all three
# conventions should give essentially the same kernel.

# %%
ESTIMATOR_MODES = ["ito", "ito_shift", "strato"]

cv_mode: dict[str, CVResult] = {}
rollout_mode: dict[str, RolloutCVResult] = {}
models_mode: dict[str, object] = {}

for idx, mode in enumerate(ESTIMATOR_MODES, start=1):
    cfg = FitConfig(
        topology=BASE_CONFIG.topology,
        n_basis_xx=BASE_CONFIG.n_basis_xx,
        n_basis_xy=BASE_CONFIG.n_basis_xy,
        r_min_xx=BASE_CONFIG.r_min_xx,
        r_max_xx=BASE_CONFIG.r_max_xx,
        r_min_xy=BASE_CONFIG.r_min_xy,
        r_max_xy=BASE_CONFIG.r_max_xy,
        basis_type=BASE_CONFIG.basis_type,
        lambda_ridge=BASE_CONFIG.lambda_ridge,
        lambda_rough=BASE_CONFIG.lambda_rough,
        basis_eval_mode=mode,
        endpoint_method=BASE_CONFIG.endpoint_method,
        endpoint_frac=BASE_CONFIG.endpoint_frac,
        diffusion_mode=BASE_CONFIG.diffusion_mode,
        dt=BASE_CONFIG.dt,
    )
    print(f"[Sweep 2 {idx}/{len(ESTIMATOR_MODES)}] mode={mode}: running 1-step LOOCV...", flush=True)
    cv_mode[mode] = cross_validate(cells, cfg)
    print(
        f"    1-step CV = {cv_mode[mode].mean_error:.4e}"
        f" ± {cv_mode[mode].fold_se:.4e}",
        flush=True,
    )
    print(
        f"    running rollout LOOCV ({len(cells)} folds x {ROLLOUT_REPS} rollouts)...",
        flush=True,
    )
    rollout_mode[mode] = run_rollout_cv(
        cells, cfg, seed=ROLLOUT_BASE_SEED + 300 + idx
    )
    print(
        f"    ensemble MSE = {rollout_ensemble_score(rollout_mode[mode]):.4e}"
        f" ± {rollout_ensemble_se(rollout_mode[mode]):.4e}"
        f"  path MSE = {rollout_path_score(rollout_mode[mode]):.4e}"
        f"  W1 total = {rollout_w1_score(rollout_mode[mode]):.4e}",
        flush=True,
    )
    models_mode[mode] = fit_model(cells, cfg)

# %%
fig_mode_cv = plot_cv_curve(cv_mode)
fig_mode_cv.axes[0].set_title("Sweep 2 — Estimator mode CV comparison")
plt.show()

# %%
fig_mode_rollout, ax_mode_rollout = plt.subplots(figsize=(7, 4))
mode_labels = list(ESTIMATOR_MODES)
mode_rollout_means = [rollout_ensemble_score(rollout_mode[m]) for m in mode_labels]
mode_rollout_stds = [rollout_ensemble_se(rollout_mode[m]) for m in mode_labels]
ax_mode_rollout.bar(
    np.arange(len(mode_labels)),
    mode_rollout_means,
    yerr=mode_rollout_stds,
    capsize=4,
    color=["C0", "C1", "C2"][: len(mode_labels)],
)
ax_mode_rollout.set_xticks(np.arange(len(mode_labels)))
ax_mode_rollout.set_xticklabels(mode_labels)
ax_mode_rollout.set_ylabel("Leave-one-out ensemble MSE")
ax_mode_rollout.set_title(
    f"Sweep 2 — Ensemble MSE comparison  ({ROLLOUT_REPS} reps/fold)"
)
fig_mode_rollout.tight_layout()
plt.show()

# %%
# Kernel shapes overlaid for each estimator mode
topology_has_chroms = WINNING_TOPOLOGY in ("poles_and_chroms", "center_and_chroms")
n_panels = 2 if topology_has_chroms else 1
MODE_COLORS = {"ito": "C0", "ito_shift": "C1", "strato": "C2"}

fig_mode_kernels, axes_mode_k = plt.subplots(
    1, n_panels, figsize=(6 * n_panels, 4), squeeze=False,
)

n_points = 200
for mode in ESTIMATOR_MODES:
    model: FittedModel = models_mode[mode]
    color = MODE_COLORS.get(mode, "k")

    if topology_has_chroms:
        r_xx = np.linspace(model.basis_xx.r_min, model.basis_xx.r_max, n_points)
        phi_xx = model.basis_xx.evaluate(r_xx)
        axes_mode_k[0, 0].plot(r_xx, phi_xx @ model.theta_xx,
                               color=color, linewidth=1.8, label=mode)

    r_xy = np.linspace(model.basis_xy.r_min, model.basis_xy.r_max, n_points)
    phi_xy = model.basis_xy.evaluate(r_xy)
    col_xy = n_panels - 1
    axes_mode_k[0, col_xy].plot(r_xy, phi_xy @ model.theta_xy,
                                color=color, linewidth=1.8, label=mode)

if topology_has_chroms:
    axes_mode_k[0, 0].axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    axes_mode_k[0, 0].set_xlabel("Distance (um)")
    axes_mode_k[0, 0].set_ylabel("Force")
    axes_mode_k[0, 0].set_title("Chrom-chrom kernel")
    axes_mode_k[0, 0].legend()

axes_mode_k[0, n_panels - 1].axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
axes_mode_k[0, n_panels - 1].set_xlabel("Distance (um)")
axes_mode_k[0, n_panels - 1].set_ylabel("Force")
axes_mode_k[0, n_panels - 1].set_title("Chrom-pole kernel")
axes_mode_k[0, n_panels - 1].legend()

fig_mode_kernels.suptitle("Sweep 2 — Kernel shapes by estimator mode", fontsize=13)
fig_mode_kernels.tight_layout()
plt.show()

best_mode = min(cv_mode, key=lambda k: cv_mode[k].mean_error)
best_mode_rollout = min(rollout_mode, key=lambda k: rollout_ensemble_score(rollout_mode[k]))
print(f"\nBest estimator mode by 1-step CV: {best_mode}  (CV MSE = {cv_mode[best_mode].mean_error:.4e})")
print(
    f"Best estimator mode by ensemble MSE: {best_mode_rollout}"
    f"  (ens MSE = {rollout_ensemble_score(rollout_mode[best_mode_rollout]):.4e})"
)

# %% [markdown]
# ## Sweep 3: Endpoint fraction
#
# Sweep the NEB-to-AO fraction, focusing on the early-phase regime where the
# gathering dynamics are most active. We also try end_sep (95% plateau).

# %%
ENDPOINT_FRACS = [0.15, 0.2, 0.25, 0.33, 0.4, 0.5]

raw_cells = cells_raw
print(f"Using {len(raw_cells)} raw CellData objects for endpoint sweep.")

# %%
cv_endpoint: dict[str, CVResult] = {}
rollout_endpoint: dict[str, RolloutCVResult] = {}
n_cells_endpoint: dict[str, int] = {}

# Sweep neb_ao_frac values
for frac in ENDPOINT_FRACS:
    label = f"frac={frac:.2f}"
    trimmed_method = []
    for raw_cell in raw_cells:
        try:
            trimmed_method.append(trim_trajectory(raw_cell, method="neb_ao_frac", frac=frac))
        except ValueError as exc:
            print(f"  Skipping {raw_cell.cell_id} for {label}: {exc}")

    n_cells_endpoint[label] = len(trimmed_method)
    if len(trimmed_method) < 3:
        print(f"  {label}: only {len(trimmed_method)} cells — skipping CV (too few).")
        continue

    cfg = FitConfig(
        topology=BASE_CONFIG.topology,
        n_basis_xx=BASE_CONFIG.n_basis_xx,
        n_basis_xy=BASE_CONFIG.n_basis_xy,
        r_min_xx=BASE_CONFIG.r_min_xx,
        r_max_xx=BASE_CONFIG.r_max_xx,
        r_min_xy=BASE_CONFIG.r_min_xy,
        r_max_xy=BASE_CONFIG.r_max_xy,
        basis_type=BASE_CONFIG.basis_type,
        lambda_ridge=BASE_CONFIG.lambda_ridge,
        lambda_rough=BASE_CONFIG.lambda_rough,
        basis_eval_mode=BASE_CONFIG.basis_eval_mode,
        endpoint_method="neb_ao_frac",
        endpoint_frac=frac,
        diffusion_mode=BASE_CONFIG.diffusion_mode,
        dt=BASE_CONFIG.dt,
    )
    print(f"[Sweep 3 {label}] running 1-step LOOCV...", flush=True)
    cv_endpoint[label] = cross_validate(trimmed_method, cfg)
    print(
        f"    n_cells={len(trimmed_method)}"
        f"  1-step CV = {cv_endpoint[label].mean_error:.4e}"
        f" ± {cv_endpoint[label].fold_se:.4e}",
        flush=True,
    )
    print(
        f"    running rollout LOOCV ({len(trimmed_method)} folds x {ROLLOUT_REPS} rollouts)...",
        flush=True,
    )
    rollout_endpoint[label] = run_rollout_cv(
        trimmed_method, cfg, seed=ROLLOUT_BASE_SEED + int(round(100 * frac))
    )
    print(
        f"    ensemble MSE = {rollout_ensemble_score(rollout_endpoint[label]):.4e}"
        f" ± {rollout_ensemble_se(rollout_endpoint[label]):.4e}"
        f"  path MSE = {rollout_path_score(rollout_endpoint[label]):.4e}"
        f"  W1 total = {rollout_w1_score(rollout_endpoint[label]):.4e}",
        flush=True,
    )

# Also try end_sep
trimmed_end_sep = []
for raw_cell in raw_cells:
    try:
        trimmed_end_sep.append(trim_trajectory(raw_cell, method="end_sep"))
    except ValueError as exc:
        print(f"  Skipping {raw_cell.cell_id} for end_sep: {exc}")

n_cells_endpoint["end_sep"] = len(trimmed_end_sep)
if len(trimmed_end_sep) >= 3:
    cfg_es = FitConfig(
        topology=BASE_CONFIG.topology,
        n_basis_xx=BASE_CONFIG.n_basis_xx,
        n_basis_xy=BASE_CONFIG.n_basis_xy,
        r_min_xx=BASE_CONFIG.r_min_xx,
        r_max_xx=BASE_CONFIG.r_max_xx,
        r_min_xy=BASE_CONFIG.r_min_xy,
        r_max_xy=BASE_CONFIG.r_max_xy,
        basis_type=BASE_CONFIG.basis_type,
        lambda_ridge=BASE_CONFIG.lambda_ridge,
        lambda_rough=BASE_CONFIG.lambda_rough,
        basis_eval_mode=BASE_CONFIG.basis_eval_mode,
        endpoint_method="end_sep",
        diffusion_mode=BASE_CONFIG.diffusion_mode,
        dt=BASE_CONFIG.dt,
    )
    print("[Sweep 3 end_sep] running 1-step LOOCV...", flush=True)
    cv_endpoint["end_sep"] = cross_validate(trimmed_end_sep, cfg_es)
    print(
        f"    n_cells={len(trimmed_end_sep)}"
        f"  1-step CV = {cv_endpoint['end_sep'].mean_error:.4e}"
        f" ± {cv_endpoint['end_sep'].fold_se:.4e}",
        flush=True,
    )
    print(
        f"    running rollout LOOCV ({len(trimmed_end_sep)} folds x {ROLLOUT_REPS} rollouts)...",
        flush=True,
    )
    rollout_endpoint["end_sep"] = run_rollout_cv(
        trimmed_end_sep, cfg_es, seed=ROLLOUT_BASE_SEED + 499
    )
    print(
        f"    ensemble MSE = {rollout_ensemble_score(rollout_endpoint['end_sep']):.4e}"
        f" ± {rollout_ensemble_se(rollout_endpoint['end_sep']):.4e}"
        f"  path MSE = {rollout_path_score(rollout_endpoint['end_sep']):.4e}"
        f"  W1 total = {rollout_w1_score(rollout_endpoint['end_sep']):.4e}",
        flush=True,
    )
else:
    print(f"  end_sep: only {len(trimmed_end_sep)} cells — skipping CV (too few).")

# %%
if cv_endpoint:
    fig_ep = plot_cv_curve(cv_endpoint)
    fig_ep.axes[0].set_title("Sweep 3 — Endpoint fraction CV comparison")
    plt.show()

    endpoint_labels = list(cv_endpoint.keys())
    endpoint_rollout_means = [rollout_ensemble_score(rollout_endpoint[k]) for k in endpoint_labels]
    endpoint_rollout_stds = [rollout_ensemble_se(rollout_endpoint[k]) for k in endpoint_labels]
    fig_ep_rollout, ax_ep_rollout = plt.subplots(figsize=(8, 4))
    ax_ep_rollout.bar(
        np.arange(len(endpoint_labels)),
        endpoint_rollout_means,
        yerr=endpoint_rollout_stds,
        capsize=4,
        color=[f"C{i}" for i in range(len(endpoint_labels))],
    )
    ax_ep_rollout.set_xticks(np.arange(len(endpoint_labels)))
    ax_ep_rollout.set_xticklabels(endpoint_labels, rotation=30, ha="right")
    ax_ep_rollout.set_ylabel("Leave-one-out ensemble MSE")
    ax_ep_rollout.set_title(
        f"Sweep 3 — Endpoint ensemble MSE comparison  ({ROLLOUT_REPS} reps/fold)"
    )
    fig_ep_rollout.tight_layout()
    plt.show()

    best_endpoint = min(cv_endpoint, key=lambda k: cv_endpoint[k].mean_error)
    best_endpoint_rollout = min(
        rollout_endpoint, key=lambda k: rollout_ensemble_score(rollout_endpoint[k])
    )
    print(f"\nBest endpoint by 1-step CV: {best_endpoint}  (CV MSE = {cv_endpoint[best_endpoint].mean_error:.4e})")
    print(
        f"Best endpoint by ensemble MSE: {best_endpoint_rollout}"
        f"  (ens MSE = {rollout_ensemble_score(rollout_endpoint[best_endpoint_rollout]):.4e})"
    )
else:
    print("No endpoint settings produced enough cells for CV.")

print("\nCell counts per endpoint setting:")
for label in list(f"frac={f:.2f}" for f in ENDPOINT_FRACS) + ["end_sep"]:
    print(f"  {label:20s}: {n_cells_endpoint.get(label, 0)} cells")

# %% [markdown]
# ## Sweep 4: Diffusion estimation mode
#
# Compare scalar D across four estimators (msd, vestergaard, weak_noise,
# f_corrected).

# %%
from chromlearn.model_fitting.basis import BSplineBasis as _BSplineBasis, HatBasis  # noqa: E402

_BasisClass = _BSplineBasis if BASE_CONFIG.basis_type == "bspline" else HatBasis

_topology_has_chroms = WINNING_TOPOLOGY in ("poles_and_chroms", "center_and_chroms")
if _topology_has_chroms:
    _basis_xx_fc = _BasisClass(
        BASE_CONFIG.r_min_xx, BASE_CONFIG.r_max_xx, BASE_CONFIG.n_basis_xx
    )
else:
    _basis_xx_fc = None

_basis_xy_fc = _BasisClass(
    BASE_CONFIG.r_min_xy, BASE_CONFIG.r_max_xy, BASE_CONFIG.n_basis_xy
)

from scipy.linalg import block_diag  # noqa: E402

_R_xx = _basis_xx_fc.roughness_matrix() if _basis_xx_fc is not None else None
_R_xy = _basis_xy_fc.roughness_matrix()
_roughness_fc = block_diag(_R_xx, _R_xy) if _R_xx is not None else _R_xy

_G_fc, _V_fc = build_design_matrix(
    cells,
    _basis_xx_fc,
    _basis_xy_fc,
    basis_eval_mode=BASE_CONFIG.basis_eval_mode,
    topology=WINNING_TOPOLOGY,
)
_fit_result_fc = fit_kernels(
    _G_fc,
    _V_fc,
    lambda_ridge=BASE_CONFIG.lambda_ridge,
    lambda_rough=BASE_CONFIG.lambda_rough,
    R=_roughness_fc,
)

print("Preliminary fit for f_corrected mode complete.")
print(f"  theta shape: {_fit_result_fc.theta.shape}")
print(f"  residuals shape: {_fit_result_fc.residuals.shape}")

# %%
DIFFUSION_MODES = ["msd", "vestergaard", "weak_noise", "f_corrected"]

D_scalar_by_mode: dict[str, float] = {}

for diff_mode in DIFFUSION_MODES:
    if diff_mode == "f_corrected":
        d_estimates = local_diffusion_estimates(
            cells,
            dt=BASE_CONFIG.dt,
            mode=diff_mode,
            fit_result=_fit_result_fc,
            basis_xx=_basis_xx_fc,
            basis_xy=_basis_xy_fc,
            topology=BASE_CONFIG.topology,
        )
    else:
        d_estimates = local_diffusion_estimates(
            cells,
            dt=BASE_CONFIG.dt,
            mode=diff_mode,
        )

    # Flatten all per-particle local estimates and take the mean.
    all_d = np.concatenate([arr.ravel() for arr in d_estimates])
    valid_d = all_d[np.isfinite(all_d)]
    scalar_d = float(np.mean(valid_d)) if valid_d.size > 0 else np.nan
    D_scalar_by_mode[diff_mode] = scalar_d
    print(f"  mode={diff_mode:15s}  D = {scalar_d:.4e} um^2/s  (n_valid={valid_d.size})")

# %%
fig_diff, ax_diff = plt.subplots(figsize=(7, 4))
modes_plotted = list(D_scalar_by_mode.keys())
d_values = [D_scalar_by_mode[m] for m in modes_plotted]
x_pos = np.arange(len(modes_plotted))
bars = ax_diff.bar(x_pos, d_values, color=["C0", "C1", "C2", "C3"][: len(modes_plotted)])
ax_diff.set_xticks(x_pos)
ax_diff.set_xticklabels(modes_plotted)
ax_diff.set_ylabel("Scalar D (um^2/s)")
ax_diff.set_title("Sweep 4 — Diffusion estimation mode comparison")
for bar, val in zip(bars, d_values):
    ax_diff.text(
        bar.get_x() + bar.get_width() / 2.0,
        val * 1.02,
        f"{val:.3e}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
fig_diff.tight_layout()
plt.show()

# %% [markdown]
# ## Summary

# %%
print("=" * 60)  # top rule
print("Hyperparameter sensitivity summary")
print("=" * 60)  # mid rule
print(f"{'Parameter':<30} {'Best value':<20} {'Ensemble MSE'}")
print("-" * 60)

# Joint grid best
best_grid_score = rollout_ensemble_score(grid_rollout[best_key])
print(
    f"  {'n_basis (joint grid)':<28} {best_n_basis_rollout:<20} "
    f"{best_grid_score:.4e}"
)
print(
    f"  {'lambda_ridge (joint grid)':<28} {best_ridge_rollout:<20.2e} "
    f"{best_grid_score:.4e}"
)
print(
    f"  {'lambda_rough (joint grid)':<28} {best_rough_rollout:<20.2e} "
    f"{best_grid_score:.4e}"
)

# Estimator mode
print(
    f"  {'basis_eval_mode (ens MSE)':<28} {best_mode_rollout:<20} "
    f"{rollout_ensemble_score(rollout_mode[best_mode_rollout]):.4e}"
)

# Endpoint method
if cv_endpoint:
    print(
        f"  {'endpoint_method (ens MSE)':<28} {best_endpoint_rollout:<20} "
        f"{rollout_ensemble_score(rollout_endpoint[best_endpoint_rollout]):.4e}"
    )
else:
    print(f"  {'endpoint_method':<28} {'N/A (CV failed)':<20}")

# Diffusion mode (no CV — just D values)
if D_scalar_by_mode:
    d_spread = max(D_scalar_by_mode.values()) - min(D_scalar_by_mode.values())
    d_mean = np.mean(list(D_scalar_by_mode.values()))
    print(
        f"  {'diffusion_mode':<28} {'(see D values)':<20} "
        f"spread = {d_spread:.3e} ({100*d_spread/d_mean:.1f}% of mean)"
    )

print("=" * 60)  # bottom rule

# %% [markdown]
# ## Metric correlation
#
# Scatter the rollout metrics against each other across all grid configs
# to check whether path MSE and W1 are measuring the same thing.

# %%
from scipy.stats import spearmanr  # noqa: E402

grid_ensemble = np.array([rollout_ensemble_score(grid_rollout[k]) for k in grid_configs])
grid_path = np.array([rollout_path_score(grid_rollout[k]) for k in grid_configs])
grid_w1 = np.array([rollout_w1_score(grid_rollout[k]) for k in grid_configs])
grid_1step = np.array([grid_cv[k].mean_error for k in grid_configs])

fig_corr, axes_corr = plt.subplots(2, 3, figsize=(16, 9))

# Row 1: ensemble MSE vs others
rho_ep, pval_ep = spearmanr(grid_ensemble, grid_path)
axes_corr[0, 0].scatter(grid_ensemble, grid_path, s=18, alpha=0.6, edgecolors="none")
axes_corr[0, 0].set_xlabel("Ensemble MSE")
axes_corr[0, 0].set_ylabel("Path MSE (per-rep)")
axes_corr[0, 0].set_title(f"Ensemble vs Path  (r={rho_ep:.2f}, p={pval_ep:.1e})")

rho_ew, pval_ew = spearmanr(grid_ensemble, grid_w1)
axes_corr[0, 1].scatter(grid_ensemble, grid_w1, s=18, alpha=0.6, edgecolors="none", color="C1")
axes_corr[0, 1].set_xlabel("Ensemble MSE")
axes_corr[0, 1].set_ylabel("Final-frame W1")
axes_corr[0, 1].set_title(f"Ensemble vs W1  (r={rho_ew:.2f}, p={pval_ew:.1e})")

rho_e1, pval_e1 = spearmanr(grid_ensemble, grid_1step)
axes_corr[0, 2].scatter(grid_ensemble, grid_1step, s=18, alpha=0.6, edgecolors="none", color="C2")
axes_corr[0, 2].set_xlabel("Ensemble MSE")
axes_corr[0, 2].set_ylabel("1-step CV MSE")
axes_corr[0, 2].set_title(f"Ensemble vs 1-step  (r={rho_e1:.2f}, p={pval_e1:.1e})")

# Row 2: pairwise among path, W1, 1-step
rho_pw, pval_pw = spearmanr(grid_path, grid_w1)
axes_corr[1, 0].scatter(grid_path, grid_w1, s=18, alpha=0.6, edgecolors="none", color="C3")
axes_corr[1, 0].set_xlabel("Path MSE")
axes_corr[1, 0].set_ylabel("Final-frame W1")
axes_corr[1, 0].set_title(f"Path vs W1  (r={rho_pw:.2f}, p={pval_pw:.1e})")

rho_p1, pval_p1 = spearmanr(grid_path, grid_1step)
axes_corr[1, 1].scatter(grid_path, grid_1step, s=18, alpha=0.6, edgecolors="none", color="C4")
axes_corr[1, 1].set_xlabel("Path MSE")
axes_corr[1, 1].set_ylabel("1-step CV MSE")
axes_corr[1, 1].set_title(f"Path vs 1-step  (r={rho_p1:.2f}, p={pval_p1:.1e})")

rho_w1, pval_w1 = spearmanr(grid_w1, grid_1step)
axes_corr[1, 2].scatter(grid_w1, grid_1step, s=18, alpha=0.6, edgecolors="none", color="C5")
axes_corr[1, 2].set_xlabel("Final-frame W1")
axes_corr[1, 2].set_ylabel("1-step CV MSE")
axes_corr[1, 2].set_title(f"W1 vs 1-step  (r={rho_w1:.2f}, p={pval_w1:.1e})")

fig_corr.suptitle("Metric correlation across hyperparameter grid (Sweep 1)", fontsize=12)
fig_corr.tight_layout()
plt.show()

print(f"Ensemble MSE range: [{grid_ensemble.min():.4e}, {grid_ensemble.max():.4e}]"
      f"  (spread = {(grid_ensemble.max()-grid_ensemble.min())/grid_ensemble.mean()*100:.2f}% of mean)")
print(f"Path MSE range:     [{grid_path.min():.4e}, {grid_path.max():.4e}]"
      f"  (spread = {(grid_path.max()-grid_path.min())/grid_path.mean()*100:.2f}% of mean)")
print(f"1-step CV range:    [{grid_1step.min():.4e}, {grid_1step.max():.4e}]"
      f"  (spread = {(grid_1step.max()-grid_1step.min())/grid_1step.mean()*100:.2f}% of mean)")
