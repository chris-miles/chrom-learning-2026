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

ROLLOUT_REPS = 4
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
) -> RolloutCVResult:
    """Deterministic rollout LOOCV helper for sweep comparisons."""
    return rollout_cross_validate(
        cells_in,
        cfg,
        n_reps=ROLLOUT_REPS,
        horizons=ROLLOUT_HORIZONS,
        rng=np.random.default_rng(seed),
    )

# %% [markdown]
# ## Sweep 1: Joint (n\_basis, lambda\_ridge, lambda\_rough) grid
#
# The old one-at-a-time sweeps can miss interactions (e.g., more basis functions
# may need stronger regularization). Instead we do a full 3D grid and visualize
# 2D slices through the best value of the third parameter.
#
# We use the rollout path MSE (per-chromosome 3D) as the selection target.
# One-step CV MSE is recorded alongside for comparison.

# %%
from itertools import product as _product  # noqa: E402

N_BASIS_GRID = [4, 8, 12, 16]
LAMBDA_RIDGE_GRID = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
LAMBDA_ROUGH_GRID = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])

grid_configs = list(_product(N_BASIS_GRID, LAMBDA_RIDGE_GRID, LAMBDA_ROUGH_GRID))
n_grid = len(grid_configs)
print(f"Joint grid: {len(N_BASIS_GRID)} x {len(LAMBDA_RIDGE_GRID)} x {len(LAMBDA_ROUGH_GRID)} = {n_grid} configs")

grid_cv: dict[tuple, CVResult] = {}
grid_rollout: dict[tuple, RolloutCVResult] = {}

for idx, (nb, lr, lro) in enumerate(grid_configs, start=1):
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
    key = (nb, lr, lro)
    print(
        f"[Grid {idx}/{n_grid}] n_basis={nb}, ridge={lr:.0e}, rough={lro:.0e}",
        end="",
        flush=True,
    )
    grid_cv[key] = cross_validate(cells, cfg)
    grid_rollout[key] = run_rollout_cv(cells, cfg, seed=ROLLOUT_BASE_SEED + idx)
    print(
        f"  1-step={grid_cv[key].mean_error:.4e}"
        f"  path_MSE={rollout_path_score(grid_rollout[key]):.4e}"
        f"  W1={rollout_w1_score(grid_rollout[key]):.4e}",
        flush=True,
    )

# %%
best_key = min(grid_configs, key=lambda k: rollout_path_score(grid_rollout[k]))
best_n_basis_rollout, best_ridge_rollout, best_rough_rollout = best_key
best_key_cv = min(grid_configs, key=lambda k: grid_cv[k].mean_error)

print(f"\nBest by rollout path MSE: n_basis={best_key[0]}, "
      f"lambda_ridge={best_key[1]:.2e}, lambda_rough={best_key[2]:.2e}  "
      f"(path MSE = {rollout_path_score(grid_rollout[best_key]):.4e})")
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
    LAMBDA_RIDGE_GRID, LAMBDA_ROUGH_GRID, grid_rollout, rollout_path_score,
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
axes_slices[0].set_title(f"Path MSE at n_basis={best_n_basis_rollout}")
fig_slices.colorbar(im0, ax=axes_slices[0], shrink=0.8)

# Slice 2: n_basis vs ridge at best rough
mat_nr = _grid_slice_2d(
    "lambda_rough", best_rough_rollout,
    np.array(N_BASIS_GRID, dtype=float), LAMBDA_RIDGE_GRID, grid_rollout, rollout_path_score,
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
axes_slices[1].set_title(f"Path MSE at lambda_rough={best_rough_rollout:.0e}")
fig_slices.colorbar(im1, ax=axes_slices[1], shrink=0.8)

# Slice 3: n_basis vs rough at best ridge
mat_nro = _grid_slice_2d(
    "lambda_ridge", best_ridge_rollout,
    np.array(N_BASIS_GRID, dtype=float), LAMBDA_ROUGH_GRID, grid_rollout, rollout_path_score,
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
axes_slices[2].set_title(f"Path MSE at lambda_ridge={best_ridge_rollout:.0e}")
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
# For each hyperparameter, show the best rollout path MSE achieved at each
# value (minimizing over the other two parameters). This gives the envelope
# of the 3D grid, analogous to the old 1D sweeps but without fixing the
# other parameters at possibly suboptimal values.

# %%
fig_marginal, axes_m = plt.subplots(1, 3, figsize=(16, 4))

# n_basis marginal
nb_best = [
    min(rollout_path_score(grid_rollout[k]) for k in grid_configs if k[0] == nb)
    for nb in N_BASIS_GRID
]
axes_m[0].plot(N_BASIS_GRID, nb_best, "o-", color="C0", linewidth=1.8)
axes_m[0].set_xlabel("n_basis")
axes_m[0].set_ylabel("Best path MSE (over ridge, rough)")
axes_m[0].set_title("n_basis marginal")
axes_m[0].set_xticks(N_BASIS_GRID)

# lambda_ridge marginal
lr_best = [
    min(rollout_path_score(grid_rollout[k]) for k in grid_configs if k[1] == lr)
    for lr in LAMBDA_RIDGE_GRID
]
axes_m[1].plot(LAMBDA_RIDGE_GRID, lr_best, "o-", color="C1", linewidth=1.8)
axes_m[1].set_xscale("log")
axes_m[1].set_xlabel("lambda_ridge")
axes_m[1].set_ylabel("Best path MSE (over n_basis, rough)")
axes_m[1].set_title("lambda_ridge marginal")

# lambda_rough marginal
lro_best = [
    min(rollout_path_score(grid_rollout[k]) for k in grid_configs if k[2] == lro)
    for lro in LAMBDA_ROUGH_GRID
]
axes_m[2].plot(LAMBDA_ROUGH_GRID, lro_best, "o-", color="C2", linewidth=1.8)
axes_m[2].set_xscale("log")
axes_m[2].set_xlabel("lambda_rough")
axes_m[2].set_ylabel("Best path MSE (over n_basis, ridge)")
axes_m[2].set_title("lambda_rough marginal")

fig_marginal.suptitle("1D marginal envelopes from the joint grid", fontsize=12)
fig_marginal.tight_layout()
plt.show()

# %% [markdown]
# ## Sweep 2: Estimator mode (Ito / Ito-shift / Stratonovich)
#
# "ito" = current positions, "ito_shift" = previous positions (decorrelates
# localisation noise), "strato" = midpoint (Stratonovich convention).

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
        f"    rollout MSE = {rollout_path_score(rollout_mode[mode]):.4e}"
        f" ± {rollout_path_se(rollout_mode[mode]):.4e}"
        f"  endpoint = {np.nanmean(rollout_mode[mode].endpoint_mean_error):.4e}"
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
mode_rollout_means = [rollout_path_score(rollout_mode[m]) for m in mode_labels]
mode_rollout_stds = [rollout_path_se(rollout_mode[m]) for m in mode_labels]
ax_mode_rollout.bar(
    np.arange(len(mode_labels)),
    mode_rollout_means,
    yerr=mode_rollout_stds,
    capsize=4,
    color=["C0", "C1", "C2"][: len(mode_labels)],
)
ax_mode_rollout.set_xticks(np.arange(len(mode_labels)))
ax_mode_rollout.set_xticklabels(mode_labels)
ax_mode_rollout.set_ylabel("Leave-one-out rollout path MSE")
ax_mode_rollout.set_title(
    f"Sweep 2 — Rollout CV comparison  ({ROLLOUT_REPS} reps/fold)"
)
fig_mode_rollout.tight_layout()
plt.show()

# %%
# Kernel shapes for each mode side by side
from chromlearn.model_fitting.model import FittedModel  # noqa: E402

topology_has_chroms = WINNING_TOPOLOGY in ("poles_and_chroms", "center_and_chroms")
n_panels_per_model = 2 if topology_has_chroms else 1
n_modes = len(ESTIMATOR_MODES)

fig_mode_kernels, axes_mode_k = plt.subplots(
    n_modes,
    n_panels_per_model,
    figsize=(6 * n_panels_per_model, 3.5 * n_modes),
    squeeze=False,
)

for row, mode in enumerate(ESTIMATOR_MODES):
    model: FittedModel = models_mode[mode]
    n_points = 200

    if topology_has_chroms:
        r_xx = np.linspace(model.basis_xx.r_min, model.basis_xx.r_max, n_points)
        phi_xx = model.basis_xx.evaluate(r_xx)
        axes_mode_k[row, 0].plot(r_xx, phi_xx @ model.theta_xx, color="C0", linewidth=2)
        axes_mode_k[row, 0].axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
        axes_mode_k[row, 0].set_xlabel("Distance (um)")
        axes_mode_k[row, 0].set_ylabel("Force")
        axes_mode_k[row, 0].set_title(f"mode={mode}  |  chrom-chrom kernel")

    r_xy = np.linspace(model.basis_xy.r_min, model.basis_xy.r_max, n_points)
    phi_xy = model.basis_xy.evaluate(r_xy)
    col_xy = n_panels_per_model - 1
    axes_mode_k[row, col_xy].plot(r_xy, phi_xy @ model.theta_xy, color="C1", linewidth=2)
    axes_mode_k[row, col_xy].axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    axes_mode_k[row, col_xy].set_xlabel("Distance (um)")
    axes_mode_k[row, col_xy].set_ylabel("Force")
    axes_mode_k[row, col_xy].set_title(f"mode={mode}  |  chrom-partner kernel")

fig_mode_kernels.suptitle("Sweep 2 — Kernel shapes by estimator mode", fontsize=13)
fig_mode_kernels.tight_layout()
plt.show()

best_mode = min(cv_mode, key=lambda k: cv_mode[k].mean_error)
best_mode_rollout = min(rollout_mode, key=lambda k: rollout_path_score(rollout_mode[k]))
print(f"\nBest estimator mode by 1-step CV: {best_mode}  (CV MSE = {cv_mode[best_mode].mean_error:.4e})")
print(
    f"Best estimator mode by rollout CV: {best_mode_rollout}"
    f"  (path MSE = {rollout_path_score(rollout_mode[best_mode_rollout]):.4e})"
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
        f"    rollout MSE = {rollout_path_score(rollout_endpoint[label]):.4e}"
        f" ± {rollout_path_se(rollout_endpoint[label]):.4e}"
        f"  endpoint = {np.nanmean(rollout_endpoint[label].endpoint_mean_error):.4e}"
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
        f"    rollout MSE = {rollout_path_score(rollout_endpoint['end_sep']):.4e}"
        f" ± {rollout_path_se(rollout_endpoint['end_sep']):.4e}"
        f"  endpoint = {np.nanmean(rollout_endpoint['end_sep'].endpoint_mean_error):.4e}"
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
    endpoint_rollout_means = [rollout_path_score(rollout_endpoint[k]) for k in endpoint_labels]
    endpoint_rollout_stds = [rollout_path_se(rollout_endpoint[k]) for k in endpoint_labels]
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
    ax_ep_rollout.set_ylabel("Leave-one-out rollout path MSE")
    ax_ep_rollout.set_title(
        f"Sweep 3 — Endpoint rollout CV comparison  ({ROLLOUT_REPS} reps/fold)"
    )
    fig_ep_rollout.tight_layout()
    plt.show()

    best_endpoint = min(cv_endpoint, key=lambda k: cv_endpoint[k].mean_error)
    best_endpoint_rollout = min(
        rollout_endpoint, key=lambda k: rollout_path_score(rollout_endpoint[k])
    )
    print(f"\nBest endpoint by 1-step CV: {best_endpoint}  (CV MSE = {cv_endpoint[best_endpoint].mean_error:.4e})")
    print(
        f"Best endpoint by rollout CV: {best_endpoint_rollout}"
        f"  (path MSE = {rollout_path_score(rollout_endpoint[best_endpoint_rollout]):.4e})"
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
print("=" * 60)
print("Hyperparameter sensitivity summary")
print("=" * 60)
print(f"{'Parameter':<30} {'Best value':<20} {'Primary score'}")
print("-" * 60)

# Joint grid best
best_grid_score = rollout_path_score(grid_rollout[best_key])
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
    f"  {'basis_eval_mode (rollout)':<28} {best_mode_rollout:<20} "
    f"{rollout_path_score(rollout_mode[best_mode_rollout]):.4e}"
)

# Endpoint method
if cv_endpoint:
    print(
        f"  {'endpoint_method (rollout)':<28} {best_endpoint_rollout:<20} "
        f"{rollout_path_score(rollout_endpoint[best_endpoint_rollout]):.4e}"
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

print("=" * 60)
