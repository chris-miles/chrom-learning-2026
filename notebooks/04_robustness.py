# %% [markdown]
# # 04 — Robustness & Hyperparameter Sensitivity
#
# Notebook 03 selected a winning interaction topology.  Here we test how
# sensitive the fitted kernels and cross-validation error are to the
# hyperparameter choices: basis size, regularisation strengths, estimator
# discretisation mode, trajectory endpoint method, and diffusion estimation
# mode.  The goal is to identify which choices matter and which do not.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in dir()
    else Path("..").resolve()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chromlearn.io.catalog import load_condition
from chromlearn.io.trajectory import trim_trajectory
from chromlearn.model_fitting import FitConfig
from chromlearn.model_fitting.basis import BSplineBasis
from chromlearn.model_fitting.diffusion import local_diffusion_estimates
from chromlearn.model_fitting.features import build_design_matrix
from chromlearn.model_fitting.fit import CVResult, cross_validate, fit_kernels, fit_model
from chromlearn.model_fitting.plotting import plot_cv_curve, plot_kernels

plt.rcParams["figure.dpi"] = 110

# %% [markdown]
# ## Setup
#
# Load cells and define the winning topology from Notebook 03.
# Update `WINNING_TOPOLOGY` below after running that notebook.

# %%
# Update this after consulting Notebook 03 results.
WINNING_TOPOLOGY = "poles"

CONDITION = "rpe18_ctr"
cells = load_condition(CONDITION)
print(f"Loaded {len(cells)} cells for condition '{CONDITION}'.")

BASE_CONFIG = FitConfig(
    topology=WINNING_TOPOLOGY,
    n_basis_xx=10,
    n_basis_xy=10,
    lambda_ridge=1e-3,
    lambda_rough=1e-3,
    basis_eval_mode="ito",
    endpoint_method="midpoint_neb_ao",
    diffusion_mode="msd",
    dt=5.0,
)

# %% [markdown]
# ## Sweep 1: Basis size
#
# How many B-spline basis functions do we need?  Too few under-fits the kernel
# shape; too many overfits and inflates CV error.  We sweep `n_basis` jointly
# for both the xx and xy kernels (they share the same count for simplicity).

# %%
N_BASIS_VALUES = [4, 6, 8, 10, 12, 16, 20]

cv_basis: dict[str, CVResult] = {}
for n in N_BASIS_VALUES:
    cfg = FitConfig(
        topology=BASE_CONFIG.topology,
        n_basis_xx=n,
        n_basis_xy=n,
        r_min_xx=BASE_CONFIG.r_min_xx,
        r_max_xx=BASE_CONFIG.r_max_xx,
        r_min_xy=BASE_CONFIG.r_min_xy,
        r_max_xy=BASE_CONFIG.r_max_xy,
        basis_type=BASE_CONFIG.basis_type,
        lambda_ridge=BASE_CONFIG.lambda_ridge,
        lambda_rough=BASE_CONFIG.lambda_rough,
        basis_eval_mode=BASE_CONFIG.basis_eval_mode,
        endpoint_method=BASE_CONFIG.endpoint_method,
        diffusion_mode=BASE_CONFIG.diffusion_mode,
        dt=BASE_CONFIG.dt,
    )
    cv_basis[str(n)] = cross_validate(cells, cfg)
    print(
        f"  n_basis={n:3d}  CV = {cv_basis[str(n)].mean_error:.4e}"
        f" ± {cv_basis[str(n)].std_error:.4e}"
    )

# %%
fig_basis, ax_basis = plt.subplots(figsize=(7, 4))
n_vals = [int(k) for k in cv_basis]
means = [cv_basis[k].mean_error for k in cv_basis]
stds = [cv_basis[k].std_error for k in cv_basis]
ax_basis.errorbar(n_vals, means, yerr=stds, fmt="o-", capsize=4, color="C0")
ax_basis.set_xlabel("Number of basis functions per kernel")
ax_basis.set_ylabel("Leave-one-out CV MSE")
ax_basis.set_title("Sweep 1 — Basis size sensitivity")
ax_basis.set_xticks(n_vals)
fig_basis.tight_layout()
plt.show()

best_n_basis = int(min(cv_basis, key=lambda k: cv_basis[k].mean_error))
print(f"\nBest n_basis: {best_n_basis}  (CV MSE = {cv_basis[str(best_n_basis)].mean_error:.4e})")

# %% [markdown]
# ## Sweep 2: Regularisation
#
# We perform two sub-sweeps:
# 1. Ridge penalty `lambda_ridge` on a log-spaced grid, with `lambda_rough` fixed.
# 2. Roughness penalty `lambda_rough` on a log-spaced grid, with `lambda_ridge` fixed.

# %%
LAMBDA_GRID = np.logspace(-6, 1, 20)
FIXED_RIDGE = 1e-3
FIXED_ROUGH = 1e-3

cv_ridge: dict[str, CVResult] = {}
for lam in LAMBDA_GRID:
    cfg = FitConfig(
        topology=BASE_CONFIG.topology,
        n_basis_xx=BASE_CONFIG.n_basis_xx,
        n_basis_xy=BASE_CONFIG.n_basis_xy,
        r_min_xx=BASE_CONFIG.r_min_xx,
        r_max_xx=BASE_CONFIG.r_max_xx,
        r_min_xy=BASE_CONFIG.r_min_xy,
        r_max_xy=BASE_CONFIG.r_max_xy,
        basis_type=BASE_CONFIG.basis_type,
        lambda_ridge=lam,
        lambda_rough=FIXED_ROUGH,
        basis_eval_mode=BASE_CONFIG.basis_eval_mode,
        endpoint_method=BASE_CONFIG.endpoint_method,
        diffusion_mode=BASE_CONFIG.diffusion_mode,
        dt=BASE_CONFIG.dt,
    )
    key = f"{lam:.2e}"
    cv_ridge[key] = cross_validate(cells, cfg)

cv_rough: dict[str, CVResult] = {}
for lam in LAMBDA_GRID:
    cfg = FitConfig(
        topology=BASE_CONFIG.topology,
        n_basis_xx=BASE_CONFIG.n_basis_xx,
        n_basis_xy=BASE_CONFIG.n_basis_xy,
        r_min_xx=BASE_CONFIG.r_min_xx,
        r_max_xx=BASE_CONFIG.r_max_xx,
        r_min_xy=BASE_CONFIG.r_min_xy,
        r_max_xy=BASE_CONFIG.r_max_xy,
        basis_type=BASE_CONFIG.basis_type,
        lambda_ridge=FIXED_RIDGE,
        lambda_rough=lam,
        basis_eval_mode=BASE_CONFIG.basis_eval_mode,
        endpoint_method=BASE_CONFIG.endpoint_method,
        diffusion_mode=BASE_CONFIG.diffusion_mode,
        dt=BASE_CONFIG.dt,
    )
    key = f"{lam:.2e}"
    cv_rough[key] = cross_validate(cells, cfg)

# %%
fig_reg, axes_reg = plt.subplots(1, 2, figsize=(12, 4))

ridge_lams = LAMBDA_GRID
ridge_means = [cv_ridge[f"{l:.2e}"].mean_error for l in ridge_lams]
ridge_stds = [cv_ridge[f"{l:.2e}"].std_error for l in ridge_lams]
axes_reg[0].errorbar(ridge_lams, ridge_means, yerr=ridge_stds, fmt="o-", capsize=3, color="C1")
axes_reg[0].set_xscale("log")
axes_reg[0].set_xlabel("lambda_ridge")
axes_reg[0].set_ylabel("Leave-one-out CV MSE")
axes_reg[0].set_title(f"Ridge penalty  (lambda_rough = {FIXED_ROUGH:.0e})")

rough_lams = LAMBDA_GRID
rough_means = [cv_rough[f"{l:.2e}"].mean_error for l in rough_lams]
rough_stds = [cv_rough[f"{l:.2e}"].std_error for l in rough_lams]
axes_reg[1].errorbar(rough_lams, rough_means, yerr=rough_stds, fmt="o-", capsize=3, color="C2")
axes_reg[1].set_xscale("log")
axes_reg[1].set_xlabel("lambda_rough")
axes_reg[1].set_ylabel("Leave-one-out CV MSE")
axes_reg[1].set_title(f"Roughness penalty  (lambda_ridge = {FIXED_RIDGE:.0e})")

fig_reg.suptitle("Sweep 2 — Regularisation sensitivity", fontsize=13)
fig_reg.tight_layout()
plt.show()

best_ridge = float(ridge_lams[int(np.argmin(ridge_means))])
best_rough = float(rough_lams[int(np.argmin(rough_means))])
print(f"Best lambda_ridge: {best_ridge:.2e}  (CV MSE = {min(ridge_means):.4e})")
print(f"Best lambda_rough: {best_rough:.2e}  (CV MSE = {min(rough_means):.4e})")

# %% [markdown]
# ## Sweep 3: Estimator mode (Ito / Ito-shift / Stratonovich)
#
# The design matrix can be built with three different discretisation
# conventions.  "ito" uses current positions; "ito_shift" uses the previous
# positions (which decorrelates localisation noise from the velocity estimate);
# "strato" uses the midpoint between consecutive frames (Stratonovich
# convention).  Here we compare CV error and kernel shape side by side.

# %%
ESTIMATOR_MODES = ["ito", "ito_shift", "strato"]

cv_mode: dict[str, CVResult] = {}
models_mode: dict[str, object] = {}

for mode in ESTIMATOR_MODES:
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
        diffusion_mode=BASE_CONFIG.diffusion_mode,
        dt=BASE_CONFIG.dt,
    )
    cv_mode[mode] = cross_validate(cells, cfg)
    models_mode[mode] = fit_model(cells, cfg)
    print(
        f"  mode={mode:12s}  CV = {cv_mode[mode].mean_error:.4e}"
        f" ± {cv_mode[mode].std_error:.4e}"
    )

# %%
fig_mode_cv = plot_cv_curve(cv_mode)
fig_mode_cv.axes[0].set_title("Sweep 3 — Estimator mode CV comparison")
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

fig_mode_kernels.suptitle("Sweep 3 — Kernel shapes by estimator mode", fontsize=13)
fig_mode_kernels.tight_layout()
plt.show()

best_mode = min(cv_mode, key=lambda k: cv_mode[k].mean_error)
print(f"\nBest estimator mode: {best_mode}  (CV MSE = {cv_mode[best_mode].mean_error:.4e})")

# %% [markdown]
# ## Sweep 4: Endpoint method
#
# The trajectory is trimmed from NEB to some endpoint.  We compare three
# strategies:
# - `"midpoint_neb_ao"` — half-way between NEB and mean AO (default)
# - `"ao_mean"` — mean of the two anaphase-onset annotations
# - `"end_sep"` — first frame where spindle separation reaches 95% of its
#   metaphase plateau
#
# Each method may yield a different number of usable cells (some cells produce
# too-short windows and are skipped with a warning).

# %%
ENDPOINT_METHODS = ["midpoint_neb_ao", "ao_mean", "end_sep"]

# Load the raw CellData so we can re-trim with different methods.
from chromlearn.io.catalog import load_condition as _load_condition  # noqa: E402
from chromlearn.io.loader import CellData  # noqa: E402

# load_condition returns TrimmedCell objects; we need raw CellData.
# Re-load raw cells from the catalog.
from chromlearn.io.catalog import CONDITIONS, list_cells  # noqa: E402
from chromlearn.io.loader import load_cell  # noqa: E402

DATA_DIR = ROOT / "data"
raw_cell_ids = list_cells(CONDITION)
raw_cells: list[CellData] = []
for cid in raw_cell_ids:
    try:
        raw_cells.append(load_cell(DATA_DIR / f"{cid}.mat"))
    except Exception as exc:
        print(f"  Warning: could not load {cid}: {exc}")

print(f"Loaded {len(raw_cells)} raw CellData objects.")

# %%
cv_endpoint: dict[str, CVResult] = {}
n_cells_endpoint: dict[str, int] = {}

for method in ENDPOINT_METHODS:
    trimmed_method = []
    for raw_cell in raw_cells:
        try:
            trimmed_method.append(trim_trajectory(raw_cell, method=method))
        except ValueError as exc:
            print(f"  Skipping {raw_cell.cell_id} for method='{method}': {exc}")

    n_cells_endpoint[method] = len(trimmed_method)
    if len(trimmed_method) < 3:
        print(
            f"  method='{method}': only {len(trimmed_method)} cells — "
            "skipping CV (too few)."
        )
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
        endpoint_method=method,
        diffusion_mode=BASE_CONFIG.diffusion_mode,
        dt=BASE_CONFIG.dt,
    )
    cv_endpoint[method] = cross_validate(trimmed_method, cfg)
    print(
        f"  method='{method}'  n_cells={len(trimmed_method)}"
        f"  CV = {cv_endpoint[method].mean_error:.4e}"
        f" ± {cv_endpoint[method].std_error:.4e}"
    )

# %%
if cv_endpoint:
    fig_ep = plot_cv_curve(cv_endpoint)
    fig_ep.axes[0].set_title("Sweep 4 — Endpoint method CV comparison")
    plt.show()

    best_endpoint = min(cv_endpoint, key=lambda k: cv_endpoint[k].mean_error)
    print(f"\nBest endpoint method: {best_endpoint}  (CV MSE = {cv_endpoint[best_endpoint].mean_error:.4e})")
else:
    print("No endpoint methods produced enough cells for CV.")

# Print summary of cell counts per method
print("\nCell counts per endpoint method:")
for method in ENDPOINT_METHODS:
    print(f"  {method:20s}: {n_cells_endpoint.get(method, 0)} cells")

# %% [markdown]
# ## Sweep 5: Diffusion estimation mode
#
# The SFI approach requires an estimate of the scalar diffusion coefficient D.
# Four estimators are available:
# - `"msd"` — naive MSD from displacement variance
# - `"vestergaard"` — 3-point estimator, robust to localisation noise
# - `"weak_noise"` — 3-point estimator, robust to drift
# - `"f_corrected"` — force-subtracted; needs a preliminary fit to remove the
#   deterministic displacement before estimating the noise variance
#
# We compare the scalar D values returned by each mode and assess whether the
# choice changes the inferred kernel significantly.

# %%
# First, build the design matrix and fit kernels to get a FitResult for
# f_corrected mode.
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
        if _basis_xx_fc is None:
            # f_corrected needs basis_xx; for topologies without it, skip gracefully.
            print(
                "  f_corrected: skipping because topology has no chrom-chrom kernel "
                "(basis_xx is None)."
            )
            continue
        d_estimates = local_diffusion_estimates(
            cells,
            dt=BASE_CONFIG.dt,
            mode=diff_mode,
            fit_result=_fit_result_fc,
            basis_xx=_basis_xx_fc,
            basis_xy=_basis_xy_fc,
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
ax_diff.set_title("Sweep 5 — Diffusion estimation mode comparison")
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
#
# Collect the best hyperparameter from each sweep and print a summary table.

# %%
print("=" * 60)
print("Hyperparameter sensitivity summary")
print("=" * 60)
print(f"{'Parameter':<30} {'Best value':<20} {'CV MSE'}")
print("-" * 60)

# Basis size
print(
    f"  {'n_basis':<28} {best_n_basis:<20} "
    f"{cv_basis[str(best_n_basis)].mean_error:.4e}"
)

# Regularisation
print(
    f"  {'lambda_ridge':<28} {best_ridge:<20.2e} "
    f"{min(ridge_means):.4e}"
)
print(
    f"  {'lambda_rough':<28} {best_rough:<20.2e} "
    f"{min(rough_means):.4e}"
)

# Estimator mode
print(
    f"  {'basis_eval_mode':<28} {best_mode:<20} "
    f"{cv_mode[best_mode].mean_error:.4e}"
)

# Endpoint method
if cv_endpoint:
    print(
        f"  {'endpoint_method':<28} {best_endpoint:<20} "
        f"{cv_endpoint[best_endpoint].mean_error:.4e}"
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
print()
print("Interpretation notes:")
print(
    "  - Basis size: CV error should plateau; choose the smallest n_basis "
    "where the curve flattens."
)
print(
    "  - Regularisation: broad flat minima indicate the fit is not highly "
    "sensitive to exact lambda values."
)
print(
    "  - Estimator mode: 'ito_shift' decorrelates localisation noise; "
    "prefer it if CV error is lower."
)
print(
    "  - Endpoint method: 'midpoint_neb_ao' is conservative (fewer frames); "
    "'ao_mean' uses the full metaphase window."
)
print(
    "  - Diffusion mode: large spread across modes indicates localisation "
    "noise or drift bias; 'vestergaard' is a good default."
)
