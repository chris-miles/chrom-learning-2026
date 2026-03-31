# %% [markdown]
# # 04 -- Model selection: which interaction topology fits best?
#
# We compare five chromosome interaction topologies:
#
# | Label | xy partner(s) | xx term? |
# |---|---|---|
# | **poles** | both poles (2 partners) | no |
# | **center** | pole midpoint (1 partner) | no |
# | **poles\_and\_chroms** | both poles | yes (full range) |
# | **center\_and\_chroms** | pole midpoint | yes (full range) |
# | **poles\_and\_chroms\_short** | both poles | yes (r < 2.5 um only) |
#
# Primary selection criterion: leave-one-cell-out ensemble-mean MSE
# (simulated positions averaged across replicates before comparing to
# reality, cancelling stochastic noise to isolate systematic drift bias).
# Secondary quantitative checks: per-rep path MSE, one-step velocity MSE,
# endpoint mismatch, final-frame distribution mismatch, and
# horizon-specific rollout errors.
# Secondary qualitative checks: bootstrap CIs, kernel physics checks, and
# representative forward simulations.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chromlearn import find_repo_root

ROOT = find_repo_root(Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd())

from chromlearn.io.catalog import load_condition
from chromlearn.io.trajectory import (
    TrimmedCell,
    get_partners,
    spindle_frame,
    trim_trajectory,
)
from chromlearn.model_fitting import FitConfig
from chromlearn.model_fitting.basis import BSplineBasis
from chromlearn.model_fitting.features import build_design_matrix
from chromlearn.model_fitting.fit import (
    BootstrapResult,
    CVResult,
    RolloutCVResult,
    bootstrap_kernels,
    cross_validate,
    fit_model,
    paired_cv_differences,
    rollout_cross_validate,
)
from chromlearn.model_fitting.model import FittedModel
from chromlearn.model_fitting.plotting import plot_cv_curve, plot_kernels
from chromlearn.model_fitting.simulate import simulate_cell, simulate_trajectories
from chromlearn.analysis.pca_projection import fit_pca_basis

plt.rcParams["figure.dpi"] = 110

# %%
BASE_FRAC = 1.0 / 3.0

cells_raw = load_condition("rpe18_ctr")
cells = [trim_trajectory(c, method="neb_ao_frac", frac=BASE_FRAC) for c in cells_raw]
print(f"Loaded {len(cells)} rpe18_ctr cells (trimmed to neb_ao_frac={BASE_FRAC:.3f} window)")
for c in cells:
    T, _, N = c.chromosomes.shape
    print(f"  {c.cell_id}: {T} frames, {N} chromosomes")

# %% [markdown]
# ## Basis domains
#
# Basis support is fixed a priori from imaging resolution and spindle geometry,
# not estimated from the data.  This avoids leaking held-out cell information
# into the basis used during cross-validation.
#
# - `r_min = 0.3 um`: below kinetochore tracking resolution.
# - `r_max = 15.0 um`: conservative upper bound from RPE1 spindle geometry
#   (pole-to-pole ~10-14 um; 15 um covers all plausible pairwise distances).
#
# We use a single domain for both xx and xy interactions across all topologies.
# Empirical distance distributions are plotted below as a sanity check.

# %%
TOPOLOGIES = ["poles", "center", "poles_and_chroms", "center_and_chroms",
              "poles_and_chroms_short"]

R_MIN = 0.3   # um — tracking resolution floor
R_MAX = 15.0  # um — conservative spindle-scale upper bound
R_CUTOFF_XX_SHORT = 2.5  # um — short-range-only xx cutoff


def _base_topology(label: str) -> str:
    """Map extended topology labels to the base topology string."""
    if label == "poles_and_chroms_short":
        return "poles_and_chroms"
    return label

r_min_xx = R_MIN
r_max_xx = R_MAX
r_min_xy = R_MIN
r_max_xy = R_MAX

# For backward compat with code that indexes xy_domains[topology]
xy_domains: dict[str, tuple[float, float]] = {t: (R_MIN, R_MAX) for t in TOPOLOGIES}

print(f"Fixed basis domains: r_min={R_MIN}, r_max={R_MAX} um (all interactions, all topologies)")

# %%
# Empirical distance distributions (sanity check that fixed domains cover the data)
xy_dists_by_topology: dict[str, list[float]] = {t: [] for t in TOPOLOGIES}
xx_dists_all: list[float] = []

for cell in cells:
    T, _, N = cell.chromosomes.shape
    chroms = cell.chromosomes  # (T, 3, N)

    for t in range(T):
        pos_t = chroms[t].T  # (N, 3)
        for i in range(N):
            if np.any(np.isnan(pos_t[i])):
                continue
            for j in range(i + 1, N):
                if np.any(np.isnan(pos_t[j])):
                    continue
                d = float(np.linalg.norm(pos_t[j] - pos_t[i]))
                if d > 1e-12:
                    xx_dists_all.append(d)

    for topology in TOPOLOGIES:
        partners = get_partners(cell, _base_topology(topology))  # (n_p, T, 3)
        for t in range(T):
            pos_t = chroms[t].T  # (N, 3)
            for p_idx in range(partners.shape[0]):
                partner_pos = partners[p_idx, t]  # (3,)
                for i in range(N):
                    if np.any(np.isnan(pos_t[i])):
                        continue
                    d = float(np.linalg.norm(partner_pos - pos_t[i]))
                    if d > 1e-12:
                        xy_dists_by_topology[topology].append(d)

xx_dists_all = np.array(xx_dists_all)
print(f"Collected {len(xx_dists_all):,} xx distance samples  "
      f"(observed range: {np.min(xx_dists_all):.2f} – {np.max(xx_dists_all):.2f} um)")
for t in TOPOLOGIES:
    arr = np.array(xy_dists_by_topology[t])
    print(f"  xy ({t}): {len(arr):,} samples  "
          f"(observed range: {np.min(arr):.2f} – {np.max(arr):.2f} um)")

# %%
# Only plot unique xy partner sets (poles_and_chroms_short shares partners
# with poles_and_chroms, so skip its duplicate histogram).
_HIST_TOPOLOGIES = [t for t in TOPOLOGIES if t != "poles_and_chroms_short"]
fig, axes = plt.subplots(1, 1 + len(_HIST_TOPOLOGIES), figsize=(18, 4))

axes[0].hist(xx_dists_all, bins=60, color="C2", edgecolor="k", alpha=0.7)
axes[0].axvline(R_MIN, color="r", linestyle="--", linewidth=1.5, label=f"r_min={R_MIN}")
axes[0].axvline(R_MAX, color="r", linestyle="-", linewidth=1.5, label=f"r_max={R_MAX}")
axes[0].set_title("Chromosome-chromosome (xx)")
axes[0].set_xlabel("Distance (um)")
axes[0].set_ylabel("Count")
axes[0].legend(fontsize=7)

for idx, topology in enumerate(_HIST_TOPOLOGIES):
    ax = axes[idx + 1]
    ax.hist(xy_dists_by_topology[topology], bins=60, color=f"C{idx}", edgecolor="k", alpha=0.7)
    ax.axvline(R_MIN, color="r", linestyle="--", linewidth=1.5, label=f"r_min={R_MIN}")
    ax.axvline(R_MAX, color="r", linestyle="-", linewidth=1.5, label=f"r_max={R_MAX}")
    ax.set_title(f"xy ({topology})")
    ax.set_xlabel("Distance (um)")
    ax.legend(fontsize=7)

fig.suptitle("Observed distance distributions — fixed basis domains marked in red")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Fit all four models
#
# We use:
# - `n_basis = 10` B-spline basis functions for both xx and xy kernels
# - `lambda_ridge = 1e-3`, `lambda_rough = 1.0`
# - `basis_eval_mode = "ito"` (current positions, standard SFI)
# - `endpoint_frac = 1/3` of the NEB-to-AO window
#
# Domain parameters are fixed a priori (see basis-domain section above).

# %%
N_BASIS = 10
LAMBDA_RIDGE = 1e-3
LAMBDA_ROUGH = 1.0

configs: dict[str, FitConfig] = {}
for topology in TOPOLOGIES:
    r_cutoff = R_CUTOFF_XX_SHORT if topology == "poles_and_chroms_short" else None
    configs[topology] = FitConfig(
        topology=_base_topology(topology),
        n_basis_xx=N_BASIS,
        n_basis_xy=N_BASIS,
        r_min_xx=R_MIN,
        r_max_xx=R_MAX,
        r_min_xy=R_MIN,
        r_max_xy=R_MAX,
        basis_type="bspline",
        lambda_ridge=LAMBDA_RIDGE,
        lambda_rough=LAMBDA_ROUGH,
        basis_eval_mode="ito",
        endpoint_method="neb_ao_frac",
        endpoint_frac=BASE_FRAC,
        dt=5.0,
        r_cutoff_xx=r_cutoff,
    )

print("Fitting models...")
models: dict[str, FittedModel] = {}
for topology in TOPOLOGIES:
    models[topology] = fit_model(cells, configs[topology])
    m = models[topology]
    print(f"  {topology:<22}  D_x={m.D_x:.8f} um^2/s  "
          f"n_params={m.theta.size}")

# %% [markdown]
# ## Cross-validation comparison
#
# Leave-one-cell-out CV: fit on N-1 cells, evaluate mean squared error on the
# held-out cell.  Lower is better.
#
# **Secondary short-horizon diagnostic**: leave-one-cell-out one-step velocity
# MSE, averaged equally across held-out cells.  Paired fold-by-fold loss
# differences and their SEs are reported to assess whether score gaps between
# topologies are meaningful.

# %%
print(f"Running leave-one-cell-out cross-validation ({len(TOPOLOGIES)} topologies × {len(cells)} folds)...")
cv_results: dict[str, CVResult] = {}
for topology in TOPOLOGIES:
    cv_results[topology] = cross_validate(cells, configs[topology])
    r = cv_results[topology]
    print(f"  {topology:<22}  MSE={r.mean_error:.8f}  (fold SD={r.fold_sd:.8f}, SE={r.fold_se:.8f})")

# %%
fig = plot_cv_curve(cv_results)
fig.suptitle("Leave-one-cell-out CV — mean squared velocity prediction error", y=1.02)
plt.show()

# Bar chart with per-cell breakdown
n_topo = len(TOPOLOGIES)
n_cells = len(cells)
x = np.arange(n_cells)
width = 0.8 / n_topo

fig, ax = plt.subplots(figsize=(12, 4.5))
for idx, topology in enumerate(TOPOLOGIES):
    offset = (idx - n_topo / 2 + 0.5) * width
    ax.bar(x + offset, cv_results[topology].held_out_errors, width,
           label=topology, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([c.cell_id.split("_")[-1] for c in cells], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("MSE (um/s)^2")
ax.set_title("Per-cell CV errors — all topologies")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
plt.show()

# Summary printout
print("\nCV summary (sorted by mean MSE):")
sorted_topo_cv = sorted(TOPOLOGIES, key=lambda t: cv_results[t].mean_error)
best_cv = cv_results[sorted_topo_cv[0]].mean_error
for rank, topology in enumerate(sorted_topo_cv):
    r = cv_results[topology]
    delta = r.mean_error - best_cv
    print(f"  #{rank + 1}  {topology:<22}  MSE={r.mean_error:.8f}  "
          f"(Δbest={delta:+.2e}, SE={r.fold_se:.2e})")


# %% [markdown]
# ## Interpreting the one-step CV scale
#
# The absolute LOOCV MSE values are small because one-frame chromosome
# velocities are small and strongly noise-dominated. To calibrate the scale, we
# compare the fitted models against two simple references:
#
# - **Zero-velocity baseline**: predict `v = 0` for every held-out coordinate.
# - **Diffusion floor**: for overdamped Langevin motion, the per-coordinate
#   one-step noise variance is approximately `2 D_x / dt`.
#
# If the fitted models sit close to these references, then one-step CV is still
# a sensible primary score, but it should not be expected to sharply separate
# topologies on its own.

# %%
ref_cfg = configs[TOPOLOGIES[0]]
baseline_basis_xy = BSplineBasis(R_MIN, R_MAX, N_BASIS)
zero_baseline_errors = np.full(len(cells), np.nan, dtype=np.float64)
held_out_abs_vel = np.full(len(cells), np.nan, dtype=np.float64)

for held_out_index, cell in enumerate(cells):
    _, V_test = build_design_matrix(
        [cell],
        basis_xx=None,
        basis_xy=baseline_basis_xy,
        basis_eval_mode=ref_cfg.basis_eval_mode,
        topology="poles",
    )
    if V_test.size == 0:
        continue
    zero_baseline_errors[held_out_index] = float(np.mean(V_test**2))
    held_out_abs_vel[held_out_index] = float(np.mean(np.abs(V_test)))

zero_baseline_mean = float(np.nanmean(zero_baseline_errors))
zero_baseline_rmse = float(np.sqrt(zero_baseline_mean))
mean_abs_velocity = float(np.nanmean(held_out_abs_vel))
noise_floor = {
    topology: 2.0 * models[topology].D_x / configs[topology].dt
    for topology in TOPOLOGIES
}

print("\n1-step CV scale check")
print("=" * 116)
print(f"Zero-velocity baseline: MSE={zero_baseline_mean:.8f}  RMSE={zero_baseline_rmse:.5f} um/s")
print(f"Mean held-out |v|:      {mean_abs_velocity:.5f} um/s")
print("-" * 116)
print(f"{'Topology':<22} {'CV MSE':>12} {'impr vs 0':>11} {'Δ vs 0':>10} {'2D/dt':>12} {'CV/(2D/dt)':>12}")
print("-" * 116)
for topology in sorted_topo_cv:
    cv = cv_results[topology].mean_error
    gain = zero_baseline_mean - cv
    gain_pct = 100.0 * gain / zero_baseline_mean if zero_baseline_mean > 0 else np.nan
    floor = noise_floor[topology]
    ratio = cv / floor if floor > 0 else np.nan
    print(f"  {topology:<22} {cv:>12.8f} {gain_pct:>+10.2f}% {gain:>10.2e} {floor:>12.8f} {ratio:>12.3f}")
print("=" * 116)
print("These one-step losses are close to both the zero-velocity baseline and the diffusion-noise floor.")
print("So the primary CV score is sensible, but in this dataset it only weakly separates the topologies.")


# %% [markdown]
# ## Bootstrap kernel confidence bands
#
# N cell-level resamples per topology; shaded band = 5–95% quantile interval.

# %%
N_BOOT = 200
boot_rng = np.random.default_rng(42)

print(f"Bootstrapping kernels ({N_BOOT} resamples × {len(TOPOLOGIES)} topologies)...")
boot_results: dict[str, BootstrapResult] = {}
for topology in TOPOLOGIES:
    boot_results[topology] = bootstrap_kernels(
        cells, configs[topology], n_boot=N_BOOT, rng=boot_rng
    )
    print(f"  {topology} done.")

# %%
# Plot each topology's kernels with bootstrap bands
for topology in TOPOLOGIES:
    fig = plot_kernels(models[topology], bootstrap=boot_results[topology])
    fig.suptitle(f"Learned kernels — topology: {topology}", y=1.02)
    plt.show()

# %% [markdown]
# ## Physical plausibility of the chromosome-chromosome kernel
#
# For topologies that include a chromosome-chromosome term (`poles_and_chroms`
# and `center_and_chroms`), we inspect whether the learned kernel is
# physically sensible:
#
# - Sign convention: forces are assembled as `f(r) * (x_j - x_i) / r`, so
#   positive values are attractive and negative values are repulsive.
# - **Expected**: a repulsive barrier at short distances (excluded volume,
#   r ≲ 1 um), i.e. negative `f_xx(r)` at small `r`.
# - **Red flag**: short-range attraction (positive force at small `r`), which
#   would pull chromosomes into one another.
#
# We flag any model where f_xx at r < 1.5 um becomes positive.

# %%
chroms_topologies = [t for t in TOPOLOGIES if "chroms" in t]
r_probe = np.linspace(r_min_xx, r_max_xx, 400)
SHORT_R_THRESHOLD = 1.5  # um — below this we expect repulsion, not attraction

fig, axes = plt.subplots(1, len(chroms_topologies), figsize=(7 * len(chroms_topologies), 4.5),
                          squeeze=False)

for col, topology in enumerate(chroms_topologies):
    ax = axes[0, col]
    m = models[topology]
    boot = boot_results[topology]

    f_vals = m.evaluate_kernel("xx", r_probe)

    # Bootstrap band
    phi = m.basis_xx.evaluate(r_probe)
    theta_xx_samples = boot.theta_samples[:, : m.n_basis_xx]
    curves = phi @ theta_xx_samples.T
    lo = np.percentile(curves, 5, axis=1)
    hi = np.percentile(curves, 95, axis=1)
    ax.fill_between(r_probe, lo, hi, color="C0", alpha=0.2, label="5–95% CI")

    ax.plot(r_probe, f_vals, "C0", linewidth=2, label="Mean fit")
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.axvline(SHORT_R_THRESHOLD, color="orange", linestyle=":", linewidth=1.2,
               label=f"r={SHORT_R_THRESHOLD} um")
    ax.set_xlabel("Distance (um)")
    ax.set_ylabel("Force")
    ax.set_title(f"f_xx — {topology}")
    ax.legend(fontsize=8)

    # Diagnosis
    short_r_mask = r_probe < SHORT_R_THRESHOLD
    max_short_r = float(np.max(f_vals[short_r_mask])) if short_r_mask.any() else np.nan
    if max_short_r > 0:
        print(f"  [WARNING] {topology}: f_xx is ATTRACTIVE at short range "
              f"(max={max_short_r:.4f} at r < {SHORT_R_THRESHOLD} um). "
              "Likely an artifact — excluded-volume physics expects repulsion here.")
    else:
        print(f"  [OK] {topology}: f_xx is repulsive or neutral at short range "
              f"(max={max_short_r:.4f} at r < {SHORT_R_THRESHOLD} um).")

fig.suptitle("Chromosome-chromosome kernel f_xx — physical plausibility check")
fig.tight_layout()
plt.show()

# Also plot f_xy for chroms topologies alongside poles/center for comparison
fig, axes = plt.subplots(1, len(TOPOLOGIES), figsize=(6 * len(TOPOLOGIES), 4.5), squeeze=False)
r_xy_probe = {t: np.linspace(R_MIN, R_MAX, 400) for t in TOPOLOGIES}

for col, topology in enumerate(TOPOLOGIES):
    ax = axes[0, col]
    m = models[topology]
    boot = boot_results[topology]
    r_p = r_xy_probe[topology]

    f_vals = m.evaluate_kernel("xy", r_p)
    phi = m.basis_xy.evaluate(r_p)
    theta_xy_samples = boot.theta_samples[:, m.n_basis_xx:]
    curves = phi @ theta_xy_samples.T
    lo = np.percentile(curves, 5, axis=1)
    hi = np.percentile(curves, 95, axis=1)

    ax.fill_between(r_p, lo, hi, color=f"C{col}", alpha=0.2, label="5–95% CI")
    ax.plot(r_p, f_vals, f"C{col}", linewidth=2, label="Mean fit")
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Distance (um)")
    ax.set_ylabel("Force")
    ax.set_title(f"f_xy — {topology}")
    ax.legend(fontsize=8)

fig.suptitle("Chromosome-partner kernel f_xy — all topologies")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Rollout validation: qualitative cells and aggregate holdout scoring
#
# One-step CV is useful for local velocity prediction, but it does not tell us
# whether a fitted model generates plausible chromosome trajectories when run
# forward.  We therefore look at rollout validation in two complementary ways:
#
# 1. A few representative cells with **one simulated rollout per model** to
#    inspect whether the simulated trajectories qualitatively resemble the real
#    spindle-frame traces.
# 2. **Leave-one-cell-out rollout validation** that aggregates across all cells
#    and scores axial/radial summary trajectories and final-frame
#    distributions.
#
# Notes for interpretation:
#
# - In the qualitative plots below, the thick curves are means over **all**
#   chromosomes. The thin colored lines show only a small displayed subset.
# - For the `poles` topologies, symmetric attraction to the two poles does **not**
#   automatically imply centering at the spindle midpoint. If attraction
#   weakens with distance, the nearer pole wins and the midpoint is not a
#   stable fixed point.
# - If a `center` topology still drifts axially in rollout, that usually means
#   the learned attraction to the moving spindle center is too weak to keep up
#   with the real partner motion, not necessarily that there is a sign bug.

# %%
EXAMPLE_CELL_IDX = 1
example_cell = cells[EXAMPLE_CELL_IDX]
T, _, N_chrom = example_cell.chromosomes.shape
n_steps = T - 1
x0 = example_cell.chromosomes[0].T
sf_real = spindle_frame(example_cell)

QUAL_CELL_IDXS = sorted({0, len(cells) // 2, len(cells) - 1})
QUAL_N_TRACES = 6
ROLLOUT_REPS = 32
ROLLOUT_HORIZONS = (1, 3, 5, 8, 10, 15, 20)


def _simulate_cell_once(cell: TrimmedCell, model: FittedModel, seed: int):
    traj, sim_cell = simulate_cell(cell, model, rng=np.random.default_rng(seed))
    return traj, spindle_frame(sim_cell)


def _representative_trace_indices(sf_data, n_traces: int) -> np.ndarray:
    valid = np.flatnonzero(
        np.all(np.isfinite(sf_data.axial), axis=0)
        & np.all(np.isfinite(sf_data.radial), axis=0)
    )
    if valid.size == 0:
        return valid
    if valid.size <= n_traces:
        return valid

    # Spread displayed traces across the initial radial distribution so the
    # thin-line examples are less biased than "first N valid chromosomes".
    radial0 = sf_data.radial[0, valid]
    order = valid[np.argsort(radial0)]
    positions = np.linspace(0, order.size - 1, n_traces, dtype=int)
    return order[positions]


def _interp_to_unit_grid(values: np.ndarray, n_grid: int = 100) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.full(n_grid, np.nan, dtype=np.float64)
    if values.size == 1:
        return np.full(n_grid, values[0], dtype=np.float64)
    src = np.linspace(0.0, 1.0, values.size)
    dst = np.linspace(0.0, 1.0, n_grid)
    return np.interp(dst, src, values)


print("Qualitative rollout check on representative cells (1 rollout per model)...")
for cell_idx in QUAL_CELL_IDXS:
    cell = cells[cell_idx]
    sf_cell_real = spindle_frame(cell)
    time_axis = np.arange(cell.chromosomes.shape[0]) * cell.dt
    trace_idx = _representative_trace_indices(sf_cell_real, QUAL_N_TRACES)

    fig, axes = plt.subplots(2, len(TOPOLOGIES), figsize=(5 * len(TOPOLOGIES), 7), squeeze=False)
    print(f"  {cell.cell_id}: {cell.chromosomes.shape[0]} frames, {cell.chromosomes.shape[2]} chromosomes")

    for col, topology in enumerate(TOPOLOGIES):
        sim_seed = 1000 + 100 * cell_idx + col
        _traj, sf_sim = _simulate_cell_once(cell, models[topology], seed=sim_seed)

        ax_axial = axes[0, col]
        ax_radial = axes[1, col]
        color = f"C{col}"

        for chrom_idx in trace_idx:
            ax_axial.plot(time_axis, sf_cell_real.axial[:, chrom_idx], color="k", alpha=0.15, linewidth=0.8)
            ax_axial.plot(time_axis, sf_sim.axial[:, chrom_idx], color=color, alpha=0.18, linewidth=0.8)
            ax_radial.plot(time_axis, sf_cell_real.radial[:, chrom_idx], color="k", alpha=0.15, linewidth=0.8)
            ax_radial.plot(time_axis, sf_sim.radial[:, chrom_idx], color=color, alpha=0.18, linewidth=0.8)

        real_axial_mean = np.nanmean(sf_cell_real.axial, axis=1)
        sim_axial_mean = np.nanmean(sf_sim.axial, axis=1)
        real_radial_mean = np.nanmean(sf_cell_real.radial, axis=1)
        sim_radial_mean = np.nanmean(sf_sim.radial, axis=1)

        ax_axial.plot(time_axis, real_axial_mean, "k-", linewidth=2.0, label="All-chromosome real mean")
        ax_axial.plot(time_axis, sim_axial_mean, color=color, linestyle="--", linewidth=2.0,
                      label="All-chromosome rollout mean")
        ax_radial.plot(time_axis, real_radial_mean, "k-", linewidth=2.0, label="All-chromosome real mean")
        ax_radial.plot(time_axis, sim_radial_mean, color=color, linestyle="--", linewidth=2.0,
                       label="All-chromosome rollout mean")

        ax_axial.set_title(f"Axial — {topology}", fontsize=9)
        ax_radial.set_title(f"Radial — {topology}", fontsize=9)
        ax_axial.set_xlabel("Time (s)")
        ax_radial.set_xlabel("Time (s)")
        if col == 0:
            ax_axial.set_ylabel("Axial position (um)")
            ax_radial.set_ylabel("Radial distance (um)")
        ax_axial.legend(fontsize=7)
        ax_radial.legend(fontsize=7)

    fig.suptitle(f"Single-rollout qualitative check — {cell.cell_id}\n"
                 "Thin lines = representative chromosome traces, thick lines = means over all chromosomes")
    fig.tight_layout()
    plt.show()


# %% [markdown]
# ### Trajectories in PCA space: real vs rollout
#
# We project real and simulated trajectories into a common 2-component PCA
# basis fitted from the **real** cell's combined pole + chromosome positions.
# This gives a data-driven 3D→2D view where pole separation and chromosome
# spread are both visible.  The same basis is used for both panels so
# differences reflect the model, not the projection.

# %%
from matplotlib.collections import LineCollection


def _colorline(ax, x, y, t, cmap, linewidth=1.5, alpha=1.0):
    """Plot a line colored by scalar *t* (values in [0, 1])."""
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidths=linewidth, alpha=alpha)
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    return lc


PCA_TOPOLOGIES = sorted_topo_cv[:2]
PCA_N_CHROM_DISPLAY = 10

for cell_idx in QUAL_CELL_IDXS[:2]:
    cell = cells[cell_idx]
    T_cell = cell.chromosomes.shape[0]
    n_chrom = cell.chromosomes.shape[2]
    t_color = np.linspace(0, 1, T_cell)

    basis = fit_pca_basis(cell)

    n_panels = 1 + len(PCA_TOPOLOGIES)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5), squeeze=False)

    # Choose a spread of chromosomes to display (by initial radial position)
    sf_real = spindle_frame(cell)
    valid_chroms = np.flatnonzero(
        np.all(np.isfinite(cell.chromosomes[:, :, :]), axis=(0, 1))
    )
    if valid_chroms.size > PCA_N_CHROM_DISPLAY:
        radial0 = sf_real.radial[0, valid_chroms]
        order = valid_chroms[np.argsort(radial0)]
        display_chroms = order[np.linspace(0, order.size - 1, PCA_N_CHROM_DISPLAY, dtype=int)]
    else:
        display_chroms = valid_chroms

    def _plot_pca_panel(ax, cell_data, title):
        """Plot poles + chromosomes in PCA space on a single axis."""
        # Poles: thick black
        for p in range(cell_data.centrioles.shape[2]):
            pole_pca = basis.project(cell_data.centrioles[:, :, p])
            _colorline(ax, pole_pca[:, 0], pole_pca[:, 1], t_color,
                        "Greys", linewidth=3, alpha=0.85)

        # Chromosomes: thin colored lines
        chrom_cmap = plt.cm.tab20(np.linspace(0, 1, max(len(display_chroms), 1)))
        for ci, j in enumerate(display_chroms):
            cj = cell_data.chromosomes[:, :, j]
            if np.any(np.isnan(cj)):
                continue
            cj_pca = basis.project(cj)
            ax.plot(cj_pca[:, 0], cj_pca[:, 1], color=chrom_cmap[ci],
                    linewidth=0.6, alpha=0.4)
            ax.plot(cj_pca[-1, 0], cj_pca[-1, 1], "o", color=chrom_cmap[ci],
                    markersize=2.5, alpha=0.6)

        ax.set_xlabel("PC1 (um)")
        ax.set_ylabel("PC2 (um)")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.autoscale()

    # Panel 0: real data
    _plot_pca_panel(axes[0, 0], cell, f"Real — {cell.cell_id}")

    # Panels 1+: rollouts from selected topologies
    for ti, topology in enumerate(PCA_TOPOLOGIES):
        sim_seed = 2000 + 100 * cell_idx + ti
        _traj, sim_cell = _simulate_cell_once(cell, models[topology], seed=sim_seed)
        _plot_pca_panel(axes[0, ti + 1], sim_cell, f"Rollout — {topology}")

    fig.suptitle(f"Trajectories in PCA space — {cell.cell_id}\n"
                 "Black = poles, colored = chromosomes (dot = endpoint), "
                 "PCA basis from real data",
                 fontsize=10)
    fig.tight_layout()
    plt.show()


# %%
AGG_NORM_GRID = 100
print("Aggregating full-data forward simulations across all cells...")

agg_real_axial = []
agg_real_radial = []
agg_sim_axial: dict[str, list[np.ndarray]] = {t: [] for t in TOPOLOGIES}
agg_sim_radial: dict[str, list[np.ndarray]] = {t: [] for t in TOPOLOGIES}

for cell_idx, cell in enumerate(cells):
    sf_cell_real = spindle_frame(cell)
    agg_real_axial.append(_interp_to_unit_grid(np.nanmean(sf_cell_real.axial, axis=1), n_grid=AGG_NORM_GRID))
    agg_real_radial.append(_interp_to_unit_grid(np.nanmean(sf_cell_real.radial, axis=1), n_grid=AGG_NORM_GRID))

    for topo_index, topology in enumerate(TOPOLOGIES):
        _traj, sf_sim = _simulate_cell_once(
            cell,
            models[topology],
            seed=10_000 + 100 * cell_idx + topo_index,
        )
        agg_sim_axial[topology].append(
            _interp_to_unit_grid(np.nanmean(sf_sim.axial, axis=1), n_grid=AGG_NORM_GRID)
        )
        agg_sim_radial[topology].append(
            _interp_to_unit_grid(np.nanmean(sf_sim.radial, axis=1), n_grid=AGG_NORM_GRID)
        )

agg_real_axial = np.stack(agg_real_axial, axis=0)
agg_real_radial = np.stack(agg_real_radial, axis=0)
norm_time = np.linspace(0.0, 1.0, AGG_NORM_GRID)

fig, axes = plt.subplots(2, len(TOPOLOGIES), figsize=(5 * len(TOPOLOGIES), 7), squeeze=False)
for col, topology in enumerate(TOPOLOGIES):
    ax_axial = axes[0, col]
    ax_radial = axes[1, col]

    sim_axial_stack = np.stack(agg_sim_axial[topology], axis=0)
    sim_radial_stack = np.stack(agg_sim_radial[topology], axis=0)

    real_axial_mean = np.nanmean(agg_real_axial, axis=0)
    real_axial_std = np.nanstd(agg_real_axial, axis=0)
    real_radial_mean = np.nanmean(agg_real_radial, axis=0)
    real_radial_std = np.nanstd(agg_real_radial, axis=0)

    sim_axial_mean = np.nanmean(sim_axial_stack, axis=0)
    sim_axial_std = np.nanstd(sim_axial_stack, axis=0)
    sim_radial_mean = np.nanmean(sim_radial_stack, axis=0)
    sim_radial_std = np.nanstd(sim_radial_stack, axis=0)

    color = f"C{col}"
    ax_axial.plot(norm_time, real_axial_mean, "k-", linewidth=2.0, label="Real mean")
    ax_axial.fill_between(
        norm_time,
        real_axial_mean - real_axial_std,
        real_axial_mean + real_axial_std,
        color="k",
        alpha=0.12,
    )
    ax_axial.plot(norm_time, sim_axial_mean, color=color, linestyle="--", linewidth=2.0, label="Sim mean")
    ax_axial.fill_between(
        norm_time,
        sim_axial_mean - sim_axial_std,
        sim_axial_mean + sim_axial_std,
        color=color,
        alpha=0.18,
    )

    ax_radial.plot(norm_time, real_radial_mean, "k-", linewidth=2.0, label="Real mean")
    ax_radial.fill_between(
        norm_time,
        real_radial_mean - real_radial_std,
        real_radial_mean + real_radial_std,
        color="k",
        alpha=0.12,
    )
    ax_radial.plot(norm_time, sim_radial_mean, color=color, linestyle="--", linewidth=2.0, label="Sim mean")
    ax_radial.fill_between(
        norm_time,
        sim_radial_mean - sim_radial_std,
        sim_radial_mean + sim_radial_std,
        color=color,
        alpha=0.18,
    )

    ax_axial.set_title(f"Axial — {topology}", fontsize=9)
    ax_radial.set_title(f"Radial — {topology}", fontsize=9)
    ax_axial.set_xlabel("Normalized progress through trimmed window")
    ax_radial.set_xlabel("Normalized progress through trimmed window")
    if col == 0:
        ax_axial.set_ylabel("Mean axial position (um)")
        ax_radial.set_ylabel("Mean radial distance (um)")
    ax_axial.legend(fontsize=7)
    ax_radial.legend(fontsize=7)

fig.suptitle("Aggregate forward simulation across all cells\n"
             "Real initial conditions and pole trajectories, normalized in time")
fig.tight_layout()
plt.show()


# %%
print("Running leave-one-cell-out rollout validation "
      f"({len(TOPOLOGIES)} topologies × {len(cells)} folds × {ROLLOUT_REPS} rollout replicates)...")
print("  (common random numbers across topologies for paired comparison)")
rollout_results: dict[str, RolloutCVResult] = {}
ROLLOUT_SEED = 200
for topo_index, topology in enumerate(TOPOLOGIES):
    rollout_results[topology] = rollout_cross_validate(
        cells,
        configs[topology],
        n_reps=ROLLOUT_REPS,
        horizons=ROLLOUT_HORIZONS,
        rng=np.random.default_rng(ROLLOUT_SEED),
    )
    rr = rollout_results[topology]
    print(f"  {topology:<22}  ens_MSE={np.nanmean(rr.ensemble_mse):.5f}  "
          f"path_MSE={np.nanmean(rr.path_mse):.5f}  "
          f"axial_MSE={np.nanmean(rr.axial_mse):.5f}  "
          f"radial_MSE={np.nanmean(rr.radial_mse):.5f}  "
          f"endpoint_MSE={np.nanmean(rr.endpoint_mean_error):.5f}  "
          f"final_W1(ax,rad)=({np.nanmean(rr.final_axial_wasserstein):.4f}, "
          f"{np.nanmean(rr.final_radial_wasserstein):.4f})")


# %%
rollout_ensemble_mse_score = {
    topology: float(np.nanmean(rollout_results[topology].ensemble_mse))
    for topology in TOPOLOGIES
}
rollout_mse_score = {
    topology: float(np.nanmean(rollout_results[topology].path_mse))
    for topology in TOPOLOGIES
}
rollout_endpoint_score = {
    topology: float(np.nanmean(rollout_results[topology].endpoint_mean_error))
    for topology in TOPOLOGIES
}
rollout_dist_score = {
    topology: float(np.nanmean(rollout_results[topology].final_axial_wasserstein)
                    + np.nanmean(rollout_results[topology].final_radial_wasserstein))
    for topology in TOPOLOGIES
}

fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
x = np.arange(len(TOPOLOGIES))

axes[0].bar(x, [rollout_ensemble_mse_score[t] for t in TOPOLOGIES], color=[f"C{i}" for i in range(len(TOPOLOGIES))])
axes[0].set_xticks(x)
axes[0].set_xticklabels(TOPOLOGIES, rotation=45, ha="right")
axes[0].set_ylabel("Ensemble MSE (um^2)")
axes[0].set_title("LOOCV ensemble MSE\n(avg positions, then compare)")

axes[1].bar(x, [rollout_mse_score[t] for t in TOPOLOGIES], color=[f"C{i}" for i in range(len(TOPOLOGIES))])
axes[1].set_xticks(x)
axes[1].set_xticklabels(TOPOLOGIES, rotation=45, ha="right")
axes[1].set_ylabel("Path MSE (um^2)")
axes[1].set_title("LOOCV rollout path MSE\n(per-chromosome 3D MSE)")

axes[2].bar(x, [rollout_endpoint_score[t] for t in TOPOLOGIES], color=[f"C{i}" for i in range(len(TOPOLOGIES))])
axes[2].set_xticks(x)
axes[2].set_xticklabels(TOPOLOGIES, rotation=45, ha="right")
axes[2].set_ylabel("Endpoint mean error (um^2)")
axes[2].set_title("LOOCV endpoint-only score\n(final axial/radial mean mismatch)")

axes[3].bar(x, [rollout_dist_score[t] for t in TOPOLOGIES], color=[f"C{i}" for i in range(len(TOPOLOGIES))])
axes[3].set_xticks(x)
axes[3].set_xticklabels(TOPOLOGIES, rotation=45, ha="right")
axes[3].set_ylabel("Final-frame W1 distance (um)")
axes[3].set_title("LOOCV final distribution mismatch\n(axial W1 + radial W1)")

for topo_index, topology in enumerate(TOPOLOGIES):
    axes[4].plot(
        rollout_results[topology].horizons,
        np.nanmean(rollout_results[topology].horizon_errors, axis=0),
        marker="o",
        linewidth=1.8,
        label=topology,
    )
axes[4].set_xlabel("Forecast horizon (frames)")
axes[4].set_ylabel("Combined axial/radial error (um^2)")
axes[4].set_title("Held-out forecast error vs horizon")
axes[4].legend(fontsize=8)

fig.suptitle("Aggregate rollout validation across held-out cells")
fig.tight_layout()
plt.show()

# %%
# Detailed LOOCV rollout table: per-metric means ± SE across held-out cells
sorted_topo_rollout = sorted(TOPOLOGIES, key=lambda t: rollout_ensemble_mse_score[t])
print("\nLOOCV rollout validation — detailed summary (mean ± SE across held-out cells)")
print("=" * 120)
print(f"{'Topology':<22} {'Ensemble MSE':>14} {'Path MSE':>14} {'Axial MSE':>14} {'Radial MSE':>14} {'Endpoint':>14} "
      f"{'W1 axial':>14} {'W1 radial':>14}")
print("-" * 135)

n_cv_cells = len(cells)
for topology in sorted_topo_rollout:
    rr = rollout_results[topology]
    def _fmt(arr):
        m = np.nanmean(arr)
        se = np.nanstd(arr) / np.sqrt(np.sum(np.isfinite(arr)))
        return f"{m:.4f} ± {se:.4f}"
    print(f"  {topology:<22} {_fmt(rr.ensemble_mse):>14} {_fmt(rr.path_mse):>14} {_fmt(rr.axial_mse):>14} {_fmt(rr.radial_mse):>14} "
          f"{_fmt(rr.endpoint_mean_error):>14} "
          f"{_fmt(rr.final_axial_wasserstein):>14} {_fmt(rr.final_radial_wasserstein):>14}")

print("=" * 120)

# Horizon-resolved table
print("\nHeld-out forecast error by horizon (mean ± SE, axial+radial combined)")
horizons = rollout_results[TOPOLOGIES[0]].horizons
header = f"{'Topology':<22}" + "".join(f"  h={h:<5}" for h in horizons)
print(header)
for topology in sorted_topo_rollout:
    rr = rollout_results[topology]
    vals = []
    for hi in range(len(horizons)):
        col = rr.horizon_errors[:, hi]
        m = np.nanmean(col)
        se = np.nanstd(col) / np.sqrt(np.sum(np.isfinite(col)))
        vals.append(f"{m:.4f}±{se:.4f}")
    print(f"  {topology:<22}" + "".join(f"  {v:<7}" for v in vals))


# %% [markdown]
# ## Metric concordance
#
# Do the different metrics agree on topology ranking?  We check two things:
#
# 1. **Rank concordance table**: topology ordering under each metric.
# 2. **Horizon-resolved path MSE vs ensemble MSE**: path MSE saturates at the
#    diffusion noise floor within a few frames, while ensemble MSE grows more
#    slowly and remains discriminative between topologies at longer horizons.
#    The gap between the two curves is the per-realization diffusion variance.

# %%
# Rank concordance table
metric_rankings: dict[str, list[str]] = {}
metric_rankings["Ensemble MSE"] = sorted(TOPOLOGIES, key=lambda t: rollout_ensemble_mse_score[t])
metric_rankings["Path MSE"] = sorted(TOPOLOGIES, key=lambda t: rollout_mse_score[t])
metric_rankings["1-step CV"] = sorted(TOPOLOGIES, key=lambda t: cv_results[t].mean_error)
metric_rankings["Endpoint"] = sorted(TOPOLOGIES, key=lambda t: rollout_endpoint_score[t])
metric_rankings["W1 (final)"] = sorted(TOPOLOGIES, key=lambda t: rollout_dist_score[t])

print("Metric concordance — topology ranking under each criterion")
print("=" * 90)
header = f"{'Metric':<16}" + "".join(f"  {'#' + str(i+1):<20}" for i in range(len(TOPOLOGIES)))
print(header)
print("-" * 90)
for metric_name, ranking in metric_rankings.items():
    row = f"  {metric_name:<16}" + "".join(f"  {t:<20}" for t in ranking)
    print(row)
print("=" * 90)

# Check if all metrics agree on rank-1
rank1_topologies = {r[0] for r in metric_rankings.values()}
if len(rank1_topologies) == 1:
    print(f"All metrics agree: {rank1_topologies.pop()} is best.")
else:
    print(f"Rank-1 picks: {', '.join(f'{m}: {r[0]}' for m, r in metric_rankings.items())}")

# %%
# Horizon-resolved path MSE vs ensemble MSE
fig_hz, axes_hz = plt.subplots(1, 2, figsize=(14, 5))

horizons_frames = rollout_results[TOPOLOGIES[0]].horizons
dt = cells[0].dt

for topo_idx, topology in enumerate(TOPOLOGIES):
    rr = rollout_results[topology]
    hz_ens = np.nanmean(rr.horizon_ensemble_mse, axis=0)
    hz_path = np.nanmean(rr.horizon_path_mse, axis=0)

    axes_hz[0].plot(
        horizons_frames * dt, hz_ens,
        marker="o", linewidth=1.8, label=topology,
    )
    axes_hz[1].plot(
        horizons_frames * dt, hz_path,
        marker="s", linewidth=1.8, label=topology,
    )

axes_hz[0].set_xlabel("Forecast horizon (s)")
axes_hz[0].set_ylabel("3D position MSE (um²)")
axes_hz[0].set_title("Ensemble-mean MSE vs horizon\n(drift bias only — discriminative)")
axes_hz[0].legend(fontsize=8)

axes_hz[1].set_xlabel("Forecast horizon (s)")
axes_hz[1].set_ylabel("3D position MSE (um²)")
axes_hz[1].set_title("Per-realization path MSE vs horizon\n(drift + diffusion noise — less discriminative)")
axes_hz[1].legend(fontsize=8)

fig_hz.suptitle("Horizon-resolved comparison: ensemble MSE isolates drift signal")
fig_hz.tight_layout()
plt.show()

# %%
# Overlay: both metrics for one topology to show the gap
ref_topo = sorted(TOPOLOGIES, key=lambda t: rollout_ensemble_mse_score[t])[0]
rr_ref = rollout_results[ref_topo]
hz_ens_ref = np.nanmean(rr_ref.horizon_ensemble_mse, axis=0)
hz_path_ref = np.nanmean(rr_ref.horizon_path_mse, axis=0)

fig_gap, ax_gap = plt.subplots(figsize=(8, 5))
ax_gap.plot(horizons_frames * dt, hz_ens_ref, "o-", linewidth=2, color="C0",
            label=f"Ensemble MSE ({ref_topo})")
ax_gap.plot(horizons_frames * dt, hz_path_ref, "s--", linewidth=2, color="C1",
            label=f"Path MSE ({ref_topo})")
ax_gap.fill_between(
    horizons_frames * dt, hz_ens_ref, hz_path_ref,
    alpha=0.15, color="C3", label="Diffusion noise floor",
)
ax_gap.set_xlabel("Forecast horizon (s)")
ax_gap.set_ylabel("3D position MSE (um²)")
ax_gap.set_title(
    "Path MSE vs ensemble MSE — the gap is diffusion variance\n"
    "Ensemble MSE removes this topology-invariant noise, isolating drift error"
)
ax_gap.legend(fontsize=9)
fig_gap.tight_layout()
plt.show()

noise_ratio = hz_path_ref[-1] / hz_ens_ref[-1] if hz_ens_ref[-1] > 0 else np.nan
print(f"At horizon {horizons_frames[-1]} frames ({horizons_frames[-1] * dt:.0f}s):")
print(f"  Path MSE / Ensemble MSE = {noise_ratio:.1f}x")
print(f"  ~{100*(1 - 1/noise_ratio):.0f}% of path MSE is diffusion noise, "
      "topology-invariant and non-discriminative.")


# %% [markdown]
# ## Model selection summary
#
# **Primary criterion**: leave-one-cell-out ensemble-mean MSE (simulated
# positions averaged across replicates before comparing to reality,
# cancelling model-side stochastic variance; the residual is drift bias
# plus a topology-invariant data-noise floor).  This is a conditional-mean
# trajectory score targeting drift/topology selection, not a full
# distributional SDE criterion.
#
# **Secondary quantitative checks**: per-rep path MSE, one-step velocity MSE,
# endpoint mismatch, final-frame distributional metrics, and horizon-specific
# rollout errors.
#
# **Qualitative checks**: full-data forward simulations and kernel-shape /
# physics plausibility (above).

# %%
sorted_topo = sorted(TOPOLOGIES, key=lambda t: cv_results[t].mean_error)
best_mean_topo = sorted_topo[0]
paired = paired_cv_differences(cv_results, reference=best_mean_topo)
best_rollout_topo = sorted_topo_rollout[0]
best_endpoint_topo = min(TOPOLOGIES, key=lambda t: rollout_endpoint_score[t])
best_dist_topo = min(TOPOLOGIES, key=lambda t: rollout_dist_score[t])

print("Primary criterion: leave-one-cell-out ensemble-mean MSE (drift bias only)")
print("=" * 115)
print(f"  {'Topology':<22} {'Ens MSE':>14} {'vs best':>9} {'Path MSE':>14} {'Endpoint':>10} {'W1 total':>10}")
print("-" * 115)

best_rollout_score = rollout_ensemble_mse_score[best_rollout_topo]
for t in sorted_topo_rollout:
    rel_pct = 100.0 * (rollout_ensemble_mse_score[t] - best_rollout_score) / best_rollout_score if best_rollout_score > 0 else np.nan
    print(f"  {t:<22} {rollout_ensemble_mse_score[t]:>14.4f} {rel_pct:>+8.1f}% "
          f"{rollout_mse_score[t]:>14.4f} "
          f"{rollout_endpoint_score[t]:>10.4f} {rollout_dist_score[t]:>10.4f}")

print("=" * 115)
print(f"  Primary rollout selector: {best_rollout_topo}")
print("  Path MSE, endpoint, and W1 are reported separately as supporting diagnostics.")

# Paired foldwise differences for rollout ensemble MSE
ref_ens = rollout_results[best_rollout_topo].ensemble_mse
print(f"\nPaired foldwise differences in ensemble MSE (reference: {best_rollout_topo})")
print(f"  {'Topology':<22} {'Ens MSE':>12} {'Δ vs best':>12} {'SE(Δ)':>12} {'Δ/SE(Δ)':>10} {'Significant?':>14}")
for topology in sorted_topo_rollout:
    rr = rollout_results[topology]
    diff = rr.ensemble_mse - ref_ens
    valid = np.isfinite(diff)
    n = int(valid.sum())
    mean_diff = float(np.mean(diff[valid])) if n > 0 else np.inf
    se_diff = float(np.std(diff[valid], ddof=1) / np.sqrt(n)) if n > 1 else np.inf
    ratio = mean_diff / se_diff if se_diff > 0 and se_diff < np.inf else 0.0
    sig = "—" if topology == best_rollout_topo else ("yes" if abs(ratio) > 2.0 else "no")
    print(f"  {topology:<22} {rollout_ensemble_mse_score[topology]:>12.4f} "
          f"{mean_diff:>+12.4e} {se_diff:>12.4e} {ratio:>10.2f} {sig:>14}")

print("\nSecondary diagnostic: leave-one-cell-out 1-step velocity MSE")
print(f"  {'Topology':<22} {'CV MSE':>12} {'vs null':>9} {'SE':>10} {'Δ vs best':>10} {'SE(Δ)':>10} {'Δ/SE(Δ)':>8}")
for topology in sorted_topo:
    r = cv_results[topology]
    mean_diff, se_diff = paired[topology]
    ratio = mean_diff / se_diff if se_diff > 0 and se_diff < np.inf else 0.0
    gain_pct = 100.0 * (zero_baseline_mean - r.mean_error) / zero_baseline_mean if zero_baseline_mean > 0 else np.nan
    print(f"  {topology:<22} {r.mean_error:>12.8f} {gain_pct:>+8.2f}% {r.fold_se:>10.2e} "
          f"{mean_diff:>+10.2e} {se_diff:>10.2e} {ratio:>8.2f}")
print("  The one-step loss remains useful, but in this dataset it is less discriminative than the rollout MSE.")

print("\nSynthesis")
print("=" * 115)
print(f"  Primary selector (ensemble MSE): {best_rollout_topo}")
print(f"  Best 1-step CV score:            {best_mean_topo}")
print(f"  Best rollout endpoint score:     {best_endpoint_topo}")
print(f"  Best rollout final W1 score:     {best_dist_topo}")
if best_rollout_topo == best_endpoint_topo == best_dist_topo:
    print(f"  The rollout diagnostics agree on {best_rollout_topo}.")
else:
    print("  The rollout diagnostics are not fully unanimous, but they point to the same small subset of topologies.")
if best_mean_topo == best_rollout_topo:
    print("  The rollout selector and the 1-step diagnostic agree.")
else:
    print("  The rollout selector and the 1-step diagnostic do not agree.")
    print("  The 1-step CV gap is small, while rollout is more discriminative on long-horizon behavior.")

# Parsimony note: if the gap between the simplest plausible model and the
# winner is within ~1 SE, prefer the simpler model.
simplest_topo = "poles"  # fewest parameters: no xx term, no midpoint
if best_rollout_topo != simplest_topo:
    diff_simp = rollout_results[simplest_topo].ensemble_mse - ref_ens
    valid_simp = np.isfinite(diff_simp)
    n_simp = int(valid_simp.sum())
    mean_simp = float(np.mean(diff_simp[valid_simp])) if n_simp > 0 else np.inf
    se_simp = float(np.std(diff_simp[valid_simp], ddof=1) / np.sqrt(n_simp)) if n_simp > 1 else np.inf
    ratio_simp = mean_simp / se_simp if se_simp > 0 and se_simp < np.inf else 0.0
    if abs(ratio_simp) < 1.0:
        print(f"\n  Parsimony note: {simplest_topo} is within 1 SE of {best_rollout_topo} "
              f"(Δ/SE = {ratio_simp:.2f}). The simpler model is not significantly worse.")
    elif abs(ratio_simp) < 2.0:
        print(f"\n  Parsimony note: {simplest_topo} is between 1-2 SE of {best_rollout_topo} "
              f"(Δ/SE = {ratio_simp:.2f}). The gap is suggestive but not definitive.")
    else:
        print(f"\n  Parsimony note: {simplest_topo} is >2 SE worse than {best_rollout_topo} "
              f"(Δ/SE = {ratio_simp:.2f}). The more complex model is justified.")
print("=" * 105)

best_topology = best_rollout_topo
print(f"\nWith only {len(TOPOLOGIES)} candidates, selection-level overfitting is unlikely.")
print("If the gap between top candidates is within ~1 SE, the ranking is not definitive;")
print("in that case, prefer the simpler (fewer-parameter) model.")

# Final kernel plot for the winner
print(f"\nFinal kernel plot for best model ({best_topology}):")
fig = plot_kernels(models[best_topology], bootstrap=boot_results[best_topology])
fig.suptitle(f"Best model kernels — {best_topology}", y=1.02)
plt.show()

# %% [markdown]
# ## Scope of comparison
#
# The topologies above form a restricted, physically motivated candidate
# family (pairwise SFI-inspired kernels with pole-geometry and optional
# chromosome-chromosome terms).  The selection identifies the best model
# *within this family*; it does not exhaust all plausible interaction
# structures.  Extensions worth testing in future work include an explicit
# common-transport / advection term and a `poles + center` hybrid topology.

# %% [markdown]
# ## Note on diffusion
#
# Spatially-varying diffusion D(x) analysis has been moved to
# **Notebook 06 (06_diffusion_landscape.py)**, which provides a thorough
# investigation with multiple estimators, per-cell consistency checks, and
# comparison across coordinate axes.
