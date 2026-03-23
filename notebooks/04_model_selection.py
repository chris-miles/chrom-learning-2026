# %% [markdown]
# # 04 -- Model selection: which interaction topology fits best?
#
# We compare four chromosome interaction topologies:
#
# | Label | xy partner(s) | xx term? |
# |---|---|---|
# | **poles** | both poles (2 partners) | no |
# | **center** | pole midpoint (1 partner) | no |
# | **poles\_and\_chroms** | both poles | yes |
# | **center\_and\_chroms** | pole midpoint | yes |
#
# Selection criteria: leave-one-cell-out CV, rollout validation, bootstrap CIs.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chromlearn.io.catalog import load_condition
from chromlearn.io.trajectory import (
    TrimmedCell,
    get_partners,
    spindle_frame,
    trim_trajectory,
)
from chromlearn.model_fitting import FitConfig
from chromlearn.model_fitting.basis import BSplineBasis
from chromlearn.model_fitting.fit import (
    BootstrapResult,
    CVResult,
    RolloutCVResult,
    bootstrap_kernels,
    cross_validate,
    fit_model,
    rollout_cross_validate,
)
from chromlearn.model_fitting.model import FittedModel
from chromlearn.model_fitting.plotting import plot_cv_curve, plot_kernels
from chromlearn.model_fitting.simulate import simulate_cell, simulate_trajectories

plt.rcParams["figure.dpi"] = 110

# %%
cells_raw = load_condition("rpe18_ctr")
cells = [trim_trajectory(c, method="neb_ao_frac") for c in cells_raw]
print(f"Loaded {len(cells)} rpe18_ctr cells (trimmed to neb_ao_frac=0.5 window)")
for c in cells:
    T, _, N = c.chromosomes.shape
    print(f"  {c.cell_id}: {T} frames, {N} chromosomes")

# %% [markdown]
# ## Estimate basis domains from data
#
# Sweep all cells to get empirical distance distributions for xy (depends on
# topology) and xx (topology-independent).  Lower limit anchored at 0.3 um
# (below tracking resolution); upper limit from observed max distance.

# %%
TOPOLOGIES = ["poles", "center", "poles_and_chroms", "center_and_chroms"]

# --------------------------------------------------------------------------
# Collect chromosome-to-partner distances per topology
# --------------------------------------------------------------------------
xy_dists_by_topology: dict[str, list[float]] = {t: [] for t in TOPOLOGIES}
xx_dists_all: list[float] = []

for cell in cells:
    T, _, N = cell.chromosomes.shape
    chroms = cell.chromosomes  # (T, 3, N)

    # xx distances: pairwise chromosome-chromosome, all frames
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

    # xy distances per topology
    for topology in TOPOLOGIES:
        partners = get_partners(cell, topology)  # (n_p, T, 3)
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
print(f"Collected {len(xx_dists_all):,} chromosome-chromosome distance samples")
for t in TOPOLOGIES:
    print(f"  xy ({t}): {len(xy_dists_by_topology[t]):,} samples")


# %%
def _domain_from_dists(
    dists: np.ndarray,
    r_min_floor: float = 0.3,
    r_max_ceil: float = 20.0,
) -> tuple[float, float]:
    """Return (r_min, r_max) domain from distance distribution."""
    r_min = float(r_min_floor)
    r_max = float(np.clip(np.max(dists), r_min_floor, r_max_ceil))
    if r_max <= r_min:
        r_max = r_min + 1.0
    return r_min, r_max


# Shared xx domain (topology-independent)
r_min_xx, r_max_xx = _domain_from_dists(xx_dists_all)

# Per-topology xy domains
xy_domains: dict[str, tuple[float, float]] = {}
for topology in TOPOLOGIES:
    xy_domains[topology] = _domain_from_dists(np.array(xy_dists_by_topology[topology]))

print(f"\nxx domain (all topologies): r_min={r_min_xx:.2f}, r_max={r_max_xx:.2f} um")
for t in TOPOLOGIES:
    rlo, rhi = xy_domains[t]
    print(f"  xy ({t}): r_min={rlo:.2f}, r_max={rhi:.2f} um")

# %%
# Plot distance distributions with chosen domains
fig, axes = plt.subplots(1, 5, figsize=(18, 4))

axes[0].hist(xx_dists_all, bins=60, color="C2", edgecolor="k", alpha=0.7)
axes[0].axvline(r_min_xx, color="r", linestyle="--", linewidth=1.5, label=f"r_min={r_min_xx:.2f}")
axes[0].axvline(r_max_xx, color="r", linestyle="-", linewidth=1.5, label=f"r_max={r_max_xx:.2f}")
axes[0].set_title("Chromosome-chromosome (xx)")
axes[0].set_xlabel("Distance (um)")
axes[0].set_ylabel("Count")
axes[0].legend(fontsize=7)

for idx, topology in enumerate(TOPOLOGIES):
    ax = axes[idx + 1]
    rlo, rhi = xy_domains[topology]
    ax.hist(xy_dists_by_topology[topology], bins=60, color=f"C{idx}", edgecolor="k", alpha=0.7)
    ax.axvline(rlo, color="r", linestyle="--", linewidth=1.5, label=f"r_min={rlo:.2f}")
    ax.axvline(rhi, color="r", linestyle="-", linewidth=1.5, label=f"r_max={rhi:.2f}")
    ax.set_title(f"xy ({topology})")
    ax.set_xlabel("Distance (um)")
    ax.legend(fontsize=7)

fig.suptitle("Observed distance distributions — basis domains marked in red")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Fit all four models
#
# We use:
# - `n_basis = 10` B-spline basis functions for both xx and xy kernels
# - `lambda_ridge = lambda_rough = 1e-3`
# - `basis_eval_mode = "ito"` (current positions, standard SFI)
#
# Domain parameters are set from the empirical distributions above.

# %%
N_BASIS = 10
LAMBDA = 1e-3

configs: dict[str, FitConfig] = {}
for topology in TOPOLOGIES:
    rlo_xy, rhi_xy = xy_domains[topology]
    configs[topology] = FitConfig(
        topology=topology,
        n_basis_xx=N_BASIS,
        n_basis_xy=N_BASIS,
        r_min_xx=r_min_xx,
        r_max_xx=r_max_xx,
        r_min_xy=rlo_xy,
        r_max_xy=rhi_xy,
        basis_type="bspline",
        lambda_ridge=LAMBDA,
        lambda_rough=LAMBDA,
        basis_eval_mode="ito",
        dt=5.0,
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
# Leave-one-cell-out CV: fit on 12 cells, evaluate mean squared error on the
# held-out cell.  Lower is better.

# %%
print("Running leave-one-cell-out cross-validation (4 topologies × 13 folds)...")
cv_results: dict[str, CVResult] = {}
for topology in TOPOLOGIES:
    cv_results[topology] = cross_validate(cells, configs[topology])
    r = cv_results[topology]
    print(f"  {topology:<22}  MSE={r.mean_error:.8f} ± {r.std_error:.8f}")

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
sorted_topo = sorted(TOPOLOGIES, key=lambda t: cv_results[t].mean_error)
best_cv = cv_results[sorted_topo[0]].mean_error
for rank, topology in enumerate(sorted_topo):
    r = cv_results[topology]
    delta = r.mean_error - best_cv
    print(f"  #{rank + 1}  {topology:<22}  MSE={r.mean_error:.8f}  "
          f"(Δbest={delta:+.2e})")


# %% [markdown]
# ## Bootstrap kernel confidence bands
#
# 250 cell-level resamples per topology; shaded band = 5–95% quantile interval.

# %%
N_BOOT = 50
boot_rng = np.random.default_rng(42)

print(f"Bootstrapping kernels ({N_BOOT} resamples × 4 topologies)...")
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
chroms_topologies = [t for t in TOPOLOGIES if t in ("poles_and_chroms", "center_and_chroms")]
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
r_xy_probe = {t: np.linspace(*xy_domains[t], 400) for t in TOPOLOGIES}

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

# %%
EXAMPLE_CELL_IDX = 1
example_cell = cells[EXAMPLE_CELL_IDX]
T, _, N_chrom = example_cell.chromosomes.shape
n_steps = T - 1
x0 = example_cell.chromosomes[0].T
sf_real = spindle_frame(example_cell)

QUAL_CELL_IDXS = sorted({0, len(cells) // 2, len(cells) - 1})
QUAL_N_TRACES = 6
ROLLOUT_REPS = 4
ROLLOUT_HORIZONS = (1, 5, 10, 20)


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

        real_axial_subset_mean = np.nanmean(sf_cell_real.axial[:, trace_idx], axis=1)
        sim_axial_subset_mean = np.nanmean(sf_sim.axial[:, trace_idx], axis=1)
        real_radial_subset_mean = np.nanmean(sf_cell_real.radial[:, trace_idx], axis=1)
        sim_radial_subset_mean = np.nanmean(sf_sim.radial[:, trace_idx], axis=1)

        ax_axial.plot(time_axis, real_axial_subset_mean, "k-", linewidth=2.0, label="Displayed real mean")
        ax_axial.plot(time_axis, sim_axial_subset_mean, color=color, linestyle="--", linewidth=2.0,
                      label="Displayed rollout mean")
        ax_radial.plot(time_axis, real_radial_subset_mean, "k-", linewidth=2.0, label="Displayed real mean")
        ax_radial.plot(time_axis, sim_radial_subset_mean, color=color, linestyle="--", linewidth=2.0,
                       label="Displayed rollout mean")

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
                 "Thin lines = representative chromosome traces, thick lines = means of displayed traces")
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
      f"({len(TOPOLOGIES)} topologies × {len(cells)} folds × {ROLLOUT_REPS} rollouts)...")
rollout_results: dict[str, RolloutCVResult] = {}
for topo_index, topology in enumerate(TOPOLOGIES):
    rollout_results[topology] = rollout_cross_validate(
        cells,
        configs[topology],
        n_reps=ROLLOUT_REPS,
        horizons=ROLLOUT_HORIZONS,
        rng=np.random.default_rng(200 + topo_index),
    )
    rr = rollout_results[topology]
    print(f"  {topology:<22}  axial_MSE={np.nanmean(rr.axial_mse):.5f}  "
          f"radial_MSE={np.nanmean(rr.radial_mse):.5f}  "
          f"endpoint_MSE={np.nanmean(rr.endpoint_mean_error):.5f}  "
          f"final_W1(ax,rad)=({np.nanmean(rr.final_axial_wasserstein):.4f}, "
          f"{np.nanmean(rr.final_radial_wasserstein):.4f})")


# %%
rollout_time_score = {
    topology: float(np.nanmean(rollout_results[topology].axial_mse)
                    + np.nanmean(rollout_results[topology].radial_mse))
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

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
x = np.arange(len(TOPOLOGIES))

axes[0].bar(x, [rollout_time_score[t] for t in TOPOLOGIES], color=[f"C{i}" for i in range(len(TOPOLOGIES))])
axes[0].set_xticks(x)
axes[0].set_xticklabels(TOPOLOGIES, rotation=45, ha="right")
axes[0].set_ylabel("Mean rollout trajectory error (um^2)")
axes[0].set_title("LOOCV rollout score\n(axial MSE + radial MSE)")

axes[1].bar(x, [rollout_endpoint_score[t] for t in TOPOLOGIES], color=[f"C{i}" for i in range(len(TOPOLOGIES))])
axes[1].set_xticks(x)
axes[1].set_xticklabels(TOPOLOGIES, rotation=45, ha="right")
axes[1].set_ylabel("Endpoint mean error (um^2)")
axes[1].set_title("LOOCV endpoint-only score\n(final axial/radial mean mismatch)")

axes[2].bar(x, [rollout_dist_score[t] for t in TOPOLOGIES], color=[f"C{i}" for i in range(len(TOPOLOGIES))])
axes[2].set_xticks(x)
axes[2].set_xticklabels(TOPOLOGIES, rotation=45, ha="right")
axes[2].set_ylabel("Final-frame W1 distance (um)")
axes[2].set_title("LOOCV final distribution mismatch\n(axial W1 + radial W1)")

for topo_index, topology in enumerate(TOPOLOGIES):
    axes[3].plot(
        rollout_results[topology].horizons,
        np.nanmean(rollout_results[topology].horizon_errors, axis=0),
        marker="o",
        linewidth=1.8,
        label=topology,
    )
axes[3].set_xlabel("Forecast horizon (frames)")
axes[3].set_ylabel("Combined axial/radial error (um^2)")
axes[3].set_title("Held-out forecast error vs horizon")
axes[3].legend(fontsize=8)

fig.suptitle("Aggregate rollout validation across held-out cells")
fig.tight_layout()
plt.show()

sorted_topo_endpoint = sorted(TOPOLOGIES, key=lambda t: rollout_endpoint_score[t])

# %%
print("=" * 112)
print(f"{'Topology':<22} {'CV MSE':>12} {'Δbest':>10} {'Rollout':>10} {'Endpoint':>10} {'Final W1':>10} {'Rank':>5} {'D_x':>12} {'n_params':>9}")
print("-" * 112)

sorted_topo = sorted(TOPOLOGIES, key=lambda t: cv_results[t].mean_error)
best_cv = cv_results[sorted_topo[0]].mean_error
sorted_topo_rollout = sorted(TOPOLOGIES, key=lambda t: rollout_time_score[t])
for rank, topology in enumerate(sorted_topo):
    r = cv_results[topology]
    m = models[topology]
    delta = r.mean_error - best_cv
    print(f"  {topology:<22} {r.mean_error:>12.8f} {delta:>+10.2e} "
          f"{rollout_time_score[topology]:>10.4f} {rollout_endpoint_score[topology]:>10.4f} "
          f"{rollout_dist_score[topology]:>10.4f} {rank + 1:>5} {m.D_x:>12.8f} {m.theta.size:>9}")

print("=" * 112)

best_topology = sorted_topo[0]
best_topology_rollout = sorted_topo_rollout[0]
best_topology_endpoint = sorted_topo_endpoint[0]
print(f"\n1-step CV winner: {best_topology}")
print(f"  CV MSE          = {cv_results[best_topology].mean_error:.8f}")
print(f"  Rollout score   = {rollout_time_score[best_topology]:.5f}")
print(f"  Endpoint score  = {rollout_endpoint_score[best_topology]:.5f}")
print(f"  Final W1 score  = {rollout_dist_score[best_topology]:.5f}")
print(f"  D_x             = {models[best_topology].D_x:.8f} um^2/s")

print(f"\nRollout winner: {best_topology_rollout}")
print(f"  Rollout score   = {rollout_time_score[best_topology_rollout]:.5f}")
print(f"  Endpoint score  = {rollout_endpoint_score[best_topology_rollout]:.5f}")
print(f"  Final W1 score  = {rollout_dist_score[best_topology_rollout]:.5f}")
print(f"  CV MSE          = {cv_results[best_topology_rollout].mean_error:.8f}")

print(f"\nEndpoint winner: {best_topology_endpoint}")
print(f"  Endpoint score  = {rollout_endpoint_score[best_topology_endpoint]:.5f}")
print(f"  Rollout score   = {rollout_time_score[best_topology_endpoint]:.5f}")
print(f"  Final W1 score  = {rollout_dist_score[best_topology_endpoint]:.5f}")
print(f"  CV MSE          = {cv_results[best_topology_endpoint].mean_error:.8f}")

# Final kernel plot for the winner
print(f"\nFinal kernel plot for best model ({best_topology}):")
fig = plot_kernels(models[best_topology], bootstrap=boot_results[best_topology])
fig.suptitle(f"Best model kernels — {best_topology}", y=1.02)
plt.show()

# %% [markdown]
# ## Note on diffusion
#
# Spatially-varying diffusion D(x) analysis has been moved to
# **Notebook 06 (06_diffusion_landscape.py)**, which provides a thorough
# investigation with multiple estimators, per-cell consistency checks, and
# comparison across coordinate axes.
