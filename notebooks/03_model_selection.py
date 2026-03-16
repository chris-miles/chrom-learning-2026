# %% [markdown]
# # 03 — Model selection: which interaction topology fits best?
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
# The analysis follows these steps:
# 1. Load and trim `rpe18_ctr` data.
# 2. Estimate basis domains from observed distance distributions.
# 3. Fit all four models with the same hyperparameters.
# 4. Compare via leave-one-cell-out cross-validation.
# 5. Inspect bootstrap confidence bands around each learned kernel.
# 6. Check physical plausibility of the chromosome-chromosome kernel.
# 7. Forward simulation: real vs simulated trajectories in spindle frame.
# 8. Verdict table.
# 9. **Bonus**: refit the winner with variable D(x) and compare.

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
    bootstrap_kernels,
    cross_validate,
    fit_model,
)
from chromlearn.model_fitting.model import FittedModel
from chromlearn.model_fitting.plotting import plot_cv_curve, plot_kernels
from chromlearn.model_fitting.simulate import simulate_trajectories

plt.rcParams["figure.dpi"] = 110

# %%
cells_raw = load_condition("rpe18_ctr")
cells = [trim_trajectory(c, method="midpoint_neb_ao") for c in cells_raw]
print(f"Loaded {len(cells)} rpe18_ctr cells (trimmed to midpoint_neb_ao window)")
for c in cells:
    T, _, N = c.chromosomes.shape
    print(f"  {c.cell_id}: {T} frames, {N} chromosomes")

# %% [markdown]
# ## Estimate basis domains from data
#
# We sweep through all cells and collect the empirical distributions of:
# - Chromosome-to-partner distances (xy) — depends on topology
# - Chromosome-to-chromosome distances (xx) — topology-independent
#
# The basis domain is set to the [2nd, 98th] percentile of observed distances,
# clipped so r_min >= 0.3 um (below our tracking resolution) and
# r_max <= 20 um.

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
def _domain_from_dists(dists: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0,
                       r_min_floor: float = 0.3, r_max_ceil: float = 20.0) -> tuple[float, float]:
    """Return (r_min, r_max) domain from distance distribution."""
    r_min = float(np.clip(np.percentile(dists, lo_pct), r_min_floor, r_max_ceil))
    r_max = float(np.clip(np.percentile(dists, hi_pct), r_min_floor, r_max_ceil))
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
    print(f"  {topology:<22}  D_x={m.D_x:.4f} um^2/s  "
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
    print(f"  {topology:<22}  MSE={r.mean_error:.5f} ± {r.std_error:.5f}")

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
for rank, topology in enumerate(sorted_topo):
    r = cv_results[topology]
    print(f"  #{rank + 1}  {topology:<22}  MSE={r.mean_error:.5f}")

# %% [markdown]
# ## Bootstrap kernel confidence bands
#
# For each topology we draw 250 bootstrap resamples of the cells and refit
# the model.  The shaded band shows the 5–95% bootstrap quantile interval.

# %%
N_BOOT = 250
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
# - **Expected**: a repulsive barrier at short distances (excluded volume,
#   r ≲ 1 um) and weak or no force at larger distances.
# - **Red flag**: short-range *attraction* (negative force at small r) is an
#   artifact of regularization or inadequate data coverage at those distances.
#   It would cause simulated chromosomes to collapse onto each other.
#
# We flag any model where the minimum of f_xx at r < 1.5 um is negative.

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
    min_short_r = float(np.min(f_vals[short_r_mask])) if short_r_mask.any() else np.nan
    if min_short_r < 0:
        print(f"  [WARNING] {topology}: f_xx is ATTRACTIVE at short range "
              f"(min={min_short_r:.4f} at r < {SHORT_R_THRESHOLD} um). "
              "Likely an artifact — excluded-volume physics expects repulsion here.")
    else:
        print(f"  [OK] {topology}: f_xx is repulsive at short range "
              f"(min={min_short_r:.4f} at r < {SHORT_R_THRESHOLD} um).")

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
# ## Forward simulation: real vs simulated trajectories
#
# We pick the representative cell (index 1) and simulate each model forward
# from its real initial conditions.  The simulated chromosomes are projected
# into the real spindle frame (using the cell's actual centrosome positions)
# and compared to the observed distribution.
#
# We run 10 independent noise realisations per model and show the resulting
# axial/radial distributions as kernel-density-estimate ribbons.

# %%
EXAMPLE_CELL_IDX = 1
sim_rng = np.random.default_rng(7)
N_SIM_REPS = 10

example_cell = cells[EXAMPLE_CELL_IDX]
T, _, N_chrom = example_cell.chromosomes.shape
n_steps = T - 1
x0 = example_cell.chromosomes[0].T  # (N, 3) — initial positions

# Real spindle-frame coordinates
sf_real = spindle_frame(example_cell)

print(f"Simulating {len(TOPOLOGIES)} models × {N_SIM_REPS} realisations on "
      f"{example_cell.cell_id} ({T} frames, {N_chrom} chromosomes)...")

sim_sf: dict[str, list] = {t: [] for t in TOPOLOGIES}

for topology in TOPOLOGIES:
    m = models[topology]

    # Build kernel callables
    def make_kernel_xy(model):
        def _f(r):
            return model.evaluate_kernel("xy", r)
        return _f

    def make_kernel_xx(model):
        if model.basis_xx is None:
            return None
        def _f(r):
            return model.evaluate_kernel("xx", r)
        return _f

    kernel_xy = make_kernel_xy(m)
    kernel_xx = make_kernel_xx(m)

    # Partner positions for simulation
    partners = get_partners(example_cell, topology)  # (n_p, T, 3)

    for rep in range(N_SIM_REPS):
        traj = simulate_trajectories(
            kernel_xx=kernel_xx,
            kernel_xy=kernel_xy,
            partner_positions=partners,
            x0=x0,
            n_steps=n_steps,
            dt=example_cell.dt,
            D_x=m.D_x,
            rng=sim_rng,
        )  # (T, 3, N)

        # Build a temporary TrimmedCell to use spindle_frame()
        sim_cell = TrimmedCell(
            cell_id=example_cell.cell_id,
            condition=example_cell.condition,
            centrioles=example_cell.centrioles,
            chromosomes=traj,
            tracked=N_chrom,
            dt=example_cell.dt,
            start_frame=example_cell.start_frame,
            end_frame=example_cell.end_frame,
        )
        sf_sim = spindle_frame(sim_cell)
        sim_sf[topology].append(sf_sim)

    print(f"  {topology} done.")

# %%
# Plot: mean axial and radial position over time — real vs simulated
fig, axes = plt.subplots(2, len(TOPOLOGIES), figsize=(5 * len(TOPOLOGIES), 8), squeeze=False)
time_axis = np.arange(T) * example_cell.dt

for col, topology in enumerate(TOPOLOGIES):
    ax_axial = axes[0, col]
    ax_radial = axes[1, col]

    # Real data
    real_axial_mean = np.nanmean(sf_real.axial, axis=1)  # (T,)
    real_axial_std = np.nanstd(sf_real.axial, axis=1)
    real_radial_mean = np.nanmean(sf_real.radial, axis=1)
    real_radial_std = np.nanstd(sf_real.radial, axis=1)

    ax_axial.plot(time_axis, real_axial_mean, "k-", linewidth=2, label="Real")
    ax_axial.fill_between(time_axis,
                          real_axial_mean - real_axial_std,
                          real_axial_mean + real_axial_std,
                          color="k", alpha=0.12)

    ax_radial.plot(time_axis, real_radial_mean, "k-", linewidth=2, label="Real")
    ax_radial.fill_between(time_axis,
                           real_radial_mean - real_radial_std,
                           real_radial_mean + real_radial_std,
                           color="k", alpha=0.12)

    # Simulated realisations
    sim_axial_stack = np.stack(
        [np.nanmean(s.axial, axis=1) for s in sim_sf[topology]], axis=0
    )  # (N_SIM_REPS, T)
    sim_radial_stack = np.stack(
        [np.nanmean(s.radial, axis=1) for s in sim_sf[topology]], axis=0
    )

    sim_ax_med = np.median(sim_axial_stack, axis=0)
    sim_ax_lo = np.percentile(sim_axial_stack, 10, axis=0)
    sim_ax_hi = np.percentile(sim_axial_stack, 90, axis=0)

    sim_rad_med = np.median(sim_radial_stack, axis=0)
    sim_rad_lo = np.percentile(sim_radial_stack, 10, axis=0)
    sim_rad_hi = np.percentile(sim_radial_stack, 90, axis=0)

    color = f"C{col}"
    ax_axial.plot(time_axis, sim_ax_med, color=color, linewidth=1.8, linestyle="--",
                  label="Simulated")
    ax_axial.fill_between(time_axis, sim_ax_lo, sim_ax_hi, color=color, alpha=0.2)

    ax_radial.plot(time_axis, sim_rad_med, color=color, linewidth=1.8, linestyle="--",
                   label="Simulated")
    ax_radial.fill_between(time_axis, sim_rad_lo, sim_rad_hi, color=color, alpha=0.2)

    ax_axial.set_title(f"Axial — {topology}", fontsize=9)
    ax_radial.set_title(f"Radial — {topology}", fontsize=9)
    ax_axial.set_xlabel("Time (s)")
    ax_radial.set_xlabel("Time (s)")
    if col == 0:
        ax_axial.set_ylabel("Mean axial position (um)")
        ax_radial.set_ylabel("Mean radial distance (um)")
    ax_axial.legend(fontsize=7)
    ax_radial.legend(fontsize=7)

fig.suptitle(f"Forward simulation — {example_cell.cell_id}\n"
             "Mean chromosome position in spindle frame (real=black, simulated=color)")
fig.tight_layout()
plt.show()

# %%
# Scatter: axial vs radial distribution at final timepoint — real vs simulated
fig, axes = plt.subplots(1, len(TOPOLOGIES) + 1, figsize=(4 * (len(TOPOLOGIES) + 1), 4.5),
                          squeeze=False)

# Real data at final frame
real_axial_t = sf_real.axial[-1]   # (N,)
real_radial_t = sf_real.radial[-1]

ax0 = axes[0, 0]
valid = np.isfinite(real_axial_t) & np.isfinite(real_radial_t)
ax0.scatter(real_axial_t[valid], real_radial_t[valid], s=20, alpha=0.8, color="k")
ax0.set_title("Real (final frame)", fontsize=9)
ax0.set_xlabel("Axial (um)")
ax0.set_ylabel("Radial (um)")

for col, topology in enumerate(TOPOLOGIES):
    ax = axes[0, col + 1]
    # Pool all realisations at final frame
    for rep_idx, sf_sim_rep in enumerate(sim_sf[topology]):
        ax_vals = sf_sim_rep.axial[-1]
        rad_vals = sf_sim_rep.radial[-1]
        valid = np.isfinite(ax_vals) & np.isfinite(rad_vals)
        ax.scatter(ax_vals[valid], rad_vals[valid], s=10, alpha=0.4, color=f"C{col}")
    ax.set_title(f"{topology}\n(final frame, {N_SIM_REPS} reps)", fontsize=9)
    ax.set_xlabel("Axial (um)")
    ax.set_ylabel("Radial (um)")

fig.suptitle(f"Chromosome spatial distribution at final frame — {example_cell.cell_id}")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Verdict: model selection summary
#
# We now compile the CV scores, plausibility flags, and simulation quality
# into a summary table and pick the best model.

# %%
print("=" * 70)
print(f"{'Topology':<24} {'CV MSE':>10} {'Rank':>5} {'D_x (um^2/s)':>14} {'n_params':>9}")
print("-" * 70)

sorted_topo = sorted(TOPOLOGIES, key=lambda t: cv_results[t].mean_error)
for rank, topology in enumerate(sorted_topo):
    r = cv_results[topology]
    m = models[topology]
    print(f"  {topology:<22} {r.mean_error:>10.5f} {rank + 1:>5} {m.D_x:>14.4f} {m.theta.size:>9}")

print("=" * 70)

best_topology = sorted_topo[0]
print(f"\nSelected topology: {best_topology}")
print(f"  CV MSE = {cv_results[best_topology].mean_error:.5f}")
print(f"  D_x    = {models[best_topology].D_x:.4f} um^2/s")
print()
print("Notes:")
print("  - 'poles' and 'center' differ in whether both poles or only their")
print("    midpoint enter the xy kernel.  'center' has half the partner count")
print("    and a sparser design matrix.")
print("  - 'poles_and_chroms' / 'center_and_chroms' add a chromosome-chromosome")
print("    term f_xx.  This roughly doubles parameter count.  CV penalises")
print("    overfitting, so improvement must be robust to be preferred.")
print("  - If CV scores are similar (< 5% difference), the simpler model")
print("    (poles or center) is preferred on parsimony grounds.")

# Final kernel plot for the winner
print(f"\nFinal kernel plot for best model ({best_topology}):")
fig = plot_kernels(models[best_topology], bootstrap=boot_results[best_topology])
fig.suptitle(f"Best model kernels — {best_topology}", y=1.02)
plt.show()

# %% [markdown]
# ## Bonus: variable D(x) on the winning model
#
# The scalar D assumption may be overly restrictive.  Chromosomes near the
# metaphase plate may have different diffusivity than those near the poles.
# We estimate D as a function of axial position using the Vestergaard estimator
# and compare the spatially-resolved D(x) to the scalar estimate.

# %%
from chromlearn.model_fitting.diffusion import BSplineBasis as _BSplineBasisDiff
from chromlearn.model_fitting.diffusion import DiffusionResult, estimate_diffusion_variable
from chromlearn.model_fitting.plotting import plot_diffusion

# Use the same basis domain as the FitConfig r_min_D / r_max_D defaults
N_BASIS_D = 8
R_MIN_D = -8.0
R_MAX_D = 8.0

basis_D = BSplineBasis(R_MIN_D, R_MAX_D, N_BASIS_D)

print(f"Estimating variable D(axial) for topology '{best_topology}'...")
diff_result = estimate_diffusion_variable(
    cells,
    basis_D=basis_D,
    coord_name="axial",
    dt=5.0,
    mode="vestergaard",
    lambda_ridge=LAMBDA,
)
print(f"  Scalar D (reference): {diff_result.D_scalar:.4f} um^2/s")

# Evaluate on a grid
coords_grid = np.linspace(R_MIN_D, R_MAX_D, 300)
D_variable_vals = diff_result.evaluate(coords_grid)
print(f"  D(x) range: [{D_variable_vals.min():.4f}, {D_variable_vals.max():.4f}] um^2/s")

# %%
fig = plot_diffusion(diff_result)
fig.suptitle(f"Variable diffusion D(axial) — {best_topology} winner", y=1.02)
plt.show()

# %%
# Compare scalar vs variable D: refit best model with variable D and simulate
winner_config_scalar = configs[best_topology]
winner_config_varD = FitConfig(
    topology=best_topology,
    n_basis_xx=N_BASIS,
    n_basis_xy=N_BASIS,
    r_min_xx=winner_config_scalar.r_min_xx,
    r_max_xx=winner_config_scalar.r_max_xx,
    r_min_xy=winner_config_scalar.r_min_xy,
    r_max_xy=winner_config_scalar.r_max_xy,
    basis_type="bspline",
    lambda_ridge=LAMBDA,
    lambda_rough=LAMBDA,
    basis_eval_mode="ito",
    dt=5.0,
    D_variable=True,
    n_basis_D=N_BASIS_D,
    r_min_D=R_MIN_D,
    r_max_D=R_MAX_D,
    D_coordinate="axial",
)

# Scalar model is already fit; simulate with variable D using the same theta
# but sampling D from D(x) at each chromosome's current axial position.
# We implement this directly here since simulate_trajectories uses a scalar D.

winner_model = models[best_topology]
winner_partners = get_partners(example_cell, best_topology)

def _kernel_xy_winner(r):
    return winner_model.evaluate_kernel("xy", r)

_kernel_xx_winner = None
if winner_model.basis_xx is not None:
    def _kernel_xx_winner(r):
        return winner_model.evaluate_kernel("xx", r)


def simulate_variable_D(
    kernel_xx,
    kernel_xy,
    partner_positions,
    x0,
    n_steps,
    dt,
    diff_result_obj,
    cell_ref,
    rng,
):
    """Forward simulation with spatially-varying D(axial position).

    At each step, the noise amplitude for chromosome i is determined by
    evaluating D(axial_i) where axial_i is the signed projection of its
    current position onto the spindle axis.
    """
    from chromlearn.io.trajectory import pole_center

    n_chromosomes = x0.shape[0]
    n_partners = partner_positions.shape[0]
    trajectory = np.zeros((n_steps + 1, 3, n_chromosomes), dtype=np.float64)
    trajectory[0] = x0.T

    # Precompute spindle axis unit vector for each frame
    centrioles = cell_ref.centrioles  # (T, 3, 2)
    center_all = pole_center(cell_ref)  # (T, 3)
    axis_raw = centrioles[:, :, 1] - centrioles[:, :, 0]
    norms = np.linalg.norm(axis_raw, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    axis_unit_all = axis_raw / norms  # (T, 3)

    for step in range(n_steps):
        positions = trajectory[step]  # (3, N)
        forces = np.zeros_like(positions)
        center_t = center_all[step]
        axis_unit_t = axis_unit_all[step]

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
                    forces[:, chrom_index] += (
                        kernel_xx(np.array([distance]))[0] * direction
                    )

            for partner_index in range(n_partners):
                delta = partner_positions[partner_index, step] - current
                distance = float(np.linalg.norm(delta))
                if distance <= 1e-12:
                    continue
                direction = delta / distance
                forces[:, chrom_index] += (
                    kernel_xy(np.array([distance]))[0] * direction
                )

        # Variable-D noise: scale each chromosome by its local D
        for chrom_index in range(n_chromosomes):
            current = positions[:, chrom_index]
            axial_val = float(np.dot(current - center_t, axis_unit_t))
            # Clip to basis domain to avoid extrapolation
            axial_clipped = np.clip(
                np.array([axial_val]),
                diff_result_obj.basis_D.r_min,
                diff_result_obj.basis_D.r_max,
            )
            D_local = float(diff_result_obj.evaluate(axial_clipped)[0])
            D_local = max(D_local, 1e-6)  # ensure positive
            noise_scale = np.sqrt(2.0 * D_local * dt)
            trajectory[step + 1, :, chrom_index] = (
                positions[:, chrom_index]
                + forces[:, chrom_index] * dt
                + noise_scale * rng.standard_normal(3)
            )

    return trajectory


print("Simulating winner model with scalar D and variable D(axial)...")
traj_scalar = simulate_trajectories(
    kernel_xx=_kernel_xx_winner,
    kernel_xy=_kernel_xy_winner,
    partner_positions=winner_partners,
    x0=x0,
    n_steps=n_steps,
    dt=example_cell.dt,
    D_x=winner_model.D_x,
    rng=np.random.default_rng(99),
)
traj_varD = simulate_variable_D(
    kernel_xx=_kernel_xx_winner,
    kernel_xy=_kernel_xy_winner,
    partner_positions=winner_partners,
    x0=x0,
    n_steps=n_steps,
    dt=example_cell.dt,
    diff_result_obj=diff_result,
    cell_ref=example_cell,
    rng=np.random.default_rng(99),
)

# Build TrimmedCells for spindle_frame()
sc_scalar = TrimmedCell(
    cell_id=example_cell.cell_id,
    condition=example_cell.condition,
    centrioles=example_cell.centrioles,
    chromosomes=traj_scalar,
    tracked=N_chrom,
    dt=example_cell.dt,
    start_frame=example_cell.start_frame,
    end_frame=example_cell.end_frame,
)
sc_varD = TrimmedCell(
    cell_id=example_cell.cell_id,
    condition=example_cell.condition,
    centrioles=example_cell.centrioles,
    chromosomes=traj_varD,
    tracked=N_chrom,
    dt=example_cell.dt,
    start_frame=example_cell.start_frame,
    end_frame=example_cell.end_frame,
)
sf_scalar = spindle_frame(sc_scalar)
sf_varD = spindle_frame(sc_varD)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
time_axis = np.arange(T) * example_cell.dt

# Axial
axes[0].plot(time_axis, np.nanmean(sf_real.axial, axis=1), "k-", linewidth=2.5, label="Real")
axes[0].fill_between(time_axis,
                     np.nanmean(sf_real.axial, axis=1) - np.nanstd(sf_real.axial, axis=1),
                     np.nanmean(sf_real.axial, axis=1) + np.nanstd(sf_real.axial, axis=1),
                     color="k", alpha=0.1)
axes[0].plot(time_axis, np.nanmean(sf_scalar.axial, axis=1), "C0--", linewidth=2,
             label=f"Scalar D ({winner_model.D_x:.4f} um^2/s)")
axes[0].plot(time_axis, np.nanmean(sf_varD.axial, axis=1), "C1-.", linewidth=2,
             label="Variable D(axial)")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Mean axial (um)")
axes[0].set_title(f"Axial — {best_topology}")
axes[0].legend(fontsize=8)

# Radial
axes[1].plot(time_axis, np.nanmean(sf_real.radial, axis=1), "k-", linewidth=2.5, label="Real")
axes[1].fill_between(time_axis,
                     np.nanmean(sf_real.radial, axis=1) - np.nanstd(sf_real.radial, axis=1),
                     np.nanmean(sf_real.radial, axis=1) + np.nanstd(sf_real.radial, axis=1),
                     color="k", alpha=0.1)
axes[1].plot(time_axis, np.nanmean(sf_scalar.radial, axis=1), "C0--", linewidth=2,
             label="Scalar D")
axes[1].plot(time_axis, np.nanmean(sf_varD.radial, axis=1), "C1-.", linewidth=2,
             label="Variable D(axial)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Mean radial (um)")
axes[1].set_title(f"Radial — {best_topology}")
axes[1].legend(fontsize=8)

fig.suptitle(f"Scalar D vs variable D(axial) — {example_cell.cell_id} — {best_topology}")
fig.tight_layout()
plt.show()

# %%
# Quantify: MSE of mean axial trajectory vs real
real_axial_mean = np.nanmean(sf_real.axial, axis=1)
mse_scalar = float(np.nanmean((np.nanmean(sf_scalar.axial, axis=1) - real_axial_mean) ** 2))
mse_varD = float(np.nanmean((np.nanmean(sf_varD.axial, axis=1) - real_axial_mean) ** 2))

print(f"Single-cell simulation MSE (axial mean trajectory):")
print(f"  Scalar D:    {mse_scalar:.5f} um^2")
print(f"  Variable D:  {mse_varD:.5f} um^2")

if mse_varD < mse_scalar * 0.9:
    print("  => Variable D noticeably improves simulation fidelity.")
elif mse_varD < mse_scalar:
    print("  => Variable D marginally improves simulation fidelity.")
else:
    print("  => Scalar D is comparable; variable D provides no clear advantage "
          "on this single cell.")

print()
print(f"Scalar D estimate:        {winner_model.D_x:.4f} um^2/s")
print(f"Vestergaard scalar mean:  {diff_result.D_scalar:.4f} um^2/s")
print(f"D(x) range (axial -8..8): [{D_variable_vals.min():.4f}, {D_variable_vals.max():.4f}] um^2/s")
