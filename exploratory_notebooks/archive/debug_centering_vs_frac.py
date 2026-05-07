# %% [markdown]
# # Debug: centering kernel sensitivity to trajectory window length
#
# Does the learned centering stiffness depend on how far into
# prometaphase/metaphase we include?
#
# If chromosomes congress early and then sit quietly near the spindle center,
# including the quiet tail dilutes the signal for the restoring force. This
# notebook fits the `center` topology at several `neb_ao_frac` values and
# compares the learned `f_xy(r)` kernels, focusing on the slope near `r = 0`
# (the effective stiffness).
#
# See `docs/centering_rollout_issue.md` for full context.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chromlearn import find_repo_root

ROOT = find_repo_root(Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd())

from chromlearn.io.catalog import load_condition
from chromlearn.io.trajectory import trim_trajectory, get_partners
from chromlearn.model_fitting import FitConfig
from chromlearn.model_fitting.fit import fit_model

plt.rcParams["figure.dpi"] = 110

# %%
cells_raw = load_condition("rpe18_ctr")
print(f"Loaded {len(cells_raw)} rpe18_ctr cells")

# %% [markdown]
# ## Fit center topology at varying `neb_ao_frac`

# %%
FRACS = np.linspace(.1, 0.5, 5)
R_MIN = 0.1
R_MAX = 12.0

models = {}
n_frames = {}

for frac in FRACS:
    cells = [trim_trajectory(c, method="neb_ao_frac", frac=frac, min_frames=10)
             for c in cells_raw]
    n_cells = len(cells)
    total_frames = sum(c.chromosomes.shape[0] for c in cells)

    config = FitConfig(
        topology="center",
        n_basis_xy=15,
        r_min_xy=R_MIN,
        r_max_xy=R_MAX,
        basis_type="bspline",
        lambda_ridge=1e-3,
        lambda_rough=10,
    )
    models[frac] = fit_model(cells, config)
    n_frames[frac] = total_frames
    print(f"frac={frac:.1f}: {n_cells} cells, {total_frames} total frames, "
          f"D_x={models[frac].D_x:.6f}")

# %% [markdown]
# ## Compare learned centering kernels `f_xy(r)`
#
# If the small-k issue is caused by including too much quiet prometaphase,
# shorter windows (smaller frac) should show a steeper kernel near `r = 0`.

# %%
r_eval = np.linspace(R_MIN, R_MAX, 300)
cmap = plt.cm.plasma
frac_colors = {frac: cmap(i / max(len(FRACS) - 1, 1)) for i, frac in enumerate(FRACS)}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Left panel: full kernel ---
ax = axes[0]
for frac in FRACS:
    f_r = models[frac].evaluate_kernel("xy", r_eval)
    ax.plot(r_eval, f_r, linewidth=2, color=frac_colors[frac], label=f"frac={frac:.2f}")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_xlabel("Distance from spindle center (µm)")
ax.set_ylabel("f_xy(r)  (µm/s per µm)")
ax.set_title("Learned centering kernel")
ax.legend(fontsize=6, ncol=2)
# Sign convention: force = f_xy(r) * (partner - chromosome) / |...|
ax.annotate("+ attractive (toward center)", xy=(0.02, 0.97),
            xycoords="axes fraction", va="top", fontsize=8, color="0.4")
ax.annotate("- repulsive (away from center)", xy=(0.02, 0.03),
            xycoords="axes fraction", va="bottom", fontsize=8, color="0.4")

# --- Right panel: zoom near r = 0 ---
ax = axes[1]
r_zoom = np.linspace(R_MIN, 3.0, 200)
for frac in FRACS:
    f_r = models[frac].evaluate_kernel("xy", r_zoom)
    ax.plot(r_zoom, f_r, linewidth=2, color=frac_colors[frac], label=f"frac={frac:.2f}")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_xlabel("Distance from spindle center (µm)")
ax.set_ylabel("f_xy(r)  (µm/s per µm)")
ax.set_title("Zoom: kernel near spindle center")
ax.legend(fontsize=6, ncol=2)
ax.annotate("+ attractive", xy=(0.02, 0.97),
            xycoords="axes fraction", va="top", fontsize=8, color="0.4")
ax.annotate("- repulsive", xy=(0.02, 0.03),
            xycoords="axes fraction", va="bottom", fontsize=8, color="0.4")

fig.suptitle("Centering kernel sensitivity to trajectory window length", y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Effective stiffness at small displacement
#
# The slope `f_xy(r) / r` near `r = 0` is the effective spring constant. If the
# trajectory window matters, shorter windows should give a steeper slope.

# %%
# Evaluate the effective stiffness as f(r)/r at a small reference distance
r_ref = np.array([0.5, 1.0, 1.5, 2.0])

print("Effective stiffness  k_eff = f_xy(r) / r  (s^{-1}):")
print(f"{'frac':>6s}", end="")
for r in r_ref:
    print(f"  r={r:.1f}µm", end="")
print(f"  {'D_x':>10s}  {'frames':>8s}")
print("-" * 72)

for frac in FRACS:
    f_vals = models[frac].evaluate_kernel("xy", r_ref)
    print(f"{frac:6.1f}", end="")
    for r, f in zip(r_ref, f_vals):
        k_eff = f / r
        print(f"  {k_eff:8.2e}", end="")
    print(f"  {models[frac].D_x:10.6f}  {n_frames[frac]:8d}")

# %% [markdown]
# ## Can the learned stiffness keep up with spindle center movement?
#
# The key question from `docs/centering_rollout_issue.md`: the steady-state lag
# of a spring model is `|y_ss| = |v| / k`. If `k` is too small relative to the
# spindle center speed, the rollout cannot keep chromosomes centered.
#
# Compare the learned restoring drift `f_xy(r)` at typical small displacements
# to the spindle center velocity, both across all cells and for the known-bad
# cell `rpe18_ctr_509`.

# %%
from chromlearn.io.trajectory import get_partners

BAD_CELL_ID = "rpe18_ctr_509"

# Compute spindle center speed statistics for each frac
print("Spindle center axial speed vs learned restoring drift")
print("=" * 80)

for frac in FRACS:
    cells = [trim_trajectory(c, method="neb_ao_frac", frac=frac, min_frames=10)
             for c in cells_raw]

    all_v_ax = []
    bad_cell_v_ax = []

    for cell in cells:
        partners = get_partners(cell, "center")  # (1, T, 3)
        center = partners[0]  # (T, 3)

        pole1 = cell.centrioles[:, :, 0]
        pole2 = cell.centrioles[:, :, 1]
        axis = pole2 - pole1
        axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)
        axis_hat = axis / np.where(axis_norm > 1e-12, axis_norm, 1.0)

        dc_dt = np.diff(center, axis=0) / cell.dt
        v_ax = np.abs(np.sum(dc_dt * axis_hat[:-1], axis=1))
        all_v_ax.append(v_ax)

        if BAD_CELL_ID in cell.cell_id:
            bad_cell_v_ax.append(v_ax)

    all_v_ax = np.concatenate(all_v_ax)

    # Learned restoring drift at typical displacement r = 1 um
    f_at_1 = abs(float(models[frac].evaluate_kernel("xy", np.array([1.0]))[0]))
    k_eff_1 = f_at_1  # k_eff = f(r)/r, so f(1) = k_eff * 1

    print(f"\nfrac={frac:.1f}:")
    print(f"  All cells  — median |v_ax|: {np.median(all_v_ax):.4e} µm/s, "
          f"mean: {np.mean(all_v_ax):.4e} µm/s")
    if bad_cell_v_ax:
        bv = np.concatenate(bad_cell_v_ax)
        print(f"  {BAD_CELL_ID} — median |v_ax|: {np.median(bv):.4e} µm/s, "
              f"mean: {np.mean(bv):.4e} µm/s")
    print(f"  Learned |f_xy(1µm)|:  {f_at_1:.4e} µm/s")
    print(f"  Ratio median(|v_ax|) / |f_xy(1µm)|:  {np.median(all_v_ax) / max(f_at_1, 1e-12):.1f}x")
    if bad_cell_v_ax:
        print(f"  Ratio for {BAD_CELL_ID}:  "
              f"{np.median(bv) / max(f_at_1, 1e-12):.1f}x")

# %% [markdown]
# If the ratio `|v_ax| / |f_xy|` shrinks at smaller frac, the learned force is
# catching up to the spindle speed — the data dilution hypothesis holds.
# If the ratio stays large, the restoring force is genuinely weak relative to
# spindle movement regardless of window length.

# %% [markdown]
# ## Non-parametric binned radial drift vs distance
#
# Raw point cloud of (distance from spindle center, radial drift toward center)
# with binned means overlaid. This is the non-parametric version of the learned
# kernel `f_xy(r)`. Same sign convention: positive = toward center (attractive).

# %%
N_BINS_SC = 20
bin_edges_sc = np.linspace(R_MIN, R_MAX, N_BINS_SC + 1)
bc_sc = 0.5 * (bin_edges_sc[:-1] + bin_edges_sc[1:])
fig, axes = plt.subplots(1, len(FRACS), figsize=(4 * len(FRACS), 4), sharey=True)

for idx, frac in enumerate(FRACS):
    cells = [trim_trajectory(c, method="neb_ao_frac", frac=frac, min_frames=10)
             for c in cells_raw]
    ax = axes[idx]

    all_r, all_vr = [], []
    for cell in cells:
        center = get_partners(cell, "center")[0]  # (T, 3)
        T_frames, _, N_chrom = cell.chromosomes.shape
        for i in range(N_chrom):
            chrom = cell.chromosomes[:, :, i]  # (T, 3)
            delta = center - chrom  # toward center
            dist = np.linalg.norm(delta, axis=1)
            safe_dist = np.where(dist > 1e-12, dist, 1.0)
            direction = delta / safe_dist[:, np.newaxis]
            # Centered difference: (x[t+1] - x[t-1]) / (2*dt)
            vel = (chrom[2:] - chrom[:-2]) / (2 * cell.dt)  # (T-2, 3)
            vr = np.sum(vel * direction[1:-1], axis=1)  # radial drift at t
            r = dist[1:-1]
            mask = ~np.isnan(vr) & ~np.isnan(r)
            all_r.append(r[mask])
            all_vr.append(vr[mask])

    all_r = np.concatenate(all_r)
    all_vr = np.concatenate(all_vr)

    # Point cloud
    ax.scatter(all_r, all_vr, s=1, alpha=0.1, rasterized=True)

    # Binned means
    bin_idx = np.digitize(all_r, bin_edges_sc) - 1
    mu_sc = np.full(N_BINS_SC, np.nan)
    se_sc = np.full(N_BINS_SC, np.nan)
    for b in range(N_BINS_SC):
        sel = bin_idx == b
        if sel.sum() > 5:
            mu_sc[b] = np.mean(all_vr[sel])
            se_sc[b] = np.std(all_vr[sel]) / np.sqrt(sel.sum())
    valid = ~np.isnan(mu_sc)
    ax.errorbar(bc_sc[valid], mu_sc[valid], yerr=se_sc[valid],
                fmt="o-", ms=4, color="red", capsize=2, linewidth=1.5,
                label="binned mean", zorder=5)

    ax.axhline(0, color="k", linestyle=":", alpha=0.3)
    ax.set_ylim(-0.01, 0.1)
    ax.set_xlabel("Distance from spindle center (µm)")
    if idx == 0:
        ax.set_ylabel("Radial drift (µm/s)")
    ax.set_title(f"frac={frac:.1f} (n={len(all_r):,})")
    ax.legend(fontsize=7, loc="upper right")

fig.suptitle("Radial drift toward center vs distance (+ = attractive)", y=1.02)
fig.tight_layout()
plt.show()
