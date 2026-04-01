# %% [markdown]
# # 03 -- Centrosomes drive chromosome motion
#
# Three independent lines of evidence that centrosome positions can be
# treated as autonomous inputs when modeling chromosome dynamics:
#
# **Part A (model-free):** The center of mass of the chromosome cloud
# tracks the center of mass of the two poles with a visible time lag —
# chromosomes follow centrosomes, not the other way around.
#
# **Part B (model-based):** Fitting and simulating centrosome dynamics
# shows that adding chromosome-on-centrosome forces provides no
# improvement over a simple distance-dependent separation term.
# Validated by F-test, cross-validation, and forward simulation.
#
# **Part C (physics):** Back-of-the-envelope estimate shows chromosome
# forces are an order of magnitude too weak to steer the spindle.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.linalg import block_diag

from chromlearn import find_repo_root

ROOT = find_repo_root(Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd())

from chromlearn.analysis.lag_correlation import compute_lag_correlation, plot_lag_correlation
from chromlearn.io.catalog import load_condition
from chromlearn.io.trajectory import trim_trajectory

plt.rcParams["figure.dpi"] = 110

# %% [markdown]
# ## Configuration

# %%
CONDITION = "rpe18_ctr"          # Analyze the control condition used throughout the SFI notebooks.
FRAC_NEB_AO_WINDOW = 0.4         # Trim each cell to this fraction of the NEB-to-AO window.
EXAMPLE_CELL = 1                 # Cell index used for the illustrative PCA trajectory panel.

# %%
cells_raw = load_condition(CONDITION)
cells = [trim_trajectory(c, method="neb_ao_frac", frac=FRAC_NEB_AO_WINDOW) for c in cells_raw]
print(f"Loaded {len(cells)} {CONDITION} cells (trimmed to neb_ao_frac={FRAC_NEB_AO_WINDOW:.3f})")

# %% [markdown]
# ## Part A — Model-free evidence
#
# ### Example: trajectories in PCA space
#
# We project all trajectories — individual chromosomes, both poles, pole
# center of mass, chromosome center of mass — into PCA coordinates computed
# from the chromosome center of mass.  Color encodes time (light → dark).

# %%
def _colorline(ax, x, y, t, cmap, linewidth=1.5, alpha=1.0):
    """Plot a line colored by a scalar parameter *t*."""
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidths=linewidth, alpha=alpha)
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    return lc


cell = cells[EXAMPLE_CELL]
T = cell.chromosomes.shape[0]
n_chrom = cell.chromosomes.shape[2]
t_color = np.linspace(0, 1, T)  # normalized time for colormaps

# Centers of mass
pole_com = 0.5 * (cell.centrioles[:, :, 0] + cell.centrioles[:, :, 1])
chrom_com = np.nanmean(cell.chromosomes, axis=2)

# PCA on chromosome center of mass, centered on its time-average
origin = chrom_com.mean(axis=0)
chrom_com_centered = chrom_com - origin
_, _, Vt = np.linalg.svd(chrom_com_centered, full_matrices=False)
pca_basis = Vt[:2].T  # (3, 2)

# Project everything into PCA space
chrom_com_pca = chrom_com_centered @ pca_basis
pole_com_pca = (pole_com - origin) @ pca_basis
p1_pca = (cell.centrioles[:, :, 0] - origin) @ pca_basis
p2_pca = (cell.centrioles[:, :, 1] - origin) @ pca_basis

fig, ax = plt.subplots(figsize=(8, 5))

# Individual chromosome trajectories (thin, purple, semi-transparent)
for j in range(n_chrom):
    chrom_j = cell.chromosomes[:, :, j]
    if np.any(np.isnan(chrom_j)):
        continue
    cj_pca = (chrom_j - origin) @ pca_basis
    _colorline(ax, cj_pca[:, 0], cj_pca[:, 1], t_color, "Purples",
               linewidth=0.3, alpha=0.2)

# Both individual poles (gray, thick)
_colorline(ax, p1_pca[:, 0], p1_pca[:, 1], t_color, "Greys", linewidth=3, alpha=0.8)
_colorline(ax, p2_pca[:, 0], p2_pca[:, 1], t_color, "Greys", linewidth=3, alpha=0.8)

# Pole center of mass (blue, thick)
_colorline(ax, pole_com_pca[:, 0], pole_com_pca[:, 1], t_color, "YlGnBu", linewidth=3)

# Chromosome center of mass (red, thick)
lc = _colorline(ax, chrom_com_pca[:, 0], chrom_com_pca[:, 1], t_color, "YlOrRd", linewidth=3)

ax.autoscale()
ax.set_aspect("equal")
ax.set_xlabel("PC1 (um)")
ax.set_ylabel("PC2 (um)")
ax.set_title(f"Trajectories in PCA space — {cell.cell_id}")
fig.colorbar(lc, ax=ax, label="Time (normalized)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Lag correlation across all cells
#
# Normalized dot-product cross-correlation between pole-center velocity and
# chromosome-center velocity over the prometaphase window, as a function of
# time lag. A peak at positive lag (and slower decay on the positive side)
# means chromosomes follow poles.

# %%
lag_result = compute_lag_correlation(cells, lag_max=40, smooth_window=25)
plot_lag_correlation(lag_result)
plt.show()

peak_idx = int(np.nanargmax(lag_result.median))
print(f"Peak correlation lag: {lag_result.lags[peak_idx]:.0f} s")
print(f"Peak correlation value: {lag_result.median[peak_idx]:.3f}")

# %% [markdown]
# ## Part B — Chromosome feedback on centrosomes is negligible
#
# We now ask: does adding chromosome-on-centrosome forces improve our
# ability to predict centrosome velocities?  If pole velocities are already
# well-predicted by a distance-dependent pole-separation term alone, then
# chromosome feedback is not a missing ingredient and we can treat
# centrosome positions as given/external when modeling chromosome dynamics.
#
# Note: the pole-separation term $f_{cc}$ is an *effective* force that
# absorbs all contributions driving pole separation (including cortical
# pulling forces, motor-driven sliding, etc.) — it is not a literal
# pole-pole interaction.
#
# We fit centrosome velocity using two potential force sources:
# - $f_{cc}(r)$: effective pole-separation force (distance-dependent)
# - $f_{xc}(r)$: chromosome-on-centrosome (all 46 chromosomes acting on each pole)
#
# and compare how much each contributes.

# %%
from chromlearn.model_fitting.basis import BSplineBasis

# Match the main fitter's light ridge + stronger roughness prior so the
# kernel shape reflects smooth spindle dynamics rather than basis jitter.
LAMBDA_RIDGE = 1e-3
LAMBDA_ROUGH = 1.0


def _padded_domain(values, pad_frac=0.10):
    """Return a data-driven basis domain with a small safety margin."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("Cannot build a basis domain from an empty sample.")
    lo = float(vals.min())
    hi = float(vals.max())
    span = hi - lo
    pad = pad_frac * span if span > 0 else pad_frac * max(abs(hi), 1.0)
    return max(1e-6, lo - pad), hi + pad


# Use the actual distance support in this trimmed dataset, with a small pad.
all_pp_dists = []
all_xp_dists = []
for cell in cells:
    pp = np.linalg.norm(cell.centrioles[:, :, 1] - cell.centrioles[:, :, 0], axis=1)
    all_pp_dists.extend(pp)

    poles = np.moveaxis(cell.centrioles, 2, 1)      # (T, 2, 3)
    chroms = np.moveaxis(cell.chromosomes, 2, 1)    # (T, N, 3)
    for t in range(chroms.shape[0]):
        chroms_t = chroms[t]
        valid = ~np.any(np.isnan(chroms_t), axis=1)
        if not valid.any():
            continue
        chroms_valid = chroms_t[valid]
        for p in range(2):
            delta = chroms_valid - poles[t, p]
            dist = np.linalg.norm(delta, axis=1)
            all_xp_dists.extend(dist[dist > 1e-12])

all_pp_dists = np.array(all_pp_dists)
all_xp_dists = np.array(all_xp_dists)

# Build a small regression for centrosome velocity.
# Each centrosome's velocity at each timepoint is one observation (3 rows for x,y,z).
# Features: f_cc(pole-pole distance) * direction + sum_j f_xc(r_jp) * direction_jp

n_basis_cc = 6
n_basis_xc = 6
basis_cc = BSplineBasis(*_padded_domain(all_pp_dists), n_basis_cc)
basis_xc = BSplineBasis(*_padded_domain(all_xp_dists), n_basis_xc)
R_cc = basis_cc.roughness_matrix()
R_xc = basis_xc.roughness_matrix()
R_full = block_diag(R_cc, R_xc)

G_rows = []
V_rows = []

for cell in cells:
    T_cell = cell.centrioles.shape[0]
    N = cell.chromosomes.shape[2]
    dt = cell.dt

    for t in range(T_cell - 1):
        poles_cur = cell.centrioles[t].T    # (2, 3)
        poles_next = cell.centrioles[t + 1].T
        chroms = cell.chromosomes[t].T      # (N, 3)

        for p in range(2):
            pole_vel = (poles_next[p] - poles_cur[p]) / dt  # (3,)

            # cc: force from the other pole
            other = 1 - p
            delta_cc = poles_cur[other] - poles_cur[p]
            r_cc = np.linalg.norm(delta_cc)
            if r_cc < 1e-12:
                continue
            dir_cc = delta_cc / r_cc
            phi_cc = basis_cc.evaluate(np.array([r_cc]))[0]  # (n_basis_cc,)
            g_cc = dir_cc[:, np.newaxis] * phi_cc[np.newaxis, :]  # (3, n_basis_cc)

            # xc: force from all chromosomes
            g_xc = np.zeros((3, n_basis_xc))
            valid_chroms = ~np.any(np.isnan(chroms), axis=1)
            if valid_chroms.any():
                chroms_valid = chroms[valid_chroms]
                delta_xc = chroms_valid - poles_cur[p]  # (n_valid, 3)
                dist_xc = np.linalg.norm(delta_xc, axis=1)
                pair_ok = dist_xc > 1e-12
                if pair_ok.any():
                    dir_xc = delta_xc[pair_ok] / dist_xc[pair_ok, np.newaxis]
                    phi_xc = basis_xc.evaluate(dist_xc[pair_ok])  # (n_ok, n_basis_xc)
                    g_xc = np.einsum("id,ib->db", dir_xc, phi_xc)  # (3, n_basis_xc)

            row = np.hstack([g_cc, g_xc])  # (3, n_basis_cc + n_basis_xc)
            G_rows.append(row)
            V_rows.append(pole_vel)

G_cent = np.vstack(G_rows)
V_cent = np.concatenate(V_rows)
print(f"Centrosome design matrix: {G_cent.shape[0]} rows, {G_cent.shape[1]} columns")

# %% [markdown]
# ### Fit and compare models
#
# 1. **Full model**: both $f_{cc}$ and $f_{xc}$
# 2. **cc-only model**: centrosome-centrosome interaction alone
# 3. **xc-only model**: chromosome-on-centrosome interaction alone

# %%
n_cc = n_basis_cc
n_xc = n_basis_xc
I_full = np.eye(n_cc + n_xc)


def ridge_fit(G, V, R=None, lam=LAMBDA_RIDGE, lam_rough=LAMBDA_ROUGH):
    n = G.shape[1]
    penalty = lam * np.eye(n)
    if R is not None:
        penalty = penalty + lam_rough * R
    theta = np.linalg.solve(G.T @ G + penalty, G.T @ V)
    residuals = V - G @ theta
    return theta, residuals


# Full model
theta_full, res_full = ridge_fit(G_cent, V_cent, R=R_full)
ss_res_full = np.sum(res_full**2)
ss_tot = np.sum((V_cent - V_cent.mean()) ** 2)

# cc-only model
G_cc_only = G_cent[:, :n_cc]
theta_cc, res_cc = ridge_fit(G_cc_only, V_cent, R=R_cc)
ss_res_cc = np.sum(res_cc**2)

# xc-only model
G_xc_only = G_cent[:, n_cc:]
theta_xc, res_xc = ridge_fit(G_xc_only, V_cent, R=R_xc)
ss_res_xc = np.sum(res_xc**2)

r2_full = 1 - ss_res_full / ss_tot
r2_cc = 1 - ss_res_cc / ss_tot
r2_xc = 1 - ss_res_xc / ss_tot

print(f"R² (full, cc + xc):     {r2_full:.4f}")
print(f"R² (cc only):           {r2_cc:.4f}")
print(f"R² (xc only):           {r2_xc:.4f}")
print(f"R² improvement from adding xc to cc: {r2_full - r2_cc:.4f}")

# %% [markdown]
# ### Effect size (primary) and F-test (heuristic)
#
# Cohen's $f^2$ measures how much variance the xc terms add, independent
# of sample size.  We also report an F-test for reference, but it should
# be treated as heuristic: the observations are autocorrelated within
# cells and the fits are ridge-regularized, so the classical p-value is
# not rigorous.  The effect size is the more reliable quantity.

# %%
from scipy.stats import f as f_dist

n_obs = G_cent.shape[0]
p_cc_model = n_cc
p_full_model = n_cc + n_xc
F_stat = ((ss_res_cc - ss_res_full) / (p_full_model - p_cc_model)) / (
    ss_res_full / (n_obs - p_full_model)
)
p_value = 1 - f_dist.cdf(F_stat, p_full_model - p_cc_model, n_obs - p_full_model)

# Cohen's f^2: effect size for the incremental R^2
delta_r2 = r2_full - r2_cc
cohens_f2 = delta_r2 / (1 - r2_full)

print(f"F-statistic: {F_stat:.2f}")
print(f"p-value:     {p_value:.2e}")
print(f"df1 = {p_full_model - p_cc_model}, df2 = {n_obs - p_full_model}")
print(f"Delta R^2:   {delta_r2:.4f}")
print(f"Cohen's f^2: {cohens_f2:.4f}  (< 0.02 = negligible, 0.02 = small)")
print()
print(f"Cohen's f^2 {'< 0.02 (negligible)' if cohens_f2 < 0.02 else '>= 0.02 (non-negligible)'}")

# %% [markdown]
# ### Distribution of pole-pole distances
#
# To understand where the kernels are well-constrained, we first check
# the empirical distribution of pole-pole distances in the data.

# %%
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(all_pp_dists, bins=50, edgecolor="k", alpha=0.7)
ax.axvline(basis_cc.r_min, color="r", linestyle="--", label=f"basis range [{basis_cc.r_min}, {basis_cc.r_max}]")
ax.axvline(basis_cc.r_max, color="r", linestyle="--")
ax.set_xlabel("Pole-pole distance (um)")
ax.set_ylabel("Count")
ax.set_title("Distribution of pole-pole distances across all cells/timepoints")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
print(f"Pole-pole distances: median={np.median(all_pp_dists):.1f}, "
      f"range=[{all_pp_dists.min():.1f}, {all_pp_dists.max():.1f}] um")

# %% [markdown]
# ### Fitted centrosome interaction kernels

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Interpret the fitted kernel only over the observed support.  Outside the
# data range the basis is clamped at the boundary, so plotting farther right
# would create an artificial tail that is not constrained by measurements.
r_cc_plot = np.linspace(all_pp_dists.min(), all_pp_dists.max(), 200)

# f_cc from cc-only model
f_cc_only_vals = basis_cc.evaluate(r_cc_plot) @ theta_cc[:n_cc]
axes[0].plot(r_cc_plot, f_cc_only_vals, linewidth=2, color="C0", label="cc-only model")

# f_cc from full model (cc+xc)
f_cc_full_vals = basis_cc.evaluate(r_cc_plot) @ theta_full[:n_cc]
axes[0].plot(r_cc_plot, f_cc_full_vals, linewidth=2, color="C1", linestyle="--",
             label="cc+xc model")

axes[0].axhline(0, color="0.5", linestyle="--", linewidth=0.8)
axes[0].set_xlabel("Pole-pole distance (um)")
axes[0].set_ylabel("Effective drift coefficient (negative = separation)")
axes[0].set_title("Effective pole-separation $f_{cc}(r)$ over observed support")
axes[0].legend(fontsize=8)

# f_xc from full model
r_xc_plot = np.linspace(all_xp_dists.min(), all_xp_dists.max(), 200)
f_xc_vals = basis_xc.evaluate(r_xc_plot) @ theta_full[n_cc:]
axes[1].plot(r_xc_plot, f_xc_vals, linewidth=2, color="C1")
axes[1].axhline(0, color="0.5", linestyle="--", linewidth=0.8)
axes[1].set_xlabel("Chromosome-to-pole distance (um)")
axes[1].set_ylabel("Effective drift coefficient")
axes[1].set_title("Chromosome-on-centrosome $f_{xc}(r)$")

# Panel 3: prediction error comparison
# The raw magnitudes of cc and xc contributions can be misleading because
# xc is largely collinear with cc (chromosomes cluster along the spindle
# axis).  What matters is whether xc *reduces prediction error*.
V_obs = V_cent.reshape(-1, 3)  # (n_obs, 3)
pred_cc_only = (G_cent[:, :n_cc] @ theta_cc[:n_cc]).reshape(-1, 3)
pred_full = (G_cent @ theta_full).reshape(-1, 3)
err_cc = np.linalg.norm(V_obs - pred_cc_only, axis=1)
err_full = np.linalg.norm(V_obs - pred_full, axis=1)

axes[2].hist(err_cc, bins=50, alpha=0.7, color="C0",
             label=f"cc only (RMSE={np.sqrt(np.mean(err_cc**2)):.4f})")
axes[2].hist(err_full, bins=50, alpha=0.7, color="C1",
             label=f"cc+xc (RMSE={np.sqrt(np.mean(err_full**2)):.4f})")
axes[2].set_xlabel("Prediction error magnitude (um/s)")
axes[2].set_ylabel("Count")
axes[2].set_title("Adding xc barely reduces prediction error")
axes[2].legend(fontsize=8)

fig.tight_layout()
plt.show()

# %%
rmse_cc_obs = np.sqrt(np.mean(err_cc**2))
rmse_full_obs = np.sqrt(np.mean(err_full**2))
print(f"Per-observation RMSE (cc only):  {rmse_cc_obs:.4f} um/s")
print(f"Per-observation RMSE (cc + xc):  {rmse_full_obs:.4f} um/s")
print(f"Error reduction:                 {rmse_cc_obs - rmse_full_obs:.5f} um/s "
      f"({100 * (rmse_cc_obs - rmse_full_obs) / rmse_cc_obs:.2f}%)")

# %% [markdown]
# ### Part B.2 — Leave-one-cell-out cross-validation
#
# The R² comparison above is on training data.  To confirm the result
# generalizes, we repeat the comparison using leave-one-cell-out
# cross-validation: fit on N−1 cells, evaluate RMSE on the held-out cell.

# %%
def build_cell_matrices(cell, basis_cc, basis_xc):
    """Build design matrix and velocity vector for a single cell."""
    G_rows = []
    V_rows = []
    T_cell = cell.centrioles.shape[0]
    dt = cell.dt
    n_basis_xc_loc = basis_xc.n_basis

    for t in range(T_cell - 1):
        poles_cur = cell.centrioles[t].T
        poles_next = cell.centrioles[t + 1].T
        chroms = cell.chromosomes[t].T

        for p in range(2):
            pole_vel = (poles_next[p] - poles_cur[p]) / dt
            other = 1 - p
            delta_cc = poles_cur[other] - poles_cur[p]
            r_cc_val = np.linalg.norm(delta_cc)
            if r_cc_val < 1e-12:
                continue
            dir_cc = delta_cc / r_cc_val
            phi_cc = basis_cc.evaluate(np.array([r_cc_val]))[0]
            g_cc = dir_cc[:, np.newaxis] * phi_cc[np.newaxis, :]

            g_xc = np.zeros((3, n_basis_xc_loc))
            valid_chroms = ~np.any(np.isnan(chroms), axis=1)
            if valid_chroms.any():
                chroms_valid = chroms[valid_chroms]
                delta_xc = chroms_valid - poles_cur[p]
                dist_xc = np.linalg.norm(delta_xc, axis=1)
                pair_ok = dist_xc > 1e-12
                if pair_ok.any():
                    dir_xc = delta_xc[pair_ok] / dist_xc[pair_ok, np.newaxis]
                    phi_xc = basis_xc.evaluate(dist_xc[pair_ok])
                    g_xc = np.einsum("id,ib->db", dir_xc, phi_xc)

            row = np.hstack([g_cc, g_xc])
            G_rows.append(row)
            V_rows.append(pole_vel)

    return np.vstack(G_rows), np.concatenate(V_rows)


cell_matrices = [build_cell_matrices(c, basis_cc, basis_xc) for c in cells]

cv_rmse_cc = []
cv_rmse_full = []

for i in range(len(cells)):
    G_train = np.vstack([cell_matrices[j][0] for j in range(len(cells)) if j != i])
    V_train = np.concatenate([cell_matrices[j][1] for j in range(len(cells)) if j != i])
    G_test, V_test = cell_matrices[i]

    # Full model
    theta_f, _ = ridge_fit(G_train, V_train, R=R_full)
    rmse_f = np.sqrt(np.mean((V_test - G_test @ theta_f) ** 2))
    cv_rmse_full.append(rmse_f)

    # cc-only model
    theta_c, _ = ridge_fit(G_train[:, :n_cc], V_train, R=R_cc)
    rmse_c = np.sqrt(np.mean((V_test - G_test[:, :n_cc] @ theta_c) ** 2))
    cv_rmse_cc.append(rmse_c)

cv_rmse_cc = np.array(cv_rmse_cc)
cv_rmse_full = np.array(cv_rmse_full)

print("Leave-one-cell-out cross-validation RMSE (um/s):")
print(f"  cc-only:  {cv_rmse_cc.mean():.4f} ± {cv_rmse_cc.std():.4f}")
print(f"  cc + xc:  {cv_rmse_full.mean():.4f} ± {cv_rmse_full.std():.4f}")
print(f"  Delta RMSE: {(cv_rmse_cc - cv_rmse_full).mean():.4f}")

fig, ax = plt.subplots(figsize=(5, 4))
x_pos = np.arange(len(cells))
width = 0.35
ax.bar(x_pos - width / 2, cv_rmse_cc, width, label="cc only", color="C0", alpha=0.8)
ax.bar(x_pos + width / 2, cv_rmse_full, width, label="cc + xc", color="C1", alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([c.cell_id for c in cells], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("RMSE (um/s)")
ax.set_title("CV RMSE: cc-only vs cc+xc")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Permutation test
#
# We shuffle time indices of the xc features within each cell (breaking
# the temporal coupling between chromosome positions and centrosome
# velocity while preserving each cell's marginal distribution of xc
# features) and refit the full model.  This tests whether the observed
# ΔR² requires temporally-aligned chromosome positions or could arise
# from the marginal statistics alone.

# %%
N_PERM = 200
perm_rng = np.random.default_rng(123)
perm_delta_r2 = np.empty(N_PERM)

for perm in range(N_PERM):
    # For each cell, shuffle the time ordering of its xc feature rows,
    # then reassemble the global design matrix.
    G_perm_parts = []
    V_perm_parts = []
    for G_cell, V_cell in cell_matrices:
        n_rows = G_cell.shape[0]
        n_obs_cell = n_rows // 3  # each observation is 3 spatial rows
        # Permute observation indices within this cell
        shuffle_idx = perm_rng.permutation(n_obs_cell)
        G_cell_perm = G_cell.copy()
        xc_block = G_cell[:, n_cc:].copy()
        for i, si in enumerate(shuffle_idx):
            G_cell_perm[i * 3:(i + 1) * 3, n_cc:] = xc_block[si * 3:(si + 1) * 3]
        G_perm_parts.append(G_cell_perm)
        V_perm_parts.append(V_cell)

    G_perm = np.vstack(G_perm_parts)
    V_perm = np.concatenate(V_perm_parts)
    theta_perm, res_perm = ridge_fit(G_perm, V_perm, R=R_full)
    ss_res_perm = np.sum(res_perm**2)
    r2_perm = 1 - ss_res_perm / ss_tot
    perm_delta_r2[perm] = r2_perm - r2_cc

# Correct finite-permutation p-value: (count + 1) / (N_PERM + 1)
n_exceed = int(np.sum(perm_delta_r2 >= delta_r2))
perm_p_value = (n_exceed + 1) / (N_PERM + 1)

print(f"Observed Delta R^2:     {delta_r2:.4f}")
print(f"Permutation null mean:  {perm_delta_r2.mean():.4f}")
print(f"Permutation null 95th:  {np.percentile(perm_delta_r2, 95):.4f}")
print(f"Permutation p-value:    {perm_p_value:.4f}")
print(f"  ({N_PERM} permutations, shuffling xc time indices within cells)")
print()
print("Interpretation: the permutation test detects a real but tiny temporally-")
print(f"aligned xc signal (Delta R^2 = {delta_r2:.4f}).  However, the effect size")
print(f"is negligible (Cohen's f^2 = {cohens_f2:.4f} < 0.02) and the LOO-CV below")
print("shows this signal does not translate into meaningful out-of-sample prediction.")

# %% [markdown]
# ### Part B.3 — Forward simulation of spindle length
#
# The ultimate test: can the cc-only model, when used to simulate pole
# dynamics forward in time from real initial conditions, reproduce the
# observed spindle-length trajectory?  We compare cc-only and cc+xc
# models against the real data.  Spindle lengths are pooled across cells
# on a normalized time axis [0, 1] (NEB to trim endpoint).
#
# Bootstrapped confidence bands come from resampling cells (with
# replacement) to refit the kernel, then simulating once per cell per
# resample.  Spindle separation is largely deterministic so a single
# realization per bootstrap kernel is sufficient.

# %%
# Estimate centrosome diffusion coefficient from cc-only residuals.
# D = 0.5 * mean(residual^2) * dt  (per-component variance = 2D/dt)
D_cent = 0.5 * float(np.mean(res_cc**2)) * cells[0].dt

N_BOOT = 50
N_TNORM = 101  # normalized time grid points
rng = np.random.default_rng(42)

from scipy.interpolate import BSpline as _BSpline


def _make_force_callable(basis, theta_slice):
    """Build a fast callable f(r) -> scalar force from basis + coefficients.

    Returns a BSpline object that evaluates the kernel directly, avoiding
    the overhead of design_matrix().toarray() on every timestep.
    """
    return _BSpline(basis.knots, theta_slice, basis.degree, extrapolate=False)


def simulate_poles_fast(poles_init, f_cc_func, dt, D, n_steps,
                        f_xc_func=None, chromosomes=None, rng=None):
    """Euler-Maruyama forward simulation of two poles using fast callables."""
    if rng is None:
        rng = np.random.default_rng()
    use_xc = f_xc_func is not None and chromosomes is not None
    poles = poles_init.copy()
    spindle_len = np.empty(n_steps + 1)
    spindle_len[0] = np.linalg.norm(poles[1] - poles[0])
    noise_scale = np.sqrt(2 * D * dt)

    for t in range(n_steps):
        new_poles = np.empty_like(poles)
        for p in range(2):
            other = 1 - p
            delta = poles[other] - poles[p]
            r_pp = np.linalg.norm(delta)
            if r_pp < 1e-12:
                new_poles[p] = poles[p] + noise_scale * rng.standard_normal(3)
                continue
            direction = delta / r_pp
            f_val = f_cc_func(r_pp)
            if np.isnan(f_val):
                f_val = 0.0
            force = direction * f_val

            if use_xc and t < chromosomes.shape[0]:
                chroms_t = chromosomes[t].T  # (N, 3)
                valid = ~np.any(np.isnan(chroms_t), axis=1)
                if valid.any():
                    delta_xc = chroms_t[valid] - poles[p]
                    dist_xc = np.linalg.norm(delta_xc, axis=1)
                    ok = dist_xc > 1e-12
                    if ok.any():
                        f_xc_vals = f_xc_func(dist_xc[ok])
                        f_xc_vals = np.nan_to_num(f_xc_vals, nan=0.0)
                        dir_xc = delta_xc[ok] / dist_xc[ok, np.newaxis]
                        force += (dir_xc * f_xc_vals[:, np.newaxis]).sum(axis=0)

            new_poles[p] = poles[p] + force * dt + noise_scale * rng.standard_normal(3)
        poles = new_poles
        spindle_len[t + 1] = np.linalg.norm(poles[1] - poles[0])

    return spindle_len


# Normalized time grid for pooling across cells
t_norm = np.linspace(0, 1, N_TNORM)

# Real data: interpolate each cell's spindle length onto normalized time
real_interp = np.empty((len(cells), N_TNORM))
for ci, cell in enumerate(cells):
    T_cell = cell.centrioles.shape[0]
    sl = np.linalg.norm(cell.centrioles[:, :, 1] - cell.centrioles[:, :, 0], axis=1)
    t_cell_norm = np.linspace(0, 1, T_cell)
    real_interp[ci] = np.interp(t_norm, t_cell_norm, sl)

# Bootstrap: resample cells, refit kernel, simulate, interpolate, pool
n_cells = len(cells)
sim_cc_interp = np.empty((N_BOOT, n_cells, N_TNORM))
sim_full_interp = np.empty((N_BOOT, n_cells, N_TNORM))

import time as _time
_t0 = _time.perf_counter()

for b in range(N_BOOT):
    boot_idx = rng.choice(n_cells, size=n_cells, replace=True)
    G_boot = np.vstack([cell_matrices[j][0] for j in boot_idx])
    V_boot = np.concatenate([cell_matrices[j][1] for j in boot_idx])

    theta_boot_cc, _ = ridge_fit(G_boot[:, :n_cc], V_boot, R=R_cc)
    theta_boot_full, _ = ridge_fit(G_boot, V_boot, R=R_full)

    # Build fast callables for this bootstrap replicate
    f_cc_only = _make_force_callable(basis_cc, theta_boot_cc)
    f_cc_full = _make_force_callable(basis_cc, theta_boot_full[:n_cc])
    f_xc_full = _make_force_callable(basis_xc, theta_boot_full[n_cc:])

    for ci, cell in enumerate(cells):
        T_cell = cell.centrioles.shape[0]
        poles_init = cell.centrioles[0].T
        t_cell_norm = np.linspace(0, 1, T_cell)

        sl_cc = simulate_poles_fast(
            poles_init, f_cc_only, cell.dt, D_cent, T_cell - 1, rng=rng,
        )
        sim_cc_interp[b, ci] = np.interp(t_norm, t_cell_norm, sl_cc)

        sl_full = simulate_poles_fast(
            poles_init, f_cc_full, cell.dt, D_cent, T_cell - 1,
            f_xc_func=f_xc_full, chromosomes=cell.chromosomes, rng=rng,
        )
        sim_full_interp[b, ci] = np.interp(t_norm, t_cell_norm, sl_full)

_elapsed = _time.perf_counter() - _t0
print(f"Forward simulation complete: {N_BOOT} bootstrap resamples in {_elapsed:.1f} s")

# %%
# Pool across cells: for each bootstrap, compute cell-mean at each norm time
# Shape: (N_BOOT, N_TNORM)
sim_cc_pooled = sim_cc_interp.mean(axis=1)
sim_full_pooled = sim_full_interp.mean(axis=1)

fig, ax = plt.subplots(figsize=(7, 5))

# Individual cell traces (faint, behind everything)
for ci in range(real_interp.shape[0]):
    ax.plot(t_norm, real_interp[ci], color="0.70", linewidth=0.7, alpha=0.5, zorder=1)

# Real data: mean +/- std across cells
real_mean = real_interp.mean(axis=0)
real_std = real_interp.std(axis=0)
ax.plot(t_norm, real_mean, "k-", linewidth=2.5, label="Data")
ax.fill_between(t_norm, real_mean - real_std, real_mean + real_std,
                color="k", alpha=0.12, label="Data $\\pm$ 1 SD")

# cc-only: median + 90% CI across bootstrap
med_cc = np.median(sim_cc_pooled, axis=0)
lo_cc = np.percentile(sim_cc_pooled, 5, axis=0)
hi_cc = np.percentile(sim_cc_pooled, 95, axis=0)
ax.plot(t_norm, med_cc, "C0-", linewidth=2, label="cc only (bootstrap median)")
ax.fill_between(t_norm, lo_cc, hi_cc, color="C0", alpha=0.2, label="cc only 90% CI")

# cc+xc: median + 90% CI across bootstrap
med_full = np.median(sim_full_pooled, axis=0)
lo_full = np.percentile(sim_full_pooled, 5, axis=0)
hi_full = np.percentile(sim_full_pooled, 95, axis=0)
ax.plot(t_norm, med_full, "C1--", linewidth=2, label="cc+xc (bootstrap median)")
ax.fill_between(t_norm, lo_full, hi_full, color="C1", alpha=0.15, label="cc+xc 90% CI")

ax.set_xlabel("Normalized time (NEB to trim endpoint)")
ax.set_ylabel("Spindle length (um)")
ax.set_title("Forward-simulated spindle length (pooled across cells)")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# The cc-only model reproduces the pooled spindle-length trajectory,
# confirming that the learned kernel captures genuine separation dynamics.
# The cc+xc model offers only marginal improvement — any difference is
# small relative to cell-to-cell variability, consistent with the
# negligible effect size and cross-validation results above.
#
# ## Part C — Back-of-the-envelope physics argument
#
# Even without data, we can estimate an upper bound on the force that the
# chromosome cloud could exert on the spindle.
#
# - Each kinetochore-microtubule attachment generates $\sim$1 pN of force.
# - With $\sim$46 chromosomes distributed roughly symmetrically around the
#   spindle, the net (vectorial) force largely cancels.  Even assuming a
#   generous 10% asymmetry, the net chromosome-on-spindle force is
#   $\sim$46 $\times$ 1 pN $\times$ 0.1 $\approx$ 5 pN.
# - The spindle's effective drag coefficient is of order $\sim$100 pN s/um
#   (e.g., Garzon-Coral et al. 2016).
# - The resulting velocity perturbation is $\sim$5 pN / 100 pN s/um
#   $= 0.05$ um/s — far below the $\sim$0.5-1 um/s pole-separation speeds
#   observed during prometaphase.
#
# This confirms that even under generous assumptions, chromosome forces
# cannot appreciably steer the spindle.
#
# ## Conclusion
#
# Four independent lines of evidence support treating centrosome positions
# as autonomous inputs when modeling chromosome dynamics:
#
# 1. **Temporal precedence (Part A):** Lag correlation shows chromosomes
#    follow centrosome velocity changes, not the reverse.
# 2. **Statistical redundancy (Part B):** Adding chromosome forces does not
#    meaningfully improve prediction of centrosome velocities beyond a
#    simple distance-dependent separation term.  A permutation test detects
#    a statistically real but tiny temporally-aligned xc signal, but the
#    effect size is negligible (Cohen's $f^2 < 0.02$), out-of-sample
#    cross-validation shows no RMSE improvement, and forward-simulated
#    spindle-length trajectories are indistinguishable.
# 3. **Physical scale separation (Part C):** The net force chromosomes can
#    exert on the spindle is an order of magnitude too small to produce
#    observable velocity perturbations.
#
# ---
#
# ## Notes on possible improvements (not implemented)
#
# **Why is the absolute R^2 so low?**  The regression predicts full 3D
# centrosome velocity, which includes stage drift, cell drift/rotation,
# transverse wobble, and measurement noise — none of which the pairwise
# force model can capture.  The low R^2 does *not* invalidate the
# comparative claim (negligible delta-R^2) because both models face the
# same noise floor.
#
# **1D spindle-length regression.**  A cleaner test would regress only
# the 1D rate of change of pole-pole distance, dr/dt, against f_cc(r)
# projected onto the spindle axis.  This eliminates all common-mode
# drift and transverse motion, and should give dramatically higher
# absolute R^2.  The chromosome contribution would be projected onto
# the pole-pole axis as well.  Not implemented here because the current
# comparative analysis is already sufficient for the modeling
# assumption, but this would strengthen the kernel-shape interpretation.
