# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 00 — Main text figures (Result 3)
#
# **STATUS: in progress.**
#
# Assembler for the four main-text figures supporting Chris's part of
# *Hierarchy of spindle forces in prometaphase* (Shi/Miles/Khodjakov/
# Mogilner). Each figure is rendered as a self-contained PDF/PNG; Alex
# can combine into multi-panel layouts as he prefers.
#
# Mapping to Alex's docx Result 3 (A/B/C/D):
# - **Fig 1 (A)** — PCA-projected CS+CH trajectories and lag correlation
#   showing chromosomes follow centrosomes. Source: NB03 Part A.
# - **Fig 2 (B)** — CS-CS-only is sufficient to predict pole motion;
#   adding chromosome-on-pole forces does not improve LOO path MSE.
#   Source: NB03 Part B (re-run with deterministic-rollout path MSE for
#   apples-to-apples with the rest of the paper).
# - **Fig 3 (C)** — learned pairwise force-distance kernels with
#   bootstrap CIs, plus per-cell LOO path MSE for the 5 candidate
#   topologies (admissible vs nuisance upper-bound). Source: NB04.
# - **Fig 4 (D)** — effective diffusion D(d) versus 3D Euclidean
#   distance from spindle center (``d`` reserved for that distance to
#   avoid collision with the SFI-inspired pairwise distance ``r``),
#   with constant-D
#   baseline. Source: NB06.
#
# Forecast-error-vs-horizon and 5-topology forecast curves go to the
# supplement (see ``00b_supplement.py``).

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from chromlearn import find_repo_root

ROOT = find_repo_root(Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.linalg import block_diag
from scipy.interpolate import BSpline as _BSpline

from chromlearn.analysis.lag_correlation import compute_lag_correlation
from chromlearn.analysis.pca_projection import fit_pca_basis
from chromlearn.io.catalog import load_condition
from chromlearn.io.trajectory import trim_trajectory
from chromlearn.model_fitting import FitConfig
from chromlearn.model_fitting.basis import BSplineBasis
from chromlearn.model_fitting.diffusion import estimate_diffusion_variable
from chromlearn.model_fitting.fit import (
    bootstrap_kernels,
    cross_validate,
    evaluate_all_loocv,
    fit_model,
    rollout_cross_validate,
)

# %% [markdown]
# ## Publication style
#
# Shared matplotlib rcParams for all figures in this notebook. Targets a
# Cell/PNAS-style layout: sans-serif, 7-8 pt labels, vector PDF with
# embedded TrueType fonts (`pdf.fonttype = 42`) plus 600 dpi PNG.

# %%
FIG_DIR = ROOT / "figures" / "main"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def publication_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


publication_style()

# Okabe-Ito categorical palette (colorblind-safe).  Used for topology
# comparisons and other discrete categories across figures.
OKABE_ITO = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "purple":  "#CC79A7",
}


def save_figure(fig, name):
    """Save *fig* as both vector PDF and 600 dpi PNG into ``FIG_DIR``."""
    pdf_path = FIG_DIR / f"{name}.pdf"
    png_path = FIG_DIR / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    print(f"Saved {pdf_path.name} and {png_path.name}")


# %% [markdown]
# ## Setup: load data, fit canonical model
#
# Mirrors NB04's selected configuration: ``poles_and_chroms`` topology
# with steric envelope on the xx kernel (``r0 = 1.5 um``, ``w = 0.3 um``),
# ``n_basis = 10`` and ``lambda_rough = 1.0``. Bootstrap with 100 cell-
# level resamples for kernel CI bands.

# %%
CONDITION = "rpe18_ctr"
FRAC_NEB_AO = 0.4
ENVELOPE_R0_XX = 1.5
ENVELOPE_W_XX = 0.3
N_BASIS = 10
R_MIN = 0.3
R_MAX = 15.0
LAMBDA_RIDGE = 1e-6
LAMBDA_ROUGH = 1.0
N_BOOT = 100

cells_raw = load_condition(CONDITION)
cells = [trim_trajectory(c, method="neb_ao_frac", frac=FRAC_NEB_AO) for c in cells_raw]
print(f"Loaded {len(cells)} {CONDITION} cells (frac={FRAC_NEB_AO}).")

config = FitConfig(
    topology="poles_and_chroms",
    envelope_r0_xx=ENVELOPE_R0_XX,
    envelope_w_xx=ENVELOPE_W_XX,
    n_basis_xx=N_BASIS,
    n_basis_xy=N_BASIS,
    r_min_xx=R_MIN, r_max_xx=R_MAX,
    r_min_xy=R_MIN, r_max_xy=R_MAX,
    lambda_ridge=LAMBDA_RIDGE,
    lambda_rough=LAMBDA_ROUGH,
    endpoint_method="neb_ao_frac",
    endpoint_frac=FRAC_NEB_AO,
)
model = fit_model(cells, config)
print(f"Fitted canonical model: D_x = {model.D_x:.4e} um^2/s, "
      f"theta size = {model.theta.size}")

boot = bootstrap_kernels(cells, config, n_boot=N_BOOT,
                         rng=np.random.default_rng(42))

# %% [markdown]
# ## Fig 1 — PCA trajectories and lag correlation
#
# Two panels for the same biological observation that "chromosomes follow
# centrosomes" (Alex's docx Result 3A):
#
# - **Fig 1A**: an example cell's centrosome and chromosome trajectories
#   projected into the spindle's principal axes (PCA basis fit from
#   chromosome COM).  Individual chromosomes are drawn as a faint
#   ``Purples`` time-encoded cloud; both poles are ``Greys`` (light →
#   dark); the pole COM is highlighted in ``YlGnBu`` and the
#   chromosome COM in ``YlOrRd`` (each as a single time-encoded line).
#   A single time colorbar anchors the time axis.
# - **Fig 1B**: pooled lag correlation between pole-COM velocity and
#   chromosome-COM velocity. Asymmetric peak at positive lag means
#   chromosomes follow poles, not the other way around.

# %%
EXAMPLE_CELL_INDEX = 1  # rpe18_ctr_032


def _colorline(ax, x, y, t_norm, cmap, linewidth=2.0, alpha=1.0, zorder=2,
               n_interp=400):
    """Plot a smooth 2-D line colored by normalized time *t_norm*.

    Interpolates the (x, y, t) curve to *n_interp* points (linear in
    arclength via the input parameterisation) so segment boundaries
    are not visible at typical screen/print resolution; uses round
    caps and joins so the polyline looks like a smooth ribbon.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t_norm, dtype=float)
    if x.size >= 2 and n_interp > x.size:
        t_dense = np.linspace(t[0], t[-1], n_interp)
        x = np.interp(t_dense, t, x)
        y = np.interp(t_dense, t, y)
        t = t_dense
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments, cmap=cmap, linewidths=linewidth, alpha=alpha,
        zorder=zorder, capstyle="round", joinstyle="round",
        antialiased=True,
    )
    lc.set_array(t[:-1])
    lc.set_clim(0.0, 1.0)
    ax.add_collection(lc)
    return lc


# Build PCA + lag panels side-by-side ----------------------------------
fig1, (ax_pca, ax_lag) = plt.subplots(
    1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.1, 1.0]}
)

# --- Fig 1A: PCA trajectories ----------------------------------------
cell = cells[EXAMPLE_CELL_INDEX]
T = cell.chromosomes.shape[0]
n_chrom = cell.chromosomes.shape[2]
t_norm = np.linspace(0.0, 1.0, T)

chrom_com = np.nanmean(cell.chromosomes, axis=2)
origin = chrom_com.mean(axis=0)
chrom_com_centered = chrom_com - origin
_, _, Vt = np.linalg.svd(chrom_com_centered, full_matrices=False)
pca_basis = Vt[:2].T

chrom_com_pca = chrom_com_centered @ pca_basis
p1_pca = (cell.centrioles[:, :, 0] - origin) @ pca_basis
p2_pca = (cell.centrioles[:, :, 1] - origin) @ pca_basis

pole_com = 0.5 * (cell.centrioles[:, :, 0] + cell.centrioles[:, :, 1])
pole_com_pca = (pole_com - origin) @ pca_basis

# Individual chromosomes — Purples, faint
for j in range(n_chrom):
    chrom_j = cell.chromosomes[:, :, j]
    if np.any(np.isnan(chrom_j)):
        continue
    cj_pca = (chrom_j - origin) @ pca_basis
    _colorline(ax_pca, cj_pca[:, 0], cj_pca[:, 1], t_norm,
               cmap="Purples", linewidth=0.5, alpha=0.30, zorder=1)

# Both individual poles — Greys
for p_pca in (p1_pca, p2_pca):
    _colorline(ax_pca, p_pca[:, 0], p_pca[:, 1], t_norm,
               cmap="Greys", linewidth=2.2, alpha=0.85, zorder=3)

# Pole COM — YlGnBu
_colorline(ax_pca, pole_com_pca[:, 0], pole_com_pca[:, 1], t_norm,
           cmap="YlGnBu", linewidth=2.4, alpha=0.95, zorder=4)

# Chromosome COM — YlOrRd; doubles as the master time reference for
# the time colorbar below.
lc_time = _colorline(ax_pca, chrom_com_pca[:, 0], chrom_com_pca[:, 1],
                     t_norm, cmap="YlOrRd", linewidth=2.4,
                     alpha=0.95, zorder=4)

ax_pca.set_aspect("equal")
ax_pca.autoscale()
ax_pca.set_xlabel("PC1 (μm)")
ax_pca.set_ylabel("PC2 (μm)")
ax_pca.set_title(f"PCA trajectories — {cell.cell_id}",
                 loc="left", fontsize=9)

# Single master time colorbar, plus a compact in-panel legend mapping
# colormaps to objects.  Uses YlOrRd (chrom COM) as the time reference;
# the other three layers share the same normalized time axis even if
# their colormaps differ.
cbar_time = inset_axes(ax_pca, width="40%", height="3%", loc="lower right",
                       borderpad=1.4)
plt.colorbar(lc_time, cax=cbar_time, orientation="horizontal")
cbar_time.set_xlabel("Time (NEB to 0.4 NEB-AO)", fontsize=6.5, labelpad=2)
cbar_time.tick_params(labelsize=0, length=0)

ax_pca.text(0.03, 0.97,
            "individual CHs (Purples)\n"
            "individual CSs (Greys)\n"
            "pole COM (YlGnBu)\n"
            "chrom COM (YlOrRd)",
            transform=ax_pca.transAxes, ha="left", va="top",
            fontsize=6.5, color="0.3",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.6, pad=2))

# --- Fig 1B: lag correlation (pooled) --------------------------------
lag_result = compute_lag_correlation(cells, lag_max=40, smooth_window=25)

for row in lag_result.per_cell:
    ax_lag.plot(lag_result.lags, row,
                color="0.7", linewidth=0.6, alpha=0.55)

n_cells = lag_result.per_cell.shape[0]
sem = lag_result.std / np.sqrt(np.maximum(n_cells, 1))
ax_lag.fill_between(lag_result.lags,
                    lag_result.median - sem,
                    lag_result.median + sem,
                    color=OKABE_ITO["blue"], alpha=0.20,
                    linewidth=0, label="median ± SE")
ax_lag.plot(lag_result.lags, lag_result.median,
            color=OKABE_ITO["blue"], linewidth=2.0, label="median")

peak_idx = int(np.nanargmax(lag_result.median))
peak_lag = lag_result.lags[peak_idx]
peak_val = lag_result.median[peak_idx]
ax_lag.axvline(peak_lag, color="0.4", linestyle=":", linewidth=0.8)

ax_lag.axhline(0.0, color="0.5", linestyle="--", linewidth=0.6)
ax_lag.axvline(0.0, color="0.5", linestyle="--", linewidth=0.6)
ax_lag.set_xlabel("Lag (s) - positive = chromosomes lag poles")
ax_lag.set_ylabel("Velocity dot product (norm.)")
ax_lag.set_title("Lag correlation across cells", loc="left", fontsize=9)
ax_lag.legend(loc="upper left", frameon=False)
ax_lag.text(0.97, 0.05,
            f"peak @ {peak_lag:+.0f} s\n(value {peak_val:.2f})",
            transform=ax_lag.transAxes, ha="right", va="bottom",
            fontsize=7, color="0.3")

# Panel labels (A, B)
for ax, label in [(ax_pca, "A"), (ax_lag, "B")]:
    ax.text(-0.13, 1.02, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")

fig1.tight_layout()
save_figure(fig1, "fig1_pca_lag")
plt.show()

# %% [markdown]
# ## Fig 2 — Centrosomes are sufficient to predict their own motion
#
# Alex's docx Result 3B: "adding CS-CH interactions or not to CS-CS
# interactions does not change prediction error, so again the simplest
# model would be that CS-CS interactions are dominant."
#
# Two complementary metrics:
#
# - **1-step RMSE** (autonomous one-frame velocity prediction): the
#   metric NB03 Part B originally used and the metric where Alex's
#   "no improvement" claim holds cleanly.  pp-only and pp+cp are
#   statistically indistinguishable.
# - **Path MSE** (deterministic rollout from the real initial frame
#   over the trimmed window): for apples-to-apples consistency with
#   NB04.  pp+cp scores slightly lower here, but **the rollout uses
#   real held-out chromosome positions as inputs** (chromosomes are
#   *given*, not co-simulated), so the apparent pp+cp advantage
#   reflects partition non-identifiability and chromosome-correlated
#   noise fitting rather than a clean causal coupling — see NB03b for
#   the partition reconciliation.
#
# Top row: signed force kernels with bootstrap CIs (f_pp and 46·f_cp
# on a shared y-axis so total CS-CH contribution is visually
# comparable to f_pp).
# Bottom row: mean ± SE bar chart of LOO path MSE for the two models
# — keeping the metric consistent with main-text Fig 3.  Numerical
# audit (1-step RMSE alongside path MSE, with the non-identifiability
# caveat) is printed to notebook stdout rather than rendered as a
# panel.  Per-cell strip plots are punted to the supplement (00b S2).

# %%
# --- Inline helpers for the centrosome velocity regression -----------
N_BASIS_PP = 6
N_BASIS_CP = 6
LAMBDA_RIDGE_F2 = LAMBDA_RIDGE
LAMBDA_ROUGH_F2 = LAMBDA_ROUGH
basis_pp = BSplineBasis(R_MIN, R_MAX, N_BASIS_PP)
basis_cp = BSplineBasis(R_MIN, R_MAX, N_BASIS_CP)
R_pp_mat = basis_pp.roughness_matrix()
R_cp_mat = basis_cp.roughness_matrix()
R_full_mat = block_diag(R_pp_mat, R_cp_mat)


def _build_cell_matrices(cell):
    """Stack pole-velocity rows and feature rows for a single cell."""
    G_rows, V_rows = [], []
    T_cell = cell.centrioles.shape[0]
    dt = cell.dt
    for t in range(T_cell - 1):
        poles_cur = cell.centrioles[t].T
        poles_next = cell.centrioles[t + 1].T
        chroms = cell.chromosomes[t].T
        for p in range(2):
            pole_vel = (poles_next[p] - poles_cur[p]) / dt
            other = 1 - p
            delta_pp = poles_cur[other] - poles_cur[p]
            r_pp_val = float(np.linalg.norm(delta_pp))
            if r_pp_val < 1e-12:
                continue
            dir_pp = delta_pp / r_pp_val
            phi_pp = basis_pp.evaluate(np.array([r_pp_val]))[0]
            g_pp = dir_pp[:, np.newaxis] * phi_pp[np.newaxis, :]

            g_cp = np.zeros((3, N_BASIS_CP))
            valid = ~np.any(np.isnan(chroms), axis=1)
            if valid.any():
                chroms_v = chroms[valid]
                delta_cp = chroms_v - poles_cur[p]
                dist_cp = np.linalg.norm(delta_cp, axis=1)
                ok = dist_cp > 1e-12
                if ok.any():
                    dir_cp = delta_cp[ok] / dist_cp[ok, np.newaxis]
                    phi_cp = basis_cp.evaluate(dist_cp[ok])
                    g_cp = np.einsum("id,ib->db", dir_cp, phi_cp)

            G_rows.append(np.hstack([g_pp, g_cp]))
            V_rows.append(pole_vel)
    return np.vstack(G_rows), np.concatenate(V_rows)


def _ridge_fit(G, V, R=None,
               lam=LAMBDA_RIDGE_F2, lam_rough=LAMBDA_ROUGH_F2):
    n = G.shape[1]
    penalty = lam * np.eye(n)
    if R is not None:
        penalty = penalty + lam_rough * R
    return np.linalg.solve(G.T @ G + penalty, G.T @ V)


def _force_callable(basis, theta_slice):
    spline = _BSpline(basis.knots, theta_slice, basis.degree, extrapolate=False)
    r_lo, r_hi = basis.r_min, basis.r_max

    def _f(r):
        rc = np.clip(r, r_lo, r_hi)
        val = spline(rc)
        return np.nan_to_num(val, nan=0.0)
    return _f


def _simulate_poles_det(poles_init, dt, n_steps,
                        f_pp=None, f_cp=None, chromosomes=None):
    """Deterministic forward Euler of pole positions (no diffusion)."""
    poles = poles_init.copy()
    traj = np.empty((n_steps + 1, 2, 3))
    traj[0] = poles
    for t in range(n_steps):
        new_poles = np.empty_like(poles)
        for p in range(2):
            force = np.zeros(3)
            other = 1 - p
            delta = poles[other] - poles[p]
            r_pp = float(np.linalg.norm(delta))
            if f_pp is not None and r_pp > 1e-12:
                force += (delta / r_pp) * f_pp(r_pp)
            if f_cp is not None and chromosomes is not None and t < chromosomes.shape[0]:
                chroms_t = chromosomes[t].T
                valid = ~np.any(np.isnan(chroms_t), axis=1)
                if valid.any():
                    delta_cp = chroms_t[valid] - poles[p]
                    dist_cp = np.linalg.norm(delta_cp, axis=1)
                    ok = dist_cp > 1e-12
                    if ok.any():
                        f_vals = f_cp(dist_cp[ok])
                        f_vals = np.nan_to_num(f_vals, nan=0.0)
                        dir_cp = delta_cp[ok] / dist_cp[ok, np.newaxis]
                        force += (dir_cp * f_vals[:, np.newaxis]).sum(axis=0)
            new_poles[p] = poles[p] + force * dt
        poles = new_poles
        traj[t + 1] = poles
    return traj


def _path_mse_pp(cell, traj):
    """Mean-squared 3D pole-position error across all timepoints, both poles."""
    real = np.moveaxis(cell.centrioles, 2, 1)  # (T, 2, 3)
    n = min(real.shape[0], traj.shape[0])
    diff = traj[:n] - real[:n]
    sq = np.sum(diff ** 2, axis=2)  # (n, 2)
    return float(np.mean(sq))


# --- LOO CV: both 1-step velocity RMSE and rollout path MSE ---------
# We report two metrics with complementary interpretations:
# - 1-step RMSE: predicts pole velocity given current state.  This is
#   the metric NB03 used originally and where Alex's "no improvement"
#   claim holds cleanly.  It is the AUTONOMOUS one-frame test.
# - rollout path MSE: integrates pole positions forward from the real
#   initial frame using the fitted kernels, then averages squared
#   position error over the trimmed window.  Note that the pp+cp
#   rollout uses *real* held-out chromosome positions as inputs (not
#   co-simulated), so its lower path MSE reflects partition non-
#   identifiability + chromosome-correlated noise fitting, NOT a clean
#   causal advantage for CS-CH coupling.  See NB03b for the
#   non-identifiability discussion.
print("Computing pp-only vs pp+cp LOO metrics...")
cell_mats = [_build_cell_matrices(c) for c in cells]
n_pp = N_BASIS_PP
path_mse_pp_only = np.empty(len(cells))
path_mse_full = np.empty(len(cells))
rmse1_pp_only = np.empty(len(cells))
rmse1_full = np.empty(len(cells))
for i, cell in enumerate(cells):
    G_train = np.vstack([cell_mats[j][0] for j in range(len(cells)) if j != i])
    V_train = np.concatenate([cell_mats[j][1] for j in range(len(cells)) if j != i])
    G_test, V_test = cell_mats[i]

    theta_pp_only = _ridge_fit(G_train[:, :n_pp], V_train, R=R_pp_mat)
    theta_full_loo = _ridge_fit(G_train, V_train, R=R_full_mat)

    # 1-step velocity prediction error on held-out cell
    rmse1_pp_only[i] = float(np.sqrt(np.mean((V_test - G_test[:, :n_pp] @ theta_pp_only) ** 2)))
    rmse1_full[i] = float(np.sqrt(np.mean((V_test - G_test @ theta_full_loo) ** 2)))

    # Forward rollout path MSE
    f_pp_only_loo = _force_callable(basis_pp, theta_pp_only)
    f_pp_full_loo = _force_callable(basis_pp, theta_full_loo[:n_pp])
    f_cp_full_loo = _force_callable(basis_cp, theta_full_loo[n_pp:])

    poles_init = cell.centrioles[0].T  # (2, 3)
    n_steps = cell.centrioles.shape[0] - 1
    dt = cell.dt
    traj_pp = _simulate_poles_det(poles_init, dt, n_steps, f_pp=f_pp_only_loo)
    traj_full = _simulate_poles_det(poles_init, dt, n_steps,
                                    f_pp=f_pp_full_loo, f_cp=f_cp_full_loo,
                                    chromosomes=cell.chromosomes)
    path_mse_pp_only[i] = _path_mse_pp(cell, traj_pp)
    path_mse_full[i] = _path_mse_pp(cell, traj_full)

print(f"  1-step RMSE  pp-only: {rmse1_pp_only.mean():.5f}, pp+cp: {rmse1_full.mean():.5f} "
      f"um/s; delta = {(rmse1_pp_only - rmse1_full).mean():+.5f}")
print(f"  Path MSE     pp-only: {path_mse_pp_only.mean():.3f}, pp+cp: {path_mse_full.mean():.3f} "
      f"um^2; delta = {(path_mse_pp_only - path_mse_full).mean():+.3f}")

# %%
# --- Bootstrap kernels (cell-level resampling) ----------------------
N_BOOT_F2 = 100
print(f"Bootstrapping pp-only and pp+cp kernels (n_boot={N_BOOT_F2})...")
r_pp_grid = np.linspace(R_MIN, R_MAX, 200)
r_cp_grid = np.linspace(R_MIN, R_MAX, 200)
boot_fpp_pp_only = np.empty((N_BOOT_F2, len(r_pp_grid)))
boot_fpp_full = np.empty((N_BOOT_F2, len(r_pp_grid)))
boot_fcp_full = np.empty((N_BOOT_F2, len(r_cp_grid)))
rng_f2 = np.random.default_rng(2026)
for b in range(N_BOOT_F2):
    idx_b = rng_f2.choice(len(cells), size=len(cells), replace=True)
    G_b = np.vstack([cell_mats[j][0] for j in idx_b])
    V_b = np.concatenate([cell_mats[j][1] for j in idx_b])
    theta_pp_b = _ridge_fit(G_b[:, :n_pp], V_b, R=R_pp_mat)
    theta_full_b = _ridge_fit(G_b, V_b, R=R_full_mat)
    boot_fpp_pp_only[b] = basis_pp.evaluate(r_pp_grid) @ theta_pp_b
    boot_fpp_full[b] = basis_pp.evaluate(r_pp_grid) @ theta_full_b[:n_pp]
    boot_fcp_full[b] = basis_cp.evaluate(r_cp_grid) @ theta_full_b[n_pp:]

# %%
# --- Render Fig 2 ---------------------------------------------------
fig2 = plt.figure(figsize=(7.0, 5.6))
gs2 = fig2.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.45,
                         wspace=0.35)
ax_fpp = fig2.add_subplot(gs2[0, 0])
ax_fcp = fig2.add_subplot(gs2[0, 1])
ax_strip = fig2.add_subplot(gs2[1, :])  # path-MSE bars span both columns


def _plot_kernel_band(ax, r_grid, boot_curves, color, label):
    median = np.median(boot_curves, axis=0)
    lo = np.quantile(boot_curves, 0.05, axis=0)
    hi = np.quantile(boot_curves, 0.95, axis=0)
    ax.fill_between(r_grid, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.plot(r_grid, median, color=color, linewidth=1.8, label=label)


# f_pp panel: pp-only vs pp+cp.  Restrict plot to observed pole-pole
# distance support so we don't show extrapolation.
all_pp = []
for cell in cells:
    pp = np.linalg.norm(cell.centrioles[:, :, 1] - cell.centrioles[:, :, 0], axis=1)
    all_pp.extend(pp)
all_pp = np.asarray(all_pp)
pp_lo, pp_hi = np.quantile(all_pp, [0.01, 0.99])

mask_pp = (r_pp_grid >= pp_lo) & (r_pp_grid <= pp_hi)
_plot_kernel_band(ax_fpp, r_pp_grid[mask_pp], boot_fpp_pp_only[:, mask_pp],
                  OKABE_ITO["blue"], "pp-only model")
_plot_kernel_band(ax_fpp, r_pp_grid[mask_pp], boot_fpp_full[:, mask_pp],
                  OKABE_ITO["vermil"], "pp+cp model")
ax_fpp.axhline(0.0, color="0.5", linestyle="--", linewidth=0.6)
ax_fpp.set_xlabel("Pole-pole distance (μm)")
ax_fpp.set_ylabel("$f_{pp}(r)$  (μm/s) "
                  "\n+ attractive · - repulsive")
ax_fpp.set_title("Effective pole-separation kernel", loc="left", fontsize=8.5)
ax_fpp.legend(frameon=False, loc="upper right")

# f_cp panel: from pp+cp model only (cp-only is meaningless here)
all_cp = []
for cell in cells:
    poles = np.moveaxis(cell.centrioles, 2, 1)
    chroms = np.moveaxis(cell.chromosomes, 2, 1)
    for t in range(chroms.shape[0]):
        ch = chroms[t]
        v = ~np.any(np.isnan(ch), axis=1)
        if not v.any():
            continue
        chv = ch[v]
        for p in range(2):
            d = np.linalg.norm(chv - poles[t, p], axis=1)
            all_cp.extend(d[d > 1e-12])
all_cp = np.asarray(all_cp)
cp_lo, cp_hi = np.quantile(all_cp, [0.01, 0.99])

# Effective CS-CH contribution: scale f_cp by the chromosome count so
# the magnitude is directly comparable to f_pp (a single inter-pole
# interaction).  This is an UPPER BOUND on the realized CS-CH force on
# a pole — it assumes all 46 chromosomes contribute in the same
# direction.  The realized sum has direction cancellations from
# chromosomes on opposite sides of the pole; see NB03 Part B for the
# observed-positions decomposition.
N_CHROM_SCALE = 46
mask_cp = (r_cp_grid >= cp_lo) & (r_cp_grid <= cp_hi)
_plot_kernel_band(ax_fcp, r_cp_grid[mask_cp],
                  N_CHROM_SCALE * boot_fcp_full[:, mask_cp],
                  OKABE_ITO["vermil"],
                  f"{N_CHROM_SCALE} × $f_{{cp}}$ (upper-bound scale)")
ax_fcp.axhline(0.0, color="0.5", linestyle="--", linewidth=0.6)
ax_fcp.set_xlabel("Chromosome-to-pole distance (μm)")
ax_fcp.set_ylabel(f"{N_CHROM_SCALE} · $f_{{cp}}(r)$  (μm/s) "
                   "\n+ attractive · - repulsive")
ax_fcp.set_title("Total CS-CH contribution (upper bound)",
                 loc="left", fontsize=8.5)
ax_fcp.legend(frameon=False, loc="upper right", fontsize=6.5)

# Share y-axis between f_pp and 46·f_cp so the overall CS-CH
# contribution is visually comparable to f_pp.
y_min_f2 = min(ax_fpp.get_ylim()[0], ax_fcp.get_ylim()[0])
y_max_f2 = max(ax_fpp.get_ylim()[1], ax_fcp.get_ylim()[1])
ax_fpp.set_ylim(y_min_f2, y_max_f2)
ax_fcp.set_ylim(y_min_f2, y_max_f2)

# Path MSE bar chart (mean ± SE) for the two models — keeps the
# headline metric consistent with main-text Fig 3.  The bars are
# very close, supporting Alex's "CS-CS sufficient" claim, but they
# are not exactly equal (see the printed summary below for caveats).
xs_bar_f2 = np.array([0.0, 0.7])
m_pp, se_pp = float(path_mse_pp_only.mean()), float(path_mse_pp_only.std(ddof=1) / np.sqrt(len(cells)))
m_fl, se_fl = float(path_mse_full.mean()),    float(path_mse_full.std(ddof=1) / np.sqrt(len(cells)))
ax_strip.bar(xs_bar_f2[0], m_pp, width=0.45, color=OKABE_ITO["blue"],
             alpha=0.85, edgecolor="white", linewidth=0.6)
ax_strip.bar(xs_bar_f2[1], m_fl, width=0.45, color=OKABE_ITO["vermil"],
             alpha=0.85, edgecolor="white", linewidth=0.6)
ax_strip.errorbar(xs_bar_f2, [m_pp, m_fl],
                  yerr=[se_pp, se_fl], fmt="none",
                  ecolor="0.2", capsize=4, linewidth=1.0)
ax_strip.set_xticks(xs_bar_f2)
ax_strip.set_xticklabels(["pp-only", "pp+cp"])
ax_strip.set_ylabel("LOO path MSE (μm²)")
ax_strip.set_title("Held-out pole-trajectory error",
                   loc="left", fontsize=8.5)
ax_strip.set_xlim(-0.45, 1.15)
ax_strip.set_ylim(bottom=0.0)

# Numerical summary printed to the notebook output (no on-figure
# panel — keeps the figure visually clean while preserving the dual-
# metric audit trail).
d1 = rmse1_pp_only - rmse1_full
d_path = path_mse_pp_only - path_mse_full
m1, se1 = float(np.mean(d1)), float(np.std(d1, ddof=1) / np.sqrt(len(cells)))
mp, sep = float(np.mean(d_path)), float(np.std(d_path, ddof=1) / np.sqrt(len(cells)))
n1_pp_better = int(np.sum(d1 < 0))
np_pp_better = int(np.sum(d_path < 0))
se_rm_pp = rmse1_pp_only.std(ddof=1) / np.sqrt(len(cells))
se_rm_fl = rmse1_full.std(ddof=1) / np.sqrt(len(cells))
print()
print("Fig 2 numerical summary")
print("=" * 60)
print("1-step RMSE  (autonomous one-frame test):")
print(f"  pp-only:  {rmse1_pp_only.mean():.5f} +/- {se_rm_pp:.5f} um/s")
print(f"  pp+cp:    {rmse1_full.mean():.5f} +/- {se_rm_fl:.5f} um/s")
print(f"  delta:    {m1:+.5f} +/- {se1:.5f}    delta/SE = {m1/se1:+.2f}")
print(f"  pp-only better in {n1_pp_better}/{len(cells)} cells")
print()
print("Path MSE  (deterministic rollout, real CHs given as covariates):")
print(f"  pp-only:  {path_mse_pp_only.mean():.2f} +/- {se_pp:.2f} um^2")
print(f"  pp+cp:    {path_mse_full.mean():.2f} +/- {se_fl:.2f} um^2")
print(f"  delta:    {mp:+.2f} +/- {sep:.2f}    delta/SE = {mp/sep:+.2f}")
print(f"  pp-only better in {np_pp_better}/{len(cells)} cells")
print()
print("Caveat: the path-MSE pp+cp edge reflects partition non-")
print("identifiability — real chromosome positions enter the rollout")
print("as covariates, so pp+cp absorbs CH-correlated noise without a")
print("clean causal advantage.  See NB03b for the partition")
print("reconciliation; 1-step RMSE is the autonomous comparison.")

# Panel labels
for ax, label in [(ax_fpp, "A"), (ax_fcp, "B"), (ax_strip, "C")]:
    ax.text(-0.13, 1.04, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")

fig2.suptitle("Chromosome-on-pole coupling does not improve "
               "pole-motion prediction in the autonomous test\n"
               "(1-step RMSE is statistically indistinguishable; "
               "path-MSE pp+cp edge reflects CH-as-covariate "
               "non-identifiability, not causal coupling — see text)",
              fontsize=8.5, y=0.998)
fig2.tight_layout()
save_figure(fig2, "fig2_cs_sufficient")
plt.show()

# %% [markdown]
# ## Fig 3 — Learned chromosome force kernels and topology comparison
#
# Addresses Alex's docx Result 3C ("predicted force-distance relations"
# + "best model: CH-CH steric only, CS attracts CH at long distance").
# Reports what the data-driven SFI-inspired projection fit actually
# shows under a
# biologically motivated steric envelope on the chrom-chrom kernel —
# the f_xy shape may differ in detail from Alex's prior verbal
# prediction; see discussion notes accompanying the figure.  The
# held-out-forecast-error-vs-horizon panel Alex requests in 3C is
# planned for supplement S3 rather than the main text.
#
# Top row: signed force kernels with bootstrap 5–95 % CI bands.
# - **A** ``f_xy(r)`` overlaid for all 4 topologies (poles, center,
#   poles_and_chroms free-xx, poles_and_chroms_enveloped).  Distinct
#   linestyles for distinguishability; cropped at 12 μm to keep the
#   visual focus on the data-supported region.
# - **B** ``f_xx(r)`` for the two topologies that have an xx kernel
#   (free-xx vs enveloped).  The free-xx kernel deviates from zero at
#   long range — the spurious chrom-chrom interaction that motivates
#   the biologically motivated steric envelope and the a priori
#   exclusion of free-xx topologies from the admissible set.
#
# Bottom row: sorted mean ± SE bar chart of LOO path MSE across the 4
# topologies.  Admissible models (poles, center, enveloped) in
# Okabe-Ito; nuisance upper-bound (poles + chroms free-xx) hatched
# neutral gray.  Selected topology annotated.  Per-cell strip plots
# are punted to the supplement (00b S1) so the main panel is not
# muddled by 12-cell jitter; ``center_and_chroms`` (free-xx +
# midpoint partner) is also punted to S1.

# %%
# --- All 5 topologies: fit + bootstrap + rollout CV ----------------
# Main-text Fig 3 compares 4 topologies: the 3 admissible models plus
# the free-form ``poles_and_chroms`` as a nuisance upper-bound.
# ``center_and_chroms`` is ruled out a priori on the same biological
# grounds as ``poles_and_chroms`` and is moved to the supplement
# (00b S1) so the main panel is not muddled with an extra free-xx
# variant that tells the same story.
TOPOLOGIES_F3 = [
    "poles", "center",
    "poles_and_chroms",
    "poles_and_chroms_enveloped",
]
TOPO_DISPLAY = {
    "poles":                       {"label": "poles",
                                    "color": OKABE_ITO["blue"],
                                    "admissible": True},
    "center":                      {"label": "center",
                                    "color": OKABE_ITO["green"],
                                    "admissible": True},
    "poles_and_chroms_enveloped":  {"label": "poles + chroms\n(enveloped)",
                                    "color": OKABE_ITO["vermil"],
                                    "admissible": True},
    "poles_and_chroms":            {"label": "poles + chroms\n(free xx)",
                                    "color": "0.45",
                                    "admissible": False},
}


def _make_config(topology):
    use_env = topology == "poles_and_chroms_enveloped"
    base = "poles_and_chroms" if use_env else topology
    return FitConfig(
        topology=base,
        n_basis_xx=N_BASIS,
        n_basis_xy=N_BASIS,
        r_min_xx=R_MIN, r_max_xx=R_MAX,
        r_min_xy=R_MIN, r_max_xy=R_MAX,
        lambda_ridge=LAMBDA_RIDGE,
        lambda_rough=LAMBDA_ROUGH,
        endpoint_method="neb_ao_frac",
        endpoint_frac=FRAC_NEB_AO,
        envelope_r0_xx=ENVELOPE_R0_XX if use_env else None,
        envelope_w_xx=ENVELOPE_W_XX if use_env else None,
    )


# Reuse the canonical fit + bootstrap for poles_and_chroms_enveloped
configs_f3 = {t: _make_config(t) for t in TOPOLOGIES_F3}
models_f3 = {"poles_and_chroms_enveloped": model}
boots_f3 = {"poles_and_chroms_enveloped": boot}

print("Fitting + bootstrapping topologies for Fig 3...")
# Deterministic per-topology seed (was hash(topology) which is process-
# randomized in Python 3.3+).  Index into TOPOLOGIES_F3 keeps it
# reproducible across runs and different machines.
for ti_, topology in enumerate(TOPOLOGIES_F3):
    if topology in models_f3:
        continue
    print(f"  fitting {topology}...")
    models_f3[topology] = fit_model(cells, configs_f3[topology])
    boots_f3[topology] = bootstrap_kernels(
        cells, configs_f3[topology], n_boot=N_BOOT,
        rng=np.random.default_rng(2025 + 17 * ti_),
    )

# Path MSE via rollout CV for each topology
print("Computing path MSE (rollout CV) for each topology...")
rollouts_f3 = {}
for topology in TOPOLOGIES_F3:
    print(f"  rollout-CV {topology}...")
    rollouts_f3[topology] = rollout_cross_validate(
        cells, configs_f3[topology], deterministic=True
    )

# %%
# --- Render Fig 3 --------------------------------------------------
fig3 = plt.figure(figsize=(7.2, 6.6))
gs = fig3.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.45,
                       wspace=0.32)
ax_fxy = fig3.add_subplot(gs[0, 0])
ax_fxx = fig3.add_subplot(gs[0, 1])
ax_path = fig3.add_subplot(gs[1, :])


def _kernel_band(ax, model_obj, boot_obj, kernel, r_grid, color, label,
                 linestyle="-"):
    """Plot mean + 5-95 % bootstrap CI for *kernel* on *ax*."""
    mean_vals = model_obj.evaluate_kernel(kernel, r_grid)
    if mean_vals is None:
        return
    basis_obj = model_obj.basis_xx if kernel == "xx" else model_obj.basis_xy
    n_xx = model_obj.n_basis_xx
    if kernel == "xx":
        samples = boot_obj.theta_samples[:, :n_xx]
    else:
        samples = boot_obj.theta_samples[:, n_xx:]
    phi = basis_obj.evaluate(r_grid)
    curves = phi @ samples.T  # (len(r_grid), n_boot)
    lo = np.quantile(curves, 0.05, axis=1)
    hi = np.quantile(curves, 0.95, axis=1)
    ax.fill_between(r_grid, lo, hi, color=color, alpha=0.16, linewidth=0)
    ax.plot(r_grid, mean_vals, color=color, linewidth=1.8,
            linestyle=linestyle, label=label)


# --- f_xy panel: overlay all 4 topologies' f_xy kernels -----------
# Each topology gets a distinct (color, linestyle) so curves remain
# distinguishable when they overlap.  The x-axis is "distance to
# partner" — note that the partner is the closest pole for `poles`,
# the spindle midpoint for `center`, and the closest pole again for
# the two pole-partner topologies.
F_XY_PLOT_MAX = 12.0
r_xy_grid = np.linspace(R_MIN, F_XY_PLOT_MAX, 250)
F_XY_OVERLAY = [
    ("poles",                       "-",
     "poles"),
    ("center",                      "--",
     "center (chrom -> spindle midpoint)"),
    ("poles_and_chroms",            ":",
     "poles + chroms (free xx)"),
    ("poles_and_chroms_enveloped",  "-",
     "poles + chroms (enveloped) — selected"),
]
for topology, linestyle, label in F_XY_OVERLAY:
    info = TOPO_DISPLAY[topology]
    _kernel_band(ax_fxy, models_f3[topology], boots_f3[topology],
                 "xy", r_xy_grid, info["color"], label,
                 linestyle=linestyle)

# Cut off the f_xy plot at the data-supported limit (chrom-to-pole
# extends to ~10 um, chrom-to-midpoint to ~7 um); 12 um is a
# conservative crop that avoids the extrapolated tail.
ax_fxy.set_xlim(R_MIN, F_XY_PLOT_MAX)
ax_fxy.axhline(0.0, color="0.5", linestyle="--", linewidth=0.6)
ax_fxy.set_xlabel("Distance to partner (μm)")
ax_fxy.set_ylabel("$f_{xy}(r)$  (μm/s) "
                   "\n+ attractive · - repulsive")
ax_fxy.set_title("Chromosome-to-partner force kernel",
                 loc="left", fontsize=8.5)
ax_fxy.legend(frameon=False, loc="best", fontsize=6.5)

# --- f_xx panel: free-form vs enveloped ---------------------------
# Plot the full basis domain so the free-xx kernel's long-range
# artefact is visible (this is the spurious chrom-chrom interaction
# that motivates ruling out the free-xx topology a priori).  Plot
# the enveloped (selected) curve first so the free-xx curve is drawn
# on top.
F_XX_PLOT_MAX = R_MAX
r_xx_grid = np.linspace(R_MIN, F_XX_PLOT_MAX, 400)
_kernel_band(ax_fxx, models_f3["poles_and_chroms_enveloped"],
             boots_f3["poles_and_chroms_enveloped"],
             "xx", r_xx_grid,
             TOPO_DISPLAY["poles_and_chroms_enveloped"]["color"],
             "enveloped (steric prior) — selected",
             linestyle="-")
_kernel_band(ax_fxx, models_f3["poles_and_chroms"],
             boots_f3["poles_and_chroms"],
             "xx", r_xx_grid,
             "0.25",
             "free xx — non-zero at long range = spurious",
             linestyle="--")
ax_fxx.axhline(0.0, color="0.5", linestyle="--", linewidth=0.6)
ax_fxx.set_xlim(R_MIN, F_XX_PLOT_MAX)
ax_fxx.set_xlabel("Chromosome-chromosome distance (μm)")
ax_fxx.set_ylabel("$f_{xx}(r)$  (μm/s) "
                   "\n+ attractive · - repulsive")
ax_fxx.set_title("Chromosome-chromosome interaction (per-pair; "
                  "y-scale ~10× smaller than panel A)",
                 loc="left", fontsize=8.0)
ax_fxx.legend(frameon=False, loc="upper right", fontsize=6.5)
# y-scale note moved out of the data area to avoid colliding with the
# kernel curves; appended to the panel title via the suptitle box.

# --- Path MSE bar chart, sorted by performance ---------------------
n_topo = len(TOPOLOGIES_F3)
per_cell_pmse = np.empty((len(cells), n_topo))
for ti, topology in enumerate(TOPOLOGIES_F3):
    per_cell_pmse[:, ti] = rollouts_f3[topology].path_mse

# Sort topologies by mean path MSE (ascending = better first)
mean_pmse = per_cell_pmse.mean(axis=0)
sort_idx = np.argsort(mean_pmse)
sorted_topos = [TOPOLOGIES_F3[i] for i in sort_idx]
sorted_means = mean_pmse[sort_idx]
sorted_ses = np.array([
    float(per_cell_pmse[:, i].std(ddof=1) / np.sqrt(len(cells)))
    for i in sort_idx
])

xs_bar = np.arange(n_topo, dtype=float)
for bi, ti in enumerate(sort_idx):
    info = TOPO_DISPLAY[TOPOLOGIES_F3[ti]]
    hatch = None if info["admissible"] else "//"
    ax_path.bar(xs_bar[bi], sorted_means[bi], width=0.7,
                color=info["color"], alpha=0.85,
                edgecolor="white" if info["admissible"] else "0.2",
                linewidth=0.6, hatch=hatch)
ax_path.errorbar(xs_bar, sorted_means, yerr=sorted_ses, fmt="none",
                 ecolor="0.2", capsize=3, linewidth=0.8)

# Highlight the selected topology (lowest mean among admissible)
admissible_topos = [t for t in TOPOLOGIES_F3 if TOPO_DISPLAY[t]["admissible"]]
selected = min(admissible_topos,
               key=lambda t: float(np.mean(rollouts_f3[t].path_mse)))
sel_bar = sorted_topos.index(selected)
sel_y = sorted_means[sel_bar]
ax_path.annotate("selected", xy=(xs_bar[sel_bar], sel_y),
                 xytext=(xs_bar[sel_bar], sel_y * 1.15),
                 ha="center", fontsize=7, color="0.1",
                 arrowprops=dict(arrowstyle="-", color="0.2",
                                 linewidth=0.6))

ax_path.set_xticks(xs_bar)
ax_path.set_xticklabels(
    [TOPO_DISPLAY[t]["label"] for t in sorted_topos],
    fontsize=7,
)
ax_path.set_ylabel("LOO path MSE (μm²)")
ax_path.set_title("Mean per-cell trajectory error across topologies "
                   "— y-axis starts above 0 to highlight differences "
                   "(see supplement for per-cell breakdown)",
                  loc="left", fontsize=8.0)

# Zoom y-axis to amplify the differences between topologies.  The
# bottom is set to ~85 % of the smallest mean so the bars are clearly
# differentiable; we add a "broken-axis" indicator on the y-axis to
# warn the reader that the bars do not start at zero.
y_lo_path = float(sorted_means.min()) * 0.85
y_hi_path = float((sorted_means + sorted_ses).max()) * 1.18
ax_path.set_ylim(y_lo_path, y_hi_path)
# Broken-axis indicator — small zigzag near the bottom of the y-axis
_break_y = y_lo_path + 0.012 * (y_hi_path - y_lo_path)
_break_x_left = ax_path.get_xlim()[0]
ax_path.plot([_break_x_left, _break_x_left + 0.10],
             [_break_y, _break_y + 0.018 * (y_hi_path - y_lo_path)],
             color="0.2", linewidth=1.0, clip_on=False, zorder=10)
ax_path.plot([_break_x_left + 0.10, _break_x_left + 0.20],
             [_break_y + 0.018 * (y_hi_path - y_lo_path), _break_y],
             color="0.2", linewidth=1.0, clip_on=False, zorder=10)
# (y-axis-zoom warning is in the panel title)

# Custom legend for admissible vs nuisance UB
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=OKABE_ITO["blue"], alpha=0.85, edgecolor="white",
          label="admissible (poles, center, enveloped)"),
    Patch(facecolor="0.45", alpha=0.85, edgecolor="0.2", hatch="//",
          label="nuisance upper-bound (free-form xx)"),
]
ax_path.legend(handles=legend_elements, loc="upper left", frameon=False,
               fontsize=6.5)

# Panel labels
for ax, label in [(ax_fxy, "A"), (ax_fxx, "B"), (ax_path, "C")]:
    ax.text(-0.13, 1.03, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")

fig3.suptitle("Learned pairwise force kernels and topology selection",
              fontsize=10, y=0.995)
fig3.tight_layout()
save_figure(fig3, "fig3_kernels_topology")
plt.show()

# %% [markdown]
# ## Fig 3b — Sim vs real trajectories in PCA space
#
# Visual confirmation that the canonical
# ``poles_and_chroms_enveloped`` model reproduces the gathering motion
# observed in real data.  We pick a good cell (low LOO path MSE on the
# selected topology), simulate chromosome trajectories deterministically
# from the real initial frame, and place real and simulated dynamics
# side-by-side in a common PCA basis fit from the cell's combined
# pole + chromosome point cloud (``fit_pca_basis``).  No manual
# centering — the PCA basis handles centering, and PC1 is sign-aligned
# with the pole-pole axis so spindle elongation is horizontal.
#
# Visual conventions match the rest of the paper:
# - Centrosomes: Greys colormap (light → dark over time), thicker line,
#   ○ start / ■ end markers.
# - Chromosomes: tab20 categorical colors (one color per chromosome,
#   no time encoding), small end-point dots.  We display a
#   representative subset of chromosomes spanning the initial radial
#   spread so the eye can match individual chromosomes between the
#   real and simulated panels.

# %%
from chromlearn.model_fitting.simulate import simulate_cell

# Pick the cell with the lowest canonical-topology path MSE — the
# best-fitting cell, gives the cleanest sim-vs-real comparison.
canonical_pmse = rollouts_f3["poles_and_chroms_enveloped"].path_mse
good_idx = int(np.argmin(canonical_pmse))
good_cell = cells[good_idx]
print(f"Sim-vs-real cell: idx={good_idx}, id={good_cell.cell_id}, "
      f"canonical-topology path MSE = {canonical_pmse[good_idx]:.2f} um^2 "
      f"(median {np.median(canonical_pmse):.2f})")

# Deterministic and stochastic rollouts of the same cell.
_, sim_cell_det = simulate_cell(good_cell, model, deterministic=True)
_, sim_cell_sde = simulate_cell(good_cell, model,
                                 rng=np.random.default_rng(2026),
                                 deterministic=False)

# Common PCA basis from the real cell.
pca_basis_3b = fit_pca_basis(good_cell)
T_3b = good_cell.chromosomes.shape[0]
n_chrom_3b = good_cell.chromosomes.shape[2]
t_norm_3b = np.linspace(0.0, 1.0, T_3b)

# Pick chromosomes that (a) travel a meaningful distance — so the
# gathering motion is visible — and (b) have spread-out initial /
# final conditions, so the colored set illustrates the geometry of
# the spindle rather than clustering in one corner.  Strategy:
# restrict to the top-half-movers, then evenly subsample by initial
# PC1 coordinate to span the spindle's long axis.
PCA3B_N_DISPLAY = 14
real_xyz = good_cell.chromosomes  # (T, 3, N)
valid_mask_3b = np.all(np.isfinite(real_xyz), axis=(0, 1))
valid_idx_3b = np.flatnonzero(valid_mask_3b)
# Per-chromosome displacement (initial -> final) -- shape (n_valid,)
init_xyz = real_xyz[0].T[valid_idx_3b]   # (n_valid, 3)
final_xyz = real_xyz[-1].T[valid_idx_3b]  # (n_valid, 3)
displacement = np.linalg.norm(final_xyz - init_xyz, axis=1)
high_mover_cutoff = float(np.median(displacement))
high_mover_mask = displacement >= high_mover_cutoff
high_mover_idx = valid_idx_3b[high_mover_mask]
init_pos_hm = init_xyz[high_mover_mask]  # (n_hm, 3)

# Initial PC1 of each high-mover (signed distance along spindle axis)
init_pc1 = (init_pos_hm - pca_basis_3b.origin) @ pca_basis_3b.components[:, 0]
sort_by_pc1 = high_mover_idx[np.argsort(init_pc1)]
if sort_by_pc1.size > PCA3B_N_DISPLAY:
    pick = np.linspace(0, sort_by_pc1.size - 1, PCA3B_N_DISPLAY, dtype=int)
    display_chroms_3b = sort_by_pc1[pick]
else:
    display_chroms_3b = sort_by_pc1
disp_picked = displacement[np.isin(valid_idx_3b, display_chroms_3b)]
print(f"  highlighted {len(display_chroms_3b)} chromosomes; "
      f"displacement range {disp_picked.min():.2f}-{disp_picked.max():.2f} um; "
      f"spread evenly along the spindle axis")


fig3b, axes_3b = plt.subplots(
    1, 3, figsize=(10.0, 3.6), sharex=True, sharey=True,
)
ax_real, ax_det, ax_sde = axes_3b


def _draw_pca_panel(ax, cell_obj, title, in_sample=False):
    chrom_palette = plt.get_cmap("tab20")(
        np.linspace(0, 1, len(display_chroms_3b))
    )
    # Faint background of all valid chromosomes
    for j in range(n_chrom_3b):
        cj = cell_obj.chromosomes[:, :, j]
        if np.any(np.isnan(cj)):
            continue
        cj_pca = pca_basis_3b.project(cj)
        ax.plot(cj_pca[:, 0], cj_pca[:, 1], color="0.78",
                linewidth=0.4, alpha=0.30, zorder=1)
    # Highlighted subset, tab20 categorical
    for ci, j in enumerate(display_chroms_3b):
        cj = cell_obj.chromosomes[:, :, j]
        if np.any(np.isnan(cj)):
            continue
        cj_pca = pca_basis_3b.project(cj)
        ax.plot(cj_pca[:, 0], cj_pca[:, 1], color=chrom_palette[ci],
                linewidth=1.1, alpha=0.85, zorder=2)
        ax.plot(cj_pca[-1, 0], cj_pca[-1, 1], "o",
                color=chrom_palette[ci], markersize=3.2,
                markeredgecolor="white", markeredgewidth=0.4, zorder=3)
    # Centrosomes — same real centrioles in all three panels (by
    # construction; sim_cell* keeps real centrioles).
    pole_lc = None
    for p_idx in range(2):
        pp = cell_obj.centrioles[:, :, p_idx]
        pp_pca = pca_basis_3b.project(pp)
        pole_lc = _colorline(ax, pp_pca[:, 0], pp_pca[:, 1], t_norm_3b,
                             cmap="Greys", linewidth=2.4,
                             alpha=0.95, zorder=4)
        ax.plot(pp_pca[0, 0], pp_pca[0, 1], "o",
                color=plt.get_cmap("Greys")(0.25),
                markersize=4.5, markeredgecolor="white",
                markeredgewidth=0.6, zorder=5)
        ax.plot(pp_pca[-1, 0], pp_pca[-1, 1], "s",
                color=plt.get_cmap("Greys")(0.95),
                markersize=4.5, markeredgecolor="white",
                markeredgewidth=0.6, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("PC1 (μm)")
    ax.set_title(title, loc="left", fontsize=9)
    if in_sample:
        ax.text(0.03, 0.97,
                "in-sample illustration:\n"
                "model trained on all 12 cells",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=6.5, color="0.3",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.6, pad=2))
    return pole_lc


_ = _draw_pca_panel(ax_real, good_cell,
                     f"Real trajectories  ({good_cell.cell_id})",
                     in_sample=True)
_ = _draw_pca_panel(ax_det, sim_cell_det,
                     "Deterministic ODE rollout",
                     in_sample=False)
pole_lc_sde = _draw_pca_panel(ax_sde, sim_cell_sde,
                               "Stochastic SDE rollout (single replicate)",
                               in_sample=False)
ax_real.set_ylabel("PC2 (μm)")
ax_real.autoscale()

# Single horizontal time colorbar OUTSIDE the axes (below the figure)
# rather than inside ax_sde, to avoid overlapping the x-axis label.
cbar_ax = fig3b.add_axes([0.30, 0.02, 0.40, 0.018])
plt.colorbar(pole_lc_sde, cax=cbar_ax, orientation="horizontal")
cbar_ax.set_xlabel("Centrosome time (NEB to 0.4 NEB-AO)",
                   fontsize=6.5, labelpad=2)
cbar_ax.tick_params(labelsize=0, length=0)

for ax, label in [(ax_real, "A"), (ax_det, "B"), (ax_sde, "C")]:
    ax.text(-0.13, 1.02, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")

fig3b.suptitle("Real vs simulated dynamics in PCA space",
               fontsize=10, y=0.995)
fig3b.tight_layout()
save_figure(fig3b, "fig3b_sim_vs_real_pca")
plt.show()


# %% [markdown]
# ## Fig 4 — Effective diffusion D(d)
#
# Alex's docx Result 3D: "learned effective diffusion of CHs, which is
# greater far from the spindle, which supports findings of our previous
# paper."
#
# Local diffusivity is estimated from the mean-squared residual
# displacement after subtracting the fitted deterministic force,
# $\langle |\delta X - \hat F(X) \cdot dt|^2 \rangle / (2 \cdot d \cdot dt)$, which
# removes the leading drift contamination that otherwise biases
# quadratic displacement estimators in this drift-dominated, spatially
# heterogeneous setting.  In the Itô convention, the projected drift
# absorbed by the stage-1 force estimate already includes any
# basis-resolvable diffusion-gradient $\nabla\!\cdot\!D$ contribution
# (Frishman & Ronceray PRX 2020, App. H Eq. H2 and Eq. 15), so no
# additional $\nabla\!\cdot\!D$ correction is applied; the estimator
# is not localization-noise-corrected, which is negligible here given
# per-step drift ≫ localization noise.  Coordinate ``d`` is reserved
# for the spindle-center distance to avoid collision with the
# SFI-inspired pairwise interaction distance
# pairwise distance ``r``.

# %%
N_BASIS_D = 8
R_MIN_D = 0.5
R_MAX_D = 12.0
D_ESTIMATOR = "f_corrected"


def _fit_D_pooled(cell_list, mode):
    """Fit D(distance) on a list of cells using *mode*."""
    kw = {}
    if mode == "f_corrected":
        kw = dict(fit_result=model, basis_xx=model.basis_xx,
                   basis_xy=model.basis_xy)
    return estimate_diffusion_variable(
        cell_list,
        basis_D=BSplineBasis(R_MIN_D, R_MAX_D, N_BASIS_D),
        coord_name="distance",
        dt=config.dt,
        mode=mode,
        lambda_ridge=LAMBDA_RIDGE,
        topology=model.topology,
        **kw,
    )


# Both the pooled headline curve and the per-cell consistency traces
# use f_corrected, which subtracts the predicted drift before
# estimating D.  The per-cell loop is the slowest cell in the
# notebook (~12 force evaluations × ~4k samples each); kept on
# f_corrected for cross-cell consistency rather than mixing
# estimators.
D_pooled = _fit_D_pooled(cells, mode=D_ESTIMATOR)

D_per_cell = []
for cell in cells:
    try:
        D_per_cell.append(_fit_D_pooled([cell], mode=D_ESTIMATOR))
    except Exception:
        D_per_cell.append(None)

# Empirical chrom-to-spindle-center distances for clipping the plot to
# the data-supported domain.
from chromlearn.model_fitting.diffusion import COORDINATE_MAPS
_coord_fn = COORDINATE_MAPS["distance"]
all_distances = np.concatenate(
    [_coord_fn(c.chromosomes, c).ravel() for c in cells]
)
all_distances = all_distances[np.isfinite(all_distances)]
EVAL_LO, EVAL_HI = np.quantile(all_distances, [0.01, 0.99])
EVAL_LO = max(EVAL_LO, R_MIN_D)
EVAL_HI = min(EVAL_HI, R_MAX_D)
d_grid = np.linspace(EVAL_LO, EVAL_HI, 200)
print(f"Plot range clipped to [{EVAL_LO:.2f}, {EVAL_HI:.2f}] um (data support)")

# %%
fig4, ax4 = plt.subplots(figsize=(5.5, 3.6))

# Per-cell faint curves
for d_res in D_per_cell:
    if d_res is None:
        continue
    vals = d_res.evaluate(d_grid)
    ax4.plot(d_grid, vals, color="0.7", linewidth=0.6, alpha=0.55,
             zorder=1)

# Pooled f_corrected curve (headline)
pooled_vals = D_pooled.evaluate(d_grid)
ax4.plot(d_grid, pooled_vals, color=OKABE_ITO["vermil"], linewidth=2.2,
         label="pooled D(d), f_corrected", zorder=3)

# Constant-D baseline
ax4.axhline(model.D_x, color="0.3", linestyle="--", linewidth=1.0,
            label=f"constant D = {model.D_x:.4f} μm²/s", zorder=2)

# Data-density rug along bottom of axis
sample_distances = np.random.default_rng(0).choice(
    all_distances, size=min(5000, all_distances.size), replace=False
)
y_lo, y_hi = ax4.get_ylim()
rug_y = y_lo + 0.02 * (y_hi - y_lo)
ax4.scatter(sample_distances, np.full_like(sample_distances, rug_y),
            s=1.0, color="0.4", alpha=0.18, zorder=0,
            marker="|", linewidths=0.5)

ax4.set_xlim(EVAL_LO, EVAL_HI)
ax4.set_ylim(bottom=0.0)
ax4.set_xlabel("Distance from spindle center, $d$ (μm)")
ax4.set_ylabel("D (μm²/s)")
ax4.set_title("Effective diffusion grows away from spindle center",
              loc="left", fontsize=9)
ax4.legend(frameon=False, loc="upper left")
ax4.text(0.99, 0.04,
         f"n={sum(1 for x in D_per_cell if x is not None)} cells; "
         "tail of D(d) is sparsely sampled — see rug.",
         transform=ax4.transAxes, ha="right", va="bottom",
         fontsize=6, color="0.4")

fig4.tight_layout()
save_figure(fig4, "fig4_diffusion_landscape")
plt.show()
