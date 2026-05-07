# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 00. Main-text figures
#
# Paper-figure assembler for the chromlearn project. Each section
# below produces one main-text figure (PDF + 600 dpi PNG) for the
# early-prometaphase spindle-force-inference manuscript. Forecast
# error vs horizon, per-cell breakdowns, and hyperparameter sensitivities
# sit in `00b_supplement.py`.

# %%
import sys
from pathlib import Path

import matplotlib
if "ipykernel" not in sys.modules:
    # Headless script execution (no Jupyter); avoid window popups.
    matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial.distance import pdist

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
# ## Setup: load data, fit selected model
#
# Selected configuration: ``poles_and_chroms`` topology with steric
# envelope on the xx kernel (``r0 = 1.5 um``, ``w = 0.3 um``),
# ``n_basis = 10`` and ``lambda_rough = 1.0``. Bootstrap with 100
# cell-level resamples for kernel CI bands.

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
print(f"Fitted selected model: D_x = {model.D_x:.4e} um^2/s, "
      f"theta size = {model.theta.size}")

boot = bootstrap_kernels(cells, config, n_boot=N_BOOT,
                         rng=np.random.default_rng(42))

_xx_pair_dists = []
for _cell in cells:
    _chroms = np.moveaxis(_cell.chromosomes, 2, 1)  # (T, N, 3)
    for _t in range(_chroms.shape[0]):
        _frame = _chroms[_t]
        _ok = ~np.any(np.isnan(_frame), axis=1)
        if _ok.sum() < 2:
            continue
        _xx_pair_dists.extend(pdist(_frame[_ok]))
_xx_pair_dists = np.asarray(_xx_pair_dists)
R_XX_MIN_PLOT = float(np.quantile(_xx_pair_dists, 0.01))
print(f"Chrom-chrom pair-distance 1%-quantile = {R_XX_MIN_PLOT:.3f} um "
      f"(used to truncate xx kernel plots)")

# %% [markdown]
# ## Fig 1. PCA trajectories and chromosome-pole lag correlation

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

# Pooled-positions PCA: fit on all centrosome and chromosome positions
# across time, with PC1 sign-aligned to the inter-pole axis.  Same
# construction used for Fig 3b so the two figures share a spindle frame.
pca = fit_pca_basis(cell)

chrom_com = np.nanmean(cell.chromosomes, axis=2)
chrom_com_pca = pca.project(chrom_com)
p1_pca = pca.project(cell.centrioles[:, :, 0])
p2_pca = pca.project(cell.centrioles[:, :, 1])

pole_com = 0.5 * (cell.centrioles[:, :, 0] + cell.centrioles[:, :, 1])
pole_com_pca = pca.project(pole_com)

# Individual chromosomes — Purples, faint
for j in range(n_chrom):
    chrom_j = cell.chromosomes[:, :, j]
    if np.any(np.isnan(chrom_j)):
        continue
    cj_pca = pca.project(chrom_j)
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
ax_pca.set_title("Centrosomes and chromosomes follow each other",
                 loc="left", fontsize=9)

# Neutral grey time gradient (decoupled from any single track's colormap).
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize as _Normalize

cbar_time = inset_axes(ax_pca, width="40%", height="4%", loc="lower right",
                       borderpad=2.6)
_time_sm = ScalarMappable(norm=_Normalize(0.0, 1.0), cmap="Greys")
_time_sm.set_array([])
plt.colorbar(_time_sm, cax=cbar_time, orientation="horizontal")
cbar_time.set_xlabel("Time", fontsize=6.5, labelpad=2)
cbar_time.set_xticks([])
cbar_time.tick_params(length=0)

_legend_specs = [
    ("individual chromosomes", "Purples", 0.65),
    ("individual centrosomes", "Greys",   0.70),
    ("pole COM",               "YlGnBu",  0.65),
    ("chromosome COM",         "YlOrRd",  0.65),
]
_stroke = [pe.withStroke(linewidth=2.2, foreground="white", alpha=0.9)]
for _i, (_label, _cmap, _t) in enumerate(_legend_specs):
    _color = plt.get_cmap(_cmap)(_t)
    ax_pca.text(
        0.035, 0.965 - 0.062 * _i, _label,
        transform=ax_pca.transAxes, ha="left", va="top",
        color=_color, fontsize=7.5, fontweight="bold",
        path_effects=_stroke, zorder=20,
    )

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

ax_lag.axhline(0.0, color="0.5", linestyle="--", linewidth=0.6)
ax_lag.axvline(0.0, color="0.5", linestyle="--", linewidth=0.6)
ax_lag.set_xlim(-100.0, 100.0)
ax_lag.set_xlabel("Lag (s) - positive = chromosomes lag poles")
ax_lag.set_ylabel("Velocity dot product (norm.)")
ax_lag.set_title("Lag correlation across cells", loc="left", fontsize=9)
ax_lag.legend(loc="upper left", frameon=False)

ax_lag.plot(peak_lag, peak_val, "o", color=OKABE_ITO["vermil"],
            markersize=5.5, markeredgecolor="white", markeredgewidth=0.6,
            zorder=5)
ax_lag.annotate(f"peak @ {peak_lag:+.0f} s",
                xy=(peak_lag, peak_val),
                xytext=(peak_lag + 50, peak_val + 0.10),
                fontsize=7, color=OKABE_ITO["vermil"],
                arrowprops=dict(arrowstyle="->", color=OKABE_ITO["vermil"],
                                lw=0.8, shrinkA=2, shrinkB=4))

# Panel labels (A, B)
for ax, label in [(ax_pca, "A"), (ax_lag, "B")]:
    ax.text(-0.13, 1.02, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")

fig1.tight_layout()
save_figure(fig1, "fig1_pca_lag")
plt.show()

# %% [markdown]
# Fig 1. Chromosome center-of-mass motion lags pole center-of-mass
# motion during early prometaphase. (A) Example RPE-1 control cell
# projected into the spindle principal-axis frame, with individual
# chromosomes, both centrosomes, and chromosome and pole center-of-mass
# tracks shown over time. (B) Pooled lag correlation between pole-COM
# and chromosome-COM velocities across all cells; the positive-lag
# peak shows that chromosome motion follows pole motion.
#
# - The PCA basis is fit on the pooled 3D positions of both centrosomes
#   and all chromosomes in this cell across time, with PC1 sign-aligned
#   to the inter-pole axis. The same construction is used in Fig 3b so
#   the two PCA panels share a spindle frame.
# - Example cell shown is `rpe18_ctr_032`.
# - Trajectories are trimmed from NEB to `frac = 0.4` of the NEB-to-AO
#   interval, the early-prometaphase gathering window in which
#   chromosomes congress while spindle elongation rate stays roughly
#   constant.
# - Lag correlation is computed per cell on the COM velocity time series
#   then pooled across cells. A positive peak at positive lag is a
#   temporal-ordering diagnostic: chromosome-COM velocity is most
#   correlated with earlier pole-COM velocity.
# - Together the panels motivate treating centrosomes as observed
#   external drivers in the chromosome force model used in Figs 3-4.

# %% [markdown]
# ## Fig 2. Pole motion is predicted from inter-pole interactions alone

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
# Two metrics with complementary interpretations:
# - 1-step RMSE: predicts pole velocity given current state — the
#   AUTONOMOUS one-frame test.
# - rollout path MSE: integrates pole positions forward from the
#   real initial frame using the fitted kernels, then averages
#   squared position error over the trimmed window.  The pp+cp
#   rollout uses *real* held-out chromosome positions as inputs
#   (not co-simulated), so its lower path MSE reflects partition
#   non-identifiability + chromosome-correlated noise fitting,
#   not a clean causal advantage for CS-CH coupling.
print("Computing pp-only vs pp+cp vs cp-only LOO metrics...")
cell_mats = [_build_cell_matrices(c) for c in cells]
n_pp = N_BASIS_PP
path_mse_pp_only = np.empty(len(cells))
path_mse_full = np.empty(len(cells))
path_mse_cp_only = np.empty(len(cells))
rmse1_pp_only = np.empty(len(cells))
rmse1_full = np.empty(len(cells))
rmse1_cp_only = np.empty(len(cells))
for i, cell in enumerate(cells):
    G_train = np.vstack([cell_mats[j][0] for j in range(len(cells)) if j != i])
    V_train = np.concatenate([cell_mats[j][1] for j in range(len(cells)) if j != i])
    G_test, V_test = cell_mats[i]

    theta_pp_only = _ridge_fit(G_train[:, :n_pp], V_train, R=R_pp_mat)
    theta_full_loo = _ridge_fit(G_train, V_train, R=R_full_mat)
    theta_cp_only = _ridge_fit(G_train[:, n_pp:], V_train, R=R_cp_mat)

    # 1-step velocity prediction error on held-out cell
    rmse1_pp_only[i] = float(np.sqrt(np.mean((V_test - G_test[:, :n_pp] @ theta_pp_only) ** 2)))
    rmse1_full[i] = float(np.sqrt(np.mean((V_test - G_test @ theta_full_loo) ** 2)))
    rmse1_cp_only[i] = float(np.sqrt(np.mean((V_test - G_test[:, n_pp:] @ theta_cp_only) ** 2)))

    # Forward rollout path MSE
    f_pp_only_loo = _force_callable(basis_pp, theta_pp_only)
    f_pp_full_loo = _force_callable(basis_pp, theta_full_loo[:n_pp])
    f_cp_full_loo = _force_callable(basis_cp, theta_full_loo[n_pp:])
    f_cp_only_loo = _force_callable(basis_cp, theta_cp_only)

    poles_init = cell.centrioles[0].T  # (2, 3)
    n_steps = cell.centrioles.shape[0] - 1
    dt = cell.dt
    traj_pp = _simulate_poles_det(poles_init, dt, n_steps, f_pp=f_pp_only_loo)
    traj_full = _simulate_poles_det(poles_init, dt, n_steps,
                                    f_pp=f_pp_full_loo, f_cp=f_cp_full_loo,
                                    chromosomes=cell.chromosomes)
    traj_cp = _simulate_poles_det(poles_init, dt, n_steps,
                                  f_cp=f_cp_only_loo,
                                  chromosomes=cell.chromosomes)
    path_mse_pp_only[i] = _path_mse_pp(cell, traj_pp)
    path_mse_full[i] = _path_mse_pp(cell, traj_full)
    path_mse_cp_only[i] = _path_mse_pp(cell, traj_cp)

print(f"  1-step RMSE  pp-only: {rmse1_pp_only.mean():.5f}, pp+cp: {rmse1_full.mean():.5f}, "
      f"cp-only: {rmse1_cp_only.mean():.5f} um/s")
print(f"  Path MSE     pp-only: {path_mse_pp_only.mean():.3f}, pp+cp: {path_mse_full.mean():.3f}, "
      f"cp-only: {path_mse_cp_only.mean():.3f} um^2")

# %%
# --- Bootstrap kernels (cell-level resampling) ----------------------
N_BOOT_F2 = 100
print(f"Bootstrapping pp-only, pp+cp, and cp-only kernels (n_boot={N_BOOT_F2})...")
r_pp_grid = np.linspace(R_MIN, R_MAX, 200)
r_cp_grid = np.linspace(R_MIN, R_MAX, 200)
boot_fpp_pp_only = np.empty((N_BOOT_F2, len(r_pp_grid)))
boot_fpp_full = np.empty((N_BOOT_F2, len(r_pp_grid)))
boot_fcp_full = np.empty((N_BOOT_F2, len(r_cp_grid)))
boot_fcp_cp_only = np.empty((N_BOOT_F2, len(r_cp_grid)))
rng_f2 = np.random.default_rng(2026)
for b in range(N_BOOT_F2):
    idx_b = rng_f2.choice(len(cells), size=len(cells), replace=True)
    G_b = np.vstack([cell_mats[j][0] for j in idx_b])
    V_b = np.concatenate([cell_mats[j][1] for j in idx_b])
    theta_pp_b = _ridge_fit(G_b[:, :n_pp], V_b, R=R_pp_mat)
    theta_full_b = _ridge_fit(G_b, V_b, R=R_full_mat)
    theta_cp_b = _ridge_fit(G_b[:, n_pp:], V_b, R=R_cp_mat)
    boot_fpp_pp_only[b] = basis_pp.evaluate(r_pp_grid) @ theta_pp_b
    boot_fpp_full[b] = basis_pp.evaluate(r_pp_grid) @ theta_full_b[:n_pp]
    boot_fcp_full[b] = basis_cp.evaluate(r_cp_grid) @ theta_full_b[n_pp:]
    boot_fcp_cp_only[b] = basis_cp.evaluate(r_cp_grid) @ theta_cp_b

# %%
# --- Render Fig 2 ---------------------------------------------------
fig2 = plt.figure(figsize=(7.0, 5.6))
gs2 = fig2.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.45,
                         wspace=0.35)
ax_fpp = fig2.add_subplot(gs2[0, 0])
ax_fcp = fig2.add_subplot(gs2[0, 1])
ax_strip = fig2.add_subplot(gs2[1, :])  # path-MSE bars span both columns


def _plot_kernel_band(ax, r_grid, boot_curves, color, label, linestyle="-"):
    median = np.median(boot_curves, axis=0)
    lo = np.quantile(boot_curves, 0.05, axis=0)
    hi = np.quantile(boot_curves, 0.95, axis=0)
    ax.fill_between(r_grid, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.plot(r_grid, median, color=color, linewidth=1.8, label=label,
            linestyle=linestyle)


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
                  OKABE_ITO["blue"], "pp", linestyle="--")
_plot_kernel_band(ax_fpp, r_pp_grid[mask_pp], boot_fpp_full[:, mask_pp],
                  OKABE_ITO["vermil"], "pp+cp", linestyle="-")
ax_fpp.axhline(0.0, color="0.5", linestyle="--", linewidth=0.6)
ax_fpp.set_xlabel("Pole-pole distance (μm)")
ax_fpp.set_ylabel("$f_{pp}(r)$  (μm/s) "
                  "\n+ attractive · - repulsive")
ax_fpp.set_title("Effective pole-separation kernel", loc="left", fontsize=8.5)
ax_fpp.set_xlim(float(pp_lo), float(pp_hi))
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

# Scale f_cp by the chromosome count so the magnitude is directly
# comparable to f_pp (a single inter-pole interaction).  Assumes all
# 46 chromosomes contribute in the same direction; in reality there
# are cancellations from chromosomes on opposite sides of the pole.
N_CHROM_SCALE = 46
mask_cp = (r_cp_grid >= cp_lo) & (r_cp_grid <= cp_hi)
_plot_kernel_band(ax_fcp, r_cp_grid[mask_cp],
                  N_CHROM_SCALE * boot_fcp_full[:, mask_cp],
                  OKABE_ITO["vermil"],
                  f"{N_CHROM_SCALE} × $f_{{cp}}$ (pp+cp)",
                  linestyle="-")
_plot_kernel_band(ax_fcp, r_cp_grid[mask_cp],
                  N_CHROM_SCALE * boot_fcp_cp_only[:, mask_cp],
                  OKABE_ITO["green"],
                  f"{N_CHROM_SCALE} × $f_{{cp}}$ (cp-only)",
                  linestyle="-.")
ax_fcp.axhline(0.0, color="0.5", linestyle="--", linewidth=0.6)
ax_fcp.set_xlabel("Chromosome-to-pole distance (μm)")
ax_fcp.set_ylabel(f"{N_CHROM_SCALE} · $f_{{cp}}(r)$  (μm/s) "
                   "\n+ attractive · - repulsive")
ax_fcp.set_title("Total CS-CH contribution",
                 loc="left", fontsize=8.5)
ax_fcp.set_xlim(float(cp_lo), float(cp_hi))
ax_fcp.legend(frameon=False, loc="upper right", fontsize=6.5)

# Share y-axis between f_pp and 46·f_cp so the overall CS-CH
# contribution is visually comparable to f_pp.
y_min_f2 = min(ax_fpp.get_ylim()[0], ax_fcp.get_ylim()[0])
y_max_f2 = max(ax_fpp.get_ylim()[1], ax_fcp.get_ylim()[1])
ax_fpp.set_ylim(y_min_f2, y_max_f2)
ax_fcp.set_ylim(y_min_f2, y_max_f2)

# Path MSE bar chart (mean ± SE) for the three models, on the same
# headline metric as main Fig 3.  pp and pp+cp bars are very close
# (the data behind the "CS-CS sufficient" framing); cp-only is
# clearly worse, confirming an inter-pole term is needed.
xs_bar_f2 = np.array([0.0, 0.7, 1.4])
m_pp, se_pp = float(path_mse_pp_only.mean()), float(path_mse_pp_only.std(ddof=1) / np.sqrt(len(cells)))
m_fl, se_fl = float(path_mse_full.mean()),    float(path_mse_full.std(ddof=1) / np.sqrt(len(cells)))
m_cp, se_cp = float(path_mse_cp_only.mean()), float(path_mse_cp_only.std(ddof=1) / np.sqrt(len(cells)))
def _darken_f2(hex_color, factor=0.55):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return (r * factor, g * factor, b * factor)

_f2_specs = [
    (xs_bar_f2[0], m_pp, se_pp, OKABE_ITO["blue"]),
    (xs_bar_f2[1], m_fl, se_fl, OKABE_ITO["vermil"]),
    (xs_bar_f2[2], m_cp, se_cp, OKABE_ITO["green"]),
]
for _x, _m, _se, _color in _f2_specs:
    ax_strip.bar(_x, _m, width=0.45, color=_color,
                 alpha=0.85, edgecolor="white", linewidth=0.6)
    ax_strip.errorbar(_x, _m, yerr=_se, fmt="none",
                      ecolor=_darken_f2(_color), capsize=4, linewidth=1.0)
ax_strip.set_xticks(xs_bar_f2)
ax_strip.set_xticklabels(["pp", "pp+cp", "cp-only"])
ax_strip.set_ylabel("LOO path MSE (μm²)")
ax_strip.set_title("Held-out pole-trajectory error",
                   loc="left", fontsize=8.5)
ax_strip.set_xlim(-0.45, 1.85)
_y_lo_f2 = min(m_pp - se_pp, m_fl - se_fl, m_cp - se_cp) * 0.85
_y_hi_f2 = max(m_pp + se_pp, m_fl + se_fl, m_cp + se_cp) * 1.18
ax_strip.set_ylim(_y_lo_f2, _y_hi_f2)

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
print("clean causal advantage; 1-step RMSE is the autonomous test.")
print()
print("cp-only:")
print(f"  1-step RMSE: {rmse1_cp_only.mean():.5f} um/s")
print(f"  Path MSE:    {path_mse_cp_only.mean():.2f} um^2")
print("  cp-only is structurally insufficient — it cannot reproduce")
print("  pole motion without an inter-pole term.  The (f_pp, f_cp)")
print("  split inside the pp+cp model is non-identifiable (Fig S1).")

# Panel labels
for ax, label in [(ax_fpp, "A"), (ax_fcp, "B"), (ax_strip, "C")]:
    ax.text(-0.13, 1.04, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")

fig2.tight_layout()
save_figure(fig2, "fig2_cs_sufficient")
plt.show()

# %% [markdown]
# Fig 2. Pole motion is predicted from inter-pole interactions alone.
# (A) The effective pole-pole term f_pp(r), as a function of inter-pole
# distance, from the pole-velocity regression. Bands are 5-95 %
# bootstrap CI over cell-level resamples. (B) The chromosome-to-pole
# contribution scaled by the typical chromosome count, 46·f_cp(r),
# so the per-pair radial term is on the same scale as the single
# inter-pole interaction. The 46x value is not the actual chromosome
# contribution to pole velocity: chromosomes pull from many
# directions and partially cancel, while this curve sums radial
# magnitudes only. The f_pp / f_cp split inside the joint pp+cp fit
# is not separately identifiable; Fig S1 shows this directly. (C)
# Held-out path MSE from deterministic rollout (mean squared 3D
# pole-position error over the trimmed early-prometaphase window) for
# three pole-velocity models: inter-pole only (pp), inter-pole plus
# chromosome-to-pole (pp+cp), and chromosome-to-pole only (cp-only).
# pp and pp+cp are close, while cp-only cannot reproduce pole motion
# without an inter-pole term.
#
# - Two complementary metrics are reported in the printed numerical
#   summary: 1-step velocity RMSE (autonomous one-frame prediction) and
#   path MSE (full deterministic rollout from the real initial frame).
#   Both support the same qualitative conclusion once the pp/cp
#   non-identifiability caveat is applied; path MSE is shown in panel C.
# - The pp+cp rollout uses real held-out chromosome positions as
#   covariates rather than co-simulating chromosomes. Its slightly lower
#   path MSE reflects partition non-identifiability and chromosome-
#   correlated noise fitting, not a clean causal advantage for pole-
#   chromosome coupling.
# - 46 is the typical chromosome count visible per cell in this
#   condition. The 46x scaling in panel B sums the radial component
#   of the per-chromosome contribution and ignores cancellation from
#   the chromosomes' angular distribution around the pole, so it
#   exceeds the actual chromosome contribution to pole velocity.
# - The basis for both kernels is `BSplineBasis(R_MIN=0.3, R_MAX=15.0,
#   n_basis=6)` with a roughness penalty `lambda_rough = 1.0` and
#   numerical-jitter ridge `lambda_ridge = 1e-6`. The penalty scales
#   are identical to the chromosome force model (Fig 3) so the same
#   smoothness prior applies to both regressions.
# - The pole-velocity regression is intentionally outside the
#   `chromlearn.model_fitting` chromosome-dynamics pipeline, since
#   here we are predicting pole motion rather than chromosome motion.
#   The same trajectory data supports both regressions.

# %% [markdown]
# ## Fig 3. Learned chromosome force kernels and topology selection

# %%
# --- 4 topologies: fit + bootstrap + rollout CV --------------------
# Three admissible models (poles, center, short range) plus the
# free-form ``poles_and_chroms`` as a nuisance upper bound.
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
    "poles_and_chroms_enveloped":  {"label": "poles + chroms\n(short range)",
                                    "color": OKABE_ITO["vermil"],
                                    "admissible": True},
    "poles_and_chroms":            {"label": "poles + chroms\n(free)",
                                    "color": OKABE_ITO["purple"],
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


# Reuse the top-level fit + bootstrap for poles_and_chroms_enveloped
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
     "poles + chroms (free)"),
    ("poles_and_chroms_enveloped",  (0, (5, 1, 1, 1)),
     "poles + chroms (short range)"),
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
r_xx_grid = np.linspace(R_XX_MIN_PLOT, F_XX_PLOT_MAX, 400)
_kernel_band(ax_fxx, models_f3["poles_and_chroms_enveloped"],
             boots_f3["poles_and_chroms_enveloped"],
             "xx", r_xx_grid,
             TOPO_DISPLAY["poles_and_chroms_enveloped"]["color"],
             "short range",
             linestyle=(0, (5, 1, 1, 1)))
_kernel_band(ax_fxx, models_f3["poles_and_chroms"],
             boots_f3["poles_and_chroms"],
             "xx", r_xx_grid,
             OKABE_ITO["purple"],
             "free",
             linestyle=":")
ax_fxx.axhline(0.0, color="0.5", linestyle="--", linewidth=0.6)
ax_fxx.set_xlim(R_XX_MIN_PLOT, F_XX_PLOT_MAX)
ax_fxx.set_xlabel("Chromosome-chromosome distance (μm)")
ax_fxx.set_ylabel("$f_{xx}(r)$  (μm/s) "
                   "\n+ attractive · - repulsive")
ax_fxx.set_title("Chromosome-chromosome interaction (per pair)",
                 loc="left", fontsize=8.5)
ax_fxx.legend(frameon=False, loc="lower right", fontsize=6.5)

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

def _darken(hex_color, factor=0.55):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return (r * factor, g * factor, b * factor)

xs_bar = np.arange(n_topo, dtype=float)
for bi, ti in enumerate(sort_idx):
    info = TOPO_DISPLAY[TOPOLOGIES_F3[ti]]
    ax_path.bar(xs_bar[bi], sorted_means[bi], width=0.7,
                color=info["color"], alpha=0.85,
                edgecolor="white", linewidth=0.6)
    ax_path.errorbar(xs_bar[bi], sorted_means[bi], yerr=sorted_ses[bi],
                     fmt="none", ecolor=_darken(info["color"]),
                     capsize=3, linewidth=1.0)

ax_path.set_xticks(xs_bar)
ax_path.set_xticklabels(
    [TOPO_DISPLAY[t]["label"] for t in sorted_topos],
    fontsize=7,
)
ax_path.set_ylabel("LOO path MSE (μm²)")
ax_path.set_title("Mean per-cell trajectory error across topologies",
                  loc="left", fontsize=8.5)

y_lo_path = float(sorted_means.min()) * 0.85
y_hi_path = float((sorted_means + sorted_ses).max()) * 1.18
ax_path.set_ylim(y_lo_path, y_hi_path)

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
# Fig 3. A short-range chromosome-chromosome prior removes the
# long-range f_xx artifact without sacrificing held-out trajectory
# accuracy; among the biologically admissible models, the short-range
# topology is selected by deterministic-rollout path MSE. (A) f_xy(r),
# the chromosome-to-partner radial force kernel, overlaid for four
# candidate topologies: chromosome attracted to nearest pole (poles),
# chromosome attracted to the spindle midpoint (center), poles plus
# chromosome-chromosome free (poles + chroms, free), and poles plus
# chromosome-chromosome enveloped (poles + chroms, short range).
# Bands are 5-95 % bootstrap CI. (B) f_xx(r), the chromosome-
# chromosome per-pair kernel, for the two topologies that include
# one. The short-range fit applies a smooth steric envelope
# `s(r) = 0.5 (1 - tanh((r - r0) / w))` with r0 = 1.5 um and
# w = 0.3 um, which suppresses the kernel beyond the contact regime.
# The free fit deviates from zero at long range, an artifact that
# motivates the envelope. The free topology is a flexible but
# biologically inadmissible upper bound, since long-range
# chromosome-chromosome forces have no known biological basis in
# mammalian mitosis. (C) Leave-one-cell-out deterministic-rollout
# path MSE for the same four topologies, sorted ascending; the
# short-range model is the selected biologically admissible topology.
#
# - Steric envelope rationale: the biologically admissible chromosome-
#   chromosome interaction is short-range steric repulsion only,
#   reflecting the kinetochore plus chromatid contact scale of ~1-2 um
#   in RPE-1 cells (Renda 2020, Gallego 2010). The envelope encodes
#   this prior in the basis itself rather than in the coefficients,
#   so all downstream code (features, simulation, plotting,
#   `diffusion.f_corrected`) is unaffected.
# - Topology admissibility: `poles` and `center` are physically
#   motivated (chromosome attracted to nearest pole vs to spindle
#   midpoint). `poles_and_chroms_enveloped` adds short-range steric
#   chromosome repulsion. `poles_and_chroms` (free) is a flexible
#   nuisance-absorbing upper bound, not interpretable as biology.
#   `center_and_chroms` is omitted from the plotted set for the same
#   reason: full-range chromosome-chromosome forces are biologically
#   inadmissible.
# - Selection criterion: leave-one-cell-out deterministic-drift-rollout
#   path MSE over the trimmed early-prometaphase window (NEB to
#   `frac = 0.4` of NEB-AO, ~150 s at dt = 5 s). Path MSE integrates
#   horizon-resolved error over the predeclared analysis window,
#   avoiding an arbitrary single-horizon choice. Held-out forecast
#   error vs horizon and per-cell breakdowns sit in Fig S2.
# - Why deterministic rollout: the fitted scalar D may be dominated by
#   measurement and tracking noise rather than thermal fluctuations, so
#   stochastic rollouts can be actively misleading. The ensemble mean
#   of many SDE replicates approximates the ODE solution when D is
#   small and the force field is not strongly curved, so deterministic
#   rollout is the cheap and noise-free version of the same quantity.
# - Hyperparameters used for the fits in panels A-C: `n_basis = 10`,
#   `lambda_rough = 1`, `lambda_ridge = 1e-6`. Sensitivity to these
#   choices is in Fig S3.
# - Cropping: f_xy is plotted out to 12 um (slightly past the
#   data-supported 10 um range to keep the comparison readable);
#   f_xx is plotted from the 1%-quantile of observed chromosome-
#   chromosome distances out to the basis upper bound, so the free-fit
#   long-range artifact is visible.

# %% [markdown]
# ## Fig 3b. Forward simulation reproduces observed gathering

# %%
from chromlearn.model_fitting.simulate import simulate_cell

selected_pmse = rollouts_f3["poles_and_chroms_enveloped"].path_mse
PREFERRED_CELL_ID = "rpe18_ctr_006"
_preferred_idx = next(
    (i for i, c in enumerate(cells) if c.cell_id == PREFERRED_CELL_ID),
    None,
)
good_idx = _preferred_idx if _preferred_idx is not None else int(np.argmin(selected_pmse))
good_cell = cells[good_idx]
print(f"Sim-vs-real cell: idx={good_idx}, id={good_cell.cell_id}, "
      f"selected-topology path MSE = {selected_pmse[good_idx]:.2f} um^2 "
      f"(median {np.median(selected_pmse):.2f})")

# Deterministic and stochastic rollouts of the same cell.  The fitted
# D_x mixes intrinsic chromosome diffusion with kinetochore localization
# noise (~50-80 nm per coordinate per Renda 2022 / RPE-1 tracking
# studies, contributing sigma_loc^2 / dt ~= 1e-3 um^2/s — comparable
# to D_fit = 0.002 um^2/s).  For the SDE rollout we use half of D_x as
# a defensible split between intrinsic and tracking-noise components,
# so the visualization reflects the intrinsic diffusion the model is
# really claiming rather than the localization-noise-inflated D_fit.
import dataclasses as _dc
SDE_D_FRACTION = 0.5
model_sde = _dc.replace(model, D_x=model.D_x * SDE_D_FRACTION)
_, sim_cell_det = simulate_cell(good_cell, model, deterministic=True)
_, sim_cell_sde = simulate_cell(good_cell, model_sde,
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


def _draw_pca_panel(ax, cell_obj, title):
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
    return pole_lc


_ = _draw_pca_panel(ax_real, good_cell, "Real trajectories")
_ = _draw_pca_panel(ax_det, sim_cell_det,
                     "Deterministic ODE rollout")
_ = _draw_pca_panel(ax_sde, sim_cell_sde,
                     "Stochastic SDE rollout (single replicate)")
ax_real.set_ylabel("PC2 (μm)")
ax_real.autoscale()

for ax, label in [(ax_real, "A"), (ax_det, "B"), (ax_sde, "C")]:
    ax.text(-0.13, 1.02, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")

fig3b.suptitle("Real vs simulated dynamics in PCA space",
               fontsize=10, y=0.995)
fig3b.tight_layout()
save_figure(fig3b, "fig3b_sim_vs_real_pca")
plt.show()


# %% [markdown]
# Fig 3b. The selected short-range model reproduces chromosome
# gathering when run forward from the real initial frame. Cell
# `rpe18_ctr_006` is shown, projected into a PCA basis fit from the
# cell's combined pole and chromosome point cloud. Left: real
# trajectories. Middle: deterministic ODE rollout under the fitted
# force kernels. Right: a single stochastic SDE replicate using
# diffusion coefficient D_x/2, splitting the fitted noise budget
# between intrinsic chromosome diffusion and localization/tracking
# error. Centrosomes are colored in greys (light to dark over time)
# with circle start and square end markers; chromosomes use one
# categorical color per chromosome, with no time encoding, so the
# eye can match individuals across panels.
#
# - Cell selection: a cell with low LOO path MSE on the selected
#   topology is chosen as a representative example. The same panels
#   for other cells are qualitatively similar but visually noisier;
#   the example here is meant to be readable, not best-case.
# - The PCA basis is fit from the combined pole-plus-chromosome point
#   cloud, then sign-aligned so PC1 lies along the pole-pole axis. No
#   manual centering is applied.
# - Why D_x/2 in the stochastic panel: the fitted scalar D absorbs both
#   genuine chromosome diffusion and kinetochore localization /
#   tracking noise. Splitting D evenly is a display convention that
#   avoids assigning all fitted residual variance to intrinsic
#   chromosome motion. Changing this split changes the visible
#   stochastic spread but not the deterministic fitted force field.
# - Only one SDE replicate is shown; multi-replicate ensemble means
#   approach the ODE rollout. The point is to show that a single noisy
#   realization remains visibly close to the data.
# - Quantitative held-out comparisons live in Fig 3 panel C and Fig S2.

# %% [markdown]
# ## Fig 4. Effective diffusion landscape and drift signal fraction

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
# Local drift signal fraction over the steady-elongation regime:
#
#   f_drift(d; T) = |F(d)|^2 T / (|F(d)|^2 T + 2 D(d))     [1D-along-F]
#
# Frishman & Ronceray PRX 2020 / arXiv:1809.09650 cast SFI capacity as the
# rate at which a drifted trajectory becomes distinguishable from pure
# Brownian motion, with signal scale F^T D^{-1} F.  Our scalar-D form is
# the per-state, force-direction-aligned reduction.
#
# T is fixed at 150 s (~2.5 min), the duration of the steady spindle-
# elongation phase ("the elongation rate of the spindle length remains
# remarkably constant for ~ 2.5 min").  This is the biologically
# motivated timescale over which the inferred constant-coefficient
# drift law is meant to apply; motors switch on/off between phases,
# so a longer accumulation window would integrate force terms
# outside the modeled regime.  T is not the per-cell trimmed-window
# length (~475 s).  The supplement (Fig S5 panel A) sweeps T to show
# the spatial ordering is robust.
#
# We compute chi_i = T_steady |F_i|^2 / (2 D_i) at the STATE level (per
# chromosome per frame), then bin medians/quantiles by spindle-center
# distance d so we get an honest aggregate without ratio-of-smoothed-mean
# artifacts.
from chromlearn.model_fitting.diffusion import _predicted_forces_all_t

T_STEADY = 150.0  # seconds; docx Result 1 anchor (steady-elongation phase)
# Per-cell trimmed window length (kept for context; NOT used to compute
# the main-panel f_drift, only reported for transparency).
T_obs_per_cell = np.array([(c.chromosomes.shape[0] - 1) * config.dt
                            for c in cells])
T_OBS_MED = float(np.median(T_obs_per_cell))
T_OBS_MIN, T_OBS_MAX = float(T_obs_per_cell.min()), float(T_obs_per_cell.max())
print(f"Trimmed-window length T_obs across cells: median {T_OBS_MED:.0f} s, "
      f"range [{T_OBS_MIN:.0f}, {T_OBS_MAX:.0f}] s "
      "(reported for context only)")
print(f"Main-panel reference timescale T_steady = {T_STEADY:.0f} s "
      "(docx Result 1: steady-elongation phase ~ 2.5 min)")

print("Computing f_drift(d; T_steady) and per-step Pe(d) at every observation...")
fmag_chunks, d_chunks, D_at_d_chunks = [], [], []
for cell in cells:
    coord_arr = _coord_fn(cell.chromosomes, cell)  # (T, N)
    F_all = _predicted_forces_all_t(  # (T-1, N, 3) — batched per cell
        cell,
        fit_result=model,
        basis_xx=model.basis_xx,
        basis_xy=model.basis_xy,
        topology=model.topology,
    )
    Fmag_all = np.linalg.norm(F_all, axis=2)  # (T-1, N)
    d_all = coord_arr[: F_all.shape[0]]       # (T-1, N)
    valid = np.isfinite(Fmag_all) & np.isfinite(d_all)
    if valid.any():
        fmag_chunks.append(Fmag_all[valid])
        d_chunks.append(d_all[valid])
        D_at_d_chunks.append(D_pooled.evaluate(d_all[valid]))

force_mags_d = np.concatenate(fmag_chunks)
d_force = np.concatenate(d_chunks)
D_at_obs = np.concatenate(D_at_d_chunks)
dt_step = config.dt

# Mask off any non-positive D(d) — the spline can dip below zero in
# undersampled tails; those points carry no signal-fraction information.
ok = np.isfinite(D_at_obs) & (D_at_obs > 0.0) & np.isfinite(force_mags_d)

# Per-step Pe (kept for the supplement)
Pe_step_obs = np.full_like(force_mags_d, np.nan)
Pe_step_obs[ok] = force_mags_d[ok] * np.sqrt(dt_step / (2.0 * D_at_obs[ok]))

# State-level chi_i = T_steady |F_i|^2 / (2 D_i) and f_drift_i
chi_obs = np.full_like(force_mags_d, np.nan)
chi_obs[ok] = T_STEADY * (force_mags_d[ok] ** 2) / (2.0 * D_at_obs[ok])
fdrift_obs = chi_obs / (chi_obs + 1.0)

# Bin by d
N_BINS_F4 = 18
bin_edges_f4 = np.linspace(EVAL_LO, EVAL_HI, N_BINS_F4 + 1)
bin_centers_f4 = 0.5 * (bin_edges_f4[:-1] + bin_edges_f4[1:])
bin_idx_f4 = np.clip(np.digitize(d_force, bin_edges_f4) - 1, 0, N_BINS_F4 - 1)

fdrift_med = np.full(N_BINS_F4, np.nan)
fdrift_lo = np.full(N_BINS_F4, np.nan)
fdrift_hi = np.full(N_BINS_F4, np.nan)
n_per_bin = np.zeros(N_BINS_F4, dtype=int)
for b in range(N_BINS_F4):
    mask = (bin_idx_f4 == b) & np.isfinite(fdrift_obs)
    n_per_bin[b] = int(mask.sum())
    if n_per_bin[b] >= 30:  # mask poorly supported bins
        fdrift_med[b] = float(np.median(fdrift_obs[mask]))
        fdrift_lo[b] = float(np.quantile(fdrift_obs[mask], 0.25))
        fdrift_hi[b] = float(np.quantile(fdrift_obs[mask], 0.75))

print(f"  f_drift overall median: {np.nanmedian(fdrift_obs):.3f} "
      f"(IQR [{np.nanquantile(fdrift_obs, 0.25):.3f}, "
      f"{np.nanquantile(fdrift_obs, 0.75):.3f}])")
print(f"  per-step Pe median:     {np.nanmedian(Pe_step_obs):.3f}")
print(f"  bins with >= 30 obs:    {int(np.sum(n_per_bin >= 30))} of {N_BINS_F4}")

# %%
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                   gridspec_kw={"wspace": 0.32})

# --- Panel A: D(d) ---
percell_curves = np.stack([
    d_res.evaluate(d_grid) for d_res in D_per_cell if d_res is not None
], axis=0)
percell_lo = np.quantile(percell_curves, 0.05, axis=0)
percell_hi = np.quantile(percell_curves, 0.95, axis=0)
ax4a.fill_between(d_grid, percell_lo, percell_hi,
                  color=OKABE_ITO["vermil"], alpha=0.18, linewidth=0,
                  label="per-cell 5-95 % range", zorder=1)
pooled_vals = D_pooled.evaluate(d_grid)
ax4a.plot(d_grid, pooled_vals, color=OKABE_ITO["vermil"], linewidth=2.2,
          label=r"pooled $D(d)$", zorder=3)
ax4a.axhline(model.D_x, color="0.3", linestyle="--", linewidth=1.0,
             label=fr"fitted constant $D = {model.D_x:.4f}\,\mu\mathrm{{m}}^2/\mathrm{{s}}$",
             zorder=2)

ax4a.set_xlim(EVAL_LO, EVAL_HI)
ax4a.set_ylim(bottom=0.0)
ax4a.set_xlabel("Distance from spindle center, $d$ (μm)")
ax4a.set_ylabel("D (μm²/s)")
ax4a.set_title("Effective diffusion grows away from spindle center",
               loc="left", fontsize=9)
ax4a.legend(frameon=False, loc="upper left", fontsize=7)

# --- Panel B: f_drift(d; T_steady) drift signal fraction over the
# steady-elongation phase ---
ax4b.fill_between(bin_centers_f4, fdrift_lo, fdrift_hi,
                  color=OKABE_ITO["blue"], alpha=0.18, linewidth=0,
                  label="IQR")
ax4b.plot(bin_centers_f4, fdrift_med, "o-", color=OKABE_ITO["blue"],
          linewidth=1.6, markersize=4.0, label="median")
ax4b.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8,
             label=r"50/50 crossover")
_valid_centers = bin_centers_f4[np.isfinite(fdrift_med)]
ax4b.set_xlim(float(_valid_centers[0]), float(_valid_centers[-1]))
ax4b.set_ylim(0.0, 1.0)
ax4b.set_xlabel("Distance from spindle center, $d$ (μm)")
ax4b.set_ylabel(r"$f_{\mathrm{drift}}(d;\,T)$")
ax4b.set_title("Drift signal fraction over the steady-elongation phase",
               loc="left", fontsize=9)
ax4b.legend(frameon=False, loc="upper left", fontsize=7)

for ax_panel, label in [(ax4a, "A"), (ax4b, "B")]:
    ax_panel.text(-0.13, 1.04, label, transform=ax_panel.transAxes,
                  fontsize=11, fontweight="bold", va="bottom", ha="left")

fig4.tight_layout()
save_figure(fig4, "fig4_diffusion_landscape")
plt.show()

# %% [markdown]
# Fig 4. Drift outweighs diffusion across most of the spindle interior.
# (A) Effective diffusivity D(d), plotted against the chromosome's
# 3D distance d from the spindle center. The vermilion line is the
# pooled fit across all cells; the band spans the per-cell 5-95 %
# range; the dashed line is the fitted constant-D scalar for
# reference. D grows away from the spindle center, consistent with
# our previous report. (B) Local drift signal fraction
# f_drift(d; T) = |F(d)|^2 T / (|F(d)|^2 T + 2 D(d)), the fraction
# of a chromosome's motion attributable to the inferred force F(d)
# rather than to random diffusion D(d), accumulated over a time
# window T. f_drift = 0.5 is the crossover at which the squared
# distance traveled under the drift force equals the variance
# accumulated by diffusion over T. T = 150 s is the duration of the
# steady spindle-elongation phase reported in Result 1 ("the
# elongation rate remains remarkably constant for ~ 2.5 min").
# Solid line is the bin median; band is IQR. Sensitivity to T is in
# Fig S5.
#
# - D(d) is estimated as a second stage with the force field held
#   fixed: residual displacements after subtracting the fitted drift,
#   averaged in distance bins and divided by 2 dt. Subtracting the
#   drift first removes the contamination that would otherwise bias a
#   plain mean-squared-displacement estimator in this drift-dominated
#   setting.
# - Default fits use the Itô convention. When D depends on position,
#   the Itô vs Stratonovich choice changes how a position-dependence
#   of D enters the drift fit. We checked that the corresponding
#   correction (the "spurious force" term proportional to the
#   gradient of D) is small relative to the inferred force here, so
#   D(d) is estimated in this second residual stage rather than
#   refit jointly with the drift. The estimator does not subtract
#   localization noise, so D conflates genuine thermal diffusion
#   with kinetochore tracking error.
# - d is the 3D Euclidean distance from the chromosome to the
#   midpoint of the two centrosomes, not the distance to the nearest
#   pole. The plot range is clipped to the 1-99 % quantile of the
#   observed d distribution so the curve is shown only on the
#   data-supported window.
# - f_drift is computed per chromosome per frame and then median- and
#   IQR-binned by d, with bins below 30 observations masked. Computing
#   the ratio at each observation and then aggregating avoids the
#   biases of smoothing |F|^2 and D separately and dividing.
# - Why T = 150 s: the docx Result 1 reports that spindle elongation
#   rate stays roughly constant for about 2.5 minutes from NEB, the
#   time window over which the inferred constant-coefficient drift
#   law is meant to apply. The full trimmed window (~475 s) is not
#   used because motors switch on and off between mitotic phases.
# - The spatial ordering and shape of f_drift(d; T) change little
#   across T; only the 50/50 crossover position shifts. The T sweep
#   is in Fig S5 panel A. Characteristic length L*(d) = 2 D / |F| and
#   timescale tau_50(d) = 2 D / |F|^2 are in Fig S5 panels B and C.
