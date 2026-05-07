# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 00b. Supplementary figures
#
# Supplement-figure assembler for the chromlearn project. Each section
# below produces one supplementary figure (PDF + 600 dpi PNG) backing
# the main-text claims in `00_main_figure.py`: per-cell breakdowns of
# the pole-velocity fit and its f_pp / f_cp non-identifiability,
# forecast error vs horizon, sensitivity to basis size, smoothness
# penalty, and Itô vs Stratonovich estimator, and characteristic
# spatial and temporal scales of the drift-vs-diffusion comparison.

# %%
import sys
from itertools import product
from pathlib import Path

import matplotlib
if "ipykernel" not in sys.modules:
    # Headless script execution (no Jupyter); avoid window popups.
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.distance import pdist

from chromlearn import find_repo_root

ROOT = find_repo_root(Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chromlearn.io.catalog import load_condition
from chromlearn.io.trajectory import trim_trajectory
from chromlearn.model_fitting import FitConfig
from chromlearn.model_fitting.basis import BSplineBasis
from chromlearn.model_fitting.fit import (
    bootstrap_kernels,
    fit_model,
    rollout_cross_validate,
)

# %% [markdown]
# ## Publication style
#
# Mirrors the main-text notebook so figures are consistent across the
# paper.

# %%
FIG_DIR = ROOT / "figures" / "supplement"
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

# Topology display order and colors.  Mirrors main Fig 3 exactly so
# the same model has the same color/linestyle across figures.
TOPOLOGY_DISPLAY = {
    "poles":                       {"label": "poles",
                                    "color": OKABE_ITO["blue"],
                                    "linestyle": "-",
                                    "admissible": True},
    "center":                      {"label": "center",
                                    "color": OKABE_ITO["green"],
                                    "linestyle": "--",
                                    "admissible": True},
    "poles_and_chroms_enveloped":  {"label": "poles + chroms\n(short range)",
                                    "color": OKABE_ITO["vermil"],
                                    "linestyle": (0, (5, 1, 1, 1)),
                                    "admissible": True},
    "poles_and_chroms":            {"label": "poles + chroms\n(free)",
                                    "color": OKABE_ITO["purple"],
                                    "linestyle": ":",
                                    "admissible": False},
}


def save_figure(fig, name):
    """Save *fig* as both vector PDF and 600 dpi PNG into ``FIG_DIR``."""
    pdf_path = FIG_DIR / f"{name}.pdf"
    png_path = FIG_DIR / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    print(f"Saved {pdf_path.name} and {png_path.name}")


# %% [markdown]
# ## Configuration
#
# Canonical configuration matching the main figure.

# %%
CONDITION = "rpe18_ctr"

# Trajectory model (chromosome dynamics)
FRAC_NEB_AO = 0.4
N_BASIS_TRAJ = 10
LAMBDA_RIDGE = 1e-6
LAMBDA_ROUGH = 1.0
R_MIN = 0.3
R_MAX = 15.0
ENVELOPE_R0_XX = 1.5
ENVELOPE_W_XX = 0.3
DT = 5.0

# Pole-velocity model (used only by Fig S1)
N_BASIS_POLE = 6
LAMBDA_ROUGH_POLE = 1.0

cells_raw = load_condition(CONDITION)
cells = [trim_trajectory(c, method="neb_ao_frac", frac=FRAC_NEB_AO) for c in cells_raw]

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
print(f"Chrom-chrom pair-distance 1%-quantile = {R_XX_MIN_PLOT:.3f} um")
print(f"Loaded {len(cells)} {CONDITION} cells (frac={FRAC_NEB_AO}).")


def make_traj_config(topology, **overrides):
    """Build a trajectory FitConfig for one of the five topology labels."""
    use_env = topology == "poles_and_chroms_enveloped"
    base_topology = "poles_and_chroms" if use_env else topology
    kwargs = dict(
        topology=base_topology,
        envelope_r0_xx=ENVELOPE_R0_XX if use_env else None,
        envelope_w_xx=ENVELOPE_W_XX if use_env else None,
        n_basis_xx=N_BASIS_TRAJ,
        n_basis_xy=N_BASIS_TRAJ,
        r_min_xx=R_MIN,
        r_max_xx=R_MAX,
        r_min_xy=R_MIN,
        r_max_xy=R_MAX,
        lambda_ridge=LAMBDA_RIDGE,
        lambda_rough=LAMBDA_ROUGH,
        basis_eval_mode="ito",
        endpoint_method="neb_ao_frac",
        endpoint_frac=FRAC_NEB_AO,
        diffusion_mode="msd",
        dt=DT,
    )
    kwargs.update(overrides)
    return FitConfig(**kwargs)


# %% [markdown]
# ## Helpers for Fig S1 (pole-velocity pp+cp model)
#
# The pp+cp pole-velocity model lives outside ``chromlearn.model_fitting``
# (which fits chromosome dynamics, not pole dynamics).  We inline the
# minimal machinery needed for the per-cell kernel fits and the
# constrained-share refit sweep.

# %%
basis_pp = BSplineBasis(R_MIN, R_MAX, N_BASIS_POLE)
basis_cp = BSplineBasis(R_MIN, R_MAX, N_BASIS_POLE)
R_pp = basis_pp.roughness_matrix()
R_cp = basis_cp.roughness_matrix()
R_full = block_diag(R_pp, R_cp)
N_PP = N_BASIS_POLE
N_CP = N_BASIS_POLE


def _ridge_fit(G, V, R):
    """Penalized least squares with light ridge + roughness penalty."""
    n = G.shape[1]
    theta = np.linalg.solve(
        G.T @ G + LAMBDA_RIDGE * np.eye(n) + LAMBDA_ROUGH_POLE * R,
        G.T @ V,
    )
    residuals = V - G @ theta
    return theta, residuals


def _chrom_com_at_frame(chroms_t):
    """Mean chromosome position at one frame, ignoring NaNs.

    ``chroms_t`` shape ``(N, 3)``.
    """
    summed = np.nansum(chroms_t, axis=0)
    counts = np.sum(~np.isnan(chroms_t), axis=0)
    return np.divide(summed, counts, out=np.full_like(summed, np.nan), where=counts > 0)


def build_pole_feature_rows(cells_in):
    """Build per-cell pole-velocity design matrices for the pp+cp model.

    Returns (cell_pole_rows, cell_sep_rows, cell_pp_dists, cell_cp_dists)
    where ``cell_pole_rows[i]`` is a list of dicts with ``Gpp`` (3, n_basis),
    ``Gcp`` (3, n_basis), ``v`` (3,) -- one entry per (frame, pole), and
    ``cell_sep_rows[i]`` is a list of scalar dicts with ``gpp`` (n_basis,),
    ``gcp`` (n_basis,), ``y`` for the spindle-separation observable.
    """
    cell_pole_rows = []
    cell_sep_rows = []
    cell_pp_dists = []
    cell_cp_dists = []

    for cell in cells_in:
        pole_rows = []
        sep_rows = []
        pp_dists_cell = []
        cp_dists_cell = []
        T = cell.centrioles.shape[0]

        for t in range(T - 1):
            poles_cur = cell.centrioles[t].T          # (2, 3)
            poles_next = cell.centrioles[t + 1].T
            chroms = cell.chromosomes[t].T            # (N, 3)
            chrom_com = _chrom_com_at_frame(chroms)
            if np.any(np.isnan(chrom_com)):
                continue

            axis = poles_cur[1] - poles_cur[0]
            r_pp = float(np.linalg.norm(axis))
            if r_pp < 1e-12:
                continue
            u_pp = axis / r_pp
            pp_dists_cell.append(r_pp)

            g_pp_per_pole = []
            g_cp_per_pole = []
            v_per_pole = []

            for p in range(2):
                pole_vel = (poles_next[p] - poles_cur[p]) / cell.dt
                other = 1 - p

                delta_pp = poles_cur[other] - poles_cur[p]
                r_pp_val = float(np.linalg.norm(delta_pp))
                dir_pp = delta_pp / r_pp_val
                phi_pp = basis_pp.evaluate(np.array([r_pp_val]))[0]
                g_pp = dir_pp[:, np.newaxis] * phi_pp[np.newaxis, :]      # (3, n_basis)

                g_cp = np.zeros((3, N_CP))
                valid = ~np.any(np.isnan(chroms), axis=1)
                if valid.any():
                    chroms_valid = chroms[valid]
                    delta_cp = chroms_valid - poles_cur[p]
                    dist_cp = np.linalg.norm(delta_cp, axis=1)
                    pair_ok = dist_cp > 1e-12
                    if pair_ok.any():
                        cp_dists_cell.extend(dist_cp[pair_ok].tolist())
                        dir_cp = delta_cp[pair_ok] / dist_cp[pair_ok, np.newaxis]
                        phi_cp = basis_cp.evaluate(dist_cp[pair_ok])
                        g_cp = np.einsum("id,ib->db", dir_cp, phi_cp)

                g_pp_per_pole.append(g_pp)
                g_cp_per_pole.append(g_cp)
                v_per_pole.append(pole_vel)
                pole_rows.append({"Gpp": g_pp, "Gcp": g_cp, "v": pole_vel})

            gpp_a, gpp_b = g_pp_per_pole
            gcp_a, gcp_b = g_cp_per_pole
            v_a, v_b = v_per_pole
            sep_rows.append({
                "gpp": np.einsum("d,db->b", u_pp, gpp_b - gpp_a),
                "gcp": np.einsum("d,db->b", u_pp, gcp_b - gcp_a),
                "y": float(np.dot(u_pp, v_b - v_a)),
            })

        cell_pole_rows.append(pole_rows)
        cell_sep_rows.append(sep_rows)
        cell_pp_dists.append(np.array(pp_dists_cell))
        cell_cp_dists.append(np.array(cp_dists_cell))

    return cell_pole_rows, cell_sep_rows, cell_pp_dists, cell_cp_dists


def stack_pole_rows(cell_rows):
    G, V = [], []
    for rows in cell_rows:
        for row in rows:
            G.append(np.hstack([row["Gpp"], row["Gcp"]]))
            V.append(row["v"])
    return np.vstack(G), np.concatenate(V)


def stack_sep_rows(cell_rows):
    G, Y = [], []
    for rows in cell_rows:
        for row in rows:
            G.append(np.concatenate([row["gpp"], row["gcp"]]))
            Y.append(row["y"])
    return np.vstack(G), np.array(Y)


# Constrained-share helpers.
def _smooth_abs(x, eps=1e-12):
    return np.sqrt(x * x + eps) - np.sqrt(eps)


def _cp_share_smooth(G, theta):
    pred_pp = G[:, :N_PP] @ theta[:N_PP]
    pred_cp = G[:, N_PP:] @ theta[N_PP:]
    mag_pp = _smooth_abs(pred_pp)
    mag_cp = _smooth_abs(pred_cp)
    return float(np.mean(mag_cp / (mag_pp + mag_cp + 1e-12)))


def _cp_share_exact(G, theta):
    pred_pp = G[:, :N_PP] @ theta[:N_PP]
    pred_cp = G[:, N_PP:] @ theta[N_PP:]
    abs_pp = np.abs(pred_pp)
    abs_cp = np.abs(pred_cp)
    return float(np.mean(abs_cp / np.where(abs_pp + abs_cp > 0, abs_pp + abs_cp, 1.0)))


def _scalar_objective(theta, G, Y, R):
    residuals = Y - G @ theta
    return float(
        residuals @ residuals
        + LAMBDA_RIDGE * (theta @ theta)
        + LAMBDA_ROUGH_POLE * (theta @ R @ theta)
    )


def _initial_theta_for_target(G, theta_full, target_share):
    theta0 = theta_full.copy()

    def mismatch(scale):
        t = theta0.copy()
        t[N_PP:] *= scale
        return (_cp_share_smooth(G, t) - target_share) ** 2

    res = minimize_scalar(mismatch, bounds=(0.0, 50.0), method="bounded")
    theta0[N_PP:] *= res.x
    return theta0


def fit_with_share_target(G, Y, target_share):
    if target_share <= 1e-8:
        theta_pp, _ = _ridge_fit(G[:, :N_PP], Y, R_pp)
        return np.concatenate([theta_pp, np.zeros(N_CP)]), {"success": True}
    if target_share >= 1 - 1e-8:
        theta_cp, _ = _ridge_fit(G[:, N_PP:], Y, R_cp)
        return np.concatenate([np.zeros(N_PP), theta_cp]), {"success": True}

    theta_full, _ = _ridge_fit(G, Y, R_full)
    theta_pp, _ = _ridge_fit(G[:, :N_PP], Y, R_pp)
    theta_cp, _ = _ridge_fit(G[:, N_PP:], Y, R_cp)
    pp_only = np.concatenate([theta_pp, np.zeros(N_CP)])
    cp_only = np.concatenate([np.zeros(N_PP), theta_cp])

    inits = [
        _initial_theta_for_target(G, theta_full, target_share),
        (1 - target_share) * pp_only + target_share * cp_only,
        theta_full.copy(),
    ]

    candidates = []
    for theta_init in inits:
        result = minimize(
            lambda th: _scalar_objective(th, G, Y, R_full),
            theta_init,
            method="SLSQP",
            constraints=[{
                "type": "eq",
                "fun": lambda th, G=G, t=target_share: _cp_share_smooth(G, th) - t,
            }],
            options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
        )
        theta_try = result.x
        share_err = abs(_cp_share_smooth(G, theta_try) - target_share)
        meta = {
            "success": bool(result.success) and share_err < 5e-3,
            "share_err": float(share_err),
        }
        candidates.append((
            meta["success"],
            _scalar_objective(theta_try, G, Y, R_full),
            theta_try,
            meta,
        ))

    feasible = [c for c in candidates if c[0]]
    pool = feasible if feasible else candidates
    _, _, best_theta, best_meta = min(pool, key=lambda c: c[1])
    return best_theta, best_meta


def constrained_share_sweep(cell_sep_rows, target_shares):
    """Refit with a constraint on the cp-force share at each target value.

    Returns (train_rows, loocv_rows): list of dicts with refit RMSE on the
    spindle-separation observable -- training (full data) and LOOCV.
    """
    G, Y = stack_sep_rows(cell_sep_rows)
    train_rows = []
    loocv_rows = []
    for target in target_shares:
        theta, meta = fit_with_share_target(G, Y, target)
        train_rows.append({
            "target": float(target),
            "actual_share": _cp_share_exact(G, theta),
            "rmse": float(np.sqrt(np.mean((Y - G @ theta) ** 2))),
            "success": bool(meta["success"]),
        })
        fold_rmses = []
        fold_shares = []
        for i in range(len(cell_sep_rows)):
            train = [rows for j, rows in enumerate(cell_sep_rows) if j != i]
            test = [cell_sep_rows[i]]
            G_train, Y_train = stack_sep_rows(train)
            G_test, Y_test = stack_sep_rows(test)
            theta_fold, _ = fit_with_share_target(G_train, Y_train, target)
            fold_rmses.append(float(np.sqrt(np.mean((Y_test - G_test @ theta_fold) ** 2))))
            fold_shares.append(_cp_share_exact(G_train, theta_fold))
        loocv_rows.append({
            "target": float(target),
            "mean_actual_share": float(np.mean(fold_shares)),
            "rmse_mean": float(np.mean(fold_rmses)),
            "rmse_se": float(np.std(fold_rmses, ddof=1) / np.sqrt(len(fold_rmses))),
        })
    return train_rows, loocv_rows


# %% [markdown]
# ## Fig S1. Per-cell pp/cp kernels and partition non-identifiability

# %%
print("Fig S1: building pole-velocity feature rows...")
cell_pole_rows, cell_sep_rows, cell_pp_dists, cell_cp_dists = build_pole_feature_rows(cells)

# Per-cell pp+cp fits and pooled fit on all cells
percell_thetas = []
for rows in cell_pole_rows:
    G_cell = np.vstack([np.hstack([r["Gpp"], r["Gcp"]]) for r in rows])
    V_cell = np.concatenate([r["v"] for r in rows])
    theta_cell, _ = _ridge_fit(G_cell, V_cell, R_full)
    percell_thetas.append(theta_cell)

G_all, V_all = stack_pole_rows(cell_pole_rows)
theta_pooled, _ = _ridge_fit(G_all, V_all, R_full)

# Plot ranges from observed support so we don't extrapolate.
all_pp = np.concatenate(cell_pp_dists)
all_cp = np.concatenate(cell_cp_dists)
r_pp_plot = np.linspace(np.percentile(all_pp, 1), np.percentile(all_pp, 99), 200)
r_cp_plot = np.linspace(np.percentile(all_cp, 1), np.percentile(all_cp, 99), 200)

phi_pp_plot = basis_pp.evaluate(r_pp_plot)
phi_cp_plot = basis_cp.evaluate(r_cp_plot)

# Constrained-share sweep
target_shares = np.linspace(0.0, 1.0, 11)
print(f"Fig S1 (c): running constrained-share sweep over {len(target_shares)} targets...")
train_rows, loocv_rows = constrained_share_sweep(cell_sep_rows, target_shares)

# Free-fit (unconstrained, full data) on the spindle-separation
# observable.  We mark the free-fit predicted-share α* as a vertical
# reference on panel (c) so the reader can see where the unconstrained
# fit lands inside the constrained-share family.  Plotting the free-fit
# LOOCV RMSE as a horizontal baseline would invite a misreading: the
# constrained sweep can dip slightly below it (regularization by
# structural constraint on a near-collinear pp/cp basis), but the dip
# is within fold-to-fold SE (~0.002) and is not statistically
# distinguishable from the free fit.  The non-identifiability story
# the panel tells is "the LOOCV curve is essentially flat across α,"
# which is what the data show.
G_sep_all, Y_sep_all = stack_sep_rows(cell_sep_rows)
theta_sep_free, _ = _ridge_fit(G_sep_all, Y_sep_all, R_full)
free_alpha = _cp_share_exact(G_sep_all, theta_sep_free)

# Render
fig_s1, axes_s1 = plt.subplots(1, 3, figsize=(11, 3.4))

percell_pp_curves = np.stack(
    [phi_pp_plot @ theta_cell[:N_PP] for theta_cell in percell_thetas], axis=0
)
percell_cp_curves = np.stack(
    [phi_cp_plot @ theta_cell[N_PP:] for theta_cell in percell_thetas], axis=0
)
median_pp = np.median(percell_pp_curves, axis=0)
median_cp = np.median(percell_cp_curves, axis=0)

N_CHROM_SCALE_S1 = 46  # match main Fig 2 panel B scaling
percell_cp_curves_scaled = N_CHROM_SCALE_S1 * percell_cp_curves
median_cp_scaled = N_CHROM_SCALE_S1 * median_cp
pooled_cp_scaled = N_CHROM_SCALE_S1 * (phi_cp_plot @ theta_pooled[N_PP:])

ax_s1A = axes_s1[0]
for curve in percell_pp_curves:
    ax_s1A.plot(r_pp_plot, curve, color="0.55", lw=0.7, alpha=0.55)
ax_s1A.plot(r_pp_plot, median_pp, color="0.15", lw=1.6, ls="--",
            label="per-cell median")
ax_s1A.plot(r_pp_plot, phi_pp_plot @ theta_pooled[:N_PP], color=OKABE_ITO["blue"],
            lw=2.0, label="pooled fit")
ax_s1A.axhline(0, color="0.5", lw=0.6, ls="--")
ax_s1A.set_xlim(r_pp_plot[0], r_pp_plot[-1])
ax_s1A.set_xlabel("Pole-pole distance (μm)")
ax_s1A.set_ylabel("$f_{pp}(r)$  (μm/s) "
                  "\n+ attractive · - repulsive")
ax_s1A.set_title("Per-cell pole-pole kernel, pp+cp model",
                 loc="left", fontsize=8.5)
ax_s1A.legend(loc="best", frameon=False, fontsize=6.5)

ax_s1B = axes_s1[1]
for curve in percell_cp_curves_scaled:
    ax_s1B.plot(r_cp_plot, curve, color="0.55", lw=0.7, alpha=0.55)
ax_s1B.plot(r_cp_plot, median_cp_scaled, color="0.15", lw=1.6, ls="--",
            label="per-cell median")
ax_s1B.plot(r_cp_plot, pooled_cp_scaled, color=OKABE_ITO["vermil"],
            lw=2.0, label="pooled fit")
ax_s1B.axhline(0, color="0.5", lw=0.6, ls="--")
ax_s1B.set_xlim(r_cp_plot[0], r_cp_plot[-1])
ax_s1B.set_xlabel("Chromosome-to-pole distance (μm)")
ax_s1B.set_ylabel(f"{N_CHROM_SCALE_S1} · $f_{{cp}}(r)$  (μm/s) "
                   "\n+ attractive · - repulsive")
ax_s1B.set_title(f"Per-cell total CS-CH contribution ({N_CHROM_SCALE_S1}× per-pair)",
                 loc="left", fontsize=8.5)
ax_s1B.legend(loc="best", frameon=False, fontsize=6.5)

_y_min_s1 = min(ax_s1A.get_ylim()[0], ax_s1B.get_ylim()[0])
_y_max_s1 = max(ax_s1A.get_ylim()[1], ax_s1B.get_ylim()[1])
ax_s1A.set_ylim(_y_min_s1, _y_max_s1)
ax_s1B.set_ylim(_y_min_s1, _y_max_s1)

ax = axes_s1[2]
sweep_actual = np.array([row["mean_actual_share"] for row in loocv_rows])
sweep_rmse = np.array([row["rmse_mean"] for row in loocv_rows])
sweep_se = np.array([row["rmse_se"] for row in loocv_rows])
ax.fill_between(sweep_actual, sweep_rmse - sweep_se, sweep_rmse + sweep_se,
                color=OKABE_ITO["green"], alpha=0.22)
ax.plot(sweep_actual, sweep_rmse, "o-", color=OKABE_ITO["green"], lw=1.5,
        markersize=5, label="constrained-refit LOOCV RMSE")
ax.axvline(free_alpha, color="0.35", ls="--", lw=0.8,
           label=fr"free fit $\alpha^*$ = {free_alpha:.2f}")
ax.set_xlabel(r"chromosome-pole predicted-contribution share"
              "\n"
              r"$\alpha = \langle |F_{cp}|/(|F_{pp}|+|F_{cp}|) \rangle$")
ax.set_ylabel(r"LOOCV RMSE on $\dot r_{pp}$ (μm/s)")
ax.set_title("Constrained-share refit sweep",
             loc="left", fontsize=8.5)
ax.legend(loc="best", frameon=False, fontsize=6.5)

# Panel letters (A/B/C) anchored above each subplot, matching main-text style
for ax_panel, label in zip(axes_s1, ["A", "B", "C"]):
    ax_panel.text(-0.13, 1.04, label, transform=ax_panel.transAxes,
                  fontsize=11, fontweight="bold", va="bottom", ha="left")

fig_s1.tight_layout()
save_figure(fig_s1, "figS1_pp_cp_kernels_partition")
plt.show()


# %% [markdown]
# Fig S1. The pp+cp pole-velocity model cannot separately identify
# the f_pp and f_cp kernels. (A) Per-cell f_pp(r) from individual
# pp+cp fits (grey lines), with the per-cell median (dashed) and the
# pooled fit (blue) overlaid. (B) Per-cell 46·f_cp(r), the per-pair
# chromosome-to-pole contribution scaled by the typical chromosome
# count, on the same y-axis as (A) for direct scale comparison. The
# 46x value sums radial-magnitude contributions and so overstates
# the actual chromosome contribution to pole velocity. (C)
# Constrained-share refit sweep on the spindle-elongation velocity:
# at each target chromosome-pole share
# alpha = <|F_cp| / (|F_pp| + |F_cp|)>, the pp+cp model is refit with
# alpha held to that value and held-out LOOCV RMSE on the elongation
# velocity is recomputed. The curve is flat across alpha, which means
# many different (f_pp, f_cp) splits fit the data equally well. The
# unconstrained free-fit alpha* is marked as a vertical reference.
#
# - The pp+cp pole-velocity regression lives outside the
#   `chromlearn.model_fitting` chromosome-dynamics pipeline, since
#   here we are predicting pole motion rather than chromosome motion.
#   The minimal design-matrix code is inlined in this notebook above.
# - Each per-cell fit uses the same B-spline basis and roughness
#   penalty as the pooled fit. Under-determined cells produce noisier
#   curves; the pooled fit averages over cells.
# - The 46x scaling on f_cp matches main Fig 2 panel B and converts
#   the per-pair kernel into the per-pole total a centrosome
#   experiences from the chromosome ensemble.
# - Constrained-share construction: for a target alpha in [0, 1] we
#   minimize the pole-velocity residual subject to a smooth equality
#   constraint that enforces the predicted-share to that target. The
#   sweep traces a continuous family of (alpha f_pp, (1 - alpha) f_cp)
#   refits.
# - Why the curve is flat rather than convex: pp+cp is a near-collinear
#   regression on pole velocity; many (f_pp, f_cp) pairs fit equally
#   well in a least-squares sense. The free-fit alpha* lies inside the
#   flat region. Mild regularization by the structural constraint can
#   even push the constrained sweep slightly below the free-fit RMSE,
#   well within fold-to-fold SE; we do not plot a free-fit horizontal
#   baseline because the dip would invite overinterpretation.
# - The non-identifiability means |F_cp| should not be quoted as a
#   separately meaningful biological force. The Fig 2 panel C
#   comparison establishes that an inter-pole term is required;
#   this panel shows that the f_pp / f_cp split inside the joint fit
#   is not separately recoverable from this data.

# %% [markdown]
# ## Fig S2. Forecast error vs horizon and per-cell path MSE

# %%
TOPOLOGIES_S2 = list(TOPOLOGY_DISPLAY.keys())
HORIZON_FRAMES_S2 = tuple(range(1, 31))
T_MAX_S2 = 60.0  # seconds

print(f"Fig S2: rollout LOOCV across {len(TOPOLOGIES_S2)} topologies, "
      f"0..{T_MAX_S2:.0f} s ({HORIZON_FRAMES_S2[-1]} frames at dt={DT:.0f} s)...")
rollout_results = {}
for topology in TOPOLOGIES_S2:
    cfg = make_traj_config(topology)
    print(f"  {topology}...", flush=True)
    rollout_results[topology] = rollout_cross_validate(
        cells, cfg, horizons=HORIZON_FRAMES_S2, deterministic=True,
    )

# Render
fig_s2, (ax_s2, ax_s2b) = plt.subplots(
    1, 2, figsize=(12.0, 4.2),
    gridspec_kw={"width_ratios": [1.0, 1.4]},
)
for topology in TOPOLOGIES_S2:
    info = TOPOLOGY_DISPLAY[topology]
    res = rollout_results[topology]
    horizons_s = res.horizons.astype(float) * DT  # seconds
    keep = horizons_s <= T_MAX_S2
    mean_curve = np.nanmean(res.horizon_ensemble_mse, axis=0)
    # Prepend (0, 0) so the curves start at the origin (every model
    # has zero ensemble MSE at horizon 0 by construction).
    xs_plot = np.concatenate([[0.0], horizons_s[keep]])
    ys_plot = np.concatenate([[0.0], mean_curve[keep]])
    lw = 1.8 if info["admissible"] else 1.4
    ax_s2.plot(xs_plot, ys_plot,
               ls=info["linestyle"], lw=lw,
               color=info["color"], label=info["label"])

ax_s2.set_xlim(0, T_MAX_S2)
ax_s2.set_xlabel("Forecast horizon (s)")
ax_s2.set_ylabel("From-NEB ensemble MSE (μm²)")
ax_s2.set_title("Held-out forecast error vs horizon",
                loc="left", fontsize=9)
ax_s2.legend(loc="upper left", frameon=False, fontsize=7)

# Inset: each curve minus the main-text short-range curve, so the
# topology winner at each horizon is visible despite the curves
# nearly coinciding on the absolute scale.
from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset_axes
ax_s2_inset = _inset_axes(
    ax_s2, width="46%", height="26%",
    bbox_to_anchor=(0.0, 0.18, 1.0, 1.0),
    bbox_transform=ax_s2.transAxes,
    loc="lower right", borderpad=0.5,
)
_canon_topo = "poles_and_chroms_enveloped"
_canon_curve = np.nanmean(rollout_results[_canon_topo].horizon_ensemble_mse, axis=0)
_canon_horizons_s = rollout_results[_canon_topo].horizons.astype(float) * DT
_inset_t_max = 40.0
_keep = _canon_horizons_s <= _inset_t_max
for topology in TOPOLOGIES_S2:
    if topology == _canon_topo:
        continue
    info = TOPOLOGY_DISPLAY[topology]
    res = rollout_results[topology]
    horizons_s = res.horizons.astype(float) * DT
    mean_curve = np.nanmean(res.horizon_ensemble_mse, axis=0)
    diff = mean_curve - _canon_curve
    xs_inset = np.concatenate([[0.0], horizons_s[_keep]])
    ys_inset = np.concatenate([[0.0], diff[_keep]])
    ax_s2_inset.plot(xs_inset, ys_inset,
                     ls=info["linestyle"], lw=1.2,
                     color=info["color"])
ax_s2_inset.axhline(0.0, color=TOPOLOGY_DISPLAY[_canon_topo]["color"],
                    lw=1.2, ls="-")
ax_s2_inset.set_xlim(0, _inset_t_max)
ax_s2_inset.set_xlabel("horizon (s)", fontsize=6.5, labelpad=1.5)
ax_s2_inset.set_ylabel("Δ MSE", fontsize=6.5, labelpad=1.5)
ax_s2_inset.tick_params(labelsize=6, pad=1.5)

# Panel B: per-cell grouped bars — one cluster of (n_topologies)
# bars per cell, cells sorted by their mean path-MSE across models
# (best cell on the left).  Truncated y-axis like main Fig 3 panel C.
percell_path_mse_s2 = {
    t: np.asarray(rollout_results[t].path_mse) for t in TOPOLOGIES_S2
}
n_cells_s2 = len(cells)
n_topo_s2 = len(TOPOLOGIES_S2)
mean_across_models = np.mean(
    np.stack([percell_path_mse_s2[t] for t in TOPOLOGIES_S2], axis=0),
    axis=0,
)
cell_order = np.argsort(mean_across_models)
cell_xs = np.arange(n_cells_s2, dtype=float)
total_group_width = 0.78
bar_width = total_group_width / n_topo_s2
all_bar_values = []
for ti, topology in enumerate(TOPOLOGIES_S2):
    info = TOPOLOGY_DISPLAY[topology]
    vals = percell_path_mse_s2[topology][cell_order]
    offsets = (ti - (n_topo_s2 - 1) / 2.0) * bar_width
    ax_s2b.bar(cell_xs + offsets, vals, width=bar_width * 0.92,
               color=info["color"], alpha=0.85, edgecolor="white",
               linewidth=0.4, label=info["label"].replace("\n", " "))
    all_bar_values.extend(vals.tolist())
all_bar_values = np.asarray(all_bar_values)
y_lo_s2b = float(all_bar_values.min()) * 0.85
y_hi_s2b = float(all_bar_values.max()) * 1.05
ax_s2b.set_ylim(y_lo_s2b, y_hi_s2b)
ax_s2b.set_xticks(cell_xs)
ax_s2b.set_xticklabels(
    [cells[i].cell_id.replace("rpe18_ctr_", "") for i in cell_order],
    fontsize=7,
)
ax_s2b.set_xlabel("Cell (sorted by mean path-MSE across models)")
ax_s2b.set_ylabel("LOO path MSE (μm²)")
ax_s2b.set_title("Per-cell, per-model path-MSE",
                 loc="left", fontsize=9)
ax_s2b.legend(loc="upper left", frameon=False, fontsize=7, ncol=2)

for ax_panel, label in [(ax_s2, "A"), (ax_s2b, "B")]:
    ax_panel.text(-0.13, 1.04, label, transform=ax_panel.transAxes,
                  fontsize=11, fontweight="bold", va="bottom", ha="left")

fig_s2.tight_layout()
save_figure(fig_s2, "figS2_forecast_horizon")
plt.show()

# %% [markdown]
# Fig S2. Forecast accuracy is similar across topologies, with cell-
# to-cell variation dominating model-to-model variation. (A) From-NEB
# ensemble MSE under deterministic rollout vs forecast horizon
# (0 to 60 s), for the same four topologies as in main Fig 3. The
# inset plots each curve minus the short-range curve so the cross-
# topology ordering at every horizon is visible despite near-overlap
# on the absolute scale. (B) Per-cell path MSE (LOOCV) for the same
# four topologies, with cells sorted by mean path MSE across models.
#
# - Horizon range follows the manuscript anchor: "show held-out
#   forecast error vs horizon for up to 10 frames." With dt = 5 s,
#   the code computes horizons through 30 frames (150 s); the
#   displayed range is 0-60 s, slightly beyond the 10-frame (50 s)
#   anchor so the early-horizon ordering remains readable.
# - The deterministic rollout integrates the fitted drift field
#   forward from the real initial frame without stochastic noise;
#   D_x is not used as a noise source in this mode. Stochastic-
#   ensemble rollouts give the same mean by construction when D is
#   small and the force field is not strongly curved (see Fig 3
#   bullets), at the cost of Monte Carlo variance.
# - The relative-difference inset reveals that even where curves
#   visually overlap on the absolute MSE scale, one topology is
#   consistently lower than the others at every horizon. This is the
#   horizon-resolved version of the path MSE selection in main
#   Fig 3 panel C.
# - Panel B shows that within a single cell, the four topologies
#   produce path MSE values that are close to each other relative to
#   the spread across cells. Cell-level structure (initial pole
#   geometry, chromosome distribution) dominates the topology choice.


# %% [markdown]
# ## Fig S3. Sensitivity to basis size, smoothness penalty, and Itô vs Stratonovich

# %%
ROLLOUT_HORIZONS_S3 = (1, 5, 10, 20)

# (a) Kernel-vs-hyperparameter sweep on the
# poles_and_chroms_enveloped topology.  Each hyperparameter is varied
# independently with the other fixed at canonical so each panel
# isolates the effect of one parameter; the bias-variance tradeoff
# is shown in kernel space rather than trajectory space.
S3A_NBASIS_SWEEP = [4, N_BASIS_TRAJ, 30]
S3A_LAMBDA_SWEEP = [0.1, LAMBDA_ROUGH, 10.0]

print(f"Fig S3 (a) col 1: n_basis sweep at lambda_rough={LAMBDA_ROUGH:.0e} "
      f"({len(S3A_NBASIS_SWEEP)} fits)...")
s3a_models_nb = {}
for nb in S3A_NBASIS_SWEEP:
    cfg = make_traj_config(
        "poles_and_chroms_enveloped",
        n_basis_xx=nb, n_basis_xy=nb,
        lambda_rough=LAMBDA_ROUGH,
    )
    print(f"  n_basis={nb}", flush=True)
    s3a_models_nb[nb] = fit_model(cells, cfg)

print(f"Fig S3 (a) col 2: lambda_rough sweep at n_basis={N_BASIS_TRAJ} "
      f"({len(S3A_LAMBDA_SWEEP)} fits)...")
s3a_models_la = {}
for la in S3A_LAMBDA_SWEEP:
    cfg = make_traj_config(
        "poles_and_chroms_enveloped",
        n_basis_xx=N_BASIS_TRAJ, n_basis_xy=N_BASIS_TRAJ,
        lambda_rough=la,
    )
    print(f"  lambda_rough={la:.0e}", flush=True)
    s3a_models_la[la] = fit_model(cells, cfg)

# (b) Estimator-mode bars (Itô vs Stratonovich only) and pooled kernel
# fits under each convention for panels (c, d).  The Stratonovich fit uses
# the midpoint-current estimator with the SFI D * div(feature) force
# correction; because D is residual-estimated and partly reflects
# localization/tracking noise, these panels are sensitivity diagnostics.
ESTIMATOR_MODES = ["ito", "strato"]
MODE_DISPLAY = {
    "ito":    {"label": "Itô",          "color": OKABE_ITO["blue"]},
    "strato": {"label": "Stratonovich", "color": OKABE_ITO["orange"]},
}
print("Fig S3 (b): rerunning rollout for Itô / Stratonovich...")
mode_results = {}
mode_models = {}
for mode in ESTIMATOR_MODES:
    cfg = make_traj_config("poles_and_chroms_enveloped", basis_eval_mode=mode)
    mode_results[mode] = rollout_cross_validate(
        cells, cfg, horizons=ROLLOUT_HORIZONS_S3, deterministic=True,
    )
    mode_models[mode] = fit_model(cells, cfg)

# Render — layout: 2x3 grid where left and middle columns hold the
# 2x2 sweep block A (xy on top, xx on bottom; n_basis on left,
# lambda_rough on the middle), and the right column stacks the
# Itô-vs-Stratonovich kernels (C: f_xy on top, D: f_xx on bottom),
# matching the vertical layout of the sweep panels.  Panel B
# (calculus-convention bars) is folded into C as a small inset to
# save space.
import matplotlib as _mpl
fig_s3 = plt.figure(figsize=(12.0, 7.2), constrained_layout=True)
gs = fig_s3.add_gridspec(
    2, 3,
    height_ratios=[1.0, 1.0],
    width_ratios=[1.0, 1.0, 1.0],
)

ax_xy_nb = fig_s3.add_subplot(gs[0, 0])
ax_xy_la = fig_s3.add_subplot(gs[0, 1])
ax_xx_nb = fig_s3.add_subplot(gs[1, 0])
ax_xx_la = fig_s3.add_subplot(gs[1, 1])
ax_c = fig_s3.add_subplot(gs[0, 2])
ax_d = fig_s3.add_subplot(gs[1, 2])

F_XY_PLOT_MAX_S3 = 12.0
r_xy_eval_s3 = np.linspace(R_MIN, F_XY_PLOT_MAX_S3, 250)
r_xx_eval_s3 = np.linspace(R_XX_MIN_PLOT, R_MAX, 300)

# Sequential viridis: low parameter -> dark, high parameter -> light;
# main-text value highlighted by linewidth + bold solid line, NOT a
# different color (so the colormap itself unambiguously conveys
# parameter ordering).
_cmap_sweep = _mpl.colormaps["viridis"]


def _sweep_styles(values, canonical):
    """Return list of (value, color, lw, ls, zorder) for an N-value sweep."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    out = []
    for i, v in enumerate(sorted_vals):
        t = i / max(n - 1, 1)
        t = 0.15 + 0.70 * t
        color = _cmap_sweep(t)
        if v == canonical:
            out.append((v, color, 2.6, "-", 10))
        else:
            out.append((v, color, 1.4, "--", 3))
    return out


for v, color, lw, ls, zo in _sweep_styles(S3A_NBASIS_SWEEP, N_BASIS_TRAJ):
    fxy = s3a_models_nb[v].evaluate_kernel("xy", r_xy_eval_s3)
    fxx = s3a_models_nb[v].evaluate_kernel("xx", r_xx_eval_s3)
    label = (f"$n_\\mathrm{{basis}}={v}$ (main text)" if v == N_BASIS_TRAJ
             else f"$n_\\mathrm{{basis}}={v}$")
    ax_xy_nb.plot(r_xy_eval_s3, fxy, color=color, lw=lw, ls=ls,
                  label=label, alpha=0.95, zorder=zo)
    ax_xx_nb.plot(r_xx_eval_s3, fxx, color=color, lw=lw, ls=ls,
                  alpha=0.95, zorder=zo)

for v, color, lw, ls, zo in _sweep_styles(S3A_LAMBDA_SWEEP, LAMBDA_ROUGH):
    fxy = s3a_models_la[v].evaluate_kernel("xy", r_xy_eval_s3)
    fxx = s3a_models_la[v].evaluate_kernel("xx", r_xx_eval_s3)
    label = (rf"$\lambda_\mathrm{{rough}}={v:g}$ (main text)" if v == LAMBDA_ROUGH
             else rf"$\lambda_\mathrm{{rough}}={v:g}$")
    ax_xy_la.plot(r_xy_eval_s3, fxy, color=color, lw=lw, ls=ls,
                  label=label, alpha=0.95, zorder=zo)
    ax_xx_la.plot(r_xx_eval_s3, fxx, color=color, lw=lw, ls=ls,
                  alpha=0.95, zorder=zo)

for ax in (ax_xy_nb, ax_xy_la):
    ax.axhline(0, color="0.5", lw=0.5, ls="--")
    ax.set_xlim(R_MIN, F_XY_PLOT_MAX_S3)
for ax in (ax_xx_nb, ax_xx_la):
    ax.axhline(0, color="0.5", lw=0.5, ls="--")
    ax.set_xlim(R_XX_MIN_PLOT, R_MAX)

# Tighten y-axes per row using the CANONICAL (main-text) curve's
# range as the reference, padded by ~40 %.  This keeps all three
# sweep curves in view (off-canonical curves stay inside this window
# almost everywhere) while dropping the small-r blow-ups of extreme
# sweep values, which would otherwise compress the visible range.
def _canonical_clip(canonical_curve, pad_frac=0.40):
    arr = np.asarray(canonical_curve)
    arr = arr[np.isfinite(arr)]
    lo = float(arr.min())
    hi = float(arr.max())
    pad = pad_frac * max(hi - lo, 1e-9)
    return lo - pad, hi + pad

_canon_xy = s3a_models_nb[N_BASIS_TRAJ].evaluate_kernel("xy", r_xy_eval_s3)
_canon_xx = s3a_models_nb[N_BASIS_TRAJ].evaluate_kernel("xx", r_xx_eval_s3)
_xy_lo, _xy_hi = _canonical_clip(_canon_xy)
_xx_lo, _xx_hi = _canonical_clip(_canon_xx)
for ax in (ax_xy_nb, ax_xy_la):
    ax.set_ylim(_xy_lo, _xy_hi)
for ax in (ax_xx_nb, ax_xx_la):
    ax.set_ylim(_xx_lo, _xx_hi)

ax_xy_nb.set_title(rf"$n_\mathrm{{basis}}$ sweep "
                    rf"($\lambda_\mathrm{{rough}}={LAMBDA_ROUGH:g}$)",
                    loc="left", fontsize=10)
ax_xy_la.set_title(rf"$\lambda_\mathrm{{rough}}$ sweep "
                    rf"($n_\mathrm{{basis}}={N_BASIS_TRAJ}$)",
                    loc="left", fontsize=10)
for _ax in (ax_xy_nb, ax_xy_la):
    _ax.set_ylabel("$f_{xy}(r)$  (μm/s)\n+ attractive · - repulsive",
                   fontsize=9)
for _ax in (ax_xx_nb, ax_xx_la):
    _ax.set_ylabel("$f_{xx}(r)$  (μm/s)\n+ attractive · - repulsive",
                   fontsize=9)
ax_xx_nb.set_xlabel("Chromosome-chromosome distance (μm)", fontsize=9)
ax_xx_la.set_xlabel("Chromosome-chromosome distance (μm)", fontsize=9)
for _ax in (ax_xy_nb, ax_xy_la, ax_xx_nb, ax_xx_la):
    _ax.tick_params(labelsize=8)
ax_xy_nb.legend(loc="best", frameon=False, fontsize=8)
ax_xy_la.legend(loc="best", frameon=False, fontsize=8)

# Panel B: Itô vs Stratonovich f_xy (with calculus-convention bar
# inset in the bottom-right corner)
for mode in ESTIMATOR_MODES:
    info = MODE_DISPLAY[mode]
    fxy = mode_models[mode].evaluate_kernel("xy", r_xy_eval_s3)
    ax_c.plot(r_xy_eval_s3, fxy, color=info["color"], lw=1.8, label=info["label"])
ax_c.axhline(0, color="0.5", lw=0.6, ls="--")
ax_c.set_xlim(R_MIN, F_XY_PLOT_MAX_S3)
ax_c.set_xlabel("Distance to partner (μm)", fontsize=9)
ax_c.set_ylabel("$f_{xy}(r)$  (μm/s) "
                "\n+ attractive · - repulsive", fontsize=9)
ax_c.set_title("Learned $f_{xy}$ kernel",
               loc="left", fontsize=10)
ax_c.tick_params(labelsize=8)
ax_c.legend(loc="upper left", frameon=False, fontsize=8)

mode_means = [float(np.nanmean(mode_results[m].path_mse)) for m in ESTIMATOR_MODES]
mode_se = [
    float(np.nanstd(mode_results[m].path_mse, ddof=1)
          / np.sqrt(np.sum(np.isfinite(mode_results[m].path_mse))))
    for m in ESTIMATOR_MODES
]
from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset_axes_s3
ax_b_inset = _inset_axes_s3(
    ax_c, width="36%", height="34%",
    bbox_to_anchor=(0.0, 0.05, 1.0, 1.0),
    bbox_transform=ax_c.transAxes,
    loc="lower right", borderpad=0.5,
)
ax_b_inset.bar(np.arange(len(ESTIMATOR_MODES)), mode_means, yerr=mode_se,
               capsize=2.5, color=[MODE_DISPLAY[m]["color"] for m in ESTIMATOR_MODES],
               width=0.55, edgecolor="white", linewidth=0.4)
ax_b_inset.set_xticks(np.arange(len(ESTIMATOR_MODES)))
ax_b_inset.set_xticklabels([MODE_DISPLAY[m]["label"] for m in ESTIMATOR_MODES],
                            fontsize=6)
ax_b_inset.set_ylabel("path MSE", fontsize=6, labelpad=1.5)
ax_b_inset.tick_params(labelsize=6, pad=1.5)
ax_b_inset.set_xlim(-0.6, len(ESTIMATOR_MODES) - 0.4)

# Panel C: Itô vs Stratonovich f_xx
for mode in ESTIMATOR_MODES:
    info = MODE_DISPLAY[mode]
    fxx = mode_models[mode].evaluate_kernel("xx", r_xx_eval_s3)
    ax_d.plot(r_xx_eval_s3, fxx, color=info["color"], lw=1.8, label=info["label"])
ax_d.axhline(0, color="0.5", lw=0.6, ls="--")
ax_d.set_xlim(R_XX_MIN_PLOT, R_MAX)
# Clip the y-axis so a Stratonovich-side blow-up at small r doesn't
# distort the visible kernel comparison.  Use the Itô curve's range
# (with a small pad) as the reference floor.
_d_ito = mode_models["ito"].evaluate_kernel("xx", r_xx_eval_s3)
_d_lo = float(np.nanmin(_d_ito))
_d_hi = float(np.nanmax(_d_ito))
_d_pad = 0.4 * (_d_hi - _d_lo + 1e-12)
ax_d.set_ylim(_d_lo - _d_pad, _d_hi + _d_pad)
ax_d.set_xlabel("Chromosome-chromosome distance (μm)", fontsize=9)
ax_d.set_ylabel("$f_{xx}(r)$  (μm/s) "
                "\n+ attractive · - repulsive", fontsize=9)
ax_d.set_title("Learned $f_{xx}$ kernel (short range)",
               loc="left", fontsize=10)
ax_d.tick_params(labelsize=8)
ax_d.legend(loc="best", frameon=False, fontsize=8)

ax_xy_nb.text(-0.18, 1.06, "A", transform=ax_xy_nb.transAxes,
              fontsize=11, fontweight="bold", va="bottom", ha="left")
for ax_panel, label in zip([ax_c, ax_d], ["B", "C"]):
    ax_panel.text(-0.13, 1.04, label, transform=ax_panel.transAxes,
                  fontsize=11, fontweight="bold", va="bottom", ha="left")

save_figure(fig_s3, "figS3_hyperparam_sensitivity")
plt.show()

# %% [markdown]
# Fig S3. The selected short-range force kernels are stable to the
# main fitting choices: basis size, smoothness penalty, and Itô vs
# Stratonovich estimator.
# (A) Independent sweeps of n_basis in {4, 10, 30} (left column) and
# lambda_rough in {0.1, 1, 10} (right column) on the
# poles_and_chroms_enveloped topology, with the other parameter held
# at its main-text value. Top row is the chromosome-to-pole kernel
# f_xy(r); bottom row is the chromosome-chromosome kernel f_xx(r).
# A sequential viridis colormap encodes parameter ordering (low to
# high, dark to light); the main-text value is drawn as a bold solid
# line. (B) f_xy(r) from the pooled fit under Ito vs Stratonovich
# estimators; the inset shows held-out path MSE under each convention.
# (C) The corresponding f_xx(r) under the two conventions.
#
# - Sweep choice: each hyperparameter is varied independently rather
#   than over a 2D grid so each panel isolates the effect of one
#   parameter. The bias-variance tradeoff is shown in kernel space
#   (the object we care about) rather than in trajectory space.
# - n_basis = 10 and lambda_rough = 1 are the main-text values; the
#   sweeps span a decade above and below in lambda_rough and a
#   meaningful range below and above in n_basis.
# - lambda_ridge is fixed at 1e-6 throughout (numerical jitter, not a
#   meaningful regularizer in this project): we are not interpreting
#   individual basis coefficients or seeking sparsity, so a coefficient-
#   norm penalty has no physical role.
# - Stratonovich estimator: midpoint-current with the SFI D times
#   div(feature) force correction. Because D is residual-estimated and
#   partly reflects localization and tracking noise, these panels are
#   sensitivity diagnostics, not a definitive Ito-vs-Stratonovich
#   resolution. The qualitative shape and sign of f_xy and f_xx are
#   preserved across conventions.
# - y-axis clipping: the main-text-curve range with ~40 % padding is
#   used for the sweep panels so the visible window is dominated by
#   the main-text fit, not by extreme-sweep small-r blow-ups; the
#   Stratonovich f_xx panel is similarly clipped to the Ito range to
#   avoid a small-r blow-up dominating the comparison.
# - Across these one-at-a-time sweeps the visible sensitivity is
#   mainly in kernel shape; held-out path MSE changes little, which
#   is the basis for calling the selected fit stable.


# %% [markdown]
# ## Fig S4. Per-cell kernels for the selected topology

# %%
print("Fig S4: pooled fit + bootstrap...")
config_s4 = make_traj_config("poles_and_chroms_enveloped")
pooled_model = fit_model(cells, config_s4)
boot = bootstrap_kernels(cells, config_s4, n_boot=100, rng=np.random.default_rng(42))

print("Fig S4: per-cell fits...")
percell_models = [fit_model([cell], config_s4) for cell in cells]

r_xy_eval = np.linspace(R_MIN, R_MAX, 200)
r_xx_eval = np.linspace(R_MIN, R_MAX, 200)
phi_xy = pooled_model.basis_xy.evaluate(r_xy_eval)

# Bootstrap CIs (use evaluate_kernel for xx so the envelope multiplier is included).
boot_xy_samples = boot.theta_samples[:, pooled_model.n_basis_xx:]
boot_xy_curves = phi_xy @ boot_xy_samples.T
xy_ci_lo = np.percentile(boot_xy_curves, 5, axis=1)
xy_ci_hi = np.percentile(boot_xy_curves, 95, axis=1)

xx_basis = pooled_model.basis_xx
boot_xx_samples = boot.theta_samples[:, : pooled_model.n_basis_xx]
xx_curves_boot = np.stack([xx_basis.evaluate(r_xx_eval) @ s for s in boot_xx_samples], axis=1)
xx_ci_lo = np.percentile(xx_curves_boot, 5, axis=1)
xx_ci_hi = np.percentile(xx_curves_boot, 95, axis=1)

percell_xy_curves = np.stack(
    [m.basis_xy.evaluate(r_xy_eval) @ m.theta_xy for m in percell_models], axis=0
)
percell_xx_curves = np.stack(
    [m.evaluate_kernel("xx", r_xx_eval) for m in percell_models], axis=0
)
median_xy = np.median(percell_xy_curves, axis=0)
median_xx = np.median(percell_xx_curves, axis=0)

fig_s4, axes_s4 = plt.subplots(1, 2, figsize=(9.5, 3.6))

ax = axes_s4[0]
ax.fill_between(r_xy_eval, xy_ci_lo, xy_ci_hi, color=OKABE_ITO["vermil"],
                alpha=0.15, label="pooled 5–95 % CI")
for curve in percell_xy_curves:
    ax.plot(r_xy_eval, curve, color="0.45", lw=0.7, alpha=0.55)
ax.plot(r_xy_eval, median_xy, color="0.15", lw=1.6, ls="--",
        label="per-cell median")
ax.plot(r_xy_eval, phi_xy @ pooled_model.theta_xy,
        color=OKABE_ITO["vermil"], lw=2.0, label="pooled fit")
ax.axhline(0, color="0.5", lw=0.6, ls="--")
ax.set_xlim(R_MIN, 12.0)
ax.set_xlabel("Distance to partner (μm)")
ax.set_ylabel("$f_{xy}(r)$  (μm/s) "
              "\n+ attractive · - repulsive")
ax.set_title("Per-cell chromosome-to-partner kernel",
             loc="left", fontsize=8.5)
ax.set_ylim(-0.04, 0.04)
ax.legend(loc="best", frameon=False, fontsize=6.5)

ax = axes_s4[1]
ax.fill_between(r_xx_eval, xx_ci_lo, xx_ci_hi, color=OKABE_ITO["vermil"],
                alpha=0.15, label="pooled 5–95 % CI")
for curve in percell_xx_curves:
    ax.plot(r_xx_eval, curve, color="0.45", lw=0.7, alpha=0.55)
ax.plot(r_xx_eval, median_xx, color="0.15", lw=1.6, ls="--",
        label="per-cell median")
_pooled_xx_curve = pooled_model.evaluate_kernel("xx", r_xx_eval)
ax.plot(r_xx_eval, _pooled_xx_curve,
        color=OKABE_ITO["vermil"], lw=2.0, label="pooled fit")
ax.axhline(0, color="0.5", lw=0.6, ls="--")
ax.set_xlim(R_XX_MIN_PLOT, r_xx_eval[-1])
ax.set_ylim(-0.004, 0.001)
ax.set_xlabel("Chromosome-chromosome distance (μm)")
ax.set_ylabel("$f_{xx}(r)$  (μm/s) "
              "\n+ attractive · - repulsive")
ax.set_title("Per-cell short-range chromosome-chromosome kernel",
             loc="left", fontsize=8.5)
ax.legend(loc="best", frameon=False, fontsize=6.5)

# Panel letters anchored above each subplot, matching main-text style
for ax_panel, label in zip(axes_s4, ["A", "B"]):
    ax_panel.text(-0.13, 1.04, label, transform=ax_panel.transAxes,
                  fontsize=11, fontweight="bold", va="bottom", ha="left")

fig_s4.tight_layout()
save_figure(fig_s4, "figS4_percell_kernels")
plt.show()

# %% [markdown]
# Fig S4. Cell-to-cell variability in the learned force kernels.
# Per-cell f_xy(r) (left) and f_xx(r) (right) from individual fits of
# the selected short-range topology, one line per cell, overlaid on
# the pooled bootstrap 5-95 % CI band. The f_xx panel is truncated at
# the 1%-quantile of observed chromosome-chromosome distances so that
# only the data-supported short-range portion of the kernel is shown.
#
# - Per-cell fits are computed by passing a single cell to `fit_model`
#   with the same `FitConfig` as the pooled fit. The pooled bootstrap
#   resamples cells; per-cell deviations smaller than the bootstrap
#   band are broadly consistent with cell-level resampling
#   variability.
# - Selected short-range topology, matching Fig 3, with the same
#   steric envelope: r0 = 1.5 um, w = 0.3 um.
# - The shape of f_xy is conserved across cells (attractive at long
#   range, vanishing or weakly repulsive at short range), with the
#   amplitude varying cell to cell.
# - For f_xx, the short-range repulsion dominates as expected from
#   the steric envelope; cell-to-cell variation is concentrated near
#   the contact regime where data density per cell is small.
# - The bootstrap band on the pooled fit reflects the cell-level
#   resampling distribution and is the preferred uncertainty summary.


# %% [markdown]
# ## Fig S5. Drift-vs-diffusion sensitivity and characteristic scales

# %%
print("Fig S5: drift-vs-diffusion sensitivity sweep...")
from chromlearn.model_fitting.diffusion import (
    COORDINATE_MAPS as _COORD_MAPS,
    _predicted_forces_all_t as _pred_F_all,
    estimate_diffusion_variable,
)
from chromlearn.model_fitting.basis import BSplineBasis as _BSplineBasis

config_s5 = make_traj_config("poles_and_chroms_enveloped")
model_s5 = fit_model(cells, config_s5)
N_BASIS_D_S5 = 8
R_MIN_D_S5 = 0.5
R_MAX_D_S5 = 12.0
D_pooled_s5 = estimate_diffusion_variable(
    cells,
    basis_D=_BSplineBasis(R_MIN_D_S5, R_MAX_D_S5, N_BASIS_D_S5),
    coord_name="distance",
    dt=config_s5.dt,
    mode="f_corrected",
    lambda_ridge=LAMBDA_RIDGE,
    topology=model_s5.topology,
    fit_result=model_s5,
    basis_xx=model_s5.basis_xx,
    basis_xy=model_s5.basis_xy,
)

T_obs_per_cell_s5 = np.array(
    [(c.chromosomes.shape[0] - 1) * config_s5.dt for c in cells]
)
T_OBS_MED_S5 = float(np.median(T_obs_per_cell_s5))

_coord_fn_s5 = _COORD_MAPS["distance"]
all_distances_s5 = np.concatenate(
    [_coord_fn_s5(c.chromosomes, c).ravel() for c in cells]
)
all_distances_s5 = all_distances_s5[np.isfinite(all_distances_s5)]
EVAL_LO_S5, EVAL_HI_S5 = np.quantile(all_distances_s5, [0.01, 0.99])
EVAL_LO_S5 = max(EVAL_LO_S5, R_MIN_D_S5)
EVAL_HI_S5 = min(EVAL_HI_S5, R_MAX_D_S5)

# Compute per-state F, D, d once (batched per cell — same vectorization
# as Fig 4 in 00_main_figure.py).
fmag_chunks_s5, d_chunks_s5, D_chunks_s5 = [], [], []
for cell in cells:
    coord_arr = _coord_fn_s5(cell.chromosomes, cell)  # (T, N)
    F_all = _pred_F_all(cell, fit_result=model_s5,
                         basis_xx=model_s5.basis_xx,
                         basis_xy=model_s5.basis_xy,
                         topology=model_s5.topology)  # (T-1, N, 3)
    Fmag_all = np.linalg.norm(F_all, axis=2)  # (T-1, N)
    d_all = coord_arr[: F_all.shape[0]]
    valid = np.isfinite(Fmag_all) & np.isfinite(d_all)
    if valid.any():
        fmag_chunks_s5.append(Fmag_all[valid])
        d_chunks_s5.append(d_all[valid])
        D_chunks_s5.append(D_pooled_s5.evaluate(d_all[valid]))

force_mags_s5 = np.concatenate(fmag_chunks_s5)
d_force_s5 = np.concatenate(d_chunks_s5)
D_at_obs_s5 = np.concatenate(D_chunks_s5)
ok_s5 = (np.isfinite(force_mags_s5) & np.isfinite(D_at_obs_s5)
         & (D_at_obs_s5 > 0))

# Bins
N_BINS_S5 = 18
bin_edges_s5 = np.linspace(EVAL_LO_S5, EVAL_HI_S5, N_BINS_S5 + 1)
bin_centers_s5 = 0.5 * (bin_edges_s5[:-1] + bin_edges_s5[1:])
bin_idx_s5 = np.clip(
    np.digitize(d_force_s5, bin_edges_s5) - 1, 0, N_BINS_S5 - 1
)


def _binmedian_s5(values, idx, n_bins, min_count=30):
    out_med = np.full(n_bins, np.nan)
    out_lo = np.full(n_bins, np.nan)
    out_hi = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = (idx == b) & np.isfinite(values)
        if int(mask.sum()) >= min_count:
            out_med[b] = float(np.median(values[mask]))
            out_lo[b] = float(np.quantile(values[mask], 0.25))
            out_hi[b] = float(np.quantile(values[mask], 0.75))
    return out_med, out_lo, out_hi


# Panel A: f_drift(d; T) for several T values.  Includes T_steady = 150 s
# (the main-text anchor, docx Result 1 steady-elongation phase) and the
# trimmed-window length T_obs (~475 s) for context.  Codex-recommended
# sweep: {50, 150, 300, 475} plus the per-step dt.
T_STEADY_S5 = 150.0
T_SWEEP = [config_s5.dt, 50.0, T_STEADY_S5, 300.0, T_OBS_MED_S5]
fdrift_curves_T = {}
for T in T_SWEEP:
    chi = np.full_like(force_mags_s5, np.nan)
    chi[ok_s5] = T * (force_mags_s5[ok_s5] ** 2) / (2.0 * D_at_obs_s5[ok_s5])
    fdrift = chi / (chi + 1.0)
    med, _, _ = _binmedian_s5(fdrift, bin_idx_s5, N_BINS_S5)
    fdrift_curves_T[T] = med

# Panel B: drift-vs-diffusion crossover length L*(d) = 2 D / |F|
# (μm).  L* is the length over which |F|·t equals sqrt(2 D t); above
# L*, drift wins; below L*, diffusion wins.
Lstar_s5 = np.full_like(force_mags_s5, np.nan)
Lstar_s5[ok_s5] = (2.0 * D_at_obs_s5[ok_s5]
                   / np.maximum(force_mags_s5[ok_s5], 1e-18))
Lstar_med, Lstar_lo, Lstar_hi = _binmedian_s5(Lstar_s5, bin_idx_s5, N_BINS_S5)
CHROM_SPACING_REF = 1.0  # μm — chromosome spacing reference

# Panel C: tau_50 = 2 D / |F|^2 (seconds)
tau50 = np.full_like(force_mags_s5, np.nan)
tau50[ok_s5] = (2.0 * D_at_obs_s5[ok_s5]
                / np.maximum(force_mags_s5[ok_s5] ** 2, 1e-18))
tau_med, tau_lo, tau_hi = _binmedian_s5(tau50, bin_idx_s5, N_BINS_S5)

# Render
fig_s5, axes_s5 = plt.subplots(1, 3, figsize=(13.5, 3.6),
                                gridspec_kw={"wspace": 0.35})

# (A) T-sweep
import matplotlib as _mpl
cmap_T = _mpl.colormaps["viridis"]
norm_T = _mpl.colors.LogNorm(vmin=min(T_SWEEP), vmax=max(T_SWEEP))
ax_a_s5 = axes_s5[0]
for T in T_SWEEP:
    color = cmap_T(norm_T(T))
    label = rf"$T={T:.0f}$ s"
    ax_a_s5.plot(bin_centers_s5, fdrift_curves_T[T], "o-", color=color,
                 lw=1.4, markersize=3.5, label=label)
ax_a_s5.axhline(0.5, color="0.5", lw=0.7, ls="--")
ax_a_s5.set_xlim(float(bin_centers_s5[0]), float(bin_centers_s5[-1]))
ax_a_s5.set_ylim(0.0, 1.0)
ax_a_s5.set_xlabel("Distance from spindle center, $d$ (μm)")
ax_a_s5.set_ylabel(r"$f_{\mathrm{drift}}(d;\,T)$")
ax_a_s5.set_title("Sensitivity to observation timescale $T$",
                   loc="left", fontsize=8.5)
ax_a_s5.legend(loc="best", frameon=False, fontsize=6.0)

# (B) Drift-vs-diffusion crossover length L*(d) = 2D/|F|
ax_b_s5 = axes_s5[1]
ax_b_s5.fill_between(bin_centers_s5, Lstar_lo, Lstar_hi,
                      color=OKABE_ITO["blue"], alpha=0.18, linewidth=0,
                      label="IQR")
ax_b_s5.plot(bin_centers_s5, Lstar_med, "o-", color=OKABE_ITO["blue"],
              lw=1.4, markersize=3.5, label="median")
ax_b_s5.axhline(CHROM_SPACING_REF, color="0.5", lw=0.7, ls="--",
                 label=rf"chrom spacing $\sim {CHROM_SPACING_REF:.0f}\,\mu$m")
ax_b_s5.set_xlim(float(bin_centers_s5[0]), float(bin_centers_s5[-1]))
ax_b_s5.set_ylim(bottom=0.0)
ax_b_s5.set_xlabel("Distance from spindle center, $d$ (μm)")
ax_b_s5.set_ylabel(r"$L^{*}(d) = 2D/|F|$ (μm)")
ax_b_s5.set_title("Drift-vs-diffusion crossover length",
                   loc="left", fontsize=8.5)
ax_b_s5.legend(loc="best", frameon=False, fontsize=6.5)

# (C) tau_50
ax_c_s5 = axes_s5[2]
ax_c_s5.fill_between(bin_centers_s5, tau_lo, tau_hi,
                      color=OKABE_ITO["green"], alpha=0.18, linewidth=0,
                      label="IQR")
ax_c_s5.plot(bin_centers_s5, tau_med, "o-", color=OKABE_ITO["green"],
              lw=1.4, markersize=3.5, label="median")
ax_c_s5.axhline(T_STEADY_S5, color="0.5", lw=0.7, ls="--",
                 label=rf"$T={T_STEADY_S5:.0f}$ s")
ax_c_s5.set_xlim(float(bin_centers_s5[0]), float(bin_centers_s5[-1]))
ax_c_s5.set_yscale("log")
ax_c_s5.set_xlabel("Distance from spindle center, $d$ (μm)")
ax_c_s5.set_ylabel(r"$\tau_{50}(d) = 2D/|F|^{2}$ (s)")
ax_c_s5.set_title("Local 50/50 drift-vs-diffusion timescale",
                   loc="left", fontsize=8.5)
ax_c_s5.legend(loc="best", frameon=False, fontsize=6.5)

# Panel letters
for ax_panel, label in zip(axes_s5, ["A", "B", "C"]):
    ax_panel.text(-0.13, 1.04, label, transform=ax_panel.transAxes,
                  fontsize=11, fontweight="bold", va="bottom", ha="left")

fig_s5.tight_layout()
save_figure(fig_s5, "figS5_drift_diffusion_sensitivity")
plt.show()

# %% [markdown]
# Fig S5. The drift-vs-diffusion split changes little across
# accumulation timescales and has interpretable spatial scales.
# (A) f_drift(d; T), the fraction of motion attributable to drift
# rather than diffusion (defined in main Fig 4 panel B), evaluated at
# several values of T: per-step dt, 50, 150, 300, and ~475 s (the
# median trimmed-window length T_obs). Curves are color-coded on a
# log-T viridis scale. The monotone shape and spatial ordering hold
# across T; only the 50/50 crossover position shifts. (B) Drift-vs-
# diffusion crossover length L*(d) = 2 D(d) / |F(d)|, the distance
# at which |F| t equals sqrt(2 D t); above L* drift dominates, below
# L* diffusion dominates. The dashed reference line marks the
# ~1 um chromosome-spacing scale. (C) Local 50/50 timescale
# tau_50(d) = 2 D(d) / |F(d)|^2, the time at which the drift force
# moves a chromosome a distance equal to its diffusive standard
# deviation. The dashed reference is T = 150 s, the value used in
# main Fig 4 panel B.
#
# - The T sweep includes the per-step dt = 5 s as a lower bound, the
#   main-text T = 150 s, and the per-cell trimmed-window length
#   T_obs ~ 475 s as an upper bound. The intermediate values 50 and
#   300 s bracket the main-text choice.
# - L* and tau_50 give the same drift-vs-diffusion comparison as
#   f_drift, in raw length and time units rather than as a bounded
#   fraction. L* anchors the comparison to biological length scales
#   (chromosome spacing); tau_50 anchors it to the steady-elongation
#   duration.
# - L* is below ~1 um across most of the spindle interior, meaning
#   drift dominates already at sub-chromosome-spacing scales. This is
#   the spatial counterpart to f_drift > 0.5.
# - tau_50 is mostly below ~150 s, the main-text accumulation window.
#   This means a typical chromosome's drift signal becomes detectable
#   well within the steady-elongation phase even where f_drift is not
#   already close to 1.
# - All three panels are computed per chromosome per frame, then
#   median- and IQR-binned by spindle-center distance d, with bins
#   below 30 observations masked; this matches the estimator used in
#   main Fig 4. Panel A traces the bin-median curve for each T and
#   omits the IQR band to keep the sweep readable.
