# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 00b -- Supplement figures
#
# Lightweight assembler for the supplementary figures supporting Chris's
# part of *Hierarchy of spindle forces in prometaphase*.  Each figure is
# rendered as a self-contained PDF/PNG; Alex can combine into multi-panel
# supplement layouts as he prefers.
#
# Scope is deliberately tight -- four figures, each load-bearing for a
# specific main-text claim:
#
# Companion to **Fig 2** (CS-CS sufficient):
# - **S1** -- per-cell ``f_pp`` and ``f_cp`` kernels from the pp+cp
#   pole-velocity model, plus the constrained-share refit sweep showing
#   the ``(f_pp, f_cp)`` partition is non-identifiable.
#
# Companions to **Fig 3** (kernels + topology comparison):
# - **S2** -- held-out forecast error vs horizon (1-30 frames, from-NEB
#   ensemble MSE) for all five topologies.  Vertical mark at ``h=10``
#   is Alex's docx Result-3C anchor.
# - **S3** -- hyperparameter and convention sensitivity on the canonical
#   ``poles_and_chroms_enveloped`` topology: ``(n_basis, λ_rough)``
#   path-MSE heatmap, Itô vs Stratonovich bars, endpoint-method bars.
# - **S4** -- per-cell ``f_xy`` and ``f_xx`` kernel spaghetti for the
#   selected topology, over the pooled bootstrap CI band.
#
# Cuts vs the longer earlier draft and why:
#
# - Aggregate per-cell pp-only-vs-pp+cp strip plot: the dual-metric
#   headline bars are already in main-text Fig 2, and the per-cell
#   heterogeneity story is now told by Fig S1 (a, b) for a single
#   model rather than as a paired comparison.
# - 5-topology per-cell path-MSE strip plot: the horizon curve in
#   S2 already shows the admissible topologies tracking each other
#   closely across horizons; the per-cell breakdown adds little.
#   Paired foldwise Δ/SE on path MSE belongs in the methods table or
#   the S2 caption, not as a separate figure.
# - D-estimator robustness overlay: the ``f_corrected`` estimator is
#   justified in methods text because it explicitly subtracts the
#   fitted drift before estimating residual variance (Frishman &
#   Ronceray 2020 App. H).  Methods text alone carries the burden.

# %%
import os
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import block_diag
from scipy.optimize import minimize, minimize_scalar

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

# Topology display order and colors.  Admissible topologies use the
# Okabe-Ito core palette; nuisance upper-bound models use neutral grays.
TOPOLOGY_DISPLAY = {
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
    "center_and_chroms":           {"label": "center + chroms\n(free xx)",
                                    "color": "0.30",
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
# Mirrors NB04's canonical configuration so panels are consistent with
# the main figure.  ``cells_raw`` is preserved for the endpoint sweep in
# Fig S3 (c), which retrims from the raw catalog.

# %%
CONDITION = "rpe18_ctr"

# Trajectory model (chromosome dynamics; matches NB04 / NB07)
FRAC_NEB_AO = 0.4
N_BASIS_TRAJ = 10
LAMBDA_RIDGE = 1e-6
LAMBDA_ROUGH = 1.0
R_MIN = 0.3
R_MAX = 15.0
ENVELOPE_R0_XX = 1.5
ENVELOPE_W_XX = 0.3
DT = 5.0

# Pole-velocity model (used only by Fig S1; matches NB03 / NB03b)
N_BASIS_POLE = 6
LAMBDA_ROUGH_POLE = 1.0

cells_raw = load_condition(CONDITION)
cells = [trim_trajectory(c, method="neb_ao_frac", frac=FRAC_NEB_AO) for c in cells_raw]
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
# minimal NB03 / NB03b machinery needed for the per-cell kernel fits and
# the constrained-share refit sweep.

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


# Constrained-share helpers (NB03b ``constrained_share_sweep``).
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

    best = None
    best_meta = None
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
        if best is None or _scalar_objective(theta_try, G, Y, R_full) < _scalar_objective(best, G, Y, R_full):
            best, best_meta = theta_try, meta
        if meta["success"]:
            return theta_try, meta
    return best, best_meta


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
# ## Fig S1 -- Per-cell pp/cp kernels and partition non-identifiability
#
# Three subpanels in one figure (companion to Fig 2):
#
# - **(a)** per-cell ``f_pp(r)`` from the pp+cp pole-velocity model.
# - **(b)** per-cell ``f_cp(r)`` from the same fits.
# - **(c)** constrained-share refit sweep.  At each target cp share the
#   model is refit subject to the constraint and held-out LOOCV RMSE on
#   the spindle-separation observable is computed.  The flat curve over
#   the ``(α·f_pp, (1-α)·f_cp)`` family makes the partition
#   non-identifiability visceral.

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

# Free-fit LOOCV RMSE on the spindle-separation observable for the
# Fig S1 (c) reference line.  Same metric as the constrained-share
# sweep (LOOCV, separation observable, full pp+cp basis).
free_loocv_rmses = []
for i in range(len(cell_sep_rows)):
    train = [rows for j, rows in enumerate(cell_sep_rows) if j != i]
    test = [cell_sep_rows[i]]
    G_train, Y_train = stack_sep_rows(train)
    G_test, Y_test = stack_sep_rows(test)
    theta_free, _ = _ridge_fit(G_train, Y_train, R_full)
    free_loocv_rmses.append(float(np.sqrt(np.mean((Y_test - G_test @ theta_free) ** 2))))
free_loocv_rmse = float(np.mean(free_loocv_rmses))

# Render
fig_s1, axes_s1 = plt.subplots(1, 3, figsize=(11, 3.4))

ax = axes_s1[0]
for theta_cell in percell_thetas:
    ax.plot(r_pp_plot, phi_pp_plot @ theta_cell[:N_PP], color="0.55", lw=0.7, alpha=0.55)
ax.plot(r_pp_plot, phi_pp_plot @ theta_pooled[:N_PP], color=OKABE_ITO["blue"],
        lw=2.0, label="pooled fit")
ax.axhline(0, color="0.5", lw=0.6, ls="--")
ax.set_xlabel("Pole-pole distance (μm)")
ax.set_ylabel("$f_{pp}(r)$  (μm/s) "
              "\n+ attractive · - repulsive")
ax.set_title("Per-cell pole-pole kernel, pp+cp model",
             loc="left", fontsize=8.5)
ax.legend(loc="best", frameon=False, fontsize=6.5)

ax = axes_s1[1]
for theta_cell in percell_thetas:
    ax.plot(r_cp_plot, phi_cp_plot @ theta_cell[N_PP:], color="0.55", lw=0.7, alpha=0.55)
ax.plot(r_cp_plot, phi_cp_plot @ theta_pooled[N_PP:], color=OKABE_ITO["vermil"],
        lw=2.0, label="pooled fit")
ax.axhline(0, color="0.5", lw=0.6, ls="--")
ax.set_xlabel("Chromosome-to-pole distance (μm)")
ax.set_ylabel("$f_{cp}(r)$  (μm/s) "
              "\n+ attractive · - repulsive")
ax.set_title("Per-cell chromosome-to-pole kernel, pp+cp model",
             loc="left", fontsize=8.5)
ax.legend(loc="best", frameon=False, fontsize=6.5)

ax = axes_s1[2]
sweep_actual = np.array([row["mean_actual_share"] for row in loocv_rows])
sweep_rmse = np.array([row["rmse_mean"] for row in loocv_rows])
sweep_se = np.array([row["rmse_se"] for row in loocv_rows])
ax.fill_between(sweep_actual, sweep_rmse - sweep_se, sweep_rmse + sweep_se,
                color=OKABE_ITO["green"], alpha=0.22)
ax.plot(sweep_actual, sweep_rmse, "o-", color=OKABE_ITO["green"], lw=1.5,
        markersize=5, label="constrained-refit LOOCV RMSE")
ax.axhline(free_loocv_rmse, color="0.4", ls="--", lw=0.8,
           label=f"free pp+cp fit (LOOCV) = {free_loocv_rmse:.3f}")
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

# Figure-level note flagging that S1 is a pole-velocity regression with
# chromosomes as observed covariates, NOT the chromosome-trajectory
# force-inference model used in S2-S4.  Without this, a reader skimming
# kernel panels with similar palette could conflate the two model
# families.
fig_s1.suptitle(
    "pp+cp pole-velocity regression: chromosomes are observed covariates, "
    "not the chromosome-trajectory force inference of S2-S4",
    fontsize=8.0, color="0.3", y=1.02,
)
fig_s1.tight_layout()
save_figure(fig_s1, "figS1_pp_cp_kernels_partition")
plt.show()


# %% [markdown]
# ## Fig S2 -- Forecast error vs horizon, 1-30 frames (companion to Fig 3)
#
# Held-out from-NEB ensemble MSE (deterministic drift rollout) vs horizon
# ``h ∈ [1, 30]`` for all five topologies.  Admissible topologies plotted
# with solid lines in Okabe-Ito colors; nuisance upper-bound topologies
# dashed in gray.  Vertical reference at ``h = 10`` marks Alex's docx
# Result-3C anchor.

# %%
TOPOLOGIES_S2 = list(TOPOLOGY_DISPLAY.keys())
HORIZONS_S2 = tuple(range(1, 31))

print(f"Fig S2: rollout LOOCV across {len(TOPOLOGIES_S2)} topologies, "
      f"horizons 1..{HORIZONS_S2[-1]}...")
rollout_results = {}
for topology in TOPOLOGIES_S2:
    cfg = make_traj_config(topology)
    print(f"  {topology}...", flush=True)
    rollout_results[topology] = rollout_cross_validate(
        cells, cfg, horizons=HORIZONS_S2, deterministic=True,
    )

# Render
fig_s2, ax_s2 = plt.subplots(figsize=(6.8, 4.2))
for topology in TOPOLOGIES_S2:
    info = TOPOLOGY_DISPLAY[topology]
    res = rollout_results[topology]
    horizons = res.horizons.astype(float) * DT  # convert to seconds for the x-axis
    mean_curve = np.nanmean(res.horizon_ensemble_mse, axis=0)
    se_curve = np.nanstd(res.horizon_ensemble_mse, axis=0, ddof=1) \
        / np.sqrt(np.sum(np.isfinite(res.horizon_ensemble_mse), axis=0).clip(min=1))
    ls = "-" if info["admissible"] else "--"
    lw = 1.8 if info["admissible"] else 1.2
    ax_s2.fill_between(horizons, mean_curve - se_curve, mean_curve + se_curve,
                       color=info["color"], alpha=0.12)
    ax_s2.plot(horizons, mean_curve, ls=ls, lw=lw, color=info["color"],
               label=info["label"])

ax_s2.axvline(10 * DT, color="0.35", lw=0.8, ls=":")
ax_s2.text(10 * DT, ax_s2.get_ylim()[1] * 0.04, " h = 10",
           color="0.35", fontsize=7, va="bottom", ha="left")
ax_s2.set_xlabel("Horizon (s)")
ax_s2.set_ylabel("From-NEB ensemble MSE (μm²)")
ax_s2.set_title("Held-out forecast error vs horizon",
                loc="left", fontsize=8.5)
ax_s2.legend(loc="best", frameon=False, fontsize=6.5, ncol=1)

fig_s2.tight_layout()
save_figure(fig_s2, "figS2_forecast_horizon")
plt.show()


# %% [markdown]
# ## Fig S3 -- Hyperparameter and convention sensitivity (companion to Fig 3)
#
# Three subpanels, all in held-out path MSE on the canonical
# ``poles_and_chroms_enveloped`` topology to match the main-text metric:
#
# - **(a)** ``(n_basis × λ_rough)`` heatmap; selected operating point
#   marked.  These are the only two tuned hyperparameters; ``λ_ridge``
#   is fixed numerical jitter.
# - **(b)** Itô vs Stratonovich bars.
# - **(c)** Endpoint-method bars (``frac`` sweep + ``end_sep``).

# %%
N_BASIS_GRID = [6, 8, 10, 12, 16]
LAMBDA_ROUGH_GRID = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
ROLLOUT_HORIZONS_S3 = (1, 5, 10, 20)

# (a) Joint grid sweep.  Use rollout_cross_validate directly (not
# evaluate_all_loocv) since only path MSE is read; this drops the
# rolling-window forecast simulation that evaluate_all_loocv runs
# internally per fold.
def _grid_one(nb, lro):
    cfg = make_traj_config(
        "poles_and_chroms_enveloped",
        n_basis_xx=nb, n_basis_xy=nb, lambda_rough=lro,
    )
    rollout_res = rollout_cross_validate(
        cells, cfg,
        horizons=ROLLOUT_HORIZONS_S3,
        deterministic=True,
    )
    return (nb, lro), float(np.nanmean(rollout_res.path_mse))


grid_pairs = list(product(N_BASIS_GRID, LAMBDA_ROUGH_GRID))
print(f"Fig S3 (a): grid sweep over {len(grid_pairs)} configs (parallel)...")
n_workers = min(os.cpu_count() or 1, len(grid_pairs))
grid_outputs = Parallel(n_jobs=n_workers, verbose=5)(
    delayed(_grid_one)(nb, lro) for nb, lro in grid_pairs
)
grid_path_mse = {key: score for key, score in grid_outputs}

heatmap = np.full((len(LAMBDA_ROUGH_GRID), len(N_BASIS_GRID)), np.nan)
for j, nb in enumerate(N_BASIS_GRID):
    for i, lro in enumerate(LAMBDA_ROUGH_GRID):
        heatmap[i, j] = grid_path_mse[(nb, lro)]

best_key = min(grid_path_mse, key=grid_path_mse.get)
best_nb, best_lro = best_key
operating_point = (N_BASIS_TRAJ, LAMBDA_ROUGH)

# (b) Estimator-mode bars (Itô vs Stratonovich only)
ESTIMATOR_MODES = ["ito", "strato"]
print("Fig S3 (b): rerunning rollout for Itô / Stratonovich...")
mode_results = {}
for mode in ESTIMATOR_MODES:
    cfg = make_traj_config("poles_and_chroms_enveloped", basis_eval_mode=mode)
    mode_results[mode] = rollout_cross_validate(
        cells, cfg, horizons=ROLLOUT_HORIZONS_S3, deterministic=True,
    )

# (c) Endpoint-method bars.  Track cell counts per condition so the
# bar comparison cannot be misread as biology when it is partly cell
# attrition; the n-cells string is annotated under each bar below.
ENDPOINT_FRACS = [0.20, 0.25, 0.33, 0.40, 0.50]
print(f"Fig S3 (c): endpoint sweep over {len(ENDPOINT_FRACS)} frac values + end_sep...")
endpoint_results = {}
endpoint_n_cells = {}
for frac in ENDPOINT_FRACS:
    trimmed = []
    for raw in cells_raw:
        try:
            trimmed.append(trim_trajectory(raw, method="neb_ao_frac", frac=frac))
        except ValueError:
            pass
    label = f"frac={frac:.2f}"
    print(f"  {label}: {len(trimmed)}/{len(cells_raw)} cells")
    if len(trimmed) >= 3:
        cfg = make_traj_config(
            "poles_and_chroms_enveloped",
            endpoint_method="neb_ao_frac", endpoint_frac=frac,
        )
        endpoint_results[label] = rollout_cross_validate(
            trimmed, cfg, horizons=ROLLOUT_HORIZONS_S3, deterministic=True,
        )
        endpoint_n_cells[label] = len(trimmed)

trimmed_es = []
for raw in cells_raw:
    try:
        trimmed_es.append(trim_trajectory(raw, method="end_sep"))
    except ValueError:
        pass
print(f"  end_sep: {len(trimmed_es)}/{len(cells_raw)} cells")
if len(trimmed_es) >= 3:
    cfg_es = make_traj_config(
        "poles_and_chroms_enveloped",
        endpoint_method="end_sep",
    )
    endpoint_results["end_sep"] = rollout_cross_validate(
        trimmed_es, cfg_es, horizons=ROLLOUT_HORIZONS_S3, deterministic=True,
    )
    endpoint_n_cells["end_sep"] = len(trimmed_es)

# Render
fig_s3 = plt.figure(figsize=(12.5, 3.6))
gs = fig_s3.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.5])

ax_a = fig_s3.add_subplot(gs[0, 0])
im = ax_a.imshow(heatmap, origin="lower", aspect="auto", cmap="viridis")
ax_a.set_xticks(range(len(N_BASIS_GRID)))
ax_a.set_xticklabels([str(nb) for nb in N_BASIS_GRID])
ax_a.set_yticks(range(len(LAMBDA_ROUGH_GRID)))
ax_a.set_yticklabels([f"{lr:.0e}" for lr in LAMBDA_ROUGH_GRID])
ax_a.set_xlabel(r"$n_\mathrm{basis}$")
ax_a.set_ylabel(r"$\lambda_\mathrm{rough}$")
op_x = N_BASIS_GRID.index(operating_point[0])
op_y = LAMBDA_ROUGH_GRID.index(operating_point[1])
ax_a.plot(op_x, op_y, marker="o", color="white", markersize=10, mfc="none", mew=1.6)
ax_a.text(op_x + 0.12, op_y + 0.12, "main-text\noperating point",
          color="white", fontsize=7, va="bottom", ha="left")
best_x = N_BASIS_GRID.index(best_nb)
best_y = LAMBDA_ROUGH_GRID.index(best_lro)
ax_a.plot(best_x, best_y, marker="*", color="white", markersize=10, mew=0)
cbar = fig_s3.colorbar(im, ax=ax_a, shrink=0.9)
cbar.set_label("path MSE (μm²)")
ax_a.set_title("Hyperparameter sensitivity",
               loc="left", fontsize=8.5)

ax_b = fig_s3.add_subplot(gs[0, 1])
mode_means = [float(np.nanmean(mode_results[m].path_mse)) for m in ESTIMATOR_MODES]
mode_se = [
    float(np.nanstd(mode_results[m].path_mse, ddof=1)
          / np.sqrt(np.sum(np.isfinite(mode_results[m].path_mse))))
    for m in ESTIMATOR_MODES
]
mode_labels = {"ito": "Itô", "strato": "Stratonovich"}
ax_b.bar(np.arange(len(ESTIMATOR_MODES)), mode_means, yerr=mode_se,
         capsize=3, color=[OKABE_ITO["blue"], OKABE_ITO["orange"]])
ax_b.set_xticks(np.arange(len(ESTIMATOR_MODES)))
ax_b.set_xticklabels([mode_labels[m] for m in ESTIMATOR_MODES])
ax_b.set_ylabel("path MSE (μm²)")
ax_b.set_title("Calculus convention",
               loc="left", fontsize=8.5)

ax_c = fig_s3.add_subplot(gs[0, 2])
ep_labels = list(endpoint_results.keys())
ep_means = [float(np.nanmean(endpoint_results[k].path_mse)) for k in ep_labels]
ep_se = [
    float(np.nanstd(endpoint_results[k].path_mse, ddof=1)
          / np.sqrt(np.sum(np.isfinite(endpoint_results[k].path_mse))))
    for k in ep_labels
]
colors = [
    OKABE_ITO["vermil"] if lbl == f"frac={FRAC_NEB_AO:.2f}" else "0.45"
    for lbl in ep_labels
]
ax_c.bar(np.arange(len(ep_labels)), ep_means, yerr=ep_se, capsize=3, color=colors)
ax_c.set_xticks(np.arange(len(ep_labels)))
ax_c.set_xticklabels(
    [f"{lbl}\nn={endpoint_n_cells[lbl]}" for lbl in ep_labels],
    rotation=25, ha="right",
)
ax_c.set_ylabel("path MSE (μm²)")
ax_c.set_title("Endpoint method",
               loc="left", fontsize=8.5)

# Panel letters anchored above each subplot, matching main-text style
for ax_panel, label in zip([ax_a, ax_b, ax_c], ["A", "B", "C"]):
    ax_panel.text(-0.13, 1.04, label, transform=ax_panel.transAxes,
                  fontsize=11, fontweight="bold", va="bottom", ha="left")

fig_s3.tight_layout()
save_figure(fig_s3, "figS3_hyperparam_sensitivity")
plt.show()


# %% [markdown]
# ## Fig S4 -- Per-cell kernels for the selected topology (companion to Fig 3)
#
# Per-cell ``f_xy(r)`` and ``f_xx(r)`` fits from the winning
# ``poles_and_chroms_enveloped`` topology, one line per cell, overlaid
# on the pooled bootstrap 5-95 % CI band.  Quantifies cell-to-cell
# heterogeneity in the selected model and answers whether the pooled
# kernel is driven by any single cell.

# %%
print("Fig S4: pooled fit + bootstrap...")
config_s4 = make_traj_config("poles_and_chroms_enveloped")
pooled_model = fit_model(cells, config_s4)
boot = bootstrap_kernels(cells, config_s4, n_boot=200, rng=np.random.default_rng(42))

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

fig_s4, axes_s4 = plt.subplots(1, 2, figsize=(9.5, 3.6))

ax = axes_s4[0]
ax.fill_between(r_xy_eval, xy_ci_lo, xy_ci_hi, color=OKABE_ITO["vermil"],
                alpha=0.15, label="pooled 5–95 % CI")
ax.plot(r_xy_eval, phi_xy @ pooled_model.theta_xy,
        color=OKABE_ITO["vermil"], lw=2.0, label="pooled fit")
for m in percell_models:
    ax.plot(r_xy_eval, m.basis_xy.evaluate(r_xy_eval) @ m.theta_xy,
            color="0.45", lw=0.7, alpha=0.55)
ax.axhline(0, color="0.5", lw=0.6, ls="--")
ax.set_xlabel("Distance to partner (μm)")
ax.set_ylabel("$f_{xy}(r)$  (μm/s) "
              "\n+ attractive · - repulsive")
ax.set_title("Per-cell chromosome-to-partner kernel",
             loc="left", fontsize=8.5)
ax.legend(loc="best", frameon=False, fontsize=6.5)

ax = axes_s4[1]
ax.fill_between(r_xx_eval, xx_ci_lo, xx_ci_hi, color=OKABE_ITO["vermil"],
                alpha=0.15, label="pooled 5–95 % CI")
ax.plot(r_xx_eval, pooled_model.evaluate_kernel("xx", r_xx_eval),
        color=OKABE_ITO["vermil"], lw=2.0, label="pooled fit")
for m in percell_models:
    ax.plot(r_xx_eval, m.evaluate_kernel("xx", r_xx_eval),
            color="0.45", lw=0.7, alpha=0.55)
ax.axhline(0, color="0.5", lw=0.6, ls="--")
ax.set_xlabel("Chromosome-chromosome distance (μm)")
ax.set_ylabel("$f_{xx}(r)$  (μm/s) "
              "\n+ attractive · - repulsive")
ax.set_title("Per-cell enveloped chromosome-chromosome kernel",
             loc="left", fontsize=8.5)
ax.legend(loc="best", frameon=False, fontsize=6.5)

# Panel letters anchored above each subplot, matching main-text style
for ax_panel, label in zip(axes_s4, ["A", "B"]):
    ax_panel.text(-0.13, 1.04, label, transform=ax_panel.transAxes,
                  fontsize=11, fontweight="bold", va="bottom", ha="left")

fig_s4.tight_layout()
save_figure(fig_s4, "figS4_percell_kernels")
plt.show()
