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
# - **S2** -- held-out forecast error vs horizon (0-60 s, from-NEB
#   ensemble MSE) for all five topologies.  Vertical mark at 25 s
#   (h = 5 frames) is the early-horizon regime where the enveloped
#   topology pulls ahead.
# - **S3** -- hyperparameter and convention sensitivity on the canonical
#   ``poles_and_chroms_enveloped`` topology: ``(n_basis, λ_rough)``
#   path-MSE heatmap, Itô vs Stratonovich path-MSE bars, plus the
#   pooled ``f_xy`` and ``f_xx`` kernels learned under each calculus
#   convention so the reader sees how the kernel itself shifts.
# - **S4** -- per-cell ``f_xy`` and ``f_xx`` kernel spaghetti for the
#   selected topology, over the pooled bootstrap CI band, with per-cell
#   median highlighted.
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
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

ax = axes_s1[0]
for curve in percell_pp_curves:
    ax.plot(r_pp_plot, curve, color="0.55", lw=0.7, alpha=0.55)
ax.plot(r_pp_plot, median_pp, color="0.15", lw=1.6, ls="--",
        label="per-cell median")
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
for curve in percell_cp_curves:
    ax.plot(r_cp_plot, curve, color="0.55", lw=0.7, alpha=0.55)
ax.plot(r_cp_plot, median_cp, color="0.15", lw=1.6, ls="--",
        label="per-cell median")
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
# ## Fig S2 -- Forecast error vs horizon, 0-60 s (companion to Fig 3)
#
# Held-out from-NEB ensemble MSE (deterministic drift rollout) vs horizon
# for all five topologies.  Plotted in seconds throughout.  Admissible
# topologies plotted with solid lines in Okabe-Ito colors; nuisance
# upper-bound topologies dashed in gray.  Vertical reference at 25 s
# (h = 5 frames) marks the early-horizon regime where the
# ``poles_and_chroms_enveloped`` topology pulls ahead of the simpler
# admissible variants.

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
fig_s2, ax_s2 = plt.subplots(figsize=(6.8, 4.2))
for topology in TOPOLOGIES_S2:
    info = TOPOLOGY_DISPLAY[topology]
    res = rollout_results[topology]
    horizons_s = res.horizons.astype(float) * DT  # seconds
    keep = horizons_s <= T_MAX_S2
    mean_curve = np.nanmean(res.horizon_ensemble_mse, axis=0)
    ls = "-" if info["admissible"] else "--"
    lw = 1.8 if info["admissible"] else 1.2
    ax_s2.plot(horizons_s[keep], mean_curve[keep], ls=ls, lw=lw,
               color=info["color"], label=info["label"])

ANCHOR_S = 5 * DT  # 25 s, h=5 frames
ax_s2.axvline(ANCHOR_S, color="0.35", lw=0.8, ls=":")
ax_s2.text(ANCHOR_S, ax_s2.get_ylim()[1] * 0.04, f" {ANCHOR_S:.0f} s",
           color="0.35", fontsize=7, va="bottom", ha="left")
ax_s2.set_xlim(0, T_MAX_S2)
ax_s2.set_xlabel("Forecast horizon (s)")
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
# - **(a)** Kernel-vs-hyperparameter sweep on the canonical
#   ``poles_and_chroms_enveloped`` topology.  Each hyperparameter is varied
#   independently with the other fixed at canonical so the column shows
#   what THAT hyperparameter does in isolation.  Left column:
#   ``n_basis`` sweep at fixed ``λ_rough``; right column: ``λ_rough``
#   sweep at fixed ``n_basis``.  Rows are ``f_xy(r)`` (top) and
#   ``f_xx(r)`` (bottom).  Held-out path MSE varies < 2% across this
#   range; the bias-variance tradeoff is shown in kernel space (where it
#   is visible) rather than trajectory space (where path-MSE smooths over
#   it).  Replaces an earlier path-MSE heatmap that, on the canonical
#   topology, was too low-pass to discriminate among kernels in this
#   regime — see ``hyperparameter_methodology_pickup.md``.
# - **(b)** Itô vs Stratonovich path-MSE bars (held-out).
# - **(c, d)** Learned ``f_xy(r)`` and ``f_xx(r)`` from the pooled fit
#   under the two calculus conventions, so the reader sees how the
#   kernel itself shifts alongside the small held-out path-MSE
#   difference in (b).

# %%
ROLLOUT_HORIZONS_S3 = (1, 5, 10, 20)

# (a) Kernel-vs-hyperparameter sweep on the canonical
# poles_and_chroms_enveloped topology.  Each hyperparameter is varied
# independently with the other fixed at canonical (codex review
# 2026-05-06: the original 2x4 regime layout conflated n_basis with
# lambda_rough; sweeping each one alone separates their bias-variance
# contributions cleanly).  Held-out path MSE varies < 2% across this
# range; the bias-variance tradeoff is shown in kernel space, not
# trajectory space (see methods).
S3A_NBASIS_SWEEP = [4, 6, 10, 16, 32, 64]
S3A_LAMBDA_SWEEP = [1e-4, 1e-2, 1.0, 1e2, 1e4]

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

# Render
import matplotlib as mpl

fig_s3 = plt.figure(figsize=(16.5, 7.2),
                     constrained_layout=True)
gs = fig_s3.add_gridspec(
    2, 4,
    height_ratios=[1.0, 1.0],
    width_ratios=[1.05, 1.05, 0.95, 0.95],
)

# Top-left 2x2 block: kernel-vs-hyperparameter sweep
ax_xy_nb = fig_s3.add_subplot(gs[0, 0])
ax_xy_la = fig_s3.add_subplot(gs[0, 1])
ax_xx_nb = fig_s3.add_subplot(gs[1, 0])
ax_xx_la = fig_s3.add_subplot(gs[1, 1])

r_kernel_eval = np.linspace(R_MIN, R_MAX, 300)

cmap_nb = mpl.colormaps["viridis"]
cmap_la = mpl.colormaps["plasma"]
norm_nb = mpl.colors.LogNorm(vmin=min(S3A_NBASIS_SWEEP),
                              vmax=max(S3A_NBASIS_SWEEP))
norm_la = mpl.colors.LogNorm(vmin=min(S3A_LAMBDA_SWEEP),
                              vmax=max(S3A_LAMBDA_SWEEP))

# Column 1: n_basis sweep at fixed lambda_rough
for nb in S3A_NBASIS_SWEEP:
    color = cmap_nb(norm_nb(nb))
    fxy = s3a_models_nb[nb].evaluate_kernel("xy", r_kernel_eval)
    fxx = s3a_models_nb[nb].evaluate_kernel("xx", r_kernel_eval)
    ax_xy_nb.plot(r_kernel_eval, fxy, color=color, lw=1.4, alpha=0.9)
    ax_xx_nb.plot(r_kernel_eval, fxx, color=color, lw=1.4, alpha=0.9)
ax_xy_nb.axhline(0, color="0.5", lw=0.5, ls="--")
ax_xx_nb.axhline(0, color="0.5", lw=0.5, ls="--")
ax_xy_nb.set_title(rf"$n_\mathrm{{basis}}$ sweep ($\lambda_\mathrm{{rough}}={LAMBDA_ROUGH:.0e}$)",
                    loc="left", fontsize=8.5)
ax_xx_nb.set_xlabel("Chromosome-chromosome distance (μm)")
ax_xy_nb.set_ylabel("$f_{xy}(r)$  (μm/s)\n+ attractive · - repulsive",
                     fontsize=8)
ax_xx_nb.set_ylabel("$f_{xx}(r)$  (μm/s)\n+ attractive · - repulsive",
                     fontsize=8)

# Column 2: lambda_rough sweep at fixed n_basis
for la in S3A_LAMBDA_SWEEP:
    color = cmap_la(norm_la(la))
    fxy = s3a_models_la[la].evaluate_kernel("xy", r_kernel_eval)
    fxx = s3a_models_la[la].evaluate_kernel("xx", r_kernel_eval)
    ax_xy_la.plot(r_kernel_eval, fxy, color=color, lw=1.4, alpha=0.9)
    ax_xx_la.plot(r_kernel_eval, fxx, color=color, lw=1.4, alpha=0.9)
ax_xy_la.axhline(0, color="0.5", lw=0.5, ls="--")
ax_xx_la.axhline(0, color="0.5", lw=0.5, ls="--")
ax_xy_la.set_title(rf"$\lambda_\mathrm{{rough}}$ sweep ($n_\mathrm{{basis}}={N_BASIS_TRAJ}$)",
                    loc="left", fontsize=8.5)
ax_xx_la.set_xlabel("Chromosome-chromosome distance (μm)")

# Share y within each kernel row so visual contrast is honest
for row_axes in [(ax_xy_nb, ax_xy_la), (ax_xx_nb, ax_xx_la)]:
    y_lo = min(ax.get_ylim()[0] for ax in row_axes)
    y_hi = max(ax.get_ylim()[1] for ax in row_axes)
    for ax in row_axes:
        ax.set_ylim(y_lo, y_hi)

# Colorbars
sm_nb = mpl.cm.ScalarMappable(cmap=cmap_nb, norm=norm_nb)
sm_nb.set_array([])
cb_nb = fig_s3.colorbar(sm_nb, ax=[ax_xy_nb, ax_xx_nb],
                         shrink=0.78, pad=0.02, location="left")
cb_nb.set_label(r"$n_\mathrm{basis}$", fontsize=8)

sm_la = mpl.cm.ScalarMappable(cmap=cmap_la, norm=norm_la)
sm_la.set_array([])
cb_la = fig_s3.colorbar(sm_la, ax=[ax_xy_la, ax_xx_la],
                         shrink=0.78, pad=0.02, location="right")
cb_la.set_label(r"$\lambda_\mathrm{rough}$", fontsize=8)

# Right side: panels B, C, D from the Itô vs Stratonovich comparison
ax_b = fig_s3.add_subplot(gs[0, 2])
mode_means = [float(np.nanmean(mode_results[m].path_mse)) for m in ESTIMATOR_MODES]
mode_se = [
    float(np.nanstd(mode_results[m].path_mse, ddof=1)
          / np.sqrt(np.sum(np.isfinite(mode_results[m].path_mse))))
    for m in ESTIMATOR_MODES
]
ax_b.bar(np.arange(len(ESTIMATOR_MODES)), mode_means, yerr=mode_se,
         capsize=3, color=[MODE_DISPLAY[m]["color"] for m in ESTIMATOR_MODES])
ax_b.set_xticks(np.arange(len(ESTIMATOR_MODES)))
ax_b.set_xticklabels([MODE_DISPLAY[m]["label"] for m in ESTIMATOR_MODES])
ax_b.set_ylabel("path MSE (μm²)")
ax_b.set_title("Calculus convention\n(held-out)",
               loc="left", fontsize=8.5)

# Pooled kernel fits under Itô vs Stratonovich, plotted on the full
# configured basis range so the small-r behavior is visible.
r_kernel_eval = np.linspace(R_MIN, R_MAX, 300)

ax_c = fig_s3.add_subplot(gs[1, 2])
for mode in ESTIMATOR_MODES:
    info = MODE_DISPLAY[mode]
    fxy = mode_models[mode].evaluate_kernel("xy", r_kernel_eval)
    ax_c.plot(r_kernel_eval, fxy, color=info["color"], lw=1.8, label=info["label"])
ax_c.axhline(0, color="0.5", lw=0.6, ls="--")
ax_c.set_xlabel("Distance to partner (μm)")
ax_c.set_ylabel("$f_{xy}(r)$  (μm/s) "
                "\n+ attractive · - repulsive")
ax_c.set_title("Learned $f_{xy}$ kernel",
               loc="left", fontsize=8.5)
ax_c.legend(loc="best", frameon=False, fontsize=6.5)

ax_d = fig_s3.add_subplot(gs[1, 3])
for mode in ESTIMATOR_MODES:
    info = MODE_DISPLAY[mode]
    fxx = mode_models[mode].evaluate_kernel("xx", r_kernel_eval)
    ax_d.plot(r_kernel_eval, fxx, color=info["color"], lw=1.8, label=info["label"])
ax_d.axhline(0, color="0.5", lw=0.6, ls="--")
ax_d.set_xlabel("Chromosome-chromosome distance (μm)")
ax_d.set_ylabel("$f_{xx}(r)$  (μm/s) "
                "\n+ attractive · - repulsive")
ax_d.set_title("Learned $f_{xx}$ kernel (enveloped)",
               loc="left", fontsize=8.5)
ax_d.legend(loc="best", frameon=False, fontsize=6.5)

# Panel letters
ax_xy_nb.text(-0.30, 1.10, "A", transform=ax_xy_nb.transAxes,
              fontsize=11, fontweight="bold", va="bottom", ha="left")
for ax_panel, label in zip([ax_b, ax_c, ax_d], ["B", "C", "D"]):
    ax_panel.text(-0.13, 1.04, label, transform=ax_panel.transAxes,
                  fontsize=11, fontweight="bold", va="bottom", ha="left")

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
ax.set_xlabel("Distance to partner (μm)")
ax.set_ylabel("$f_{xy}(r)$  (μm/s) "
              "\n+ attractive · - repulsive")
ax.set_title("Per-cell chromosome-to-partner kernel",
             loc="left", fontsize=8.5)
ax.set_ylim(-0.10, 0.05)
ax.legend(loc="best", frameon=False, fontsize=6.5)

ax = axes_s4[1]
ax.fill_between(r_xx_eval, xx_ci_lo, xx_ci_hi, color=OKABE_ITO["vermil"],
                alpha=0.15, label="pooled 5–95 % CI")
for curve in percell_xx_curves:
    ax.plot(r_xx_eval, curve, color="0.45", lw=0.7, alpha=0.55)
ax.plot(r_xx_eval, median_xx, color="0.15", lw=1.6, ls="--",
        label="per-cell median")
ax.plot(r_xx_eval, pooled_model.evaluate_kernel("xx", r_xx_eval),
        color=OKABE_ITO["vermil"], lw=2.0, label="pooled fit")
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


# %% [markdown]
# ## Fig S5 -- Drift-vs-diffusion sensitivity (companion to Fig 4)
#
# Companion analysis for the main-text Fig 4B drift signal fraction
# panel.  Three panels:
#
# - **(A)** Sensitivity to the choice of accumulation timescale $T$.
#   Curves of $f_{\mathrm{drift}}(d; T)$ for $T \in \{\mathrm{dt}, 50,
#   T_{\mathrm{steady}} = 150, 300, T_{\mathrm{obs}} \approx 475\}$ s.
#   $T_{\mathrm{steady}} = 150$ s is the main-panel anchor (docx Result 1
#   steady-elongation phase); $T_{\mathrm{obs}}$ is the trimmed-window
#   length, shown for context.  The monotone shape and spatial ordering
#   are robust across $T$; the 50/50 crossover position shifts smoothly
#   from outside the supported domain (small $T$) to small $d$ (large
#   $T$).  Single-frame curves stay near zero across all $d$,
#   consistent with one-step motion being noise-dominated even at large
#   $d$ (panel B).
# - **(B)** Per-step Peclet number
#   $\mathrm{Pe}_{\Delta t}(d) = |F(d)|\sqrt{\Delta t / (2\,D(d))}$
#   reported for transparency.  Per-step is well below the 1-D 50/50
#   threshold $\mathrm{Pe} = 1$ everywhere, consistent with one-step
#   motion being noise-dominated; the trajectory-scale story in Fig 4B
#   is a multi-frame consequence (Frishman & Ronceray PRX 2020 capacity
#   formalism).
# - **(C)** Local 50/50 timescale $\tau_{50}(d) = 2\,D(d)/|F(d)|^{2}$,
#   the time at which drift-squared equals diffusive variance along
#   the force direction.  Horizontal reference at the median
#   $T_{\mathrm{obs}}$.  Where $\tau_{50}(d) < T_{\mathrm{obs}}$, drift
#   wins over the window; where $\tau_{50}(d) > T_{\mathrm{obs}}$,
#   noise wins.

# %%
print("Fig S5: drift-vs-diffusion sensitivity sweep...")
from chromlearn.model_fitting.diffusion import (
    COORDINATE_MAPS as _COORD_MAPS,
    _predicted_force as _pred_F,
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

# Compute per-state F, D, d once
fmag_chunks_s5, d_chunks_s5, D_chunks_s5 = [], [], []
for cell in cells:
    coord_arr = _coord_fn_s5(cell.chromosomes, cell)
    T_frames = cell.chromosomes.shape[0]
    for t in range(T_frames - 1):
        F = _pred_F(cell, t, fit_result=model_s5,
                    basis_xx=model_s5.basis_xx, basis_xy=model_s5.basis_xy,
                    topology=model_s5.topology)
        Fmag = np.linalg.norm(F, axis=1)
        d_t = coord_arr[t]
        valid = np.isfinite(Fmag) & np.isfinite(d_t)
        if valid.any():
            fmag_chunks_s5.append(Fmag[valid])
            d_chunks_s5.append(d_t[valid])
            D_chunks_s5.append(D_pooled_s5.evaluate(d_t[valid]))

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

# Panel B: per-step Pe
Pe_step_s5 = np.full_like(force_mags_s5, np.nan)
Pe_step_s5[ok_s5] = (force_mags_s5[ok_s5]
                     * np.sqrt(config_s5.dt
                               / (2.0 * D_at_obs_s5[ok_s5])))
pe_med_s5, pe_lo_s5, pe_hi_s5 = _binmedian_s5(Pe_step_s5, bin_idx_s5, N_BINS_S5)

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
    if abs(T - T_STEADY_S5) < 0.5:
        label = rf"$T={T:.0f}$ s ($T_{{\mathrm{{steady}}}}$, main panel)"
    elif abs(T - T_OBS_MED_S5) < 0.5:
        label = rf"$T={T:.0f}$ s ($T_{{\mathrm{{obs}}}}$, trim window)"
    else:
        label = rf"$T={T:.0f}$ s"
    ax_a_s5.plot(bin_centers_s5, fdrift_curves_T[T], "o-", color=color,
                 lw=1.4, markersize=3.5, label=label)
ax_a_s5.axhline(0.5, color="0.5", lw=0.7, ls="--")
ax_a_s5.set_xlim(EVAL_LO_S5, EVAL_HI_S5)
ax_a_s5.set_ylim(0.0, 1.0)
ax_a_s5.set_xlabel("Distance from spindle center, $d$ (μm)")
ax_a_s5.set_ylabel(r"$f_{\mathrm{drift}}(d;\,T)$")
ax_a_s5.set_title("Sensitivity to observation timescale $T$",
                   loc="left", fontsize=8.5)
ax_a_s5.legend(loc="best", frameon=False, fontsize=6.0)

# (B) Per-step Pe
ax_b_s5 = axes_s5[1]
ax_b_s5.fill_between(bin_centers_s5, pe_lo_s5, pe_hi_s5,
                      color=OKABE_ITO["blue"], alpha=0.18, linewidth=0,
                      label="IQR")
ax_b_s5.plot(bin_centers_s5, pe_med_s5, "o-", color=OKABE_ITO["blue"],
              lw=1.4, markersize=3.5, label="median")
ax_b_s5.axhline(1.0, color="0.5", lw=0.7, ls="--",
                 label=r"$\mathrm{Pe} = 1$ (1-D 50/50)")
ax_b_s5.set_xlim(EVAL_LO_S5, EVAL_HI_S5)
ax_b_s5.set_ylim(bottom=0.0)
ax_b_s5.set_xlabel("Distance from spindle center, $d$ (μm)")
ax_b_s5.set_ylabel(r"$\mathrm{Pe}_{\Delta t}(d) = |F|\sqrt{\Delta t/(2D)}$")
ax_b_s5.set_title("Per-step Peclet number (transparency)",
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
                 label=rf"$T_{{\mathrm{{steady}}}}={T_STEADY_S5:.0f}$ s "
                        "(main panel)")
ax_c_s5.axhline(T_OBS_MED_S5, color="0.7", lw=0.5, ls=":",
                 label=rf"$T_{{\mathrm{{obs}}}}\approx{T_OBS_MED_S5:.0f}$ s "
                        "(trim window)")
ax_c_s5.set_xlim(EVAL_LO_S5, EVAL_HI_S5)
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
