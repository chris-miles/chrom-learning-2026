# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 00b — Supplement figures
#
# **STATUS: in progress.**
#
# Lightweight assembler for the supplementary figures supporting Chris's
# part of *Hierarchy of spindle forces in prometaphase*. Each figure is
# rendered as a self-contained PDF/PNG; Alex can combine into multi-panel
# supplement layouts as he prefers.
#
# Scope was deliberately trimmed to keep the supplement focused on the
# robustness claims that actually support the main-text narrative. Drops:
#
# - **Data-density / extrapolation rugs** — instead handled inline on
#   main-text Fig 3 by shading sparse-data regions on the kernel plots.
# - **Envelope-shape methodology and ``(r0, w)`` sensitivity** — held for
#   revision; the envelope is a fixed biological prior and the choice
#   is justified in CLAUDE.md / methods text rather than as its own
#   figure. We can add it back if reviewers ask.
#
# Surviving panels — each maps to a robustness story for one of the
# main-text figures:
#
# Companions to **Fig 2** (CS-CS sufficient):
# - **S1** — pp-only vs pp+cp per-cell strip plot for both metrics
#   (1-step RMSE and rollout path MSE).  The headline mean ± SE bars
#   are in Fig 2; the per-cell breakdown lives here for fold-level
#   inspection.
# - **S2** — pp/cp partition non-identifiability (NB03b
#   reconciliation).  Demonstrates that the apparent path-MSE pp+cp
#   edge in Fig 2 is consistent with the analytical partition
#   non-identifiability when chromosomes are observed covariates.
#
# Companions to **Fig 3** (kernels + topology comparison):
# - **S3** — 5-topology per-cell breakdown (LOO path MSE) including
#   ``center_and_chroms`` (free-xx + midpoint partner; dropped from
#   Fig 3 main text).  Admissibility annotation as in Fig 3.
# - **S4** — held-out forecast error vs horizon (1–30 frames) for all
#   5 topologies.  Directly addresses Alex's docx ask
#   (*"held-out forecast error vs horizon for up to 10 frames"*),
#   extended to 30 frames for the tail.  Admissible solid, nuisance
#   upper-bound dashed.
# - **S5** — hyperparameter sensitivity (from NB05): ``(n_basis ×
#   λ_rough)`` path-MSE heatmap; estimator-mode bars (Itô / Itô-shift
#   / Stratonovich); endpoint-method bars (``frac`` sweep +
#   ``end_sep``).
# - **S6** — per-cell kernel variability (from NB07): spaghetti of
#   per-cell ``f_xy`` overlaid on pooled bootstrap CI band.
#
# Companion to **Fig 4** (D(d)):
# - **S7** — D-estimator robustness: pooled D(d) curves from
#   ``f_corrected`` (main-text headline), ``vestergaard``, ``msd``,
#   and ``weak_noise`` overlaid on the same axes.  Confirms the
#   "D grows away from spindle" pattern is robust to estimator
#   choice; the absolute magnitudes differ (drift-contamination of
#   Vestergaard, etc.) but the qualitative trend agrees.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chromlearn import find_repo_root

ROOT = find_repo_root(Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chromlearn.io.catalog import load_condition
from chromlearn.io.trajectory import trim_trajectory

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
# Okabe-Ito core palette; nuisance upper-bound models use neutral grays
# with hatching.
TOPOLOGY_DISPLAY = {
    "poles":                       {"label": "poles",
                                    "color": OKABE_ITO["blue"],
                                    "admissible": True},
    "center":                      {"label": "center",
                                    "color": OKABE_ITO["green"],
                                    "admissible": True},
    "poles_and_chroms_enveloped":  {"label": "poles + chroms (enveloped)",
                                    "color": OKABE_ITO["vermil"],
                                    "admissible": True},
    "poles_and_chroms":            {"label": "poles + chroms (free xx)",
                                    "color": "0.55",
                                    "admissible": False},
    "center_and_chroms":           {"label": "center + chroms (free xx)",
                                    "color": "0.35",
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
# ## Setup
#
# Mirrors NB04's canonical configuration so panels are consistent with
# the main figure.

# %%
CONDITION = "rpe18_ctr"
FRAC_NEB_AO = 0.4

cells_raw = load_condition(CONDITION)
cells = [trim_trajectory(c, method="neb_ao_frac", frac=FRAC_NEB_AO) for c in cells_raw]
print(f"Loaded {len(cells)} {CONDITION} cells.")

# %% [markdown]
# ## Fig S1 — pp-only vs pp+cp per-cell breakdown (companion to Fig 2)
#
# Per-cell paired strip plots for both metrics that anchor main-text
# Fig 2: 1-step LOO RMSE and deterministic-rollout LOO path MSE.
# Light connecting lines show within-cell paired changes.  Mean ± SE
# diamonds.
#
# **TODO.** Build pole-velocity design matrices, run LOO over the 12
# cells (or load cached results), and render the two strip plots
# side-by-side.

# %%
# TODO: Fig S1 implementation

# %% [markdown]
# ## Fig S2 — pp/cp partition non-identifiability (companion to Fig 2)
#
# Source: NB03b.  Demonstrates analytically and empirically that the
# apparent path-MSE pp+cp edge in Fig 2 is consistent with the
# partition non-identifiability when chromosomes are observed as
# covariates: a continuous family of (f_pp, f_cp) pairs gives nearly
# identical predictions, so the slight pp+cp gain is fitting
# CH-correlated noise rather than capturing a real causal coupling.
#
# **TODO.** Reuse NB03b's force-partition reconciliation analysis;
# render a 2-panel figure (analytical degeneracy + empirical
# coefficient swap).

# %%
# TODO: Fig S2 implementation

# %% [markdown]
# ## Fig S3 — 5-topology per-cell breakdown (companion to Fig 3)
#
# Companion to main-text Fig 3, which shows only the topology
# mean ± SE bars and drops ``center_and_chroms`` to keep the panel
# focused.  This panel exposes the per-cell distribution across all
# 5 topologies (poles, center, poles_and_chroms_enveloped,
# poles_and_chroms, center_and_chroms).  Light connecting lines
# highlight fold-by-fold consistency.
#
# **TODO.** Run ``rollout_cross_validate`` for each of the 5
# topologies on the same trimmed cells (or load NB04's saved results
# if cached), then render the strip plot.  Cells that blow up on
# isolated topologies (typically ``center_and_chroms``) are flagged
# as annotated outliers rather than clipping the y-axis.

# %%
# TODO: Fig S3 implementation

# %% [markdown]
# ## Fig S4 — Forecast error vs horizon, 1–30 frames (companion to Fig 3)
#
# Directly addresses Alex's docx Result 3C ask
# (*"held-out forecast error vs horizon for up to 10 frames"*),
# extended to 30 frames for the tail.  Ensemble MSE vs horizon for
# all 5 topologies; admissible solid, nuisance upper-bound dashed.
# Vertical annotation at h = 10 marks Alex's anchor.
#
# **TODO.** Pull from NB04's ``forecast_horizon_cross_validate``
# output (or recompute via ``evaluate_all_loocv``).

# %%
# TODO: Fig S4 implementation

# %% [markdown]
# ## Fig S5 — Hyperparameter sensitivity (companion to Fig 3)
#
# Source: NB05.  Three sub-panels in one figure:
#
# - Heatmap of path MSE over the ``(n_basis, λ_rough)`` grid for the
#   canonical ``poles_and_chroms_enveloped`` topology, with the
#   selected operating point marked.
# - Bars of path MSE for estimator modes (Itô / Itô-shift /
#   Stratonovich).
# - Bars of path MSE for endpoint methods (``frac`` sweep +
#   ``end_sep``).

# %%
# TODO: Fig S5 implementation

# %% [markdown]
# ## Fig S6 — Per-cell kernel variability (companion to Fig 3)
#
# Source: NB07.  Spaghetti of per-cell ``f_xy`` fits overlaid on the
# pooled bootstrap 5–95 % CI band.  Quantifies cell-to-cell
# variability vs pooled-fit uncertainty.

# %%
# TODO: Fig S6 implementation

# %% [markdown]
# ## Fig S7 — D-estimator robustness (companion to Fig 4)
#
# Pooled D(d) curves from all four estimators (``f_corrected``,
# ``vestergaard``, ``msd``, ``weak_noise``) overlaid on a single axis.
# The main-text Fig 4 reports ``f_corrected`` because it explicitly
# subtracts the fitted drift before estimating residual variance,
# which matters in this drift-dominated, spatially heterogeneous
# regime (Frishman & Ronceray PRX 2020, App. H).  This supplement
# panel confirms the qualitative "D grows away from spindle center"
# pattern is robust to estimator choice; the absolute magnitudes
# differ by a factor of ~2-3× across estimators (Vestergaard sits
# systematically high due to drift contamination, ``f_corrected``
# overlaps the scalar-D baseline).
#
# **TODO.** Run ``estimate_diffusion_variable`` with all four modes,
# overlay on one panel, annotate which estimator each curve uses.

# %%
# TODO: Fig S7 implementation
