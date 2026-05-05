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
# Surviving panels:
# - **S1** — 5-topology per-cell breakdown (LOO path MSE).  The per-cell
#   strip plot is in the supplement rather than the main text because
#   the bulk pattern is best summarised by the sorted mean ± SE bars
#   in main-text Fig 3; reviewers who want fold-level granularity can
#   look here.  Admissibility annotation as in Fig 3.
# - **S2** — pp-only vs pp+cp per-cell breakdown for the centrosome-
#   prediction analysis (companion to main-text Fig 2).  Same logic:
#   the headline mean ± SE bars are in Fig 2, and the per-cell strip
#   plot lives here for fold-level inspection.  Includes both 1-step
#   RMSE and rollout path MSE.
# - **S3** — held-out forecast error vs horizon (1–30 frames) for all 5
#   topologies. Directly addresses Alex's docx ask
#   (*"held-out forecast error vs horizon for up to 10 frames"*),
#   extended to 30 frames for the tail.  Admissible solid, nuisance
#   upper-bound dashed.
# - **S4** — hyperparameter sensitivity (NB05): ``(n_basis × λ_rough)``
#   path-MSE heatmap; estimator-mode bars (Itô / Itô-shift / Stratonovich);
#   endpoint-method bars (``frac`` sweep + ``end_sep``).
# - **S5** — per-cell kernel variability (NB07): spaghetti of per-cell
#   ``f_xy`` overlaid on pooled bootstrap CI band.

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
# ## Fig S1 — 5-topology per-cell breakdown (LOO path MSE)
#
# Companion to main-text Fig 3, which only shows the topology mean ±
# SE.  This panel exposes the per-cell distribution: each cell is a
# row, with one paired strip across the 5 topologies.  Light connecting
# lines highlight fold-by-fold consistency (or lack thereof).
# Admissible models (poles, center, ``poles_and_chroms_enveloped``) in
# Okabe-Ito categorical colors; nuisance upper-bound models
# (``poles_and_chroms``, ``center_and_chroms`` with free-form xx) in
# neutral gray.
#
# **TODO.** Run ``rollout_cross_validate`` for each of the 5
# topologies on the same trimmed cells (or load NB04's saved results
# if cached), then render the strip plot.  Cells that blow up on
# isolated topologies (typically ``center_and_chroms``) are flagged as
# annotated outliers rather than clipping the y-axis.

# %%
# TODO: Fig S1 implementation

# %% [markdown]
# ## Fig S2 — pp-only vs pp+cp per-cell breakdown
#
# Companion to main-text Fig 2.  Per-cell paired strip plots for both
# metrics (1-step RMSE and rollout path MSE), with light connecting
# lines showing within-cell paired changes.  Mean ± SE diamonds.
#
# **TODO.** Build pole-velocity design matrices, run LOO over the 12
# cells (or load cached results), and render the two strip plots
# side-by-side.

# %%
# TODO: Fig S2 implementation

# %% [markdown]
# ## Fig S3 — Forecast error vs horizon (1–30 frames)
#
# **TODO.** Ensemble MSE vs horizon for all 5 topologies, horizons
# 1 → 30 frames.  Admissible solid, nuisance upper-bound dashed.
# Vertical annotation at h = 10 marking Alex's docx anchor.  Pulled
# from NB04's ``forecast_horizon_cross_validate`` output (or recomputed
# via ``evaluate_all_loocv``).

# %%
# TODO: Fig S3 implementation

# %% [markdown]
# ## Fig S4 — Hyperparameter sensitivity (NB05)
#
# **TODO.** Three sub-panels in one figure:
#
# - Heatmap of path MSE over the ``(n_basis, lambda_rough)`` grid for
#   the canonical ``poles_and_chroms_enveloped`` topology, with the
#   selected operating point marked.
# - Bars of path MSE for estimator modes (Itô / Itô-shift / Stratonovich).
# - Bars of path MSE for endpoint methods (``frac`` sweep + ``end_sep``).
#
# Pulled from NB05.

# %%
# TODO: Fig S3 implementation

# %% [markdown]
# ## Fig S5 — Per-cell kernel variability
#
# **TODO.** Spaghetti of per-cell ``f_xy`` fits overlaid on pooled
# bootstrap 5–95 % CI band. Quantifies cell-to-cell variability vs
# pooled-fit uncertainty. Pulled from NB07.

# %%
# TODO: Fig S5 implementation
