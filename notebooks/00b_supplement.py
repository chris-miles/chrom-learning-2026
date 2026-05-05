# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 11 — Supplement figures
#
# **STATUS: SKELETON (in progress).**  Panel implementations are TODOs;
# do not use this notebook for paper figures yet.
#
# Lightweight assembler for the supplementary figures supporting Chris's
# part of the spindle-forces paper.  Pulls panel content from NB04
# (topology comparison, horizon-resolved curves), NB05 (hyperparameter
# robustness), NB06 (D(x) diagnostics), and NB07 (per-cell variability).
#
# Panels (subject to Alex feedback):
# - **S1**: 5-topology comparison table + path-MSE bar chart (with
#   biological-admissibility annotation)
# - **S2**: forecast error vs horizon (1-30 frames) for all topologies —
#   directly addresses Alex's "show held-out forecast error vs horizon
#   for up to 10 frames" request, extended to 30 frames for the tail
# - **S3**: hyperparameter sensitivity — (n_basis × λ_rough) heatmap
#   showing path-MSE robustness; estimator-mode and endpoint-method bars
# - **S4**: per-cell kernel variability (NB07's spaghetti plot vs pooled
#   bootstrap CI)
# - **S5**: data-density on each kernel domain — rug plot showing where
#   ``f_xy`` and ``f_xx`` are data-supported vs penalty-extrapolated
# - **S6**: methodology — envelope shape; sensitivity sweeps to
#   ``(envelope_r0, envelope_w)`` if reviewers ask

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

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

# %% [markdown]
# ## Setup
#
# Mirrors NB04's canonical configuration so panels are consistent with the
# main figure.

# %%
CONDITION = "rpe18_ctr"
FRAC_NEB_AO = 0.4

cells_raw = load_condition(CONDITION)
cells = [trim_trajectory(c, method="neb_ao_frac", frac=FRAC_NEB_AO) for c in cells_raw]
print(f"Loaded {len(cells)} {CONDITION} cells.")

# %% [markdown]
# ## Panel S1 — Topology comparison
#
# Bar chart of path MSE for the 5 candidate topologies, annotated with
# biological admissibility (admissible: poles, center, enveloped;
# nuisance UB: full-range xx variants).  Numerical values from NB04.

# %%
# TODO: re-run NB04's topology comparison or load saved results;
# render the bar chart with admissibility annotations.

# %% [markdown]
# ## Panel S2 — Forecast error vs horizon (1–30 frames)
#
# Alex's docx specifically requests "held-out forecast error vs horizon
# for up to 10 frames".  We extend to 30 frames so the curve can be
# inspected past the early-prometaphase analysis window.  Show all 5
# topologies; admissible models in solid lines, nuisance UB models
# dashed.

# %%
# TODO: pull forecast_horizon_cross_validate output from NB04 or recompute;
# plot ensemble-MSE curves vs horizon.

# %% [markdown]
# ## Panel S3 — Hyperparameter sensitivity (NB05)
#
# Three sub-panels:
# - Heatmap of path MSE over (n_basis × λ_rough) — show robustness
# - Estimator mode bars (Ito vs Ito-shift vs Strato)
# - Endpoint method bars (frac sweep + end_sep)

# %%
# TODO: pull NB05 sweep outputs.

# %% [markdown]
# ## Panel S4 — Per-cell kernel variability
#
# Spaghetti of per-cell ``f_xy`` fits vs pooled bootstrap CI.  Tells the
# reviewer how much cell-to-cell variability there is in the inferred
# kernel.  From NB07.

# %%
# TODO: pull NB07's per-cell + pooled overlay.

# %% [markdown]
# ## Panel S5 — Data-density and extrapolation regions
#
# For ``f_xy`` and ``f_xx``, show empirical pairwise-distance histograms
# overlaid on the kernel domain.  Shade the region where data is sparse
# (e.g., ``f_xy`` below 2 um — fewer than 100 of 112k pairs) so reviewers
# can see what's data-driven vs penalty-extrapolated.

# %%
# TODO: data-density panels.

# %% [markdown]
# ## Panel S6 — Envelope methodology
#
# - Envelope shape ``s(r) = 0.5 * (1 - tanh((r - r0) / w))`` overlaid
#   on raw spline basis.
# - Optional: sensitivity to ``(r0, w)`` if reviewers question the
#   choice; otherwise hold for revision.

# %%
# TODO: envelope panel.

# %% [markdown]
# ## Supplement assembly

# %%
# TODO: stack panels S1–S6 into one or more supplement figures.
