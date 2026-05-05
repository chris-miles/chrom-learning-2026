# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 10 — Main text figure (Fig 3)
#
# **STATUS: SKELETON (in progress).**  Panel implementations are TODOs; do
# not use this notebook for paper figures yet.  The setup section fits the
# canonical model so panel work can drop in.
#
# Lightweight assembler for the paper's main figure on Chris's part of the
# spindle-forces draft.  Pulls panel content from NB03 (PCA + lag), NB04
# (force-distance kernels), and NB06 (effective diffusion).  This notebook
# does NOT do model selection or hyperparameter sweeps — it loads the
# canonical config (poles_and_chroms_enveloped, n_basis=10, lambda_rough=1)
# and produces publication-ready panels.
#
# Panels (per Alex's docx Result 3):
# - **A**: PCA trajectories of centrosomes and chromosomes
# - **B**: lag-correlation showing chromosomes follow centrosomes
# - **C**: learned force–distance kernels (f_xy, f_xx) with bootstrap CIs;
#   data-density rug; sparse-data regions shaded as "extrapolated"
# - **D**: effective diffusion D(distance) along spindle radial coordinate

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
from chromlearn.model_fitting import FitConfig
from chromlearn.model_fitting.fit import bootstrap_kernels, fit_model

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

# %% [markdown]
# ## Canonical fit
#
# Mirrors NB04's selected configuration: ``poles_and_chroms_enveloped``
# with ``n_basis = 10`` and ``lambda_rough = 1.0``.  Envelope parameters
# from the steric-prior choice in NB04 (``r0 = 1.5 um``, ``w = 0.3 um``).

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

cells_raw = load_condition(CONDITION)
cells = [trim_trajectory(c, method="neb_ao_frac", frac=FRAC_NEB_AO) for c in cells_raw]

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

# Bootstrap for kernel CI bands (matches NB04 setup)
N_BOOT = 100
boot = bootstrap_kernels(cells, config, n_boot=N_BOOT,
                         rng=np.random.default_rng(42))

# %% [markdown]
# ## Panel A — PCA trajectories
#
# TODO: pull PCA logic from NB03.  Plot CS and CH trajectories projected
# onto the spindle's principal axes.

# %%
# TODO: panel A implementation

# %% [markdown]
# ## Panel B — Lag correlation: chromosomes follow centrosomes
#
# TODO: pull lag correlation from NB03.  Show CS-CH velocity cross-correlation
# vs lag, with the asymmetric peak at positive lag (chromosomes following).

# %%
# TODO: panel B implementation

# %% [markdown]
# ## Panel C — Force–distance kernels
#
# Show ``f_xy(r)`` (chromosome-to-pole) and ``f_xx(r)`` (chromosome-chromosome,
# enveloped).  Each panel: bootstrap 5–95% CI band, mean fit, data-density
# rug along the x-axis, sparse regions shaded as extrapolation territory.
# Alex's prediction (docx): f_xy is "almost constant attractive at long
# distances and repels at spindle-width-scale distances" — verify visually.

# %%
# TODO: panel C implementation

# %% [markdown]
# ## Panel D — Effective diffusion D(x)
#
# TODO: pull D(distance) from NB06 (vestergaard estimator, restricted to
# data-supported domain).

# %%
# TODO: panel D implementation

# %% [markdown]
# ## Figure assembly

# %%
# TODO: combine A-D into a single multi-panel figure with consistent styling
