# %% [markdown]
# # 08 -- Cross-condition comparison of learned kernels
#
# The pooled fit (NB04) learns one set of interaction kernels for the control
# condition (rpe18_ctr).  Here we ask how the learned force law changes across
# experimental perturbations:
#
# 1. **Rod-delta/delta (rod311_ctr)**: Corona-deficient mutant with ~3 skMTs
#    instead of ~12.  The PNAS companion paper shows a weaker velocity-distance
#    sigmoid; does the learned f_xy kernel reflect this?
# 2. **CENP-E inhibition (rpe18_gsk)**: Blocks a plus-end-directed motor;
#    expect altered long-range kernel.
# 3. **PRC1 depletion (rpe18_prc)**: Disrupts antiparallel MT overlap; expect
#    changes to spindle geometry but possibly not the poleward kernel.
# 4. **Kid/Kif4A depletion (rpe18_siKidKif4A)**: Removes chromokinesins (arm
#    motors); should mainly affect polar ejection (short-range repulsion).
#
# This notebook is exploratory: we may or may not include it in the paper
# depending on whether the cross-condition kernel differences are informative
# beyond what the PNAS paper already showed.

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
from chromlearn.io.trajectory import (
    TrimmedCell,
    spindle_frame,
    trim_trajectory,
)
from chromlearn.model_fitting import FitConfig
from chromlearn.model_fitting.fit import bootstrap_kernels, fit_model
from chromlearn.model_fitting.model import FittedModel
from chromlearn.model_fitting.simulate import simulate_cell

plt.rcParams["figure.dpi"] = 110

# %% [markdown]
# ## Configuration
#
# Same fit config as NB04/05 (poles topology, 10 B-spline bases, 1e-3 reg).

# %%
FRAC_NEB_AO_WINDOW = 0.4         # Baseline trajectory window as a fraction of NEB-to-AO.
TOPOLOGY = "poles"               # Drift model compared across perturbation conditions.
N_BASIS_XX = 10                  # Number of spline basis functions for chromosome-chromosome kernels.
N_BASIS_XY = 10                  # Number of spline basis functions for pole-chromosome kernels.
R_MIN = 0.3                      # Lower basis cutoff in microns.
R_MAX = 15.0                     # Upper basis cutoff in microns.
BASIS_TYPE = "bspline"           # Functional basis used for the learned kernels.
LAMBDA_RIDGE = 1e-3              # L2 penalty on coefficient magnitude.
LAMBDA_ROUGH = 1.0               # Smoothness penalty on neighboring spline coefficients.
BASIS_EVAL_MODE = "ito"          # Drift-evaluation convention used in the fit.
DT = 5.0                         # Frame interval in seconds.
CONFIG = FitConfig(
    topology=TOPOLOGY,
    n_basis_xx=N_BASIS_XX,
    n_basis_xy=N_BASIS_XY,
    r_min_xx=R_MIN,
    r_max_xx=R_MAX,
    r_min_xy=R_MIN,
    r_max_xy=R_MAX,
    basis_type=BASIS_TYPE,
    lambda_ridge=LAMBDA_RIDGE,
    lambda_rough=LAMBDA_ROUGH,
    basis_eval_mode=BASIS_EVAL_MODE,
    endpoint_method="neb_ao_frac",
    endpoint_frac=FRAC_NEB_AO_WINDOW,
    dt=DT,
)

# Conditions to compare.  Kid/Kif4A excluded: only 1/11 cells survives
# the default min_frames=25 trimming, too thin for meaningful fitting.
CONDITIONS = {
    "rpe18_ctr": "Control",
    "rod311_ctr": "Rod-d/d",
    "rpe18_gsk": "CENP-E inh.",
    "rpe18_prc": "PRC1 dep.",
}

COLORS = {
    "rpe18_ctr": "C0",
    "rod311_ctr": "C1",
    "rpe18_gsk": "C2",
    "rpe18_prc": "C3",
}

N_BOOT = 200
RNG_SEED = 42

# %% [markdown]
# ## Load, fit, and bootstrap all conditions

# %%
results: dict[str, dict] = {}

for cond, label in CONDITIONS.items():
    print(f"\n--- {label} ({cond}) ---")
    cells_raw = load_condition(cond)
    cells = []
    for c in cells_raw:
        try:
            cells.append(trim_trajectory(c, method="neb_ao_frac", frac=FRAC_NEB_AO_WINDOW))
        except ValueError as e:
            print(f"  skip {c.cell_id}: {e}")
    print(f"  {len(cells)}/{len(cells_raw)} cells")

    model = fit_model(cells, CONFIG)
    print(f"  D = {model.D_x:.6f} um^2/s, n_params = {model.theta.size}")

    boot = bootstrap_kernels(
        cells, CONFIG, n_boot=N_BOOT,
        rng=np.random.default_rng(RNG_SEED),
    )
    print(f"  Bootstrap: {boot.theta_samples.shape[0]} resamples")

    results[cond] = {
        "label": label,
        "cells": cells,
        "model": model,
        "boot": boot,
    }

# %% [markdown]
# ## Overlay f_xy kernels across conditions
#
# The chromosome-to-pole kernel is the primary observable.  We plot each
# condition's pooled f_xy with its 90% bootstrap CI.

# %%
r_eval = np.linspace(CONFIG.r_min_xy, CONFIG.r_max_xy, 200)

fig, ax = plt.subplots(figsize=(9, 5.5))

for cond, info in results.items():
    model = info["model"]
    boot = info["boot"]
    color = COLORS[cond]

    phi_xy = model.basis_xy.evaluate(r_eval)
    pooled_fxy = phi_xy @ model.theta_xy

    boot_xy = boot.theta_samples[:, model.n_basis_xx:]
    boot_curves = phi_xy @ boot_xy.T
    ci_lo = np.percentile(boot_curves, 5, axis=1)
    ci_hi = np.percentile(boot_curves, 95, axis=1)

    ax.fill_between(r_eval, ci_lo, ci_hi, color=color, alpha=0.12)
    ax.plot(r_eval, pooled_fxy, color=color, linewidth=2, label=info["label"])

ax.axhline(0, color="0.7", linestyle="--", linewidth=0.5)
ax.set_xlabel("Distance to pole (um)")
ax.set_ylabel("f_xy (force)")
ax.set_title("Chromosome-to-pole kernel f_xy across conditions")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Overlay f_xx kernels (if present)

# %%
has_xx = any(r["model"].basis_xx is not None and r["model"].n_basis_xx > 0
             for r in results.values())

if has_xx:
    r_eval_xx = np.linspace(CONFIG.r_min_xx, CONFIG.r_max_xx, 200)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cond, info in results.items():
        model = info["model"]
        if model.basis_xx is None:
            continue
        phi_xx = model.basis_xx.evaluate(r_eval_xx)
        pooled_fxx = phi_xx @ model.theta_xx
        boot_xx = info["boot"].theta_samples[:, :model.n_basis_xx]
        boot_curves = phi_xx @ boot_xx.T
        ci_lo = np.percentile(boot_curves, 5, axis=1)
        ci_hi = np.percentile(boot_curves, 95, axis=1)
        ax.fill_between(r_eval_xx, ci_lo, ci_hi, color=COLORS[cond], alpha=0.12)
        ax.plot(r_eval_xx, pooled_fxx, color=COLORS[cond], linewidth=2,
                label=info["label"])
    ax.axhline(0, color="0.7", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Inter-chromosome distance (um)")
    ax.set_ylabel("f_xx (force)")
    ax.set_title("Chromosome-chromosome kernel f_xx across conditions")
    ax.legend()
    fig.tight_layout()
    plt.show()
else:
    print("No f_xx kernels in these models (poles topology has n_basis_xx=10 "
          "but check whether basis_xx is None).")

# %% [markdown]
# ## Scalar summary per condition
#
# Extract scalar features from each condition's f_xy kernel.
# Sign convention: positive f_xy = attractive (toward pole).

# %%
print(f"{'Condition':<20} {'Cells':>5} {'D':>10} {'Peak attr':>10} "
      f"{'Attr r':>8} {'f_xy@5um':>10}")
print("-" * 70)

for cond, info in results.items():
    model = info["model"]
    phi_xy = model.basis_xy.evaluate(r_eval)
    curve = phi_xy @ model.theta_xy
    peak_attract = float(np.max(curve))
    peak_attract_loc = float(r_eval[np.argmax(curve)])
    ref_idx = np.argmin(np.abs(r_eval - 5.0))
    ref_val = float(curve[ref_idx])
    print(f"{info['label']:<20} {len(info['cells']):>5} {model.D_x:>10.6f} "
          f"{peak_attract:>10.6f} {peak_attract_loc:>8.2f} {ref_val:>10.6f}")

# %% [markdown]
# ## D(x) comparison across conditions
#
# If NB06 showed a real spatial diffusion gradient for control, does the same
# gradient appear in other conditions?  Differences in D(x) between conditions
# would suggest the noise structure of chromosome motion is perturbation-dependent.

# %%
from chromlearn.model_fitting.basis import BSplineBasis
from chromlearn.model_fitting.diffusion import estimate_diffusion_variable

D_COORD = "distance"
N_BASIS_D = 8
R_MIN_D = 0.5
R_MAX_D = 12.0
LAMBDA_D = 1e-2

eval_coords = np.linspace(R_MIN_D, R_MAX_D, 200)

fig, ax = plt.subplots(figsize=(9, 5.5))

for cond, info in results.items():
    model = info["model"]
    dr = estimate_diffusion_variable(
        info["cells"],
        basis_D=BSplineBasis(R_MIN_D, R_MAX_D, N_BASIS_D),
        coord_name=D_COORD,
        dt=CONFIG.dt,
        mode="vestergaard",
        lambda_ridge=LAMBDA_D,
        topology=model.topology,
    )
    D_curve = dr.evaluate(eval_coords)
    ax.plot(eval_coords, D_curve, color=COLORS[cond], linewidth=2,
            label=info["label"])
    # Scalar D as horizontal line
    ax.axhline(model.D_x, color=COLORS[cond], linestyle=":", linewidth=0.8,
               alpha=0.5)

ax.set_xlabel("Distance from spindle center (um)")
ax.set_ylabel("D (um$^2$/s)")
ax.set_title("Spatially-varying diffusion D(distance) across conditions")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Rollout validation: control vs Rod
#
# Simulate a few control cells with the control model and a few Rod cells with
# the Rod model.  Compare mean radial/axial trajectories to verify that the
# condition-specific fits capture the qualitative dynamics of each perturbation.

# %%
ROLLOUT_CONDITIONS = ["rpe18_ctr", "rod311_ctr"]
N_ROLLOUT = 3  # cells per condition

fig, axes = plt.subplots(len(ROLLOUT_CONDITIONS) * N_ROLLOUT, 2,
                         figsize=(12, 4 * len(ROLLOUT_CONDITIONS) * N_ROLLOUT),
                         squeeze=False)

row = 0
for cond in ROLLOUT_CONDITIONS:
    info = results[cond]
    cells = info["cells"]
    model = info["model"]
    idxs = np.linspace(0, len(cells) - 1, N_ROLLOUT, dtype=int)

    for cell_idx in idxs:
        cell = cells[cell_idx]
        T, _, N = cell.chromosomes.shape
        rng = np.random.default_rng(200 + row)
        _, sim_cell = simulate_cell(cell, model, rng=rng)

        sf_real = spindle_frame(cell)
        sf_sim = spindle_frame(sim_cell)
        time_axis = np.arange(T) * CONFIG.dt

        n_show = min(10, N)
        for i in range(n_show):
            axes[row, 0].plot(time_axis, sf_real.radial[:, i],
                              "k-", alpha=0.15, linewidth=0.5)
            axes[row, 0].plot(time_axis, sf_sim.radial[:, i],
                              "C0-", alpha=0.15, linewidth=0.5)
        axes[row, 0].plot(time_axis, np.nanmean(sf_real.radial, axis=1),
                          "k-", linewidth=2, label="Real")
        axes[row, 0].plot(time_axis, np.nanmean(sf_sim.radial, axis=1),
                          "C0--", linewidth=2, label="Simulated")
        axes[row, 0].set_ylabel("Radial (um)")
        axes[row, 0].set_title(f"{info['label']}: {cell.cell_id} -- radial")
        axes[row, 0].legend(fontsize=7)

        for i in range(n_show):
            axes[row, 1].plot(time_axis, sf_real.axial[:, i],
                              "k-", alpha=0.15, linewidth=0.5)
            axes[row, 1].plot(time_axis, sf_sim.axial[:, i],
                              "C0-", alpha=0.15, linewidth=0.5)
        axes[row, 1].plot(time_axis, np.nanmean(sf_real.axial, axis=1),
                          "k-", linewidth=2, label="Real")
        axes[row, 1].plot(time_axis, np.nanmean(sf_sim.axial, axis=1),
                          "C0--", linewidth=2, label="Simulated")
        axes[row, 1].set_ylabel("Axial (um)")
        axes[row, 1].set_title(f"{info['label']}: {cell.cell_id} -- axial")
        axes[row, 1].legend(fontsize=7)
        row += 1

axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")
fig.suptitle("Rollout: condition-specific models vs real data")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# **Key questions:**
#
# 1. Does the learned f_xy kernel differ qualitatively between control and
#    Rod-d/d?  The PNAS paper predicts a weaker sigmoid for Rod (fewer skMTs),
#    which should manifest as a shallower attractive well.
# 2. Do CENP-E and Kid/Kif4A perturbations show kernel changes consistent with
#    motor-specific effects (long-range vs short-range)?
# 3. Is the D(x) spatial gradient conserved across conditions, or does it
#    change shape with perturbation?
# 4. Are the cross-condition differences larger than bootstrap uncertainty?
#    If the CIs overlap heavily, the conditions are not distinguishable by
#    our method.
