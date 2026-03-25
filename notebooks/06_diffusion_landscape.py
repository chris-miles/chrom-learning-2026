# %% [markdown]
# # 06 -- Effective diffusion landscape D(x)
#
# The companion PNAS paper characterizes mean velocity-distance curves but never
# examines the **noise structure** of chromosome motion.  The SFI framework
# infers both drift (interaction kernels) and diffusion, so we can ask:
#
# 1. Is the effective diffusion coefficient D spatially varying?
# 2. Is that spatial structure robust across different local-D estimators?
# 3. Is the signal consistent across individual cells, or dominated by one or
#    two outliers?
#
# If D(x) shows a real gradient -- e.g., higher fluctuations in the periphery
# that collapse near the spindle center -- that would be a genuinely new
# observable reflecting the stochastic averaging over skMT-spindle contacts.

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
from chromlearn.io.trajectory import TrimmedCell, trim_trajectory, spindle_frame
from chromlearn.model_fitting import FitConfig
from chromlearn.model_fitting.basis import BSplineBasis
from chromlearn.model_fitting.diffusion import (
    COORDINATE_MAPS,
    DiffusionResult,
    estimate_diffusion_variable,
    local_diffusion_estimates,
)
from chromlearn.model_fitting.fit import fit_model
from chromlearn.model_fitting.plotting import plot_diffusion, plot_kernels
from chromlearn.model_fitting.simulate import simulate_cell

plt.rcParams["figure.dpi"] = 110

# %% [markdown]
# ## Load and fit the control model
#
# We use the `poles` topology with the same hyperparameters as NB04/05.
# The fit gives us the drift kernels and a scalar D; we then investigate
# whether D is better described as a function of position.

# %%
CONDITION = "rpe18_ctr"
cells_raw = load_condition(CONDITION)
cells = [trim_trajectory(c, method="neb_ao_frac") for c in cells_raw]
print(f"Loaded {len(cells)} {CONDITION} cells")

# Use the same config as NB04 (poles topology, 10 basis functions, 1e-3 reg)
config = FitConfig(
    topology="poles",
    n_basis_xx=10,
    n_basis_xy=10,
    r_min_xx=0.3,
    r_max_xx=15.0,
    r_min_xy=0.3,
    r_max_xy=15.0,
    basis_type="bspline",
    lambda_ridge=1e-3,
    lambda_rough=1.0,
    basis_eval_mode="ito",
    dt=5.0,
)

model = fit_model(cells, config)
print(f"Scalar D = {model.D_x:.6f} um^2/s")

# %% [markdown]
# ## Local D estimates from all four estimators
#
# We compute per-particle, per-timepoint D estimates using four methods:
#
# | Estimator      | Formula                                          | Bias properties |
# |----------------|--------------------------------------------------|-----------------|
# | MSD            | |dX|^2 / (2d dt)                                 | Simple, 2-point |
# | Vestergaard    | 3-point, cancels localization noise               | Noise-robust    |
# | Weak-noise     | |dX - dX_prev|^2 / (4d dt)                       | Drift-robust    |
# | F-corrected    | |dX - F_pred*dt|^2 / (2d dt)                     | Subtracts drift |
#
# If the spatial gradient is robust across estimators, it is more likely
# biological.  If it only appears in one estimator, suspect artifact.

# %%
ESTIMATORS = ["msd", "vestergaard", "weak_noise", "f_corrected"]
COORD = "distance"  # distance from spindle center

D_locals_by_est: dict[str, list[np.ndarray]] = {}
for est in ESTIMATORS:
    kwargs = {}
    if est == "f_corrected":
        kwargs = dict(
            fit_result=model,
            basis_xx=model.basis_xx,
            basis_xy=model.basis_xy,
            topology=model.topology,
        )
    D_locals_by_est[est] = local_diffusion_estimates(
        cells, dt=config.dt, mode=est, **kwargs
    )
    all_vals = np.concatenate([d.ravel() for d in D_locals_by_est[est]])
    valid = all_vals[np.isfinite(all_vals)]
    print(f"{est:14s}: {len(valid):>7,} valid samples, "
          f"mean D = {np.mean(valid):.6f}, median = {np.median(valid):.6f} um^2/s")

# %% [markdown]
# ## Fit D(distance) with basis expansion for each estimator
#
# We use a B-spline basis along the distance-from-spindle-center coordinate
# and ridge regression to get a smooth D(x) profile.

# %%
D_COORD = "distance"
N_BASIS_D = 8
R_MIN_D = 0.5
R_MAX_D = 12.0
LAMBDA_D = 1e-2

basis_D = BSplineBasis(R_MIN_D, R_MAX_D, N_BASIS_D)
eval_coords = np.linspace(R_MIN_D, R_MAX_D, 200)

diff_results: dict[str, DiffusionResult] = {}
for est in ESTIMATORS:
    kwargs = {}
    if est == "f_corrected":
        kwargs = dict(
            fit_result=model,
            basis_xx=model.basis_xx,
            basis_xy=model.basis_xy,
        )
    diff_results[est] = estimate_diffusion_variable(
        cells,
        basis_D=BSplineBasis(R_MIN_D, R_MAX_D, N_BASIS_D),
        coord_name=D_COORD,
        dt=config.dt,
        mode=est,
        lambda_ridge=LAMBDA_D,
        topology=model.topology,
        **kwargs,
    )

# %%
# Overlay all four estimator D(x) profiles
fig, ax = plt.subplots(figsize=(8, 5))
colors = {"msd": "C0", "vestergaard": "C1", "weak_noise": "C2", "f_corrected": "C3"}

for est in ESTIMATORS:
    D_vals = diff_results[est].evaluate(eval_coords)
    ax.plot(eval_coords, D_vals, color=colors[est], linewidth=2, label=est)

ax.axhline(model.D_x, color="0.5", linestyle="--", linewidth=0.8, label="Scalar D (MSD residual)")
ax.set_xlabel("Distance from spindle center (um)")
ax.set_ylabel("D (um$^2$/s)")
ax.set_title("Effective diffusion vs distance -- all estimators")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Per-cell D(x) consistency
#
# We check whether the spatial gradient is consistent across individual cells
# or driven by one or two outliers.  For each cell, we fit D(distance) using
# the Vestergaard estimator (a good compromise between noise-robustness and
# simplicity).

# %%
PERCELL_EST = "vestergaard"
fig, ax = plt.subplots(figsize=(8, 5))

percell_D_curves = []
for cell in cells:
    dr = estimate_diffusion_variable(
        [cell],
        basis_D=BSplineBasis(R_MIN_D, R_MAX_D, N_BASIS_D),
        coord_name=D_COORD,
        dt=config.dt,
        mode=PERCELL_EST,
        lambda_ridge=LAMBDA_D,
        topology=model.topology,
    )
    D_curve = dr.evaluate(eval_coords)
    percell_D_curves.append(D_curve)
    ax.plot(eval_coords, D_curve, color="C0", alpha=0.25, linewidth=1)

percell_D_curves = np.array(percell_D_curves)
ax.plot(eval_coords, np.median(percell_D_curves, axis=0),
        color="k", linewidth=2.5, label="Median across cells")
ax.fill_between(
    eval_coords,
    np.percentile(percell_D_curves, 25, axis=0),
    np.percentile(percell_D_curves, 75, axis=0),
    color="C0", alpha=0.15, label="IQR",
)
# Overlay pooled fit
pooled_D = diff_results[PERCELL_EST].evaluate(eval_coords)
ax.plot(eval_coords, pooled_D, color="C1", linewidth=2, linestyle="--",
        label=f"Pooled fit ({PERCELL_EST})")

ax.set_xlabel("Distance from spindle center (um)")
ax.set_ylabel("D (um$^2$/s)")
ax.set_title(f"Per-cell D(distance) -- {PERCELL_EST} estimator")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## D along alternative coordinate axes
#
# Check whether the signal looks different along the axial (spindle-axis
# projection) vs radial (perpendicular distance from spindle axis) coordinates.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
coord_names = ["distance", "axial", "radial"]
coord_domains = {
    "distance": (0.5, 12.0),
    "axial": (-8.0, 8.0),
    "radial": (0.5, 10.0),
}

for idx, coord in enumerate(coord_names):
    rlo, rhi = coord_domains[coord]
    dr = estimate_diffusion_variable(
        cells,
        basis_D=BSplineBasis(rlo, rhi, N_BASIS_D),
        coord_name=coord,
        dt=config.dt,
        mode="vestergaard",
        lambda_ridge=LAMBDA_D,
        topology=model.topology,
    )
    x = np.linspace(rlo, rhi, 200)
    axes[idx].plot(x, dr.evaluate(x), color="C0", linewidth=2)
    axes[idx].axhline(dr.D_scalar, color="0.5", linestyle="--", linewidth=0.8)
    axes[idx].set_xlabel(f"{coord} coordinate (um)")
    axes[idx].set_ylabel("D (um$^2$/s)")
    axes[idx].set_title(f"D({coord})")

fig.suptitle("Diffusion landscape along three coordinate axes (Vestergaard)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Rollout validation: scalar D model
#
# As a sanity check, we simulate a few representative cells with the fitted
# model (scalar D) and compare spindle-frame trajectories to real data.

# %%
from chromlearn.model_fitting.simulate import simulate_cell

# Pick a few representative cells
ROLLOUT_CELLS = [0, len(cells) // 2, len(cells) - 1]
ROLLOUT_SEED = 42

fig, axes = plt.subplots(len(ROLLOUT_CELLS), 2, figsize=(12, 4 * len(ROLLOUT_CELLS)),
                         squeeze=False)

for row, cell_idx in enumerate(ROLLOUT_CELLS):
    cell = cells[cell_idx]
    T, _, N = cell.chromosomes.shape

    _, sim_cell = simulate_cell(cell, model,
                                rng=np.random.default_rng(ROLLOUT_SEED + cell_idx))

    sf_real = spindle_frame(cell)
    sf_sim = spindle_frame(sim_cell)

    time_axis = np.arange(T) * config.dt

    # Plot radial (distance to spindle axis) for a subset of chromosomes
    n_show = min(10, N)
    for i in range(n_show):
        axes[row, 0].plot(time_axis, sf_real.radial[:, i], "k-", alpha=0.2, linewidth=0.5)
        axes[row, 0].plot(time_axis, sf_sim.radial[:, i], "C0-", alpha=0.2, linewidth=0.5)
    axes[row, 0].plot(time_axis, np.nanmean(sf_real.radial, axis=1),
                      "k-", linewidth=2, label="Real mean")
    axes[row, 0].plot(time_axis, np.nanmean(sf_sim.radial, axis=1),
                      "C0--", linewidth=2, label="Sim mean (scalar D)")
    axes[row, 0].set_ylabel("Radial distance (um)")
    axes[row, 0].set_title(f"{cell.cell_id} -- radial")
    axes[row, 0].legend(fontsize=7)

    # Plot axial
    for i in range(n_show):
        axes[row, 1].plot(time_axis, sf_real.axial[:, i], "k-", alpha=0.2, linewidth=0.5)
        axes[row, 1].plot(time_axis, sf_sim.axial[:, i], "C0-", alpha=0.2, linewidth=0.5)
    axes[row, 1].plot(time_axis, np.nanmean(sf_real.axial, axis=1),
                      "k-", linewidth=2, label="Real mean")
    axes[row, 1].plot(time_axis, np.nanmean(sf_sim.axial, axis=1),
                      "C0--", linewidth=2, label="Sim mean (scalar D)")
    axes[row, 1].set_ylabel("Axial position (um)")
    axes[row, 1].set_title(f"{cell.cell_id} -- axial")
    axes[row, 1].legend(fontsize=7)

axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")
fig.suptitle("Rollout validation: real vs simulated (scalar D, poles model)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Diffusion-gradient correction magnitude
#
# In a full SFI treatment, D(x) enters the force inference jointly: the Ito
# drift includes a "spurious force" term grad(D).  Our two-stage approach
# (fit force first, then estimate D from residuals) neglects this correction.
# Here we check whether grad(D) is small compared to the inferred force,
# which would justify the decoupled approach.
#
# For radial pairwise forces, the relevant comparison is dD/dr vs F_xy(r)
# at the same distances, since the centrosome-chromosome interaction dominates
# the force budget.

# %%
# Numerical derivative of D(distance) from the Vestergaard fit
dr_vest = diff_results["vestergaard"]
eval_r = np.linspace(R_MIN_D + 0.2, R_MAX_D - 0.2, 180)
D_vals = dr_vest.evaluate(eval_r)

# Central-difference gradient dD/dr
dr_step = eval_r[1] - eval_r[0]
dD_dr = np.gradient(D_vals, dr_step)

# Inferred centrosome-chromosome force at the same distances
F_xy = model.evaluate_kernel("xy", eval_r)

# Also compute for f_corrected estimator as a cross-check
dr_fcorr = diff_results["f_corrected"]
D_vals_fc = dr_fcorr.evaluate(eval_r)
dD_dr_fc = np.gradient(D_vals_fc, dr_step)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left panel: force vs dD/dr
ax = axes[0]
ax.plot(eval_r, np.abs(F_xy), "C0-", linewidth=2, label="|F_xy(r)|")
ax.plot(eval_r, np.abs(dD_dr), "C2--", linewidth=2, label="|dD/dr| (Vestergaard)")
ax.plot(eval_r, np.abs(dD_dr_fc), "C3:", linewidth=2, label="|dD/dr| (f-corrected)")
ax.set_xlabel("Distance from spindle center (um)")
ax.set_ylabel("Magnitude (um/s or um$^2$/s/um)")
ax.set_title("Inferred force vs diffusion gradient")
ax.legend(fontsize=8)
ax.set_yscale("log")

# Right panel: ratio |dD/dr| / |F_xy|
ax = axes[1]
safe_F = np.where(np.abs(F_xy) > 1e-12, np.abs(F_xy), np.nan)
ratio_vest = np.abs(dD_dr) / safe_F
ratio_fc = np.abs(dD_dr_fc) / safe_F
ax.plot(eval_r, ratio_vest, "C2-", linewidth=2, label="Vestergaard")
ax.plot(eval_r, ratio_fc, "C3--", linewidth=2, label="f-corrected")
ax.axhline(0.1, color="0.5", linestyle=":", linewidth=1, label="10% threshold")
ax.set_xlabel("Distance from spindle center (um)")
ax.set_ylabel("|dD/dr| / |F_xy|")
ax.set_title("Diffusion gradient as fraction of inferred force")
ax.legend(fontsize=8)
ax.set_ylim(0, min(2.0, np.nanmax(ratio_vest) * 1.2))

fig.suptitle("Diffusion-gradient correction: magnitude check")
fig.tight_layout()
plt.show()

# %%
# Print summary statistics
median_ratio_vest = float(np.nanmedian(ratio_vest))
median_ratio_fc = float(np.nanmedian(ratio_fc))
max_ratio_vest = float(np.nanmax(ratio_vest[np.isfinite(ratio_vest)]))
max_ratio_fc = float(np.nanmax(ratio_fc[np.isfinite(ratio_fc)]))
print(f"Vestergaard:  median |dD/dr|/|F| = {median_ratio_vest:.3f}, "
      f"max = {max_ratio_vest:.3f}")
print(f"F-corrected:  median |dD/dr|/|F| = {median_ratio_fc:.3f}, "
      f"max = {max_ratio_fc:.3f}")
if median_ratio_vest < 0.1 and median_ratio_fc < 0.1:
    print("=> Diffusion gradient is small relative to inferred force.")
    print("   Two-stage (force first, then D) approach is justified.")
else:
    print("=> Diffusion gradient is NOT negligible. Consider joint inference.")

# %% [markdown]
# ## Summary
#
# **Questions answered:**
#
# 1. Is D(x) spatially varying? -- Check the multi-estimator overlay above.
# 2. Is the gradient robust? -- If all four estimator curves show the same
#    qualitative shape, the signal is credible.
# 3. Is it per-cell consistent? -- Check whether the per-cell curves cluster
#    tightly or scatter widely.
#
# **Interpretation guide:**
#
# - Higher D in the periphery + lower D near the center would suggest that
#   far-out chromosomes experience more stochastic fluctuation per step,
#   consistent with fewer effective skMT-spindle averaging contacts.
# - Flat D would mean the noise is spatially homogeneous and the scalar D is
#   sufficient.
# - D(axial) showing structure while D(radial) does not (or vice versa)
#   would point to anisotropic noise worth further investigation.
