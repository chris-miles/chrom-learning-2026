# Notebooks 3 & 4: Model Selection and Robustness

## Overview

Notebook 3 compares four chromosome interaction topologies to determine which
pairwise forces are needed. Notebook 4 tests the sensitivity of the winning
model to hyperparameter and estimator choices.

## Model topologies (Notebook 3)

| Model | Label | Partners | Kernels |
|-------|-------|----------|---------|
| M1 | `poles` | Each centrosome independently | f_xy(r) |
| M2 | `center` | Centrosome midpoint | f_xc(r) |
| M3 | `poles_and_chroms` | Each centrosome + other chromosomes | f_xy(r) + f_xx(r) |
| M4 | `center_and_chroms` | Centrosome midpoint + other chromosomes | f_xc(r) + f_xx(r) |

The chromosome-chromosome kernel f_xx is a single shared kernel over pairwise
distance (no sister/non-sister distinction at this level).

## Code changes

### Partner-list abstraction in features.py

Refactor `_build_cell_design_matrix` so that the "pole" interaction block
operates on a generic `partners` array of shape `(n_partners, T, 3)` rather
than reading `cell.centrioles` directly. The caller constructs the partner
array based on the topology:

- `"poles"`: stack pole1, pole2 → (2, T, 3)
- `"center"`: compute midpoint → (1, T, 3)

The partner array is stacked into `eval_poles` exactly as the current code
does with `centrioles[t].T`. The vectorized einsum contraction is unchanged;
only the input array differs. No per-partner Python loops.

When `basis_xx is None`, the chromosome-chromosome block is skipped entirely
and the design matrix has only the partner columns.

### basis_xx=None contract

When a topology excludes chromosome-chromosome interactions, `basis_xx=None`
propagates through the pipeline with these rules:

- **features.py:** `n_cols = n_bxy` (no xx columns). The xx einsum block is
  skipped. `build_design_matrix` returns `(n_obs, n_bxy)`.
- **fit.py / `_block_roughness`:** Returns only `R_xy` (no xx roughness
  block). The penalty matrix is `(n_bxy, n_bxy)`.
- **fit.py / `cross_validate`, `bootstrap_kernels`:** `theta_samples` shape
  uses `n_bxy` only. These functions should accept `FitConfig` and construct
  bases internally (paralleling `fit_model`) to reduce signature drift.
- **model.py / `FittedModel`:** `n_basis_xx = 0`, `theta_xx` returns empty
  array. `evaluate_kernel("f_xx", r)` returns `None`.
- **model.py / `save`/`load`:** Handle `n_basis_xx=0`. `load` defaults
  `topology="poles"` when field is absent (backward compat with old files).
- **plotting.py:** Shows one kernel panel instead of two when xx is absent.

### Partner distance domains

The `"center"` topology produces systematically different chromosome-partner
distances than `"poles"` (closer to midpoint than to individual poles). Each
topology should estimate its own basis domain from the observed distance
distribution (e.g., 1st–99th percentile), or the notebook should set
`r_min_xy`/`r_max_xy` per topology. The spec does not mandate one approach;
the notebook should document whichever is used.

### Helper function: get_partners

Add `get_partners(cell: TrimmedCell, topology: str) -> np.ndarray` to
`chromlearn/io/trajectory.py`. Returns shape `(n_partners, T, 3)`:
- `"poles"` / `"poles_and_chroms"` → `cell.centrioles.transpose(2,0,1)`
- `"center"` / `"center_and_chroms"` → `pole_center(cell)[np.newaxis]`

This centralizes partner construction out of the notebooks.

### FitConfig

Add field `topology: str = "poles"`. Valid values: `"poles"`, `"center"`,
`"poles_and_chroms"`, `"center_and_chroms"`.

Backward compatible: default is `"poles"`, which produces the same design
matrix as the current code.

### Threading through fit.py

`fit_model`, `cross_validate`, `bootstrap_kernels` accept `FitConfig` and
construct bases internally. When topology includes chromosomes, `basis_xx`
is built from config; otherwise `basis_xx=None`.

### FittedModel (model.py)

Stores `topology` so downstream code knows which kernels exist. See
basis_xx=None contract above for detailed behavior.

### simulate.py

Rename `centrosome_positions` to `partner_positions` with shape
`(T, 3, n_partners)`. The inner loop iterates over `n_partners` (1 or 2)
instead of hardcoded `range(2)`. `generate_synthetic_data` gains a
`topology` parameter.

### plotting.py

Handle models with one or two kernel types. Panels adapt to what's available.
Kernel labels reflect the topology (e.g., "chromosome-pole" vs
"chromosome-center").

## Notebook 3: Model Selection

**File:** `notebooks/03_model_selection.py`

### Flow

1. **Setup** — Load `rpe18_ctr` cells, trim trajectories, set reasonable
   default hyperparameters (n_basis, lambda values).

2. **Define topologies** — Helper functions mapping `TrimmedCell` →
   partner arrays for each of the 4 models.

3. **Fit all 4 models** — Loop over topologies, fit each, store results.

4. **Cross-validation comparison** — Leave-one-cell-out CV error for each
   model. Bar chart or table comparing prediction errors.

5. **Learned kernel plots** — Side-by-side panels showing each model's
   learned kernels with bootstrap confidence bands.

6. **Physical plausibility** — For models with f_xx: inspect the
   chromosome-chromosome kernel. Flag nonphysical long-range forces.
   Discuss attractive/repulsive character and length scales. Specifically
   check for a repulsive barrier as r → 0 (excluded volume); short-range
   attraction between chromosomes is likely an artifact of optical
   diffraction or tracking merges and should be diagnosed as such.

7. **Forward simulation** — Simulate trajectories using learned kernels +
   estimated D for each model. Overlay with real trajectories in spindle
   frame. Visual comparison of congression/spreading behavior.

8. **Verdict** — Summarize which topology wins on CV error, plausibility,
   and simulation fidelity.

9. **Bonus: variable D(x) for the winner** — Refit the winning topology
   with variable D(x) (using the existing diffusion infrastructure).
   Compare scalar D vs D(x) via CV error. If variable D improves the
   fit, plot D(coordinate) and discuss whether the spatial variation is
   physically reasonable. Note: if D(x) varies significantly, the
   Ito/Stratonovich choice in NB4 becomes more consequential due to
   noise-induced drift (∇·D term).

## Notebook 4: Robustness and Hyperparameter Sensitivity

**File:** `notebooks/04_robustness.py`

### Flow

1. **Setup** — Load cells, fix the winning topology from Notebook 3.

2. **Basis size sweep** — Vary n_basis (4, 6, 8, 10, 12, 16, 20) for each
   kernel. Plot training error and CV error vs n_basis. Identify overfitting
   onset.

3. **Regularization sweep** — Log-spaced grid of lambda_ridge and
   lambda_rough. Heatmap or line plot of CV error. Check kernel shape
   stability across reasonable lambda range.

4. **Estimator mode comparison** — Fit with Ito, Ito-shift, Stratonovich.
   Compare CV errors and learned kernel shapes. Disagreement flags
   finite-dt bias.

5. **Endpoint method comparison** — Fit with midpoint_neb_ao, ao_mean,
   end_sep. Test sensitivity to trajectory time window choice.

6. **Diffusion estimation comparison** — Compare scalar D across modes
   (MSD, Vestergaard, weak-noise, f-corrected). Optionally test variable
   D(x) vs scalar D.

7. **Summary** — Table of recommended hyperparameters. Discussion of which
   choices matter and which are inconsequential.

### Iterative refinement

If Notebook 4 reveals that a particular estimator or hyperparameter setting
materially changes the model topology ranking, Notebook 3 should be rerun
with those settings and the verdict updated.

## Future notebooks (rough sketches, not fully designed)

### Notebook 5 — Cell-to-cell heterogeneity

Per-cell fits, comparison of learned kernels across cells, clustering of
kernel shapes, identification of outlier cells. Tests whether a single
shared kernel is adequate or whether subpopulations exist.

**Time-stationarity note:** The current pipeline fits a single stationary
kernel across the trajectory window. Preliminary evidence from
`old_code/fig2_histogram_binned_v.m` (a MATLAB script that bins
velocities by time or distance) suggests that the dynamics are
approximately time-stationary and depend more on spatial position than
on time. A Python reproduction of this analysis belongs here in NB5 to
formally justify the stationarity assumption used in NB3/NB4.

### Notebook 6 — Angular and rotational effects (bonus)

Investigates whether chromosome forces depend on orientation relative to
the spindle axis, not just radial distance. This extends the isotropic
pairwise kernel f(r) to an anisotropic kernel f(r, theta).

**Scientific questions:**
- Do chromosomes experience different forces along the spindle axis vs
  perpendicular to it? (Axial-radial anisotropy)
- Does the angle between a chromosome's KT-KT axis and the spindle axis
  affect its dynamics? (Chromosome orientation coupling)
- Are there rotational/torque-like forces visible in the residuals of the
  isotropic model?

**Possible model extensions:**
1. **Axial/radial decomposition:** Decompose the chromosome-pole force
   into spindle-axis and perpendicular components, each with its own
   kernel: f_axial(r_axial) + f_radial(r_radial). The partner-list
   abstraction supports this — the "feature extractor" computes
   (r_axial, r_radial) instead of scalar r, and the basis becomes 2D
   (or two separate 1D bases on each coordinate).
2. **Angular basis terms:** Add angular basis functions phi(theta) that
   multiply the radial kernel: f(r, theta) = sum_b c_b * phi_b(r) *
   psi_b(theta). The einsum contraction generalizes — phi becomes a
   product of radial and angular evaluations, direction vectors remain
   the same.
3. **KT-KT orientation coupling:** If sister kinetochore positions are
   available (not centroids), compute the KT-KT axis orientation
   relative to the spindle and test whether it correlates with
   chromosome velocity residuals. This is more exploratory / model-free.

**Code implications:** The partner-list + feature-extractor design in
features.py accommodates this. The key change is that the "distance
computation" step generalizes to a "coordinate computation" step that
returns (r, theta) or (r_axial, r_radial) instead of just r. The basis
evaluation and einsum contraction patterns stay the same. This is why
the partner interface is kept generic rather than hardcoding pole logic.

**Approach:** Start with residual analysis from the isotropic model
(NB3 winner). If residuals show systematic angular structure (e.g.,
plot residuals vs spindle-axis angle), then fit the extended models
and compare via CV.

## Implementation notes

These details must be followed to prevent runtime errors during
implementation:

1. **Shape consistency:** `get_partners` returns `(n_partners, T, 3)`.
   `simulate_trajectories` must adopt the same convention for
   `partner_positions` — i.e., `(n_partners, T, 3)`, not the current
   `(T, 3, n_poles)`. Access pattern becomes
   `partner_positions[partner_index, step]`.

2. **kernel_xx=None in simulation:** `simulate_trajectories` and
   `generate_synthetic_data` must skip the chromosome-chromosome force
   loop when `kernel_xx is None`, not just when `basis_xx is None` in
   the fitting path.

3. **Defensive save/load:** `FittedModel.save()` must guard
   `self.basis_xx is not None` before accessing `self.basis_xx.r_min`,
   `self.basis_xx.basis_type`, etc. to avoid `AttributeError`.

4. **Topology in build_design_matrix signature:** Add
   `topology: str = "poles"` to `build_design_matrix` so it can call
   `get_partners` internally and pass the partners array down to
   `_build_cell_design_matrix`.

5. **Type hints:** Update `basis_xx` type annotations to
   `BSplineBasis | HatBasis | None` in features.py, fit.py, and
   model.py.

## Regression tests

After the partner-list refactor, verify that `topology="poles"` produces
bit-identical design matrices to the pre-refactor code on the same input
cells. This is a one-time regression check during implementation.

## Performance constraints

All numerical changes must preserve the current vectorization strategy.
The partner-list refactor stacks arrays before the existing einsum
contractions — no new Python-level loops over partners or timepoints.
