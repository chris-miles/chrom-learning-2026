# chromlearn

Learning pairwise interaction kernels from chromosome and centrosome trajectories during mitosis, using Stochastic Force Inference (SFI/Ronceray).

## What this does

Infers effective distance-dependent forces between chromosomes (and from centrosomes to chromosomes) from 3D microscopy tracking data of dividing cells. Uses an overdamped Langevin model with pairwise radial interaction kernels expanded in a B-spline basis, solved via penalized linear regression.

## Structure

- `chromlearn/` — Python package
  - `io/` — Data loading, trajectory processing, cell catalog
  - `model_fitting/` — Basis functions, design matrix, regression, simulation, validation, diffusion estimation
  - `analysis/` — Supporting analyses (lag correlation, trajectory visualization, velocity-vs-distance)
- `notebooks/` — Jupyter-compatible `.py` notebooks (primary interface)
  - `01_explore_data.py` — Data loading, visualization, trajectory inspection
  - `02_velocity_spatial_not_temporal.py` — Velocity depends on distance, not time (binned comparison, effect sizes, chromosome-level permutation test)
  - `03_chromosomes_follow_centrosomes.py` — Justification for treating centrosomes as autonomous inputs (lag correlation, model comparison, forward simulation, physics argument)
  - `04_model_selection.py` — Compares 4 interaction topologies (poles, center, ±chromosomes) via cross-validation, kernel plots, physical plausibility, and forward simulation
  - `05_robustness.py` — Hyperparameter sensitivity: basis size, regularization, estimator mode, endpoint method, diffusion estimation
  - `06_diffusion_landscape.py` — Spatially-varying diffusion D(x): multi-estimator comparison, per-cell consistency, coordinate axis comparison
  - `07_per_cell_heterogeneity.py` — Per-cell kernel variability vs pooled bootstrap uncertainty, correlation with cell features
  - `08_cross_condition.py` — Cross-condition kernel comparison (control, Rod, CENP-E, PRC1)
- `data/` — Raw `.mat` trajectory files (not tracked in git)
- `docs/` — Design spec and planning documents

## Setup

```bash
pip install -e .
# or: pip install numpy scipy matplotlib jupyter h5py
```

## Data

Each `.mat` file contains one cell's tracked trajectories:
- ~46 chromosomes (3D positions from kinetochore tracking)
- 2 centrosomes (3D positions)
- Metadata: NEB frame, anaphase onset estimates

Files with `neb = NaN` are treated as anaphase-only and are excluded from the
prometaphase fitting pipeline.

Primary fitting dataset: `rpe18_ctr` NEB-annotated subset
(13 RPE1 control cells with NEB annotations, 5 s frame interval, micron units).

## Key options

All options are configured via `FitConfig` (see `chromlearn/model_fitting/__init__.py`):

- **Topology** (`topology`): `"poles"` (default), `"center"` (midpoint), `"poles_and_chroms"`, `"center_and_chroms"` — selects which pairwise interactions to include
- **Basis evaluation mode** (`basis_eval_mode`): `"ito"` (default), `"ito_shift"` (decorrelates localization noise), `"strato"` (midpoint, reduces finite-dt bias)
- **Diffusion estimator** (`diffusion_mode`): `"msd"` (default), `"vestergaard"` (noise-robust), `"weak_noise"` (drift-robust), `"f_corrected"` (subtracts inferred force)
- **Variable D** (`D_variable`): fit D as a function of position along the spindle axis, radial distance, or distance from spindle center
- **Endpoint method** (`endpoint_method`): `"neb_ao_frac"` (default, `endpoint_frac=0.5`) or `"end_sep"`
- **Basis type** (`basis_type`): `"bspline"` (default) or `"hat"`

## Docs

- `docs/design.md` — Full design specification (architecture, model, decisions, future extensions)
- `docs/llm_fitting_plan.md` — Mathematical background and roadmap
- `docs/specs/` — Detailed feature specs (multi-point estimators, variable diffusion, etc.)
- `docs/implementation_plan.md` — Step-by-step build plan
