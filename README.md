# chromlearn

Learning pairwise interaction kernels from chromosome and centrosome trajectories during mitosis, using an approach inspired by [Stochastic Force Inference](https://github.com/ronceray/StochasticForceInference) (SFI/Ronceray).

## What this does

Infers effective distance-dependent forces between chromosomes (and from centrosomes to chromosomes) from 3D microscopy tracking data of dividing cells.

The core method treats chromosome motion as overdamped Langevin dynamics driven by pairwise radial forces. SFI projects the observed velocity field onto a basis of pairwise interaction functions and solves for the coefficients via penalized linear regression, recovering the effective force law without assuming a parametric model. We expand the interaction kernels in a B-spline basis and compare candidate interaction topologies (which pairs of particles interact) via leave-one-out cross-validation.

As independent validation, notebook 09 uses a [Neural Relational Inference](https://github.com/ethanfetaya/NRI) (NRI) approach: a variational graph encoder infers a latent interaction graph from trajectory windows, checking whether a neural model independently recovers the same topology.

## Structure

- `chromlearn/` — Python package
  - `io/` — Data loading, trajectory processing, cell catalog
  - `model_fitting/` — Basis functions, design matrix, regression, simulation, validation, diffusion estimation
  - `analysis/` — Supporting analyses (lag correlation, trajectory visualization, velocity-vs-distance)
- `notebooks/` — Jupytext percent-format `.py` notebooks (source of truth for all edits)
  - `ipynb/` — Auto-generated `.ipynb` files for GitHub rendering (may be out of date; regenerate with `bash scripts/execute_notebooks.sh`)
  - `01_explore_data.py` — Data loading, visualization, trajectory inspection
  - `02_velocity_spatial_not_temporal.py` — Velocity depends on distance, not time (binned comparison, effect sizes, chromosome-level permutation test)
  - `03_chromosomes_follow_centrosomes.py` — Justification for treating centrosomes as autonomous inputs (lag correlation, model comparison, forward simulation, physics argument)
  - `04_model_selection.py` — Compares 4 interaction topologies via leave-one-cell-out CV (primary: 1-step velocity MSE with paired fold-difference SEs; secondary: rollout validation), kernel plausibility, and forward simulation
  - `05_robustness.py` — Hyperparameter sensitivity: basis size, regularization, estimator mode, endpoint method, diffusion estimation
  - `06_diffusion_landscape.py` — Spatially-varying diffusion D(x): multi-estimator comparison, per-cell consistency, coordinate axis comparison
  - `07_per_cell_heterogeneity.py` — Per-cell kernel variability vs pooled bootstrap uncertainty, correlation with cell features
  - `08_cross_condition.py` — Cross-condition kernel comparison (control, Rod, CENP-E, PRC1)
  - `09_neural_relational_inference.py` — NRI-lite latent topology inference: variational graph encoder infers which edges matter, independently validating SFI topology (requires `torch`)
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

## Methodology

This project uses SFI-inspired projection inference with cross-validated interaction topologies. We fit pairwise radial kernels via penalized regression (as in SFI's projection framework) but differ from the full SFI/PASTIS pipeline in two ways: (1) model selection compares a small set of physically motivated topologies rather than sparse selection over a large basis library, and (2) spatially varying diffusion D(x) is estimated in a second stage from residuals rather than jointly inferred. Notebook 06 validates that the diffusion-gradient correction is negligible for our data.

Model topology is selected using leave-one-cell-out cross-validated one-step velocity MSE as the primary criterion. Paired foldwise loss differences are reported to assess whether score gaps between topologies are meaningful. Rollout validation, kernel plausibility checks, and NRI analysis serve as supporting evidence. Basis domains are fixed a priori from imaging resolution and spindle geometry.

### References

- **SFI (OLI)**: A. Frishman & P. Ronceray, *Learning force fields from stochastic trajectories*, Phys. Rev. X 10, 021009 (2020).
- **ULI**: D. B. Bruckner, P. Ronceray & C. P. Broedersz, *Inferring the dynamics of underdamped stochastic systems*, Phys. Rev. Lett. 125, 058103 (2020).
- **PASTIS**: A. Gerardos & P. Ronceray, *Parsimonious model selection for stochastic dynamics*, arXiv:2501.10339 (2025).
- **SFI code**: https://github.com/ronceray/StochasticForceInference
- **NRI**: T. Kipf, E. Fetaya, K.-C. Wang, M. Welling & R. Zemel, *Neural relational inference for interacting systems*, ICML 2018. Code: https://github.com/ethanfetaya/NRI

## Docs

Build-phase design docs, specs, and implementation plans are in `docs/archive/`.
