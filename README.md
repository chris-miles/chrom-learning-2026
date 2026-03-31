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
  - `analysis/` — Supporting analyses (lag correlation, trajectory visualization, velocity-vs-distance, PCA trajectory projection)
- `notebooks/` — Jupytext percent-format `.py` notebooks (source of truth for all edits)
  - `ipynb/` — Auto-generated `.ipynb` files for GitHub rendering (may be out of date; regenerate with `bash scripts/execute_notebooks.sh`)
  - `01_explore_data.py` — Data loading, visualization, trajectory inspection
  - `02_velocity_spatial_not_temporal.py` — Velocity depends on distance, not time (binned comparison, effect sizes, chromosome-level permutation test)
  - `03_chromosomes_follow_centrosomes.py` — Justification for treating centrosomes as autonomous inputs (lag correlation, model comparison, forward simulation, physics argument)
  - `04_model_selection.py` — Compares 5 interaction topologies via leave-one-cell-out CV (primary: ensemble-mean MSE with paired foldwise Δ/SE(Δ); supporting: per-rep path MSE, 1-step velocity MSE, W1, horizon-resolved error curves), metric concordance analysis, kernel plausibility, forward simulation, and PCA-space trajectory comparison (real vs rollout)
  - `05_robustness.py` — Hyperparameter sensitivity: joint (n_basis, ridge, roughness) grid sweep, estimator mode, endpoint method, diffusion estimation
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
prometaphase fitting pipeline. Cells with anomalous spindle geometry are moved
to subdirectories (`data/excluded_horizontal/`, `data/excluded_invagination/`,
`data/excluded_outlier/`) and excluded from fitting; notebook 01 visualizes why.

Primary fitting dataset: `rpe18_ctr` NEB-annotated subset
(12 RPE1 control cells with NEB annotations, 5 s frame interval, micron units;
507 excluded as oblique outlier).

## Key options

All options are configured via `FitConfig` (see `chromlearn/model_fitting/__init__.py`):

- **Topology** (`topology`): `"poles"` (default), `"center"` (midpoint), `"poles_and_chroms"`, `"center_and_chroms"` — selects which pairwise interactions to include. Notebook 04 also tests `"poles_and_chroms_short"` (short-range-only xx via `r_cutoff_xx`)
- **XX cutoff** (`r_cutoff_xx`): If set (e.g. 2.5 um), chromosome-chromosome forces are zeroed beyond this distance in fitting, kernel evaluation, simulation, and diffusion estimation. Persisted through model save/load. Tests whether xx benefit is local steric repulsion vs long-range nuisance absorption
- **Basis evaluation mode** (`basis_eval_mode`): `"ito"` (default), `"ito_shift"` (decorrelates localization noise), `"strato"` (midpoint, reduces finite-dt bias)
- **Diffusion estimator** (`diffusion_mode`): `"msd"` (default), `"vestergaard"` (noise-robust), `"weak_noise"` (drift-robust), `"f_corrected"` (subtracts inferred force)
- **Variable D** (`D_variable`): fit D as a function of position along the spindle axis, radial distance, or distance from spindle center
- **Endpoint method** (`endpoint_method`): `"neb_ao_frac"` (default, `endpoint_frac=1/3`) or `"end_sep"`
- **Basis type** (`basis_type`): `"bspline"` (default) or `"hat"`

## Methodology

This project uses SFI-inspired projection inference with cross-validated interaction topologies. We fit pairwise radial kernels via penalized regression (as in SFI's projection framework) but differ from the full SFI/PASTIS pipeline in two ways: (1) model selection compares a small set of physically motivated topologies rather than sparse selection over a large basis library, and (2) spatially varying diffusion D(x) is estimated in a second stage from residuals rather than jointly inferred. Notebook 06 validates that the diffusion-gradient correction is negligible for our data.

Model topology is selected using leave-one-cell-out ensemble-mean MSE (simulated positions averaged across replicates before comparing to reality, cancelling model-side stochastic variance) as the primary criterion. Paired foldwise differences (Δ/SE(Δ)) quantify whether gaps between topologies are statistically meaningful, with a parsimony rule favoring simpler models when the gap is within ~1 SE. Per-rep path MSE, one-step velocity MSE, endpoint mismatch, final-frame Wasserstein, and horizon-specific errors are reported as supporting diagnostics. Horizon-resolved curves for both path MSE and ensemble MSE show the diffusion noise floor and illustrate why ensemble MSE is more discriminative. Rollout CV uses common random numbers across topologies. Kernel plausibility checks and NRI analysis provide additional evidence. Basis domains are fixed a priori from imaging resolution and spindle geometry.

### References

- **SFI (OLI)**: A. Frishman & P. Ronceray, *Learning force fields from stochastic trajectories*, Phys. Rev. X 10, 021009 (2020).
- **ULI**: D. B. Bruckner, P. Ronceray & C. P. Broedersz, *Inferring the dynamics of underdamped stochastic systems*, Phys. Rev. Lett. 125, 058103 (2020).
- **PASTIS**: A. Gerardos & P. Ronceray, *Parsimonious model selection for stochastic dynamics*, arXiv:2501.10339 (2025).
- **SFI code**: https://github.com/ronceray/StochasticForceInference
- **NRI**: T. Kipf, E. Fetaya, K.-C. Wang, M. Welling & R. Zemel, *Neural relational inference for interacting systems*, ICML 2018. Code: https://github.com/ethanfetaya/NRI

## Docs

Build-phase design docs, specs, and implementation plans are in `docs/archive/`.
