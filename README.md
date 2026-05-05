# chromlearn

Learning pairwise interaction kernels from chromosome and centrosome trajectories during mitosis, using an approach inspired by [Stochastic Force Inference](https://github.com/ronceray/StochasticForceInference) (SFI/Ronceray).

## What this does

Infers effective distance-dependent forces between chromosomes (and from centrosomes to chromosomes) from 3D microscopy tracking data of dividing cells.

The core method treats chromosome motion as overdamped Langevin dynamics driven by pairwise radial forces. SFI projects the observed velocity field onto a basis of pairwise interaction functions and solves for the coefficients via penalized linear regression, recovering the effective force law without assuming a parametric model. We expand the interaction kernels in a B-spline basis and compare candidate interaction topologies (which pairs of particles interact) via leave-one-out cross-validation.

## Structure

- `chromlearn/` — Python package
  - `io/` — Data loading, trajectory processing, cell catalog
  - `model_fitting/` — Basis functions, design matrix, regression, simulation, validation, diffusion estimation
  - `analysis/` — Supporting analyses (lag correlation, trajectory visualization, velocity-vs-distance, PCA trajectory projection)
- `notebooks/` — Jupytext percent-format `.py` notebooks (source of truth for all edits)
  - `ipynb/` — Auto-generated `.ipynb` files for GitHub rendering (may be out of date; regenerate with `bash scripts/execute_notebooks.sh`)
  - `01_explore_data.py` — Data loading, visualization, trajectory inspection
  - `02_velocity_spatial_not_temporal.py` — Velocity depends on distance, not time
  - `02b_explore_chrom_pole_asymm.py`, `02c_chrom_pole_projection_test.py` — Pole/chromosome asymmetry diagnostics
  - `03_chromosomes_follow_centrosomes.py` — Lag correlation, model comparison, forward simulation, physics argument (Fig 3 panels A, B)
  - `03b_force_partition_reconciliation.py` — Reconciles `02c` lab-frame and `03` regression views of the chromosome-pole force partition
  - `04_model_selection.py` — Compares 5 interaction topologies via leave-one-cell-out CV. **Primary criterion**: deterministic drift-rollout *path MSE* (full-trajectory ensemble MSE over the trimmed early-prometaphase window). Supporting diagnostics: ensemble MSE at `H_PRIMARY = 10` frames (Alex's docx anchor), rolling-window forecast (1-30 frames), 1-step velocity MSE, final-frame W1, endpoint MSE. Free-form xx topologies (`poles_and_chroms`, `center_and_chroms`) are reported as nuisance-absorbing upper bounds; selection is over the biologically admissible set (`poles`, `center`, `poles_and_chroms_enveloped`)
  - `05_robustness.py` — Hyperparameter sensitivity: (n_basis, lambda_rough) joint grid, estimator mode, endpoint method. `lambda_ridge` is fixed at `1e-6` (numerical jitter only); we don't interpret individual basis coefficients, so coefficient sparsity is not a goal
  - `06_diffusion_landscape.py` — Spatially-varying diffusion D(x): multi-estimator comparison, per-cell consistency (Fig 3 panel D)
  - `07_per_cell_heterogeneity.py` — Per-cell kernel variability vs pooled bootstrap uncertainty
  - `archive/` — Notebooks out of scope for the current paper (`08_cross_condition.py`, `09_neural_relational_inference.py`, `debug_centering_vs_frac.py`)
- `data/` — Raw `.mat` trajectory files (not tracked in git)

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

- **Topology** (`topology`): `"poles"` (default), `"center"` (midpoint), `"poles_and_chroms"`, `"center_and_chroms"` — selects which pairwise interactions to include. Notebook 04 also tests `"poles_and_chroms_enveloped"` (chrom-chrom kernel multiplied by a smooth steric envelope; see below)
- **Steric envelope on xx kernel** (`envelope_r0_xx`, `envelope_w_xx`): When both are set, the xx B-spline basis is multiplied by a smooth envelope `0.5 * (1 - tanh((r - r0) / w))` that decays to ~0 past `r0 + ~2w`, enforcing a biologically motivated short-range steric prior (kinetochore-scale, ~1-2 um for RPE-1 chromatids). Defaults `r0 = 1.5`, `w = 0.3` give a kernel that is ~1 below 1 um, 0.5 at 1.5 um, ~0 by 2 um. Implemented as a thin `EnvelopedBasis` wrapper around the underlying basis; the envelope is part of the basis itself, so feature construction, kernel evaluation, simulation, diffusion estimation, and plotting all work transparently. Persisted through model save/load. The deprecated hard-cutoff path (`r_cutoff_xx`) is retained for backward compatibility but is mutually exclusive with the envelope path
- **Basis evaluation mode** (`basis_eval_mode`): `"ito"` (default), `"ito_shift"` (decorrelates localization noise), `"strato"` (midpoint, reduces finite-dt bias)
- **Diffusion estimator** (`diffusion_mode`): `"msd"` (default), `"vestergaard"` (noise-robust), `"weak_noise"` (drift-robust), `"f_corrected"` (subtracts inferred force)
- **Variable D** (`D_variable`): fit D as a function of position along the spindle axis, radial distance, or distance from spindle center
- **Endpoint method** (`endpoint_method`): `"neb_ao_frac"` (default, `endpoint_frac=0.4`) or `"end_sep"`
- **Basis type** (`basis_type`): `"bspline"` (default) or `"hat"`

## Methodology

This project uses SFI-inspired projection inference with cross-validated interaction topologies. We fit pairwise radial kernels via penalized regression (as in SFI's projection framework) but differ from the full SFI/PASTIS pipeline in two ways: (1) model selection compares a small set of physically motivated topologies rather than sparse selection over a large basis library, and (2) spatially varying diffusion D(x) is estimated in a second stage from residuals rather than jointly inferred. Notebook 06 validates that the diffusion-gradient correction is negligible for our data.

Model topology is selected using leave-one-cell-out deterministic drift-rollout **path MSE** — full-trajectory ensemble MSE over the trimmed early-prometaphase window (NEB to `frac=0.4` of NEB-AO, ~150 s with `dt=5 s`). Path MSE integrates the horizon-resolved error over the predeclared analysis window, avoiding an arbitrary single-horizon choice. The from-NEB ensemble MSE at `H_PRIMARY = 10` frames (Alex's docx anchor), the full horizon-resolved curve (1-30 frames), the rolling-window forecast, 1-step velocity MSE, final-frame Wasserstein, and endpoint MSE are reported as supporting diagnostics — they show whether the path-MSE conclusion is hiding a single-horizon failure mode. Selection is restricted to the *biologically admissible* topology set (`poles`, `center`, `poles_and_chroms_enveloped`); the free-form xx variants (`poles_and_chroms`, `center_and_chroms`) are reported but treated as nuisance-absorbing upper bounds, since there is no known biological basis for long-range chromosome-chromosome forces in mammalian mitosis. Paired foldwise differences (Δ/SE(Δ)) on path MSE quantify whether gaps between admissible topologies are statistically meaningful. Basis domains are fixed a priori from imaging resolution and spindle geometry.

### References

- **SFI**: A. Frishman & P. Ronceray, *Learning force fields from stochastic trajectories*, Phys. Rev. X 10, 021009 (2020).
- **ULI**: D. B. Bruckner, P. Ronceray & C. P. Broedersz, *Inferring the dynamics of underdamped stochastic systems*, Phys. Rev. Lett. 125, 058103 (2020).
- **PASTIS**: A. Gerardos & P. Ronceray, *Parsimonious model selection for stochastic dynamics*, arXiv:2501.10339 (2025).
- **SFI code**: https://github.com/ronceray/StochasticForceInference
- **NRI** (held-out metric reference): T. Kipf, E. Fetaya, K.-C. Wang, M. Welling & R. Zemel, *Neural relational inference for interacting systems*, ICML 2018.
- **Latent ODE**: Y. Rubanova, R. T. Q. Chen & D. Duvenaud, *Latent ODEs for irregularly-sampled time series*, NeurIPS 2019.

