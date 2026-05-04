# AGENTS.md
#
# This file mirrors CLAUDE.md. Keep them in sync when either is updated.

## Project

Python codebase for learning pairwise interaction kernels from chromosome/centrosome mitosis trajectories using Stochastic Force Inference (Ronceray/SFI approach).

## Architecture

Python package is `chromlearn/` with three subpackages:
- `chromlearn/io/` — Data loading (`.mat` files), trajectory trimming, spindle-frame transforms, cell catalog
- `chromlearn/model_fitting/` — Basis functions, design matrix, penalized regression, simulation, validation, plotting, multi-point estimators, variable D(x)
- `chromlearn/analysis/` — Independent analyses (lag correlation, trajectory visualization, velocity-vs-distance, PCA trajectory projection)

Notebooks in `notebooks/` are the primary interface. Raw data lives in `data/` (MATLAB `.mat` files). Old MATLAB code in `old_code/` for reference.

## Data conventions

- dt = 5 seconds, spatial units = microns
- Chromosome position = centroid of sister kinetochores (single 3D particle)
- Centrosomes are external/given (not modeled); justified in notebook 03
- Trajectories start at NEB, endpoint is configurable (default: `neb_ao_frac` with `frac=0.4`, capturing the early gathering phase before deep metaphase)
- Files with `neb = NaN` are anaphase-only and should be ignored
- Primary condition for fitting: `rpe18_ctr` (12 cells with NEB annotations after loading; 507 excluded as oblique outlier)

## Methodology and relation to SFI

This project uses SFI-inspired projection inference, not the full SFI/PASTIS pipeline. Key differences from the reference SFI implementation (github.com/ronceray/StochasticForceInference):

- **Model selection**: We compare a small set of physically motivated interaction topologies via leave-one-cell-out CV, rather than sparse selection over a large operator library (PASTIS). Primary criterion is leave-one-cell-out deterministic drift-rollout ensemble MSE at a fixed forecast horizon `H_PRIMARY = 10` frames (50 s with dt=5 s): a single noise-free ODE forward integration of the fitted drift field from real initial conditions, scored at the primary horizon. This matches the standard held-out metric across the dynamics-learning literature (Kipf et al. 2018 NRI, Rubanova et al. 2019 Latent ODE, Yildiz et al. 2019 ODE2VAE all report MSE at fixed multi-step horizons). The horizon parameter is exposed via `H_PRIMARY` in `notebooks/04_model_selection.py`; the full horizon-resolved curve (1-30 frames) is reported as a sweep diagnostic for the supplement. The residual at the chosen horizon is drift bias plus a topology-invariant data-noise floor. Rationale for deterministic rather than stochastic-ensemble rollout: (1) the fitted D may be dominated by measurement/tracking noise rather than genuine thermal fluctuations, making stochastic rollouts actively misleading; (2) the ensemble mean of many SDE replicates approximates the ODE solution when D is small and the force field is not strongly curved; (3) the deterministic rollout eliminates Monte Carlo noise from the CV score and is much cheaper. One-step velocity MSE, endpoint mismatch, final-frame Wasserstein, and horizon-resolved errors are reported as supporting diagnostics. Paired foldwise differences (delta/SE(delta)) and a parsimony rule assess whether topology rankings are statistically meaningful. Basis domains are fixed a priori from imaging resolution and spindle geometry to avoid preprocessing leakage. The stochastic rollout path (with replicate averaging) remains available via `deterministic=False` in `rollout_cross_validate()`.
- **Short-range xx test**: The free-form chromosome-chromosome (xx) kernel can fit arbitrary forces out to 10-15 um. The concern is that a long-range xx term may act as a flexible nuisance absorber — soaking up missing physics (e.g. common spindle transport) rather than reflecting genuine chromosome-chromosome biology. The `poles_and_chroms_short` topology tests this by restricting xx forces to short range (`r_cutoff_xx`, default 2.5 um) via explicit distance masks (not via basis `r_max`, since basis evaluation clamps rather than zeros out-of-domain distances). The cutoff is enforced consistently in feature construction, `evaluate_kernel()`, simulation, diffusion estimation (`f_corrected` mode), plotting (including bootstrap CI bands), and is persisted through model save/load. Interpretation: if `poles_and_chroms_short` performs comparably to full-range `poles_and_chroms`, the xx benefit is local excluded-volume/steric repulsion; if performance drops substantially, the long-range xx tail was capturing real structure (or absorbing missing model terms). A short-range-only xx kernel is biologically interpretable: chromosomes repel at close range but do not exert learned forces at spindle-scale distances.
- **Variable diffusion**: D(x) is estimated in a second stage from residuals, not jointly inferred with the force. Notebook 06 includes a quantitative check showing the diffusion-gradient correction (grad(D), the "spurious force" in Ito convention) is small relative to the inferred force, justifying the decoupled approach.
- **Regularization**: Only `lambda_rough` (integrated 2nd-derivative penalty controlling kernel smoothness) is a tuned hyperparameter. `lambda_ridge` is fixed at `1e-6` everywhere as numerical jitter for the normal-equations solve, not as a meaningful regularizer. Rationale: we are not interpreting individual basis coefficients or seeking sparsity in the coefficient vector; only the output kernel function predictions matter, so a coefficient-norm penalty has no physical role here. Notebook 05 sweeps `(n_basis, lambda_rough)` only.
- **Stochastic calculus convention**: Default is Ito; sensitivity to Ito/Ito-shift/Stratonovich is checked in notebook 05.
- **Paper framing**: "SFI-inspired projection inference with cross-validated interaction topologies."

## Code style

- Python 3.10+, numpy/scipy/matplotlib
- Dataclasses for configuration (`FitConfig`) and data containers (`CellData`)
- Notebooks are Jupytext percent-format `.py` files in `notebooks/` (source of truth for all edits)
- `.ipynb` files in `notebooks/ipynb/` are auto-generated; the pre-commit hook converts staged `.py` files (code-only, no execution). Run `bash scripts/execute_notebooks.sh` to regenerate with outputs. The `.ipynb` files may be out of date relative to the `.py` sources.
- Modules contain reusable logic; notebooks are the primary interface

## Active vs archived notebooks

Active notebooks (paper-relevant; primary interface):
- `01_explore_data.py` -- data loading, trajectory inspection
- `02_velocity_spatial_not_temporal.py` -- velocity-distance relation
- `02b_explore_chrom_pole_asymm.py`, `02c_chrom_pole_projection_test.py` -- pole/chromosome asymmetry diagnostics
- `03_chromosomes_follow_centrosomes.py` -- PCA + lag correlation, model comparison (Fig 3 panels A, B)
- `03b_force_partition_reconciliation.py` -- pp/cp partition reconciliation
- `04_model_selection.py` -- topology comparison, primary criterion (Fig 3 panel C kernels and forecast-vs-horizon)
- `05_robustness.py` -- hyperparameter sensitivity, basis sweep
- `06_diffusion_landscape.py` -- D(x) (Fig 3 panel D)
- `07_per_cell_heterogeneity.py` -- per-cell kernel variability

Archived in `notebooks/archive/` (superseded or out of paper scope):
- `08_cross_condition.py` -- cross-condition comparison (out of scope for current paper)
- `09_neural_relational_inference.py` -- NRI-lite topology validation (out of scope; required `torch`)
- `debug_centering_vs_frac.py` -- one-off diagnostic
