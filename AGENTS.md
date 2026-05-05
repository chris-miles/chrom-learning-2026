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

- **Model selection**: We compare a small set of physically motivated interaction topologies via leave-one-cell-out CV, rather than sparse selection over a large operator library (PASTIS). **Primary criterion**: leave-one-cell-out deterministic drift-rollout *path MSE* — full-trajectory ensemble MSE over the trimmed early-prometaphase window (NEB to `frac=0.4` of NEB-AO, ~150 s with dt=5 s). Path MSE integrates the horizon-resolved error over the predeclared analysis window, avoiding an arbitrary single-horizon choice. **Supporting diagnostics** (computed and reported, not used for selection): from-NEB ensemble MSE at `H_PRIMARY = 10` frames (Alex's docx anchor — *"show held-out forecast error vs horizon for up to 10 frames"*), the full horizon-resolved curve (1-30 frames), rolling-window forecast at h=1..30, 1-step velocity MSE, final-frame Wasserstein, endpoint MSE. **Topology admissibility**: `poles_and_chroms` and `center_and_chroms` (free-form full-range xx) are ruled out a priori as biologically inadmissible — there is no known biological basis for long-range chromosome-chromosome forces in mammalian mitosis. They are reported as flexible nuisance-absorbing upper bounds (they may capture missing physics like common spindle transport) but are not physically interpretable. Selection is over the biologically admissible set: `poles`, `center`, `poles_and_chroms_enveloped`. Paired foldwise differences (delta/SE(delta)) on path MSE quantify whether gaps between admissible topologies are statistically meaningful. Rationale for deterministic rather than stochastic-ensemble rollout: (1) the fitted D may be dominated by measurement/tracking noise rather than genuine thermal fluctuations, making stochastic rollouts actively misleading; (2) the ensemble mean of many SDE replicates approximates the ODE solution when D is small and the force field is not strongly curved; (3) the deterministic rollout eliminates Monte Carlo noise from the CV score and is much cheaper. Basis domains are fixed a priori from imaging resolution and spindle geometry to avoid preprocessing leakage. The stochastic rollout path remains available via `deterministic=False` in `rollout_cross_validate()`. Held-out metric tradition references: Kipf et al. 2018 NRI, Rubanova et al. 2019 Latent ODE, Yildiz et al. 2019 ODE2VAE all report MSE at fixed multi-step horizons; we report both single-horizon (h=10) and full-trajectory (path MSE) variants.
- **Steric envelope on the chrom-chrom (xx) kernel**: The biologically admissible chrom-chrom interaction is short-range steric repulsion only (kinetochore + chromatid contact ~1-2 um in RPE-1 cells; Renda 2020, Gallego 2010). The `poles_and_chroms_enveloped` topology encodes this prior by multiplying the xx B-spline basis by a smooth steric envelope `s(r) = 0.5 * (1 - tanh((r - r0) / w))` with `r0 = 1.5 um`, `w = 0.3 um`, so the kernel is ~1 below 1 um, 0.5 at 1.5 um, ~0 by 2 um. The envelope is a fixed prior (not CV-tuned); the params are not swept in `notebooks/05_robustness.py`. Implemented as a thin `EnvelopedBasis` wrapper around the underlying B-spline (`chromlearn/model_fitting/basis.py`). Because the envelope lives in the basis itself, downstream code (`features.py`, `evaluate_kernel`, simulation, plotting, `diffusion.py:f_corrected`) is unaffected — every consumer just calls `basis.evaluate(r)`. The envelope params are persisted through model save/load inside the `basis_xx` payload. The deprecated hard-cutoff path (`r_cutoff_xx`) is retained on `FitConfig` and `FittedModel` for backward compatibility with old saved models, but is mutually exclusive with the envelope path; new fits should use the envelope.
- **Variable diffusion**: D(x) is estimated in a second stage from residuals, not jointly inferred with the force. Notebook 06 includes a quantitative check showing the diffusion-gradient correction (grad(D), the "spurious force" in Ito convention) is small relative to the inferred force, justifying the decoupled approach.
- **Regularization**: Only `lambda_rough` (integrated 2nd-derivative penalty controlling kernel smoothness) is a tuned hyperparameter. `lambda_ridge` is fixed at `1e-6` everywhere as numerical jitter for the normal-equations solve, not as a meaningful regularizer. Rationale: we are not interpreting individual basis coefficients or seeking sparsity in the coefficient vector; only the output kernel function predictions matter, so a coefficient-norm penalty has no physical role here. Notebook 05 sweeps `(n_basis, lambda_rough)` only.
- **Stochastic calculus convention**: Default is Ito; sensitivity to Ito/Ito-shift/Stratonovich is checked in notebook 05.
- **Paper framing**: "SFI-inspired projection inference with cross-validated interaction topologies."

## Version tags

- `stable-apr3-2026` — Stable baseline before envelope refactor (commit `deff017`). All notebooks, CV pipeline, and the (deprecated) `r_cutoff_xx` hard-cutoff path working. Roll back with `git checkout stable-apr3-2026`.

## Code style

- Python 3.10+, numpy/scipy/matplotlib
- Dataclasses for configuration (`FitConfig`) and data containers (`CellData`)
- Notebooks are Jupytext percent-format `.py` files in `notebooks/` (source of truth for all edits)
- `.ipynb` files in `notebooks/ipynb/` are auto-generated; the pre-commit hook converts staged `.py` files (code-only, no execution). Run `bash scripts/execute_notebooks.sh` to regenerate with outputs. The `.ipynb` files may be out of date relative to the `.py` sources.
- Modules contain reusable logic; notebooks are the primary interface

## Active vs archived notebooks

Active notebooks (paper-relevant; primary interface):
Paper-figure assemblers (skeleton; in progress):
- `00_main_figure.py` -- main-text Fig 3 assembler (panels A: PCA, B: lag, C: kernels, D: D(x)). Loads canonical envelope model and bootstrap; panel implementations TODO.
- `00b_supplement.py` -- supplement assembler (topology comparison, horizon forecast, hyperparameter robustness, per-cell variability, data-density / extrapolation shading, envelope methodology). Panels TODO.

Internal / technical notebooks (paper-relevant; primary interface for the underlying analysis):
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
