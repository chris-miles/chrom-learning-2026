# CLAUDE.md
#
# AGENTS.md mirrors this file. Keep them in sync when either is updated.

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

- **Model selection**: We compare a small set of physically motivated interaction topologies via leave-one-cell-out CV, rather than sparse selection over a large operator library (PASTIS). Primary criterion is leave-one-cell-out ensemble-mean MSE (simulated positions averaged across replicates before comparing to reality, cancelling model-side stochastic variance and leaving drift bias plus a topology-invariant data-noise floor). This scores the conditional-mean trajectory, not the full stochastic distribution; it is appropriate here because the goal is drift/topology selection and the diffusion landscape is nearly spatially uniform (notebook 06). Per-rep path MSE, one-step velocity MSE, endpoint mismatch, final-frame Wasserstein, and horizon-specific errors are reported as supporting diagnostics. Paired foldwise differences (Δ/SE(Δ)) and a parsimony rule are used to assess whether topology rankings are statistically meaningful. Basis domains are fixed a priori from imaging resolution and spindle geometry to avoid preprocessing leakage. Rollout CV uses common random numbers (same RNG seed) across topologies for paired comparison.
- **Short-range xx test**: The free-form chromosome-chromosome (xx) kernel can fit arbitrary forces out to 10-15 um. The concern is that a long-range xx term may act as a flexible nuisance absorber — soaking up missing physics (e.g. common spindle transport) rather than reflecting genuine chromosome-chromosome biology. The `poles_and_chroms_short` topology tests this by restricting xx forces to short range (`r_cutoff_xx`, default 2.5 um) via explicit distance masks (not via basis `r_max`, since basis evaluation clamps rather than zeros out-of-domain distances). The cutoff is enforced consistently in feature construction, `evaluate_kernel()`, simulation, diffusion estimation (`f_corrected` mode), plotting (including bootstrap CI bands), and is persisted through model save/load. Interpretation: if `poles_and_chroms_short` performs comparably to full-range `poles_and_chroms`, the xx benefit is local excluded-volume/steric repulsion; if performance drops substantially, the long-range xx tail was capturing real structure (or absorbing missing model terms). A short-range-only xx kernel is biologically interpretable: chromosomes repel at close range but do not exert learned forces at spindle-scale distances.
- **Variable diffusion**: D(x) is estimated in a second stage from residuals, not jointly inferred with the force. Notebook 06 includes a quantitative check showing the diffusion-gradient correction (grad(D), the "spurious force" in Ito convention) is small relative to the inferred force, justifying the decoupled approach.
- **Stochastic calculus convention**: Default is Ito; sensitivity to Ito/Ito-shift/Stratonovich is checked in notebook 05.
- **NRI validation**: Notebook 09 uses a Neural Relational Inference (NRI-lite) model as independent topology validation. The variational graph encoder infers latent edge types from trajectory windows; if it assigns high activity to pole->chromosome edges, that confirms the SFI topology selection without sharing any of the SFI assumptions.
- **Paper framing**: "SFI-inspired projection inference with cross-validated interaction topologies."

## Code style

- Python 3.10+, numpy/scipy/matplotlib
- Dataclasses for configuration (`FitConfig`) and data containers (`CellData`)
- PyTorch used only in notebook 09 (NRI-lite); all other code is numpy/scipy
- Notebooks are Jupytext percent-format `.py` files in `notebooks/` (source of truth for all edits)
- `.ipynb` files in `notebooks/ipynb/` are auto-generated; the pre-commit hook converts staged `.py` files (code-only, no execution). Run `bash scripts/execute_notebooks.sh` to regenerate with outputs. The `.ipynb` files may be out of date relative to the `.py` sources.
- Modules contain reusable logic; notebooks are the primary interface
