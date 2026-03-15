# CLAUDE.md

## Project

Python codebase for learning pairwise interaction kernels from chromosome/centrosome mitosis trajectories using Stochastic Force Inference (Ronceray/SFI approach).

## Key docs

- `docs/design.md` — Full design spec: architecture, model, all design decisions and future extensions
- `docs/llm_fitting_plan.md` — Mathematical background (model equations, basis expansion, regression, validation)

## Architecture

Python package is `chromlearn/` with three subpackages:
- `chromlearn/io/` — Data loading (`.mat` files), trajectory trimming, spindle-frame transforms, cell catalog
- `chromlearn/model_fitting/` — Basis functions, design matrix, penalized regression, simulation, validation, plotting
- `chromlearn/analysis/` — Independent analyses (lag correlation, trajectory visualization)

Notebooks in `notebooks/` are the primary interface. Raw data lives in `data/` (MATLAB `.mat` files). Old MATLAB code in `old_code/` for reference.

## Data conventions

- dt = 5 seconds, spatial units = microns
- Chromosome position = centroid of sister kinetochores (single 3D particle)
- Centrosomes are external/given (not modeled)
- Trajectories start at NEB, endpoint is configurable (default: midpoint NEB-AO)
- Primary condition: `rpe18_ctr` (28 cells), pooled for fitting

## Code style

- Python 3.10+, numpy/scipy/matplotlib
- Dataclasses for configuration (`FitConfig`) and data containers (`CellData`)
- No heavy ML frameworks in initial implementation
- Jupyter notebooks for running analyses; modules for reusable logic
