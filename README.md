# chromlearn

Learning pairwise interaction kernels from chromosome and centrosome trajectories during mitosis, using Stochastic Force Inference (SFI/Ronceray).

## What this does

Infers effective distance-dependent forces between chromosomes (and from centrosomes to chromosomes) from 3D microscopy tracking data of dividing cells. Uses an overdamped Langevin model with pairwise radial interaction kernels expanded in a B-spline basis, solved via penalized linear regression.

## Structure

- `chromlearn/` — Python package
  - `io/` — Data loading, trajectory processing, cell catalog
  - `model_fitting/` — Basis functions, design matrix, regression, simulation, validation
  - `analysis/` — Supporting analyses (lag correlation, trajectory visualization)
- `notebooks/` — Jupyter notebooks (primary interface)
- `data/` — Raw `.mat` trajectory files
- `old_code/` — Reference MATLAB implementation
- `docs/` — Design spec and planning documents

## Setup

```bash
pip install numpy scipy matplotlib jupyter h5py
```

## Data

Each `.mat` file contains one cell's tracked trajectories:
- ~46 chromosomes (3D positions from kinetochore tracking)
- 2 centrosomes (3D positions)
- Metadata: NEB frame, anaphase onset estimates

Primary dataset: `rpe18_ctr` (28 RPE1 control cells, 5s frame interval, micron units).

## Docs

- `docs/design.md` — Full design specification (architecture, model, decisions, future extensions)
- `docs/llm_fitting_plan.md` — Mathematical background and roadmap
