# chromlearn

Learning pairwise interaction kernels from chromosome and centrosome trajectories during mitosis, using Stochastic Force Inference (SFI/Ronceray).

## What this does

Infers effective distance-dependent forces between chromosomes (and from centrosomes to chromosomes) from 3D microscopy tracking data of dividing cells. Uses an overdamped Langevin model with pairwise radial interaction kernels expanded in a B-spline basis, solved via penalized linear regression.

## Structure

- `chromlearn/` — Python package
  - `io/` — Data loading, trajectory processing, cell catalog
  - `model_fitting/` — Basis functions, design matrix, regression, simulation, validation, diffusion estimation
  - `analysis/` — Supporting analyses (lag correlation, trajectory visualization)
- `notebooks/` — Jupyter notebooks (primary interface)
- `data/` — Raw `.mat` trajectory files
- `old_code/` — Reference MATLAB implementation
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
(`7` of `28` RPE1 control files, 5 s frame interval, micron units).

## Key options

All options are configured via `FitConfig` (see `chromlearn/model_fitting/__init__.py`):

- **Basis evaluation mode** (`basis_eval_mode`): `"ito"` (default), `"ito_shift"` (decorrelates localization noise), `"strato"` (midpoint, reduces finite-dt bias)
- **Diffusion estimator** (`diffusion_mode`): `"msd"` (default), `"vestergaard"` (noise-robust), `"weak_noise"` (drift-robust), `"f_corrected"` (subtracts inferred force)
- **Variable D** (`D_variable`): fit D as a function of position along the spindle axis, radial distance, or distance from spindle center
- **Endpoint method** (`endpoint_method`): `"midpoint_neb_ao"` (default), `"ao_mean"`, `"end_sep"`
- **Basis type** (`basis_type`): `"bspline"` (default) or `"hat"`

## Docs

- `docs/design.md` — Full design specification (architecture, model, decisions, future extensions)
- `docs/llm_fitting_plan.md` — Mathematical background and roadmap
