# chromlearn: Learning Pairwise Interactions from Mitotic Chromosome Trajectories

## Overview

Python codebase for inferring effective pairwise interaction kernels between chromosomes and centrosomes during mitosis, using an overdamped Langevin / Stochastic Force Inference (SFI/Ronceray) framework. Replaces the previous MATLAB implementation with a cleaner architecture, modern tooling, and extensibility to new experimental conditions and model variants.

This repository also contains independent supporting analyses (e.g., lag correlation between centrosome and chromosome motion) that provide biological justification for modeling choices.

---

## Scientific Context

During mitosis, chromosomes and centrosomes interact through motor proteins and microtubule-mediated forces. We observe 3D trajectories of ~46 chromosomes and 2 centrosomes per cell, tracked at 5-second intervals in microns. The goal is to infer the effective distance-dependent forces governing chromosome motion from these noisy microscopy tracks.

The previous paper (Miles et al., "Transient dynein interactions drive rapid centripetal convergence of chromosomes in early prometaphase") used Chebyshev polynomial basis expansion in MATLAB. This rewrite adopts the Ronceray/SFI approach — designed for stochastic data with explicit drift + diffusion inference — rather than the Maggioni/Lu interaction-kernel framework.

---

## Model

### Governing equations

Overdamped Langevin dynamics for chromosome positions x_i(t) in R^3:

```
dx_i = F_i(X, Y) dt + sqrt(2 D_x) dW_i
```

where Y = {y_1, y_2} are the two centrosome positions, treated as **external/given** (not modeled).

### Pairwise radial interaction ansatz

```
F_i = sum_{k != i} f_xx(r_ik) * r_hat_ik      (chromosome-chromosome)
    + sum_j       f_xy(rho_ij) * rho_hat_ij    (centrosome-on-chromosome)
```

- r_ik = ||x_k - x_i||, r_hat_ik = (x_k - x_i) / r_ik
- rho_ij = ||y_j - x_i||, rho_hat_ij = (y_j - x_i) / rho_ij
- f_xx: chromosome-chromosome effective interaction kernel
- f_xy: centrosome effect on chromosome kernel

Centrosomes are not modeled (no f_yy or f_yx equations). Justification: lag correlation analysis shows chromosomes follow centrosomes, not vice versa.

### Basis expansion

Each kernel is expanded in a finite basis on [0, r_max]:

```
f_xx(r) = sum_m a_m * phi_m(r)
f_xy(r) = sum_m b_m * phi_m(r)
```

Default basis: cubic B-splines with 8-15 functions per kernel. Piecewise linear hat functions available for debugging. The previous MATLAB implementation used Chebyshev polynomials, which are intentionally replaced — B-splines have local support, making them better suited for regularization and compactness constraints.

Each kernel has its own basis domain since chromosome-chromosome and chromosome-centrosome distances have different typical ranges:
- f_xx: basis on [r_min_xx, r_max_xx], set from empirical chromosome-chromosome distance distribution (e.g., 5th to 95th percentile)
- f_xy: basis on [r_min_xy, r_max_xy], set from empirical chromosome-centrosome distance distribution

These are configurable parameters. Boundary condition: by default, kernels are not constrained at the cutoff, but an option to enforce f(r_max) = 0 can be added if needed for smoothness at the boundary.

### Regression

Euler-Maruyama discretization gives:

```
delta_x_i^n / dt ≈ F_i(t_n) + noise
```

Because the drift is linear in coefficients, this becomes a linear regression problem:

```
V ≈ G * theta + epsilon
```

where V = stacked displacement velocities, G = design matrix of basis features, theta = kernel coefficients.

Objective:

```
minimize ||V - G*theta||^2 + lambda_ridge * ||theta||^2 + lambda_rough * theta^T R theta
```

R is the roughness penalty matrix (integrated squared second derivative of the basis).

### Basis evaluation modes

The positions at which basis functions are evaluated can be shifted to reduce
noise bias (following Ronceray SFI v2):

- **Ito** (default): Basis evaluated at current positions X(t).
- **Ito-shift**: Basis evaluated at previous positions X(t-1).  Decorrelates
  localization noise from the displacement used as the response variable.
- **Stratonovich**: Basis evaluated at the midpoint (X(t)+X(t+1))/2.  Reduces
  finite-dt bias.

The velocity is always the forward difference (X(t+1)-X(t))/dt regardless of
mode.  Controlled by the `basis_eval_mode` parameter in `FitConfig` and
`build_design_matrix`.

### Diffusion estimation

Two-stage: fit drift first, then estimate D_x from residual variance.

**Scalar D (default):**
```
D_x = mean(residual^2) * dt / 2
```
where residuals are velocity residuals (displacement/dt minus predicted force).

**Local diffusion estimators** (per-particle, per-timepoint):

| Estimator | Formula | Notes |
|-----------|---------|-------|
| MSD (default) | \|dX\|^2 / (2 d dt) | Simple, 2-point |
| Vestergaard | (\|dX\|^2+\|dX_minus\|^2)/(4 d dt) + (dX.dX_minus)/(2 d dt) | 3-point, cancels localization noise |
| Weak-noise | \|dX-dX_minus\|^2 / (4 d dt) | 3-point, removes drift bias |
| F-corrected | \|dX-F*dt\|^2 / (2 d dt) | Subtracts inferred force first |

**Variable D(x):** D can optionally be expanded in a 1-D basis along a
configurable position coordinate (axial, radial, or distance from spindle
center).  Local D estimates are projected onto the basis via ridge regression.
See `chromlearn/model_fitting/diffusion.py`.

---

## Data

### Source format

MATLAB `.mat` files, one per cell. Each contains:
- `centrioles`: (time x 3 x 2) — 3D positions of two centrosome poles
- `kinetochores`: (time x 6 x N) — columns 1:3 and 4:6 are sister kinetochore positions per chromosome
- `neb`: scalar — frame index of nuclear envelope breakdown
- `ao1`, `ao2`: scalars — two estimates of anaphase onset frame
- `tracked`: scalar — number of tracked chromosomes

### Preprocessing

- Chromosome position = centroid of two sister kinetochores (mean of columns 1:3 and 4:6)
- Each chromosome treated as a single 3D particle
- dt = 5 seconds between frames
- Spatial units = microns

### Missing data handling

The `.mat` files may contain NaN entries for chromosomes that are not tracked at certain timepoints (e.g., out of focal plane, tracking lost). The pipeline handles this as follows:

- **Detection:** After computing chromosome centroids, any position that is NaN in any coordinate is flagged as missing for that chromosome at that timepoint.
- **Design matrix masking:** When building the design matrix, rows corresponding to missing observations are excluded. Specifically, if chromosome i has a NaN position at time t_n or t_{n+1}, the displacement for (i, n) is excluded from V, and the corresponding row is excluded from G. Similarly, when computing pairwise features for chromosome i at time t_n, any neighbor k with a NaN position is excluded from the pairwise sum.
- **No imputation:** We do not interpolate or fill missing positions. Missing data simply reduces the number of observations contributing to the regression.

Some files have `neb = NaN`. These are treated as anaphase-only datasets and
are excluded entirely from the prometaphase pipeline rather than being imputed
or trimmed from frame 1.

### Time window

Trajectories are trimmed from NEB to a configurable endpoint:
- `"neb_ao_frac"` (default): `NEB + frac * (AO - NEB)` where AO = mean(ao1, ao2).
  Default `frac=0.5` (midpoint); `frac=1.0` gives the full window to AO.
- `"end_sep"`: first frame where a running average of smoothed pole-pole distance
  reaches 95% of the metaphase-region maximum. Non-default; can be sensitive to
  smoothing parameters.

Same time window for all chromosomes within a cell (no per-chromosome attachment masking in initial implementation).

### Cell conditions

Primary focus: the `rpe18_ctr` NEB-annotated subset (7 of 28 files). Other
conditions available for future extension:
- rod311_ctr (34 cells), rod311_prc (9), rod311_rev (15)
- rpe18_rev (22), rpe18_cytoD (7), rpe18_hesp (4), rpe18_zm (2)

### Pooling strategy

All NEB-annotated `rpe18_ctr` cells are pooled into one design matrix for
primary fits. Bootstrap over trajectories for uncertainty. Per-cell fits planned
as future diagnostic.

---

## Architecture

```
chrom_learning_2026/               # repository root
├── chromlearn/                    # Python package
│   ├── __init__.py
│   ├── io/                        # Shared data layer
│   │   ├── __init__.py
│   │   ├── loader.py              # Load .mat files → CellData dataclass
│   │   ├── trajectory.py          # Time windowing, derived quantities, spindle frame
│   │   └── catalog.py             # Cell database, batch loading by condition
│   ├── model_fitting/             # SFI kernel learning pipeline
│   │   ├── __init__.py            # FitConfig dataclass
│   │   ├── basis.py               # B-spline / hat basis, roughness matrices
│   │   ├── features.py            # Design matrix construction (pairwise basis features)
│   │   ├── fit.py                 # Penalized regression, CV, bootstrap, diffusion estimation
│   │   ├── diffusion.py           # Coordinate maps, local D estimators, variable D(x) fitting
│   │   ├── model.py               # Fitted model container, kernel evaluation, save/load
│   │   ├── simulate.py            # Euler-Maruyama simulator, synthetic data generation
│   │   ├── validate.py            # Prediction error, residual diagnostics, recovery metrics
│   │   └── plotting.py            # Kernel plots, CV curves, residual plots, D(x) plots
│   └── analysis/                  # Independent supporting analyses
│       ├── __init__.py
│       ├── lag_correlation.py     # Velocity autocorrelation (centrosome vs chromosome)
│       ├── velocity_vs_distance.py # Binned velocity analysis: spatial vs temporal dependence
│       └── trajectory_viz.py      # Single-cell trajectory visualization, spindle-frame plots
├── notebooks/                     # Primary interface (percent-cell .py format)
│   ├── 01_explore_data.py         # Data loading, metadata, trajectory visualization
│   ├── 02_velocity_spatial_not_temporal.py   # Velocity depends on distance, not time
│   ├── 03_chromosomes_follow_centrosomes.py  # Centrosome autonomy justification
│   ├── 04_model_selection.py      # 4-topology comparison, CV, forward simulation
│   ├── 05_robustness.py           # Hyperparameter sensitivity sweeps
│   ├── 06_diffusion_landscape.py  # Spatially-varying D(x), multi-estimator
│   ├── 07_per_cell_heterogeneity.py  # Per-cell kernel variability
│   └── 08_cross_condition.py      # Cross-condition kernel comparison
├── data/                          # .mat files (existing raw data, not a Python package)
├── old_code/                      # MATLAB reference (existing)
└── docs/                          # Documentation (existing)
```

Note: The Python data-loading subpackage is named `chromlearn/io/` (not `chromlearn/data/`) to avoid collision with the top-level `data/` directory containing raw `.mat` files.

### Configuration

Parameters are collected in a `FitConfig` dataclass passed to fitting functions. This avoids scattered keyword arguments while staying lightweight (no config files for a research codebase).

```python
@dataclass
class FitConfig:
    # Time window
    endpoint_method: str = "neb_ao_frac"  # or "end_sep"
    endpoint_frac: float = 0.5  # fraction of [NEB, AO] window
    # Basis
    n_basis_xx: int = 10
    n_basis_xy: int = 10
    r_min_xx: float = 0.5    # set from empirical distance distribution
    r_max_xx: float = 10.0
    r_min_xy: float = 0.5
    r_max_xy: float = 12.0
    basis_type: str = "bspline"  # or "hat"
    # Regularization
    lambda_ridge: float = 1e-3
    lambda_rough: float = 1e-3
    # Data
    dt: float = 5.0  # seconds
    d: int = 3        # spatial dimension
    # Multi-point estimators
    basis_eval_mode: str = "ito"     # "ito", "ito_shift", or "strato"
    diffusion_mode: str = "msd"      # "msd", "vestergaard", "weak_noise", or "f_corrected"
    # Variable diffusion
    D_variable: bool = False
    n_basis_D: int = 6
    r_min_D: float = -8.0
    r_max_D: float = 8.0
    D_coordinate: str = "axial"      # "axial", "radial", or "distance"
    # Interaction topology
    topology: str = "poles"          # "poles", "center", "poles_and_chroms", "center_and_chroms"
```

The `topology` field controls which interaction partners are used:
- `"poles"`: Each centrosome independently (2 partners, no chromosome-chromosome)
- `"center"`: Centrosome midpoint only (1 partner, no chromosome-chromosome)
- `"poles_and_chroms"`: Each centrosome + chromosome-chromosome interactions
- `"center_and_chroms"`: Centrosome midpoint + chromosome-chromosome interactions

### Module responsibilities

**io/loader.py**
- `load_cell(path) -> CellData`: Parse .mat, extract centrioles, compute chromosome centroids, extract metadata
- Rejects files with `neb = NaN` as anaphase-only
- `CellData` dataclass: cell_id, condition, centrioles (T x 3 x 2), chromosomes (T x 3 x N), neb, ao1, ao2, tracked, dt

**io/trajectory.py**
- `trim_trajectory(cell, endpoint_method) -> TrimmedCell`: Apply time window
- `compute_end_sep(cell) -> int`: Detect 95%-plateau of smoothed pole-pole distance
- `spindle_frame(cell) -> SpindleFrameData`: Compute axial/radial coordinates relative to pole-pole axis (for visualization)
- `pole_pole_distance(cell) -> array`: Distance between centrosomes over time
- `pole_center(cell) -> array`: Midpoint of two centrosomes over time
- `get_partners(cell, topology) -> array`: Construct interaction partner array (n_partners, T, 3) based on topology

**io/catalog.py**
- `list_cells(condition) -> list[str]`: List available cell IDs for a condition
- `load_condition(condition, data_dir) -> list[CellData]`: Batch-load all valid cells for a condition. Discovers cells by glob-matching filenames with prefix `{condition}_*.mat` and filters out files with `neb = NaN`.
- `CONDITIONS`: dict mapping condition names to filename prefixes (e.g., `"rpe18_ctr"` → `"rpe18_ctr"`)

**model_fitting/basis.py**
- `BSplineBasis(r_min, r_max, n_basis) -> Basis`: Cubic B-spline basis on interval
- `HatBasis(r_min, r_max, n_basis) -> Basis`: Piecewise linear basis (debugging)
- `Basis.evaluate(r) -> array`: Evaluate all basis functions at distances r
- `Basis.roughness_matrix() -> array`: Integrated squared second derivative penalty

**model_fitting/features.py**
- `build_design_matrix(cells, basis_xx, basis_xy, basis_eval_mode="ito", topology="poles") -> (G, V)`: Construct stacked design matrix and response vector from list of trimmed cells. Uses `get_partners(cell, topology)` to obtain interaction partner trajectories. When `basis_xx=None`, the chromosome-chromosome block is skipped.
- `basis_eval_mode`: controls where basis functions are evaluated ("ito", "ito_shift", "strato")
- Internally computes all pairwise distances, unit vectors, and basis feature vectors in 3D via vectorized einsum contractions

**model_fitting/fit.py**
- `fit_kernels(G, V, lambda_ridge, lambda_rough, R) -> FitResult`: Penalized least squares
- `fit_model(cells, config: FitConfig) -> FittedModel`: High-level wrapper that builds bases, design matrix, fits, and estimates D
- `cross_validate(cells, config: FitConfig) -> CVResult`: Leave-one-cell-out CV (one-step velocity prediction). Constructs bases from config, respects topology.
- `rollout_cross_validate(cells, config, n_reps, horizons, rng) -> RolloutCVResult`: Leave-one-cell-out rollout validation. Fits on N-1 cells, simulates the held-out cell forward, scores via pathwise MSE (axial/radial), endpoint error, final-frame Wasserstein distance, and per-horizon errors.
- `bootstrap_kernels(cells, config: FitConfig, n_boot, rng) -> BootstrapResult`: Bootstrap over cells
- `estimate_diffusion(V, G, theta, dt, d=3) -> float`: Residual-based D_x

**model_fitting/model.py**
- `FittedModel`: Stores theta, basis configs, D_x, topology, metadata. `basis_xx` can be `None` for topologies without chromosome-chromosome interactions.
- `FittedModel.evaluate_kernel("xx" or "xy", r_values) -> array | None`: Returns `None` for xx when `basis_xx is None`
- `FittedModel.save(path)` / `FittedModel.load(path)`: Persistence. Handles `basis_xx=None` and backward-compatible loading (missing topology defaults to "poles").

**model_fitting/simulate.py**
- `simulate_trajectories(kernel_xx, kernel_xy, partner_positions, x0, n_steps, dt, D_x, rng) -> array`: Euler-Maruyama forward simulation. `partner_positions` shape `(n_partners, T, 3)`. `kernel_xx=None` skips chromosome-chromosome forces. Vectorized pairwise force computation.
- `kernel_callables(model) -> (kernel_xx, kernel_xy)`: Build callable pair from a `FittedModel`. Returns `kernel_xx=None` when the model has no xx basis.
- `simulate_cell(cell, model, rng) -> (trajectory, sim_cell)`: Simulate a `TrimmedCell` forward using a `FittedModel` and the cell's real partner trajectories. Returns the raw trajectory and a `TrimmedCell` wrapping it.
- `generate_synthetic_data(kernel_xx, kernel_xy, ...) -> SyntheticDataset`: Create benchmark with known ground truth
- `add_localization_noise(trajectories, sigma) -> array`: Add Gaussian noise to positions

**model_fitting/validate.py**
- `one_step_prediction_error(model, cells) -> float`
- `residual_diagnostics(model, cells) -> dict`: Normality, whiteness, per-component checks
- `kernel_recovery_error(fitted, true_kernels) -> dict`: L2 error on synthetic benchmarks
- `summary_statistics(trajectories) -> dict`: Radial distribution, spacing, MSD

**model_fitting/diffusion.py**
- `COORDINATE_MAPS`: dict of predefined coordinate maps ("axial", "radial", "distance")
- `local_diffusion_estimates(cells, dt, mode, ...) -> list[array]`: Per-particle, per-timepoint local D estimates using MSD, Vestergaard, weak-noise, or force-corrected estimators
- `estimate_diffusion_variable(cells, basis_D, coord_name, dt, ...) -> DiffusionResult`: Fit D(coordinate) as a basis expansion via ridge regression on local D estimates
- `DiffusionResult`: Stores coefficients, basis, coordinate name, scalar D; has `evaluate(coords)` method

**model_fitting/plotting.py**
- `plot_kernels(model, bootstrap_result) -> fig`: Learned kernels with confidence bands
- `plot_cv_curve(cv_result) -> fig`: Error vs basis size
- `plot_residuals(model, cells) -> fig`: Residual diagnostics
- `plot_recovery(fitted, true_kernels) -> fig`: Synthetic benchmark comparison
- `plot_diffusion(diffusion_result) -> fig`: D(coordinate) curve with scalar average reference

**analysis/lag_correlation.py**
- `compute_lag_correlation(cells, max_lag, smooth_window) -> LagResult`: Velocity dot-product autocorrelation between pole-center and chromosome-center motion
- `plot_lag_correlation(result) -> fig`

**analysis/trajectory_viz.py**
- `plot_cell_trajectories(cell, frame="spindle") -> fig`: Color-coded chromosome tracks
- `plot_chromosome_cloud(cell, timepoint) -> fig`: Snapshot of chromosome positions

---

## Notebook Descriptions

**01_explore_data.ipynb**: Load a few rpe18_ctr cells, inspect array shapes, plot raw 3D trajectories, verify NEB/AO timing, sanity-check pole-pole distances and chromosome counts. Purpose: confirm data loading works correctly.

**02_velocity_spatial_not_temporal.py**: Establishes that chromosome velocity is determined by distance from the spindle center, not time. Replicates the old paper's binned velocity comparison (Fig 2A-B) with Cohen's d effect sizes and a chromosome-level permutation test. Justifies dropping time-dependence before fitting spatial force models.

**02_lag_correlation.ipynb**: Run lag correlation analysis across all rpe18_ctr cells. Produce the supplementary figure showing chromosomes follow centrosomes. This justifies treating centrosomes as external in the model.

**03_synthetic_validation.ipynb**: Generate synthetic trajectories from known kernels (e.g., exponential attraction toward centrosomes, soft repulsion between chromosomes). Fit the model and verify kernel recovery. Test robustness to noise levels, dt, particle count. This is mandatory before real-data fitting.

**04_fit_kernels.ipynb**: Fit f_xx and f_xy on pooled rpe18_ctr data. Plot learned kernels with bootstrap confidence bands. Run residual diagnostics. Primary results notebook.

**05_model_selection.ipynb**: Systematic comparison across basis sizes (4-20), regularization strengths, endpoint fraction, and end_sep. Possible future: with/without spindle-axis angle dependence. Cross-validation curves.

**06_forward_simulation.ipynb**: Simulate from the learned model using real centrosome trajectories as input. Compare simulated chromosome statistics to real data: radial distributions, spacing, MSD, congression dynamics.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Chromosome representation | Centroid of sister kinetochores, single 3D particle | Simplest; sister angle extension planned for later |
| Centrosome treatment | External/given, not modeled | Lag correlation shows chromosomes follow centrosomes; reduces parameters by half |
| Inference dimensionality | Full 3D | Radial kernels are dimension-agnostic; avoids coordinate-transform artifacts |
| Visualization coordinates | 2D spindle frame (axial + radial) | Biologically intuitive for plotting, not used in inference |
| Time window | NEB to configurable endpoint (default: midpoint NEB-AO) | Configurable to compare; midpoint cleanest to explain in paper |
| Per-chromosome start times | Not used; all chromosomes share cell time window | Pairwise features require all positions at each timepoint; most chromosomes move shortly after NEB anyway |
| Pooling | All cells in a condition pooled into one design matrix | Maximizes data for stable kernel estimates; per-cell fits planned as diagnostic |
| Basis functions | Cubic B-splines, 8-15 per kernel | Good smoothness-flexibility tradeoff; roughness penalty natural |
| Methodology | Ronceray/SFI, not Maggioni/Lu | SFI designed for stochastic data, handles noise, includes drift + diffusion inference |
| dt | 5 seconds | Consistent across all data |
| Spatial units | Microns | Native units from tracking |

---

## Future Extensions

### Fit centrosome equations (f_yy, f_yx)
Fitting the full four-kernel system and showing f_yx ≈ 0 would be stronger evidence than the lag correlation alone that centrosomes are autonomous. **Trade-off:** doubles the parameter count, and the centrosome equation has only 2 particles so f_yy is very poorly constrained (only one pairwise distance at each timepoint). Worth doing as a supplementary check, not the primary model.

### Per-cell fits
Pooling assumes all cells in a condition share the same interaction kernels. If kernels vary systematically (e.g., with cell size or chromosome count), pooling masks this. Per-cell fits would reveal heterogeneity but may be noisy — each cell has ~46 chromosomes x ~50-100 timepoints, which might be marginal for stable kernel estimation. **Trade-off:** a middle ground is to fit per-cell, then check if the spread exceeds bootstrap uncertainty from the pooled fit.

### Per-chromosome masking / HMM
Early after NEB, some chromosomes haven't engaged motors yet and are essentially diffusing. Including these timepoints dilutes the learned force signal. Masking by attachment state would sharpen the fit, but requires either the attachment-time heuristic (old code) or a learned hidden state. **Trade-off:** complexity of state inference vs cleaner force estimates. The HMM version is scientifically interesting in its own right — jointly learning forces and attachment states.

### Spindle-axis angle dependence
The pairwise radial model assumes interactions are isotropic. But chromosome-centrosome interactions likely depend on whether the chromosome is axially aligned with or perpendicular to the spindle axis. Adding f(r, cos(theta)) would capture this anisotropy. **Trade-off:** increases basis dimensionality from 1D to 2D per kernel, requiring more data and careful regularization. Probably the most biologically motivated extension — comparing isotropic vs anisotropic fits could be a main paper result.

### Chromosome sister angle
The orientation of the kinetochore pair relative to the spindle axis reflects attachment state (bi-oriented vs mono-oriented vs unattached). Could serve as an input feature rather than just a diagnostic. **Trade-off:** adds complexity and is partially redundant with the angle-dependent kernel extension. Most useful if we want to condition forces on attachment state without an explicit HMM.

### Direction-decomposed learning
Instead of a single scalar kernel f(r) producing forces along the inter-particle direction, learn separate axial and radial force components in the spindle frame. Biologically motivated if chromosomes experience a "polar wind" along the spindle axis that isn't captured by radial pairwise forces. **Trade-off:** requires spindle-frame coordinates for inference (not just visualization) and doubles the kernel count. Breaks the clean pairwise radial assumption.

### Other experimental conditions
Fitting rod311 (dynein inhibition), CENP-E inhibition, hesperidin, etc. and comparing learned kernels to rpe18_ctr is the core scientific payoff — seeing which interactions change under perturbation. **Trade-off:** minimal technical cost once rpe18_ctr works; the catalog/pipeline design supports this directly.

### Multi-lag / SFI noise-aware estimators
~~Plain one-step Euler-Maruyama increments are biased when localization noise is significant relative to true displacement.~~ **Implemented:** Multi-point basis evaluation modes (Ito-shift, Stratonovich) and noise-robust diffusion estimators (Vestergaard, weak-noise) are now available as options. Variable D(x) fitting is also implemented. See `diffusion.py` and the `basis_eval_mode` parameter.

### Near-term scientific priorities

Based on code audit (March 2026):

1. **Add a one-body spindle-frame drift baseline** — `F(z, r_perp)` with no chromosome-chromosome kernel. The learned `f_xx(r)` can absorb omitted spindle geometry or tracking artifacts; a spindle-frame-only baseline would expose this.
2. **Add a simpler parametric external-force baseline** — linear/piecewise-linear pole attraction. If this performs comparably, flexible spline kernels shouldn't be over-interpreted.
3. **Wire variable D(x) into rollout scoring** or keep it explicitly exploratory. Currently D(x) is fitted but not used in CV or rollout validation.
4. **Interpret f_xx cautiously** — short-range repulsion is physically plausible; long-range features may reflect shared geometry or tracking artifacts rather than direct chromosome-chromosome forces.

### Neural baselines (GNN/NRI)
A graph neural network could learn more flexible (non-radial, many-body) interactions. Useful as comparison: if it fits much better, the pairwise radial assumption is missing something; if similar, it validates the simpler model. **Trade-off:** harder to interpret, easier to overfit, heavier dependencies (PyTorch/JAX). Should only be attempted after the interpretable pipeline is solid.

---

## Dependencies

- numpy, scipy (core computation, .mat loading via scipy.io)
- matplotlib (plotting)
- jupyter (notebooks)
- h5py (if any .mat files are v7.3/HDF5 format)

No heavy ML frameworks needed for the initial implementation.
