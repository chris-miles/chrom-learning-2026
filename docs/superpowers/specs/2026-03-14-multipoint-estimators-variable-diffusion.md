# Multi-point Estimators and Variable Diffusion

**Date:** 2026-03-14
**Status:** Approved

## Motivation

The current pipeline uses naive one-step forward-difference velocity estimation
(`v = (x(t+dt) - x(t)) / dt`) and estimates a single scalar diffusion
coefficient D from residuals.  Two improvements from the Ronceray SFI v2
framework are worth adopting:

1. **Multi-point estimators** that reduce bias from localization noise and
   finite sampling interval.
2. **Variable diffusion D(x)** that captures position-dependent diffusivity
   along biologically meaningful coordinates (spindle axis, radial distance,
   etc.).

Both are implemented as options/flags — the current behaviour remains the
default.

---

## Feature 1: Multi-point Force and Diffusion Estimators

### What changes

Following Ronceray, the velocity stencil stays as forward difference `dX/dt`.
The multi-point improvements enter through (a) **where the basis functions are
evaluated** when building the design matrix, and (b) **which local diffusion
estimator** is used.

### Basis evaluation modes (`basis_eval_mode`)

Added as a parameter to `build_design_matrix`:

| Mode | Basis evaluated at | Effect |
|------|-------------------|--------|
| `"ito"` (default) | `X(t)` | Current behaviour. |
| `"ito_shift"` | `X(t) - dX_minus = X(t-1)` | Decorrelates localization noise from displacement. Requires `t >= 1`. |
| `"strato"` | `[X(t) + X(t+1)] / 2` | Stratonovich midpoint. Reduces finite-dt bias. Note: when combined with variable D(x), a spurious-drift correction `div(D)` is technically needed but is negligible for our regime and omitted. |

For `"ito_shift"`, the loop starts at `time_index = 1` instead of 0 (need
access to `X(t-1)`).  For `"strato"`, the basis is evaluated at the mean of
current and next positions, which is already available in the inner loop.

### Diffusion estimator modes (`diffusion_mode`)

Added as a parameter to a new `estimate_diffusion_local` function that returns
per-timepoint, per-particle local D estimates (for use with variable-D fitting)
and to the existing `estimate_diffusion` for scalar D:

| Mode | Formula (scalar D per particle per timepoint) | Points | Notes |
|------|------------------------------------------------|--------|-------|
| `"msd"` (default) | `\|dX\|^2 / (2 d dt)` | 2 | Current behaviour. |
| `"vestergaard"` | `(\|dX\|^2 + \|dX_minus\|^2) / (4 d dt) + (dX . dX_minus) / (2 d dt)` | 3 | Cancels localization noise to leading order (Vestergaard et al. 2014). |
| `"weak_noise"` | `\|dX - dX_minus\|^2 / (4 d dt)` | 3 | Removes drift bias; second difference `X(t+1)-2X(t)+X(t-1)`. |
| `"f_corrected"` | `\|dX - F*dt\|^2 / (2 d dt)` | 2 | Requires prior force estimate (two-pass). |

Here `d=3` is spatial dimension, `.` is dot product, `\|.\|^2` is squared norm.

For scalar D, these local estimates are averaged over all particles and
timepoints.  For variable D(x), they become the response vector in a second
regression (see Feature 2).

The 3-point estimators require access to `dX_minus[t] = X(t) - X(t-1)`, so the
valid timepoint range shrinks by 1 at the start.

### Where these go in the code

- `features.py`: `build_design_matrix` gains a `basis_eval_mode` parameter.
  The inner loop changes which position array is passed to `_pairwise_feature_sum`.
- `diffusion.py` (new): `local_diffusion_estimates(cells, dt, mode, fit_result=None)`
  returns per-(particle, timepoint) scalar D estimates from raw displacements.
  Multi-point estimators (vestergaard, weak_noise) need access to raw `dX` and
  `dX_minus` arrays, not regression residuals.  The `f_corrected` mode requires
  a prior `fit_result` to subtract predicted forces.
- `fit.py`: `estimate_diffusion` keeps its current residual-based interface for
  backward compatibility (`"msd"` mode).  New modes delegate to `diffusion.py`.
  `bootstrap_kernels` and `cross_validate` gain `basis_eval_mode` passthrough.
- `FitConfig`: gains `basis_eval_mode: str = "ito"` and
  `diffusion_mode: str = "msd"` fields.

---

## Feature 2: Variable Diffusion D(x)

### Design

D(x) is expanded in a 1-D basis along a configurable **position coordinate**:

```
D(x) = sum_a  d_a * psi_a(coord(x))
```

where `coord(x)` maps a 3-D chromosome position to a scalar.

### Position coordinate system (`CoordinateMap`)

A lightweight callable that converts `(chromosome_position, TrimmedCell)` →
scalar.  Predefined options:

| Name | Formula | Biological meaning |
|------|---------|-------------------|
| `"axial"` | Signed projection of `(x - pole_center)` onto spindle axis | Distance along spindle; 0 at metaphase plate |
| `"radial"` | Perpendicular distance from spindle axis | Distance from spindle axis |
| `"distance"` | `sqrt(axial^2 + radial^2)` | Distance from spindle center |

These are already computed by `spindle_frame()` in `trajectory.py`, so we reuse
that infrastructure.

The coordinate map is a general interface: `coord(positions, cell) -> scalars`.
This keeps the door open for future extensions (e.g., angular coordinates,
additive multi-coordinate models like `D(r, z) = D_r(r) + D_z(z)`).

### Fitting procedure

Two-stage, following Ronceray (no iteration):

1. **Stage 1 (force inference):** Same as current — build design matrix, solve
   penalized least squares.  Optionally use `basis_eval_mode != "ito"` and
   weight by `A_inv = (2 * D_avg)^{-1}` where `D_avg` is a quick scalar
   estimate.  For scalar D this weighting cancels and doesn't change theta.

2. **Stage 2 (diffusion inference):** Compute local D estimates at each
   (particle, timepoint) using the chosen `diffusion_mode`.  Map each particle
   position to the chosen coordinate.  Solve a second linear regression:
   ```
   D_local[i,t] = sum_a  d_a * psi_a(coord(x_i(t)))  + noise
   ```
   This is ordinary least squares (or ridge-regularised).  Returns coefficient
   vector `d` and the basis object, from which `D(coord)` can be evaluated at
   arbitrary positions.

### Where this goes in the code

- **New file:** `chromlearn/model_fitting/diffusion.py`
  - `CoordinateMap` protocol/callable
  - `coordinate_maps` dict of predefined maps (`"axial"`, `"radial"`, `"distance"`)
  - `estimate_diffusion_variable(cells, fit_result, basis_D, coord_map, ...)` →
    returns `DiffusionResult` with `d_coeffs`, `basis_D`, `coord_map`, and a
    method `evaluate(coord_values) -> D_values`
  - `estimate_diffusion_local(chromosomes, centrioles, dt, mode)` → local D
    array

- **Modified:** `FitConfig` gains:
  - `diffusion_mode: str = "msd"`
  - `D_variable: bool = False`
  - `n_basis_D: int = 6`
  - `r_min_D: float = -8.0`  (signed for axial; user should set per coordinate)
  - `r_max_D: float = 8.0`
  - `D_coordinate: str = "axial"`

- **Modified:** `FittedModel` gains an optional `diffusion_model` field
  (None for scalar D, `DiffusionResult` for variable D).  `save/load` extended
  to serialise diffusion basis parameters and coefficients.

- **Modified:** `plotting.py` gains `plot_diffusion(diffusion_result)` for
  visualising D(coord).

### Data flow

```
TrimmedCells
    │
    ├─► build_design_matrix(basis_eval_mode) ──► G, V
    │                                              │
    │                                              ▼
    │                                         fit_kernels() ──► theta, residuals
    │                                              │
    │   ┌──────────────────────────────────────────┘
    │   │
    │   ▼
    ├─► estimate_diffusion(mode="msd"|"vestergaard"|..., D_variable=False)
    │       → scalar D_x
    │
    └─► estimate_diffusion_variable(cells, fit_result, basis_D, coord_map, mode)
            → DiffusionResult with D(coord) as basis expansion
```

---

## What does NOT change

- Force kernels f_xx and f_xy remain functions of pairwise distance `||x_i - x_j||`.
- The regression structure for force inference stays the same (penalised least squares).
- Default behaviour is identical to current code (`basis_eval_mode="ito"`,
  `diffusion_mode="msd"`, `D_variable=False`).
- The coordinate map infrastructure is general enough that force kernels could
  later be made position-dependent, but that is out of scope here.

---

## New files

```
chromlearn/model_fitting/diffusion.py    — variable D fitting, coordinate maps, local D estimators
tests/test_diffusion.py                  — tests for the above
```

## Modified files

```
chromlearn/model_fitting/__init__.py     — FitConfig gains new fields
chromlearn/model_fitting/features.py     — basis_eval_mode parameter
chromlearn/model_fitting/fit.py          — diffusion_mode parameter on estimate_diffusion
chromlearn/model_fitting/model.py        — optional diffusion_model field, save/load
chromlearn/model_fitting/plotting.py     — plot_diffusion function
```

---

## Testing strategy

- **Stencil tests:** Synthetic data with known localization noise.  Verify that
  `ito_shift` and `strato` produce lower-bias force estimates than `ito` when
  noise is present.
- **Diffusion estimator tests:** Pure diffusion (zero force) with known D.
  Verify all estimators recover D.  Add localization noise and verify
  `vestergaard` is more accurate than `msd`.
- **Variable D test:** Synthetic data where D varies linearly along the
  x-axis.  Fit with variable D and verify recovery.
- **Backward compatibility:** All existing tests pass unchanged (defaults
  preserve current behaviour).
