# Model Selection & Robustness Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add topology-aware model comparison (4 interaction topologies) with cross-validation, forward simulation, and hyperparameter sensitivity analysis across notebooks 3 and 4.

**Architecture:** Refactor the design matrix builder to accept generic interaction partners instead of hardcoded centrioles. Thread a `topology` field through `FitConfig` → `features.py` → `fit.py` → `model.py` → `simulate.py` → `plotting.py`. Build two notebooks: NB3 (model selection) and NB4 (robustness sweeps).

**Tech Stack:** Python 3.10+, numpy, scipy, matplotlib. Existing `chromlearn` package.

**Spec:** `docs/specs/2026-03-15-model-selection-robustness-design.md`

---

## File Structure

### Modified files
- `chromlearn/model_fitting/__init__.py` — Add `topology` field to `FitConfig`
- `chromlearn/io/trajectory.py` — Add `get_partners()` helper
- `chromlearn/model_fitting/features.py` — Accept `topology` + `partners`, support `basis_xx=None`
- `chromlearn/model_fitting/fit.py` — Thread topology through `fit_model`, `cross_validate`, `bootstrap_kernels`; handle `basis_xx=None` in `_block_roughness`
- `chromlearn/model_fitting/model.py` — `FittedModel` supports `topology`, `basis_xx=None`, defensive save/load
- `chromlearn/model_fitting/simulate.py` — `partner_positions` shape `(n_partners, T, 3)`, `kernel_xx=None` support
- `chromlearn/model_fitting/plotting.py` — Adaptive panels for 1 or 2 kernels, topology-aware labels

### New files
- `notebooks/03_model_selection.py` — Notebook 3
- `notebooks/04_robustness.py` — Notebook 4

### Modified test files
- `tests/test_features.py` — Topology + basis_xx=None tests
- `tests/test_fit.py` — Topology threading, basis_xx=None in CV/bootstrap
- `tests/test_simulate.py` — Partner shape, kernel_xx=None
- `tests/test_model.py` — Save/load with topology, basis_xx=None
- `tests/test_trajectory.py` — get_partners tests

---

## Chunk 1: Core Pipeline Refactor

### Task 1: Add `topology` to FitConfig and `get_partners` to trajectory.py

**Files:**
- Modify: `chromlearn/model_fitting/__init__.py:37-56`
- Modify: `chromlearn/io/trajectory.py:1-48`
- Test: `tests/test_trajectory.py`

- [ ] **Step 1: Write failing tests for get_partners**

Add to `tests/test_trajectory.py`:

```python
import numpy as np
from chromlearn.io.trajectory import TrimmedCell, get_partners, pole_center


def _make_cell(T=20, N=4):
    centrioles = np.zeros((T, 3, 2))
    centrioles[:, 0, 0] = -5.0
    centrioles[:, 0, 1] = 5.0
    chromosomes = np.zeros((T, 3, N))
    return TrimmedCell(
        cell_id="test", condition="test",
        centrioles=centrioles, chromosomes=chromosomes,
        tracked=N, dt=5.0, start_frame=0, end_frame=T - 1,
    )


def test_get_partners_poles():
    cell = _make_cell()
    partners = get_partners(cell, "poles")
    assert partners.shape == (2, 20, 3)
    np.testing.assert_allclose(partners[0, :, 0], -5.0)
    np.testing.assert_allclose(partners[1, :, 0], 5.0)


def test_get_partners_center():
    cell = _make_cell()
    partners = get_partners(cell, "center")
    assert partners.shape == (1, 20, 3)
    expected = pole_center(cell)
    np.testing.assert_allclose(partners[0], expected)


def test_get_partners_poles_and_chroms():
    cell = _make_cell()
    partners = get_partners(cell, "poles_and_chroms")
    assert partners.shape == (2, 20, 3)


def test_get_partners_center_and_chroms():
    cell = _make_cell()
    partners = get_partners(cell, "center_and_chroms")
    assert partners.shape == (1, 20, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trajectory.py::test_get_partners_poles -v`
Expected: FAIL — `ImportError: cannot import name 'get_partners'`

- [ ] **Step 3: Implement get_partners and add topology to FitConfig**

In `chromlearn/io/trajectory.py`, add after `pole_center`:

```python
VALID_TOPOLOGIES = ("poles", "center", "poles_and_chroms", "center_and_chroms")


def get_partners(cell: CellData | TrimmedCell, topology: str) -> np.ndarray:
    """Construct interaction partner trajectories for a given topology.

    Returns:
        Array of shape ``(n_partners, T, 3)``.
    """
    if topology not in VALID_TOPOLOGIES:
        raise ValueError(
            f"Unknown topology '{topology}'. Expected one of {VALID_TOPOLOGIES}."
        )
    if topology in ("poles", "poles_and_chroms"):
        return cell.centrioles.transpose(2, 0, 1)  # (2, T, 3)
    # center or center_and_chroms
    return pole_center(cell)[np.newaxis]  # (1, T, 3)
```

In `chromlearn/model_fitting/__init__.py`, add field after `D_coordinate`:

```python
    topology: str = "poles"
```

And add to the docstring: `topology: Interaction topology — "poles" (default), "center", "poles_and_chroms", or "center_and_chroms".`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_trajectory.py -k get_partners -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add chromlearn/io/trajectory.py chromlearn/model_fitting/__init__.py tests/test_trajectory.py
git commit -m "feat: add get_partners helper and topology field to FitConfig"
```

---

### Task 2: Refactor features.py for partner-list abstraction and basis_xx=None

**Files:**
- Modify: `chromlearn/model_fitting/features.py:1-183`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write regression test — topology="poles" matches old behavior**

Add to `tests/test_features.py`:

```python
def test_topology_poles_matches_default():
    """Regression: topology='poles' produces identical output to pre-refactor code."""
    cell = make_simple_trimmed_cell(T=20, N=4)
    basis_xx = HatBasis(0.0, 10.0, n_basis=5)
    basis_xy = HatBasis(0.0, 15.0, n_basis=5)
    G_old, V_old = build_design_matrix([cell], basis_xx, basis_xy)
    G_new, V_new = build_design_matrix(
        [cell], basis_xx, basis_xy, topology="poles",
    )
    np.testing.assert_allclose(G_new, G_old)
    np.testing.assert_allclose(V_new, V_old)
```

- [ ] **Step 2: Write tests for basis_xx=None (poles-only topology)**

```python
def test_basis_xx_none_poles_only():
    """topology='poles' with basis_xx=None skips xx block."""
    cell = make_simple_trimmed_cell(T=20, N=4)
    basis_xy = HatBasis(0.0, 15.0, n_basis=5)
    G, V = build_design_matrix([cell], None, basis_xy, topology="poles")
    # Only xy columns
    assert G.shape[1] == 5
    assert G.shape[0] == 4 * 19 * 3
    assert V.shape[0] == G.shape[0]


def test_basis_xx_none_center_topology():
    """topology='center' uses pole midpoint as single partner."""
    cell = make_simple_trimmed_cell(T=20, N=4)
    basis_xy = HatBasis(0.0, 15.0, n_basis=5)
    G_center, V_center = build_design_matrix(
        [cell], None, basis_xy, topology="center",
    )
    G_poles, V_poles = build_design_matrix(
        [cell], None, basis_xy, topology="poles",
    )
    # Same number of rows, same number of columns
    assert G_center.shape[1] == G_poles.shape[1] == 5
    assert G_center.shape[0] == G_poles.shape[0]
    # But different values (different partner positions)
    assert not np.allclose(G_center, G_poles)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_features.py -k "topology" -v`
Expected: FAIL — `build_design_matrix() got an unexpected keyword argument 'topology'`

- [ ] **Step 4: Refactor features.py**

Modify `build_design_matrix` signature to accept `topology: str = "poles"`. Inside, use `get_partners` to construct the partner array, then pass it to `_build_cell_design_matrix`.

Modify `_build_cell_design_matrix` to accept `partners: np.ndarray` (shape `(n_partners, T, 3)`) and `basis_xx: BSplineBasis | HatBasis | None`.

Key changes in `_build_cell_design_matrix`:
- Replace `centrioles = cell.centrioles` with the passed `partners` array
- Replace `n_poles = centrioles.shape[2]` with `n_partners = partners.shape[0]`
- Replace `centrioles[t].T` with `partners[:, t, :].copy()` → shape `(n_partners, 3)`
- For the evaluation positions, use `partners[:, t, :]`, `partners[:, t-1, :]`, or midpoint accordingly
- When `basis_xx is None`: skip the entire chromosome-chromosome block, set `n_bxx = 0`, produce `g_xx` as empty
- `n_cols = n_bxx + n_bxy` still works (n_bxx=0 when None)

**Critical:** The vectorized einsum contractions remain identical. The only change is which arrays feed into `eval_poles` and whether the xx block runs.

Full implementation of `build_design_matrix`:

```python
def build_design_matrix(
    cells: list[TrimmedCell],
    basis_xx,
    basis_xy,
    basis_eval_mode: str = "ito",
    topology: str = "poles",
) -> tuple[np.ndarray, np.ndarray]:
    """Build the stacked regression design matrix and velocity response.

    Parameters
    ----------
    cells : list[TrimmedCell]
    basis_xx : Basis or None
        Basis for chromosome-chromosome interactions.  ``None`` to skip.
    basis_xy : Basis
        Basis for chromosome-partner interactions.
    basis_eval_mode : str
    topology : str
        Interaction topology — controls which partner positions are used.
    """
    from chromlearn.io.trajectory import get_partners

    if basis_eval_mode not in ("ito", "ito_shift", "strato"):
        raise ValueError(
            f"Unknown basis_eval_mode={basis_eval_mode!r}. "
            "Expected 'ito', 'ito_shift', or 'strato'."
        )

    all_G: list[np.ndarray] = []
    all_V: list[np.ndarray] = []

    for cell in cells:
        partners = get_partners(cell, topology)
        G_cell, V_cell = _build_cell_design_matrix(
            cell, partners, basis_xx, basis_xy, basis_eval_mode,
        )
        if G_cell.size > 0:
            all_G.append(G_cell)
            all_V.append(V_cell)

    n_bxx = basis_xx.n_basis if basis_xx is not None else 0
    n_cols = n_bxx + basis_xy.n_basis
    if not all_G:
        return np.zeros((0, n_cols), dtype=np.float64), np.zeros(0, dtype=np.float64)

    return np.vstack(all_G), np.concatenate(all_V)
```

Full implementation of `_build_cell_design_matrix` — same vectorized logic, but with `partners` parameter and `basis_xx=None` guard:

```python
def _build_cell_design_matrix(
    cell: TrimmedCell,
    partners: np.ndarray,
    basis_xx,
    basis_xy,
    basis_eval_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized design matrix construction for a single cell.

    Parameters
    ----------
    partners : np.ndarray
        Shape ``(n_partners, T, 3)`` — external interaction partner trajectories.
    basis_xx : Basis or None
        ``None`` to skip chromosome-chromosome interactions.
    """
    chromosomes = cell.chromosomes
    T = chromosomes.shape[0]
    N = chromosomes.shape[2]
    n_partners = partners.shape[0]
    n_bxx = basis_xx.n_basis if basis_xx is not None else 0
    n_bxy = basis_xy.n_basis
    dt = cell.dt

    t_start = 1 if basis_eval_mode == "ito_shift" else 0

    rows_G: list[np.ndarray] = []
    rows_V: list[np.ndarray] = []

    for t in range(t_start, T - 1):
        pos_cur = chromosomes[t].T      # (N, 3)
        pos_next = chromosomes[t + 1].T  # (N, 3)

        if basis_eval_mode == "ito":
            eval_chroms = pos_cur
            eval_partners = partners[:, t, :]       # (n_partners, 3)
        elif basis_eval_mode == "ito_shift":
            eval_chroms = chromosomes[t - 1].T
            eval_partners = partners[:, t - 1, :]
        else:  # strato
            eval_chroms = 0.5 * (pos_cur + pos_next)
            eval_partners = 0.5 * (partners[:, t, :] + partners[:, t + 1, :])

        valid = (
            ~np.any(np.isnan(pos_cur), axis=1)
            & ~np.any(np.isnan(pos_next), axis=1)
            & ~np.any(np.isnan(eval_chroms), axis=1)
        )
        n_valid = valid.sum()
        if n_valid == 0:
            continue

        valid_idx = np.flatnonzero(valid)
        eval_valid = eval_chroms[valid_idx]  # (n_valid, 3)

        # --- Chromosome-chromosome interactions ---
        if basis_xx is not None:
            delta_xx = eval_chroms[np.newaxis, :, :] - eval_valid[:, np.newaxis, :]
            dist_xx = np.linalg.norm(delta_xx, axis=2)
            neighbor_valid = ~np.any(np.isnan(eval_chroms), axis=1)
            pair_mask = np.ones((n_valid, N), dtype=bool)
            pair_mask[np.arange(n_valid), valid_idx] = False
            pair_mask[:, ~neighbor_valid] = False
            pair_mask &= dist_xx > 1e-12
            safe_dist = np.where(pair_mask, dist_xx, 1.0)
            direction_xx = delta_xx / safe_dist[:, :, np.newaxis]
            direction_xx[~pair_mask] = 0.0
            flat_dist_xx = dist_xx[pair_mask]
            phi_xx_flat = basis_xx.evaluate(flat_dist_xx)
            phi_xx = np.zeros((n_valid, N, n_bxx), dtype=np.float64)
            phi_xx[pair_mask] = phi_xx_flat
            g_xx = np.einsum("ijd,ijb->idb", direction_xx, phi_xx)
        else:
            g_xx = np.zeros((n_valid, 3, 0), dtype=np.float64)

        # --- Chromosome-partner interactions ---
        delta_xy = eval_partners[np.newaxis, :, :] - eval_valid[:, np.newaxis, :]
        dist_xy = np.linalg.norm(delta_xy, axis=2)
        pole_mask = dist_xy > 1e-12
        safe_dist_xy = np.where(pole_mask, dist_xy, 1.0)
        direction_xy = delta_xy / safe_dist_xy[:, :, np.newaxis]
        direction_xy[~pole_mask] = 0.0
        flat_dist_xy = dist_xy[pole_mask]
        phi_xy_flat = basis_xy.evaluate(flat_dist_xy)
        phi_xy = np.zeros((n_valid, n_partners, n_bxy), dtype=np.float64)
        phi_xy[pole_mask] = phi_xy_flat
        g_xy = np.einsum("ipd,ipb->idb", direction_xy, phi_xy)

        G_block = np.concatenate([g_xx, g_xy], axis=2)
        velocity = (pos_next[valid_idx] - pos_cur[valid_idx]) / dt
        rows_G.append(G_block.reshape(-1, n_bxx + n_bxy))
        rows_V.append(velocity.reshape(-1))

    n_cols = n_bxx + n_bxy
    if not rows_G:
        return np.zeros((0, n_cols), dtype=np.float64), np.zeros(0, dtype=np.float64)

    return np.vstack(rows_G), np.concatenate(rows_V)
```

- [ ] **Step 5: Run all feature tests**

Run: `python -m pytest tests/test_features.py -v`
Expected: All tests PASS (including old tests — backward compatible)

- [ ] **Step 6: Commit**

```bash
git add chromlearn/model_fitting/features.py tests/test_features.py
git commit -m "feat: refactor design matrix for partner-list abstraction and basis_xx=None"
```

---

### Task 3: Update fit.py — thread topology, handle basis_xx=None

**Files:**
- Modify: `chromlearn/model_fitting/fit.py:33-278`
- Test: `tests/test_fit.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_fit.py`:

```python
from chromlearn.model_fitting import FitConfig
from chromlearn.model_fitting.fit import fit_model


def test_fit_model_poles_only_no_xx():
    """fit_model with topology='poles' (no chromosome-chromosome)."""
    cell, _, basis_xy, _ = make_synthetic_inference_cell()
    config = FitConfig(
        topology="poles",
        n_basis_xx=10, n_basis_xy=4,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat",
        lambda_ridge=1e-6, lambda_rough=0.0,
        dt=0.1,
    )
    model = fit_model([cell], config)
    assert model.topology == "poles"
    assert model.n_basis_xx == 0
    assert model.basis_xx is None
    assert model.theta.shape[0] == 4  # only xy coeffs


def test_fit_model_center_topology():
    """fit_model with topology='center'."""
    cell, _, basis_xy, _ = make_synthetic_inference_cell()
    config = FitConfig(
        topology="center",
        n_basis_xy=4,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat",
        lambda_ridge=1e-6, lambda_rough=0.0,
        dt=0.1,
    )
    model = fit_model([cell], config)
    assert model.topology == "center"
    assert model.n_basis_xx == 0


def test_fit_model_poles_and_chroms():
    """fit_model with topology='poles_and_chroms' includes both kernels."""
    cell, _, _, _ = make_synthetic_inference_cell()
    config = FitConfig(
        topology="poles_and_chroms",
        n_basis_xx=4, n_basis_xy=4,
        r_min_xx=0.0, r_max_xx=8.0,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat",
        lambda_ridge=1e-6, lambda_rough=0.0,
        dt=0.1,
    )
    model = fit_model([cell], config)
    assert model.topology == "poles_and_chroms"
    assert model.n_basis_xx == 4
    assert model.basis_xx is not None
    assert model.theta.shape[0] == 8


def test_cross_validate_poles_only():
    """CV works with basis_xx=None."""
    cells = [
        make_synthetic_inference_cell(T=90, N=12, dt=0.15, seed=s)[0]
        for s in range(4)
    ]
    config = FitConfig(
        topology="poles",
        n_basis_xy=4,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat",
        lambda_ridge=1e-6, lambda_rough=0.0,
        dt=0.15,
    )
    cv = cross_validate(cells, config)
    assert np.all(np.isfinite(cv.held_out_errors))
    assert cv.mean_error > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fit.py -k "poles_only_no_xx or center_topology or poles_and_chroms or cross_validate_poles_only" -v`
Expected: FAIL

- [ ] **Step 3: Implement changes to fit.py**

Key changes:

1. `_block_roughness` handles `None`:

```python
def _block_roughness(R_xx: np.ndarray | None, R_xy: np.ndarray) -> np.ndarray:
    if R_xx is None:
        return R_xy
    return block_diag(R_xx, R_xy)
```

2. Helper to determine if topology includes chromosomes:

```python
def _topology_has_chroms(topology: str) -> bool:
    return topology in ("poles_and_chroms", "center_and_chroms")
```

3. `fit_model` uses topology to decide whether to build `basis_xx`:

```python
def fit_model(cells, config=None):
    if config is None:
        config = FitConfig()
    BasisClass = BSplineBasis if config.basis_type == "bspline" else HatBasis
    if _topology_has_chroms(config.topology):
        basis_xx = BasisClass(config.r_min_xx, config.r_max_xx, config.n_basis_xx)
    else:
        basis_xx = None
    basis_xy = BasisClass(config.r_min_xy, config.r_max_xy, config.n_basis_xy)

    G, V = build_design_matrix(
        cells, basis_xx, basis_xy,
        basis_eval_mode=config.basis_eval_mode,
        topology=config.topology,
    )
    R_xx = basis_xx.roughness_matrix() if basis_xx is not None else None
    roughness = _block_roughness(R_xx, basis_xy.roughness_matrix())
    result = fit_kernels(G, V, config.lambda_ridge, config.lambda_rough, roughness)
    D_x = estimate_diffusion(V, G, result.theta, dt=config.dt)

    return FittedModel(
        theta=result.theta,
        n_basis_xx=basis_xx.n_basis if basis_xx is not None else 0,
        n_basis_xy=basis_xy.n_basis,
        basis_xx=basis_xx,
        basis_xy=basis_xy,
        D_x=D_x,
        dt=config.dt,
        topology=config.topology,
        metadata={"n_cells": len(cells)},
    )
```

4. Refactor `cross_validate` and `bootstrap_kernels` to accept `FitConfig` instead of individual basis args. Keep backward-compatible overloads or just switch the signature (breaking change is acceptable since these are internal APIs).

New `cross_validate` signature:

```python
def cross_validate(
    cells: list[TrimmedCell],
    config: FitConfig,
) -> CVResult:
```

Internally constructs bases from config, uses `build_design_matrix(..., topology=config.topology)`.

New `bootstrap_kernels` signature:

```python
def bootstrap_kernels(
    cells: list[TrimmedCell],
    config: FitConfig,
    n_boot: int = 250,
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
```

- [ ] **Step 4: Update old tests that call cross_validate/bootstrap_kernels with old signatures**

The test `test_cross_validate_and_bootstrap_support_shifted_basis_eval_mode` in `tests/test_fit.py` passes `basis_xx`, `basis_xy` directly. Update it to use `FitConfig`:

```python
def test_cross_validate_and_bootstrap_support_shifted_basis_eval_mode() -> None:
    cells = [
        make_synthetic_inference_cell(T=90, N=12, dt=0.15, seed=seed)[0]
        for seed in range(4)
    ]
    _, basis_xx, basis_xy, theta_true = make_synthetic_inference_cell(T=90, N=12, dt=0.15)

    config = FitConfig(
        topology="poles_and_chroms",
        n_basis_xx=4, n_basis_xy=4,
        r_min_xx=0.0, r_max_xx=8.0,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat",
        lambda_ridge=1e-6, lambda_rough=0.0,
        basis_eval_mode="ito_shift",
        dt=0.15,
    )
    cv_result = cross_validate(cells, config)
    G, V = build_design_matrix(
        cells, basis_xx, basis_xy,
        basis_eval_mode="ito_shift",
        topology="poles_and_chroms",
    )
    baseline_error = float(np.mean(V**2))
    assert np.all(np.isfinite(cv_result.held_out_errors))
    assert cv_result.mean_error < baseline_error

    bootstrap_result = bootstrap_kernels(
        cells, config, n_boot=8, rng=np.random.default_rng(0),
    )
    assert bootstrap_result.theta_samples.shape == (8, theta_true.size)
    assert np.all(np.isfinite(bootstrap_result.theta_std))
    bootstrap_rmse = float(np.sqrt(np.mean((bootstrap_result.theta_mean - theta_true) ** 2)))
    assert bootstrap_rmse < 0.08
```

- [ ] **Step 5: Run all fit tests**

Run: `python -m pytest tests/test_fit.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add chromlearn/model_fitting/fit.py tests/test_fit.py
git commit -m "feat: thread topology through fit pipeline, support basis_xx=None"
```

---

### Task 4: Update model.py — topology field, basis_xx=None in save/load

**Files:**
- Modify: `chromlearn/model_fitting/model.py:1-133`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_model.py`:

```python
def test_fitted_model_no_xx_kernel():
    """FittedModel with basis_xx=None."""
    basis_xy = HatBasis(0.0, 10.0, n_basis=5)
    model = FittedModel(
        theta=np.arange(5, dtype=float),
        n_basis_xx=0,
        n_basis_xy=5,
        basis_xx=None,
        basis_xy=basis_xy,
        D_x=0.4,
        dt=5.0,
        topology="poles",
    )
    assert model.evaluate_kernel("xx", np.array([1.0, 2.0])) is None
    result_xy = model.evaluate_kernel("xy", np.array([1.0, 2.0]))
    assert result_xy.shape == (2,)
    assert model.theta_xx.shape == (0,)
    assert model.theta_xy.shape == (5,)


def test_save_load_no_xx(tmp_path):
    """Round-trip save/load with basis_xx=None."""
    basis_xy = HatBasis(0.0, 10.0, n_basis=5)
    model = FittedModel(
        theta=np.arange(5, dtype=float),
        n_basis_xx=0,
        n_basis_xy=5,
        basis_xx=None,
        basis_xy=basis_xy,
        D_x=0.4,
        dt=5.0,
        topology="poles",
    )
    path = tmp_path / "model_no_xx.npz"
    model.save(path)
    loaded = FittedModel.load(path)
    np.testing.assert_allclose(loaded.theta, model.theta)
    assert loaded.basis_xx is None
    assert loaded.n_basis_xx == 0
    assert loaded.topology == "poles"


def test_load_backward_compat_no_topology(tmp_path):
    """Loading old model files without topology field defaults to 'poles'."""
    basis_xx = HatBasis(0.0, 8.0, n_basis=4)
    basis_xy = HatBasis(0.0, 10.0, n_basis=5)
    # Save without topology field (simulating old format)
    np.savez(
        tmp_path / "old_model.npz",
        theta=np.arange(9, dtype=float),
        n_basis_xx=4, n_basis_xy=5,
        D_x=0.4, dt=5.0,
        metadata=np.array(None, dtype=object),
        basis_xx_type="hat", basis_xx_r_min=0.0, basis_xx_r_max=8.0, basis_xx_n_basis=4,
        basis_xy_type="hat", basis_xy_r_min=0.0, basis_xy_r_max=10.0, basis_xy_n_basis=5,
        diffusion_has_model=False,
    )
    loaded = FittedModel.load(tmp_path / "old_model.npz")
    assert loaded.topology == "poles"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_model.py -k "no_xx or backward_compat" -v`
Expected: FAIL

- [ ] **Step 3: Implement model.py changes**

Add `topology: str = "poles"` field to `FittedModel`.

Update `theta_xx` property:
```python
@property
def theta_xx(self) -> np.ndarray:
    return self.theta[: self.n_basis_xx]
```
(This already works — returns empty array when n_basis_xx=0.)

Update `evaluate_kernel`:
```python
def evaluate_kernel(self, kernel: str, r: np.ndarray) -> np.ndarray | None:
    values = np.asarray(r, dtype=np.float64)
    if kernel == "xx":
        if self.basis_xx is None:
            return None
        return self.basis_xx.evaluate(values) @ self.theta_xx
    if kernel == "xy":
        return self.basis_xy.evaluate(values) @ self.theta_xy
    raise ValueError(f"Unknown kernel '{kernel}'. Use 'xx' or 'xy'.")
```

Update `save` — guard `basis_xx`:
```python
def save(self, path):
    output_path = Path(path)
    # ... diffusion_payload same as before ...
    basis_xx_payload = {}
    if self.basis_xx is not None:
        basis_xx_payload = {
            "basis_xx_type": "bspline" if isinstance(self.basis_xx, BSplineBasis) else "hat",
            "basis_xx_r_min": self.basis_xx.r_min,
            "basis_xx_r_max": self.basis_xx.r_max,
            "basis_xx_n_basis": self.basis_xx.n_basis,
        }
    np.savez(
        output_path,
        theta=self.theta,
        n_basis_xx=self.n_basis_xx,
        n_basis_xy=self.n_basis_xy,
        D_x=self.D_x,
        dt=self.dt,
        topology=self.topology,
        metadata=np.array(self.metadata, dtype=object),
        basis_xy_type="bspline" if isinstance(self.basis_xy, BSplineBasis) else "hat",
        basis_xy_r_min=self.basis_xy.r_min,
        basis_xy_r_max=self.basis_xy.r_max,
        basis_xy_n_basis=self.basis_xy.n_basis,
        **basis_xx_payload,
        **diffusion_payload,
    )
```

Update `load` — handle missing topology and missing basis_xx:
```python
@classmethod
def load(cls, path):
    data = np.load(Path(path), allow_pickle=True)
    # ...
    topology = str(data["topology"]) if "topology" in data else "poles"
    n_basis_xx = int(data["n_basis_xx"])
    basis_xx = None
    if n_basis_xx > 0 and "basis_xx_type" in data:
        basis_xx = make_basis(
            str(data["basis_xx_type"]),
            float(data["basis_xx_r_min"]),
            float(data["basis_xx_r_max"]),
            n_basis_xx,
        )
    # ... rest same ...
    return cls(
        theta=..., n_basis_xx=n_basis_xx, n_basis_xy=...,
        basis_xx=basis_xx, basis_xy=...,
        D_x=..., dt=..., topology=topology,
        metadata=metadata, diffusion_model=diffusion_model,
    )
```

- [ ] **Step 4: Run all model tests**

Run: `python -m pytest tests/test_model.py -v`
Expected: All PASS (including existing round-trip test)

- [ ] **Step 5: Commit**

```bash
git add chromlearn/model_fitting/model.py tests/test_model.py
git commit -m "feat: FittedModel supports topology field and basis_xx=None save/load"
```

---

### Task 5: Update simulate.py — partner_positions shape, kernel_xx=None

**Files:**
- Modify: `chromlearn/model_fitting/simulate.py:1-149`
- Test: `tests/test_simulate.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_simulate.py`:

```python
def test_simulate_with_partner_positions():
    """simulate_trajectories accepts (n_partners, T, 3) partner shape."""
    rng = np.random.default_rng(42)
    n_steps = 50
    n_chrom = 10
    # (2, T, 3) shape
    partners = np.zeros((2, n_steps + 1, 3))
    partners[0, :, 0] = -5.0
    partners[1, :, 0] = 5.0
    x0 = rng.normal(0.0, 2.0, size=(n_chrom, 3))
    trajectory = simulate_trajectories(
        kernel_xx=lambda r: np.zeros_like(r),
        kernel_xy=lambda r: np.zeros_like(r),
        partner_positions=partners,
        x0=x0, n_steps=n_steps, dt=5.0, D_x=0.1, rng=rng,
    )
    assert trajectory.shape == (n_steps + 1, 3, n_chrom)


def test_simulate_single_partner():
    """simulate_trajectories works with 1 partner (center topology)."""
    rng = np.random.default_rng(42)
    n_steps = 50
    partners = np.zeros((1, n_steps + 1, 3))
    x0 = rng.normal(0.0, 2.0, size=(5, 3))
    trajectory = simulate_trajectories(
        kernel_xx=None,
        kernel_xy=lambda r: -0.01 * r,
        partner_positions=partners,
        x0=x0, n_steps=n_steps, dt=5.0, D_x=0.1, rng=rng,
    )
    assert trajectory.shape == (n_steps + 1, 3, 5)


def test_simulate_kernel_xx_none():
    """kernel_xx=None skips chromosome-chromosome forces."""
    rng = np.random.default_rng(42)
    n_steps = 50
    partners = np.zeros((2, n_steps + 1, 3))
    partners[0, :, 0] = -5.0
    partners[1, :, 0] = 5.0
    x0 = rng.normal(0.0, 2.0, size=(5, 3))
    trajectory = simulate_trajectories(
        kernel_xx=None,
        kernel_xy=lambda r: np.zeros_like(r),
        partner_positions=partners,
        x0=x0, n_steps=n_steps, dt=5.0, D_x=0.1, rng=rng,
    )
    assert trajectory.shape == (n_steps + 1, 3, 5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_simulate.py -k "partner_positions or single_partner or kernel_xx_none" -v`
Expected: FAIL

- [ ] **Step 3: Implement simulate.py changes**

Update `simulate_trajectories`:
- Rename `centrosome_positions` to `partner_positions`
- Change expected shape from `(T, 3, n_poles)` to `(n_partners, T, 3)`
- Handle `kernel_xx=None` by skipping chromosome-chromosome loop
- Update partner access: `partner_positions[p, step, :]` instead of `poles[:, pole_index]`

```python
def simulate_trajectories(
    kernel_xx: Callable[[np.ndarray], np.ndarray] | None,
    kernel_xy: Callable[[np.ndarray], np.ndarray],
    partner_positions: np.ndarray,
    x0: np.ndarray,
    n_steps: int,
    dt: float,
    D_x: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Euler-Maruyama forward simulation of overdamped Langevin dynamics.

    Args:
        kernel_xx: f_xx(r) or None to skip chromosome-chromosome forces.
        kernel_xy: f_xy(r) for chromosome-partner forces.
        partner_positions: Shape ``(n_partners, n_steps+1, 3)``.
        x0: Initial chromosome positions, shape ``(N, 3)``.
        ...
    """
    if rng is None:
        rng = np.random.default_rng()

    n_chromosomes = x0.shape[0]
    n_partners = partner_positions.shape[0]
    trajectory = np.zeros((n_steps + 1, 3, n_chromosomes), dtype=np.float64)
    trajectory[0] = x0.T
    noise_scale = np.sqrt(2.0 * D_x * dt)

    for step in range(n_steps):
        positions = trajectory[step]
        forces = np.zeros_like(positions)

        for chrom_index in range(n_chromosomes):
            current = positions[:, chrom_index]

            if kernel_xx is not None:
                for neighbor_index in range(n_chromosomes):
                    if neighbor_index == chrom_index:
                        continue
                    delta = positions[:, neighbor_index] - current
                    distance = float(np.linalg.norm(delta))
                    if distance <= 1e-12:
                        continue
                    direction = delta / distance
                    forces[:, chrom_index] += kernel_xx(np.array([distance]))[0] * direction

            for partner_index in range(n_partners):
                delta = partner_positions[partner_index, step] - current
                distance = float(np.linalg.norm(delta))
                if distance <= 1e-12:
                    continue
                direction = delta / distance
                forces[:, chrom_index] += kernel_xy(np.array([distance]))[0] * direction

        noise = noise_scale * rng.standard_normal(size=(3, n_chromosomes))
        trajectory[step + 1] = positions + forces * dt + noise

    return trajectory
```

Update `generate_synthetic_data` similarly — construct `partners` in `(n_partners, T, 3)` shape:

```python
def generate_synthetic_data(
    kernel_xx: Callable[[np.ndarray], np.ndarray] | None,
    kernel_xy: Callable[[np.ndarray], np.ndarray],
    n_chromosomes: int = 20,
    n_steps: int = 100,
    dt: float = 5.0,
    D_x: float = 0.1,
    pole_separation: float = 10.0,
    rng: np.random.Generator | None = None,
) -> SyntheticDataset:
    if rng is None:
        rng = np.random.default_rng()

    partners = np.zeros((2, n_steps + 1, 3), dtype=np.float64)
    partners[0, :, 0] = -0.5 * pole_separation
    partners[1, :, 0] = 0.5 * pole_separation
    x0 = rng.normal(0.0, 2.0, size=(n_chromosomes, 3))
    chromosomes = simulate_trajectories(
        kernel_xx=kernel_xx, kernel_xy=kernel_xy,
        partner_positions=partners,
        x0=x0, n_steps=n_steps, dt=dt, D_x=D_x, rng=rng,
    )
    return SyntheticDataset(
        chromosomes=chromosomes,
        centrosomes=partners.transpose(1, 2, 0),  # back to (T, 3, 2) for compat
        kernel_xx=kernel_xx, kernel_xy=kernel_xy,
        D_x=D_x, dt=dt,
    )
```

- [ ] **Step 4: Update old tests that use centrosome_positions kwarg**

In `tests/test_simulate.py`, update existing tests to use `partner_positions` with shape `(n_partners, T, 3)`:

```python
def test_simulate_output_shape() -> None:
    rng = np.random.default_rng(42)
    n_chrom = 10
    n_steps = 50
    partners = np.zeros((2, n_steps + 1, 3))
    partners[0, :, 0] = -5.0
    partners[1, :, 0] = 5.0
    x0 = rng.normal(0.0, 2.0, size=(n_chrom, 3))
    trajectory = simulate_trajectories(
        kernel_xx=lambda r: np.zeros_like(r),
        kernel_xy=lambda r: np.zeros_like(r),
        partner_positions=partners,
        x0=x0, n_steps=n_steps, dt=5.0, D_x=0.1, rng=rng,
    )
    assert trajectory.shape == (n_steps + 1, 3, n_chrom)
```

Also update `test_pure_diffusion_msd` similarly. Update `tests/test_fit.py:make_synthetic_inference_cell` to construct partner-shaped centrosomes and use `partner_positions`:

```python
centrioles = np.zeros((T + 1, 3, 2))
centrioles[:, 0, 0] = -3.0
centrioles[:, 0, 1] = 3.0
partners = centrioles.transpose(2, 0, 1)  # (2, T+1, 3)
# ...
chromosomes = simulate_trajectories(
    kernel_xx=kernel_xx, kernel_xy=kernel_xy,
    partner_positions=partners,
    x0=x0, n_steps=T, dt=dt, D_x=D_x, rng=rng,
)
```

- [ ] **Step 5: Run all simulate and fit tests**

Run: `python -m pytest tests/test_simulate.py tests/test_fit.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add chromlearn/model_fitting/simulate.py tests/test_simulate.py tests/test_fit.py
git commit -m "feat: simulate.py uses partner_positions shape and supports kernel_xx=None"
```

---

### Task 6: Update plotting.py — adaptive panels, topology-aware labels

**Files:**
- Modify: `chromlearn/model_fitting/plotting.py:1-108`

- [ ] **Step 1: Update plot_kernels to handle basis_xx=None**

```python
def plot_kernels(
    model: FittedModel,
    bootstrap: BootstrapResult | None = None,
    n_points: int = 200,
    ci_levels: list[float] | None = None,
) -> plt.Figure:
    if ci_levels is None:
        ci_levels = [0.05]

    has_xx = model.basis_xx is not None

    if has_xx:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    else:
        figure, ax_xy = plt.subplots(1, 1, figsize=(6, 4.5))
        axes = [None, ax_xy]

    # Determine partner label based on topology
    topology = getattr(model, "topology", "poles")
    partner_label = (
        "Chromosome \u2190 pole center"
        if topology in ("center", "center_and_chroms")
        else "Chromosome \u2190 poles"
    )

    specs = []
    if has_xx:
        specs.append(("xx", model.basis_xx, model.theta_xx,
                       "Chromosome \u2190 chromosome", axes[0]))
    specs.append(("xy", model.basis_xy, model.theta_xy,
                   partner_label, axes[-1]))

    for name, basis, theta, title, axis in specs:
        radius = np.linspace(basis.r_min, basis.r_max, n_points)
        phi = basis.evaluate(radius)
        axis.plot(radius, phi @ theta, color="C0", linewidth=2)
        if bootstrap is not None:
            if name == "xx":
                samples = bootstrap.theta_samples[:, : model.n_basis_xx]
            else:
                samples = bootstrap.theta_samples[:, model.n_basis_xx :]
            curves = phi @ samples.T
            for level in ci_levels:
                lo = np.quantile(curves, level, axis=1)
                hi = np.quantile(curves, 1.0 - level, axis=1)
                axis.fill_between(radius, lo, hi, color="C0", alpha=0.2)
        axis.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
        axis.set_xlabel("Distance (\u00b5m)")
        axis.set_ylabel("Force")
        axis.set_title(title)

    figure.tight_layout()
    return figure
```

- [ ] **Step 2: Verify visually (no automated test needed for plotting)**

Run: `python -c "from chromlearn.model_fitting.plotting import plot_kernels; print('import OK')"`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add chromlearn/model_fitting/plotting.py
git commit -m "feat: plot_kernels adapts panels and labels to topology"
```

---

### Task 7: Run full test suite — verify nothing is broken

- [ ] **Step 1: Run entire test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Fix any failures, re-run**

- [ ] **Step 3: Commit any fixes**

---

## Chunk 2: Notebooks

### Task 8: Write Notebook 3 — Model Selection

**Files:**
- Create: `notebooks/03_model_selection.py`

- [ ] **Step 1: Write notebook 3**

Follow the pattern from notebooks 01 and 02 (percent-cell format with `# %%` markers and `# %% [markdown]` for prose).

```python
# %% [markdown]
# # 03 — Model selection: which interactions matter?
#
# Compare four interaction topologies for chromosome dynamics:
#
# | Model | Partners | Kernels |
# |-------|----------|---------|
# | M1 (poles) | Each centrosome | f_xy(r) |
# | M2 (center) | Centrosome midpoint | f_xc(r) |
# | M3 (poles+chroms) | Centrosomes + chromosomes | f_xy(r) + f_xx(r) |
# | M4 (center+chroms) | Midpoint + chromosomes | f_xc(r) + f_xx(r) |
#
# Evaluation: leave-one-cell-out cross-validation, learned kernel plots
# with bootstrap confidence bands, physical plausibility checks, and
# forward simulation comparison.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chromlearn.io.catalog import load_condition
from chromlearn.io.trajectory import trim_trajectory, spindle_frame, get_partners
from chromlearn.model_fitting import FitConfig
from chromlearn.model_fitting.fit import fit_model, cross_validate, bootstrap_kernels
from chromlearn.model_fitting.plotting import plot_kernels, plot_cv_curve
from chromlearn.model_fitting.simulate import simulate_trajectories

plt.rcParams["figure.dpi"] = 110

# %%
cells_raw = load_condition("rpe18_ctr")
cells = [trim_trajectory(c, method="midpoint_neb_ao") for c in cells_raw]
print(f"Loaded {len(cells)} rpe18_ctr cells")

# %% [markdown]
# ## Step 1: Define topologies and fit all four models
#
# We use the same basis size and regularisation for all models to ensure
# a fair comparison.  The basis domain for the partner kernel is set per
# topology based on the observed distance distribution.

# %%
# Estimate basis domains from data
def estimate_partner_domain(cells, topology, percentiles=(1, 99)):
    """Compute domain for partner basis from observed distances."""
    all_dists = []
    for cell in cells:
        partners = get_partners(cell, topology)
        chroms = cell.chromosomes  # (T, 3, N)
        for t in range(chroms.shape[0]):
            pos = chroms[t].T  # (N, 3)
            valid = ~np.any(np.isnan(pos), axis=1)
            if not valid.any():
                continue
            for p in range(partners.shape[0]):
                diffs = pos[valid] - partners[p, t]
                dists = np.linalg.norm(diffs, axis=1)
                all_dists.append(dists)
    all_dists = np.concatenate(all_dists)
    lo, hi = np.percentile(all_dists, percentiles)
    return max(0.3, lo), hi


def estimate_xx_domain(cells, percentiles=(1, 99)):
    """Compute domain for chromosome-chromosome basis."""
    all_dists = []
    for cell in cells:
        chroms = cell.chromosomes
        for t in range(chroms.shape[0]):
            pos = chroms[t].T
            valid = ~np.any(np.isnan(pos), axis=1)
            pos_valid = pos[valid]
            if pos_valid.shape[0] < 2:
                continue
            diffs = pos_valid[:, None, :] - pos_valid[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            mask = np.triu(np.ones(dists.shape, dtype=bool), k=1)
            all_dists.append(dists[mask])
    all_dists = np.concatenate(all_dists)
    lo, hi = np.percentile(all_dists, percentiles)
    return max(0.3, lo), hi


r_min_xx, r_max_xx = estimate_xx_domain(cells)
print(f"Chromosome-chromosome domain: [{r_min_xx:.2f}, {r_max_xx:.2f}] um")

# %%
TOPOLOGIES = {
    "M1: poles": "poles",
    "M2: center": "center",
    "M3: poles+chroms": "poles_and_chroms",
    "M4: center+chroms": "center_and_chroms",
}

N_BASIS = 10
LAMBDA_RIDGE = 1e-3
LAMBDA_ROUGH = 1e-3

models = {}
configs = {}

for label, topo in TOPOLOGIES.items():
    r_min_xy, r_max_xy = estimate_partner_domain(cells, topo)
    print(f"{label}: partner domain [{r_min_xy:.2f}, {r_max_xy:.2f}] um")

    cfg = FitConfig(
        topology=topo,
        n_basis_xx=N_BASIS,
        n_basis_xy=N_BASIS,
        r_min_xx=r_min_xx,
        r_max_xx=r_max_xx,
        r_min_xy=r_min_xy,
        r_max_xy=r_max_xy,
        lambda_ridge=LAMBDA_RIDGE,
        lambda_rough=LAMBDA_ROUGH,
    )
    configs[label] = cfg
    models[label] = fit_model(cells, cfg)
    print(f"  {label}: {models[label].theta.shape[0]} coefficients, D = {models[label].D_x:.4f}")

# %% [markdown]
# ## Step 2: Cross-validation comparison

# %%
cv_results = {}
for label, cfg in configs.items():
    print(f"CV for {label}...")
    cv_results[label] = cross_validate(cells, cfg)
    print(f"  MSE = {cv_results[label].mean_error:.6f} +/- {cv_results[label].std_error:.6f}")

# %%
fig = plot_cv_curve(cv_results)
fig.suptitle("Leave-one-cell-out cross-validation", y=1.02)
plt.show()

# %% [markdown]
# ## Step 3: Learned kernel plots with bootstrap confidence bands

# %%
for label, cfg in configs.items():
    print(f"Bootstrap for {label}...")
    boot = bootstrap_kernels(cells, cfg, n_boot=200, rng=np.random.default_rng(42))
    fig = plot_kernels(models[label], bootstrap=boot)
    fig.suptitle(label, y=1.02)
    plt.show()

# %% [markdown]
# ## Step 4: Physical plausibility
#
# For models with chromosome-chromosome interactions (M3, M4), inspect
# the learned f_xx kernel:
# - **Short-range repulsion** (excluded volume) should appear as f_xx > 0
#   at small r.  Short-range *attraction* is likely an artifact of optical
#   diffraction or tracking merges.
# - **Long-range forces** should decay to zero.  Significant forces at
#   large r (> 5 um) are nonphysical and suggest overfitting or model
#   misspecification.

# %%
for label in ["M3: poles+chroms", "M4: center+chroms"]:
    model = models[label]
    r = np.linspace(r_min_xx, r_max_xx, 200)
    f_xx = model.evaluate_kernel("xx", r)
    if f_xx is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(r, f_xx, linewidth=2)
        ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Distance (um)")
        ax.set_ylabel("f_xx(r)")
        ax.set_title(f"{label}: chromosome-chromosome kernel")
        # Annotate plausibility
        if f_xx[0] < 0:
            ax.annotate("Warning: short-range attraction",
                        xy=(r[0], f_xx[0]), fontsize=9, color="red")
        plt.show()

# %% [markdown]
# ## Step 5: Forward simulation comparison
#
# Simulate trajectories under each model using learned kernels and
# estimated diffusion coefficient.  Compare to real trajectories in
# spindle frame.

# %%
# Pick a representative cell for simulation comparison
ref_cell = cells[0]
ref_sf = spindle_frame(ref_cell)
n_steps = ref_cell.chromosomes.shape[0] - 1

fig, axes = plt.subplots(2, len(TOPOLOGIES), figsize=(5 * len(TOPOLOGIES), 8),
                          sharex=True, sharey="row")

for col, (label, topo) in enumerate(TOPOLOGIES.items()):
    model = models[label]
    partners = get_partners(ref_cell, topo)

    def make_kernel(kernel_name, m=model):
        def k(r):
            result = m.evaluate_kernel(kernel_name, r)
            return result if result is not None else np.zeros_like(r)
        return k

    kernel_xy = make_kernel("xy")
    kernel_xx_fn = None
    if model.basis_xx is not None:
        kernel_xx_fn = make_kernel("xx")

    x0 = ref_cell.chromosomes[0].T  # (N, 3)
    sim = simulate_trajectories(
        kernel_xx=kernel_xx_fn,
        kernel_xy=kernel_xy,
        partner_positions=partners,
        x0=x0,
        n_steps=n_steps,
        dt=ref_cell.dt,
        D_x=model.D_x,
        rng=np.random.default_rng(42),
    )

    # Convert simulated to spindle frame (approximate: use same poles)
    from chromlearn.io.trajectory import TrimmedCell as TC
    sim_cell = TC(
        cell_id="sim", condition="sim",
        centrioles=ref_cell.centrioles,
        chromosomes=sim,
        tracked=sim.shape[2], dt=ref_cell.dt,
        start_frame=0, end_frame=n_steps,
    )
    sim_sf = spindle_frame(sim_cell)

    # Real
    for chrom_idx in range(ref_sf.axial.shape[1]):
        axes[0, col].plot(ref_sf.axial[:, chrom_idx],
                          ref_sf.radial[:, chrom_idx],
                          "k-", alpha=0.15, linewidth=0.5)
    axes[0, col].set_title(f"Real")
    axes[0, col].set_ylabel("Radial (um)")

    # Simulated
    for chrom_idx in range(sim_sf.axial.shape[1]):
        axes[1, col].plot(sim_sf.axial[:, chrom_idx],
                          sim_sf.radial[:, chrom_idx],
                          "C0-", alpha=0.15, linewidth=0.5)
    axes[1, col].set_title(f"Sim: {label}")
    axes[1, col].set_xlabel("Axial (um)")
    axes[1, col].set_ylabel("Radial (um)")

fig.suptitle("Real vs simulated trajectories (spindle frame)", y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Step 6: Verdict
#
# Compare all models on:
# 1. Cross-validation error
# 2. Physical plausibility of learned kernels
# 3. Forward simulation fidelity

# %%
print("=" * 60)
print("Model comparison summary")
print("=" * 60)
for label in TOPOLOGIES:
    cv = cv_results[label]
    m = models[label]
    n_params = m.theta.shape[0]
    print(f"{label:25s}  CV MSE = {cv.mean_error:.6f} +/- {cv.std_error:.6f}  "
          f"(n_params = {n_params}, D = {m.D_x:.4f})")

best_label = min(cv_results, key=lambda k: cv_results[k].mean_error)
print(f"\nBest model by CV: {best_label}")

# %% [markdown]
# ## Bonus: Variable D(x) for the winning model
#
# Refit the winning topology with position-dependent diffusion D(x)
# and compare to scalar D via cross-validation.

# %%
from chromlearn.model_fitting.diffusion import estimate_diffusion_variable

best_config = configs[best_label]
best_model = models[best_label]

# Fit variable D along axial coordinate
from chromlearn.model_fitting.basis import BSplineBasis
from chromlearn.model_fitting.features import build_design_matrix
from chromlearn.model_fitting.fit import fit_kernels, _block_roughness

# Re-fit to get residuals for diffusion estimation
if best_model.basis_xx is not None:
    basis_xx = BSplineBasis(best_config.r_min_xx, best_config.r_max_xx, best_config.n_basis_xx)
else:
    basis_xx = None
basis_xy = BSplineBasis(best_config.r_min_xy, best_config.r_max_xy, best_config.n_basis_xy)

G, V = build_design_matrix(cells, basis_xx, basis_xy,
                            basis_eval_mode=best_config.basis_eval_mode,
                            topology=best_config.topology)
R_xx = basis_xx.roughness_matrix() if basis_xx is not None else None
R = _block_roughness(R_xx, basis_xy.roughness_matrix())
fit_result = fit_kernels(G, V, best_config.lambda_ridge, best_config.lambda_rough, R)

basis_D = BSplineBasis(-8.0, 8.0, n_basis=6)
diffusion_result = estimate_diffusion_variable(
    cells, basis_D, coord_name="axial", dt=best_config.dt,
    mode="msd", fit_result=fit_result, basis_xx=basis_xx, basis_xy=basis_xy,
)

print(f"Scalar D = {best_model.D_x:.4f}")
print(f"Variable D range: [{diffusion_result.evaluate(np.array([-6.0]))[0]:.4f}, "
      f"{diffusion_result.evaluate(np.array([6.0]))[0]:.4f}]")

from chromlearn.model_fitting.plotting import plot_diffusion
fig = plot_diffusion(diffusion_result)
fig.suptitle(f"Variable D(axial) for {best_label}", y=1.02)
plt.show()
```

- [ ] **Step 2: Test that the notebook runs without errors**

Run: `cd "/c/Google Drive/chrom_learning_2026" && python notebooks/03_model_selection.py`
Expected: Completes without errors. May take a few minutes for CV/bootstrap.

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_model_selection.py
git commit -m "feat: notebook 03 — model selection across 4 interaction topologies"
```

---

### Task 9: Write Notebook 4 — Robustness

**Files:**
- Create: `notebooks/04_robustness.py`

- [ ] **Step 1: Write notebook 4**

```python
# %% [markdown]
# # 04 — Robustness and hyperparameter sensitivity
#
# Tests how sensitive the winning model from Notebook 03 is to:
# 1. Number of basis functions
# 2. Regularisation strength (lambda_ridge, lambda_rough)
# 3. Estimator mode (Ito, Ito-shift, Stratonovich)
# 4. Endpoint method (midpoint_neb_ao, ao_mean, end_sep)
# 5. Diffusion estimation mode

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chromlearn.io.catalog import load_condition
from chromlearn.io.trajectory import trim_trajectory
from chromlearn.model_fitting import FitConfig
from chromlearn.model_fitting.fit import fit_model, cross_validate
from chromlearn.model_fitting.plotting import plot_kernels, plot_cv_curve

plt.rcParams["figure.dpi"] = 110

# %%
# --- CONFIGURE: set the winning topology from NB3 ---
WINNING_TOPOLOGY = "poles"  # Update after running NB3
# ---------------------------------------------------

cells_raw = load_condition("rpe18_ctr")
cells = [trim_trajectory(c, method="midpoint_neb_ao") for c in cells_raw]
print(f"Loaded {len(cells)} cells")

# %% [markdown]
# ## Sweep 1: Number of basis functions
#
# Vary n_basis while keeping regularisation fixed.  Plot training MSE
# and CV MSE vs n_basis to identify overfitting onset.

# %%
BASIS_SIZES = [4, 6, 8, 10, 12, 16, 20]
cv_by_nbasis = {}
train_by_nbasis = {}

for n in BASIS_SIZES:
    print(f"n_basis = {n}...")
    cfg = FitConfig(
        topology=WINNING_TOPOLOGY,
        n_basis_xx=n, n_basis_xy=n,
        lambda_ridge=1e-3, lambda_rough=1e-3,
    )
    cv_by_nbasis[f"n={n}"] = cross_validate(cells, cfg)
    model = fit_model(cells, cfg)
    train_by_nbasis[n] = float(np.mean(model.theta**2))  # placeholder; use residuals
    print(f"  CV MSE = {cv_by_nbasis[f'n={n}'].mean_error:.6f}")

# %%
fig = plot_cv_curve(cv_by_nbasis)
fig.axes[0].set_xlabel("Number of basis functions")
fig.suptitle("CV error vs basis size", y=1.02)
plt.show()

# %% [markdown]
# ## Sweep 2: Regularisation strength
#
# Log-spaced grid of lambda_ridge and lambda_rough.

# %%
LAMBDAS = np.logspace(-5, 0, 8)
cv_ridge = {}
cv_rough = {}

# Sweep lambda_ridge (fix lambda_rough)
for lam in LAMBDAS:
    label = f"{lam:.1e}"
    cfg = FitConfig(topology=WINNING_TOPOLOGY, lambda_ridge=lam, lambda_rough=1e-3)
    cv_ridge[label] = cross_validate(cells, cfg)

# Sweep lambda_rough (fix lambda_ridge)
for lam in LAMBDAS:
    label = f"{lam:.1e}"
    cfg = FitConfig(topology=WINNING_TOPOLOGY, lambda_ridge=1e-3, lambda_rough=lam)
    cv_rough[label] = cross_validate(cells, cfg)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

labels_r = list(cv_ridge)
axes[0].errorbar(range(len(labels_r)),
                 [cv_ridge[l].mean_error for l in labels_r],
                 yerr=[cv_ridge[l].std_error for l in labels_r],
                 fmt="o-", capsize=4)
axes[0].set_xticks(range(len(labels_r)))
axes[0].set_xticklabels(labels_r, rotation=45, ha="right")
axes[0].set_ylabel("CV MSE")
axes[0].set_title("lambda_ridge sweep")

labels_s = list(cv_rough)
axes[1].errorbar(range(len(labels_s)),
                 [cv_rough[l].mean_error for l in labels_s],
                 yerr=[cv_rough[l].std_error for l in labels_s],
                 fmt="o-", capsize=4)
axes[1].set_xticks(range(len(labels_s)))
axes[1].set_xticklabels(labels_s, rotation=45, ha="right")
axes[1].set_ylabel("CV MSE")
axes[1].set_title("lambda_rough sweep")

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Sweep 3: Estimator mode comparison
#
# Fit with Ito, Ito-shift, and Stratonovich.  Compare CV errors and
# kernel shapes.  Disagreement flags finite-dt bias.

# %%
MODES = ["ito", "ito_shift", "strato"]
cv_modes = {}
models_modes = {}

for mode in MODES:
    print(f"Mode: {mode}...")
    cfg = FitConfig(topology=WINNING_TOPOLOGY, basis_eval_mode=mode)
    cv_modes[mode] = cross_validate(cells, cfg)
    models_modes[mode] = fit_model(cells, cfg)
    print(f"  CV MSE = {cv_modes[mode].mean_error:.6f}")

# %%
fig = plot_cv_curve(cv_modes)
fig.suptitle("Estimator mode comparison", y=1.02)
plt.show()

# %%
# Compare kernel shapes across modes
for mode in MODES:
    fig = plot_kernels(models_modes[mode])
    fig.suptitle(f"Kernels ({mode})", y=1.02)
    plt.show()

# %% [markdown]
# ## Sweep 4: Endpoint method comparison
#
# Test sensitivity to trajectory time window.

# %%
ENDPOINT_METHODS = ["midpoint_neb_ao", "ao_mean", "end_sep"]
cv_endpoints = {}

for method in ENDPOINT_METHODS:
    print(f"Endpoint: {method}...")
    cells_m = [trim_trajectory(c, method=method) for c in cells_raw]
    cfg = FitConfig(topology=WINNING_TOPOLOGY, endpoint_method=method)
    cv_endpoints[method] = cross_validate(cells_m, cfg)
    print(f"  CV MSE = {cv_endpoints[method].mean_error:.6f}")

# %%
fig = plot_cv_curve(cv_endpoints)
fig.suptitle("Endpoint method comparison", y=1.02)
plt.show()

# %% [markdown]
# ## Sweep 5: Diffusion estimation mode

# %%
from chromlearn.model_fitting.diffusion import local_diffusion_estimates
from chromlearn.model_fitting.basis import BSplineBasis
from chromlearn.model_fitting.features import build_design_matrix
from chromlearn.model_fitting.fit import fit_kernels as _fit_kernels, _block_roughness

best_cfg = FitConfig(topology=WINNING_TOPOLOGY)
best_model = fit_model(cells, best_cfg)

if best_model.basis_xx is not None:
    bxx = BSplineBasis(best_cfg.r_min_xx, best_cfg.r_max_xx, best_cfg.n_basis_xx)
else:
    bxx = None
bxy = BSplineBasis(best_cfg.r_min_xy, best_cfg.r_max_xy, best_cfg.n_basis_xy)

G, V = build_design_matrix(cells, bxx, bxy, topology=best_cfg.topology)
R_xx = bxx.roughness_matrix() if bxx is not None else None
R = _block_roughness(R_xx, bxy.roughness_matrix())
fit_res = _fit_kernels(G, V, best_cfg.lambda_ridge, best_cfg.lambda_rough, R)

D_MODES = ["msd", "vestergaard", "weak_noise"]
print("Diffusion estimates by mode:")
for dmode in D_MODES:
    try:
        D_locals = local_diffusion_estimates(
            cells, dt=best_cfg.dt, mode=dmode,
            fit_result=fit_res if dmode == "f_corrected" else None,
            basis_xx=bxx, basis_xy=bxy,
        )
        D_mean = float(np.nanmean(D_locals))
        print(f"  {dmode:15s}: D = {D_mean:.4f}")
    except Exception as e:
        print(f"  {dmode:15s}: skipped ({e})")

# %% [markdown]
# ## Summary

# %%
print("=" * 60)
print("Robustness summary")
print("=" * 60)
print(f"\nTopology: {WINNING_TOPOLOGY}")
print(f"\nBasis size: best n_basis by CV = "
      f"{min(cv_by_nbasis, key=lambda k: cv_by_nbasis[k].mean_error)}")
print(f"\nBest lambda_ridge = "
      f"{min(cv_ridge, key=lambda k: cv_ridge[k].mean_error)}")
print(f"\nBest lambda_rough = "
      f"{min(cv_rough, key=lambda k: cv_rough[k].mean_error)}")
print(f"\nBest estimator mode = "
      f"{min(cv_modes, key=lambda k: cv_modes[k].mean_error)}")
print(f"\nBest endpoint = "
      f"{min(cv_endpoints, key=lambda k: cv_endpoints[k].mean_error)}")
```

- [ ] **Step 2: Test that the notebook runs without errors**

Run: `cd "/c/Google Drive/chrom_learning_2026" && python notebooks/04_robustness.py`
Expected: Completes without errors. Will take longer due to many CV runs.

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_robustness.py
git commit -m "feat: notebook 04 — robustness and hyperparameter sensitivity"
```

---

### Task 10: Final integration test

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Verify notebooks import without errors**

Run: `python -c "exec(open('notebooks/03_model_selection.py').read())" 2>&1 | tail -5`

- [ ] **Step 3: Final commit if any cleanup needed**
