# chromlearn Implementation Plan

> **For agentic workers:** Use this plan to implement the chromlearn codebase step by step. Steps use checkbox (`- [ ]`) syntax for tracking. Read `docs/design.md` for the full design spec and `CLAUDE.md` for project conventions before starting.

**Goal:** Build a Python pipeline that learns pairwise interaction kernels (chromosome-chromosome, centrosome-on-chromosome) from 3D mitosis trajectory data using penalized basis-expansion regression (SFI/Ronceray approach).

**Architecture:** A `chromlearn/` Python package with three subpackages: `io/` (data loading), `model_fitting/` (basis functions, design matrix, regression, simulation, validation), and `analysis/` (lag correlation, trajectory visualization). Jupyter notebooks in `notebooks/` are the primary user interface.

**Tech Stack:** Python 3.10+, numpy, scipy, matplotlib, jupyter, h5py

**Spec document:** `docs/design.md`

---

## File Map

Files to create (in implementation order):

```
chromlearn/__init__.py
chromlearn/io/__init__.py
chromlearn/io/loader.py
chromlearn/io/trajectory.py
chromlearn/io/catalog.py
chromlearn/model_fitting/__init__.py
chromlearn/model_fitting/basis.py
chromlearn/model_fitting/features.py
chromlearn/model_fitting/fit.py
chromlearn/model_fitting/model.py
chromlearn/model_fitting/simulate.py
chromlearn/model_fitting/validate.py
chromlearn/model_fitting/plotting.py
chromlearn/analysis/__init__.py
chromlearn/analysis/lag_correlation.py
chromlearn/analysis/trajectory_viz.py
tests/test_loader.py
tests/test_trajectory.py
tests/test_catalog.py
tests/test_basis.py
tests/test_features.py
tests/test_fit.py
tests/test_simulate.py
tests/test_validate.py
tests/test_lag_correlation.py
notebooks/01_explore_data.ipynb
notebooks/02_lag_correlation.ipynb
notebooks/03_synthetic_validation.ipynb
notebooks/04_fit_kernels.ipynb
notebooks/05_model_selection.ipynb
notebooks/06_forward_simulation.ipynb
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `chromlearn/__init__.py`, `chromlearn/io/__init__.py`, `chromlearn/model_fitting/__init__.py`, `chromlearn/analysis/__init__.py`, `tests/__init__.py`, `requirements.txt`

- [ ] **Step 1: Create package structure**

Create all `__init__.py` files. The top-level `chromlearn/__init__.py` should just define `__version__ = "0.1.0"`. All subpackage `__init__.py` files should be empty.

- [ ] **Step 2: Create requirements.txt**

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
jupyter
h5py
pytest
```

- [ ] **Step 3: Install dependencies**

Run: `pip install -r requirements.txt`

- [ ] **Step 4: Create tests/__init__.py**

Empty file.

- [ ] **Step 5: Verify pytest runs**

Run: `pytest tests/ -v`
Expected: "no tests ran" (0 collected), exit code 5 (no tests found), no import errors.

---

## Task 2: Data Loader (`chromlearn/io/loader.py`)

**Files:**
- Create: `chromlearn/io/loader.py`, `tests/test_loader.py`

**Context:** Each `.mat` file (in the `data/` directory) contains one cell's tracking data. The MATLAB variables inside are:
- `centrioles`: array of shape `(T, 3, 2)` — 3D positions of 2 centrosome poles at each timepoint. First index is time, second is xyz coordinates, third is pole index (0 or 1).
- `kinetochores`: array of shape `(T, 6, N)` — each chromosome has 6 columns: columns 0:3 are sister kinetochore 1 (x,y,z), columns 3:6 are sister kinetochore 2 (x,y,z). Third index is chromosome index.
- `neb`: scalar integer — frame index when nuclear envelope breaks down
- `ao1`, `ao2`: scalar integers — two estimates of anaphase onset frame
- `tracked`: scalar integer — number of tracked chromosomes

The `.mat` files may be MATLAB v5 format (use `scipy.io.loadmat`) or v7.3/HDF5 format (use `h5py`). Try scipy first, fall back to h5py.

Some files have `neb = NaN`. Treat those as anaphase-only and exclude them from
the pipeline. `load_cell` should raise on those files, and `catalog.py` should
filter them out.

- [ ] **Step 1: Write test for load_cell**

```python
# tests/test_loader.py
import numpy as np
import pytest
from pathlib import Path
from chromlearn.io.loader import load_cell, CellData

DATA_DIR = Path(__file__).parent.parent / "data"

def get_sample_cell_path():
    """Return path to first available rpe18_ctr cell."""
    cells = sorted(DATA_DIR.glob("rpe18_ctr_*.mat"))
    if not cells:
        pytest.skip("No data files found")
    return cells[0]

def test_load_cell_returns_celldata():
    path = get_sample_cell_path()
    cell = load_cell(path)
    assert isinstance(cell, CellData)

def test_load_cell_fields():
    path = get_sample_cell_path()
    cell = load_cell(path)
    # Check cell_id is parsed from filename
    assert cell.cell_id == path.stem  # e.g. "rpe18_ctr_500"
    # Check condition is parsed from cell_id
    assert cell.condition == "rpe18_ctr"
    # Check array shapes
    T = cell.centrioles.shape[0]
    N = cell.tracked
    assert cell.centrioles.shape == (T, 3, 2)
    assert cell.chromosomes.shape == (T, 3, N)
    # Check metadata types
    assert isinstance(cell.neb, int)
    assert isinstance(cell.ao1, int)
    assert isinstance(cell.ao2, int)
    assert isinstance(cell.tracked, int)
    assert cell.dt == 5.0

def test_load_cell_chromosomes_are_centroids():
    """Chromosomes should be the mean of the two sister kinetochores."""
    path = get_sample_cell_path()
    cell = load_cell(path)
    # Verify no chromosome position is exactly a sister position
    # (centroids should be between the two sisters)
    assert cell.chromosomes.shape[1] == 3  # 3D

def test_load_cell_nan_handling():
    """NaN values in kinetochores should propagate to chromosome centroids as NaN."""
    path = get_sample_cell_path()
    cell = load_cell(path)
    # Check that chromosomes array has float dtype (can hold NaN)
    assert cell.chromosomes.dtype == np.float64 or cell.chromosomes.dtype == np.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chromlearn.io.loader'`

- [ ] **Step 3: Implement loader.py**

```python
# chromlearn/io/loader.py
from dataclasses import dataclass
import numpy as np
from pathlib import Path


@dataclass
class CellData:
    """Container for one cell's trajectory data."""
    cell_id: str           # e.g. "rpe18_ctr_500"
    condition: str         # e.g. "rpe18_ctr"
    centrioles: np.ndarray # (T, 3, 2) — two centrosome poles
    chromosomes: np.ndarray # (T, 3, N) — chromosome centroids
    neb: int               # frame index of nuclear envelope breakdown
    ao1: int               # anaphase onset estimate 1
    ao2: int               # anaphase onset estimate 2
    tracked: int           # number of tracked chromosomes
    dt: float = 5.0        # seconds between frames


def _parse_condition(cell_id: str) -> str:
    """Extract condition from cell_id.

    e.g. 'rpe18_ctr_500' -> 'rpe18_ctr'
         'rod311_ctr_503' -> 'rod311_ctr'

    Convention: last underscore-separated token is the cell number,
    everything before it is the condition.
    """
    parts = cell_id.rsplit("_", 1)
    return parts[0]


def _load_mat_scipy(path: Path) -> dict:
    """Load .mat file using scipy (MATLAB v5 format)."""
    import scipy.io
    return scipy.io.loadmat(str(path), squeeze_me=True)


def _load_mat_h5py(path: Path) -> dict:
    """Load .mat file using h5py (MATLAB v7.3 / HDF5 format)."""
    import h5py
    data = {}
    with h5py.File(str(path), "r") as f:
        for key in f.keys():
            val = f[key][()]
            if isinstance(val, np.ndarray):
                data[key] = val
            else:
                data[key] = val
    return data


def _load_mat(path: Path) -> dict:
    """Load .mat file, trying scipy first then h5py."""
    try:
        return _load_mat_scipy(path)
    except NotImplementedError:
        return _load_mat_h5py(path)


def _extract_scalar_int(val) -> int:
    """Extract an integer from a MATLAB scalar (may be array or scalar)."""
    if isinstance(val, np.ndarray):
        return int(val.flat[0])
    return int(val)


def load_cell(path: Path | str, dt: float = 5.0) -> CellData:
    """Load a single cell's trajectory data from a .mat file.

    Args:
        path: Path to the .mat file
        dt: Time interval between frames in seconds (default 5.0)

    Returns:
        CellData with centrosome and chromosome positions
    """
    path = Path(path)
    raw = _load_mat(path)

    cell_id = path.stem
    condition = _parse_condition(cell_id)

    # Extract centrioles: (T, 3, 2)
    centrioles = np.array(raw["centrioles"], dtype=np.float64)
    # Ensure shape is (T, 3, 2) — scipy.io may squeeze differently
    if centrioles.ndim == 2:
        # Shouldn't happen with squeeze_me=True and 3D data, but handle
        raise ValueError(f"Unexpected centrioles shape: {centrioles.shape}")

    # Extract kinetochores: (T, 6, N)
    kk = np.array(raw["kinetochores"], dtype=np.float64)

    # Compute chromosome centroids: mean of sisters
    # kk[:, 0:3, j] = sister 1 xyz, kk[:, 3:6, j] = sister 2 xyz
    sister1 = kk[:, 0:3, :]  # (T, 3, N)
    sister2 = kk[:, 3:6, :]  # (T, 3, N)
    # Use nanmean so that if one sister is NaN, we still get the other
    # If both are NaN, result is NaN (which is correct — missing data)
    chromosomes = np.nanmean(np.stack([sister1, sister2], axis=0), axis=0)  # (T, 3, N)

    neb = _extract_scalar_int(raw["neb"])
    ao1 = _extract_scalar_int(raw["ao1"])
    ao2 = _extract_scalar_int(raw["ao2"])
    tracked = _extract_scalar_int(raw["tracked"])

    return CellData(
        cell_id=cell_id,
        condition=condition,
        centrioles=centrioles,
        chromosomes=chromosomes,
        neb=neb,
        ao1=ao1,
        ao2=ao2,
        tracked=tracked,
        dt=dt,
    )
```

**Important MATLAB indexing note:** MATLAB uses 1-based indexing. The `neb`, `ao1`, `ao2` values from the `.mat` files are 1-based frame indices. When using them as Python array indices, subtract 1. This conversion should happen in `trajectory.py` (at the trim stage), NOT in the loader — the loader stores raw values as they come from the file. Document this in the loader docstring.

Add this note to the `CellData` docstring:
```python
    # Note: neb, ao1, ao2 are 1-based (MATLAB convention).
    # Convert to 0-based when indexing into arrays: arr[cell.neb - 1]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_loader.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add chromlearn/ tests/ requirements.txt
git commit -m "feat: add data loader for .mat trajectory files"
```

---

## Task 3: Trajectory Processing (`chromlearn/io/trajectory.py`)

**Files:**
- Create: `chromlearn/io/trajectory.py`, `tests/test_trajectory.py`

**Context:** This module trims trajectories to a time window (NEB to endpoint) and computes derived quantities like pole-pole distance, pole center, and spindle-frame coordinates. The `neb`, `ao1`, `ao2` values from the `.mat` files are **1-based** (MATLAB convention). Convert to 0-based Python indices here.

The `end_sep` computation (inspired by the MATLAB code in `old_code/aggregate_trajs_dec2022.m`):
1. Compute pole-pole distance over time
2. Smooth with Savitzky-Golay filter (window=51)
3. Compute reference maximum over the core metaphase region `[NEB, midpoint(NEB, AO)]`
4. Compute a running average (window=5 frames) of the smoothed distance
5. `end_sep` = first frame where the running average reaches 95% of the reference maximum

Note: the original MATLAB code used a velocity-threshold rule; the Python
implementation uses a 95%-plateau rule instead, which detects when the spindle
reaches its target separation rather than when it stops elongating.

- [ ] **Step 1: Write tests**

```python
# tests/test_trajectory.py
import numpy as np
import pytest
from chromlearn.io.loader import CellData
from chromlearn.io.trajectory import (
    trim_trajectory,
    compute_end_sep,
    pole_pole_distance,
    pole_center,
    spindle_frame,
    TrimmedCell,
)


def make_fake_cell(T=100, N=10, neb=10, ao1=80, ao2=82):
    """Create a synthetic CellData for testing."""
    rng = np.random.default_rng(42)
    # Two poles separating linearly along x-axis
    centrioles = np.zeros((T, 3, 2))
    for t in range(T):
        sep = 2.0 + 0.1 * t  # increasing separation
        centrioles[t, 0, 0] = -sep / 2  # pole 1 at (-sep/2, 0, 0)
        centrioles[t, 0, 1] = sep / 2   # pole 2 at (sep/2, 0, 0)
    # Chromosomes scattered around origin
    chromosomes = rng.normal(0, 2, (T, 3, N))
    return CellData(
        cell_id="test_cell_001",
        condition="test",
        centrioles=centrioles,
        chromosomes=chromosomes,
        neb=neb,  # 1-based
        ao1=ao1,
        ao2=ao2,
        tracked=N,
    )


def test_pole_pole_distance():
    cell = make_fake_cell()
    ppd = pole_pole_distance(cell)
    assert ppd.shape == (100,)
    # At t=0, separation = 2.0
    np.testing.assert_allclose(ppd[0], 2.0, atol=1e-10)
    # Distance should be increasing
    assert np.all(np.diff(ppd) > 0)


def test_pole_center():
    cell = make_fake_cell()
    pc = pole_center(cell)
    assert pc.shape == (100, 3)
    # Poles are symmetric around origin, so center should be at origin
    np.testing.assert_allclose(pc, 0.0, atol=1e-10)


def test_trim_trajectory_midpoint():
    cell = make_fake_cell(T=100, neb=10, ao1=80, ao2=82)
    trimmed = trim_trajectory(cell, method="midpoint_neb_ao")
    assert isinstance(trimmed, TrimmedCell)
    # neb=10 (1-based) -> index 9 (0-based)
    # ao_mean = (80+82)/2 = 81 (1-based) -> index 80 (0-based)
    # midpoint = (9 + 80) // 2 = 44
    # So trimmed window is frames 9..44 inclusive
    expected_len = 44 - 9 + 1
    assert trimmed.chromosomes.shape[0] == expected_len
    assert trimmed.centrioles.shape[0] == expected_len


def test_trim_trajectory_ao_mean():
    cell = make_fake_cell(T=100, neb=10, ao1=80, ao2=82)
    trimmed = trim_trajectory(cell, method="ao_mean")
    # ao_mean = 81 (1-based) -> 80 (0-based)
    # neb = 10 (1-based) -> 9 (0-based)
    expected_len = 80 - 9 + 1
    assert trimmed.chromosomes.shape[0] == expected_len


def test_trim_preserves_particle_count():
    cell = make_fake_cell(N=15)
    trimmed = trim_trajectory(cell, method="ao_mean")
    assert trimmed.chromosomes.shape[2] == 15
    assert trimmed.centrioles.shape[2] == 2


def test_spindle_frame_axes():
    cell = make_fake_cell()
    trimmed = trim_trajectory(cell, method="ao_mean")
    sf = spindle_frame(trimmed)
    # sf.axial should be along the pole-pole axis (x-axis in our fake data)
    # sf.radial should be perpendicular distance
    assert sf.axial.shape == trimmed.chromosomes.shape  # (T', 1, N) or (T', N)
    assert sf.radial.shape[0] == trimmed.chromosomes.shape[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trajectory.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement trajectory.py**

```python
# chromlearn/io/trajectory.py
from dataclasses import dataclass
import numpy as np
from scipy.signal import savgol_filter
from chromlearn.io.loader import CellData


@dataclass
class TrimmedCell:
    """Cell data trimmed to a time window."""
    cell_id: str
    condition: str
    centrioles: np.ndarray   # (T', 3, 2)
    chromosomes: np.ndarray  # (T', 3, N)
    tracked: int
    dt: float
    start_frame: int  # 0-based index in original data
    end_frame: int    # 0-based index in original data (inclusive)


@dataclass
class SpindleFrameData:
    """Chromosome positions in spindle-aligned cylindrical coordinates."""
    axial: np.ndarray    # (T', N) — signed distance along pole-pole axis
    radial: np.ndarray   # (T', N) — perpendicular distance from pole-pole axis


def pole_pole_distance(cell: CellData | TrimmedCell) -> np.ndarray:
    """Euclidean distance between the two centrosome poles at each timepoint.

    Returns:
        1D array of shape (T,)
    """
    return np.linalg.norm(
        cell.centrioles[:, :, 0] - cell.centrioles[:, :, 1], axis=1
    )


def pole_center(cell: CellData | TrimmedCell) -> np.ndarray:
    """Midpoint of the two centrosome poles at each timepoint.

    Returns:
        Array of shape (T, 3)
    """
    return 0.5 * (cell.centrioles[:, :, 0] + cell.centrioles[:, :, 1])


def compute_end_sep(cell: CellData, smooth_window: int = 51) -> int:
    """Detect when spindle poles stop separating.

    Ported from old_code/aggregate_trajs_dec2022.m lines 156-170.
    Returns 0-based frame index.

    Algorithm:
    1. Compute pole-pole distance
    2. Smooth with Savitzky-Golay filter
    3. Normalize to [0, 1]
    4. Compute velocity (diff)
    5. Smooth velocity
    6. Normalize velocity by max |velocity|
    7. First frame > 50 where |normalized velocity| < 0.1
    """
    ppd = pole_pole_distance(cell)
    T = len(ppd)
    # Smooth
    if T < smooth_window:
        smooth_window = T if T % 2 == 1 else T - 1
    smoothed = savgol_filter(ppd, window_length=smooth_window, polyorder=3)
    # Normalize to [0, 1]
    smin, smax = smoothed.min(), smoothed.max()
    if smax - smin < 1e-10:
        return T - 1  # no separation detected
    normalized = (smoothed - smin) / (smax - smin)
    # Velocity
    vel = np.diff(normalized)
    if len(vel) < smooth_window:
        sw = len(vel) if len(vel) % 2 == 1 else len(vel) - 1
    else:
        sw = smooth_window
    smooth_vel = savgol_filter(vel, window_length=sw, polyorder=3)
    # Normalize velocity
    max_abs_vel = np.max(np.abs(smooth_vel))
    if max_abs_vel < 1e-10:
        return T - 1
    norm_vel = smooth_vel / max_abs_vel
    # Find first frame > 50 where normalized velocity < 0.1
    for i in range(50, len(norm_vel)):
        if norm_vel[i] < 0.1:
            return i
    return T - 1


def _compute_endpoint(cell: CellData, method: str) -> int:
    """Compute the 0-based end frame index for trimming.

    Args:
        cell: CellData with 1-based neb, ao1, ao2
        method: one of "midpoint_neb_ao", "ao_mean", "end_sep"

    Returns:
        0-based frame index (inclusive endpoint)
    """
    neb_0 = cell.neb - 1  # convert to 0-based
    if method == "ao_mean":
        ao_mean_0 = int(round((cell.ao1 + cell.ao2) / 2)) - 1
        return ao_mean_0
    elif method == "midpoint_neb_ao":
        ao_mean_0 = int(round((cell.ao1 + cell.ao2) / 2)) - 1
        return (neb_0 + ao_mean_0) // 2
    elif method == "end_sep":
        return compute_end_sep(cell)
    else:
        raise ValueError(f"Unknown endpoint method: {method}")


def trim_trajectory(
    cell: CellData, method: str = "midpoint_neb_ao"
) -> TrimmedCell:
    """Trim cell trajectories to [NEB, endpoint].

    Args:
        cell: Raw cell data (neb, ao1, ao2 are 1-based MATLAB indices)
        method: Endpoint method — "midpoint_neb_ao" (default), "ao_mean", "end_sep"

    Returns:
        TrimmedCell with arrays sliced to the time window
    """
    start = cell.neb - 1  # 0-based
    end = _compute_endpoint(cell, method)
    # Clamp to valid range
    T = cell.centrioles.shape[0]
    start = max(0, start)
    end = min(T - 1, end)

    sl = slice(start, end + 1)  # inclusive end
    return TrimmedCell(
        cell_id=cell.cell_id,
        condition=cell.condition,
        centrioles=cell.centrioles[sl],
        chromosomes=cell.chromosomes[sl],
        tracked=cell.tracked,
        dt=cell.dt,
        start_frame=start,
        end_frame=end,
    )


def spindle_frame(trimmed: TrimmedCell) -> SpindleFrameData:
    """Project chromosome positions into spindle-aligned cylindrical coordinates.

    At each timepoint:
    - The spindle axis is the unit vector from pole 1 to pole 2
    - 'axial' = signed projection of (chromosome - pole_center) onto spindle axis
    - 'radial' = perpendicular distance from spindle axis

    Returns:
        SpindleFrameData with axial (T', N) and radial (T', N) arrays
    """
    T = trimmed.centrioles.shape[0]
    N = trimmed.tracked
    pc = pole_center(trimmed)  # (T, 3)

    # Spindle axis at each timepoint
    axis = trimmed.centrioles[:, :, 1] - trimmed.centrioles[:, :, 0]  # (T, 3)
    axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)  # (T, 1)
    axis_unit = axis / axis_norm  # (T, 3)

    axial = np.zeros((T, N))
    radial = np.zeros((T, N))

    for j in range(N):
        # Vector from pole center to chromosome j
        delta = trimmed.chromosomes[:, :, j] - pc  # (T, 3)
        # Axial projection
        ax = np.sum(delta * axis_unit, axis=1)  # (T,)
        # Radial = perpendicular component magnitude
        perp = delta - ax[:, np.newaxis] * axis_unit  # (T, 3)
        rad = np.linalg.norm(perp, axis=1)  # (T,)
        axial[:, j] = ax
        radial[:, j] = rad

    return SpindleFrameData(axial=axial, radial=radial)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_trajectory.py -v`
Expected: All PASS. You may need to adjust the `test_spindle_frame_axes` test assertions to match the actual output shapes `(T', N)`.

- [ ] **Step 5: Commit**

```bash
git add chromlearn/io/trajectory.py tests/test_trajectory.py
git commit -m "feat: add trajectory trimming and spindle-frame projection"
```

---

## Task 4: Cell Catalog (`chromlearn/io/catalog.py`)

**Files:**
- Create: `chromlearn/io/catalog.py`, `tests/test_catalog.py`

**Context:** This module discovers and batch-loads cells from the `data/` directory by condition name. Cell `.mat` files follow the naming pattern `{condition}_{id}.mat`, e.g. `rpe18_ctr_500.mat`. The catalog globs for `{condition}_*.mat` and excludes files with `neb = NaN`.

- [ ] **Step 1: Write tests**

```python
# tests/test_catalog.py
import pytest
from pathlib import Path
from chromlearn.io.catalog import list_cells, load_condition
from chromlearn.io.loader import CellData

DATA_DIR = Path(__file__).parent.parent / "data"


def test_list_cells_rpe18_ctr():
    cells = list_cells("rpe18_ctr", data_dir=DATA_DIR)
    assert len(cells) == 7  # NEB-annotated subset
    assert all(isinstance(c, str) for c in cells)
    assert all("rpe18_ctr" in c for c in cells)


def test_list_cells_returns_sorted():
    cells = list_cells("rpe18_ctr", data_dir=DATA_DIR)
    assert cells == sorted(cells)


def test_list_cells_unknown_condition():
    cells = list_cells("nonexistent_condition", data_dir=DATA_DIR)
    assert cells == []


def test_load_condition():
    cells = load_condition("rpe18_ctr", data_dir=DATA_DIR)
    assert len(cells) == 7
    assert all(isinstance(c, CellData) for c in cells)
    assert all(c.condition == "rpe18_ctr" for c in cells)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_catalog.py -v`

- [ ] **Step 3: Implement catalog.py**

```python
# chromlearn/io/catalog.py
from pathlib import Path
from chromlearn.io.loader import load_cell, CellData

# Default data directory (relative to repo root)
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def list_cells(condition: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> list[str]:
    """List available cell IDs for a given condition.

    Discovers cells by globbing for {condition}_*.mat in data_dir.

    Args:
        condition: Condition prefix, e.g. "rpe18_ctr"
        data_dir: Directory containing .mat files

    Returns:
        Sorted list of cell IDs (filename stems)
    """
    data_dir = Path(data_dir)
    pattern = f"{condition}_*.mat"
    paths = sorted(data_dir.glob(pattern))
    return [p.stem for p in paths]


def load_condition(
    condition: str,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    dt: float = 5.0,
) -> list[CellData]:
    """Load all cells for a given condition.

    Args:
        condition: Condition prefix, e.g. "rpe18_ctr"
        data_dir: Directory containing .mat files
        dt: Time interval between frames in seconds

    Returns:
        List of CellData objects, sorted by cell_id
    """
    data_dir = Path(data_dir)
    pattern = f"{condition}_*.mat"
    paths = sorted(data_dir.glob(pattern))
    return [load_cell(p, dt=dt) for p in paths]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_catalog.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chromlearn/io/catalog.py tests/test_catalog.py
git commit -m "feat: add cell catalog for batch loading by condition"
```

---

## Task 5: Basis Functions (`chromlearn/model_fitting/basis.py`)

**Files:**
- Create: `chromlearn/model_fitting/basis.py`, `tests/test_basis.py`

**Context:** We need basis functions to represent the interaction kernels f_xx(r) and f_xy(r) on bounded intervals [r_min, r_max]. The primary basis is cubic B-splines. We also provide piecewise linear "hat" functions for debugging. Each basis must:
1. Evaluate all basis functions at arbitrary distance values: `evaluate(r) -> (len(r), n_basis)` matrix
2. Compute a roughness penalty matrix R: `roughness_matrix() -> (n_basis, n_basis)` — this is the Gram matrix of second derivatives, used for smoothness regularization

For cubic B-splines, use `scipy.interpolate.BSpline`. Place `n_basis + 4` knots (with repeated boundary knots for clamped splines) uniformly on [r_min, r_max]. Each basis function is a single B-spline basis element.

The roughness matrix R_{ij} = integral of phi_i''(r) * phi_j''(r) dr over [r_min, r_max]. Compute this numerically using Gaussian quadrature (`scipy.integrate.fixed_quad` or simply dense evaluation and trapezoidal rule).

- [ ] **Step 1: Write tests**

```python
# tests/test_basis.py
import numpy as np
import pytest
from chromlearn.model_fitting.basis import BSplineBasis, HatBasis


class TestBSplineBasis:
    def test_shape(self):
        basis = BSplineBasis(r_min=0.0, r_max=10.0, n_basis=8)
        r = np.linspace(0, 10, 50)
        phi = basis.evaluate(r)
        assert phi.shape == (50, 8)

    def test_partition_of_unity(self):
        """B-spline basis should sum to ~1 in the interior."""
        basis = BSplineBasis(r_min=0.0, r_max=10.0, n_basis=10)
        r = np.linspace(0.5, 9.5, 100)  # interior points
        phi = basis.evaluate(r)
        row_sums = phi.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.05)

    def test_nonnegative(self):
        basis = BSplineBasis(r_min=0.0, r_max=10.0, n_basis=8)
        r = np.linspace(0, 10, 100)
        phi = basis.evaluate(r)
        assert np.all(phi >= -1e-15)

    def test_outside_support_is_zero(self):
        basis = BSplineBasis(r_min=1.0, r_max=5.0, n_basis=6)
        r_outside = np.array([0.0, 0.5, 5.5, 10.0])
        phi = basis.evaluate(r_outside)
        np.testing.assert_allclose(phi, 0.0, atol=1e-15)

    def test_roughness_matrix_shape(self):
        basis = BSplineBasis(r_min=0.0, r_max=10.0, n_basis=8)
        R = basis.roughness_matrix()
        assert R.shape == (8, 8)

    def test_roughness_matrix_symmetric_positive_semidefinite(self):
        basis = BSplineBasis(r_min=0.0, r_max=10.0, n_basis=8)
        R = basis.roughness_matrix()
        np.testing.assert_allclose(R, R.T, atol=1e-10)
        eigenvalues = np.linalg.eigvalsh(R)
        assert np.all(eigenvalues >= -1e-10)

    def test_linear_kernel_has_zero_roughness(self):
        """A linear function has zero second derivative, so roughness penalty = 0."""
        basis = BSplineBasis(r_min=0.0, r_max=10.0, n_basis=8)
        R = basis.roughness_matrix()
        # Coefficients for a linear function: f(r) = a + b*r
        # We can approximate this by fitting a linear function to the basis
        r = np.linspace(0, 10, 200)
        phi = basis.evaluate(r)
        f_linear = r  # simple linear
        # Least-squares fit
        theta, _, _, _ = np.linalg.lstsq(phi, f_linear, rcond=None)
        roughness = theta @ R @ theta
        assert roughness < 0.1  # should be near zero


class TestHatBasis:
    def test_shape(self):
        basis = HatBasis(r_min=0.0, r_max=10.0, n_basis=5)
        r = np.linspace(0, 10, 50)
        phi = basis.evaluate(r)
        assert phi.shape == (50, 5)

    def test_partition_of_unity(self):
        basis = HatBasis(r_min=0.0, r_max=10.0, n_basis=5)
        r = np.linspace(0, 10, 100)
        phi = basis.evaluate(r)
        row_sums = phi.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_nonnegative(self):
        basis = HatBasis(r_min=0.0, r_max=10.0, n_basis=5)
        r = np.linspace(0, 10, 100)
        phi = basis.evaluate(r)
        assert np.all(phi >= -1e-15)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_basis.py -v`

- [ ] **Step 3: Implement basis.py**

```python
# chromlearn/model_fitting/basis.py
import numpy as np
from scipy.interpolate import BSpline


class BSplineBasis:
    """Cubic B-spline basis on [r_min, r_max].

    Creates n_basis cubic B-spline basis functions with uniform knot spacing.
    """

    def __init__(self, r_min: float, r_max: float, n_basis: int):
        self.r_min = r_min
        self.r_max = r_max
        self.n_basis = n_basis
        self.degree = 3

        # Create knot vector for n_basis cubic B-splines
        # Need n_basis + degree + 1 knots total
        # Interior knots: n_basis - degree + 1 = n_basis - 2 uniformly spaced
        n_interior = n_basis - self.degree + 1
        interior = np.linspace(r_min, r_max, n_interior)
        # Clamp: repeat boundary knots degree times
        self.knots = np.concatenate([
            np.full(self.degree, r_min),
            interior,
            np.full(self.degree, r_max),
        ])

        # Pre-build individual BSpline objects for each basis function
        self._splines = []
        for i in range(n_basis):
            coeffs = np.zeros(n_basis)
            coeffs[i] = 1.0
            self._splines.append(BSpline(self.knots, coeffs, self.degree, extrapolate=False))

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """Evaluate all basis functions at distances r.

        Args:
            r: 1D array of distances

        Returns:
            Array of shape (len(r), n_basis). Values outside [r_min, r_max] are 0.
        """
        r = np.asarray(r, dtype=np.float64)
        result = np.zeros((len(r), self.n_basis))
        for i, spline in enumerate(self._splines):
            vals = spline(r)
            # extrapolate=False returns NaN outside support; replace with 0
            vals = np.nan_to_num(vals, nan=0.0)
            result[:, i] = vals
        return result

    def roughness_matrix(self, n_quad: int = 500) -> np.ndarray:
        """Compute roughness penalty matrix R.

        R[i, j] = integral of phi_i''(r) * phi_j''(r) dr over [r_min, r_max].
        Computed via dense evaluation and trapezoidal integration.

        Returns:
            Symmetric positive semi-definite matrix of shape (n_basis, n_basis)
        """
        r = np.linspace(self.r_min, self.r_max, n_quad)
        # Evaluate second derivatives of each basis function
        d2 = np.zeros((n_quad, self.n_basis))
        for i, spline in enumerate(self._splines):
            d2_spline = spline.derivative(2)
            vals = d2_spline(r)
            vals = np.nan_to_num(vals, nan=0.0)
            d2[:, i] = vals
        # R = integral of d2^T d2 dr, via trapezoidal rule
        R = np.trapz(d2[:, :, np.newaxis] * d2[:, np.newaxis, :], r, axis=0)
        return R


class HatBasis:
    """Piecewise linear hat basis on [r_min, r_max].

    Creates n_basis evenly spaced hat (tent) functions.
    Useful for debugging.
    """

    def __init__(self, r_min: float, r_max: float, n_basis: int):
        self.r_min = r_min
        self.r_max = r_max
        self.n_basis = n_basis
        self.centers = np.linspace(r_min, r_max, n_basis)
        if n_basis > 1:
            self.width = self.centers[1] - self.centers[0]
        else:
            self.width = r_max - r_min

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """Evaluate all hat functions at distances r.

        Returns:
            Array of shape (len(r), n_basis)
        """
        r = np.asarray(r, dtype=np.float64)
        result = np.zeros((len(r), self.n_basis))
        for i, c in enumerate(self.centers):
            result[:, i] = np.maximum(0.0, 1.0 - np.abs(r - c) / self.width)
        return result

    def roughness_matrix(self) -> np.ndarray:
        """Roughness matrix for hat basis.

        Hat functions have piecewise constant first derivative and
        zero second derivative (except at knots). Return zeros for simplicity.
        Roughness regularization is less meaningful for hat functions;
        use ridge regularization instead.
        """
        return np.zeros((self.n_basis, self.n_basis))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_basis.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chromlearn/model_fitting/basis.py tests/test_basis.py
git commit -m "feat: add B-spline and hat basis functions with roughness matrices"
```

---

## Task 6: Design Matrix Construction (`chromlearn/model_fitting/features.py`)

**Files:**
- Create: `chromlearn/model_fitting/features.py`, `tests/test_features.py`

**Context:** This is the mathematical core. For each chromosome i at each timepoint t_n, we need:

1. **Response:** `v_i^n = (x_i(t_{n+1}) - x_i(t_n)) / dt` — the displacement velocity (3D vector)
2. **Chromosome-chromosome features:** For each basis function m:
   `g^xx_{i,m}(t_n) = sum_{k != i} phi_m(r_ik(t_n)) * r_hat_ik(t_n)`
   where r_ik = ||x_k - x_i||, r_hat_ik = (x_k - x_i) / r_ik. This is a 3D vector.
3. **Centrosome-on-chromosome features:** For each basis function m:
   `g^xy_{i,m}(t_n) = sum_j phi_m(rho_ij(t_n)) * rho_hat_ij(t_n)`
   where rho_ij = ||y_j - x_i||, rho_hat_ij = (y_j - x_i) / rho_ij. This is a 3D vector.

The design matrix G has rows indexed by (chromosome i, timepoint n, spatial dimension d). Each row has `n_basis_xx + n_basis_xy` columns (xx basis features then xy basis features). The response vector V has the same rows, each being one component of the displacement velocity.

So for P chromosomes, T-1 timepoints (increments), and d=3 dimensions:
- G has shape `(P * (T-1) * 3, n_basis_xx + n_basis_xy)`
- V has shape `(P * (T-1) * 3,)`

**NaN handling:** Skip any (i, n) pair where chromosome i has NaN position at t_n or t_{n+1}. When computing pairwise sums, skip neighbors k with NaN positions.

For multiple cells, stack the rows from each cell vertically.

- [ ] **Step 1: Write tests**

```python
# tests/test_features.py
import numpy as np
import pytest
from chromlearn.io.trajectory import TrimmedCell
from chromlearn.model_fitting.basis import BSplineBasis, HatBasis
from chromlearn.model_fitting.features import build_design_matrix


def make_simple_trimmed_cell(T=20, N=4):
    """Create a simple trimmed cell with known geometry.

    4 chromosomes at corners of a square, 2 centrosomes on x-axis.
    All stationary (displacements = 0).
    """
    centrioles = np.zeros((T, 3, 2))
    centrioles[:, 0, 0] = -5.0  # pole 1 at (-5, 0, 0)
    centrioles[:, 0, 1] = 5.0   # pole 2 at (5, 0, 0)

    chromosomes = np.zeros((T, 3, N))
    # Place chromosomes at (1,1,0), (-1,1,0), (-1,-1,0), (1,-1,0)
    positions = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]], dtype=float)
    for j in range(N):
        chromosomes[:, :, j] = positions[j]

    return TrimmedCell(
        cell_id="test_001", condition="test",
        centrioles=centrioles, chromosomes=chromosomes,
        tracked=N, dt=5.0, start_frame=0, end_frame=T - 1,
    )


def test_design_matrix_shape():
    cell = make_simple_trimmed_cell(T=20, N=4)
    basis_xx = HatBasis(0.0, 10.0, n_basis=5)
    basis_xy = HatBasis(0.0, 15.0, n_basis=5)
    G, V = build_design_matrix([cell], basis_xx, basis_xy)
    # 4 chromosomes, 19 increments (T-1), 3 dimensions
    expected_rows = 4 * 19 * 3
    expected_cols = 5 + 5  # n_basis_xx + n_basis_xy
    assert G.shape == (expected_rows, expected_cols)
    assert V.shape == (expected_rows,)


def test_stationary_particles_zero_response():
    """If particles don't move, all displacement velocities should be 0."""
    cell = make_simple_trimmed_cell(T=20, N=4)
    basis_xx = HatBasis(0.0, 10.0, n_basis=5)
    basis_xy = HatBasis(0.0, 15.0, n_basis=5)
    G, V = build_design_matrix([cell], basis_xx, basis_xy)
    np.testing.assert_allclose(V, 0.0, atol=1e-15)


def test_design_matrix_nonzero_features():
    """Features should be nonzero (particles have neighbors in basis support)."""
    cell = make_simple_trimmed_cell(T=20, N=4)
    basis_xx = HatBasis(0.0, 10.0, n_basis=5)
    basis_xy = HatBasis(0.0, 15.0, n_basis=5)
    G, V = build_design_matrix([cell], basis_xx, basis_xy)
    # At least some features should be nonzero
    assert np.any(np.abs(G) > 1e-10)


def test_multiple_cells_stacked():
    cell1 = make_simple_trimmed_cell(T=10, N=3)
    cell2 = make_simple_trimmed_cell(T=15, N=3)
    basis_xx = HatBasis(0.0, 10.0, n_basis=4)
    basis_xy = HatBasis(0.0, 15.0, n_basis=4)
    G, V = build_design_matrix([cell1, cell2], basis_xx, basis_xy)
    # cell1: 3 * 9 * 3 = 81 rows, cell2: 3 * 14 * 3 = 126 rows
    expected_rows = 81 + 126
    assert G.shape[0] == expected_rows


def test_nan_handling():
    """Rows with NaN positions should be excluded."""
    cell = make_simple_trimmed_cell(T=20, N=4)
    # Introduce NaN for chromosome 0 at timepoint 5
    cell.chromosomes[5, :, 0] = np.nan
    basis_xx = HatBasis(0.0, 10.0, n_basis=5)
    basis_xy = HatBasis(0.0, 15.0, n_basis=5)
    G, V = build_design_matrix([cell], basis_xx, basis_xy)
    # Should have fewer rows than the full case
    full_rows = 4 * 19 * 3
    assert G.shape[0] < full_rows
    # No NaN values in the output
    assert not np.any(np.isnan(G))
    assert not np.any(np.isnan(V))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features.py -v`

- [ ] **Step 3: Implement features.py**

```python
# chromlearn/model_fitting/features.py
import numpy as np
from chromlearn.io.trajectory import TrimmedCell


def build_design_matrix(
    cells: list[TrimmedCell],
    basis_xx,
    basis_xy,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the stacked design matrix G and response vector V.

    For each chromosome i at each timepoint t_n:
    - Response: v_i^n = (x_i(t_{n+1}) - x_i(t_n)) / dt  (3D vector)
    - Features: basis evaluations * unit direction vectors for
      chromosome-chromosome (xx) and centrosome-on-chromosome (xy) interactions

    Rows with NaN chromosome positions are excluded.

    Args:
        cells: List of TrimmedCell objects (possibly from multiple cells)
        basis_xx: Basis object for chromosome-chromosome kernel (has .evaluate(r) method)
        basis_xy: Basis object for centrosome-on-chromosome kernel

    Returns:
        G: Design matrix, shape (n_valid_obs * 3, n_basis_xx + n_basis_xy)
        V: Response vector, shape (n_valid_obs * 3,)
    """
    G_blocks = []
    V_blocks = []

    for cell in cells:
        _build_cell_features(cell, basis_xx, basis_xy, G_blocks, V_blocks)

    if not G_blocks:
        n_cols = basis_xx.n_basis + basis_xy.n_basis
        return np.zeros((0, n_cols)), np.zeros(0)

    G = np.vstack(G_blocks)
    V = np.concatenate(V_blocks)
    return G, V


def _build_cell_features(
    cell: TrimmedCell,
    basis_xx,
    basis_xy,
    G_blocks: list,
    V_blocks: list,
):
    """Append design matrix rows for one cell."""
    chrom = cell.chromosomes  # (T, 3, N)
    centr = cell.centrioles   # (T, 3, 2)
    T, _, N = chrom.shape
    dt = cell.dt
    n_xx = basis_xx.n_basis
    n_xy = basis_xy.n_basis

    for n in range(T - 1):
        for i in range(N):
            # Check if chromosome i is valid at t_n and t_{n+1}
            pos_n = chrom[n, :, i]    # (3,)
            pos_n1 = chrom[n + 1, :, i]  # (3,)
            if np.any(np.isnan(pos_n)) or np.any(np.isnan(pos_n1)):
                continue

            # Response: displacement velocity
            v = (pos_n1 - pos_n) / dt  # (3,)

            # --- Chromosome-chromosome features (xx) ---
            g_xx = np.zeros((3, n_xx))
            for k in range(N):
                if k == i:
                    continue
                pos_k = chrom[n, :, k]
                if np.any(np.isnan(pos_k)):
                    continue
                delta = pos_k - pos_n  # (3,)
                r_ik = np.linalg.norm(delta)
                if r_ik < 1e-12:
                    continue  # avoid division by zero
                r_hat = delta / r_ik  # (3,) unit vector
                phi_vals = basis_xx.evaluate(np.array([r_ik]))  # (1, n_xx)
                # Outer product: each basis function contributes phi_m * r_hat
                g_xx += r_hat[:, np.newaxis] * phi_vals  # (3, n_xx)

            # --- Centrosome-on-chromosome features (xy) ---
            g_xy = np.zeros((3, n_xy))
            for j in range(2):  # 2 centrosomes
                pos_j = centr[n, :, j]
                delta = pos_j - pos_n  # (3,)
                rho_ij = np.linalg.norm(delta)
                if rho_ij < 1e-12:
                    continue
                rho_hat = delta / rho_ij  # (3,)
                phi_vals = basis_xy.evaluate(np.array([rho_ij]))  # (1, n_xy)
                g_xy += rho_hat[:, np.newaxis] * phi_vals  # (3, n_xy)

            # Stack xx and xy features: (3, n_xx + n_xy)
            g_row = np.hstack([g_xx, g_xy])  # (3, n_total)

            # Append 3 rows (one per spatial dimension)
            G_blocks.append(g_row)
            V_blocks.append(v)
```

**Performance note:** This nested-loop implementation is O(N^2 * T) per cell and will be slow for 46 chromosomes x 100 timepoints x 7 cells in the current NEB-annotated `rpe18_ctr` subset. It is correct and clear. After verifying correctness, it can be vectorized in a follow-up. For now, prioritize clarity. A full vectorization can be done later — the design matrix only needs to be built once per fit.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_features.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chromlearn/model_fitting/features.py tests/test_features.py
git commit -m "feat: add design matrix construction for pairwise kernel regression"
```

---

## Task 7: Regression and Fitting (`chromlearn/model_fitting/fit.py`)

**Files:**
- Create: `chromlearn/model_fitting/fit.py`, `tests/test_fit.py`

**Context:** Given the design matrix G and response vector V from Task 6, solve the penalized least-squares problem:

```
theta_hat = argmin ||V - G*theta||^2 + lambda_ridge * ||theta||^2 + lambda_rough * theta^T R theta
```

This has closed-form solution:
```
theta_hat = (G^T G + lambda_ridge * I + lambda_rough * R)^{-1} G^T V
```

Use `numpy.linalg.solve` (or `scipy.linalg.solve` with `assume_a='pos'`) for numerical stability.

Also implement:
- `estimate_diffusion`: compute D_x from residuals
- `bootstrap_kernels`: resample cells, refit, return distribution of theta
- `cross_validate`: leave-one-cell-out CV

- [ ] **Step 1: Write tests**

```python
# tests/test_fit.py
import numpy as np
import pytest
from chromlearn.model_fitting.fit import (
    fit_kernels,
    estimate_diffusion,
)


def test_fit_kernels_recovers_known_theta():
    """If V = G @ theta_true + small noise, fit should recover theta_true."""
    rng = np.random.default_rng(42)
    n_obs, n_basis = 500, 6
    theta_true = rng.standard_normal(n_basis)
    G = rng.standard_normal((n_obs, n_basis))
    noise = 0.01 * rng.standard_normal(n_obs)
    V = G @ theta_true + noise

    R = np.zeros((n_basis, n_basis))  # no roughness
    result = fit_kernels(G, V, lambda_ridge=1e-6, lambda_rough=0.0, R=R)
    np.testing.assert_allclose(result.theta, theta_true, atol=0.05)


def test_fit_kernels_with_roughness():
    """Adding roughness regularization should not crash and should return theta."""
    rng = np.random.default_rng(42)
    n_obs, n_basis = 200, 5
    G = rng.standard_normal((n_obs, n_basis))
    V = rng.standard_normal(n_obs)
    R = np.eye(n_basis)  # simple roughness

    result = fit_kernels(G, V, lambda_ridge=0.01, lambda_rough=0.01, R=R)
    assert result.theta.shape == (n_basis,)


def test_fit_kernels_result_has_residuals():
    rng = np.random.default_rng(42)
    G = rng.standard_normal((100, 4))
    V = rng.standard_normal(100)
    R = np.zeros((4, 4))
    result = fit_kernels(G, V, lambda_ridge=0.01, lambda_rough=0.0, R=R)
    assert hasattr(result, "residuals")
    assert result.residuals.shape == (100,)


def test_estimate_diffusion():
    """Diffusion estimate from known noise level."""
    rng = np.random.default_rng(42)
    D_true = 0.5
    dt = 5.0
    d = 3
    n_obs = 10000
    # Simulate: V = 0 + noise with variance 2*D*dt per component
    # Since we divide displacements by dt to get V, the noise variance
    # of V is 2*D/(dt) per component
    noise_std = np.sqrt(2 * D_true / dt)
    V = noise_std * rng.standard_normal(n_obs * d)
    G = np.zeros((n_obs * d, 1))
    theta = np.array([0.0])

    D_hat = estimate_diffusion(V, G, theta, dt=dt, d=d)
    np.testing.assert_allclose(D_hat, D_true, rtol=0.1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fit.py -v`

- [ ] **Step 3: Implement fit.py**

```python
# chromlearn/model_fitting/fit.py
from dataclasses import dataclass
import numpy as np
from chromlearn.io.trajectory import TrimmedCell


@dataclass
class FitResult:
    """Result of kernel regression."""
    theta: np.ndarray         # fitted coefficients, shape (n_basis_total,)
    residuals: np.ndarray     # V - G @ theta, shape (n_obs,)
    D_x: float                # estimated diffusion coefficient


@dataclass
class BootstrapResult:
    """Result of bootstrap over cells."""
    theta_samples: np.ndarray  # (n_boot, n_basis_total)
    theta_mean: np.ndarray     # (n_basis_total,)
    theta_std: np.ndarray      # (n_basis_total,)


@dataclass
class CVResult:
    """Result of cross-validation."""
    held_out_errors: np.ndarray  # (n_folds,) mean squared error per fold
    mean_error: float
    std_error: float


def fit_kernels(
    G: np.ndarray,
    V: np.ndarray,
    lambda_ridge: float,
    lambda_rough: float,
    R: np.ndarray,
) -> FitResult:
    """Solve penalized least squares for kernel coefficients.

    Solves: theta = argmin ||V - G @ theta||^2 + lambda_ridge * ||theta||^2
                                                + lambda_rough * theta^T R theta

    Closed-form: theta = (G^T G + lambda_ridge * I + lambda_rough * R)^{-1} G^T V

    Args:
        G: Design matrix, shape (n_obs, n_basis)
        V: Response vector, shape (n_obs,)
        lambda_ridge: Ridge regularization strength
        lambda_rough: Roughness regularization strength
        R: Roughness penalty matrix, shape (n_basis, n_basis)

    Returns:
        FitResult with theta, residuals, and D_x=0 (call estimate_diffusion separately)
    """
    n_basis = G.shape[1]
    A = G.T @ G + lambda_ridge * np.eye(n_basis) + lambda_rough * R
    b = G.T @ V
    theta = np.linalg.solve(A, b)
    residuals = V - G @ theta
    return FitResult(theta=theta, residuals=residuals, D_x=0.0)


def estimate_diffusion(
    V: np.ndarray,
    G: np.ndarray,
    theta: np.ndarray,
    dt: float,
    d: int = 3,
) -> float:
    """Estimate diffusion coefficient from regression residuals.

    D_x = (1 / (2 * d * N_obs)) * sum ||residual||^2 * dt

    where N_obs = len(V) / d (number of particle-timestep pairs),
    and residuals are already in velocity units (displacement / dt).

    The variance of displacement noise is 2*D*dt per component.
    The variance of velocity noise (displacement/dt) is 2*D/dt per component.
    So D = var(residuals) * dt / 2.

    Args:
        V: Response vector, shape (n_obs * d,) — flattened velocity components
        G: Design matrix
        theta: Fitted coefficients
        dt: Time step in seconds
        d: Spatial dimension (default 3)

    Returns:
        Estimated diffusion coefficient D_x
    """
    residuals = V - G @ theta
    # Mean squared residual per component
    msr = np.mean(residuals**2)
    # D = msr * dt / 2
    return msr * dt / 2.0


def bootstrap_kernels(
    cells: list[TrimmedCell],
    basis_xx,
    basis_xy,
    n_boot: int = 250,
    lambda_ridge: float = 1e-3,
    lambda_rough: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """Bootstrap kernel fits by resampling cells with replacement.

    Args:
        cells: List of TrimmedCell objects
        basis_xx, basis_xy: Basis objects
        n_boot: Number of bootstrap iterations
        lambda_ridge, lambda_rough: Regularization strengths
        rng: Random number generator

    Returns:
        BootstrapResult with theta_samples, theta_mean, theta_std
    """
    from chromlearn.model_fitting.features import build_design_matrix

    if rng is None:
        rng = np.random.default_rng()

    n_cells = len(cells)
    n_basis = basis_xx.n_basis + basis_xy.n_basis
    R_xx = basis_xx.roughness_matrix()
    R_xy = basis_xy.roughness_matrix()
    R = _block_roughness(R_xx, R_xy)

    theta_samples = np.zeros((n_boot, n_basis))

    for b in range(n_boot):
        idx = rng.choice(n_cells, size=n_cells, replace=True)
        boot_cells = [cells[i] for i in idx]
        G, V = build_design_matrix(boot_cells, basis_xx, basis_xy)
        result = fit_kernels(G, V, lambda_ridge, lambda_rough, R)
        theta_samples[b] = result.theta

    return BootstrapResult(
        theta_samples=theta_samples,
        theta_mean=theta_samples.mean(axis=0),
        theta_std=theta_samples.std(axis=0),
    )


def cross_validate(
    cells: list[TrimmedCell],
    basis_xx,
    basis_xy,
    lambda_ridge: float = 1e-3,
    lambda_rough: float = 1e-3,
) -> CVResult:
    """Leave-one-cell-out cross-validation.

    For each fold:
    1. Exclude one cell
    2. Build design matrix from remaining cells
    3. Fit theta
    4. Build design matrix for held-out cell
    5. Compute mean squared prediction error on held-out cell

    Returns:
        CVResult with per-fold errors and summary statistics
    """
    from chromlearn.model_fitting.features import build_design_matrix

    n_cells = len(cells)
    R_xx = basis_xx.roughness_matrix()
    R_xy = basis_xy.roughness_matrix()
    R = _block_roughness(R_xx, R_xy)

    errors = np.zeros(n_cells)

    for i in range(n_cells):
        train_cells = [cells[j] for j in range(n_cells) if j != i]
        test_cell = [cells[i]]

        G_train, V_train = build_design_matrix(train_cells, basis_xx, basis_xy)
        result = fit_kernels(G_train, V_train, lambda_ridge, lambda_rough, R)

        G_test, V_test = build_design_matrix(test_cell, basis_xx, basis_xy)
        if len(V_test) == 0:
            errors[i] = np.nan
            continue
        pred = G_test @ result.theta
        errors[i] = np.mean((V_test - pred) ** 2)

    return CVResult(
        held_out_errors=errors,
        mean_error=np.nanmean(errors),
        std_error=np.nanstd(errors),
    )


def _block_roughness(R_xx: np.ndarray, R_xy: np.ndarray) -> np.ndarray:
    """Build block-diagonal roughness matrix from per-kernel matrices."""
    from scipy.linalg import block_diag
    return block_diag(R_xx, R_xy)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fit.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chromlearn/model_fitting/fit.py tests/test_fit.py
git commit -m "feat: add penalized regression, diffusion estimation, bootstrap, and CV"
```

---

## Task 8: Fitted Model Container (`chromlearn/model_fitting/model.py`)

**Files:**
- Create: `chromlearn/model_fitting/model.py`

**Context:** Lightweight container that holds fitted coefficients and basis configs, and can evaluate the learned kernels at arbitrary distances. Also supports save/load via numpy `.npz` files.

- [ ] **Step 1: Implement model.py**

```python
# chromlearn/model_fitting/model.py
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from chromlearn.model_fitting.basis import BSplineBasis, HatBasis


@dataclass
class FittedModel:
    """Container for a fitted interaction kernel model."""
    theta: np.ndarray          # all coefficients (n_basis_xx + n_basis_xy,)
    n_basis_xx: int
    n_basis_xy: int
    basis_xx: BSplineBasis | HatBasis
    basis_xy: BSplineBasis | HatBasis
    D_x: float                 # estimated diffusion coefficient
    dt: float
    metadata: dict | None = None  # optional: condition, n_cells, etc.

    @property
    def theta_xx(self) -> np.ndarray:
        return self.theta[:self.n_basis_xx]

    @property
    def theta_xy(self) -> np.ndarray:
        return self.theta[self.n_basis_xx:]

    def evaluate_kernel(self, kernel: str, r: np.ndarray) -> np.ndarray:
        """Evaluate a learned kernel at distances r.

        Args:
            kernel: "xx" for chromosome-chromosome, "xy" for centrosome-on-chromosome
            r: 1D array of distances

        Returns:
            1D array of kernel values f(r)
        """
        r = np.asarray(r)
        if kernel == "xx":
            phi = self.basis_xx.evaluate(r)  # (len(r), n_basis_xx)
            return phi @ self.theta_xx
        elif kernel == "xy":
            phi = self.basis_xy.evaluate(r)  # (len(r), n_basis_xy)
            return phi @ self.theta_xy
        else:
            raise ValueError(f"Unknown kernel: {kernel}. Use 'xx' or 'xy'.")

    def save(self, path: str | Path):
        """Save model to .npz file."""
        path = Path(path)
        np.savez(
            path,
            theta=self.theta,
            n_basis_xx=self.n_basis_xx,
            n_basis_xy=self.n_basis_xy,
            D_x=self.D_x,
            dt=self.dt,
            # Basis params
            basis_xx_r_min=self.basis_xx.r_min,
            basis_xx_r_max=self.basis_xx.r_max,
            basis_xx_n_basis=self.basis_xx.n_basis,
            basis_xx_type="bspline" if isinstance(self.basis_xx, BSplineBasis) else "hat",
            basis_xy_r_min=self.basis_xy.r_min,
            basis_xy_r_max=self.basis_xy.r_max,
            basis_xy_n_basis=self.basis_xy.n_basis,
            basis_xy_type="bspline" if isinstance(self.basis_xy, BSplineBasis) else "hat",
        )

    @classmethod
    def load(cls, path: str | Path) -> "FittedModel":
        """Load model from .npz file."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)

        def make_basis(type_str, r_min, r_max, n_basis):
            if type_str == "bspline":
                return BSplineBasis(float(r_min), float(r_max), int(n_basis))
            else:
                return HatBasis(float(r_min), float(r_max), int(n_basis))

        basis_xx = make_basis(
            str(data["basis_xx_type"]), data["basis_xx_r_min"],
            data["basis_xx_r_max"], data["basis_xx_n_basis"]
        )
        basis_xy = make_basis(
            str(data["basis_xy_type"]), data["basis_xy_r_min"],
            data["basis_xy_r_max"], data["basis_xy_n_basis"]
        )

        return cls(
            theta=data["theta"],
            n_basis_xx=int(data["n_basis_xx"]),
            n_basis_xy=int(data["n_basis_xy"]),
            basis_xx=basis_xx,
            basis_xy=basis_xy,
            D_x=float(data["D_x"]),
            dt=float(data["dt"]),
        )
```

- [ ] **Step 2: Commit**

```bash
git add chromlearn/model_fitting/model.py
git commit -m "feat: add FittedModel container with save/load and kernel evaluation"
```

---

## Task 9: Simulator (`chromlearn/model_fitting/simulate.py`)

**Files:**
- Create: `chromlearn/model_fitting/simulate.py`, `tests/test_simulate.py`

**Context:** Euler-Maruyama forward simulator for the overdamped Langevin model. Used for:
1. Generating synthetic benchmark data with known kernels to validate the fitting pipeline
2. Forward-simulating from a learned model to compare statistics with real data

The simulator takes kernel functions (callables), centrosome trajectories (fixed/given), initial chromosome positions, and diffusion coefficient.

- [ ] **Step 1: Write tests**

```python
# tests/test_simulate.py
import numpy as np
import pytest
from chromlearn.model_fitting.simulate import (
    simulate_trajectories,
    generate_synthetic_data,
    add_localization_noise,
)


def test_simulate_output_shape():
    """Simulator should return correct shape."""
    n_chrom = 10
    n_steps = 50
    # Centrosomes: fixed at (-5,0,0) and (5,0,0)
    centrosomes = np.zeros((n_steps + 1, 3, 2))
    centrosomes[:, 0, 0] = -5.0
    centrosomes[:, 0, 1] = 5.0
    # Zero kernels (pure diffusion)
    kernel_xx = lambda r: np.zeros_like(r)
    kernel_xy = lambda r: np.zeros_like(r)
    # Random initial positions
    rng = np.random.default_rng(42)
    x0 = rng.normal(0, 2, (n_chrom, 3))

    traj = simulate_trajectories(
        kernel_xx=kernel_xx, kernel_xy=kernel_xy,
        centrosome_positions=centrosomes,
        x0=x0, n_steps=n_steps, dt=5.0, D_x=0.1, rng=rng,
    )
    assert traj.shape == (n_steps + 1, 3, n_chrom)


def test_pure_diffusion_msd():
    """With zero forces, MSD should grow linearly with slope 2*d*D."""
    rng = np.random.default_rng(42)
    n_chrom = 100
    n_steps = 200
    dt = 1.0
    D = 0.5
    d = 3

    centrosomes = np.zeros((n_steps + 1, 3, 2))
    x0 = np.zeros((n_chrom, 3))

    traj = simulate_trajectories(
        kernel_xx=lambda r: np.zeros_like(r),
        kernel_xy=lambda r: np.zeros_like(r),
        centrosome_positions=centrosomes,
        x0=x0, n_steps=n_steps, dt=dt, D_x=D, rng=rng,
    )
    # MSD at lag 1
    displacements = traj[1:] - traj[:-1]  # (n_steps, 3, n_chrom)
    msd_per_step = np.mean(np.sum(displacements**2, axis=1))
    expected = 2 * d * D * dt
    np.testing.assert_allclose(msd_per_step, expected, rtol=0.15)


def test_add_localization_noise():
    rng = np.random.default_rng(42)
    traj = np.zeros((50, 3, 10))
    noisy = add_localization_noise(traj, sigma=0.1, rng=rng)
    assert noisy.shape == traj.shape
    assert not np.allclose(noisy, traj)
    # Noise std should be approximately sigma
    np.testing.assert_allclose(np.std(noisy), 0.1, rtol=0.2)


def test_generate_synthetic_data():
    result = generate_synthetic_data(
        kernel_xx=lambda r: -0.01 * r,  # weak repulsion
        kernel_xy=lambda r: 0.05 * np.ones_like(r),  # attraction to centrosomes
        n_chromosomes=10, n_steps=50, dt=5.0, D_x=0.1,
        rng=np.random.default_rng(42),
    )
    assert result.chromosomes.shape == (51, 3, 10)
    assert result.centrosomes.shape == (51, 3, 2)
    assert result.kernel_xx is not None
    assert result.kernel_xy is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_simulate.py -v`

- [ ] **Step 3: Implement simulate.py**

```python
# chromlearn/model_fitting/simulate.py
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class SyntheticDataset:
    """Container for synthetic benchmark data."""
    chromosomes: np.ndarray     # (T, 3, N) simulated chromosome trajectories
    centrosomes: np.ndarray     # (T, 3, 2) centrosome positions used
    kernel_xx: Callable         # true chromosome-chromosome kernel
    kernel_xy: Callable         # true centrosome-on-chromosome kernel
    D_x: float                  # true diffusion coefficient
    dt: float


def simulate_trajectories(
    kernel_xx: Callable[[np.ndarray], np.ndarray],
    kernel_xy: Callable[[np.ndarray], np.ndarray],
    centrosome_positions: np.ndarray,
    x0: np.ndarray,
    n_steps: int,
    dt: float,
    D_x: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Euler-Maruyama forward simulation of overdamped Langevin dynamics.

    Args:
        kernel_xx: f_xx(r) — chromosome-chromosome kernel. Takes 1D array of
            distances, returns 1D array of force magnitudes.
        kernel_xy: f_xy(r) — centrosome-on-chromosome kernel. Same signature.
        centrosome_positions: (n_steps+1, 3, 2) — centrosome positions over time
        x0: (N, 3) — initial chromosome positions
        n_steps: number of time steps to simulate
        dt: time step size
        D_x: diffusion coefficient
        rng: random number generator

    Returns:
        Trajectory array of shape (n_steps+1, 3, N)
    """
    if rng is None:
        rng = np.random.default_rng()

    N = x0.shape[0]
    d = 3
    noise_std = np.sqrt(2 * D_x * dt)

    # Initialize trajectory
    traj = np.zeros((n_steps + 1, d, N))
    traj[0] = x0.T  # (3, N)

    for n in range(n_steps):
        x_n = traj[n]  # (3, N)
        centr_n = centrosome_positions[n]  # (3, 2)

        # Compute forces on each chromosome
        forces = np.zeros((d, N))
        for i in range(N):
            xi = x_n[:, i]  # (3,)

            # Chromosome-chromosome interactions
            for k in range(N):
                if k == i:
                    continue
                delta = x_n[:, k] - xi  # (3,)
                r = np.linalg.norm(delta)
                if r < 1e-12:
                    continue
                r_hat = delta / r
                f_val = kernel_xx(np.array([r]))[0]
                forces[:, i] += f_val * r_hat

            # Centrosome-on-chromosome interactions
            for j in range(2):
                delta = centr_n[:, j] - xi
                rho = np.linalg.norm(delta)
                if rho < 1e-12:
                    continue
                rho_hat = delta / rho
                f_val = kernel_xy(np.array([rho]))[0]
                forces[:, i] += f_val * rho_hat

        # Euler-Maruyama step
        noise = noise_std * rng.standard_normal((d, N))
        traj[n + 1] = x_n + forces * dt + noise

    return traj


def generate_synthetic_data(
    kernel_xx: Callable,
    kernel_xy: Callable,
    n_chromosomes: int = 20,
    n_steps: int = 100,
    dt: float = 5.0,
    D_x: float = 0.1,
    pole_separation: float = 10.0,
    rng: np.random.Generator | None = None,
) -> SyntheticDataset:
    """Generate a full synthetic benchmark dataset.

    Creates centrosomes at fixed positions on x-axis (symmetric around origin),
    random initial chromosome positions, and simulates forward.

    Args:
        kernel_xx, kernel_xy: True interaction kernels
        n_chromosomes: Number of chromosomes
        n_steps: Number of time steps
        dt: Time step
        D_x: Diffusion coefficient
        pole_separation: Distance between centrosomes
        rng: Random number generator

    Returns:
        SyntheticDataset with trajectories and ground truth
    """
    if rng is None:
        rng = np.random.default_rng()

    # Fixed centrosomes
    centrosomes = np.zeros((n_steps + 1, 3, 2))
    centrosomes[:, 0, 0] = -pole_separation / 2
    centrosomes[:, 0, 1] = pole_separation / 2

    # Random initial positions (scattered around origin)
    x0 = rng.normal(0, 2.0, (n_chromosomes, 3))

    chromosomes = simulate_trajectories(
        kernel_xx=kernel_xx, kernel_xy=kernel_xy,
        centrosome_positions=centrosomes,
        x0=x0, n_steps=n_steps, dt=dt, D_x=D_x, rng=rng,
    )

    return SyntheticDataset(
        chromosomes=chromosomes,
        centrosomes=centrosomes,
        kernel_xx=kernel_xx,
        kernel_xy=kernel_xy,
        D_x=D_x,
        dt=dt,
    )


def add_localization_noise(
    trajectories: np.ndarray,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add isotropic Gaussian localization noise to trajectory positions.

    Args:
        trajectories: (T, 3, N) position array
        sigma: Standard deviation of localization noise in each coordinate
        rng: Random number generator

    Returns:
        Noisy trajectory array of same shape
    """
    if rng is None:
        rng = np.random.default_rng()
    return trajectories + sigma * rng.standard_normal(trajectories.shape)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_simulate.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chromlearn/model_fitting/simulate.py tests/test_simulate.py
git commit -m "feat: add Euler-Maruyama simulator and synthetic data generation"
```

---

## Task 10: Validation (`chromlearn/model_fitting/validate.py`)

**Files:**
- Create: `chromlearn/model_fitting/validate.py`, `tests/test_validate.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_validate.py
import numpy as np
import pytest
from chromlearn.model_fitting.validate import (
    one_step_prediction_error,
    residual_diagnostics,
    kernel_recovery_error,
)


def test_one_step_prediction_error_perfect_fit():
    """If model predicts perfectly, error should be ~0."""
    V = np.array([1.0, 2.0, 3.0])
    G = np.eye(3)
    theta = np.array([1.0, 2.0, 3.0])
    err = one_step_prediction_error(V, G, theta)
    np.testing.assert_allclose(err, 0.0, atol=1e-15)


def test_residual_diagnostics_returns_dict():
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal(300)
    diag = residual_diagnostics(residuals)
    assert "mean" in diag
    assert "std" in diag
    assert "skewness" in diag
    assert "kurtosis" in diag


def test_kernel_recovery_error():
    r = np.linspace(0, 10, 100)
    true_vals = np.sin(r)
    fitted_vals = np.sin(r) + 0.01 * np.ones_like(r)
    err = kernel_recovery_error(r, true_vals, fitted_vals)
    assert err < 0.02
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement validate.py**

```python
# chromlearn/model_fitting/validate.py
import numpy as np
from scipy import stats


def one_step_prediction_error(
    V: np.ndarray, G: np.ndarray, theta: np.ndarray
) -> float:
    """Mean squared one-step prediction error."""
    pred = G @ theta
    return float(np.mean((V - pred) ** 2))


def residual_diagnostics(residuals: np.ndarray) -> dict:
    """Compute diagnostic statistics for regression residuals.

    Args:
        residuals: 1D array of residuals (V - G @ theta)

    Returns:
        Dict with mean, std, skewness, kurtosis, and normality test p-value
    """
    result = {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "skewness": float(stats.skew(residuals)),
        "kurtosis": float(stats.kurtosis(residuals)),
    }
    # Shapiro-Wilk test (on subsample if too large)
    if len(residuals) > 5000:
        rng = np.random.default_rng(0)
        sub = rng.choice(residuals, size=5000, replace=False)
    else:
        sub = residuals
    if len(sub) >= 3:
        _, p_value = stats.shapiro(sub)
        result["normality_p_value"] = float(p_value)
    return result


def kernel_recovery_error(
    r: np.ndarray,
    true_values: np.ndarray,
    fitted_values: np.ndarray,
) -> float:
    """L2 error between true and fitted kernel values.

    Computes: sqrt(mean((true - fitted)^2))

    Args:
        r: Distance values where kernels are evaluated
        true_values: True kernel f(r)
        fitted_values: Fitted kernel f_hat(r)

    Returns:
        RMS error
    """
    return float(np.sqrt(np.mean((true_values - fitted_values) ** 2)))


def summary_statistics(chromosomes: np.ndarray, centrosomes: np.ndarray) -> dict:
    """Compute summary statistics for trajectory comparison.

    Useful for comparing simulated vs real trajectories.

    Args:
        chromosomes: (T, 3, N) chromosome positions
        centrosomes: (T, 3, 2) centrosome positions

    Returns:
        Dict with various summary statistics
    """
    T, _, N = chromosomes.shape
    pc = 0.5 * (centrosomes[:, :, 0] + centrosomes[:, :, 1])  # (T, 3)

    # Distance from pole center for each chromosome
    dists = np.zeros((T, N))
    for j in range(N):
        dists[:, j] = np.linalg.norm(chromosomes[:, :, j] - pc, axis=1)

    # Pairwise chromosome distances (at each timepoint, sample)
    pairwise_dists = []
    for t in range(0, T, max(1, T // 10)):  # subsample timepoints
        for i in range(N):
            for k in range(i + 1, N):
                d = np.linalg.norm(chromosomes[t, :, i] - chromosomes[t, :, k])
                pairwise_dists.append(d)

    # MSD at lag 1
    displacements = chromosomes[1:] - chromosomes[:-1]
    msd_lag1 = np.mean(np.sum(displacements**2, axis=1))

    return {
        "mean_dist_from_center": float(np.mean(dists)),
        "std_dist_from_center": float(np.std(dists)),
        "mean_pairwise_dist": float(np.mean(pairwise_dists)),
        "std_pairwise_dist": float(np.std(pairwise_dists)),
        "msd_lag1": float(msd_lag1),
    }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_validate.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chromlearn/model_fitting/validate.py tests/test_validate.py
git commit -m "feat: add validation diagnostics and summary statistics"
```

---

## Task 11: Plotting (`chromlearn/model_fitting/plotting.py`)

**Files:**
- Create: `chromlearn/model_fitting/plotting.py`

**Context:** Plotting functions for learned kernels, cross-validation curves, residual diagnostics, and synthetic recovery comparisons. These are called from notebooks. Use matplotlib.

- [ ] **Step 1: Implement plotting.py**

```python
# chromlearn/model_fitting/plotting.py
import numpy as np
import matplotlib.pyplot as plt
from chromlearn.model_fitting.model import FittedModel
from chromlearn.model_fitting.fit import BootstrapResult, CVResult


def plot_kernels(
    model: FittedModel,
    bootstrap: BootstrapResult | None = None,
    n_points: int = 200,
    ci_levels: list[float] | None = None,
) -> plt.Figure:
    """Plot learned interaction kernels with optional bootstrap confidence bands.

    Args:
        model: Fitted model
        bootstrap: Optional bootstrap result for confidence bands
        n_points: Number of evaluation points
        ci_levels: Confidence levels for bands, e.g. [0.05, 0.01]
    """
    if ci_levels is None:
        ci_levels = [0.05]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, kernel_name, basis, title in [
        (axes[0], "xx", model.basis_xx, r"$f_{xx}(r)$: chromosome-chromosome"),
        (axes[1], "xy", model.basis_xy, r"$f_{xy}(r)$: centrosome $\to$ chromosome"),
    ]:
        r = np.linspace(basis.r_min, basis.r_max, n_points)
        f = model.evaluate_kernel(kernel_name, r)

        if bootstrap is not None:
            # Evaluate all bootstrap samples
            phi = basis.evaluate(r)
            if kernel_name == "xx":
                samples = phi @ bootstrap.theta_samples[:, :model.n_basis_xx].T
            else:
                samples = phi @ bootstrap.theta_samples[:, model.n_basis_xx:].T

            for level in ci_levels:
                lo = np.quantile(samples, level, axis=1)
                hi = np.quantile(samples, 1 - level, axis=1)
                ax.fill_between(r, lo, hi, alpha=0.2, color="C0")

        ax.plot(r, f, linewidth=2, color="C0")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Distance (μm)")
        ax.set_ylabel("Force (μm/s)")
        ax.set_title(title)

    fig.tight_layout()
    return fig


def plot_cv_curve(cv_results: dict[str, CVResult]) -> plt.Figure:
    """Plot cross-validation error vs configuration.

    Args:
        cv_results: Dict mapping label -> CVResult
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(cv_results.keys())
    means = [cv_results[k].mean_error for k in labels]
    stds = [cv_results[k].std_error for k in labels]
    x = range(len(labels))
    ax.errorbar(x, means, yerr=stds, fmt="o-", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean squared error")
    ax.set_title("Cross-validation")
    fig.tight_layout()
    return fig


def plot_residuals(residuals: np.ndarray) -> plt.Figure:
    """Plot residual diagnostics: histogram and QQ plot."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(residuals, bins=50, density=True, alpha=0.7)
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Residual distribution")

    from scipy import stats
    stats.probplot(residuals, plot=axes[1])
    axes[1].set_title("Q-Q plot")

    fig.tight_layout()
    return fig


def plot_recovery(
    r: np.ndarray,
    true_values: np.ndarray,
    fitted_values: np.ndarray,
    kernel_name: str = "",
) -> plt.Figure:
    """Plot true vs recovered kernel on synthetic data."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r, true_values, "k--", linewidth=2, label="True")
    ax.plot(r, fitted_values, "C0-", linewidth=2, label="Fitted")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Distance (μm)")
    ax.set_ylabel("Force")
    ax.set_title(f"Kernel recovery: {kernel_name}")
    ax.legend()
    fig.tight_layout()
    return fig
```

- [ ] **Step 2: Commit**

```bash
git add chromlearn/model_fitting/plotting.py
git commit -m "feat: add plotting functions for kernels, CV, residuals, and recovery"
```

---

## Task 12: FitConfig Dataclass

**Files:**
- Modify: `chromlearn/model_fitting/__init__.py`

- [ ] **Step 1: Add FitConfig to the model_fitting package init**

```python
# chromlearn/model_fitting/__init__.py
from dataclasses import dataclass


@dataclass
class FitConfig:
    """Configuration for the kernel fitting pipeline."""
    # Time window
    endpoint_method: str = "midpoint_neb_ao"
    # Basis
    n_basis_xx: int = 10
    n_basis_xy: int = 10
    r_min_xx: float = 0.5
    r_max_xx: float = 10.0
    r_min_xy: float = 0.5
    r_max_xy: float = 12.0
    basis_type: str = "bspline"
    # Regularization
    lambda_ridge: float = 1e-3
    lambda_rough: float = 1e-3
    # Data
    dt: float = 5.0
    d: int = 3
```

- [ ] **Step 2: Commit**

```bash
git add chromlearn/model_fitting/__init__.py
git commit -m "feat: add FitConfig dataclass"
```

---

## Task 13: Lag Correlation Analysis (`chromlearn/analysis/lag_correlation.py`)

**Files:**
- Create: `chromlearn/analysis/lag_correlation.py`, `tests/test_lag_correlation.py`

**Context:** This is a port of `old_code/suppfig2_corr.m` and `old_code/dot_autocov.m`. It computes the lagged velocity dot-product autocorrelation between centrosome-center motion and chromosome-center (center of mass of all chromosomes) motion. This is the key evidence that chromosomes "follow" centrosomes.

Algorithm (per cell):
1. Compute pole center = midpoint of two centrosomes at each timepoint (T, 3)
2. Compute chromosome center = mean position of all chromosomes at each timepoint (T, 3)
3. Smooth both with Savitzky-Golay filter (window=31, polyorder=3)
4. Compute velocities: diff of smoothed positions → (T-1, 3)
5. For each lag L in [-lag_max, ..., lag_max]:
   - For valid indices i: compute dot product of unit velocity vectors:
     `dot(pole_vel[i] / |pole_vel[i]|, chrom_vel[i+L] / |chrom_vel[i+L]|)`
   - Average over all valid i → gives autocorrelation at lag L

Aggregate across cells: median and std of per-cell autocorrelation curves.

- [ ] **Step 1: Write tests**

```python
# tests/test_lag_correlation.py
import numpy as np
import pytest
from chromlearn.io.loader import CellData
from chromlearn.analysis.lag_correlation import (
    compute_lag_correlation_single,
    compute_lag_correlation,
    LagResult,
)


def make_correlated_cell(T=200, N=10, lag_offset=3):
    """Cell where chromosome center follows pole center with a lag."""
    rng = np.random.default_rng(42)
    # Random walk for pole center
    pole_vel = rng.standard_normal((T - 1, 3)) * 0.1
    pole_center = np.cumsum(np.vstack([np.zeros((1, 3)), pole_vel]), axis=0)

    # Centrosomes symmetric around pole center
    centrioles = np.zeros((T, 3, 2))
    centrioles[:, 0, 0] = pole_center[:, 0] - 5
    centrioles[:, 0, 1] = pole_center[:, 0] + 5
    centrioles[:, 1:, 0] = pole_center[:, 1:]
    centrioles[:, 1:, 1] = pole_center[:, 1:]

    # Chromosome center follows pole center with a lag
    chrom_center = np.zeros((T, 3))
    for t in range(T):
        t_src = max(0, t - lag_offset)
        chrom_center[t] = pole_center[t_src] + rng.normal(0, 0.05, 3)

    chromosomes = np.zeros((T, 3, N))
    for j in range(N):
        chromosomes[:, :, j] = chrom_center + rng.normal(0, 0.5, (T, 3))

    return CellData(
        cell_id="test_corr_001", condition="test",
        centrioles=centrioles, chromosomes=chromosomes,
        neb=1, ao1=T - 10, ao2=T - 8, tracked=N,
    )


def test_lag_correlation_single_shape():
    cell = make_correlated_cell()
    lags, ac = compute_lag_correlation_single(cell, lag_max=20, smooth_window=31)
    assert len(lags) == 41  # -20 to 20
    assert len(ac) == 41


def test_lag_correlation_peak_near_positive_lag():
    """If chromosomes follow with a lag, peak should be at a positive lag."""
    cell = make_correlated_cell(lag_offset=5)
    lags, ac = compute_lag_correlation_single(cell, lag_max=20, smooth_window=31)
    peak_lag = lags[np.argmax(ac)]
    # Peak should be at a positive lag (chromosomes lag behind centrosomes)
    assert peak_lag >= 0


def test_compute_lag_correlation_aggregated():
    cells = [make_correlated_cell(lag_offset=3) for _ in range(5)]
    result = compute_lag_correlation(cells, lag_max=20, smooth_window=31)
    assert isinstance(result, LagResult)
    assert len(result.lags) == 41
    assert result.median.shape == (41,)
    assert result.std.shape == (41,)
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement lag_correlation.py**

```python
# chromlearn/analysis/lag_correlation.py
from dataclasses import dataclass
import numpy as np
from scipy.signal import savgol_filter
from chromlearn.io.loader import CellData
import matplotlib.pyplot as plt


@dataclass
class LagResult:
    """Result of lag correlation analysis across cells."""
    lags: np.ndarray          # lag values in seconds (1D)
    per_cell: np.ndarray      # (n_cells, n_lags) autocorrelation per cell
    median: np.ndarray        # (n_lags,) median across cells
    std: np.ndarray           # (n_lags,) std across cells


def compute_lag_correlation_single(
    cell: CellData,
    lag_max: int = 101,
    smooth_window: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lagged velocity dot-product correlation for one cell.

    Computes the normalized dot product between centrosome-center velocity
    and chromosome-center velocity at various lags.

    Args:
        cell: CellData object
        lag_max: Maximum lag in frames (positive and negative)
        smooth_window: Savitzky-Golay smoothing window (must be odd)

    Returns:
        lags: 1D array of lag values in frames, from -lag_max to +lag_max
        ac: 1D array of autocorrelation values at each lag
    """
    T = cell.centrioles.shape[0]
    N = cell.tracked

    # Pole center
    pc = 0.5 * (cell.centrioles[:, :, 0] + cell.centrioles[:, :, 1])  # (T, 3)

    # Chromosome center (nanmean over chromosomes)
    chrom_cent = np.nanmean(cell.chromosomes, axis=2)  # (T, 3)

    # Smooth
    if smooth_window > T:
        smooth_window = T if T % 2 == 1 else T - 1
    pc_smooth = savgol_filter(pc, window_length=smooth_window, polyorder=3, axis=0)
    cc_smooth = savgol_filter(chrom_cent, window_length=smooth_window, polyorder=3, axis=0)

    # Velocities
    pc_vel = np.diff(pc_smooth, axis=0)  # (T-1, 3)
    cc_vel = np.diff(cc_smooth, axis=0)  # (T-1, 3)

    n = len(pc_vel)
    lags_arr = np.arange(-lag_max, lag_max + 1)
    ac = np.zeros(len(lags_arr))

    for ll, lag in enumerate(lags_arr):
        dots = []
        if lag < 0:
            idx_range = range(-lag, n)
        else:
            idx_range = range(0, n - lag)

        for i in idx_range:
            x = pc_vel[i]
            y = cc_vel[i + lag]
            nx = np.linalg.norm(x)
            ny = np.linalg.norm(y)
            if nx < 1e-12 or ny < 1e-12:
                continue
            dots.append(np.dot(x, y) / (nx * ny))

        if dots:
            ac[ll] = np.nanmean(dots)
        else:
            ac[ll] = np.nan

    return lags_arr, ac


def compute_lag_correlation(
    cells: list[CellData],
    lag_max: int = 101,
    smooth_window: int = 31,
) -> LagResult:
    """Compute lag correlation across multiple cells.

    Args:
        cells: List of CellData objects
        lag_max: Maximum lag in frames
        smooth_window: Smoothing window for Savitzky-Golay filter

    Returns:
        LagResult with per-cell and aggregated autocorrelations
    """
    all_ac = []
    lags = None

    for cell in cells:
        l, ac = compute_lag_correlation_single(cell, lag_max, smooth_window)
        all_ac.append(ac)
        lags = l

    per_cell = np.array(all_ac)  # (n_cells, n_lags)
    lags_seconds = lags * cells[0].dt  # convert to seconds

    return LagResult(
        lags=lags_seconds,
        per_cell=per_cell,
        median=np.nanmedian(per_cell, axis=0),
        std=np.nanstd(per_cell, axis=0),
    )


def plot_lag_correlation(result: LagResult) -> plt.Figure:
    """Plot lag correlation with individual cell traces and median.

    Args:
        result: LagResult from compute_lag_correlation

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Individual cell traces
    for i in range(result.per_cell.shape[0]):
        ax.plot(result.lags, result.per_cell[i], linewidth=0.75, alpha=0.5)

    # Median
    ax.plot(result.lags, result.median, "k", linewidth=3, label="Median")

    # Confidence band
    ax.fill_between(
        result.lags,
        result.median - result.std,
        result.median + result.std,
        alpha=0.2, color="gray",
    )

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Lag (seconds)")
    ax.set_ylabel("Normalized velocity dot product")
    ax.set_title("Centrosome–chromosome velocity correlation")
    ax.set_xlim(-200, 200)
    ax.set_ylim(-0.25, 1)
    ax.legend()
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_lag_correlation.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chromlearn/analysis/lag_correlation.py tests/test_lag_correlation.py
git commit -m "feat: add lag correlation analysis (centrosome-chromosome velocity following)"
```

---

## Task 14: Trajectory Visualization (`chromlearn/analysis/trajectory_viz.py`)

**Files:**
- Create: `chromlearn/analysis/trajectory_viz.py`

**Context:** Visualization of single-cell chromosome trajectories, optionally in spindle-frame coordinates. Port of `old_code/suppfig21_example_centerofmass.m`.

- [ ] **Step 1: Implement trajectory_viz.py**

```python
# chromlearn/analysis/trajectory_viz.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from chromlearn.io.loader import CellData
from chromlearn.io.trajectory import (
    TrimmedCell, trim_trajectory, spindle_frame, pole_center, pole_pole_distance,
)


def plot_cell_trajectories(
    cell: CellData | TrimmedCell,
    frame: str = "lab",
    method: str = "midpoint_neb_ao",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot chromosome trajectories for a single cell, color-coded by time.

    Args:
        cell: CellData or TrimmedCell
        frame: "lab" for raw 3D (projected to xy), "spindle" for spindle-frame
        method: Endpoint method if cell is CellData (needs trimming)
        ax: Optional matplotlib axes

    Returns:
        matplotlib Figure
    """
    if isinstance(cell, CellData):
        trimmed = trim_trajectory(cell, method=method)
    else:
        trimmed = cell

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    T = trimmed.chromosomes.shape[0]
    N = trimmed.tracked
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, T))

    if frame == "spindle":
        sf = spindle_frame(trimmed)
        for j in range(N):
            for t in range(T - 1):
                ax.plot(
                    [sf.axial[t, j], sf.axial[t + 1, j]],
                    [sf.radial[t, j], sf.radial[t + 1, j]],
                    color=colors[t], linewidth=0.5,
                )
        ax.set_xlabel("Axial distance (μm)")
        ax.set_ylabel("Radial distance (μm)")
    else:
        # Lab frame, project to xy
        for j in range(N):
            for t in range(T - 1):
                ax.plot(
                    [trimmed.chromosomes[t, 0, j], trimmed.chromosomes[t + 1, 0, j]],
                    [trimmed.chromosomes[t, 1, j], trimmed.chromosomes[t + 1, 1, j]],
                    color=colors[t], linewidth=0.5,
                )
        # Plot centrosomes
        ax.plot(trimmed.centrioles[:, 0, 0], trimmed.centrioles[:, 1, 0],
                "r-", linewidth=2, label="Pole 1")
        ax.plot(trimmed.centrioles[:, 0, 1], trimmed.centrioles[:, 1, 1],
                "b-", linewidth=2, label="Pole 2")
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("y (μm)")
        ax.legend()

    ax.set_title(f"{trimmed.cell_id}")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_chromosome_cloud(
    cell: CellData | TrimmedCell,
    timepoint: int = 0,
    method: str = "midpoint_neb_ao",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot chromosome positions as a point cloud at a single timepoint.

    Args:
        cell: CellData or TrimmedCell
        timepoint: Index within trimmed trajectory
        method: Endpoint method if cell is CellData
        ax: Optional axes

    Returns:
        matplotlib Figure
    """
    if isinstance(cell, CellData):
        trimmed = trim_trajectory(cell, method=method)
    else:
        trimmed = cell

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    chrom = trimmed.chromosomes[timepoint]  # (3, N)
    centr = trimmed.centrioles[timepoint]    # (3, 2)

    ax.scatter(chrom[0], chrom[1], c="gray", s=30, alpha=0.7, label="Chromosomes")
    ax.scatter(centr[0, 0], centr[1, 0], c="red", s=100, marker="^", label="Pole 1")
    ax.scatter(centr[0, 1], centr[1, 1], c="blue", s=100, marker="^", label="Pole 2")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.set_title(f"{trimmed.cell_id} — t={timepoint}")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    return fig
```

- [ ] **Step 2: Commit**

```bash
git add chromlearn/analysis/trajectory_viz.py
git commit -m "feat: add trajectory visualization in lab and spindle frames"
```

---

## Task 15: Notebook 01 — Explore Data

**Files:**
- Create: `notebooks/01_explore_data.ipynb`

**Context:** This notebook validates the data loading pipeline. Another LLM should create it as a Jupyter notebook (`.ipynb` format) with the following cells. If creating `.ipynb` is difficult, create a `.py` file with `# %%` cell markers (percent format) instead — it can be converted later.

- [ ] **Step 1: Create the notebook**

The notebook should contain these sections:

**Cell 1: Imports and setup**
```python
import sys
sys.path.insert(0, "..")  # add repo root to path

import numpy as np
import matplotlib.pyplot as plt
from chromlearn.io.loader import load_cell
from chromlearn.io.catalog import list_cells, load_condition
from chromlearn.io.trajectory import (
    trim_trajectory, pole_pole_distance, pole_center, spindle_frame,
)
from chromlearn.analysis.trajectory_viz import plot_cell_trajectories, plot_chromosome_cloud

%matplotlib inline
plt.rcParams["figure.dpi"] = 100
```

**Cell 2: List available cells**
```python
cells_ctr = list_cells("rpe18_ctr")
print(f"Found {len(cells_ctr)} rpe18_ctr cells:")
for c in cells_ctr:
    print(f"  {c}")
```

**Cell 3: Load one cell and inspect**
```python
from pathlib import Path
DATA_DIR = Path("../data")
cell = load_cell(DATA_DIR / f"{cells_ctr[0]}.mat")
print(f"Cell ID: {cell.cell_id}")
print(f"Condition: {cell.condition}")
print(f"Centrioles shape: {cell.centrioles.shape}")
print(f"Chromosomes shape: {cell.chromosomes.shape}")
print(f"NEB frame: {cell.neb} (1-based)")
print(f"AO1: {cell.ao1}, AO2: {cell.ao2}")
print(f"Tracked chromosomes: {cell.tracked}")
print(f"NaN count in chromosomes: {np.isnan(cell.chromosomes).sum()}")
```

**Cell 4: Pole-pole distance over time**
```python
ppd = pole_pole_distance(cell)
fig, ax = plt.subplots(figsize=(8, 3))
time = np.arange(len(ppd)) * cell.dt
ax.plot(time, ppd)
ax.axvline((cell.neb - 1) * cell.dt, color="r", linestyle="--", label="NEB")
ao_mean = (cell.ao1 + cell.ao2) / 2
ax.axvline((ao_mean - 1) * cell.dt, color="g", linestyle="--", label="AO (mean)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Pole-pole distance (μm)")
ax.legend()
ax.set_title(cell.cell_id)
plt.show()
```

**Cell 5: Trim and visualize trajectories**
```python
trimmed = trim_trajectory(cell, method="midpoint_neb_ao")
print(f"Trimmed: frames {trimmed.start_frame}-{trimmed.end_frame} "
      f"({trimmed.chromosomes.shape[0]} timepoints)")

fig = plot_cell_trajectories(trimmed, frame="lab")
plt.show()

fig = plot_cell_trajectories(trimmed, frame="spindle")
plt.show()
```

**Cell 6: Chromosome cloud at NEB**
```python
fig = plot_chromosome_cloud(trimmed, timepoint=0)
plt.show()
```

**Cell 7: Load all rpe18_ctr and summarize**
```python
all_cells = load_condition("rpe18_ctr")
print(f"Loaded {len(all_cells)} cells")
for c in all_cells:
    trimmed = trim_trajectory(c, method="midpoint_neb_ao")
    ppd = pole_pole_distance(c)
    print(f"  {c.cell_id}: {c.tracked} chromosomes, "
          f"{trimmed.chromosomes.shape[0]} trimmed timepoints, "
          f"max pole-pole dist = {ppd.max():.1f} μm")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/01_explore_data.ipynb  # or .py
git commit -m "feat: add data exploration notebook"
```

---

## Task 16: Notebook 02 — Lag Correlation

**Files:**
- Create: `notebooks/02_lag_correlation.ipynb`

- [ ] **Step 1: Create the notebook**

**Cell 1: Imports**
```python
import sys
sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt
from chromlearn.io.catalog import load_condition
from chromlearn.analysis.lag_correlation import compute_lag_correlation, plot_lag_correlation
%matplotlib inline
```

**Cell 2: Load data and compute**
```python
cells = load_condition("rpe18_ctr")
result = compute_lag_correlation(cells, lag_max=40, smooth_window=31)
```

**Cell 3: Plot**
```python
fig = plot_lag_correlation(result)
plt.show()
```

**Cell 4: Interpretation**
```python
# The peak at positive lag shows that chromosomes follow centrosomes.
# This justifies treating centrosomes as external/given in our model.
peak_idx = np.argmax(result.median)
print(f"Peak correlation at lag = {result.lags[peak_idx]:.0f} seconds")
print(f"Peak value = {result.median[peak_idx]:.3f}")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/02_lag_correlation.ipynb
git commit -m "feat: add lag correlation notebook"
```

---

## Task 17: Notebook 03 — Synthetic Validation

**Files:**
- Create: `notebooks/03_synthetic_validation.ipynb`

**Context:** This is the mandatory validation step. Generate synthetic data with known kernels, fit the model, and verify recovery. This must pass before fitting real data.

- [ ] **Step 1: Create the notebook**

**Cell 1: Imports**
```python
import sys
sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt
from chromlearn.model_fitting.simulate import generate_synthetic_data, add_localization_noise
from chromlearn.model_fitting.basis import BSplineBasis
from chromlearn.model_fitting.features import build_design_matrix
from chromlearn.model_fitting.fit import fit_kernels, estimate_diffusion
from chromlearn.model_fitting.model import FittedModel
from chromlearn.model_fitting.validate import kernel_recovery_error, residual_diagnostics
from chromlearn.model_fitting.plotting import plot_recovery, plot_residuals
from chromlearn.io.trajectory import TrimmedCell
%matplotlib inline
```

**Cell 2: Define true kernels**
```python
# True kernels:
# f_xx: weak short-range repulsion (negative = repulsive since force points away)
def true_xx(r):
    return -0.02 * np.exp(-r / 1.5)

# f_xy: attraction toward centrosomes (positive = attractive)
def true_xy(r):
    return 0.05 * np.exp(-r / 3.0)

D_true = 0.05
dt = 5.0
```

**Cell 3: Generate synthetic data**
```python
rng = np.random.default_rng(42)
synth = generate_synthetic_data(
    kernel_xx=true_xx, kernel_xy=true_xy,
    n_chromosomes=30, n_steps=100, dt=dt, D_x=D_true,
    pole_separation=10.0, rng=rng,
)
print(f"Chromosomes shape: {synth.chromosomes.shape}")
print(f"Centrosomes shape: {synth.centrosomes.shape}")
```

**Cell 4: Convert synthetic data to TrimmedCell format**
```python
# Wrap synthetic data as a TrimmedCell so build_design_matrix can use it
synth_cell = TrimmedCell(
    cell_id="synthetic_001", condition="synthetic",
    centrioles=synth.centrosomes, chromosomes=synth.chromosomes,
    tracked=synth.chromosomes.shape[2], dt=dt,
    start_frame=0, end_frame=synth.chromosomes.shape[0] - 1,
)
```

**Cell 5: Set up basis and fit**
```python
basis_xx = BSplineBasis(r_min=0.5, r_max=10.0, n_basis=10)
basis_xy = BSplineBasis(r_min=0.5, r_max=12.0, n_basis=10)

G, V = build_design_matrix([synth_cell], basis_xx, basis_xy)
print(f"Design matrix: {G.shape}, Response: {V.shape}")

from scipy.linalg import block_diag
R = block_diag(basis_xx.roughness_matrix(), basis_xy.roughness_matrix())

result = fit_kernels(G, V, lambda_ridge=1e-4, lambda_rough=1e-4, R=R)
D_hat = estimate_diffusion(V, G, result.theta, dt=dt, d=3)
print(f"True D = {D_true}, Estimated D = {D_hat:.4f}")
```

**Cell 6: Compare recovered vs true kernels**
```python
model = FittedModel(
    theta=result.theta,
    n_basis_xx=basis_xx.n_basis, n_basis_xy=basis_xy.n_basis,
    basis_xx=basis_xx, basis_xy=basis_xy,
    D_x=D_hat, dt=dt,
)

r_xx = np.linspace(0.5, 10.0, 200)
r_xy = np.linspace(0.5, 12.0, 200)

fig = plot_recovery(r_xx, true_xx(r_xx), model.evaluate_kernel("xx", r_xx), "f_xx")
plt.show()

fig = plot_recovery(r_xy, true_xy(r_xy), model.evaluate_kernel("xy", r_xy), "f_xy")
plt.show()

# Quantitative error
err_xx = kernel_recovery_error(r_xx, true_xx(r_xx), model.evaluate_kernel("xx", r_xx))
err_xy = kernel_recovery_error(r_xy, true_xy(r_xy), model.evaluate_kernel("xy", r_xy))
print(f"RMS recovery error: f_xx = {err_xx:.4f}, f_xy = {err_xy:.4f}")
```

**Cell 7: Residual diagnostics**
```python
diag = residual_diagnostics(result.residuals)
print("Residual diagnostics:")
for k, v in diag.items():
    print(f"  {k}: {v:.4f}")

fig = plot_residuals(result.residuals)
plt.show()
```

**Cell 8: Test with localization noise**
```python
# Add localization noise and re-fit
noisy_chrom = add_localization_noise(synth.chromosomes, sigma=0.1, rng=rng)
noisy_cell = TrimmedCell(
    cell_id="synthetic_noisy", condition="synthetic",
    centrioles=synth.centrosomes, chromosomes=noisy_chrom,
    tracked=noisy_chrom.shape[2], dt=dt,
    start_frame=0, end_frame=noisy_chrom.shape[0] - 1,
)

G_noisy, V_noisy = build_design_matrix([noisy_cell], basis_xx, basis_xy)
result_noisy = fit_kernels(G_noisy, V_noisy, lambda_ridge=1e-4, lambda_rough=1e-3, R=R)

model_noisy = FittedModel(
    theta=result_noisy.theta,
    n_basis_xx=basis_xx.n_basis, n_basis_xy=basis_xy.n_basis,
    basis_xx=basis_xx, basis_xy=basis_xy,
    D_x=estimate_diffusion(V_noisy, G_noisy, result_noisy.theta, dt=dt), dt=dt,
)

fig = plot_recovery(r_xx, true_xx(r_xx), model_noisy.evaluate_kernel("xx", r_xx), "f_xx (noisy)")
plt.show()

fig = plot_recovery(r_xy, true_xy(r_xy), model_noisy.evaluate_kernel("xy", r_xy), "f_xy (noisy)")
plt.show()
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/03_synthetic_validation.ipynb
git commit -m "feat: add synthetic validation notebook"
```

---

## Task 18: Notebook 04 — Fit Kernels on Real Data

**Files:**
- Create: `notebooks/04_fit_kernels.ipynb`

- [ ] **Step 1: Create the notebook**

**Cell 1: Imports**
```python
import sys
sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt
from chromlearn.io.catalog import load_condition
from chromlearn.io.trajectory import trim_trajectory
from chromlearn.model_fitting.basis import BSplineBasis
from chromlearn.model_fitting.features import build_design_matrix
from chromlearn.model_fitting.fit import fit_kernels, estimate_diffusion, bootstrap_kernels
from chromlearn.model_fitting.model import FittedModel
from chromlearn.model_fitting.validate import residual_diagnostics
from chromlearn.model_fitting.plotting import plot_kernels, plot_residuals
from chromlearn.model_fitting import FitConfig
from scipy.linalg import block_diag
%matplotlib inline
```

**Cell 2: Load and trim data**
```python
config = FitConfig()
cells_raw = load_condition("rpe18_ctr")
cells = [trim_trajectory(c, method=config.endpoint_method) for c in cells_raw]
print(f"Loaded {len(cells)} cells")
for c in cells:
    print(f"  {c.cell_id}: {c.tracked} chromosomes, {c.chromosomes.shape[0]} timepoints")
```

**Cell 3: Explore distance distributions (to set r_min/r_max)**
```python
# Collect pairwise distances to inform basis domain
chr_chr_dists = []
chr_cen_dists = []
for c in cells:
    T, _, N = c.chromosomes.shape
    for t in range(0, T, 5):  # subsample
        for i in range(N):
            pos_i = c.chromosomes[t, :, i]
            if np.any(np.isnan(pos_i)):
                continue
            for k in range(i+1, N):
                pos_k = c.chromosomes[t, :, k]
                if np.any(np.isnan(pos_k)):
                    continue
                chr_chr_dists.append(np.linalg.norm(pos_k - pos_i))
            for j in range(2):
                chr_cen_dists.append(np.linalg.norm(c.centrioles[t, :, j] - pos_i))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(chr_chr_dists, bins=80, density=True)
axes[0].set_title("Chromosome-chromosome distances")
axes[0].set_xlabel("Distance (μm)")
axes[1].hist(chr_cen_dists, bins=80, density=True)
axes[1].set_title("Chromosome-centrosome distances")
axes[1].set_xlabel("Distance (μm)")
plt.tight_layout()
plt.show()

# Print percentiles to guide r_min/r_max
p = [5, 25, 50, 75, 95]
print("Chr-chr percentiles:", {pi: f"{np.percentile(chr_chr_dists, pi):.1f}" for pi in p})
print("Chr-cen percentiles:", {pi: f"{np.percentile(chr_cen_dists, pi):.1f}" for pi in p})
```

**Cell 4: Set up basis and build design matrix**
```python
# Set r_min/r_max from distance distributions above
# (adjust these after seeing the histograms)
basis_xx = BSplineBasis(r_min=config.r_min_xx, r_max=config.r_max_xx, n_basis=config.n_basis_xx)
basis_xy = BSplineBasis(r_min=config.r_min_xy, r_max=config.r_max_xy, n_basis=config.n_basis_xy)

G, V = build_design_matrix(cells, basis_xx, basis_xy)
print(f"Design matrix: {G.shape}")
print(f"Response vector: {V.shape}")
```

**Cell 5: Fit**
```python
R = block_diag(basis_xx.roughness_matrix(), basis_xy.roughness_matrix())
result = fit_kernels(G, V, config.lambda_ridge, config.lambda_rough, R)
D_hat = estimate_diffusion(V, G, result.theta, dt=config.dt, d=config.d)
print(f"Estimated diffusion D_x = {D_hat:.4f} μm²/s")
```

**Cell 6: Bootstrap for confidence bands**
```python
boot = bootstrap_kernels(
    cells, basis_xx, basis_xy,
    n_boot=250,
    lambda_ridge=config.lambda_ridge,
    lambda_rough=config.lambda_rough,
)
```

**Cell 7: Build model and plot**
```python
model = FittedModel(
    theta=result.theta,
    n_basis_xx=basis_xx.n_basis, n_basis_xy=basis_xy.n_basis,
    basis_xx=basis_xx, basis_xy=basis_xy,
    D_x=D_hat, dt=config.dt,
    metadata={"condition": "rpe18_ctr", "n_cells": len(cells)},
)

fig = plot_kernels(model, bootstrap=boot, ci_levels=[0.05, 0.01])
plt.show()
```

**Cell 8: Residual diagnostics**
```python
diag = residual_diagnostics(result.residuals)
print("Residual diagnostics:")
for k, v in diag.items():
    print(f"  {k}: {v:.4f}")

fig = plot_residuals(result.residuals)
plt.show()
```

**Cell 9: Save model**
```python
model.save("../results/rpe18_ctr_model.npz")
print("Model saved.")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/04_fit_kernels.ipynb
git commit -m "feat: add real-data kernel fitting notebook"
```

---

## Task 19: Notebook 05 — Model Selection

**Files:**
- Create: `notebooks/05_model_selection.ipynb`

- [ ] **Step 1: Create the notebook**

**Cell 1: Imports and data loading** (same as notebook 04)

**Cell 2: Cross-validation over basis sizes**
```python
from chromlearn.model_fitting.fit import cross_validate
from chromlearn.model_fitting.plotting import plot_cv_curve

basis_sizes = [4, 6, 8, 10, 12, 15, 18]
cv_results = {}

for n in basis_sizes:
    print(f"Testing n_basis = {n}...")
    bxx = BSplineBasis(r_min=config.r_min_xx, r_max=config.r_max_xx, n_basis=n)
    bxy = BSplineBasis(r_min=config.r_min_xy, r_max=config.r_max_xy, n_basis=n)
    cv = cross_validate(cells, bxx, bxy,
                        lambda_ridge=config.lambda_ridge,
                        lambda_rough=config.lambda_rough)
    cv_results[str(n)] = cv
    print(f"  MSE = {cv.mean_error:.6f} ± {cv.std_error:.6f}")

fig = plot_cv_curve(cv_results)
plt.show()
```

**Cell 3: Cross-validation over regularization strength**
```python
lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
cv_lambda_results = {}

for lam in lambdas:
    print(f"Testing lambda = {lam}...")
    cv = cross_validate(cells, basis_xx, basis_xy,
                        lambda_ridge=lam, lambda_rough=lam)
    cv_lambda_results[f"{lam:.0e}"] = cv
    print(f"  MSE = {cv.mean_error:.6f} ± {cv.std_error:.6f}")

fig = plot_cv_curve(cv_lambda_results)
plt.xlabel("Lambda")
plt.show()
```

**Cell 4: Compare endpoint methods**
```python
for method in ["midpoint_neb_ao", "ao_mean", "end_sep"]:
    cells_m = [trim_trajectory(c, method=method) for c in cells_raw]
    G_m, V_m = build_design_matrix(cells_m, basis_xx, basis_xy)
    res_m = fit_kernels(G_m, V_m, config.lambda_ridge, config.lambda_rough, R)
    model_m = FittedModel(
        theta=res_m.theta, n_basis_xx=basis_xx.n_basis, n_basis_xy=basis_xy.n_basis,
        basis_xx=basis_xx, basis_xy=basis_xy,
        D_x=estimate_diffusion(V_m, G_m, res_m.theta, dt=config.dt), dt=config.dt,
    )
    r_plot = np.linspace(config.r_min_xy, config.r_max_xy, 200)
    plt.plot(r_plot, model_m.evaluate_kernel("xy", r_plot), label=method)

plt.xlabel("Distance (μm)")
plt.ylabel("f_xy")
plt.title("Centrosome kernel: effect of endpoint method")
plt.legend()
plt.show()
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/05_model_selection.ipynb
git commit -m "feat: add model selection notebook (basis size, lambda, endpoint)"
```

---

## Task 20: Notebook 06 — Forward Simulation

**Files:**
- Create: `notebooks/06_forward_simulation.ipynb`

- [ ] **Step 1: Create the notebook**

**Cell 1: Imports and load fitted model**
```python
import sys
sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt
from chromlearn.model_fitting.model import FittedModel
from chromlearn.model_fitting.simulate import simulate_trajectories
from chromlearn.model_fitting.validate import summary_statistics
from chromlearn.io.catalog import load_condition
from chromlearn.io.trajectory import trim_trajectory
%matplotlib inline

model = FittedModel.load("../results/rpe18_ctr_model.npz")
```

**Cell 2: Simulate using learned kernels and real centrosome trajectories**
```python
cells_raw = load_condition("rpe18_ctr")
# Pick a representative cell
cell = trim_trajectory(cells_raw[0], method="midpoint_neb_ao")

rng = np.random.default_rng(42)
# Use real centrosome trajectories
sim_traj = simulate_trajectories(
    kernel_xx=lambda r: model.evaluate_kernel("xx", r),
    kernel_xy=lambda r: model.evaluate_kernel("xy", r),
    centrosome_positions=cell.centrioles,
    x0=cell.chromosomes[0].T,  # start from real initial positions
    n_steps=cell.chromosomes.shape[0] - 1,
    dt=model.dt, D_x=model.D_x, rng=rng,
)
print(f"Simulated trajectory shape: {sim_traj.shape}")
```

**Cell 3: Compare summary statistics**
```python
real_stats = summary_statistics(cell.chromosomes, cell.centrioles)
sim_stats = summary_statistics(sim_traj, cell.centrioles)

print(f"{'Statistic':<30} {'Real':>10} {'Simulated':>10}")
print("-" * 52)
for key in real_stats:
    print(f"{key:<30} {real_stats[key]:>10.3f} {sim_stats[key]:>10.3f}")
```

**Cell 4: Visual comparison — distance from center**
```python
from chromlearn.io.trajectory import pole_center
pc = pole_center(cell)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, traj, title in [(axes[0], cell.chromosomes, "Real"), (axes[1], sim_traj, "Simulated")]:
    T, _, N = traj.shape
    for j in range(N):
        d = np.linalg.norm(traj[:, :, j] - pc, axis=1)
        ax.plot(np.arange(T) * model.dt, d, linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance from center (μm)")
    ax.set_title(title)
plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/06_forward_simulation.ipynb
git commit -m "feat: add forward simulation comparison notebook"
```

---

## Task 21: Final Integration Test

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: End-to-end smoke test**

Run notebook 03 (synthetic validation) end to end. The fitted kernels should visually match the true kernels, with RMS recovery error < 0.01 for the noise-free case.

- [ ] **Step 3: Run notebook 01 on real data**

Load all rpe18_ctr cells, trim, and visualize. Verify no crashes and plots look reasonable.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: chromlearn v0.1 — complete pipeline for pairwise kernel learning"
```

---

## Summary

| Task | Component | Key Files |
|------|-----------|-----------|
| 1 | Scaffolding | `__init__.py` files, `requirements.txt` |
| 2 | Data loader | `chromlearn/io/loader.py` |
| 3 | Trajectory processing | `chromlearn/io/trajectory.py` |
| 4 | Cell catalog | `chromlearn/io/catalog.py` |
| 5 | Basis functions | `chromlearn/model_fitting/basis.py` |
| 6 | Design matrix | `chromlearn/model_fitting/features.py` |
| 7 | Regression/fitting | `chromlearn/model_fitting/fit.py` |
| 8 | Model container | `chromlearn/model_fitting/model.py` |
| 9 | Simulator | `chromlearn/model_fitting/simulate.py` |
| 10 | Validation | `chromlearn/model_fitting/validate.py` |
| 11 | Plotting | `chromlearn/model_fitting/plotting.py` |
| 12 | FitConfig | `chromlearn/model_fitting/__init__.py` |
| 13 | Lag correlation | `chromlearn/analysis/lag_correlation.py` |
| 14 | Trajectory viz | `chromlearn/analysis/trajectory_viz.py` |
| 15-20 | Notebooks | `notebooks/01-06_*.ipynb` |
| 21 | Integration test | Full pipeline validation |

**Dependencies between tasks:**
- Tasks 1-4 (data layer) must be done first and in order
- Tasks 5-12 (model fitting) depend on tasks 1-4 but can be done in the listed order
- Tasks 13-14 (analysis) depend only on tasks 1-4
- Tasks 15-20 (notebooks) depend on all prior tasks
- Task 21 depends on everything
