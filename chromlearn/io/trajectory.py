from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from chromlearn.io.loader import CellData


@dataclass
class TrimmedCell:
    """Cell data restricted to one analysis window."""

    cell_id: str
    condition: str
    centrioles: np.ndarray
    chromosomes: np.ndarray
    tracked: int
    dt: float
    start_frame: int
    end_frame: int


@dataclass
class SpindleFrameData:
    """Chromosome positions in spindle-aligned coordinates."""

    axial: np.ndarray
    radial: np.ndarray


def pole_pole_distance(cell: CellData | TrimmedCell) -> np.ndarray:
    """Euclidean distance between the two centrosome poles at each timepoint.

    Returns:
        1-D array of shape ``(T,)``.
    """
    return np.linalg.norm(cell.centrioles[:, :, 0] - cell.centrioles[:, :, 1], axis=1)


def pole_center(cell: CellData | TrimmedCell) -> np.ndarray:
    """Midpoint of the two centrosome poles at each timepoint.

    Returns:
        Array of shape ``(T, 3)``.
    """
    return 0.5 * (cell.centrioles[:, :, 0] + cell.centrioles[:, :, 1])


VALID_TOPOLOGIES = ("poles", "center", "poles_and_chroms", "center_and_chroms")


def get_partners(cell: CellData | TrimmedCell, topology: str) -> np.ndarray:
    """Construct interaction partner trajectories for a given topology.

    For ``"poles"`` / ``"poles_and_chroms"``: returns both centrosomes as
    separate partners — shape ``(2, T, 3)``.
    For ``"center"`` / ``"center_and_chroms"``: returns the centrosome
    midpoint as a single partner — shape ``(1, T, 3)``.

    The ``_and_chroms`` suffix does not affect the partner array; it signals
    downstream code to include chromosome-chromosome interactions.
    """
    if topology not in VALID_TOPOLOGIES:
        raise ValueError(
            f"Unknown topology '{topology}'. Expected one of {VALID_TOPOLOGIES}."
        )
    if topology in ("poles", "poles_and_chroms"):
        return cell.centrioles.transpose(2, 0, 1)  # (2, T, 3)
    # center or center_and_chroms
    return pole_center(cell)[np.newaxis]  # (1, T, 3)


def _savgol_window(length: int, desired: int) -> int | None:
    if length < 3:
        return None
    window = min(desired, length if length % 2 == 1 else length - 1)
    if window < 3:
        return None
    return window


def compute_end_sep(
    cell: CellData,
    smooth_window: int = 51,
    plateau_frac: float = 0.95,
    avg_halfwin: int = 2,
) -> int:
    """Detect the first frame where spindle separation reaches its metaphase plateau.

    1. Smooth the pole-pole distance with a Savitzky-Golay filter.
    2. Compute the reference maximum over ``[NEB, midpoint(NEB, AO)]`` — the
       core metaphase region — to avoid inflation from late pre-anaphase ramp.
    3. Compute a running average (window = ``2 * avg_halfwin + 1``, default 5
       frames) of the smoothed distance.
    4. Return the first frame where that running average reaches *plateau_frac*
       (default 95%) of the reference maximum.
    """
    neb_frame = cell.neb - 1
    ao_frame = _ao_min_index(cell)
    midpoint_frame = (neb_frame + ao_frame) // 2

    distances = pole_pole_distance(cell)
    if distances.size <= 2:
        return max(0, distances.size - 1)

    window = _savgol_window(distances.size, smooth_window)
    if window is None:
        return distances.size - 1

    smoothed = savgol_filter(
        distances,
        window_length=window,
        polyorder=min(3, window - 1),
    )
    # Reference max from the core metaphase region.
    ref_start = max(0, neb_frame)
    ref_end = min(midpoint_frame + 1, len(smoothed))
    metaphase_max = float(smoothed[ref_start:ref_end].max()) if ref_end > ref_start else float(smoothed.max())
    threshold = plateau_frac * metaphase_max

    # Running average of the smoothed signal.
    kernel_size = 2 * avg_halfwin + 1
    kernel = np.ones(kernel_size) / kernel_size
    running_avg = np.convolve(smoothed, kernel, mode="same")

    candidates = np.flatnonzero(running_avg >= threshold)
    if candidates.size == 0:
        return max(0, len(smoothed) - 1)
    return int(candidates[0])


def _ao_min_index(cell: CellData) -> int:
    return min(cell.ao1, cell.ao2) - 1


def _ao_mean_index(cell: CellData) -> int:
    return int(round((cell.ao1 + cell.ao2) / 2.0)) - 1


def _compute_endpoint(cell: CellData, method: str) -> int:
    neb_index = cell.neb - 1
    if method == "ao_mean":
        return _ao_mean_index(cell)
    if method == "midpoint_neb_ao":
        return (neb_index + _ao_mean_index(cell)) // 2
    if method == "end_sep":
        return compute_end_sep(cell)
    raise ValueError(
        f"Unknown endpoint method '{method}'. "
        "Expected 'midpoint_neb_ao', 'ao_mean', or 'end_sep'."
    )


def trim_trajectory(
    cell: CellData, method: str = "midpoint_neb_ao", min_frames: int = 100
) -> TrimmedCell:
    """Trim cell trajectories to the window ``[NEB, endpoint]``.

    Args:
        cell: Raw cell data (``neb``, ``ao1``, ``ao2`` are 1-based MATLAB indices).
        method: Endpoint strategy -- ``"midpoint_neb_ao"`` (default),
            ``"ao_mean"``, or ``"end_sep"``.
        min_frames: Minimum number of frames required after trimming.

    Returns:
        TrimmedCell with arrays sliced to the chosen time window.

    Raises:
        ValueError: If the trimmed trajectory has fewer than *min_frames* frames.
    """
    start = max(0, cell.neb - 1)
    end = min(cell.centrioles.shape[0] - 1, _compute_endpoint(cell, method))
    if end < start:
        end = start
    n_frames = end - start + 1
    if n_frames < min_frames:
        raise ValueError(
            f"{cell.cell_id}: trimmed trajectory has {n_frames} frames, "
            f"fewer than the minimum {min_frames}."
        )
    window = slice(start, end + 1)
    return TrimmedCell(
        cell_id=cell.cell_id,
        condition=cell.condition,
        centrioles=cell.centrioles[window],
        chromosomes=cell.chromosomes[window],
        tracked=cell.tracked,
        dt=cell.dt,
        start_frame=start,
        end_frame=end,
    )


def spindle_frame(cell: TrimmedCell) -> SpindleFrameData:
    """Project chromosome positions into spindle-aligned cylindrical coordinates.

    At each timepoint the spindle axis runs from pole 1 to pole 2.
    ``axial`` is the signed projection onto that axis relative to the pole
    midpoint; ``radial`` is the perpendicular distance from it.

    Returns:
        SpindleFrameData with ``axial`` and ``radial`` arrays of shape ``(T', N)``.
    """
    center = pole_center(cell)
    axis = cell.centrioles[:, :, 1] - cell.centrioles[:, :, 0]
    norms = np.linalg.norm(axis, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    axis_unit = axis / safe_norms

    delta = np.moveaxis(cell.chromosomes, 2, 1) - center[:, np.newaxis, :]
    axial = np.sum(delta * axis_unit[:, np.newaxis, :], axis=2)
    perpendicular = delta - axial[:, :, np.newaxis] * axis_unit[:, np.newaxis, :]
    radial = np.linalg.norm(perpendicular, axis=2)
    return SpindleFrameData(axial=axial, radial=radial)
