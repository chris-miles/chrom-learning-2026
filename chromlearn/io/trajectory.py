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


def _savgol_window(length: int, desired: int) -> int | None:
    if length < 3:
        return None
    window = min(desired, length if length % 2 == 1 else length - 1)
    if window < 3:
        return None
    return window


def compute_end_sep(cell: CellData, smooth_window: int = 51) -> int:
    """Detect the first frame where spindle separation has largely plateaued."""

    distances = pole_pole_distance(cell)
    if distances.size <= 2:
        return max(0, distances.size - 1)

    end_smooth_window = min(200, distances.size)
    distance_window = _savgol_window(end_smooth_window, smooth_window)
    if distance_window is None:
        return distances.size - 1

    smoothed = savgol_filter(
        distances[:end_smooth_window],
        window_length=distance_window,
        polyorder=min(3, distance_window - 1),
    )
    span = float(smoothed.max() - smoothed.min())
    if span <= 1e-12:
        return distances.size - 1

    normalized = (smoothed - smoothed.min()) / span
    velocity = np.diff(normalized)
    velocity_window = _savgol_window(velocity.size, smooth_window)
    if velocity_window is None:
        return distances.size - 1

    smooth_velocity = savgol_filter(
        velocity,
        window_length=velocity_window,
        polyorder=min(3, velocity_window - 1),
    )
    denom = float(np.max(np.abs(smooth_velocity)))
    if denom <= 1e-12:
        return distances.size - 1

    normalized_velocity = smooth_velocity / denom
    candidates = np.flatnonzero((normalized_velocity < 0.1) & (np.arange(velocity.size) > 50))
    if candidates.size == 0:
        return distances.size - 1
    return int(candidates[0])


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


def trim_trajectory(cell: CellData, method: str = "midpoint_neb_ao") -> TrimmedCell:
    """Trim cell trajectories to the window ``[NEB, endpoint]``.

    Args:
        cell: Raw cell data (``neb``, ``ao1``, ``ao2`` are 1-based MATLAB indices).
        method: Endpoint strategy -- ``"midpoint_neb_ao"`` (default),
            ``"ao_mean"``, or ``"end_sep"``.

    Returns:
        TrimmedCell with arrays sliced to the chosen time window.
    """
    start = max(0, cell.neb - 1)
    end = min(cell.centrioles.shape[0] - 1, _compute_endpoint(cell, method))
    if end < start:
        end = start
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
