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


def _normalize_range(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = float(np.max(values) - np.min(values))
    if span <= 1e-12:
        return np.zeros_like(values)
    return (values - np.min(values)) / span


def _normalize_max_abs(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(np.max(np.abs(values)))
    if scale <= 1e-12:
        return np.zeros_like(values)
    return values / scale


def compute_end_sep(
    cell: CellData,
    smooth_window: int = 50,
    min_frames_after_neb: int = 50,
    velocity_threshold: float = 0.1,
) -> int:
    """Detect the end of spindle separation using the old MATLAB velocity rule.

    This mirrors the legacy MATLAB logic:
    1. Smooth the pole-pole distance.
    2. Normalize the smoothed separation to ``[0, 1]``.
    3. Differentiate, smooth again, and normalize by its max absolute value.
    4. Return the first frame where the normalized separation velocity falls
       below ``velocity_threshold`` at least ``min_frames_after_neb`` frames
       past NEB.

    To avoid contamination from anaphase onset, only frames strictly before the
    first AO annotation are used when constructing the separation velocity.
    """
    neb_frame = cell.neb - 1
    ao_frame = _ao_min_index(cell)

    distances = pole_pole_distance(cell)
    if distances.size <= 2:
        return max(0, distances.size - 1)

    search_stop = min(distances.size, ao_frame)
    if search_stop <= 1:
        return max(neb_frame, min(distances.size - 1, max(0, ao_frame - 1)))

    segment = distances[:search_stop]
    window = _savgol_window(segment.size, smooth_window)
    if window is None:
        smoothed = segment.astype(float, copy=True)
    else:
        smoothed = savgol_filter(
            segment,
            window_length=window,
            polyorder=min(3, window - 1),
        )

    normalized_sep = _normalize_range(smoothed)
    sep_velocity = np.diff(normalized_sep)
    if sep_velocity.size == 0:
        return max(neb_frame, search_stop - 1)

    vel_window = _savgol_window(sep_velocity.size, smooth_window)
    if vel_window is None:
        smoothed_velocity = sep_velocity.astype(float, copy=True)
    else:
        smoothed_velocity = savgol_filter(
            sep_velocity,
            window_length=vel_window,
            polyorder=min(3, vel_window - 1),
        )

    normalized_velocity = _normalize_max_abs(smoothed_velocity)
    first_allowed = neb_frame + min_frames_after_neb
    candidates = np.flatnonzero(
        (normalized_velocity < velocity_threshold)
        & (np.arange(normalized_velocity.size) >= first_allowed)
    )
    if candidates.size == 0:
        return max(neb_frame, search_stop - 1)
    return int(candidates[0])


def _ao_min_index(cell: CellData) -> int:
    return min(cell.ao1, cell.ao2) - 1


def _ao_mean_index(cell: CellData) -> int:
    return int(round((cell.ao1 + cell.ao2) / 2.0)) - 1


def _compute_endpoint(cell: CellData, method: str, frac: float = 1.0 / 3.0) -> int:
    neb_index = cell.neb - 1
    if method == "neb_ao_frac":
        ao_index = _ao_mean_index(cell)
        return int(round(neb_index + frac * (ao_index - neb_index)))
    if method == "end_sep":
        return compute_end_sep(cell)
    raise ValueError(
        f"Unknown endpoint method '{method}'. "
        "Expected 'neb_ao_frac' or 'end_sep'."
    )


def trim_trajectory(
    cell: CellData,
    method: str = "neb_ao_frac",
    frac: float = 0.4,
    min_frames: int = 25,
) -> TrimmedCell:
    """Trim cell trajectories to the window ``[NEB, endpoint]``.

    Args:
        cell: Raw cell data (``neb``, ``ao1``, ``ao2`` are 1-based MATLAB indices).
        method: Endpoint strategy -- ``"neb_ao_frac"`` (default) or
            ``"end_sep"``.
        frac: Fraction of the ``[NEB, AO]`` window to use (only for
            ``"neb_ao_frac"``).  ``0.5`` = midpoint (default), ``1.0`` = full
            window to AO.
        min_frames: Minimum number of frames required after trimming.

    Returns:
        TrimmedCell with arrays sliced to the chosen time window.

    Raises:
        ValueError: If the trimmed trajectory has fewer than *min_frames* frames.
    """
    start = max(0, cell.neb - 1)
    end = min(cell.centrioles.shape[0] - 1, _compute_endpoint(cell, method, frac))
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
