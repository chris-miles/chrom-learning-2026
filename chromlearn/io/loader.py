from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CellData:
    """Container for one cell's trajectory data.

    Note: ``neb``, ``ao1``, and ``ao2`` are stored exactly as loaded from the
    MATLAB files and therefore use 1-based indexing. Convert to 0-based before
    using them as NumPy indices, e.g. ``arr[cell.neb - 1]``.
    """

    cell_id: str
    condition: str
    centrioles: np.ndarray
    chromosomes: np.ndarray
    neb: int
    ao1: int
    ao2: int
    tracked: int
    dt: float = 5.0


def _parse_condition(cell_id: str) -> str:
    parts = cell_id.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Could not parse condition from cell id '{cell_id}'.")
    return parts[0].lower()


def _load_mat_scipy(path: Path) -> dict[str, Any]:
    import scipy.io

    return scipy.io.loadmat(str(path), squeeze_me=True)


def _load_mat_h5py(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Loading MATLAB v7.3 files requires the optional dependency 'h5py'."
        ) from exc

    data: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for key in handle.keys():
            data[key] = handle[key][()]
    return data


def _load_mat(path: Path) -> dict[str, Any]:
    try:
        return _load_mat_scipy(path)
    except (NotImplementedError, ValueError):
        return _load_mat_h5py(path)


def _extract_scalar_int(value: Any) -> int:
    if isinstance(value, np.ndarray):
        value = np.asarray(value).reshape(-1)[0]
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        raise ValueError("Encountered NaN where an integer metadata value was required.")
    return int(value)


def _extract_scalar_int_with_default(value: Any, default: int) -> int:
    try:
        return _extract_scalar_int(value)
    except ValueError:
        return default


def has_valid_neb(path: Path | str) -> bool:
    """Return True if the file contains a usable NEB frame."""

    raw = _load_mat(Path(path))
    try:
        _extract_scalar_int(raw["neb"])
    except (KeyError, ValueError):
        return False
    return True


def _normalize_centrioles(raw_centrioles: Any) -> np.ndarray:
    centrioles = np.asarray(raw_centrioles, dtype=np.float64)
    if centrioles.ndim != 3:
        raise ValueError(
            f"Expected centrioles to have 3 dimensions, got shape {centrioles.shape}."
        )
    if centrioles.shape[1] == 3:
        return centrioles
    if centrioles.shape[1] >= 6:
        # The raw files include duplicated/auxiliary pole coordinates. The
        # MATLAB reference pipeline uses the first xyz triplet for each pole.
        return centrioles[:, 0:3, :]
    raise ValueError(
        "Unsupported centrioles layout. Expected shape (T, 3, 2) or (T, >=6, 2), "
        f"got {centrioles.shape}."
    )


def _compute_chromosome_centroids(raw_kinetochores: Any) -> np.ndarray:
    kinetochores = np.asarray(raw_kinetochores, dtype=np.float64)
    if kinetochores.ndim != 3 or kinetochores.shape[1] < 6:
        raise ValueError(
            "Expected kinetochores to have shape (T, 6, N) or compatible, "
            f"got {kinetochores.shape}."
        )
    sister_1 = kinetochores[:, 0:3, :]
    sister_2 = kinetochores[:, 3:6, :]
    valid_counts = (~np.isnan(sister_1)).astype(np.int64) + (~np.isnan(sister_2)).astype(
        np.int64
    )
    summed = np.nan_to_num(sister_1, nan=0.0) + np.nan_to_num(sister_2, nan=0.0)
    chromosomes = np.divide(
        summed,
        valid_counts,
        out=np.full_like(summed, np.nan),
        where=valid_counts > 0,
    )
    return chromosomes


def load_cell(path: Path | str, dt: float = 5.0) -> CellData:
    """Load one cell from a MATLAB trajectory file.

    The returned metadata frame indices remain 1-based to preserve the source
    MATLAB convention. Convert them during trimming/indexing.

    Files with ``neb = NaN`` are treated as anaphase-only and rejected.
    """

    path = Path(path)
    raw = _load_mat(path)

    required = {"centrioles", "kinetochores", "neb", "tracked"}
    missing = sorted(required.difference(raw))
    if missing:
        raise KeyError(f"Missing required variables in {path.name}: {missing}")
    # Support files that store a single 'ao' instead of separate 'ao1'/'ao2'.
    # If the value is NaN, fall back to the last frame.
    has_ao_pair = "ao1" in raw and "ao2" in raw
    if not has_valid_neb(path):
        raise ValueError(
            f"{path.name} has neb=NaN and is treated as an anaphase-only file; "
            "it is excluded from chromlearn analyses."
        )

    centrioles = _normalize_centrioles(raw["centrioles"])
    chromosomes = _compute_chromosome_centroids(raw["kinetochores"])

    tracked = _extract_scalar_int(raw["tracked"])
    if chromosomes.shape[2] != tracked:
        chromosomes = chromosomes[:, :, :tracked]
    if centrioles.shape[2] != 2:
        raise ValueError(
            f"Expected two centrosome poles, got centrioles shape {centrioles.shape}."
        )

    cell_id = path.stem
    return CellData(
        cell_id=cell_id,
        condition=_parse_condition(cell_id),
        centrioles=centrioles,
        chromosomes=chromosomes,
        neb=_extract_scalar_int(raw["neb"]),
        ao1=_extract_scalar_int_with_default(
            raw["ao1"] if has_ao_pair else raw.get("ao", np.nan),
            default=centrioles.shape[0],  # last frame, 1-based
        ),
        ao2=_extract_scalar_int_with_default(
            raw["ao2"] if has_ao_pair else raw.get("ao", np.nan),
            default=centrioles.shape[0],
        ),
        tracked=tracked,
        dt=dt,
    )
