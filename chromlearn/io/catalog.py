from __future__ import annotations

from pathlib import Path

from chromlearn.io.loader import CellData, has_valid_neb, load_cell

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

CONDITIONS = {
    "rpe18_ctr": "rpe18_ctr",
    "rod311_ctr": "rod311_ctr",
    "rod311_prc": "rod311_prc",
    "rod311_rev": "rod311_rev",
    "rpe18_rev": "rpe18_rev",
    "rpe18_cytoD": "rpe18_cytoD",
    "rpe18_hesp": "rpe18_hesp",
    "rpe18_zm": "rpe18_zm",
}


def _matching_paths(condition: str, data_dir: Path) -> list[Path]:
    prefix = CONDITIONS.get(condition, condition).lower() + "_"
    return sorted(
        path
        for path in data_dir.glob("*.mat")
        if path.stem.lower().startswith(prefix)
        and has_valid_neb(path)
    )


def list_cells(condition: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> list[str]:
    """List available cell IDs for a given condition.

    Discovers cells by globbing for ``{condition}_*.mat`` in *data_dir* and
    excludes files whose ``neb`` field is NaN (anaphase-only).

    Returns:
        Sorted list of cell-ID strings (filename stems).
    """
    data_dir = Path(data_dir)
    return [path.stem for path in _matching_paths(condition, data_dir)]


def load_condition(
    condition: str,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    dt: float = 5.0,
) -> list[CellData]:
    """Load all valid cells for a given experimental condition.

    Args:
        condition: Condition prefix, e.g. ``"rpe18_ctr"``.
        data_dir: Directory containing ``.mat`` files.
        dt: Time interval between frames in seconds.

    Returns:
        List of :class:`CellData` objects, sorted by cell ID.
    """
    data_dir = Path(data_dir)
    return [load_cell(path, dt=dt) for path in _matching_paths(condition, data_dir)]
