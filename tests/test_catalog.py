from pathlib import Path

from chromlearn.io.catalog import list_cells, load_condition
from chromlearn.io.loader import CellData

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_list_cells_rpe18_ctr() -> None:
    cells = list_cells("rpe18_ctr", data_dir=DATA_DIR)
    assert len(cells) >= 7, f"Expected at least 7 rpe18_ctr cells, got {len(cells)}"
    assert all(isinstance(cell, str) for cell in cells)
    assert all("rpe18_ctr" in cell for cell in cells)


def test_list_cells_returns_sorted() -> None:
    cells = list_cells("rpe18_ctr", data_dir=DATA_DIR)
    assert cells == sorted(cells)


def test_list_cells_unknown_condition() -> None:
    assert list_cells("nonexistent_condition", data_dir=DATA_DIR) == []


def test_load_condition() -> None:
    cells = load_condition("rpe18_ctr", data_dir=DATA_DIR)
    assert len(cells) >= 7, f"Expected at least 7 rpe18_ctr cells, got {len(cells)}"
    assert all(isinstance(cell, CellData) for cell in cells)
    assert all(cell.condition == "rpe18_ctr" for cell in cells)


def test_catalog_skips_anaphase_only_files() -> None:
    cells = list_cells("rpe18_ctr", data_dir=DATA_DIR)
    assert "rpe18_ctr_622" not in cells
