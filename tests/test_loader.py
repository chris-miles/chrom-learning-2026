from pathlib import Path

import numpy as np
import pytest

from chromlearn.io.loader import CellData, has_valid_neb, load_cell

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def get_sample_cell_path() -> Path:
    cells = sorted(DATA_DIR.glob("rpe18_ctr_*.mat"))
    if not cells:
        pytest.skip("No data files found")
    return cells[0]


def test_load_cell_returns_celldata() -> None:
    cell = load_cell(get_sample_cell_path())
    assert isinstance(cell, CellData)


def test_load_cell_fields() -> None:
    path = get_sample_cell_path()
    cell = load_cell(path)
    assert cell.cell_id == path.stem
    assert cell.condition == "rpe18_ctr"
    n_timepoints = cell.centrioles.shape[0]
    assert cell.centrioles.shape == (n_timepoints, 3, 2)
    assert cell.chromosomes.shape == (n_timepoints, 3, cell.tracked)
    assert isinstance(cell.neb, int)
    assert isinstance(cell.ao1, int)
    assert isinstance(cell.ao2, int)
    assert isinstance(cell.tracked, int)
    assert cell.dt == 5.0


def test_load_cell_chromosomes_are_centroids() -> None:
    cell = load_cell(get_sample_cell_path())
    assert cell.chromosomes.shape[1] == 3


def test_load_cell_nan_handling() -> None:
    cell = load_cell(get_sample_cell_path())
    assert cell.chromosomes.dtype in (np.float32, np.float64)


def test_has_valid_neb_detects_anaphase_only_files() -> None:
    assert has_valid_neb(DATA_DIR / "rpe18_ctr_500.mat")
    assert not has_valid_neb(DATA_DIR / "rpe18_ctr_622.mat")


def test_load_cell_rejects_anaphase_only_files() -> None:
    with pytest.raises(ValueError, match="neb=NaN"):
        load_cell(DATA_DIR / "rpe18_ctr_622.mat")
