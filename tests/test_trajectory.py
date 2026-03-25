import numpy as np

from chromlearn.io.loader import CellData
from chromlearn.io.trajectory import (
    SpindleFrameData,
    TrimmedCell,
    compute_end_sep,
    pole_center,
    pole_pole_distance,
    spindle_frame,
    trim_trajectory,
)


def make_fake_cell(T: int = 250, N: int = 10, neb: int = 10, ao1: int = 220, ao2: int = 225) -> CellData:
    rng = np.random.default_rng(42)
    centrioles = np.zeros((T, 3, 2))
    for time_index in range(T):
        separation = 2.0 + 0.1 * time_index
        centrioles[time_index, 0, 0] = -0.5 * separation
        centrioles[time_index, 0, 1] = 0.5 * separation
    chromosomes = rng.normal(0.0, 2.0, size=(T, 3, N))
    return CellData(
        cell_id="test_cell_001",
        condition="test",
        centrioles=centrioles,
        chromosomes=chromosomes,
        neb=neb,
        ao1=ao1,
        ao2=ao2,
        tracked=N,
    )


def test_pole_pole_distance() -> None:
    cell = make_fake_cell()
    distances = pole_pole_distance(cell)
    assert distances.shape == (cell.centrioles.shape[0],)
    np.testing.assert_allclose(distances[0], 2.0)
    assert np.all(np.diff(distances) > 0)


def test_pole_center() -> None:
    cell = make_fake_cell()
    center = pole_center(cell)
    assert center.shape == (cell.centrioles.shape[0], 3)
    np.testing.assert_allclose(center, 0.0)


def test_trim_trajectory_default_frac() -> None:
    cell = make_fake_cell()
    trimmed = trim_trajectory(cell, method="neb_ao_frac")
    assert isinstance(trimmed, TrimmedCell)
    # neb=10 (1-based) -> start=9; ao_mean=222 (1-based) -> 221 (0-based)
    # endpoint = round(9 + (1/3) * (221 - 9)) = round(9 + 70.667) = 80
    # window = 80 - 9 + 1 = 72
    expected_len = 72
    assert trimmed.chromosomes.shape[0] == expected_len
    assert trimmed.centrioles.shape[0] == expected_len


def test_trim_trajectory_full_window() -> None:
    cell = make_fake_cell()
    trimmed = trim_trajectory(cell, method="neb_ao_frac", frac=1.0)
    # ao_mean = round((220+225)/2) = 222 (1-based) -> 221 (0-based)
    # endpoint = round(9 + 1.0 * (221 - 9)) = 221; window = 221 - 9 + 1 = 213
    expected_len = 213
    assert trimmed.chromosomes.shape[0] == expected_len


def test_trim_preserves_particle_count() -> None:
    cell = make_fake_cell(N=15)
    trimmed = trim_trajectory(cell, method="neb_ao_frac", frac=1.0)
    assert trimmed.chromosomes.shape[2] == 15
    assert trimmed.centrioles.shape[2] == 2


def test_compute_end_sep_returns_int() -> None:
    cell = make_fake_cell(T=120)
    end_sep = compute_end_sep(cell)
    assert isinstance(end_sep, int)
    assert 0 <= end_sep < cell.centrioles.shape[0]


def test_spindle_frame_axes() -> None:
    cell = make_fake_cell()
    trimmed = trim_trajectory(cell, method="neb_ao_frac", frac=1.0)
    frame = spindle_frame(trimmed)
    assert isinstance(frame, SpindleFrameData)
    assert frame.axial.shape == (trimmed.chromosomes.shape[0], trimmed.chromosomes.shape[2])
    assert frame.radial.shape == (trimmed.chromosomes.shape[0], trimmed.chromosomes.shape[2])


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
