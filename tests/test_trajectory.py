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


def make_fake_cell(T: int = 100, N: int = 10, neb: int = 10, ao1: int = 80, ao2: int = 82) -> CellData:
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
    assert distances.shape == (100,)
    np.testing.assert_allclose(distances[0], 2.0)
    assert np.all(np.diff(distances) > 0)


def test_pole_center() -> None:
    cell = make_fake_cell()
    center = pole_center(cell)
    assert center.shape == (100, 3)
    np.testing.assert_allclose(center, 0.0)


def test_trim_trajectory_midpoint() -> None:
    cell = make_fake_cell(T=100, neb=10, ao1=80, ao2=82)
    trimmed = trim_trajectory(cell, method="midpoint_neb_ao")
    assert isinstance(trimmed, TrimmedCell)
    expected_len = 44 - 9 + 1
    assert trimmed.chromosomes.shape[0] == expected_len
    assert trimmed.centrioles.shape[0] == expected_len


def test_trim_trajectory_ao_mean() -> None:
    cell = make_fake_cell(T=100, neb=10, ao1=80, ao2=82)
    trimmed = trim_trajectory(cell, method="ao_mean")
    expected_len = 80 - 9 + 1
    assert trimmed.chromosomes.shape[0] == expected_len


def test_trim_preserves_particle_count() -> None:
    cell = make_fake_cell(N=15)
    trimmed = trim_trajectory(cell, method="ao_mean")
    assert trimmed.chromosomes.shape[2] == 15
    assert trimmed.centrioles.shape[2] == 2


def test_compute_end_sep_returns_int() -> None:
    cell = make_fake_cell(T=120)
    end_sep = compute_end_sep(cell)
    assert isinstance(end_sep, int)
    assert 0 <= end_sep < cell.centrioles.shape[0]


def test_spindle_frame_axes() -> None:
    cell = make_fake_cell()
    trimmed = trim_trajectory(cell, method="ao_mean")
    frame = spindle_frame(trimmed)
    assert isinstance(frame, SpindleFrameData)
    assert frame.axial.shape == (trimmed.chromosomes.shape[0], trimmed.chromosomes.shape[2])
    assert frame.radial.shape == (trimmed.chromosomes.shape[0], trimmed.chromosomes.shape[2])
