import numpy as np

from chromlearn.analysis.lag_correlation import (
    LagResult,
    compute_lag_correlation,
    compute_lag_correlation_single,
)
from chromlearn.io.loader import CellData


def make_correlated_cell(T: int = 200, N: int = 10, lag_offset: int = 3) -> CellData:
    rng = np.random.default_rng(42)
    pole_velocity = rng.standard_normal((T - 1, 3)) * 0.1
    pole_center = np.cumsum(np.vstack([np.zeros((1, 3)), pole_velocity]), axis=0)

    centrioles = np.zeros((T, 3, 2))
    centrioles[:, 0, 0] = pole_center[:, 0] - 5.0
    centrioles[:, 0, 1] = pole_center[:, 0] + 5.0
    centrioles[:, 1:, 0] = pole_center[:, 1:]
    centrioles[:, 1:, 1] = pole_center[:, 1:]

    chrom_center = np.zeros((T, 3))
    for time_index in range(T):
        chrom_center[time_index] = pole_center[max(0, time_index - lag_offset)] + rng.normal(0, 0.05, size=3)

    chromosomes = np.zeros((T, 3, N))
    for chrom_index in range(N):
        chromosomes[:, :, chrom_index] = chrom_center + rng.normal(0, 0.5, size=(T, 3))

    return CellData(
        cell_id="test_corr_001",
        condition="test",
        centrioles=centrioles,
        chromosomes=chromosomes,
        neb=1,
        ao1=T - 10,
        ao2=T - 8,
        tracked=N,
    )


def test_lag_correlation_single_shape() -> None:
    lags, values = compute_lag_correlation_single(make_correlated_cell(), lag_max=20, smooth_window=31)
    assert len(lags) == 41
    assert len(values) == 41


def test_lag_correlation_peak_near_positive_lag() -> None:
    lags, values = compute_lag_correlation_single(make_correlated_cell(lag_offset=5), lag_max=20, smooth_window=31)
    assert lags[int(np.nanargmax(values))] >= 0


def test_compute_lag_correlation_aggregated() -> None:
    result = compute_lag_correlation(
        [make_correlated_cell(lag_offset=3) for _ in range(5)],
        lag_max=20,
        smooth_window=31,
    )
    assert isinstance(result, LagResult)
    assert len(result.lags) == 41
    assert result.median.shape == (41,)
    assert result.std.shape == (41,)
