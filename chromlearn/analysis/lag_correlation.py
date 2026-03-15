from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from chromlearn.io.loader import CellData


@dataclass
class LagResult:
    lags: np.ndarray
    per_cell: np.ndarray
    median: np.ndarray
    std: np.ndarray


def _adjust_window(length: int, desired: int) -> int | None:
    if length < 3:
        return None
    window = min(desired, length if length % 2 == 1 else length - 1)
    if window < 3:
        return None
    return window


def compute_lag_correlation_single(
    cell: CellData,
    lag_max: int = 101,
    smooth_window: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    pole_centers = 0.5 * (cell.centrioles[:, :, 0] + cell.centrioles[:, :, 1])
    chromosome_center = np.nanmean(cell.chromosomes, axis=2)

    window = _adjust_window(len(pole_centers), smooth_window)
    if window is not None:
        pole_centers = savgol_filter(
            pole_centers,
            window_length=window,
            polyorder=min(3, window - 1),
            axis=0,
        )
        chromosome_center = savgol_filter(
            chromosome_center,
            window_length=window,
            polyorder=min(3, window - 1),
            axis=0,
        )

    pole_velocity = np.diff(pole_centers, axis=0)
    chromosome_velocity = np.diff(chromosome_center, axis=0)
    n_velocity = len(pole_velocity)

    lags = np.arange(-lag_max, lag_max + 1)
    autocorrelation = np.full(lags.shape, np.nan, dtype=np.float64)

    for lag_index, lag in enumerate(lags):
        dots: list[float] = []
        if lag < 0:
            valid_indices = range(-lag, n_velocity)
        else:
            valid_indices = range(0, n_velocity - lag)

        for index in valid_indices:
            pole_vec = pole_velocity[index]
            chrom_vec = chromosome_velocity[index + lag]
            pole_norm = float(np.linalg.norm(pole_vec))
            chrom_norm = float(np.linalg.norm(chrom_vec))
            if pole_norm <= 1e-12 or chrom_norm <= 1e-12:
                continue
            dots.append(float(np.dot(pole_vec, chrom_vec) / (pole_norm * chrom_norm)))

        if dots:
            autocorrelation[lag_index] = float(np.mean(dots))

    return lags, autocorrelation


def compute_lag_correlation(
    cells: list[CellData],
    lag_max: int = 101,
    smooth_window: int = 31,
) -> LagResult:
    if not cells:
        raise ValueError("compute_lag_correlation requires at least one cell.")

    per_cell = []
    lags = None
    for cell in cells:
        lags, autocorrelation = compute_lag_correlation_single(
            cell,
            lag_max=lag_max,
            smooth_window=smooth_window,
        )
        per_cell.append(autocorrelation)

    per_cell_array = np.asarray(per_cell, dtype=np.float64)
    return LagResult(
        lags=lags * cells[0].dt,
        per_cell=per_cell_array,
        median=np.nanmedian(per_cell_array, axis=0),
        std=np.nanstd(per_cell_array, axis=0),
    )


def plot_lag_correlation(result: LagResult) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6, 5))
    for row in result.per_cell:
        axis.plot(result.lags, row, color="C1", alpha=0.35, linewidth=0.9)
    axis.plot(result.lags, result.median, color="k", linewidth=2.5, label="Median")
    axis.fill_between(
        result.lags,
        result.median - result.std,
        result.median + result.std,
        color="0.5",
        alpha=0.2,
        label="±1 SD",
    )
    axis.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    axis.axvline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Lag (seconds)")
    axis.set_ylabel("Normalized velocity dot product")
    axis.set_title("Centrosome-chromosome lag correlation")
    axis.legend()
    figure.tight_layout()
    return figure
