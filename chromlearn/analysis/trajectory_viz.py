from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from chromlearn.io.loader import CellData
from chromlearn.io.trajectory import TrimmedCell, spindle_frame, trim_trajectory


def plot_cell_trajectories(
    cell: CellData | TrimmedCell,
    frame: str = "lab",
    method: str = "neb_ao_frac",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    trimmed = trim_trajectory(cell, method=method) if isinstance(cell, CellData) else cell

    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 6))
    else:
        figure = ax.figure

    n_timepoints = trimmed.chromosomes.shape[0]
    colors = plt.cm.viridis(np.linspace(0, 1, n_timepoints))

    if frame == "spindle":
        spindle = spindle_frame(trimmed)
        for chrom_index in range(trimmed.tracked):
            for time_index in range(n_timepoints - 1):
                ax.plot(
                    spindle.axial[time_index : time_index + 2, chrom_index],
                    spindle.radial[time_index : time_index + 2, chrom_index],
                    color=colors[time_index],
                    linewidth=0.7,
                )
        ax.set_xlabel("Axial distance (um)")
        ax.set_ylabel("Radial distance (um)")
    elif frame == "lab":
        for chrom_index in range(trimmed.tracked):
            for time_index in range(n_timepoints - 1):
                ax.plot(
                    trimmed.chromosomes[time_index : time_index + 2, 0, chrom_index],
                    trimmed.chromosomes[time_index : time_index + 2, 1, chrom_index],
                    color=colors[time_index],
                    linewidth=0.7,
                )
        ax.plot(
            trimmed.centrioles[:, 0, 0],
            trimmed.centrioles[:, 1, 0],
            color="red",
            linewidth=2,
            label="Pole 1",
        )
        ax.plot(
            trimmed.centrioles[:, 0, 1],
            trimmed.centrioles[:, 1, 1],
            color="blue",
            linewidth=2,
            label="Pole 2",
        )
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.legend()
    else:
        raise ValueError("frame must be 'lab' or 'spindle'.")

    ax.set_title(trimmed.cell_id)
    ax.set_aspect("equal")
    figure.tight_layout()
    return figure


def plot_chromosome_cloud(
    cell: CellData | TrimmedCell,
    timepoint: int = 0,
    method: str = "neb_ao_frac",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    trimmed = trim_trajectory(cell, method=method) if isinstance(cell, CellData) else cell
    if not 0 <= timepoint < trimmed.chromosomes.shape[0]:
        raise IndexError("timepoint is out of bounds for the trimmed trajectory.")

    if ax is None:
        figure, ax = plt.subplots(figsize=(6, 6))
    else:
        figure = ax.figure

    chrom = trimmed.chromosomes[timepoint]
    poles = trimmed.centrioles[timepoint]
    ax.scatter(chrom[0], chrom[1], color="0.4", s=30, alpha=0.75, label="Chromosomes")
    ax.scatter(poles[0, 0], poles[1, 0], color="red", s=110, marker="^", label="Pole 1")
    ax.scatter(poles[0, 1], poles[1, 1], color="blue", s=110, marker="^", label="Pole 2")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(f"{trimmed.cell_id} t={timepoint}")
    ax.set_aspect("equal")
    ax.legend()
    figure.tight_layout()
    return figure
