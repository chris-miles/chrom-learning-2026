# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chromlearn.analysis.trajectory_viz import plot_cell_trajectories, plot_chromosome_cloud
from chromlearn.io.catalog import list_cells, load_condition
from chromlearn.io.loader import load_cell
from chromlearn.io.trajectory import pole_pole_distance, trim_trajectory

plt.rcParams["figure.dpi"] = 110

# %%
cells_ctr = list_cells("rpe18_ctr")
print(f"Found {len(cells_ctr)} rpe18_ctr cells:")
for cell_id in cells_ctr:
    print(f"  {cell_id}")

# %%
DATA_DIR = ROOT / "data"
cell = load_cell(DATA_DIR / f"{cells_ctr[0]}.mat")
print(f"Cell ID: {cell.cell_id}")
print(f"Condition: {cell.condition}")
print(f"Centrioles shape: {cell.centrioles.shape}")
print(f"Chromosomes shape: {cell.chromosomes.shape}")
print(f"NEB frame: {cell.neb} (1-based)")
print(f"AO1: {cell.ao1}, AO2: {cell.ao2}")
print(f"Tracked chromosomes: {cell.tracked}")
print(f"NaN count in chromosomes: {np.isnan(cell.chromosomes).sum()}")

# %%
ppd = pole_pole_distance(cell)
time = np.arange(len(ppd)) * cell.dt
figure, axis = plt.subplots(figsize=(8, 3))
axis.plot(time, ppd)
axis.axvline((cell.neb - 1) * cell.dt, color="r", linestyle="--", label="NEB")
ao_mean = (cell.ao1 + cell.ao2) / 2.0
axis.axvline((ao_mean - 1) * cell.dt, color="g", linestyle="--", label="AO mean")
axis.set_xlabel("Time (s)")
axis.set_ylabel("Pole-pole distance (um)")
axis.set_title(cell.cell_id)
axis.legend()
plt.show()

# %%
trimmed = trim_trajectory(cell, method="midpoint_neb_ao")
print(
    f"Trimmed frames: {trimmed.start_frame}-{trimmed.end_frame} "
    f"({trimmed.chromosomes.shape[0]} timepoints)"
)
plot_cell_trajectories(trimmed, frame="lab")
plt.show()
plot_cell_trajectories(trimmed, frame="spindle")
plt.show()

# %%
plot_chromosome_cloud(trimmed, timepoint=0)
plt.show()

# %%
all_cells = load_condition("rpe18_ctr")
print(f"Loaded {len(all_cells)} cells")
for loaded in all_cells:
    trimmed_cell = trim_trajectory(loaded, method="midpoint_neb_ao")
    max_ppd = pole_pole_distance(loaded).max()
    print(
        f"{loaded.cell_id}: tracked={loaded.tracked}, "
        f"trimmed_timepoints={trimmed_cell.chromosomes.shape[0]}, "
        f"max_pole_pole_distance={max_ppd:.2f}"
    )
