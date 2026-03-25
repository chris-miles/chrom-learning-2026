# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chromlearn.io.catalog import load_condition
from chromlearn.io.trajectory import trim_trajectory
from chromlearn.model_fitting.basis import BSplineBasis
from chromlearn.model_fitting.features import build_design_matrix
from chromlearn.model_fitting.fit import estimate_diffusion, fit_kernels
from chromlearn.model_fitting.model import FittedModel
from chromlearn.model_fitting.plotting import plot_kernels
from chromlearn.model_fitting.simulate import simulate_trajectories
from chromlearn.model_fitting.validate import summary_statistics

# %%
cells = [trim_trajectory(cell, method="neb_ao_frac") for cell in load_condition("rpe18_ctr")]
reference = cells[0]
basis_xx = BSplineBasis(0.5, 10.0, 8)
basis_xy = BSplineBasis(0.5, 12.0, 8)
G, V = build_design_matrix(cells, basis_xx, basis_xy)
roughness = np.block(
    [
        [basis_xx.roughness_matrix(), np.zeros((basis_xx.n_basis, basis_xy.n_basis))],
        [np.zeros((basis_xy.n_basis, basis_xx.n_basis)), basis_xy.roughness_matrix()],
    ]
)
fit_result = fit_kernels(G, V, lambda_ridge=1e-3, lambda_rough=1e-3, R=roughness)
model = FittedModel(
    theta=fit_result.theta,
    n_basis_xx=basis_xx.n_basis,
    n_basis_xy=basis_xy.n_basis,
    basis_xx=basis_xx,
    basis_xy=basis_xy,
    D_x=estimate_diffusion(V, G, fit_result.theta, dt=reference.dt),
    dt=reference.dt,
    metadata={"condition": "rpe18_ctr"},
)

# %%
simulated = simulate_trajectories(
    kernel_xx=lambda r: model.evaluate_kernel("xx", r),
    kernel_xy=lambda r: model.evaluate_kernel("xy", r),
    centrosome_positions=reference.centrioles,
    x0=reference.chromosomes[0].T,
    n_steps=reference.chromosomes.shape[0] - 1,
    dt=reference.dt,
    D_x=model.D_x,
    rng=np.random.default_rng(42),
)

# %%
real_stats = summary_statistics(reference.chromosomes, reference.centrioles)
sim_stats = summary_statistics(simulated, reference.centrioles)
print("Real statistics:", real_stats)
print("Simulated statistics:", sim_stats)

# %%
plot_kernels(model)
plt.show()
