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
from chromlearn.model_fitting.fit import bootstrap_kernels, estimate_diffusion, fit_kernels
from chromlearn.model_fitting.model import FittedModel
from chromlearn.model_fitting.plotting import plot_kernels, plot_residuals
from chromlearn.model_fitting.validate import residual_diagnostics

# %%
cells = [trim_trajectory(cell, method="neb_ao_frac") for cell in load_condition("rpe18_ctr")]
basis_xx = BSplineBasis(0.5, 10.0, 10)
basis_xy = BSplineBasis(0.5, 12.0, 10)
G, V = build_design_matrix(cells, basis_xx, basis_xy)
roughness = np.block(
    [
        [basis_xx.roughness_matrix(), np.zeros((basis_xx.n_basis, basis_xy.n_basis))],
        [np.zeros((basis_xy.n_basis, basis_xx.n_basis)), basis_xy.roughness_matrix()],
    ]
)
fit_result = fit_kernels(G, V, lambda_ridge=1e-3, lambda_rough=1e-3, R=roughness)
bootstrap = bootstrap_kernels(
    cells,
    basis_xx,
    basis_xy,
    n_boot=100,
    lambda_ridge=1e-3,
    lambda_rough=1e-3,
)
model = FittedModel(
    theta=fit_result.theta,
    n_basis_xx=basis_xx.n_basis,
    n_basis_xy=basis_xy.n_basis,
    basis_xx=basis_xx,
    basis_xy=basis_xy,
    D_x=estimate_diffusion(V, G, fit_result.theta, dt=cells[0].dt),
    dt=cells[0].dt,
    metadata={"condition": "rpe18_ctr", "n_cells": len(cells)},
)

# %%
print("Fitted diffusion:", model.D_x)
print(residual_diagnostics(fit_result.residuals))

# %%
plot_kernels(model, bootstrap=bootstrap)
plt.show()
plot_residuals(fit_result.residuals)
plt.show()
