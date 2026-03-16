# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chromlearn.model_fitting.basis import BSplineBasis
from chromlearn.model_fitting.features import build_design_matrix
from chromlearn.model_fitting.fit import estimate_diffusion, fit_kernels
from chromlearn.model_fitting.model import FittedModel
from chromlearn.model_fitting.plotting import plot_recovery, plot_residuals
from chromlearn.model_fitting.simulate import add_localization_noise, generate_synthetic_data
from chromlearn.model_fitting.validate import kernel_recovery_error, residual_diagnostics
from chromlearn.io.trajectory import TrimmedCell

# %%
def true_xx(r: np.ndarray) -> np.ndarray:
    return -0.02 * np.exp(-r / 1.5)


def true_xy(r: np.ndarray) -> np.ndarray:
    return 0.05 * np.exp(-r / 6.0)


rng = np.random.default_rng(42)
dataset = generate_synthetic_data(
    kernel_xx=true_xx,
    kernel_xy=true_xy,
    n_chromosomes=20,
    n_steps=80,
    dt=5.0,
    D_x=0.1,
    rng=rng,
)
noisy_chromosomes = add_localization_noise(dataset.chromosomes, sigma=0.05, rng=rng)

# %%
trimmed = TrimmedCell(
    cell_id="synthetic_001",
    condition="synthetic",
    centrioles=dataset.centrosomes,
    chromosomes=noisy_chromosomes,
    tracked=noisy_chromosomes.shape[2],
    dt=dataset.dt,
    start_frame=0,
    end_frame=noisy_chromosomes.shape[0] - 1,
)

basis_xx = BSplineBasis(0.5, 8.0, 8)
basis_xy = BSplineBasis(0.5, 12.0, 8)
G, V = build_design_matrix([trimmed], basis_xx, basis_xy)
roughness = np.block(
    [
        [basis_xx.roughness_matrix(), np.zeros((basis_xx.n_basis, basis_xy.n_basis))],
        [np.zeros((basis_xy.n_basis, basis_xx.n_basis)), basis_xy.roughness_matrix()],
    ]
)
fit_result = fit_kernels(G, V, lambda_ridge=1e-3, lambda_rough=1e-3, R=roughness)
D_x_hat = estimate_diffusion(V, G, fit_result.theta, dt=trimmed.dt)
model = FittedModel(
    theta=fit_result.theta,
    n_basis_xx=basis_xx.n_basis,
    n_basis_xy=basis_xy.n_basis,
    basis_xx=basis_xx,
    basis_xy=basis_xy,
    D_x=D_x_hat,
    dt=trimmed.dt,
    metadata={"source": "synthetic_validation"},
)

# %%
r_xx = np.linspace(basis_xx.r_min, basis_xx.r_max, 200)
r_xy = np.linspace(basis_xy.r_min, basis_xy.r_max, 200)
fitted_xx = model.evaluate_kernel("xx", r_xx)
fitted_xy = model.evaluate_kernel("xy", r_xy)
true_xx_values = true_xx(r_xx)
true_xy_values = true_xy(r_xy)
print("Kernel recovery error xx:", kernel_recovery_error(r_xx, true_xx_values, fitted_xx))
print("Kernel recovery error xy:", kernel_recovery_error(r_xy, true_xy_values, fitted_xy))
print("Estimated diffusion:", model.D_x)

# %%
plot_recovery(r_xx, true_xx_values, fitted_xx, kernel_name="f_xx")
plt.show()
plot_recovery(r_xy, true_xy_values, fitted_xy, kernel_name="f_xy")
plt.show()

# %%
print(residual_diagnostics(fit_result.residuals))
plot_residuals(fit_result.residuals)
plt.show()
