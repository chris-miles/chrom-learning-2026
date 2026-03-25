# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chromlearn.io.catalog import load_condition
from chromlearn.io.trajectory import trim_trajectory
from chromlearn.model_fitting.basis import BSplineBasis
from chromlearn.model_fitting.fit import cross_validate
from chromlearn.model_fitting.plotting import plot_cv_curve

# %%
cells = [trim_trajectory(cell, method="neb_ao_frac") for cell in load_condition("rpe18_ctr")]
configs = {}
for n_basis in [4, 6, 8, 10]:
    basis_xx = BSplineBasis(0.5, 10.0, n_basis)
    basis_xy = BSplineBasis(0.5, 12.0, n_basis)
    configs[f"bspline-{n_basis}"] = cross_validate(
        cells,
        basis_xx,
        basis_xy,
        lambda_ridge=1e-3,
        lambda_rough=1e-3,
    )

# %%
for label, result in configs.items():
    print(label, result.mean_error, result.std_error)

# %%
plot_cv_curve(configs)
plt.show()
