import numpy as np

from chromlearn.model_fitting.basis import HatBasis
from chromlearn.model_fitting.diffusion import DiffusionResult
from chromlearn.model_fitting.model import FittedModel


def test_fitted_model_round_trip_preserves_diffusion_model(tmp_path) -> None:
    basis_xx = HatBasis(0.0, 8.0, n_basis=4)
    basis_xy = HatBasis(0.0, 10.0, n_basis=5)
    basis_D = HatBasis(-6.0, 6.0, n_basis=3)
    diffusion_model = DiffusionResult(
        d_coeffs=np.array([0.3, 0.5, 0.7]),
        basis_D=basis_D,
        coord_name="axial",
        D_scalar=0.5,
    )
    model = FittedModel(
        theta=np.arange(9, dtype=float),
        n_basis_xx=4,
        n_basis_xy=5,
        basis_xx=basis_xx,
        basis_xy=basis_xy,
        D_x=0.4,
        dt=5.0,
        metadata={"condition": "test"},
        diffusion_model=diffusion_model,
    )

    path = tmp_path / "model_with_diffusion.npz"
    model.save(path)
    loaded = FittedModel.load(path)

    np.testing.assert_allclose(loaded.theta, model.theta)
    assert loaded.metadata == model.metadata
    assert loaded.diffusion_model is not None
    assert loaded.diffusion_model.coord_name == diffusion_model.coord_name
    assert loaded.diffusion_model.D_scalar == diffusion_model.D_scalar
    np.testing.assert_allclose(loaded.diffusion_model.d_coeffs, diffusion_model.d_coeffs)
    assert loaded.diffusion_model.basis_D.r_min == diffusion_model.basis_D.r_min
    assert loaded.diffusion_model.basis_D.r_max == diffusion_model.basis_D.r_max
    assert loaded.diffusion_model.basis_D.n_basis == diffusion_model.basis_D.n_basis
