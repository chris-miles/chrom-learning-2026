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


def test_fitted_model_no_xx_kernel():
    """FittedModel with basis_xx=None."""
    basis_xy = HatBasis(0.0, 10.0, n_basis=5)
    model = FittedModel(
        theta=np.arange(5, dtype=float),
        n_basis_xx=0,
        n_basis_xy=5,
        basis_xx=None,
        basis_xy=basis_xy,
        D_x=0.4,
        dt=5.0,
        topology="poles",
    )
    assert model.evaluate_kernel("xx", np.array([1.0, 2.0])) is None
    result_xy = model.evaluate_kernel("xy", np.array([1.0, 2.0]))
    assert result_xy.shape == (2,)
    assert model.theta_xx.shape == (0,)
    assert model.theta_xy.shape == (5,)


def test_save_load_no_xx(tmp_path):
    """Round-trip save/load with basis_xx=None."""
    basis_xy = HatBasis(0.0, 10.0, n_basis=5)
    model = FittedModel(
        theta=np.arange(5, dtype=float),
        n_basis_xx=0,
        n_basis_xy=5,
        basis_xx=None,
        basis_xy=basis_xy,
        D_x=0.4,
        dt=5.0,
        topology="poles",
    )
    path = tmp_path / "model_no_xx.npz"
    model.save(path)
    loaded = FittedModel.load(path)
    np.testing.assert_allclose(loaded.theta, model.theta)
    assert loaded.basis_xx is None
    assert loaded.n_basis_xx == 0
    assert loaded.topology == "poles"


def test_save_load_r_cutoff_xx(tmp_path):
    """Round-trip save/load preserves r_cutoff_xx."""
    basis_xx = HatBasis(0.0, 8.0, n_basis=4)
    basis_xy = HatBasis(0.0, 10.0, n_basis=5)
    model = FittedModel(
        theta=np.arange(9, dtype=float),
        n_basis_xx=4,
        n_basis_xy=5,
        basis_xx=basis_xx,
        basis_xy=basis_xy,
        D_x=0.4,
        dt=5.0,
        topology="poles_and_chroms",
        r_cutoff_xx=2.5,
    )
    path = tmp_path / "model_cutoff.npz"
    model.save(path)
    loaded = FittedModel.load(path)
    assert loaded.r_cutoff_xx == 2.5
    assert loaded.topology == "poles_and_chroms"


def test_save_load_r_cutoff_xx_none(tmp_path):
    """Round-trip save/load with r_cutoff_xx=None (default)."""
    basis_xy = HatBasis(0.0, 10.0, n_basis=5)
    model = FittedModel(
        theta=np.arange(5, dtype=float),
        n_basis_xx=0,
        n_basis_xy=5,
        basis_xx=None,
        basis_xy=basis_xy,
        D_x=0.4,
        dt=5.0,
    )
    path = tmp_path / "model_no_cutoff.npz"
    model.save(path)
    loaded = FittedModel.load(path)
    assert loaded.r_cutoff_xx is None


def test_evaluate_kernel_xx_respects_cutoff():
    """evaluate_kernel('xx', r) returns zero above r_cutoff_xx."""
    basis_xx = HatBasis(0.0, 8.0, n_basis=4)
    basis_xy = HatBasis(0.0, 10.0, n_basis=5)
    model = FittedModel(
        theta=np.ones(9, dtype=float),
        n_basis_xx=4,
        n_basis_xy=5,
        basis_xx=basis_xx,
        basis_xy=basis_xy,
        D_x=0.4,
        dt=5.0,
        r_cutoff_xx=2.5,
    )
    r = np.array([1.0, 2.0, 3.0, 5.0, 7.0])
    result = model.evaluate_kernel("xx", r)
    # Values at r <= 2.5 may be nonzero; values at r > 2.5 must be zero
    assert np.all(result[r > 2.5] == 0.0)
    # Sanity: at least one value below cutoff should be nonzero
    assert np.any(result[r <= 2.5] != 0.0)


def test_load_backward_compat_no_topology(tmp_path):
    """Loading old model files without topology field defaults to 'poles'."""
    np.savez(
        tmp_path / "old_model.npz",
        theta=np.arange(9, dtype=float),
        n_basis_xx=4, n_basis_xy=5,
        D_x=0.4, dt=5.0,
        metadata=np.array(None, dtype=object),
        basis_xx_type="hat", basis_xx_r_min=0.0, basis_xx_r_max=8.0, basis_xx_n_basis=4,
        basis_xy_type="hat", basis_xy_r_min=0.0, basis_xy_r_max=10.0, basis_xy_n_basis=5,
        diffusion_has_model=False,
    )
    loaded = FittedModel.load(tmp_path / "old_model.npz")
    assert loaded.topology == "poles"
