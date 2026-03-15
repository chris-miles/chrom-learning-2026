import numpy as np
import pytest

from chromlearn.io.trajectory import TrimmedCell
from chromlearn.model_fitting.basis import BSplineBasis, HatBasis
from chromlearn.model_fitting.diffusion import (
    COORDINATE_MAPS,
    DiffusionResult,
    estimate_diffusion_variable,
    local_diffusion_estimates,
)


def make_diffusion_cell(T: int = 50, N: int = 6) -> TrimmedCell:
    """Synthetic cell with poles on x-axis, chromosomes near origin."""
    rng = np.random.default_rng(99)
    centrioles = np.zeros((T, 3, 2))
    centrioles[:, 0, 0] = -5.0
    centrioles[:, 0, 1] = 5.0
    chromosomes = rng.normal(0.0, 2.0, size=(T, 3, N))
    return TrimmedCell(
        cell_id="diff_test_001",
        condition="test",
        centrioles=centrioles,
        chromosomes=chromosomes,
        tracked=N,
        dt=5.0,
        start_frame=0,
        end_frame=T - 1,
    )


# ---- Coordinate maps ----


def test_coordinate_maps_keys() -> None:
    assert set(COORDINATE_MAPS.keys()) == {"axial", "radial", "distance"}


def test_coord_axial_shape() -> None:
    cell = make_diffusion_cell()
    coords = COORDINATE_MAPS["axial"](cell.chromosomes, cell)
    assert coords.shape == (50, 6)


def test_coord_radial_nonnegative() -> None:
    cell = make_diffusion_cell()
    coords = COORDINATE_MAPS["radial"](cell.chromosomes, cell)
    assert np.all(coords >= 0)


def test_coord_distance_ge_radial() -> None:
    cell = make_diffusion_cell()
    radial = COORDINATE_MAPS["radial"](cell.chromosomes, cell)
    distance = COORDINATE_MAPS["distance"](cell.chromosomes, cell)
    np.testing.assert_array_less(radial - 1e-10, distance)


# ---- Local diffusion estimators ----


def make_pure_diffusion_cell(T: int = 200, N: int = 20, D: float = 0.5, dt: float = 1.0) -> TrimmedCell:
    """Pure Brownian motion with known D, no drift."""
    rng = np.random.default_rng(42)
    noise_std = np.sqrt(2.0 * D * dt)
    chromosomes = np.zeros((T, 3, N))
    for t in range(1, T):
        chromosomes[t] = chromosomes[t - 1] + noise_std * rng.standard_normal((3, N))
    centrioles = np.zeros((T, 3, 2))
    centrioles[:, 0, 0] = -5.0
    centrioles[:, 0, 1] = 5.0
    return TrimmedCell(
        cell_id="brownian",
        condition="test",
        centrioles=centrioles,
        chromosomes=chromosomes,
        tracked=N,
        dt=dt,
        start_frame=0,
        end_frame=T - 1,
    )


def test_msd_recovers_D() -> None:
    D_true = 0.5
    cell = make_pure_diffusion_cell(T=500, N=30, D=D_true, dt=1.0)
    D_local = local_diffusion_estimates([cell], dt=1.0, mode="msd")
    assert len(D_local) == 1
    D_mean = float(np.nanmean(D_local[0]))
    np.testing.assert_allclose(D_mean, D_true, rtol=0.15)


def test_vestergaard_recovers_D() -> None:
    D_true = 0.5
    cell = make_pure_diffusion_cell(T=500, N=30, D=D_true, dt=1.0)
    D_local = local_diffusion_estimates([cell], dt=1.0, mode="vestergaard")
    assert len(D_local) == 1
    # vestergaard has T-2 valid timepoints
    assert D_local[0].shape[0] == cell.chromosomes.shape[0] - 2
    D_mean = float(np.nanmean(D_local[0]))
    np.testing.assert_allclose(D_mean, D_true, rtol=0.15)


def test_weak_noise_recovers_D() -> None:
    D_true = 0.5
    cell = make_pure_diffusion_cell(T=500, N=30, D=D_true, dt=1.0)
    D_local = local_diffusion_estimates([cell], dt=1.0, mode="weak_noise")
    assert len(D_local) == 1
    D_mean = float(np.nanmean(D_local[0]))
    # weak_noise has higher variance but should still be in the ballpark
    np.testing.assert_allclose(D_mean, D_true, rtol=0.3)


def test_local_diffusion_unknown_mode() -> None:
    cell = make_diffusion_cell()
    with pytest.raises(ValueError, match="Unknown mode"):
        local_diffusion_estimates([cell], dt=5.0, mode="bogus")


def test_f_corrected_requires_fit_result() -> None:
    cell = make_diffusion_cell()
    with pytest.raises(ValueError, match="f_corrected"):
        local_diffusion_estimates([cell], dt=5.0, mode="f_corrected")


# ---- Variable diffusion fitting ----


def test_estimate_diffusion_variable_runs() -> None:
    cell = make_pure_diffusion_cell(T=100, N=10, D=0.5, dt=1.0)
    basis_D = HatBasis(r_min=-6.0, r_max=6.0, n_basis=5)
    result = estimate_diffusion_variable(
        [cell], basis_D, coord_name="axial", dt=1.0, mode="msd",
    )
    assert isinstance(result, DiffusionResult)
    assert result.d_coeffs.shape == (5,)
    assert result.D_scalar > 0


def test_diffusion_result_evaluate() -> None:
    cell = make_pure_diffusion_cell(T=100, N=10, D=0.5, dt=1.0)
    basis_D = HatBasis(r_min=-6.0, r_max=6.0, n_basis=5)
    result = estimate_diffusion_variable(
        [cell], basis_D, coord_name="axial", dt=1.0, mode="msd",
    )
    coords = np.linspace(-5.0, 5.0, 20)
    D_vals = result.evaluate(coords)
    assert D_vals.shape == (20,)
    # For constant-D data, the fitted D(coord) should be roughly flat
    assert np.std(D_vals) < 0.5 * np.mean(D_vals)


def test_unknown_coord_name_raises() -> None:
    cell = make_diffusion_cell()
    basis_D = HatBasis(r_min=0.0, r_max=10.0, n_basis=3)
    with pytest.raises(ValueError, match="Unknown coordinate"):
        estimate_diffusion_variable([cell], basis_D, coord_name="bogus", dt=5.0)
