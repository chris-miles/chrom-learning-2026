import numpy as np
import pytest
from types import SimpleNamespace

from chromlearn.io.trajectory import TrimmedCell
from chromlearn.model_fitting.basis import BSplineBasis, HatBasis
from chromlearn.model_fitting.diffusion import (
    COORDINATE_MAPS,
    DiffusionResult,
    estimate_diffusion_variable,
    local_diffusion_estimates,
)
from chromlearn.model_fitting.simulate import simulate_trajectories


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


def make_observed_diffusion_cell(
    T: int = 2000,
    N: int = 100,
    D: float = 0.5,
    dt: float = 1.0,
    localization_sigma: float = 0.0,
    drift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> TrimmedCell:
    """Brownian motion with optional constant drift and observation noise."""
    rng = np.random.default_rng(42)
    drift_vec = np.asarray(drift, dtype=float).reshape(3, 1)
    noise_std = np.sqrt(2.0 * D * dt)
    true_chromosomes = np.zeros((T, 3, N))
    for t in range(1, T):
        true_chromosomes[t] = (
            true_chromosomes[t - 1]
            + drift_vec
            + noise_std * rng.standard_normal((3, N))
        )

    chromosomes = true_chromosomes
    if localization_sigma > 0.0:
        chromosomes = true_chromosomes + localization_sigma * rng.standard_normal(true_chromosomes.shape)

    centrioles = np.zeros((T, 3, 2))
    centrioles[:, 0, 0] = -5.0
    centrioles[:, 0, 1] = 5.0
    return TrimmedCell(
        cell_id="observed_brownian",
        condition="test",
        centrioles=centrioles,
        chromosomes=chromosomes,
        tracked=N,
        dt=dt,
        start_frame=0,
        end_frame=T - 1,
    )


def make_variable_diffusion_cell(
    T: int = 800,
    N: int = 80,
    dt: float = 1.0,
    base_D: float = 0.45,
    slope_D: float = 0.08,
    kappa: float = 0.15,
) -> tuple[TrimmedCell, float, float]:
    """OU-like motion with diffusion varying linearly along the spindle axis."""
    rng = np.random.default_rng(0)
    centrioles = np.zeros((T, 3, 2))
    centrioles[:, 0, 0] = -5.0
    centrioles[:, 0, 1] = 5.0
    chromosomes = np.zeros((T, 3, N))
    chromosomes[0] = rng.normal(0.0, 1.0, size=(3, N))
    for t in range(T - 1):
        axial = chromosomes[t, 0]
        D_axial = np.clip(base_D + slope_D * axial, 0.12, 0.9)
        chromosomes[t + 1, 0] = (
            chromosomes[t, 0]
            - kappa * axial * dt
            + np.sqrt(2.0 * D_axial * dt) * rng.standard_normal(N)
        )
        chromosomes[t + 1, 1] = (
            chromosomes[t, 1]
            + np.sqrt(2.0 * 0.35 * dt) * rng.standard_normal(N)
        )
        chromosomes[t + 1, 2] = (
            chromosomes[t, 2]
            + np.sqrt(2.0 * 0.35 * dt) * rng.standard_normal(N)
        )

    cell = TrimmedCell(
        cell_id="variable_diffusion",
        condition="test",
        centrioles=centrioles,
        chromosomes=chromosomes,
        tracked=N,
        dt=dt,
        start_frame=0,
        end_frame=T - 1,
    )
    return cell, base_D, slope_D


def make_force_corrected_diffusion_cell() -> tuple[TrimmedCell, HatBasis, HatBasis, np.ndarray, float]:
    """Synthetic trajectories with known force so f_corrected can be checked."""
    rng = np.random.default_rng(0)
    T, N, dt, D_true = 300, 18, 0.4, 0.02
    basis_xx = HatBasis(0.0, 8.0, n_basis=4)
    basis_xy = HatBasis(0.0, 10.0, n_basis=4)
    theta_xx = np.zeros(4)
    theta_xy = np.array([0.0, 1.0, 0.0, 0.0])
    theta_true = np.concatenate([theta_xx, theta_xy])

    def kernel_xx(r: np.ndarray) -> np.ndarray:
        return basis_xx.evaluate(r) @ theta_xx

    def kernel_xy(r: np.ndarray) -> np.ndarray:
        return basis_xy.evaluate(r) @ theta_xy

    centrioles = np.zeros((T + 1, 3, 2))
    centrioles[:, 0, 0] = -3.0
    centrioles[:, 0, 1] = 3.0
    x0 = rng.normal(0.0, 0.8, size=(N, 3))
    chromosomes = simulate_trajectories(
        kernel_xx=kernel_xx,
        kernel_xy=kernel_xy,
        centrosome_positions=centrioles,
        x0=x0,
        n_steps=T,
        dt=dt,
        D_x=D_true,
        rng=rng,
    )
    cell = TrimmedCell(
        cell_id="force_corrected",
        condition="test",
        centrioles=centrioles,
        chromosomes=chromosomes,
        tracked=N,
        dt=dt,
        start_frame=0,
        end_frame=T,
    )
    return cell, basis_xx, basis_xy, theta_true, D_true


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


def test_vestergaard_beats_msd_with_localization_noise() -> None:
    D_true = 0.5
    cell = make_observed_diffusion_cell(
        T=2000,
        N=100,
        D=D_true,
        dt=1.0,
        localization_sigma=0.5,
    )
    msd_mean = float(np.nanmean(local_diffusion_estimates([cell], dt=1.0, mode="msd")[0]))
    vestergaard_mean = float(
        np.nanmean(local_diffusion_estimates([cell], dt=1.0, mode="vestergaard")[0])
    )
    assert abs(vestergaard_mean - D_true) < abs(msd_mean - D_true)


def test_weak_noise_beats_msd_with_constant_drift() -> None:
    D_true = 0.5
    cell = make_observed_diffusion_cell(
        T=2000,
        N=100,
        D=D_true,
        dt=1.0,
        drift=(0.4, 0.0, 0.0),
    )
    msd_mean = float(np.nanmean(local_diffusion_estimates([cell], dt=1.0, mode="msd")[0]))
    weak_noise_mean = float(
        np.nanmean(local_diffusion_estimates([cell], dt=1.0, mode="weak_noise")[0])
    )
    assert abs(weak_noise_mean - D_true) < abs(msd_mean - D_true)


def test_f_corrected_beats_msd_when_true_force_is_known() -> None:
    cell, basis_xx, basis_xy, theta_true, D_true = make_force_corrected_diffusion_cell()
    msd_mean = float(np.nanmean(local_diffusion_estimates([cell], dt=cell.dt, mode="msd")[0]))
    f_corrected_mean = float(
        np.nanmean(
            local_diffusion_estimates(
                [cell],
                dt=cell.dt,
                mode="f_corrected",
                fit_result=SimpleNamespace(theta=theta_true),
                basis_xx=basis_xx,
                basis_xy=basis_xy,
            )[0]
        )
    )
    assert abs(f_corrected_mean - D_true) < abs(msd_mean - D_true)
    np.testing.assert_allclose(f_corrected_mean, D_true, rtol=0.1)


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


def test_variable_diffusion_recovers_nonconstant_axial_profile() -> None:
    cell, base_D, slope_D = make_variable_diffusion_cell()
    basis_D = HatBasis(r_min=-4.0, r_max=4.0, n_basis=9)
    result = estimate_diffusion_variable(
        [cell],
        basis_D,
        coord_name="axial",
        dt=cell.dt,
        mode="weak_noise",
    )
    coords = np.linspace(-3.0, 3.0, 25)
    true_D = np.clip(base_D + slope_D * coords, 0.12, 0.9)
    fitted_D = result.evaluate(coords)
    rmse = float(np.sqrt(np.mean((fitted_D - true_D) ** 2)))
    correlation = float(np.corrcoef(true_D, fitted_D)[0, 1])

    assert rmse < 0.15
    assert correlation > 0.8
    assert fitted_D[-1] > fitted_D[0]


def test_unknown_coord_name_raises() -> None:
    cell = make_diffusion_cell()
    basis_D = HatBasis(r_min=0.0, r_max=10.0, n_basis=3)
    with pytest.raises(ValueError, match="Unknown coordinate"):
        estimate_diffusion_variable([cell], basis_D, coord_name="bogus", dt=5.0)
