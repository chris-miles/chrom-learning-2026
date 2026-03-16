import numpy as np

from chromlearn.io.trajectory import TrimmedCell
from chromlearn.model_fitting import FitConfig
from chromlearn.model_fitting.basis import HatBasis
from chromlearn.model_fitting.fit import (
    bootstrap_kernels,
    cross_validate,
    estimate_diffusion,
    fit_kernels,
    fit_model,
)
from chromlearn.model_fitting.features import build_design_matrix
from chromlearn.model_fitting.simulate import add_localization_noise, simulate_trajectories


def make_synthetic_inference_cell(
    T: int = 160,
    N: int = 16,
    dt: float = 0.1,
    D_x: float = 0.02,
    localization_sigma: float = 0.0,
    seed: int = 0,
) -> tuple[TrimmedCell, HatBasis, HatBasis, np.ndarray]:
    rng = np.random.default_rng(seed)
    basis_xx = HatBasis(0.0, 8.0, n_basis=4)
    basis_xy = HatBasis(0.0, 10.0, n_basis=4)
    theta_xx = np.array([0.0, -0.2, 0.0, 0.0])
    theta_xy = np.array([0.0, 0.4, 0.0, 0.0])
    theta_true = np.concatenate([theta_xx, theta_xy])

    def kernel_xx(r: np.ndarray) -> np.ndarray:
        return basis_xx.evaluate(r) @ theta_xx

    def kernel_xy(r: np.ndarray) -> np.ndarray:
        return basis_xy.evaluate(r) @ theta_xy

    centrioles = np.zeros((T + 1, 3, 2))
    centrioles[:, 0, 0] = -3.0
    centrioles[:, 0, 1] = 3.0
    x0 = rng.normal(0.0, 1.0, size=(N, 3))
    partners = centrioles.transpose(2, 0, 1)  # (2, T+1, 3)
    chromosomes = simulate_trajectories(
        kernel_xx=kernel_xx,
        kernel_xy=kernel_xy,
        partner_positions=partners,
        x0=x0,
        n_steps=T,
        dt=dt,
        D_x=D_x,
        rng=rng,
    )
    if localization_sigma > 0.0:
        chromosomes = add_localization_noise(chromosomes, sigma=localization_sigma, rng=rng)

    cell = TrimmedCell(
        cell_id="synthetic_fit",
        condition="test",
        centrioles=centrioles,
        chromosomes=chromosomes,
        tracked=N,
        dt=dt,
        start_frame=0,
        end_frame=T,
    )
    return cell, basis_xx, basis_xy, theta_true


def test_fit_kernels_recovers_known_theta() -> None:
    rng = np.random.default_rng(42)
    n_obs, n_basis = 500, 6
    theta_true = rng.standard_normal(n_basis)
    G = rng.standard_normal((n_obs, n_basis))
    V = G @ theta_true + 0.01 * rng.standard_normal(n_obs)
    result = fit_kernels(G, V, lambda_ridge=1e-6, lambda_rough=0.0, R=np.zeros((n_basis, n_basis)))
    np.testing.assert_allclose(result.theta, theta_true, atol=0.05)


def test_fit_kernels_with_roughness() -> None:
    rng = np.random.default_rng(42)
    G = rng.standard_normal((200, 5))
    V = rng.standard_normal(200)
    result = fit_kernels(G, V, lambda_ridge=0.01, lambda_rough=0.01, R=np.eye(5))
    assert result.theta.shape == (5,)


def test_fit_kernels_result_has_residuals() -> None:
    rng = np.random.default_rng(42)
    G = rng.standard_normal((100, 4))
    V = rng.standard_normal(100)
    result = fit_kernels(G, V, lambda_ridge=0.01, lambda_rough=0.0, R=np.zeros((4, 4)))
    assert result.residuals.shape == (100,)


def test_estimate_diffusion() -> None:
    rng = np.random.default_rng(42)
    D_true = 0.5
    dt = 5.0
    d = 3
    n_obs = 10_000
    V = np.sqrt(2 * D_true / dt) * rng.standard_normal(n_obs * d)
    G = np.zeros((n_obs * d, 1))
    theta = np.array([0.0])
    D_hat = estimate_diffusion(V, G, theta, dt=dt, d=d)
    np.testing.assert_allclose(D_hat, D_true, rtol=0.1)


def test_end_to_end_recovers_known_kernels_on_synthetic_data() -> None:
    cell, basis_xx, basis_xy, theta_true = make_synthetic_inference_cell()
    G, V = build_design_matrix([cell], basis_xx, basis_xy)
    result = fit_kernels(
        G,
        V,
        lambda_ridge=1e-6,
        lambda_rough=0.0,
        R=np.zeros((theta_true.size, theta_true.size)),
    )

    rmse = float(np.sqrt(np.mean((result.theta - theta_true) ** 2)))
    assert rmse < 0.1
    np.testing.assert_allclose(result.theta[[1, 5]], theta_true[[1, 5]], atol=0.1)


def test_shifted_basis_eval_modes_reduce_localization_noise_bias() -> None:
    cell, basis_xx, basis_xy, theta_true = make_synthetic_inference_cell(
        T=120,
        N=14,
        localization_sigma=0.25,
    )
    roughness = np.zeros((theta_true.size, theta_true.size))
    errors = {}
    for mode in ("ito", "ito_shift", "strato"):
        G, V = build_design_matrix([cell], basis_xx, basis_xy, basis_eval_mode=mode)
        result = fit_kernels(G, V, lambda_ridge=1e-6, lambda_rough=0.0, R=roughness)
        errors[mode] = float(np.sqrt(np.mean((result.theta - theta_true) ** 2)))

    assert errors["ito_shift"] < errors["ito"] * 0.5
    assert errors["strato"] < errors["ito"] * 0.5
    assert errors["ito_shift"] < 0.2
    assert errors["strato"] < 0.2


def test_cross_validate_and_bootstrap_support_shifted_basis_eval_mode() -> None:
    cells = [
        make_synthetic_inference_cell(T=90, N=12, dt=0.15, seed=seed)[0]
        for seed in range(4)
    ]
    _, basis_xx, basis_xy, theta_true = make_synthetic_inference_cell(T=90, N=12, dt=0.15)

    config = FitConfig(
        topology="poles_and_chroms",
        n_basis_xx=4, n_basis_xy=4,
        r_min_xx=0.0, r_max_xx=8.0,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat",
        lambda_ridge=1e-6, lambda_rough=0.0,
        basis_eval_mode="ito_shift",
        dt=0.15,
    )
    cv_result = cross_validate(cells, config)
    G, V = build_design_matrix(
        cells, basis_xx, basis_xy,
        basis_eval_mode="ito_shift",
        topology="poles_and_chroms",
    )
    baseline_error = float(np.mean(V**2))
    assert np.all(np.isfinite(cv_result.held_out_errors))
    assert cv_result.mean_error < baseline_error

    bootstrap_result = bootstrap_kernels(
        cells, config, n_boot=8, rng=np.random.default_rng(0),
    )
    assert bootstrap_result.theta_samples.shape == (8, theta_true.size)
    assert np.all(np.isfinite(bootstrap_result.theta_std))
    bootstrap_rmse = float(np.sqrt(np.mean((bootstrap_result.theta_mean - theta_true) ** 2)))
    assert bootstrap_rmse < 0.08


def test_fit_model_poles_only_no_xx() -> None:
    """fit_model with topology='poles' produces no chromosome-chromosome kernel."""
    cell, _, _, _ = make_synthetic_inference_cell()
    config = FitConfig(
        topology="poles",
        n_basis_xx=10, n_basis_xy=4,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat",
        lambda_ridge=1e-6, lambda_rough=0.0,
        dt=0.1,
    )
    model = fit_model([cell], config)
    assert model.n_basis_xx == 0
    assert model.basis_xx is None
    assert model.theta.shape[0] == 4


def test_fit_model_poles_and_chroms() -> None:
    """fit_model with topology='poles_and_chroms' includes both kernels."""
    cell, _, _, _ = make_synthetic_inference_cell()
    config = FitConfig(
        topology="poles_and_chroms",
        n_basis_xx=4, n_basis_xy=4,
        r_min_xx=0.0, r_max_xx=8.0,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat",
        lambda_ridge=1e-6, lambda_rough=0.0,
        dt=0.1,
    )
    model = fit_model([cell], config)
    assert model.n_basis_xx == 4
    assert model.basis_xx is not None
    assert model.theta.shape[0] == 8


def test_fit_model_preserves_topology() -> None:
    """fit_model stores config.topology in the returned FittedModel."""
    cell, _, _, _ = make_synthetic_inference_cell()
    for topo in ("poles", "center", "poles_and_chroms", "center_and_chroms"):
        config = FitConfig(
            topology=topo,
            n_basis_xx=4, n_basis_xy=4,
            r_min_xx=0.0, r_max_xx=8.0,
            r_min_xy=0.0, r_max_xy=10.0,
            basis_type="hat",
            lambda_ridge=1e-6, lambda_rough=0.0,
            dt=0.1,
        )
        model = fit_model([cell], config)
        assert model.topology == topo, f"Expected topology={topo}, got {model.topology}"


def test_fit_model_vestergaard_diffusion() -> None:
    """fit_model honours diffusion_mode='vestergaard'."""
    cell, _, _, _ = make_synthetic_inference_cell()
    config_msd = FitConfig(
        topology="poles", n_basis_xy=4,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat", lambda_ridge=1e-6, lambda_rough=0.0,
        dt=0.1, diffusion_mode="msd",
    )
    config_vest = FitConfig(
        topology="poles", n_basis_xy=4,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat", lambda_ridge=1e-6, lambda_rough=0.0,
        dt=0.1, diffusion_mode="vestergaard",
    )
    model_msd = fit_model([cell], config_msd)
    model_vest = fit_model([cell], config_vest)
    # Both should produce positive D, but they use different estimators
    assert model_msd.D_x > 0
    assert model_vest.D_x > 0
    # They should differ (different estimators on the same data)
    assert model_msd.D_x != model_vest.D_x


def test_fit_model_variable_diffusion() -> None:
    """fit_model with D_variable=True returns a diffusion_model."""
    cell, _, _, _ = make_synthetic_inference_cell()
    config = FitConfig(
        topology="poles", n_basis_xy=4,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat", lambda_ridge=1e-6, lambda_rough=0.0,
        dt=0.1, D_variable=True, n_basis_D=4,
        r_min_D=-5.0, r_max_D=5.0,
    )
    model = fit_model([cell], config)
    assert model.diffusion_model is not None
    assert model.diffusion_model.d_coeffs.shape == (4,)


def test_cross_validate_poles_only() -> None:
    """CV works with topology='poles' (basis_xx=None)."""
    cells = [
        make_synthetic_inference_cell(T=90, N=12, dt=0.15, seed=s)[0]
        for s in range(4)
    ]
    config = FitConfig(
        topology="poles",
        n_basis_xy=4,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat",
        lambda_ridge=1e-6, lambda_rough=0.0,
        dt=0.15,
    )
    cv = cross_validate(cells, config)
    assert np.all(np.isfinite(cv.held_out_errors))
    assert cv.mean_error > 0
