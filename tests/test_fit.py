import numpy as np

from chromlearn.model_fitting.fit import estimate_diffusion, fit_kernels


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
