import numpy as np

from chromlearn.model_fitting.validate import (
    kernel_recovery_error,
    one_step_prediction_error,
    residual_diagnostics,
)


def test_one_step_prediction_error_perfect_fit() -> None:
    V = np.array([1.0, 2.0, 3.0])
    G = np.eye(3)
    theta = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(one_step_prediction_error(V, G, theta), 0.0, atol=1e-15)


def test_residual_diagnostics_returns_dict() -> None:
    diagnostics = residual_diagnostics(np.random.default_rng(42).standard_normal(300))
    for key in ["mean", "std", "skewness", "kurtosis"]:
        assert key in diagnostics


def test_kernel_recovery_error() -> None:
    r = np.linspace(0, 10, 100)
    true_values = np.sin(r)
    fitted_values = np.sin(r) + 0.01
    assert kernel_recovery_error(r, true_values, fitted_values) < 0.02
