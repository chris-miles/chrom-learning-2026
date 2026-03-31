import numpy as np

from chromlearn.model_fitting.basis import BSplineBasis, HatBasis


class TestBSplineBasis:
    def test_shape(self) -> None:
        basis = BSplineBasis(r_min=0.0, r_max=10.0, n_basis=8)
        values = basis.evaluate(np.linspace(0, 10, 50))
        assert values.shape == (50, 8)

    def test_partition_of_unity(self) -> None:
        basis = BSplineBasis(r_min=0.0, r_max=10.0, n_basis=10)
        values = basis.evaluate(np.linspace(0.5, 9.5, 100))
        np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=0.05)

    def test_nonnegative(self) -> None:
        basis = BSplineBasis(r_min=0.0, r_max=10.0, n_basis=8)
        values = basis.evaluate(np.linspace(0, 10, 100))
        assert np.all(values >= -1e-15)

    def test_outside_support_is_clamped(self) -> None:
        basis = BSplineBasis(r_min=1.0, r_max=5.0, n_basis=6)
        r_out = np.array([0.0, 0.5, 5.5, 10.0])
        values = basis.evaluate(r_out)
        # Out-of-domain values should clamp to the nearest boundary
        at_min = basis.evaluate(np.array([1.0]))
        at_max = basis.evaluate(np.array([5.0]))
        np.testing.assert_allclose(values[0], at_min[0], atol=1e-15)
        np.testing.assert_allclose(values[1], at_min[0], atol=1e-15)
        np.testing.assert_allclose(values[2], at_max[0], atol=1e-15)
        np.testing.assert_allclose(values[3], at_max[0], atol=1e-15)

    def test_roughness_matrix_shape(self) -> None:
        basis = BSplineBasis(r_min=0.0, r_max=10.0, n_basis=8)
        roughness = basis.roughness_matrix()
        assert roughness.shape == (8, 8)

    def test_roughness_matrix_symmetric_positive_semidefinite(self) -> None:
        basis = BSplineBasis(r_min=0.0, r_max=10.0, n_basis=8)
        roughness = basis.roughness_matrix()
        np.testing.assert_allclose(roughness, roughness.T, atol=1e-10)
        eigenvalues = np.linalg.eigvalsh(roughness)
        assert np.all(eigenvalues >= -1e-8)

    def test_linear_kernel_has_small_roughness(self) -> None:
        basis = BSplineBasis(r_min=0.0, r_max=10.0, n_basis=8)
        roughness = basis.roughness_matrix()
        grid = np.linspace(0, 10, 200)
        phi = basis.evaluate(grid)
        theta, *_ = np.linalg.lstsq(phi, grid, rcond=None)
        assert float(theta @ roughness @ theta) < 0.1


class TestHatBasis:
    def test_shape(self) -> None:
        basis = HatBasis(r_min=0.0, r_max=10.0, n_basis=5)
        values = basis.evaluate(np.linspace(0, 10, 50))
        assert values.shape == (50, 5)

    def test_partition_of_unity(self) -> None:
        basis = HatBasis(r_min=0.0, r_max=10.0, n_basis=5)
        values = basis.evaluate(np.linspace(0, 10, 100))
        np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-10)

    def test_nonnegative(self) -> None:
        basis = HatBasis(r_min=0.0, r_max=10.0, n_basis=5)
        values = basis.evaluate(np.linspace(0, 10, 100))
        assert np.all(values >= -1e-15)
