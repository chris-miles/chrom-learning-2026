from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import BSpline


@dataclass
class BSplineBasis:
    """Clamped cubic B-spline basis on a bounded interval."""

    r_min: float
    r_max: float
    n_basis: int
    degree: int = 3

    def __post_init__(self) -> None:
        if self.r_max <= self.r_min:
            raise ValueError("r_max must be larger than r_min.")
        if self.n_basis < self.degree + 1:
            raise ValueError(
                f"n_basis must be at least {self.degree + 1} for degree-{self.degree} splines."
            )

        n_interior = self.n_basis - self.degree - 1
        if n_interior > 0:
            interior = np.linspace(self.r_min, self.r_max, n_interior + 2)[1:-1]
        else:
            interior = np.array([], dtype=float)
        self.knots = np.concatenate(
            [
                np.repeat(self.r_min, self.degree + 1),
                interior,
                np.repeat(self.r_max, self.degree + 1),
            ]
        )
        self._splines = []
        for index in range(self.n_basis):
            coeffs = np.zeros(self.n_basis)
            coeffs[index] = 1.0
            self._splines.append(
                BSpline(self.knots, coeffs, self.degree, extrapolate=False)
            )

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """Evaluate all basis functions at distances *r*.

        Args:
            r: 1-D array of distances.

        Returns:
            Array of shape ``(len(r), n_basis)``.  Values outside
            ``[r_min, r_max]`` are clamped to the nearest boundary so
            that the force at the domain edge extends smoothly.
        """
        values = np.asarray(r, dtype=np.float64).reshape(-1)
        n = values.size
        result = np.zeros((n, self.n_basis), dtype=np.float64)
        if n == 0:
            return result
        clamped = np.clip(values, self.r_min, self.r_max)
        sparse_result = BSpline.design_matrix(clamped, self.knots, self.degree)
        result[:] = np.nan_to_num(sparse_result.toarray(), nan=0.0)
        return result

    def roughness_matrix(self, n_quad: int = 800) -> np.ndarray:
        """Roughness penalty matrix R.

        ``R[i, j] = integral of phi_i''(r) * phi_j''(r) dr`` over
        ``[r_min, r_max]``, computed via trapezoidal quadrature.

        Returns:
            Symmetric positive-semidefinite matrix of shape
            ``(n_basis, n_basis)``.
        """
        grid = np.linspace(self.r_min, self.r_max, n_quad)
        second_derivatives = np.zeros((grid.size, self.n_basis))
        for index, spline in enumerate(self._splines):
            second = spline.derivative(2)(grid)
            second_derivatives[:, index] = np.nan_to_num(second, nan=0.0)
        integrand = (
            second_derivatives[:, :, np.newaxis] * second_derivatives[:, np.newaxis, :]
        )
        roughness = trapezoid(integrand, grid, axis=0)
        return 0.5 * (roughness + roughness.T)


@dataclass
class HatBasis:
    """Piecewise-linear hat basis on a bounded interval."""

    r_min: float
    r_max: float
    n_basis: int

    def __post_init__(self) -> None:
        if self.r_max <= self.r_min:
            raise ValueError("r_max must be larger than r_min.")
        if self.n_basis < 1:
            raise ValueError("n_basis must be positive.")
        self.centers = np.linspace(self.r_min, self.r_max, self.n_basis)
        self.width = (
            (self.r_max - self.r_min) / (self.n_basis - 1) if self.n_basis > 1 else self.r_max - self.r_min
        )

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """Evaluate hat basis at distances *r*.

        Values outside ``[r_min, r_max]`` are clamped to the nearest
        boundary so that the force at the domain edge extends smoothly.
        """
        values = np.asarray(r, dtype=np.float64).reshape(-1)
        clamped = np.clip(values, self.r_min, self.r_max)
        if self.n_basis == 1:
            return np.ones((clamped.size, 1), dtype=np.float64)
        result = np.maximum(0.0, 1.0 - np.abs(clamped[:, np.newaxis] - self.centers[np.newaxis, :]) / self.width)
        return result

    def roughness_matrix(self) -> np.ndarray:
        return np.zeros((self.n_basis, self.n_basis), dtype=np.float64)
