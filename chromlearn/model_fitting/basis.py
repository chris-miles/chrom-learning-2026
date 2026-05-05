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


@dataclass
class EnvelopedBasis:
    """Wraps a base basis with a smooth steric envelope.

    The envelope ``e(r) = 0.5 * (1 - tanh((r - r0) / w))`` is multiplied
    into every basis column, so the resulting kernel decays smoothly
    to zero past ``r0`` instead of being clipped by a hard cutoff.

    All public attributes (``r_min``, ``r_max``, ``n_basis``) and the
    ``evaluate`` / ``roughness_matrix`` API mirror the inner basis so
    consumers don't need to know an envelope is present.
    """

    inner: "BSplineBasis | HatBasis"
    envelope_r0: float
    envelope_w: float

    def __post_init__(self) -> None:
        if self.envelope_w <= 0:
            raise ValueError("envelope_w must be positive.")

    @property
    def r_min(self) -> float:
        return self.inner.r_min

    @property
    def r_max(self) -> float:
        return self.inner.r_max

    @property
    def n_basis(self) -> int:
        return self.inner.n_basis

    def envelope(self, r: np.ndarray) -> np.ndarray:
        values = np.asarray(r, dtype=np.float64)
        return 0.5 * (1.0 - np.tanh((values - self.envelope_r0) / self.envelope_w))

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        values = np.asarray(r, dtype=np.float64).reshape(-1)
        phi = self.inner.evaluate(values)
        env = self.envelope(values)
        return phi * env[:, np.newaxis]

    def roughness_matrix(self, n_quad: int = 800) -> np.ndarray:
        """Numerical roughness matrix for the enveloped basis.

        Uses central finite differences on a dense quadrature grid so
        the penalty correctly captures both the spline and envelope
        curvature.
        """
        grid = np.linspace(self.inner.r_min, self.inner.r_max, n_quad)
        phi = self.evaluate(grid)
        h = grid[1] - grid[0]
        phi_dd = np.zeros_like(phi)
        phi_dd[1:-1] = (phi[2:] - 2.0 * phi[1:-1] + phi[:-2]) / (h * h)
        integrand = phi_dd[:, :, np.newaxis] * phi_dd[:, np.newaxis, :]
        roughness = trapezoid(integrand, grid, axis=0)
        return 0.5 * (roughness + roughness.T)


def make_basis_with_envelope(
    base_class,
    r_min: float,
    r_max: float,
    n_basis: int,
    envelope_r0: float | None = None,
    envelope_w: float | None = None,
):
    """Build a basis, optionally wrapped in an EnvelopedBasis.

    Pass ``envelope_r0`` and ``envelope_w`` together to get a smooth
    short-range basis; pass neither for the bare basis.
    """
    inner = base_class(r_min, r_max, n_basis)
    if envelope_r0 is None and envelope_w is None:
        return inner
    if envelope_r0 is None or envelope_w is None:
        raise ValueError(
            "envelope_r0 and envelope_w must both be set or both be None."
        )
    return EnvelopedBasis(inner=inner, envelope_r0=float(envelope_r0),
                          envelope_w=float(envelope_w))
