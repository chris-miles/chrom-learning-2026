from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from chromlearn.io.trajectory import TrimmedCell, pole_center
from chromlearn.model_fitting.basis import BSplineBasis, HatBasis


# ---------------------------------------------------------------------------
# 1. Coordinate maps
# ---------------------------------------------------------------------------


def _spindle_axis_unit(cell: TrimmedCell) -> tuple[np.ndarray, np.ndarray]:
    """Return the spindle center (T,3) and axis unit vector (T,3)."""
    center = pole_center(cell)  # (T, 3)
    axis = cell.centrioles[:, :, 1] - cell.centrioles[:, :, 0]  # (T, 3)
    norms = np.linalg.norm(axis, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    axis_unit = axis / safe_norms
    return center, axis_unit


def _coord_axial(positions: np.ndarray, cell: TrimmedCell) -> np.ndarray:
    """Signed projection onto the spindle axis.

    Args:
        positions: Array of shape ``(T, 3, N)``.
        cell: TrimmedCell providing centrosome positions.

    Returns:
        Array of shape ``(T, N)``.
    """
    center, axis_unit = _spindle_axis_unit(cell)
    # delta: (T, N, 3) — displacement of each particle from spindle center
    delta = np.moveaxis(positions, 2, 1) - center[:, np.newaxis, :]
    axial = np.sum(delta * axis_unit[:, np.newaxis, :], axis=2)
    return axial


def _coord_radial(positions: np.ndarray, cell: TrimmedCell) -> np.ndarray:
    """Perpendicular distance from the spindle axis.

    Args:
        positions: Array of shape ``(T, 3, N)``.
        cell: TrimmedCell providing centrosome positions.

    Returns:
        Array of shape ``(T, N)``.
    """
    center, axis_unit = _spindle_axis_unit(cell)
    delta = np.moveaxis(positions, 2, 1) - center[:, np.newaxis, :]
    axial = np.sum(delta * axis_unit[:, np.newaxis, :], axis=2)
    perpendicular = delta - axial[:, :, np.newaxis] * axis_unit[:, np.newaxis, :]
    radial = np.linalg.norm(perpendicular, axis=2)
    return radial


def _coord_distance(positions: np.ndarray, cell: TrimmedCell) -> np.ndarray:
    """Distance from the spindle center (Euclidean norm of axial + radial).

    Args:
        positions: Array of shape ``(T, 3, N)``.
        cell: TrimmedCell providing centrosome positions.

    Returns:
        Array of shape ``(T, N)``.
    """
    center, _axis_unit = _spindle_axis_unit(cell)
    delta = np.moveaxis(positions, 2, 1) - center[:, np.newaxis, :]
    return np.linalg.norm(delta, axis=2)


COORDINATE_MAPS: dict[str, Callable[[np.ndarray, TrimmedCell], np.ndarray]] = {
    "axial": _coord_axial,
    "radial": _coord_radial,
    "distance": _coord_distance,
}
"""Mapping from coordinate name to callable ``(positions, cell) -> (T, N)``."""


# ---------------------------------------------------------------------------
# 2. Local diffusion estimators
# ---------------------------------------------------------------------------


def _predicted_force(
    cell: TrimmedCell,
    time_index: int,
    fit_result,
    basis_xx,
    basis_xy,
) -> np.ndarray:
    """Compute predicted force vectors at one timepoint for all chromosomes.

    Returns:
        Array of shape ``(N, 3)`` with the predicted force (velocity) for each
        chromosome at the given timepoint.
    """
    chromosomes = cell.chromosomes  # (T, 3, N)
    centrioles = cell.centrioles    # (T, 3, 2)
    n_chromosomes = chromosomes.shape[2]
    theta = fit_result.theta
    n_xx = basis_xx.n_basis

    positions = chromosomes[time_index].T   # (N, 3)
    poles = centrioles[time_index].T        # (2, 3)

    forces = np.full((n_chromosomes, 3), np.nan, dtype=np.float64)
    for i in range(n_chromosomes):
        pos_i = positions[i]
        if np.any(np.isnan(pos_i)):
            continue

        # chromosome-chromosome interactions
        force_vec = np.zeros(3, dtype=np.float64)
        for j in range(n_chromosomes):
            if j == i:
                continue
            pos_j = positions[j]
            if np.any(np.isnan(pos_j)):
                continue
            delta = pos_j - pos_i
            dist = float(np.linalg.norm(delta))
            if dist <= 1e-12:
                continue
            direction = delta / dist
            phi = basis_xx.evaluate(np.array([dist]))[0]  # (n_basis_xx,)
            force_vec += direction * (phi @ theta[:n_xx])

        # chromosome-pole interactions
        for p in range(2):
            delta = poles[p] - pos_i
            dist = float(np.linalg.norm(delta))
            if dist <= 1e-12:
                continue
            direction = delta / dist
            phi = basis_xy.evaluate(np.array([dist]))[0]  # (n_basis_xy,)
            force_vec += direction * (phi @ theta[n_xx:])

        forces[i] = force_vec

    return forces


def local_diffusion_estimates(
    cells: list[TrimmedCell],
    dt: float,
    mode: str,
    fit_result=None,
    basis_xx=None,
    basis_xy=None,
) -> list[np.ndarray]:
    """Compute per-particle, per-timepoint local diffusion estimates.

    Args:
        cells: List of trimmed cell trajectories.
        dt: Time step in seconds.
        mode: One of ``"msd"``, ``"vestergaard"``, ``"weak_noise"``,
            or ``"f_corrected"``.
        fit_result: Required for ``"f_corrected"`` mode.  A :class:`FitResult`
            (or :class:`FittedModel`) with ``theta`` attribute.
        basis_xx: Basis for chromosome-chromosome kernel (required for
            ``"f_corrected"``).
        basis_xy: Basis for chromosome-pole kernel (required for
            ``"f_corrected"``).

    Returns:
        List (one per cell) of arrays with shape ``(T_valid, N)`` containing
        local scalar diffusion estimates.  ``T_valid`` depends on the mode.

    Raises:
        ValueError: If *mode* is unknown or ``"f_corrected"`` dependencies
            are missing.
    """
    d = 3  # spatial dimension

    if mode == "f_corrected" and (fit_result is None or basis_xx is None or basis_xy is None):
        raise ValueError(
            "f_corrected mode requires fit_result, basis_xx, and basis_xy."
        )

    if mode not in ("msd", "vestergaard", "weak_noise", "f_corrected"):
        raise ValueError(
            f"Unknown mode '{mode}'. "
            "Expected 'msd', 'vestergaard', 'weak_noise', or 'f_corrected'."
        )

    results: list[np.ndarray] = []

    for cell in cells:
        chromosomes = cell.chromosomes  # (T, 3, N)
        T, _, N = chromosomes.shape

        if mode == "msd":
            # dX[t] = X(t+1) - X(t), shape (T-1, 3, N)
            dX = np.diff(chromosomes, axis=0)
            # |dX|^2 summed over spatial dims -> (T-1, N)
            dX_sq = np.sum(dX ** 2, axis=1)
            D = dX_sq / (2.0 * d * dt)
            # Propagate NaN: if either position is NaN, dX is NaN -> D is NaN
            results.append(D)

        elif mode == "vestergaard":
            # dX[t] = X(t+1) - X(t) for t in [0, T-2] -> (T-1, 3, N)
            dX_all = np.diff(chromosomes, axis=0)
            if T < 3:
                results.append(np.full((0, N), np.nan, dtype=np.float64))
                continue
            # dX_plus[t] = dX[t] for t in [1, T-2] -> index 1: in dX_all
            dX_plus = dX_all[1:]       # (T-2, 3, N)
            dX_minus = dX_all[:-1]     # (T-2, 3, N)
            sq_plus = np.sum(dX_plus ** 2, axis=1)
            sq_minus = np.sum(dX_minus ** 2, axis=1)
            dot_term = np.sum(dX_plus * dX_minus, axis=1)
            D = (sq_plus + sq_minus) / (4.0 * d * dt) + dot_term / (2.0 * d * dt)
            results.append(D)

        elif mode == "weak_noise":
            dX_all = np.diff(chromosomes, axis=0)
            if T < 3:
                results.append(np.full((0, N), np.nan, dtype=np.float64))
                continue
            # second difference: dX[t] - dX[t-1] = X(t+1) - 2*X(t) + X(t-1)
            ddX = dX_all[1:] - dX_all[:-1]  # (T-2, 3, N)
            ddX_sq = np.sum(ddX ** 2, axis=1)
            D = ddX_sq / (4.0 * d * dt)
            results.append(D)

        elif mode == "f_corrected":
            # D[t,i] = |dX[t] - F_pred[t]*dt|^2 / (2*d*dt)
            dX_all = np.diff(chromosomes, axis=0)  # (T-1, 3, N)
            D = np.full((T - 1, N), np.nan, dtype=np.float64)
            for t in range(T - 1):
                forces = _predicted_force(cell, t, fit_result, basis_xx, basis_xy)
                # forces: (N, 3), dX_all[t]: (3, N)
                residual = dX_all[t] - (forces.T * dt)  # (3, N)
                D[t] = np.sum(residual ** 2, axis=0) / (2.0 * d * dt)
            results.append(D)

    return results


# ---------------------------------------------------------------------------
# 3. Variable diffusion fitting
# ---------------------------------------------------------------------------


@dataclass
class DiffusionResult:
    """Result of variable diffusion estimation.

    Stores the basis-function coefficients for D(coordinate) along with the
    basis object and scalar summary.
    """

    d_coeffs: np.ndarray
    """Fitted coefficients of shape ``(n_basis_D,)``."""

    basis_D: BSplineBasis | HatBasis
    """Basis used to expand D as a function of the chosen coordinate."""

    coord_name: str
    """Name of the coordinate map (``"axial"``, ``"radial"``, or ``"distance"``)."""

    D_scalar: float
    """Trajectory-averaged scalar diffusion coefficient for reference."""

    def evaluate(self, coord_values: np.ndarray) -> np.ndarray:
        """Evaluate D(coord) at arbitrary coordinate values.

        Args:
            coord_values: 1-D array of coordinate values.

        Returns:
            1-D array of diffusion values, same length as *coord_values*.
        """
        phi = self.basis_D.evaluate(coord_values)
        return phi @ self.d_coeffs


def estimate_diffusion_variable(
    cells: list[TrimmedCell],
    basis_D: BSplineBasis | HatBasis,
    coord_name: str,
    dt: float,
    mode: str = "vestergaard",
    fit_result=None,
    basis_xx=None,
    basis_xy=None,
    lambda_ridge: float = 1e-3,
) -> DiffusionResult:
    """Fit a spatially-varying diffusion coefficient D(coordinate).

    1. Computes local D estimates via :func:`local_diffusion_estimates`.
    2. Evaluates the chosen coordinate map on each cell's chromosomes.
    3. Assembles valid (non-NaN) pairs ``(coordinate, D_local)`` and solves
       ridge regression in the basis ``basis_D``.

    Args:
        cells: List of trimmed cell trajectories.
        basis_D: Basis for expanding D as a function of coordinate.
        coord_name: Which coordinate to use (``"axial"``, ``"radial"``, or
            ``"distance"``).
        dt: Time step in seconds.
        mode: Local estimator mode; see :func:`local_diffusion_estimates`.
        fit_result: Needed for ``"f_corrected"`` mode.
        basis_xx: Needed for ``"f_corrected"`` mode.
        basis_xy: Needed for ``"f_corrected"`` mode.
        lambda_ridge: Ridge regularisation strength for the D regression.

    Returns:
        :class:`DiffusionResult` with fitted coefficients and scalar summary.

    Raises:
        ValueError: If *coord_name* is not a recognised coordinate map.
    """
    if coord_name not in COORDINATE_MAPS:
        raise ValueError(
            f"Unknown coordinate name '{coord_name}'. "
            f"Expected one of {list(COORDINATE_MAPS.keys())}."
        )

    coord_map = COORDINATE_MAPS[coord_name]

    # Step 1: local D estimates
    D_locals = local_diffusion_estimates(
        cells, dt, mode,
        fit_result=fit_result,
        basis_xx=basis_xx,
        basis_xy=basis_xy,
    )

    # Step 2–3: collect valid (coordinate, D) pairs
    all_coords: list[np.ndarray] = []
    all_D: list[np.ndarray] = []

    for cell, D_cell in zip(cells, D_locals):
        chromosomes = cell.chromosomes  # (T, 3, N)
        T_full = chromosomes.shape[0]
        T_valid = D_cell.shape[0]

        # Determine time offset: msd and f_corrected use t in [0, T-2],
        # vestergaard and weak_noise use t in [1, T-2].
        if mode in ("vestergaard", "weak_noise"):
            t_start = 1
        else:
            t_start = 0

        # Coordinate values for the valid time window
        positions_window = chromosomes[t_start : t_start + T_valid]  # (T_valid, 3, N)
        coords = coord_map(positions_window, cell.__class__(
            cell_id=cell.cell_id,
            condition=cell.condition,
            centrioles=cell.centrioles[t_start : t_start + T_valid],
            chromosomes=positions_window,
            tracked=cell.tracked,
            dt=cell.dt,
            start_frame=cell.start_frame + t_start,
            end_frame=cell.start_frame + t_start + T_valid - 1,
        ))  # (T_valid, N)

        # Flatten and filter NaN
        flat_D = D_cell.ravel()
        flat_coord = coords.ravel()
        valid = np.isfinite(flat_D) & np.isfinite(flat_coord)
        all_D.append(flat_D[valid])
        all_coords.append(flat_coord[valid])

    coords_cat = np.concatenate(all_coords) if all_coords else np.array([], dtype=np.float64)
    D_cat = np.concatenate(all_D) if all_D else np.array([], dtype=np.float64)

    # Step 6: scalar D as mean of all valid local estimates
    D_scalar = float(np.mean(D_cat)) if D_cat.size > 0 else 0.0

    # Step 4–5: ridge regression  D_local ~ Phi @ d_coeffs
    n_basis = basis_D.n_basis
    if coords_cat.size == 0:
        return DiffusionResult(
            d_coeffs=np.zeros(n_basis, dtype=np.float64),
            basis_D=basis_D,
            coord_name=coord_name,
            D_scalar=D_scalar,
        )

    Phi = basis_D.evaluate(coords_cat)  # (K, n_basis)
    A = Phi.T @ Phi + lambda_ridge * np.eye(n_basis)
    b = Phi.T @ D_cat
    d_coeffs = np.linalg.solve(A, b)

    return DiffusionResult(
        d_coeffs=d_coeffs,
        basis_D=basis_D,
        coord_name=coord_name,
        D_scalar=D_scalar,
    )
