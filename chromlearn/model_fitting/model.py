from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chromlearn.model_fitting.basis import BSplineBasis, HatBasis


@dataclass
class FittedModel:
    """Container for a fitted pairwise-kernel model."""

    theta: np.ndarray
    n_basis_xx: int
    n_basis_xy: int
    basis_xx: BSplineBasis | HatBasis
    basis_xy: BSplineBasis | HatBasis
    D_x: float
    dt: float
    metadata: dict | None = None

    @property
    def theta_xx(self) -> np.ndarray:
        return self.theta[: self.n_basis_xx]

    @property
    def theta_xy(self) -> np.ndarray:
        return self.theta[self.n_basis_xx :]

    def evaluate_kernel(self, kernel: str, r: np.ndarray) -> np.ndarray:
        values = np.asarray(r, dtype=np.float64)
        if kernel == "xx":
            return self.basis_xx.evaluate(values) @ self.theta_xx
        if kernel == "xy":
            return self.basis_xy.evaluate(values) @ self.theta_xy
        raise ValueError(f"Unknown kernel '{kernel}'. Use 'xx' or 'xy'.")

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        np.savez(
            output_path,
            theta=self.theta,
            n_basis_xx=self.n_basis_xx,
            n_basis_xy=self.n_basis_xy,
            D_x=self.D_x,
            dt=self.dt,
            metadata=np.array(self.metadata, dtype=object),
            basis_xx_type="bspline" if isinstance(self.basis_xx, BSplineBasis) else "hat",
            basis_xx_r_min=self.basis_xx.r_min,
            basis_xx_r_max=self.basis_xx.r_max,
            basis_xx_n_basis=self.basis_xx.n_basis,
            basis_xy_type="bspline" if isinstance(self.basis_xy, BSplineBasis) else "hat",
            basis_xy_r_min=self.basis_xy.r_min,
            basis_xy_r_max=self.basis_xy.r_max,
            basis_xy_n_basis=self.basis_xy.n_basis,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FittedModel":
        input_path = Path(path)
        data = np.load(input_path, allow_pickle=True)

        def make_basis(kind: str, r_min: float, r_max: float, n_basis: int):
            if kind == "bspline":
                return BSplineBasis(float(r_min), float(r_max), int(n_basis))
            return HatBasis(float(r_min), float(r_max), int(n_basis))

        metadata = data["metadata"].item() if data["metadata"].shape == () else None
        return cls(
            theta=np.asarray(data["theta"], dtype=np.float64),
            n_basis_xx=int(data["n_basis_xx"]),
            n_basis_xy=int(data["n_basis_xy"]),
            basis_xx=make_basis(
                str(data["basis_xx_type"]),
                float(data["basis_xx_r_min"]),
                float(data["basis_xx_r_max"]),
                int(data["basis_xx_n_basis"]),
            ),
            basis_xy=make_basis(
                str(data["basis_xy_type"]),
                float(data["basis_xy_r_min"]),
                float(data["basis_xy_r_max"]),
                int(data["basis_xy_n_basis"]),
            ),
            D_x=float(data["D_x"]),
            dt=float(data["dt"]),
            metadata=metadata,
        )
