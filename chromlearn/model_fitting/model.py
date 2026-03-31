from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from chromlearn.model_fitting.basis import BSplineBasis, HatBasis
from chromlearn.model_fitting.diffusion import DiffusionResult


@dataclass
class FittedModel:
    """Container for a fitted pairwise-kernel model.

    Attributes:
        theta: Combined coefficient vector [theta_xx ; theta_xy].
        n_basis_xx: Number of chromosome-chromosome basis functions (0 if no xx).
        n_basis_xy: Number of chromosome-partner basis functions.
        basis_xx: Basis for xx kernel, or None for topologies without xx.
        basis_xy: Basis for xy (chromosome-partner) kernel.
        D_x: Scalar effective diffusion coefficient (um^2/s).
        dt: Time step (seconds).
        metadata: Optional dict of fitting metadata.
        diffusion_model: Optional fitted D(x) model from second-stage estimation.
        topology: Base interaction topology name (e.g. "poles", "poles_and_chroms").
        r_cutoff_xx: If set, xx forces are zeroed beyond this distance (um).
            Persisted through save/load.  Enforced in evaluate_kernel(),
            simulation, and diffusion estimation.
    """

    theta: np.ndarray
    n_basis_xx: int
    n_basis_xy: int
    basis_xx: BSplineBasis | HatBasis | None
    basis_xy: BSplineBasis | HatBasis
    D_x: float
    dt: float
    metadata: dict | None = None
    diffusion_model: DiffusionResult | None = None
    topology: str = "poles"
    r_cutoff_xx: float | None = None

    @property
    def theta_xx(self) -> np.ndarray:
        """Chromosome-chromosome kernel coefficients (empty if no xx basis)."""
        return self.theta[: self.n_basis_xx]

    @property
    def theta_xy(self) -> np.ndarray:
        """Chromosome-partner kernel coefficients."""
        return self.theta[self.n_basis_xx :]

    def evaluate_kernel(self, kernel: str, r: np.ndarray) -> np.ndarray | None:
        """Evaluate a fitted kernel at distances *r*.

        For ``"xx"`` with ``r_cutoff_xx`` set, values at ``r > r_cutoff_xx``
        are zeroed.  Returns ``None`` when the model has no xx basis.
        """
        values = np.asarray(r, dtype=np.float64)
        if kernel == "xx":
            if self.basis_xx is None:
                return None
            result = self.basis_xx.evaluate(values) @ self.theta_xx
            if self.r_cutoff_xx is not None:
                result[values > self.r_cutoff_xx] = 0.0
            return result
        if kernel == "xy":
            return self.basis_xy.evaluate(values) @ self.theta_xy
        raise ValueError(f"Unknown kernel '{kernel}'. Use 'xx' or 'xy'.")

    def save(self, path: str | Path) -> None:
        """Save the model to an ``.npz`` file.

        Persists theta, basis configurations, scalar D, metadata, topology,
        ``r_cutoff_xx``, and (if present) the variable-diffusion model.
        """
        output_path = Path(path)
        has_diffusion_model = self.diffusion_model is not None
        if has_diffusion_model:
            diffusion_basis = self.diffusion_model.basis_D
            diffusion_payload = {
                "diffusion_has_model": True,
                "diffusion_coord_name": self.diffusion_model.coord_name,
                "diffusion_D_scalar": self.diffusion_model.D_scalar,
                "diffusion_d_coeffs": self.diffusion_model.d_coeffs,
                "diffusion_basis_type": (
                    "bspline" if isinstance(diffusion_basis, BSplineBasis) else "hat"
                ),
                "diffusion_basis_r_min": diffusion_basis.r_min,
                "diffusion_basis_r_max": diffusion_basis.r_max,
                "diffusion_basis_n_basis": diffusion_basis.n_basis,
            }
        else:
            diffusion_payload = {"diffusion_has_model": False}

        basis_xx_payload = {}
        if self.basis_xx is not None:
            basis_xx_payload = {
                "basis_xx_type": "bspline" if isinstance(self.basis_xx, BSplineBasis) else "hat",
                "basis_xx_r_min": self.basis_xx.r_min,
                "basis_xx_r_max": self.basis_xx.r_max,
                "basis_xx_n_basis": self.basis_xx.n_basis,
            }

        cutoff_payload = {}
        if self.r_cutoff_xx is not None:
            cutoff_payload["r_cutoff_xx"] = self.r_cutoff_xx

        np.savez(
            output_path,
            theta=self.theta,
            n_basis_xx=self.n_basis_xx,
            n_basis_xy=self.n_basis_xy,
            D_x=self.D_x,
            dt=self.dt,
            topology=self.topology,
            metadata=np.array(self.metadata, dtype=object),
            basis_xy_type="bspline" if isinstance(self.basis_xy, BSplineBasis) else "hat",
            basis_xy_r_min=self.basis_xy.r_min,
            basis_xy_r_max=self.basis_xy.r_max,
            basis_xy_n_basis=self.basis_xy.n_basis,
            **basis_xx_payload,
            **diffusion_payload,
            **cutoff_payload,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FittedModel":
        """Load a model from an ``.npz`` file saved by :meth:`save`."""
        input_path = Path(path)
        data = np.load(input_path, allow_pickle=True)

        _VALID_KINDS = {"bspline", "hat"}

        def make_basis(kind: str, r_min: float, r_max: float, n_basis: int):
            if kind not in _VALID_KINDS:
                raise ValueError(
                    f"Unknown basis type {kind!r} in saved model; "
                    f"must be one of {sorted(_VALID_KINDS)}"
                )
            if kind == "bspline":
                return BSplineBasis(float(r_min), float(r_max), int(n_basis))
            return HatBasis(float(r_min), float(r_max), int(n_basis))

        metadata = data["metadata"].item() if data["metadata"].shape == () else None
        topology = str(data["topology"]) if "topology" in data else "poles"

        n_basis_xx = int(data["n_basis_xx"])
        basis_xx = None
        if n_basis_xx > 0 and "basis_xx_type" in data:
            basis_xx = make_basis(
                str(data["basis_xx_type"]),
                float(data["basis_xx_r_min"]),
                float(data["basis_xx_r_max"]),
                n_basis_xx,
            )

        diffusion_model = None
        if "diffusion_has_model" in data and bool(data["diffusion_has_model"]):
            basis_D = make_basis(
                str(data["diffusion_basis_type"]),
                float(data["diffusion_basis_r_min"]),
                float(data["diffusion_basis_r_max"]),
                int(data["diffusion_basis_n_basis"]),
            )
            diffusion_model = DiffusionResult(
                d_coeffs=np.asarray(data["diffusion_d_coeffs"], dtype=np.float64),
                basis_D=basis_D,
                coord_name=str(data["diffusion_coord_name"]),
                D_scalar=float(data["diffusion_D_scalar"]),
            )

        r_cutoff_xx = None
        if "r_cutoff_xx" in data:
            val = float(data["r_cutoff_xx"])
            if np.isfinite(val):
                r_cutoff_xx = val

        return cls(
            theta=np.asarray(data["theta"], dtype=np.float64),
            n_basis_xx=n_basis_xx,
            n_basis_xy=int(data["n_basis_xy"]),
            basis_xx=basis_xx,
            basis_xy=make_basis(
                str(data["basis_xy_type"]),
                float(data["basis_xy_r_min"]),
                float(data["basis_xy_r_max"]),
                int(data["basis_xy_n_basis"]),
            ),
            D_x=float(data["D_x"]),
            dt=float(data["dt"]),
            metadata=metadata,
            diffusion_model=diffusion_model,
            topology=topology,
            r_cutoff_xx=r_cutoff_xx,
        )
