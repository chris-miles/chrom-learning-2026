from dataclasses import dataclass


@dataclass
class FitConfig:
    """Configuration for the kernel fitting pipeline."""

    endpoint_method: str = "midpoint_neb_ao"
    n_basis_xx: int = 10
    n_basis_xy: int = 10
    r_min_xx: float = 0.5
    r_max_xx: float = 10.0
    r_min_xy: float = 0.5
    r_max_xy: float = 12.0
    basis_type: str = "bspline"
    lambda_ridge: float = 1e-3
    lambda_rough: float = 1e-3
    dt: float = 5.0
    d: int = 3
