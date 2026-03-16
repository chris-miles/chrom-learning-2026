from dataclasses import dataclass


@dataclass
class FitConfig:
    """Configuration for the kernel fitting pipeline.

    Attributes:
        endpoint_method: Trajectory trimming endpoint — ``"midpoint_neb_ao"``
            (default), ``"ao_mean"``, or ``"end_sep"``.
        n_basis_xx: Number of basis functions for chromosome-chromosome kernel.
        n_basis_xy: Number of basis functions for centrosome-on-chromosome kernel.
        r_min_xx, r_max_xx: Domain of the xx basis (microns).
        r_min_xy, r_max_xy: Domain of the xy basis (microns).
        basis_type: ``"bspline"`` (default) or ``"hat"``.
        lambda_ridge: Ridge (L2) regularisation strength.
        lambda_rough: Roughness penalty strength (integrated squared 2nd deriv).
        dt: Time between frames in seconds.
        d: Spatial dimension (3).
        basis_eval_mode: Where basis functions are evaluated when building the
            design matrix — ``"ito"`` (default, current positions),
            ``"ito_shift"`` (previous positions, decorrelates localization noise),
            or ``"strato"`` (midpoint, Stratonovich-style).
        diffusion_mode: Local diffusion estimator — ``"msd"`` (default),
            ``"vestergaard"`` (3-point, noise-robust), ``"weak_noise"``
            (3-point, drift-robust), or ``"f_corrected"`` (force-subtracted).
        D_variable: If True, fit D as a function of position coordinate instead
            of a scalar.
        n_basis_D: Number of basis functions for the D(coordinate) expansion.
        r_min_D, r_max_D: Domain of the diffusion basis.  Note: for ``"axial"``
            coordinate, positions can be negative (signed distance from metaphase
            plate), so the default range is ``[-8, 8]``.
        D_coordinate: Position coordinate for variable D — ``"axial"`` (default,
            along spindle axis), ``"radial"`` (perpendicular to spindle axis),
            or ``"distance"`` (from spindle center).
        topology: Interaction topology — "poles" (default), "center",
            "poles_and_chroms", or "center_and_chroms".
    """

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
    basis_eval_mode: str = "ito"
    diffusion_mode: str = "msd"
    D_variable: bool = False
    n_basis_D: int = 6
    r_min_D: float = -8.0
    r_max_D: float = 8.0
    D_coordinate: str = "axial"
    topology: str = "poles"
