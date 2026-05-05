from dataclasses import dataclass
from typing import ClassVar


@dataclass
class FitConfig:
    """Configuration for the kernel fitting pipeline.

    Attributes:
        endpoint_method: Trajectory trimming endpoint — ``"neb_ao_frac"``
            (default) or ``"end_sep"``.
        endpoint_frac: Fraction of the ``[NEB, AO]`` window to use (only for
            ``"neb_ao_frac"``).  ``1/3`` keeps the early gathering phase
            (default); ``0.5`` = midpoint, ``1.0`` = full window to AO.
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
        r_cutoff_xx: Hard cutoff (microns) above which xx forces are zeroed.
            Deprecated; prefer ``envelope_r0_xx`` / ``envelope_w_xx`` for a
            smooth (no-kink) short-range kernel.  Mutually exclusive with the
            envelope parameters.
        envelope_r0_xx: Center of the smooth steric envelope (microns) applied
            to the xx basis.  When set, every xx basis column is multiplied by
            ``e(r) = 0.5 * (1 - tanh((r - r0)/w))`` before fitting, yielding a
            kernel that decays smoothly to zero past ``r0``.  Must be paired
            with ``envelope_w_xx``.
        envelope_w_xx: Width (microns) of the smooth envelope transition.
    """

    _VALID_BASIS_TYPES: ClassVar = frozenset({"bspline", "hat"})
    _VALID_TOPOLOGIES: ClassVar = frozenset({"poles", "center", "poles_and_chroms", "center_and_chroms"})
    _VALID_DIFFUSION_MODES: ClassVar = frozenset({"msd", "vestergaard", "weak_noise", "f_corrected"})
    _VALID_ENDPOINT_METHODS: ClassVar = frozenset({"neb_ao_frac", "end_sep"})

    endpoint_method: str = "neb_ao_frac"
    endpoint_frac: float = 0.4
    n_basis_xx: int = 10
    n_basis_xy: int = 10
    r_min_xx: float = 0.5
    r_max_xx: float = 10.0
    r_min_xy: float = 0.5
    r_max_xy: float = 12.0
    basis_type: str = "bspline"
    lambda_ridge: float = 1e-6  # Numerical jitter only; not a tuning knob.
    lambda_rough: float = 1.0
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
    r_cutoff_xx: float | None = None
    envelope_r0_xx: float | None = None
    envelope_w_xx: float | None = None

    def __post_init__(self) -> None:
        if self.basis_type not in self._VALID_BASIS_TYPES:
            raise ValueError(
                f"Unknown basis_type {self.basis_type!r}; "
                f"must be one of {sorted(self._VALID_BASIS_TYPES)}"
            )
        if self.topology not in self._VALID_TOPOLOGIES:
            raise ValueError(
                f"Unknown topology {self.topology!r}; "
                f"must be one of {sorted(self._VALID_TOPOLOGIES)}"
            )
        if self.diffusion_mode not in self._VALID_DIFFUSION_MODES:
            raise ValueError(
                f"Unknown diffusion_mode {self.diffusion_mode!r}; "
                f"must be one of {sorted(self._VALID_DIFFUSION_MODES)}"
            )
        if self.endpoint_method not in self._VALID_ENDPOINT_METHODS:
            raise ValueError(
                f"Unknown endpoint_method {self.endpoint_method!r}; "
                f"must be one of {sorted(self._VALID_ENDPOINT_METHODS)}"
            )
        envelope_set = (self.envelope_r0_xx is not None) or (self.envelope_w_xx is not None)
        if envelope_set and (self.envelope_r0_xx is None or self.envelope_w_xx is None):
            raise ValueError(
                "envelope_r0_xx and envelope_w_xx must both be set or both be None."
            )
        if envelope_set and self.r_cutoff_xx is not None:
            raise ValueError(
                "r_cutoff_xx and envelope_{r0,w}_xx are mutually exclusive; "
                "use the envelope (smooth) form going forward."
            )
