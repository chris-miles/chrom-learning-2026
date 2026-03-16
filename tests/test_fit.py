import numpy as np

from chromlearn.io.trajectory import TrimmedCell
from chromlearn.model_fitting import FitConfig
from chromlearn.model_fitting.basis import HatBasis
from chromlearn.model_fitting.fit import (
    bootstrap_kernels,
    cross_validate,
    estimate_diffusion,
    fit_kernels,
    fit_model,
    rollout_cross_validate,
)
from chromlearn.model_fitting.features import build_design_matrix
from chromlearn.model_fitting.simulate import add_localization_noise, simulate_trajectories


def make_synthetic_inference_cell(
    T: int = 160,
    N: int = 16,
    dt: float = 0.1,
    D_x: float = 0.02,
    localization_sigma: float = 0.0,
    seed: int = 0,
) -> tuple[TrimmedCell, HatBasis, HatBasis, np.ndarray]:
    rng = np.random.default_rng(seed)
    basis_xx = HatBasis(0.0, 8.0, n_basis=4)
    basis_xy = HatBasis(0.0, 10.0, n_basis=4)
    theta_xx = np.array([0.0, -0.2, 0.0, 0.0])
    theta_xy = np.array([0.0, 0.4, 0.0, 0.0])
    theta_true = np.concatenate([theta_xx, theta_xy])

    def kernel_xx(r: np.ndarray) -> np.ndarray:
        return basis_xx.evaluate(r) @ theta_xx

    def kernel_xy(r: np.ndarray) -> np.ndarray:
        return basis_xy.evaluate(r) @ theta_xy

    centrioles = np.zeros((T + 1, 3, 2))
    centrioles[:, 0, 0] = -3.0
    centrioles[:, 0, 1] = 3.0
    x0 = rng.normal(0.0, 1.0, size=(N, 3))
    partners = centrioles.transpose(2, 0, 1)  # (2, T+1, 3)
    chromosomes = simulate_trajectories(
        kernel_xx=kernel_xx,
        kernel_xy=kernel_xy,
        partner_positions=partners,
        x0=x0,
        n_steps=T,
        dt=dt,
        D_x=D_x,
        rng=rng,
    )
    if localization_sigma > 0.0:
        chromosomes = add_localization_noise(chromosomes, sigma=localization_sigma, rng=rng)

    cell = TrimmedCell(
        cell_id="synthetic_fit",
        condition="test",
        centrioles=centrioles,
        chromosomes=chromosomes,
        tracked=N,
        dt=dt,
        start_frame=0,
        end_frame=T,
    )
    return cell, basis_xx, basis_xy, theta_true


def _make_rotating_centrioles(T: int, dt: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(T + 1, dtype=np.float64) * dt
    sep = 4.5 + 1.0 * np.sin(0.25 * t + rng.uniform(0.0, 2.0 * np.pi))
    theta = 0.35 * np.sin(0.11 * t + rng.uniform(0.0, 2.0 * np.pi))
    phi = 0.45 * np.cos(0.07 * t + rng.uniform(0.0, 2.0 * np.pi))
    axis = np.column_stack([
        np.cos(theta) * np.cos(phi),
        np.sin(theta) * np.cos(phi),
        np.sin(phi),
    ])
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    center = np.column_stack([
        0.5 * np.sin(0.09 * t + rng.uniform(0.0, 2.0 * np.pi)),
        0.4 * np.cos(0.05 * t + rng.uniform(0.0, 2.0 * np.pi)),
        0.3 * np.sin(0.13 * t + rng.uniform(0.0, 2.0 * np.pi)),
    ])

    centrioles = np.zeros((T + 1, 3, 2), dtype=np.float64)
    centrioles[:, :, 0] = center - 0.5 * sep[:, np.newaxis] * axis
    centrioles[:, :, 1] = center + 0.5 * sep[:, np.newaxis] * axis
    return centrioles


def make_synthetic_model_selection_cells(
    topology: str,
    n_cells: int = 4,
    T: int = 60,
    N: int = 8,
    dt: float = 0.12,
    D_x: float = 8e-4,
) -> list[TrimmedCell]:
    """Generate synthetic cells whose true topology is identifiable in LOOCV.

    The poles rotate and change separation over time so that a midpoint-only
    model cannot simply mimic the two-pole geometry, and vice versa.
    """
    if topology not in ("poles", "center", "poles_and_chroms", "center_and_chroms"):
        raise ValueError(f"Unsupported synthetic topology: {topology}")

    basis_xy = HatBasis(0.3, 8.0, n_basis=5)
    basis_xx = HatBasis(0.3, 6.0, n_basis=5)

    if topology in ("poles", "poles_and_chroms"):
        theta_xy = np.array([-0.10, 0.18, 0.10, -0.04, 0.00], dtype=np.float64)
    else:
        theta_xy = np.array([-0.20, 0.28, 0.06, 0.00, 0.00], dtype=np.float64)

    if topology in ("poles_and_chroms", "center_and_chroms"):
        theta_xx = np.array([0.00, -0.18, -0.06, 0.00, 0.00], dtype=np.float64)
    else:
        theta_xx = None

    def kernel_xy(r: np.ndarray) -> np.ndarray:
        return basis_xy.evaluate(r) @ theta_xy

    def kernel_xx(r: np.ndarray) -> np.ndarray:
        assert theta_xx is not None
        return basis_xx.evaluate(r) @ theta_xx

    cells: list[TrimmedCell] = []
    for seed in range(n_cells):
        rng = np.random.default_rng(seed)
        centrioles = _make_rotating_centrioles(T=T, dt=dt, seed=seed + 500)
        if topology in ("poles", "poles_and_chroms"):
            partners = centrioles.transpose(2, 0, 1)
        else:
            partners = (0.5 * (centrioles[:, :, 0] + centrioles[:, :, 1]))[np.newaxis, :, :]

        x0 = np.column_stack([
            rng.normal(0.0, 1.0, size=N),
            rng.normal(0.0, 1.0, size=N),
            rng.normal(0.0, 1.0, size=N),
        ])
        chromosomes = simulate_trajectories(
            kernel_xx=kernel_xx if theta_xx is not None else None,
            kernel_xy=kernel_xy,
            partner_positions=partners,
            x0=x0,
            n_steps=T,
            dt=dt,
            D_x=D_x,
            rng=rng,
        )
        cells.append(
            TrimmedCell(
                cell_id=f"synthetic_{topology}_{seed}",
                condition="synthetic_model_selection",
                centrioles=centrioles,
                chromosomes=chromosomes,
                tracked=N,
                dt=dt,
                start_frame=0,
                end_frame=T,
            )
        )
    return cells


def _simulate_trajectories_reference(
    kernel_xx,
    kernel_xy,
    partner_positions,
    x0,
    n_steps,
    dt,
    D_x,
    rng,
):
    """Pre-vectorization reference implementation for regression tests."""
    n_chromosomes = x0.shape[0]
    n_partners = partner_positions.shape[0]
    trajectory = np.zeros((n_steps + 1, 3, n_chromosomes), dtype=np.float64)
    trajectory[0] = x0.T
    noise_scale = np.sqrt(2.0 * D_x * dt)

    for step in range(n_steps):
        positions = trajectory[step]
        forces = np.zeros_like(positions)

        for chrom_index in range(n_chromosomes):
            current = positions[:, chrom_index]

            if kernel_xx is not None:
                for neighbor_index in range(n_chromosomes):
                    if neighbor_index == chrom_index:
                        continue
                    delta = positions[:, neighbor_index] - current
                    distance = float(np.linalg.norm(delta))
                    if distance <= 1e-12:
                        continue
                    direction = delta / distance
                    forces[:, chrom_index] += kernel_xx(np.array([distance]))[0] * direction

            for partner_index in range(n_partners):
                delta = partner_positions[partner_index, step] - current
                distance = float(np.linalg.norm(delta))
                if distance <= 1e-12:
                    continue
                direction = delta / distance
                forces[:, chrom_index] += kernel_xy(np.array([distance]))[0] * direction

        noise = noise_scale * rng.standard_normal(size=(3, n_chromosomes))
        trajectory[step + 1] = positions + forces * dt + noise

    return trajectory


def test_fit_kernels_recovers_known_theta() -> None:
    rng = np.random.default_rng(42)
    n_obs, n_basis = 500, 6
    theta_true = rng.standard_normal(n_basis)
    G = rng.standard_normal((n_obs, n_basis))
    V = G @ theta_true + 0.01 * rng.standard_normal(n_obs)
    result = fit_kernels(G, V, lambda_ridge=1e-6, lambda_rough=0.0, R=np.zeros((n_basis, n_basis)))
    np.testing.assert_allclose(result.theta, theta_true, atol=0.05)


def test_simulate_trajectories_matches_reference_loop() -> None:
    rng_seed = 123

    def kernel_xx(r: np.ndarray) -> np.ndarray:
        return -0.05 * np.exp(-0.2 * r)

    def kernel_xy(r: np.ndarray) -> np.ndarray:
        return 0.08 - 0.01 * r

    x0 = np.array([
        [-1.0, 0.2, 0.0],
        [0.5, -0.3, 0.1],
        [1.2, 0.4, -0.2],
    ])
    partner_positions = np.zeros((2, 6, 3), dtype=float)
    partner_positions[0, :, 0] = -2.0
    partner_positions[1, :, 0] = 2.0

    traj_ref = _simulate_trajectories_reference(
        kernel_xx=kernel_xx,
        kernel_xy=kernel_xy,
        partner_positions=partner_positions,
        x0=x0,
        n_steps=5,
        dt=0.2,
        D_x=0.01,
        rng=np.random.default_rng(rng_seed),
    )
    traj_vec = simulate_trajectories(
        kernel_xx=kernel_xx,
        kernel_xy=kernel_xy,
        partner_positions=partner_positions,
        x0=x0,
        n_steps=5,
        dt=0.2,
        D_x=0.01,
        rng=np.random.default_rng(rng_seed),
    )

    np.testing.assert_allclose(traj_vec, traj_ref, atol=1e-12)


def test_fit_kernels_with_roughness() -> None:
    rng = np.random.default_rng(42)
    G = rng.standard_normal((200, 5))
    V = rng.standard_normal(200)
    result = fit_kernels(G, V, lambda_ridge=0.01, lambda_rough=0.01, R=np.eye(5))
    assert result.theta.shape == (5,)


def test_fit_kernels_result_has_residuals() -> None:
    rng = np.random.default_rng(42)
    G = rng.standard_normal((100, 4))
    V = rng.standard_normal(100)
    result = fit_kernels(G, V, lambda_ridge=0.01, lambda_rough=0.0, R=np.zeros((4, 4)))
    assert result.residuals.shape == (100,)


def test_estimate_diffusion() -> None:
    rng = np.random.default_rng(42)
    D_true = 0.5
    dt = 5.0
    d = 3
    n_obs = 10_000
    V = np.sqrt(2 * D_true / dt) * rng.standard_normal(n_obs * d)
    G = np.zeros((n_obs * d, 1))
    theta = np.array([0.0])
    D_hat = estimate_diffusion(V, G, theta, dt=dt, d=d)
    np.testing.assert_allclose(D_hat, D_true, rtol=0.1)


def test_end_to_end_recovers_known_kernels_on_synthetic_data() -> None:
    cell, basis_xx, basis_xy, theta_true = make_synthetic_inference_cell()
    G, V = build_design_matrix([cell], basis_xx, basis_xy)
    result = fit_kernels(
        G,
        V,
        lambda_ridge=1e-6,
        lambda_rough=0.0,
        R=np.zeros((theta_true.size, theta_true.size)),
    )

    rmse = float(np.sqrt(np.mean((result.theta - theta_true) ** 2)))
    assert rmse < 0.1
    np.testing.assert_allclose(result.theta[[1, 5]], theta_true[[1, 5]], atol=0.1)


def test_shifted_basis_eval_modes_reduce_localization_noise_bias() -> None:
    cell, basis_xx, basis_xy, theta_true = make_synthetic_inference_cell(
        T=120,
        N=14,
        localization_sigma=0.25,
    )
    roughness = np.zeros((theta_true.size, theta_true.size))
    errors = {}
    for mode in ("ito", "ito_shift", "strato"):
        G, V = build_design_matrix([cell], basis_xx, basis_xy, basis_eval_mode=mode)
        result = fit_kernels(G, V, lambda_ridge=1e-6, lambda_rough=0.0, R=roughness)
        errors[mode] = float(np.sqrt(np.mean((result.theta - theta_true) ** 2)))

    assert errors["ito_shift"] < errors["ito"] * 0.5
    assert errors["strato"] < errors["ito"] * 0.5
    assert errors["ito_shift"] < 0.2
    assert errors["strato"] < 0.2


def test_cross_validate_and_bootstrap_support_shifted_basis_eval_mode() -> None:
    cells = [
        make_synthetic_inference_cell(T=90, N=12, dt=0.15, seed=seed)[0]
        for seed in range(4)
    ]
    _, basis_xx, basis_xy, theta_true = make_synthetic_inference_cell(T=90, N=12, dt=0.15)

    config = FitConfig(
        topology="poles_and_chroms",
        n_basis_xx=4, n_basis_xy=4,
        r_min_xx=0.0, r_max_xx=8.0,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat",
        lambda_ridge=1e-6, lambda_rough=0.0,
        basis_eval_mode="ito_shift",
        dt=0.15,
    )
    cv_result = cross_validate(cells, config)
    G, V = build_design_matrix(
        cells, basis_xx, basis_xy,
        basis_eval_mode="ito_shift",
        topology="poles_and_chroms",
    )
    baseline_error = float(np.mean(V**2))
    assert np.all(np.isfinite(cv_result.held_out_errors))
    assert cv_result.mean_error < baseline_error

    bootstrap_result = bootstrap_kernels(
        cells, config, n_boot=8, rng=np.random.default_rng(0),
    )
    assert bootstrap_result.theta_samples.shape == (8, theta_true.size)
    assert np.all(np.isfinite(bootstrap_result.theta_std))
    bootstrap_rmse = float(np.sqrt(np.mean((bootstrap_result.theta_mean - theta_true) ** 2)))
    assert bootstrap_rmse < 0.08


def test_fit_model_poles_only_no_xx() -> None:
    """fit_model with topology='poles' produces no chromosome-chromosome kernel."""
    cell, _, _, _ = make_synthetic_inference_cell()
    config = FitConfig(
        topology="poles",
        n_basis_xx=10, n_basis_xy=4,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat",
        lambda_ridge=1e-6, lambda_rough=0.0,
        dt=0.1,
    )
    model = fit_model([cell], config)
    assert model.n_basis_xx == 0
    assert model.basis_xx is None
    assert model.theta.shape[0] == 4


def test_fit_model_poles_and_chroms() -> None:
    """fit_model with topology='poles_and_chroms' includes both kernels."""
    cell, _, _, _ = make_synthetic_inference_cell()
    config = FitConfig(
        topology="poles_and_chroms",
        n_basis_xx=4, n_basis_xy=4,
        r_min_xx=0.0, r_max_xx=8.0,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat",
        lambda_ridge=1e-6, lambda_rough=0.0,
        dt=0.1,
    )
    model = fit_model([cell], config)
    assert model.n_basis_xx == 4
    assert model.basis_xx is not None
    assert model.theta.shape[0] == 8


def test_fit_model_preserves_topology() -> None:
    """fit_model stores config.topology in the returned FittedModel."""
    cell, _, _, _ = make_synthetic_inference_cell()
    for topo in ("poles", "center", "poles_and_chroms", "center_and_chroms"):
        config = FitConfig(
            topology=topo,
            n_basis_xx=4, n_basis_xy=4,
            r_min_xx=0.0, r_max_xx=8.0,
            r_min_xy=0.0, r_max_xy=10.0,
            basis_type="hat",
            lambda_ridge=1e-6, lambda_rough=0.0,
            dt=0.1,
        )
        model = fit_model([cell], config)
        assert model.topology == topo, f"Expected topology={topo}, got {model.topology}"


def test_fit_model_vestergaard_diffusion() -> None:
    """fit_model honours diffusion_mode='vestergaard'."""
    cell, _, _, _ = make_synthetic_inference_cell()
    config_msd = FitConfig(
        topology="poles", n_basis_xy=4,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat", lambda_ridge=1e-6, lambda_rough=0.0,
        dt=0.1, diffusion_mode="msd",
    )
    config_vest = FitConfig(
        topology="poles", n_basis_xy=4,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat", lambda_ridge=1e-6, lambda_rough=0.0,
        dt=0.1, diffusion_mode="vestergaard",
    )
    model_msd = fit_model([cell], config_msd)
    model_vest = fit_model([cell], config_vest)
    # Both should produce positive D, but they use different estimators
    assert model_msd.D_x > 0
    assert model_vest.D_x > 0
    # They should differ (different estimators on the same data)
    assert model_msd.D_x != model_vest.D_x


def test_fit_model_variable_diffusion() -> None:
    """fit_model with D_variable=True returns a diffusion_model."""
    cell, _, _, _ = make_synthetic_inference_cell()
    config = FitConfig(
        topology="poles", n_basis_xy=4,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat", lambda_ridge=1e-6, lambda_rough=0.0,
        dt=0.1, D_variable=True, n_basis_D=4,
        r_min_D=-5.0, r_max_D=5.0,
    )
    model = fit_model([cell], config)
    assert model.diffusion_model is not None
    assert model.diffusion_model.d_coeffs.shape == (4,)


def test_cross_validate_poles_only() -> None:
    """CV works with topology='poles' (basis_xx=None)."""
    cells = [
        make_synthetic_inference_cell(T=90, N=12, dt=0.15, seed=s)[0]
        for s in range(4)
    ]
    config = FitConfig(
        topology="poles",
        n_basis_xy=4,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat",
        lambda_ridge=1e-6, lambda_rough=0.0,
        dt=0.15,
    )
    cv = cross_validate(cells, config)
    assert np.all(np.isfinite(cv.held_out_errors))
    assert cv.mean_error > 0


def test_cross_validate_recovers_true_topology_on_synthetic_cells() -> None:
    topologies = ("poles", "center", "poles_and_chroms", "center_and_chroms")

    for true_topology in topologies:
        cells = make_synthetic_model_selection_cells(true_topology)
        scores: dict[str, float] = {}
        for candidate_topology in topologies:
            config = FitConfig(
                topology=candidate_topology,
                n_basis_xx=5,
                n_basis_xy=5,
                r_min_xx=0.3,
                r_max_xx=6.0,
                r_min_xy=0.3,
                r_max_xy=8.0,
                basis_type="hat",
                lambda_ridge=1e-6,
                lambda_rough=0.0,
                dt=0.12,
            )
            scores[candidate_topology] = cross_validate(cells, config).mean_error

        winner = min(scores, key=scores.get)
        assert winner == true_topology, (
            f"Expected synthetic {true_topology} cells to prefer {true_topology}, "
            f"got {winner} with scores {scores}"
        )


def test_rollout_cross_validate_returns_finite_metrics() -> None:
    cells = [
        make_synthetic_inference_cell(
            T=60,
            N=10,
            dt=0.15,
            D_x=0.005,
            seed=seed,
        )[0]
        for seed in range(4)
    ]
    config = FitConfig(
        topology="poles_and_chroms",
        n_basis_xx=4, n_basis_xy=4,
        r_min_xx=0.0, r_max_xx=8.0,
        r_min_xy=0.0, r_max_xy=10.0,
        basis_type="hat",
        lambda_ridge=1e-6, lambda_rough=0.0,
        dt=0.15,
    )

    rollout = rollout_cross_validate(
        cells,
        config,
        n_reps=4,
        horizons=(1, 5, 10),
        rng=np.random.default_rng(0),
    )

    assert np.all(np.isfinite(rollout.axial_mse))
    assert np.all(np.isfinite(rollout.radial_mse))
    assert np.all(np.isfinite(rollout.endpoint_mean_error))
    assert np.all(np.isfinite(rollout.final_axial_wasserstein))
    assert np.all(np.isfinite(rollout.final_radial_wasserstein))
    assert rollout.horizon_errors.shape == (len(cells), 3)
    assert np.all(np.isfinite(rollout.mean_horizon_errors))
    assert rollout.mean_axial_mse < 2.0
    assert rollout.mean_radial_mse < 2.0
    assert rollout.mean_endpoint_mean_error < 2.0
