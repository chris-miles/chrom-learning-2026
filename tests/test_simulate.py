import numpy as np

from chromlearn.model_fitting.simulate import (
    add_localization_noise,
    generate_synthetic_data,
    simulate_trajectories,
)


def test_simulate_output_shape() -> None:
    rng = np.random.default_rng(42)
    n_chrom = 10
    n_steps = 50
    partners = np.zeros((2, n_steps + 1, 3))
    partners[0, :, 0] = -5.0
    partners[1, :, 0] = 5.0
    x0 = rng.normal(0.0, 2.0, size=(n_chrom, 3))
    trajectory = simulate_trajectories(
        kernel_xx=lambda r: np.zeros_like(r),
        kernel_xy=lambda r: np.zeros_like(r),
        partner_positions=partners,
        x0=x0,
        n_steps=n_steps,
        dt=5.0,
        D_x=0.1,
        rng=rng,
    )
    assert trajectory.shape == (n_steps + 1, 3, n_chrom)


def test_pure_diffusion_msd() -> None:
    rng = np.random.default_rng(42)
    n_chrom = 100
    n_steps = 200
    dt = 1.0
    D = 0.5
    partners = np.zeros((2, n_steps + 1, 3))
    x0 = np.zeros((n_chrom, 3))
    trajectory = simulate_trajectories(
        kernel_xx=lambda r: np.zeros_like(r),
        kernel_xy=lambda r: np.zeros_like(r),
        partner_positions=partners,
        x0=x0,
        n_steps=n_steps,
        dt=dt,
        D_x=D,
        rng=rng,
    )
    displacements = trajectory[1:] - trajectory[:-1]
    msd_per_step = np.mean(np.sum(displacements**2, axis=1))
    np.testing.assert_allclose(msd_per_step, 2 * 3 * D * dt, rtol=0.15)


def test_add_localization_noise() -> None:
    rng = np.random.default_rng(42)
    trajectory = np.zeros((50, 3, 10))
    noisy = add_localization_noise(trajectory, sigma=0.1, rng=rng)
    assert noisy.shape == trajectory.shape
    assert not np.allclose(noisy, trajectory)
    np.testing.assert_allclose(np.std(noisy), 0.1, rtol=0.2)


def test_generate_synthetic_data() -> None:
    dataset = generate_synthetic_data(
        kernel_xx=lambda r: -0.01 * r,
        kernel_xy=lambda r: 0.05 * np.ones_like(r),
        n_chromosomes=10,
        n_steps=50,
        dt=5.0,
        D_x=0.1,
        rng=np.random.default_rng(42),
    )
    assert dataset.chromosomes.shape == (51, 3, 10)
    assert dataset.centrosomes.shape == (51, 3, 2)
    assert dataset.kernel_xx is not None
    assert dataset.kernel_xy is not None


def test_simulate_single_partner():
    """simulate_trajectories works with 1 partner (center topology)."""
    rng = np.random.default_rng(42)
    n_steps = 50
    partners = np.zeros((1, n_steps + 1, 3))
    x0 = rng.normal(0.0, 2.0, size=(5, 3))
    trajectory = simulate_trajectories(
        kernel_xx=None,
        kernel_xy=lambda r: -0.01 * r,
        partner_positions=partners,
        x0=x0,
        n_steps=n_steps,
        dt=5.0,
        D_x=0.1,
        rng=rng,
    )
    assert trajectory.shape == (n_steps + 1, 3, 5)


def test_simulate_kernel_xx_none_differs_from_repulsive_xx():
    """kernel_xx=None should produce different trajectories than repulsive xx."""
    seed = 42
    n_steps = 30
    partners = np.zeros((2, n_steps + 1, 3))
    partners[0, :, 0] = -5.0
    partners[1, :, 0] = 5.0
    x0 = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])  # two nearby chromosomes

    traj_no_xx = simulate_trajectories(
        kernel_xx=None,
        kernel_xy=lambda r: -0.01 * r,
        partner_positions=partners,
        x0=x0,
        n_steps=n_steps,
        dt=1.0,
        D_x=0.001,
        rng=np.random.default_rng(seed),
    )
    traj_with_xx = simulate_trajectories(
        kernel_xx=lambda r: -0.5 * np.ones_like(r),  # strong repulsion
        kernel_xy=lambda r: -0.01 * r,
        partner_positions=partners,
        x0=x0,
        n_steps=n_steps,
        dt=1.0,
        D_x=0.001,
        rng=np.random.default_rng(seed),
    )
    assert traj_no_xx.shape == (n_steps + 1, 3, 2)
    # The repulsive xx should push the two chromosomes apart more
    final_sep_no_xx = np.linalg.norm(traj_no_xx[-1, :, 0] - traj_no_xx[-1, :, 1])
    final_sep_with_xx = np.linalg.norm(traj_with_xx[-1, :, 0] - traj_with_xx[-1, :, 1])
    assert final_sep_with_xx > final_sep_no_xx, (
        f"Repulsive xx should increase separation: {final_sep_with_xx:.3f} vs {final_sep_no_xx:.3f}"
    )
