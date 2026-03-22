"""Velocity-binned analysis: spatial vs temporal dependence of chromosome motion.

Replicates the analysis from the old paper (Fig 2A–B) that justifies dropping
time-dependence in the force model.  The key idea:

A) Chromosomes that *started* far vs close from the spindle center have
   indistinguishable velocity distributions when observed in the *same* spatial
   region — ruling out time/history dependence.

B) The *same* chromosomes (started far) show significantly different velocities
   at different distances — confirming spatial dependence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from chromlearn.io.loader import CellData
from chromlearn.io.trajectory import TrimmedCell, pole_center


@dataclass
class BinnedVelocityResult:
    """Aggregated per-frame velocity magnitudes split by initial distance and
    current spatial region."""

    # Panel A: same region, different starting distance
    startclose_region1: np.ndarray
    startfar_region1: np.ndarray
    ttest_a: stats.TtestResult
    ks_a: stats.KstestResult

    # Panel B: same starting distance (far), different region
    startfar_region1_b: np.ndarray  # alias, same data as startfar_region1
    startfar_region2: np.ndarray
    ttest_b: stats.TtestResult
    ks_b: stats.KstestResult

    # Thresholds used
    far_threshold: float
    region1_bounds: tuple[float, float]
    region2_bounds: tuple[float, float]


def compute_binned_velocities(
    cells: list[CellData] | list[TrimmedCell],
    far_threshold: float = 5.0,
    region1_bounds: tuple[float, float] = (3.0, 5.0),
    region2_bounds: tuple[float, float] = (5.0, 15.0),
) -> BinnedVelocityResult:
    """Compute velocity distributions split by initial distance and spatial bin.

    Each chromosome in each cell is treated as an independent trajectory.
    Velocities are frame-to-frame displacement magnitudes in µm/min.

    Parameters
    ----------
    cells : list of CellData or TrimmedCell
    far_threshold : µm — chromosomes starting above this are "far"
    region1_bounds : (lo, hi) µm — the shared spatial region for panel A
    region2_bounds : (lo, hi) µm — the far spatial region for panel B
    """
    startclose_r1: list[float] = []
    startfar_r1: list[float] = []
    startfar_r2: list[float] = []

    for cell in cells:
        center = pole_center(cell)  # (T, 3)
        dt_min = cell.dt / 60.0  # seconds → minutes

        for k in range(cell.tracked):
            pos = cell.chromosomes[:, :, k]  # (T, 3)
            if np.any(np.isnan(pos)):
                continue

            # Distance to spindle center at each frame
            dist = np.linalg.norm(pos - center, axis=1)  # (T,)

            # Per-frame velocity magnitude in µm/min
            vel = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt_min  # (T-1,)

            # Initial distance (first frame)
            init_dist = dist[0]
            is_far = init_dist > far_threshold

            # Distance at velocity frames (use distance at start of each step)
            dist_at_vel = dist[:-1]

            # Find frames in each region
            in_r1 = (dist_at_vel >= region1_bounds[0]) & (dist_at_vel < region1_bounds[1])
            in_r2 = (dist_at_vel >= region2_bounds[0]) & (dist_at_vel < region2_bounds[1])

            if is_far:
                startfar_r1.extend(vel[in_r1].tolist())
                startfar_r2.extend(vel[in_r2].tolist())
            else:
                startclose_r1.extend(vel[in_r1].tolist())

    startclose_r1_arr = np.asarray(startclose_r1)
    startfar_r1_arr = np.asarray(startfar_r1)
    startfar_r2_arr = np.asarray(startfar_r2)

    # Statistical tests
    ttest_a = stats.ttest_ind(startfar_r1_arr, startclose_r1_arr, equal_var=False)
    ks_a = stats.ks_2samp(startfar_r1_arr, startclose_r1_arr)

    ttest_b = stats.ttest_ind(startfar_r1_arr, startfar_r2_arr, equal_var=False)
    ks_b = stats.ks_2samp(startfar_r1_arr, startfar_r2_arr)

    return BinnedVelocityResult(
        startclose_region1=startclose_r1_arr,
        startfar_region1=startfar_r1_arr,
        ttest_a=ttest_a,
        ks_a=ks_a,
        startfar_region1_b=startfar_r1_arr,
        startfar_region2=startfar_r2_arr,
        ttest_b=ttest_b,
        ks_b=ks_b,
        far_threshold=far_threshold,
        region1_bounds=region1_bounds,
        region2_bounds=region2_bounds,
    )


def plot_binned_velocities(
    result: BinnedVelocityResult,
    vel_max: float = 9.0,
    n_bins: int = 31,
) -> "matplotlib.figure.Figure":
    """Four-panel figure: histograms and ECDFs for panels A and B."""
    import matplotlib.pyplot as plt

    bins = np.linspace(0, vel_max, n_bins)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    r1lo, r1hi = result.region1_bounds
    r2lo, r2hi = result.region2_bounds
    thr = result.far_threshold

    # --- Panel A: histogram ---
    ax = axes[0, 0]
    ax.hist(result.startfar_region1, bins=bins, density=True, histtype="step",
            linewidth=1.5, label=f"started far (>{thr} µm)")
    ax.hist(result.startclose_region1, bins=bins, density=True, histtype="step",
            linewidth=1.5, label=f"started close (≤{thr} µm)")
    ax.set_title(f"A: same region ({r1lo}–{r1hi} µm), different start")
    ax.set_xlabel("Speed (µm/min)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    _style(ax, vel_max)

    # --- Panel A: ECDF ---
    ax = axes[1, 0]
    _plot_ecdf(ax, result.startfar_region1, vel_max, label=f"started far", color="C0")
    _plot_ecdf(ax, result.startclose_region1, vel_max, label=f"started close", color="C1")
    p_t = result.ttest_a.pvalue
    p_ks = result.ks_a.pvalue
    ax.set_title(f"t p={p_t:.3g}, KS p={p_ks:.3g}", fontsize=9)
    ax.set_xlabel("Speed (µm/min)")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=7)
    _style(ax, vel_max)

    # --- Panel B: histogram ---
    ax = axes[0, 1]
    ax.hist(result.startfar_region1_b, bins=bins, density=True, histtype="step",
            linewidth=1.5, label=f"region {r1lo}–{r1hi} µm")
    ax.hist(result.startfar_region2, bins=bins, density=True, histtype="step",
            linewidth=1.5, label=f"region {r2lo}–{r2hi} µm")
    ax.set_title(f"B: started far, different regions")
    ax.set_xlabel("Speed (µm/min)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    _style(ax, vel_max)

    # --- Panel B: ECDF ---
    ax = axes[1, 1]
    _plot_ecdf(ax, result.startfar_region1_b, vel_max, label=f"{r1lo}–{r1hi} µm", color="C0")
    _plot_ecdf(ax, result.startfar_region2, vel_max, label=f"{r2lo}–{r2hi} µm", color="C1")
    p_t = result.ttest_b.pvalue
    p_ks = result.ks_b.pvalue
    ax.set_title(f"t p={p_t:.3g}, KS p={p_ks:.3g}", fontsize=9)
    ax.set_xlabel("Speed (µm/min)")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=7)
    _style(ax, vel_max)

    fig.tight_layout()
    return fig


def _plot_ecdf(ax, data, vel_max, **kwargs):
    """Plot empirical CDF, clipped to vel_max."""
    clipped = data[data < vel_max]
    sorted_vals = np.sort(clipped)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.step(sorted_vals, cdf, where="post", **kwargs)


def _style(ax, vel_max):
    ax.set_xlim(0, vel_max)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


# ---------------------------------------------------------------------------
# Permutation test on Panel A
# ---------------------------------------------------------------------------

@dataclass
class PermutationResult:
    """Result of a chromosome-level permutation test."""

    observed_diff: float       # observed difference in means (far - close)
    p_value: float             # two-sided permutation p-value
    n_permutations: int
    null_distribution: np.ndarray  # Δmean under each permutation
    null_ci: tuple[float, float]   # 95% CI of null distribution
    n_far_chroms: int
    n_close_chroms: int


def permutation_test_panel_a(
    cells: list[CellData] | list[TrimmedCell],
    far_threshold: float = 5.0,
    region1_bounds: tuple[float, float] = (3.0, 5.0),
    n_permutations: int = 10000,
    rng_seed: int = 42,
) -> PermutationResult:
    """Chromosome-level permutation test for Panel A.

    Each chromosome contributes its mean velocity in the shared spatial region
    as one data point.  The "started far" vs "started close" labels are then
    shuffled across chromosomes (not frames), respecting the autocorrelation
    structure within each trajectory.
    """
    rng = np.random.default_rng(rng_seed)

    # Collect per-chromosome mean velocity in region 1
    chrom_means: list[float] = []
    chrom_is_far: list[bool] = []

    for cell in cells:
        center = pole_center(cell)
        dt_min = cell.dt / 60.0
        for k in range(cell.tracked):
            pos = cell.chromosomes[:, :, k]
            if np.any(np.isnan(pos)):
                continue
            dist = np.linalg.norm(pos - center, axis=1)
            vel = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt_min
            dist_at_vel = dist[:-1]
            in_r1 = (dist_at_vel >= region1_bounds[0]) & (dist_at_vel < region1_bounds[1])
            if in_r1.sum() == 0:
                continue
            chrom_means.append(float(np.mean(vel[in_r1])))
            chrom_is_far.append(dist[0] > far_threshold)

    means_arr = np.array(chrom_means)
    labels = np.array(chrom_is_far)
    n_far = int(labels.sum())
    n_close = len(labels) - n_far

    observed_diff = float(means_arr[labels].mean() - means_arr[~labels].mean())

    null_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        rng.shuffle(labels)
        null_diffs[i] = means_arr[labels].mean() - means_arr[~labels].mean()

    p_value = float(np.mean(np.abs(null_diffs) >= np.abs(observed_diff)))
    ci = (float(np.percentile(null_diffs, 2.5)), float(np.percentile(null_diffs, 97.5)))

    return PermutationResult(
        observed_diff=observed_diff,
        p_value=p_value,
        n_permutations=n_permutations,
        null_distribution=null_diffs,
        null_ci=ci,
        n_far_chroms=n_far,
        n_close_chroms=n_close,
    )


# ---------------------------------------------------------------------------
# More rigorous test: within-distance-bin partial correlation with time
# ---------------------------------------------------------------------------

@dataclass
class PartialCorrelationResult:
    """Within-distance-bin correlation of velocity with time-since-NEB."""

    bin_centers: np.ndarray
    correlations: np.ndarray          # Spearman rho per distance bin
    p_values: np.ndarray              # per-bin p-values
    counts: np.ndarray                # samples per bin
    fisher_chi2: float                # Fisher combined chi2 statistic
    fisher_p: float                   # Fisher combined p-value
    permutation_p: float | None       # permutation test p-value (if computed)


def _collect_dist_time_vel(
    cells: list[CellData] | list[TrimmedCell],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (distance, time_fraction, velocity) triples from all chromosomes."""
    all_dist, all_time, all_vel = [], [], []
    for cell in cells:
        center = pole_center(cell)
        dt_min = cell.dt / 60.0
        n_frames = cell.chromosomes.shape[0]
        # Normalized time: 0 at start, 1 at end of trajectory
        t_frac = np.arange(n_frames - 1) / max(n_frames - 2, 1)

        for k in range(cell.tracked):
            pos = cell.chromosomes[:, :, k]
            if np.any(np.isnan(pos)):
                continue
            dist = np.linalg.norm(pos - center, axis=1)
            vel = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt_min
            all_dist.append(dist[:-1])
            all_time.append(t_frac)
            all_vel.append(vel)

    return np.concatenate(all_dist), np.concatenate(all_time), np.concatenate(all_vel)


def compute_partial_correlation(
    cells: list[CellData] | list[TrimmedCell],
    dist_bins: np.ndarray | None = None,
    n_permutations: int = 0,
    rng_seed: int = 42,
) -> PartialCorrelationResult:
    """Test whether velocity correlates with time *after* controlling for distance.

    Within each distance bin, compute the Spearman correlation between velocity
    and normalized time-since-NEB.  If velocity is purely spatial, all these
    correlations should be zero.  Combine p-values across bins with Fisher's
    method.

    Optionally run a permutation test: shuffle time labels within each distance
    bin and recompute the aggregate Fisher statistic to get a non-parametric
    p-value.
    """
    all_dist, all_time, all_vel = _collect_dist_time_vel(cells)

    if dist_bins is None:
        dist_bins = np.arange(0, 16, 1.0)

    bin_indices = np.digitize(all_dist, dist_bins) - 1
    n_bins = len(dist_bins) - 1
    bin_centers = 0.5 * (dist_bins[:-1] + dist_bins[1:])

    correlations = np.full(n_bins, np.nan)
    p_values = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    valid_p = []
    for i in range(n_bins):
        mask = bin_indices == i
        counts[i] = mask.sum()
        if counts[i] < 20:
            continue
        rho, p = stats.spearmanr(all_time[mask], all_vel[mask])
        correlations[i] = rho
        p_values[i] = p
        if p > 0:
            valid_p.append(p)

    # Fisher's combined test
    if valid_p:
        chi2 = -2.0 * np.sum(np.log(valid_p))
        fisher_p = float(stats.chi2.sf(chi2, df=2 * len(valid_p)))
    else:
        chi2, fisher_p = np.nan, np.nan

    # Permutation test
    perm_p = None
    if n_permutations > 0 and valid_p:
        rng = np.random.default_rng(rng_seed)
        observed_chi2 = chi2
        count_ge = 0
        for _ in range(n_permutations):
            perm_pvals = []
            for i in range(n_bins):
                mask = bin_indices == i
                if counts[i] < 20:
                    continue
                shuffled_time = all_time[mask].copy()
                rng.shuffle(shuffled_time)
                _, p = stats.spearmanr(shuffled_time, all_vel[mask])
                if p > 0:
                    perm_pvals.append(p)
            if perm_pvals:
                perm_chi2 = -2.0 * np.sum(np.log(perm_pvals))
                if perm_chi2 >= observed_chi2:
                    count_ge += 1
        perm_p = (count_ge + 1) / (n_permutations + 1)

    return PartialCorrelationResult(
        bin_centers=bin_centers,
        correlations=correlations,
        p_values=p_values,
        counts=counts,
        fisher_chi2=chi2,
        fisher_p=fisher_p,
        permutation_p=perm_p,
    )


def plot_partial_correlation(result: PartialCorrelationResult) -> "matplotlib.figure.Figure":
    """Plot within-bin time-velocity correlations."""
    import matplotlib.pyplot as plt

    valid = ~np.isnan(result.correlations)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    ax1.bar(result.bin_centers[valid], result.correlations[valid], width=0.8, alpha=0.7)
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.set_xlabel("Distance to spindle center (µm)")
    ax1.set_ylabel("Spearman ρ (velocity vs time)")
    ax1.set_title("Within-bin velocity–time correlation")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Significance markers
    for i in np.where(valid)[0]:
        if result.p_values[i] < 0.05:
            ax1.text(result.bin_centers[i], result.correlations[i],
                     "*", ha="center", va="bottom", fontsize=12, color="red")

    ax2.bar(result.bin_centers[valid], result.counts[valid], width=0.8, alpha=0.7, color="C1")
    ax2.set_xlabel("Distance to spindle center (µm)")
    ax2.set_ylabel("Sample count")
    ax2.set_title("Samples per distance bin")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    title = f"Fisher combined p = {result.fisher_p:.3g}"
    if result.permutation_p is not None:
        title += f", permutation p = {result.permutation_p:.3g}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
