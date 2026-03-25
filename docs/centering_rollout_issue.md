# Centering Rollout Issue

## Purpose of This Note

This memo records the current understanding of the "rollout drifts away from
the spindle center" issue seen in `notebooks/04_model_selection.py`, so we do
not keep rediscovering the same arguments.

It is intentionally long and self-contained. It summarizes:

- what the symptom is,
- what code path produces it,
- what was checked directly,
- what the quantitative diagnostics say,
- why this does not contradict notebook 03,
- what Alex/Chris concluded about lab frame vs spindle frame,
- what we are and are not planning to change.

Short version:

- This does **not** currently look like a sign bug.
- The effect persists in deterministic rollouts, so it is **not** Monte Carlo
  noise from a bad random draw.
- The most likely explanation is a **model/data limitation**:
  the current pairwise distance-only chromosome model can capture some weak
  residual centering behavior, but it does not capture the full common
  transport of chromosomes with a translating/turning spindle, especially in
  hard cells like `rpe18_ctr_509`.
- We are **not** switching the main analysis wholesale into spindle
  coordinates. The main analysis stays in the lab frame.

## The Symptom

In the qualitative rollout section of `notebooks/04_model_selection.py`, some
cells, especially `rpe18_ctr_509`, show a striking pathology:

- the thick black line (real mean in spindle-frame coordinates) stays roughly
  centered axially near `x = 0`,
- but the thick dashed model rollout mean drifts away from the spindle center,
- and this happens for **all** four model topologies,
- with both axial and radial spindle-frame summaries looking wrong.

At first glance this looks impossible, because the learned chromosome-partner
kernel looks centering/attractive in the kernel plots.

That prompted the question:

> Is this a bug in the code, or is the data/model just weird?

## The Relevant Code Path

The important pieces of the implementation are:

### Qualitative rollouts in notebook 04

`notebooks/04_model_selection.py`:

- fits the four topologies on all cells,
- for selected example cells, runs **one simulated rollout per topology**,
- converts both real and simulated trajectories into spindle-frame
  coordinates,
- plots thin per-chromosome traces plus thick means over all chromosomes.

Important detail:

- the qualitative panel uses **one rollout only** per topology for each cell.
- however, the thick dashed curve is still the mean over **all chromosomes** in
  that rollout, not the mean of the displayed subset.

### Simulator

`chromlearn/model_fitting/simulate.py`:

- simulates chromosome trajectories in the **lab frame**,
- uses real partner trajectories from the held cell (`poles` or pole midpoint),
- evaluates forces from the current geometry,
- advances with Euler-Maruyama:

`X(t + dt) = X(t) + F(X(t)) * dt + sqrt(2 D dt) * xi`

The chromosome-partner force direction is built from:

`partner_position - chromosome_position`

so a positive `f_xy(r)` is attractive toward the current partner.

### Spindle-frame coordinates

`chromlearn/io/trajectory.py`:

- computes the instantaneous pole midpoint,
- computes the instantaneous pole-to-pole axis,
- projects chromosome positions relative to that moving midpoint/axis into:
  - `axial`: signed coordinate along the spindle axis,
  - `radial`: distance from the spindle axis.

So the rollout is simulated in the lab frame, but displayed in a
**moving spindle frame**.

## Direct Checks That Were Performed

The following checks were run specifically to distinguish a code bug from a
model/data issue.

### 1. The effect is not a random one-off stochastic rollout

The first suspicion was that the qualitative plot only shows one stochastic
rollout per model, so perhaps `rpe18_ctr_509` just got unlucky noise.

That is not what happens.

For `rpe18_ctr_509`, the same basic drift remains when the rollout is repeated
with:

- the same fitted model,
- the same real partner trajectories,
- but `D_x = 0` (deterministic rollout, no diffusion noise).

So the excursion is not a random Monte Carlo artifact.

### 2. The effect is not a `nanmean` / missing-data artifact

For `rpe18_ctr_509`, all 46 chromosomes are finite at all 156 frames in the
trimmed window.

So the thick black and dashed means are not being distorted by time-varying
numbers of valid chromosomes.

### 3. The force sign path looks correct

The code path is internally consistent:

- notebook 04 documents the sign convention as
  `f(r) * (x_j - x_i) / r`, so positive means attractive,
- the simulator uses `partner - position` for the direction vector,
- then adds `f_xy(r) * direction_xy`.

That is the expected attractive sign convention.

No sign inversion was found in the rollout code.

### 4. The kernel itself is attractive at the relevant distances

For the full-data `center` fit, the learned `f_xy(r)` is positive at the
distances most chromosomes occupy in `rpe18_ctr_509`. For example:

| Distance `r` (um) | `f_xy(r)` |
| --- | ---: |
| 0.5 | 0.00356 |
| 1.0 | 0.00132 |
| 2.0 | 0.00169 |
| 3.0 | 0.00148 |
| 4.0 | 0.00037 |
| 5.0 | 0.00420 |
| 6.0 | 0.01283 |
| 8.0 | 0.02211 |
| 10.0 | 0.01286 |
| 12.0 | -0.00970 |

And the mean chromosome-to-center distance in `rpe18_ctr_509` is roughly:

| Time | Mean distance to pole midpoint (um) |
| --- | ---: |
| 0 s | 4.786 |
| 250 s | 3.804 |
| 500 s | 3.688 |
| 750 s | 3.640 |

So at the typical `3-6 um` distances occupied by this cell, the kernel is
indeed attractive.

### 5. The bad cell is a real spindle-motion outlier

For `rpe18_ctr_509`:

- 156 frames,
- 46 chromosomes,
- `dt = 5 s`.

Compared with the 13 `rpe18_ctr` cells, `509` ranks:

| Metric | Rank among 13 cells | Value |
| --- | --- | ---: |
| Total spindle-center displacement | 2 / 13 | 6.079 um |
| Cumulative spindle-center path length | 3 / 13 | 16.049 um |
| Mean spindle-center speed | 6 / 13 | 0.0207 um/s |
| Total spindle-axis turning | 3 / 13 | 3.281 rad |
| Net spindle-axis reorientation | 3 / 13 | 1.219 rad |

So this is not just a random ugly plot. It is one of the harder cells in the
dataset from the standpoint of spindle translation and reorientation.

### 6. Across cells, rollout error increases with spindle-center motion

For the full-data `center` model, if one computes a simple per-cell rollout
error and compares it with how much the spindle center moves, the relationship
is strong:

- correlation between rollout error and total spindle-center displacement:
  about `0.90`,
- correlation between rollout error and total spindle-axis turning:
  about `0.54`.

This strongly suggests that the worst rollout failures are concentrated in
cells where the spindle itself moves a lot.

## Concrete Numbers for `rpe18_ctr_509`

The real spindle-frame mean trajectory for `509` stays fairly close to center:

| Time | Real mean axial (um) | Real mean radial (um) |
| --- | ---: | ---: |
| 0 s | -0.534 | 4.654 |
| 100 s | -0.802 | 3.619 |
| 200 s | -0.211 | 3.525 |
| 300 s | -0.234 | 3.359 |
| 400 s | -0.192 | 3.421 |
| 500 s | -0.502 | 3.435 |
| 600 s | -0.635 | 3.438 |
| 700 s | -0.344 | 3.475 |
| 775 s | -0.253 | 3.429 |

By contrast, deterministic (`D_x = 0`) rollouts already drift strongly:

| Topology | Deterministic axial mean at 500 s (um) | Deterministic radial mean at 500 s (um) |
| --- | ---: | ---: |
| `poles` | 1.350 | 3.124 |
| `center` | 1.165 | 3.365 |
| `poles_and_chroms` | 0.807 | 3.739 |
| `center_and_chroms` | 0.314 | 4.012 |

The same sign of failure appears across all four topologies:

- the simulated mean drifts to positive axial values in spindle coordinates,
- while the real mean stays modestly negative and near center.

## What This Means Mechanistically

The key point is:

- the simulation is performed in the **lab frame**,
- but the plotted coordinates are in the **moving spindle frame**.

That distinction matters.

### Simplified midpoint-only picture

Let:

- `x(t)` = chromosome position in lab coordinates,
- `c(t)` = spindle center / pole midpoint,
- `y(t) = x(t) - c(t)` = position relative to the spindle center.

If the model says

`dx/dt = F(x - c(t)) + noise`

then

`dy/dt = F(y) - dc/dt + noise`

So even if `F(y)` points toward the current spindle center, the cloud can still
lag the moving center if `dc/dt` is large relative to the restoring drift.

### Full spindle-frame picture

The actual plotted coordinates are not just `x - c`. They are projected onto a
time-varying spindle axis. So the moving-frame dynamics also inherit a frame
rotation term.

Qualitatively:

- midpoint translation contributes a `-dc/dt` term,
- spindle-axis rotation contributes an additional transport term,
- neither is explicitly represented by a pure radial distance-based kernel.

So it is entirely possible for the force to be attractive in the lab frame
while the trajectory still moves away from zero in spindle-frame coordinates.

## Direct Evidence for "lagging a moving frame"

The strongest evidence came from inspecting the deterministic `center` rollout
around the visibly bad interval near `475-520 s`.

During that interval:

- the mean restoring motion toward the center is only about
  `0.002-0.003 um/s`,
- while the spindle midpoint is moving by roughly
  `0.01-0.03 um/s` axially,
- and the spindle also has radial/rotational motion of comparable scale.

In other words:

- the force points the right way,
- but it is too weak to keep the simulated chromosome cloud comoving with the
  measured spindle motion in this cell.

That is why both axial and radial spindle-frame summaries can look wrong even
with an attractive `f_xy(r)`.

## Why This Does Not Contradict Notebook 03

This was the main conceptual sticking point in the chat, so it is worth being
explicit.

Notebook 03 shows:

- the **chromosome-cloud center of mass** follows the **pole center of mass**
  with a lag,
- and therefore centrosome motion is a sensible external driver for the
  chromosome problem.

Notebook 03 does **not** show:

- that each individual chromosome behaves like a stiff spring tethered to the
  pole midpoint,
- or that a per-chromosome radial kernel should be able to reproduce all
  chromosome transport.

That distinction matters a lot.

### COM-level coupling is weaker evidence than it first appears

The lag-correlation analysis in notebook 03:

- reduces the chromosome cloud to a single center-of-mass trajectory,
- reduces the poles to a single midpoint trajectory,
- smooths both,
- then compares their velocities.

This is useful and correct for the intended causal point:

- chromosomes follow centrosomes,
- not the other way around.

But it is an aggregate statement about the **cloud center of mass**, not a
direct test of a per-chromosome centering spring.

### Why `x(t) - c(t) approx 0` is misleading here

The tempting argument was:

> If chromosomes stay centered while the spindle center moves, shouldn't the
> restoring stiffness `k` have to be large?

That would only be decisive if `x(t) - c(t)` referred to the position of each
chromosome relative to the midpoint.

But in the actual data:

- individual chromosomes sit several microns away from the midpoint because the
  metaphase cloud is broad,
- it is mainly the **cloud COM** that stays near the spindle center.

So "chromosomes stay centered" at the cloud level does **not** imply that the
per-chromosome distance-to-center force must be a strong spring.

### The more precise interpretation

The data support:

- strong **velocity-level coupling** between chromosome-cloud motion and
  spindle motion,

but they do not necessarily support:

- a strong **position-dependent restoring force** from each chromosome to the
  spindle midpoint.

Those are different physical pictures.

The current pairwise `f(r)` model encodes the second picture, not the first.

## Competing Explanations for the Small Learned Centering Strength

The current understanding is that there are at least two plausible
contributors, and they are not mutually exclusive.

### A. Omitted common transport / mechanical attachment

Chromosomes may stay near the spindle center primarily because they are
mechanically carried with the spindle through kinetochore-microtubule
attachment, not because a large position-dependent force pulls them back toward
the midpoint.

In that picture:

- the dominant effect is common transport with the spindle,
- the learned `f_xy(r)` captures only weak residual centering dynamics,
- long rollouts fail because the pairwise radial kernel is missing the main
  transport mechanism.

This is the cleanest explanation of why:

- notebook 03 sees strong spindle-following at the cloud level,
- but notebook 04 still learns a weak centering kernel and has poor rollouts in
  moving-frame coordinates.

### B. Windowing / regime mixing / pooled-cell heterogeneity

There is also a less structural explanation.

The current trimmed window is `neb_ao_frac = 0.5`, i.e. the midpoint of the
NEB-AO interval.

If that window includes:

- an early active congression regime,
- followed by a longer, quieter late-prometaphase / early-metaphase regime,

then the regression is averaging across regimes with different signal-to-noise.

Likewise, fitting one shared kernel across all cells mixes:

- fast-moving spindles,
- slower spindles,
- different amounts of axis turning,
- different transport environments.

That can flatten the globally fitted residual centering signal even if some
subsets of the data have stronger effective centering.

This possibility is plausible and should not be ignored.

## Current Best Interpretation

The best synthesis of all checks so far is:

1. The rollout pathology is **real**.
2. It is **not** a simple implementation bug.
3. It is most visible in cells where the spindle itself moves a lot.
4. It likely reflects a **model/data limitation**, not a broken sign
   convention:
   - either omitted common transport,
   - or pooled/regime-mixed fits,
   - or both.

So the right framing is:

> The current pairwise chromosome model learns a weak residual centering effect,
> but it does not fully reproduce bulk chromosome transport with a moving
> spindle in hard cells such as `rpe18_ctr_509`.

## Lab Frame vs Spindle Frame

This was explicitly discussed with Alex Mogilner by email on 2026-03-24.

### Alex's framing of the paper questions

Alex separated the project into three questions:

1. Do centrosome-chromosome forces affect centrosome motion?
2. Do chromosomes interact with each other?
3. What are the relative roles of interpolar vs astral forces for centrosome
   motion?

His view was:

- question 1 should be handled in the **lab frame**,
- question 2 could in principle be done in either frame,
- the previous PNAS paper already addressed the spindle-centric chromosome
  problem under a simpler assumption set.

### Current decision

We should **not** switch the main analysis wholesale into the spindle frame.

The current decision is:

- keep the main chromosome fitting and model-selection story in the **lab
  frame**,
- keep notebook 03's conclusion that centrosomes are valid external drivers,
- treat spindle-frame or advection-augmented fits as optional **sensitivity
  analyses**, not as the main pipeline.

### Why not just "fix" it by switching frames?

A spindle-frame fit may be useful diagnostically, but it should not be sold as
solving the problem.

If one subtracts spindle transport first, then the model is only being asked to
explain residual within-spindle dynamics. That may be scientifically useful,
but it is a different target.

The danger is that a spindle-frame rollout can look good simply because the
frame change supplied the dominant transport mechanism "for free."

So:

- spindle-frame fitting is not forbidden,
- but it is not the right replacement for the main lab-frame story.

## What We Are Doing With This in the Paper/Code

### Main stance

For now:

- keep the lab-frame code,
- keep one-step LOOCV as the primary quantitative topology criterion,
- keep rollout failures as a diagnostic of model-class limitations,
- do **not** present the issue as a resolved code bug.

### How to talk about the pathology

The right language is something like:

> In some cells with large spindle translation/reorientation, long-horizon
> forward simulations drift in spindle-frame coordinates even though the learned
> chromosome-partner kernel is attractive. Deterministic checks indicate this is
> not a force-sign bug but a limitation of the current pairwise model class in
> reproducing common chromosome transport with the moving spindle.

### What not to claim

We should **not** claim:

- that the model is broken because the dashed line drifts,
- that a spindle-frame reformulation has been shown to be the correct fix,
- that notebook 03 proves the per-chromosome centering stiffness must be large.

## Recommended Next Checks if We Revisit This

These are optional, not urgent.

### 1. Endpoint sensitivity

Check whether shortening the trajectory window strengthens the residual
centering signal.

The existing debugging notebook `notebooks/debug_centering_vs_frac.py` is the
right place for this.

Reason:

- if `neb_ao_frac = 0.5` includes too much quiet late prometaphase / early
  metaphase, the active centering signal may be diluted.

### 2. Deterministic overlays in notebook 04

If the qualitative rollout panel remains confusing, a small supplemental plot
showing deterministic `D_x = 0` rollouts for a few cells would make it obvious
that the issue is not just diffusion noise.

### 3. Spindle-motion diagnostics for hard cells

For example, add:

- pole-midpoint trajectory,
- midpoint speed,
- spindle-axis turning,

for cells like `rpe18_ctr_509`.

This would visually connect the rollout failure to the motion of the reference
frame itself.

### 4. Optional nuisance-term sensitivity analysis

As a supplemental check only, one could augment the regression with:

- spindle-midpoint velocity,
- perhaps spindle angular velocity,

to test whether explicitly modeling transport reduces the rollout pathology.

Again: this would be a sensitivity analysis, not the new main model.

## Practical Conclusion

There may be no single "fix" here, in the narrow bug-fixing sense.

The current best reading is:

- the data are heterogeneous,
- `rpe18_ctr_509` is a hard cell,
- the current model class is not rich enough to reproduce all long-horizon
  chromosome transport effects,
- and that is itself useful information.

So the action item is mostly **documentation and careful framing**, not a
desperate code rewrite.

## Final Takeaway

If this issue comes up again, the default answer should be:

1. No, it does not currently look like a sign bug.
2. Yes, the effect is reproducible in deterministic rollouts.
3. Yes, the force is attractive at the relevant distances.
4. The pathology is strongest in cells with unusually large spindle motion.
5. This does not contradict notebook 03, because notebook 03 is a cloud-level
   spindle-following result, not proof of a strong per-chromosome centering
   spring.
6. We are keeping the main analysis in the lab frame.
7. Treat this as a documented model/data limitation unless future targeted
   sensitivity analyses show otherwise.
