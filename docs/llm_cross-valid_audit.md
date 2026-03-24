# Cross-Validation Audit Synthesis for `nb04`

## Scope
This document consolidates four inputs:

- the Yates et al. review: *Cross validation for model selection: A review with examples from ecology* (Ecological Monographs, 2022; read from `docs/Ecological Monographs - 2022 - Yates - Cross validation for model selection A review with examples from ecology.pdf` via `pdftotext`)
- `docs/Generated File March 23, 2026 - 3_50PM.markdown`
- `docs/gemini_crossvalid_audit.md`
- a direct code audit of `notebooks/04_model_selection.py`, `chromlearn/model_fitting/fit.py`, `chromlearn/model_fitting/plotting.py`, with some supporting context from `notebooks/05_robustness.py` and `llm_audit_2026-03-23.md`

## Bottom Line
The main cross-validation backbone in `nb04` is good: leave-one-cell-out is the right grouped split for these data, and one-step held-out MSE is a defensible regression score. The notebook is not failing because it uses the wrong fold structure.

The main weaknesses are higher-level:

1. It picks raw minima instead of using a calibrated selection rule.
2. It mixes several quantitative metrics and prints several different “winners,” which is not a clean paper story.
3. It leaks held-out information through full-dataset basis-domain estimation.
4. It labels fold-to-fold spread as `std_error`, even though the code stores a standard deviation, not a standard error.

For the paper, the cleanest story is:

- **Primary model-selection criterion:** leave-one-cell-out one-step velocity MSE.
- **Secondary quantitative checks:** leave-one-cell-out rollout metrics.
- **Secondary qualitative / physics checks:** full-data forward simulations, kernel-shape sanity checks, and detection of nonphysical interactions such as implausible long-range chromosome-chromosome attraction.

That structure is compatible with Yates and with the scientific need for hard visual sanity checks on difficult long-horizon forecasts.

## What Yates et al. Actually Recommend
The points most relevant here are:

- When data have grouping, temporal dependence, or other structural dependence, the split must respect that structure. For grouped data, leave-one-group-out is recommended.
- Leave-one-out or approximate leave-one-out is preferred when practical because it minimizes bias.
- Bias correction matters mainly when using smaller `k`, or when blocked/grouped deletions are large enough to make the training set much smaller than the full data set.
- Raw “pick the minimum CV score” selection overfits because score differences are noisy. Yates recommend calibrated selection, specifically their modified one-standard-error rule (m-OSE), or closely related paired score-difference uncertainty calculations.
- Post-selection inference is biased: once a model is selected, uncertainty on its parameter estimates no longer reflects the full selection uncertainty.
- Score choice should match the modeling target. For least-squares regression, squared error is acceptable; for a full predictive-density story, log score is the gold standard.

## Audit of the Current `nb04`

### 1. Fold structure
**Assessment:** Strong alignment with best practice.

`cross_validate` leaves out one entire cell per fold and evaluates on that held-out cell (`chromlearn/model_fitting/fit.py:205-270`). That is the correct grouped split if the generalization target is a new cell. It avoids the obvious leakage that would happen if timepoints from the same cell were split across train and test.

`rollout_cross_validate` uses the same leave-one-cell-out structure for forward simulation (`chromlearn/model_fitting/fit.py:273-380`). That is also appropriate as a secondary holdout check.

### 2. Primary score choice
**Assessment:** Mostly strong, with one important paper-story clarification.

The primary one-step score is held-out mean squared velocity prediction error (`chromlearn/model_fitting/fit.py:256-264`; `notebooks/04_model_selection.py:205-248`). For this least-squares regression problem, that is a reasonable predictive score. It is also much cleaner than the rollout metrics because:

- it directly matches the fitted objective
- it is deterministic once the fold is defined
- it supports paired fold-by-fold comparisons between topologies
- it avoids extra Monte Carlo noise

A good feature of the current implementation is that it computes one loss per held-out cell and then averages across cells, rather than pooling all observations across all cells. That means the target is “performance on a new cell,” which is the right scientific unit here.

For the paper, this should be the single main quantitative criterion.

### 3. Bias correction
**Assessment:** Acceptable as currently implemented.

With 13 cells, leave-one-cell-out removes about 7.7% of the cells per fold. That is not a regime where I would treat bias correction as mandatory. Yates do note that grouped/blocked deletions can need correction when the deleted block is large, and give 10% as a useful example threshold rather than a hard law.

So the current omission of formal bias correction is defensible. I would not make this the main methodological concern.

### 4. Selection rule and uncertainty accounting
**Assessment:** This is the main quantitative weakness.

`nb04` ranks models by the raw mean CV score and then reports a `1-step CV winner`, a `Rollout winner`, and an `Endpoint winner` (`notebooks/04_model_selection.py:664-697`). That is not a coherent selection protocol.

Two concrete problems:

- The code uses raw minimum score selection rather than a calibrated rule.
- The uncertainty object is mislabeled. `CVResult.std_error` is computed as `np.nanstd(errors)` in `chromlearn/model_fitting/fit.py:266-269`, so it is a fold-level standard deviation, not a standard error. `plot_cv_curve` then plots that quantity as an error bar (`chromlearn/model_fitting/plotting.py:62-68`), and the notebook prints it with `±` language (`notebooks/04_model_selection.py:211-220`).

This matters because Yates’ recommendation is not “compare means using unpaired error bars.” It is to use the covariance-aware uncertainty of **paired fold differences** or m-OSE. Since every topology is evaluated on the same held-out cells, the correct uncertainty target is the foldwise difference in losses between models.

The two Gemini audits were directionally right that the notebook should not just pick the minimum CV mean. They were not careful enough about the fact that the current code does not actually compute a standard error.

### 5. Leakage through basis-domain estimation
**Assessment:** Small but real methodological leakage.

Before any CV is run, `nb04` estimates `r_min_xx`, `r_max_xx`, and the topology-specific `xy` basis domains using **all cells**, including the cell that will later be held out in each fold (`notebooks/04_model_selection.py:74-131`). Those domains are then frozen into `FitConfig` once (`notebooks/04_model_selection.py:178-194`), and `cross_validate` builds fold-specific bases from that precomputed config (`chromlearn/model_fitting/fit.py:224-229`).

This is a form of preprocessing leakage. It is probably not the dominant issue in this project, but it is not ideal. If the held-out cell contains unusually large distances, it can influence the support of the basis used in its own fold.

Best practice is one of:

- recompute basis domains inside each training fold
- or pre-specify a fixed domain from external knowledge and use it for every fold

### 6. Rollout validation
**Assessment:** Good secondary validation, not ideal as the primary selection metric.

The leave-one-cell-out rollout validation is a legitimate out-of-sample forecast check (`chromlearn/model_fitting/fit.py:273-380`; `notebooks/04_model_selection.py:584-655`). It is scientifically useful because one-step accuracy alone does not guarantee plausible trajectories.

But it should remain secondary because:

- it is not the fitted objective
- it uses several different summary losses rather than one clearly predeclared score
- it includes Monte Carlo noise from finite simulation replicates (`ROLLOUT_REPS = 4` in `notebooks/04_model_selection.py:391-394`)
- some rollout diagnostics are better interpreted as task-specific summaries than as the main formal model-selection score

So rollout validation is valuable, but not the cleanest single criterion for the paper.

### 7. Full-data forward simulations and kernel sanity checks
**Assessment:** Keep these, but label them correctly.

There are two useful qualitative checks in `nb04`:

- representative single-rollout plots for selected cells (`notebooks/04_model_selection.py:431-480`)
- aggregate full-data forward simulations across all cells (`notebooks/04_model_selection.py:483-580`)

These are **not** rigorous holdout validation, because they use `models[topology]` fit on all cells (`notebooks/04_model_selection.py:196-202`) and then simulate those same cells. But that does not make them useless. They are important visual validations because long-horizon forward simulation is hard, and these plots can reveal:

- obvious instability
- trajectory shapes that are qualitatively wrong
- unrealistic radial or axial drift
- nonphysical learned interactions that only become obvious when rolled forward

The same logic applies to physics sanity checks. The existing notebook already checks for forbidden short-range chromosome-chromosome attraction (`notebooks/04_model_selection.py:287-333`). For the paper, I would keep such checks and broaden them to explicitly discuss implausible long-range `xx` attraction as a coarse-graining artifact/sanity issue rather than a selection metric.

So these plots should be presented as **qualitative validation / pathology detection**, not as the primary quantitative evidence.

### 8. Post-selection inference
**Assessment:** Needs an explicit caveat in the paper.

The bootstrap bands are descriptive and useful (`notebooks/04_model_selection.py:254-358`). But if the paper first selects a topology and then interprets the selected topology’s kernel confidence bands as if the model had been fixed a priori, that will understate uncertainty.

Yates are explicit on this point. The safest wording is that the bootstrap intervals are conditional on the chosen topology and do not include topology-selection uncertainty.

### 9. Hyperparameter tuning and `nb05`
**Assessment:** Potential issue, depending on how the paper is written.

`nb04` hardcodes `n_basis = 10` and `lambda_ridge = lambda_rough = 1e-3` (`notebooks/04_model_selection.py:175-193`). On its own, that is fine if those are treated as fixed analysis settings.

But `notebooks/05_robustness.py` explicitly sweeps basis size and regularization using the same CV machinery on the same dataset (`notebooks/05_robustness.py:38-104` and `notebooks/05_robustness.py:112-186`).

Therefore:

- if `nb05` is presented only as a robustness/sensitivity analysis after topology selection, no problem
- if those sweeps are used to tune the settings and then the paper reuses the same data to claim a final topology winner, then strict best practice would require nested CV or a frozen analysis plan

## Synthesis of the Prior Audits

### What survives from the Gemini drafts
- They correctly identify leave-one-cell-out as the right grouped split.
- They correctly identify raw-minimum selection as the main methodological gap.
- They correctly view rollout validation as useful and beyond a minimal CV workflow.
- The longer draft is right to raise post-selection inference as a caveat.

### What needed correction or tightening
- Both Gemini drafts treat `std_error` as a standard error, but the code stores a standard deviation.
- Neither Gemini draft caught the basis-domain leakage from estimating domains on the full dataset before CV.
- Neither draft cleanly separates in-sample full-data rollouts from genuine holdout validation.
- Neither draft addressed the fact that the current notebook prints several different “winners,” which is a paper-story problem even if the underlying computations are individually reasonable.
- The shorter Gemini audit treats the 10% bias-correction threshold too mechanically. In Yates it is a practical warning sign, not a hard cutoff.

## Recommended Paper Protocol

### Primary model-selection criterion
Use:

> **Leave-one-cell-out one-step velocity MSE, averaged equally across held-out cells.**

This is the cleanest choice for the paper because it is aligned with the regression target and the grouped CV structure.

### Selection rule
Do **not** pick the raw minimum mean score.

Instead:

1. Predefine a complexity ordering.
2. Compute paired held-out-cell loss differences relative to the best mean-scoring model.
3. Apply either:
   - the modified one-standard-error rule (preferred if implemented cleanly), or
   - a paired standard-error-of-the-difference rule as a close practical approximation

A reasonable complexity ordering here is:

- simpler class: `center`, `poles`
- more complex class: `center_and_chroms`, `poles_and_chroms`

Within a complexity class, choose by score. Across complexity classes, only move to the more complex class if the improvement is large relative to paired uncertainty.

### Secondary quantitative checks
Report these separately and explicitly as secondary diagnostics:

- leave-one-cell-out rollout axial/radial MSE
- leave-one-cell-out endpoint mismatch
- leave-one-cell-out final-frame distribution mismatch

These should support or challenge the primary choice, but not replace the primary criterion unless a model fails badly.

### Secondary qualitative / physics checks
Keep these and present them as valuable sanity checks:

- full-data forward simulations for visual validation
- representative-cell rollouts
- short-range `xx` repulsion sanity check
- explicit discussion of implausible long-range `xx` attraction if it appears

These are scientifically important because long-horizon rollout quality and physical plausibility matter, even when they are not the formal selection score.

### Implementation fixes
Before the paper, I recommend:

1. Rename `std_error` to something like `fold_sd`.
2. Add a true standard error or paired-difference uncertainty calculation for selection.
3. Move basis-domain estimation inside each training fold, or pre-specify domains once from external reasoning.
4. Keep qualitative full-data rollouts, but label them as in-sample visual validation.
5. Avoid printing multiple “winners.” Print one primary winner, then report the secondary checks beneath it.

## Suggested Paper Wording
One clean version is:

> Model topology was selected using leave-one-cell-out cross-validated one-step velocity MSE as the primary criterion. Because the candidate topologies were evaluated on the same held-out cells, model differences were assessed using paired foldwise loss differences and a calibrated parsimony rule rather than by selecting the raw minimum score. Forward rollout metrics and qualitative full-data simulations were treated as secondary validation checks on long-horizon realism and physical plausibility, not as independent primary selection criteria.

That is the story I would recommend carrying through the paper.
