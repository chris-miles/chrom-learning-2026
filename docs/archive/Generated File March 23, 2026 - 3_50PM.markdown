# Cross-Validation Audit of `04_model_selection.py`

This report provides a holistic assessment of the cross-validation and model selection procedures used in `notebooks/04_model_selection.py`, evaluated against the best practices outlined in the provided reference: *Yates et al. (2022). Cross validation for model selection: A review with examples from ecology*.

## Executive Summary
The model selection pipeline implemented in the `chromlearn` notebook is highly sophisticated and aligns well with several advanced recommendations from the statistical ecology literature. By explicitly testing out-of-sample forward simulations (Rollout CV) alongside step-wise velocity predictions, the project avoids many common pitfalls. 

However, the methodology currently relies on an "absolute minimum score" selection criterion. According to Yates et al., this significantly increases the risk of overfitting due to score-estimation uncertainty. Adopting a calibrated selection rule (such as the modified one-standard-error rule) is the primary recommendation to fully align the notebook with current best practices.

---

## 1. Strengths: Alignment with Best Practices

### A. Data-Splitting Scheme for Structured Data
* **Yates et al. Principle:** When data exhibit structural dependencies (e.g., spatial, temporal, or hierarchical/grouped structure), ordinary randomized $k$-fold cross-validation suffers from data leakage, leading to underestimated prediction errors. *Leave-one-group-out* (LOGO) or blocked cross-validation is strictly recommended.
* **Audit of `nb04`:** The notebook uses a **Leave-One-Cell-Out** cross-validation scheme (`cross_validate(cells, ...)`). Since the trajectory data contains strong intra-cell temporal autocorrelation but cells are independent, leaving out an entire cell perfectly matches the LOGO recommendation. It correctly evaluates the *marginal predictive performance*—the model's capacity to generalize to a completely unseen biological sample.

### B. Choice of Predictive Scores
* **Yates et al. Principle:** Model selection should be based on strictly proper scoring rules. For continuous regression settings with Gaussian errors, squared error and log-likelihood are appropriate and theoretically robust. 
* **Audit of `nb04`:** The notebook correctly utilizes Mean Squared Error (MSE) on the step-wise velocity predictions (via the Stochastic Force Inference projection). Furthermore, the notebook augments this with **Rollout Validation** (calculating out-of-sample axial/radial trajectory MSE, endpoint error, and final Wasserstein distances). This provides an exceptional, multi-faceted reality check on the model's physical consistency that goes above and beyond standard statistical score evaluation.

---

## 2. Areas for Improvement: Gaps & Recommendations

### A. Mitigating Overfitting via Calibrated Selection Rules (Critical)
* **The Problem:** The notebook currently selects the "winner" by simply ranking the mean CV MSE (`best_cv = cv_results[sorted_topo[0]].mean_error`) and picking the absolute minimum. Yates et al. heavily emphasize that **picking the absolute minimum score leads to overfitting** due to score-estimation uncertainty. When comparing models with different complexities (e.g., `poles` with ~10 parameters vs. `poles_and_chroms` with ~20 parameters), the complex model might slightly edge out the simpler one simply due to random sample variation, not true generalizability.
* **Yates et al. Recommendation:** Use a calibrated selection rule, such as the **Modified One-Standard-Error (m-OSE) rule** or calculate the standard error of the score differences ($\sigma_m^{	ext{diff}}$).
* **Actionable Fix for `nb04`:**
  Instead of picking `sorted_topo[0]`, calculate the standard error of the difference in fold errors between the best model and simpler candidates.
  ```python
  import numpy as np
  
  best_topo = min(TOPOLOGIES, key=lambda t: cv_results[t].mean_error)
  best_errors = cv_results[best_topo].held_out_errors
  n_folds = len(cells)
  
  for topo in TOPOLOGIES: # preferably ordered from simplest to most complex
      diffs = cv_results[topo].held_out_errors - best_errors
      mean_diff = np.mean(diffs)
      se_diff = np.std(diffs, ddof=1) / np.sqrt(n_folds)
      
      # If mean_diff is within 1 SE of 0, the models are statistically indistinguishable.
      # You should select the simpler topology if it falls in this range.
  ```
  Implementing this will ensure that the 20-parameter models are only selected if they provide a *statistically meaningful* improvement over the 10-parameter models.

### B. Consideration of Bias Correction
* **The Problem:** Cross-validation generally overestimates the expected loss because the model is trained on less data than the full dataset. Yates et al. note that if the left-out block is large (e.g., >10% of the data), models with higher complexity are disproportionately penalized, which can lead to underfitting.
* **Audit of `nb04`:** Leaving 1 cell out of 13 removes ~7.7% of the data per fold. This is borderline. Because the topologies differ by a factor of 2 in parameter count (10 vs 20), this 7.7% reduction *might* slightly and unfairly penalize the models that include chromosome-chromosome interactions.
* **Actionable Fix:** The bias is likely small, but it is worth noting in the text that Leave-One-Cell-Out is slightly conservative for the more complex models. If precision is paramount, one could implement Burman's bias correction (Equation 5 in Yates et al.), though the current Rollout CV provides enough corroborating evidence that this is likely unnecessary.

### C. Post-Selection Inference Risks
* **Yates et al. Principle:** Using a single dataset to make model selection decisions and then making inferences on the parameter estimates (e.g., confidence intervals) of the winning model results in biased effect sizes and artificially tight confidence intervals (Box 1: Valid post-selection inference).
* **Audit of `nb04`:** The notebook plots the bootstrap confidence bands for the kernels. Fortunately, `nb04` mitigates the worst of the "post-selection inference" bias by bootstrapping and plotting the kernels for *all* topologies independently to check for physical plausibility (e.g., a repulsive barrier at $r < 1.5 \mu m$). 
* **Actionable Fix:** When writing the final paper, ensure that if you present the bootstrap confidence intervals of the *selected* model, you explicitly acknowledge that these intervals do not account for selection uncertainty. Alternatively, presenting the full model (`poles_and_chroms`) as the primary inference object—as recommended by some statisticians in Box 1—circumvents this issue entirely.

---

## Conclusion
The `chromlearn` pipeline exhibits a robust, modern approach to model validation, particularly through its use of Leave-One-Cell-Out and Rollout CV. To elevate the analysis to fully comply with modern ecological statistical guidelines, the adoption of a **One-Standard-Error Rule** based on the paired cross-validation fold differences is highly recommended. This will definitively protect the study against claims of overfitting the interaction topologies.
