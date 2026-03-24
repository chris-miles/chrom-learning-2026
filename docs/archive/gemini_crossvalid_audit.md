# Audit of Cross-Validation Practices in `nb04` against Yates et al. (2022)

## 1. Holistic Summary of Best Practices (Yates et al., 2022)
Based on a review of *Ecological Monographs - 2022 - Yates - Cross validation for model selection: A review with examples from ecology*, the key recommendations for cross-validation in regression/modeling settings are:

1. **Structured Data Splitting (Leave-One-Group-Out):** When model residuals are not independent due to temporal, spatial, or hierarchical grouping structures, data splitting must respect this structure. The paper strongly recommends **leave-one-group-out (LOGO)** or blocked cross-validation to assess predictive performance on new, unseen groups.
2. **Minimizing Bias:** Leave-one-out (LOO) or approximate LOO minimizes bias. If $k$-fold is used, $k$ should be as large as practically possible ($k \ge 10$). 
3. **Bias Correction:** When blocked or group-out CV leaves out a large proportion of the data (e.g., > 10% of the dataset per fold), score estimation bias increases, and formal bias correction (e.g., Burman 1989) is needed.
4. **Mitigating Overfitting (The Modified One-Standard-Error Rule):** Selecting the model with the absolute best cross-validation score often leads to overfitting due to score-estimation uncertainty. The paper strongly advocates for using calibrated selection—specifically the **modified one-standard-error rule**—which selects the simplest model whose predictive performance is within one standard error of the best-scoring model.

---

## 2. Assessment of `notebooks/04_model_selection.py` (`nb04`)

We evaluated the cross-validation strategy in `nb04` against the best practices outlined above.

### A. Data Splitting: Leave-One-Cell-Out CV
**Status: Excellent Alignment**
The data consists of time-series trajectories of chromosomes grouped by cell. `nb04` correctly identifies this hierarchical/grouped structure and implements `cross_validate(cells, configs)` as a **leave-one-cell-out** cross-validation over the 13 cells. This perfectly aligns with Yates et al.'s recommendation to use leave-one-group-out (LOGO) for structured data to assess the marginal performance on new cells.

### B. Bias Correction
**Status: Acceptable / No Action Required**
By leaving out 1 cell out of 13, the cross-validation deletes approximately ~7.7% of the data in each fold. Yates et al. suggest that bias correction is typically required if the left-out block exceeds 10% of the data. Thus, `nb04` is safely under this threshold, and omitting bias correction is justifiable.

### C. Overfitting Mitigation & Model Selection Rule
**Status: Needs Improvement (Recommendation: Adopt One-Standard-Error Rule)**
Currently, `nb04` selects the "winning" topology strictly by looking at the minimum Mean Squared Error (MSE):
```python
sorted_topo = sorted(TOPOLOGIES, key=lambda t: cv_results[t].mean_error)
best_cv = cv_results[sorted_topo[0]].mean_error
```
While `nb04` correctly calculates and prints the standard error of the MSE (`MSE={r.mean_error:.8f} ± {r.std_error:.8f}`), it **does not use this uncertainty to calibrate model selection**. 

**Actionable Advice:** To fully align with Yates et al., `nb04` should implement the *one-standard-error rule*. If a simpler topology (e.g., `poles` or `center`) has an MSE that is within one standard error (`r.std_error`) of the absolute best, more complex topology (e.g., `poles_and_chroms` or `center_and_chroms`), the simpler model should be selected to avoid overfitting.

### D. Multi-Step / Rollout Validation
**Status: Beyond Standard Practice (Excellent)**
While Yates et al. primarily discuss standard predictive scores, `nb04` implements forward-simulated **rollout validation** (scoring multi-step trajectories and endpoint Wasserstein distances). This goes beyond standard 1-step cross-validation and provides rigorous, domain-specific validation of the models' physical plausibility.

## Conclusion
Our implementation in `nb04` is highly robust and correctly handles the hierarchical nature of the data using leave-one-group-out (cell) cross-validation. The primary area for improvement is moving away from strictly picking the "lowest MSE" model to adopting the **one-standard-error rule** favored by Yates et al., which systematically penalizes complexity and mitigates overfitting.