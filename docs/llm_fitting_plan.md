# Learning interactions from chromosome and centrosome trajectories in mitosis
## Code planning document (Ronceray / SFI-first approach)

## 1. Goal

We want to infer effective interactions from tracked trajectories of:
- chromosomes
- centrosomes

during mitosis, using an overdamped stochastic interacting-particle model.

The initial goal is **not** to build the most flexible predictor possible. The goal is to build the most interpretable and robust inference pipeline that can recover biologically meaningful effective interactions from noisy trajectory data.

This suggests starting with:
1. an overdamped Langevin model,
2. pairwise interaction kernels,
3. constant diffusion (at first),
4. a basis-expansion regression / quasi-likelihood approach,
5. SFI-style noise-aware estimators and model selection.

Neural approaches can be added later as a comparison baseline, but they should not be the first implementation.

---

## 2. Why this approach first

There are two closely related methodological families relevant here:

### A. Interaction-kernel learning (Lu / Maggioni / Tang)
These papers formulate learning pairwise distance-dependent interaction kernels from trajectory data as a nonparametric regression problem, with strong theory for learnability and convergence.

### B. Stochastic Force Inference (Frishman / Ronceray)
These papers formulate inference directly at the SDE level, treating stochasticity seriously and focusing on drift/force and diffusion inference from noisy trajectories, including model selection and robustness to measurement noise.

For our purposes, the two viewpoints are very close mathematically in the pairwise radial setting. The key reason to start from the Ronceray/SFI perspective is practical:
- it is built for stochastic data,
- it explicitly addresses drift + diffusion,
- it includes tools for noisy / discrete / low-time-resolution trajectories,
- it is a good match to microscopy tracks.

---

## 3. First-pass model

Let chromosome positions be \(x_i(t) \in \mathbb{R}^d\), \(i=1,\dots,N_x\), and centrosome positions be \(y_j(t) \in \mathbb{R}^d\), \(j=1,\dots,N_y\).

We begin with an overdamped Langevin model:

\[
dx_i(t) = F_i^{(x)}(X(t),Y(t))\,dt + \sqrt{2D_x}\,dW_i(t),
\]

\[
dy_j(t) = F_j^{(y)}(X(t),Y(t))\,dt + \sqrt{2D_y}\,dB_j(t),
\]

where:
- \(D_x\) and \(D_y\) are constant diffusion coefficients,
- \(F^{(x)}\), \(F^{(y)}\) are unknown effective drifts,
- \(d = 2\) or \(3\), depending on data.

### 3.1 Pairwise radial interaction ansatz

First assume forces are sums of pairwise radial interactions:

For chromosomes:
\[
F_i^{(x)} =
\sum_{k\neq i} f_{xx}(r_{ik})\,\hat r_{ik}
+
\sum_j f_{xy}(\rho_{ij})\,\hat \rho_{ij}
+ \text{optional external/confinement terms},
\]

For centrosomes:
\[
F_j^{(y)} =
\sum_{l\neq j} f_{yy}(s_{jl})\,\hat s_{jl}
+
\sum_i f_{yx}(\rho_{ij})\,(-\hat \rho_{ij})
+ \text{optional external/confinement terms}.
\]

Definitions:
- \(r_{ik} = \|x_k - x_i\|\), \(\hat r_{ik} = (x_k - x_i)/r_{ik}\)
- \(\rho_{ij} = \|y_j - x_i\|\), \(\hat \rho_{ij} = (y_j - x_i)/\rho_{ij}\)
- \(s_{jl} = \|y_l - y_j\|\), \(\hat s_{jl} = (y_l - y_j)/s_{jl}\)

Interpretation:
- \(f_{xx}\): chromosome-chromosome effective interaction
- \(f_{xy}\): centrosome effect on chromosomes
- \(f_{yx}\): chromosome effect on centrosomes
- \(f_{yy}\): centrosome-centrosome interaction

In the simplest symmetric setup, one may enforce \(f_{xy} = f_{yx}\) up to sign conventions, but we should not assume that immediately.

---

## 4. Why pairwise radial first

This is the best complexity / learnability tradeoff.

Advantages:
- interpretable
- low-dimensional compared to general neural drifts
- close to the Lu/Maggioni interaction-kernel setup
- directly compatible with SFI basis projection
- easier to regularize for smoothness and locality
- easier to validate biologically

What this model does **not** capture:
- angle-dependent anisotropy
- explicit spindle-axis directional terms unless added
- history dependence / switching attachment states
- many-body crowding terms
- state-dependent diffusion

These can be added later only if the pairwise model clearly fails.

---

## 5. Basis expansion of kernels

Each unknown radial kernel is represented in a basis.

Example:
\[
f_{xx}(r) = \sum_{m=1}^{M_{xx}} a_m \phi_m(r),
\]
\[
f_{xy}(r) = \sum_{m=1}^{M_{xy}} b_m \phi_m(r),
\]
\[
f_{yy}(r) = \sum_{m=1}^{M_{yy}} c_m \phi_m(r).
\]

Choices for \(\phi_m(r)\):
- cubic B-splines
- compact radial basis functions
- compact Wendland functions
- piecewise linear “hat” functions for debugging

Recommended first implementation:
- cubic B-splines or simple compact piecewise-linear basis
- finite support on \([0, r_{\max}]\)
- modest basis size: e.g. 8–15 basis functions per kernel

Important:
- we should impose a cutoff \(r_c\) or at least basis support over a finite distance range
- we should regularize curvature to avoid noisy oscillatory kernels

---

## 6. Discrete-time regression form

Suppose trajectories are observed at times \(t_n\), with step \(\Delta t = t_{n+1}-t_n\).

Euler-Maruyama gives:
\[
\Delta x_i^n := x_i(t_{n+1}) - x_i(t_n)
\approx
F_i^{(x)}(t_n)\Delta t + \eta_i^n,
\]
with
\[
\eta_i^n \sim \mathcal{N}(0, 2D_x\Delta t\,I),
\]
and similarly for centrosomes.

Therefore
\[
\frac{\Delta x_i^n}{\Delta t}
\approx
F_i^{(x)}(t_n) + \text{noise}.
\]

Because the kernels are expanded in a basis, the drift is linear in coefficients.

### 6.1 Feature construction

For each particle and timepoint, define basis-feature vectors.

For chromosome \(i\), basis feature for chromosome-chromosome term:
\[
g^{xx}_{i,m}(t_n)
=
\sum_{k\neq i} \phi_m(r_{ik}(t_n)) \hat r_{ik}(t_n).
\]

For chromosome-centrosome term:
\[
g^{xy}_{i,m}(t_n)
=
\sum_j \phi_m(\rho_{ij}(t_n)) \hat \rho_{ij}(t_n).
\]

Then
\[
F_i^{(x)}(t_n)
=
\sum_m a_m g^{xx}_{i,m}(t_n)
+
\sum_m b_m g^{xy}_{i,m}(t_n)
+ \cdots
\]

Likewise define \(g^{yy}\), \(g^{yx}\) for centrosomes.

This turns the problem into linear regression for vector-valued responses.

---

## 7. Regression objective

Stack all particle-time increments into one response vector:
\[
V = \left\{ \frac{\Delta x_i^n}{\Delta t}, \frac{\Delta y_j^n}{\Delta t} \right\}.
\]

Stack all features into a design matrix \(G\), so that:
\[
V \approx G\theta + \epsilon,
\]
where \(\theta\) contains all kernel coefficients.

We then solve:
\[
\hat\theta
=
\arg\min_\theta
\|V - G\theta\|^2
+ \lambda_{\text{ridge}}\|\theta\|^2
+ \lambda_{\text{rough}} \theta^\top R \theta.
\]

Where:
- ridge penalty stabilizes correlated features
- roughness penalty discourages high-curvature kernels
- \(R\) is the usual spline roughness matrix or a finite-difference approximation

Possible constraints:
- compact support
- kernel goes to zero at cutoff
- optional sign constraints if biologically justified
- optional symmetry constraints between \(x\leftrightarrow y\)

---

## 8. Learning diffusion

At first, allow separate constant diffusion coefficients:
- \(D_x\) for chromosomes
- \(D_y\) for centrosomes

### 8.1 Two-stage estimate
A robust first route:
1. fit the drift coefficients \(\theta\),
2. estimate diffusion from residuals.

For chromosomes:
\[
\hat D_x
\approx
\frac{1}{2 d N_{\text{obs},x}}
\sum_{i,n}
\frac{\|\Delta x_i^n - \hat F_i^{(x)}(t_n)\Delta t\|^2}{\Delta t}.
\]

Similarly for centrosomes.

### 8.2 Joint estimation
Later, move to a quasi-likelihood formulation:
\[
\mathcal{L}(\theta,D)
=
\sum_{i,n}
\left[
\frac{\|\Delta x_i^n - F_i^{(x)}\Delta t\|^2}{4D_x\Delta t}
+ \frac{d}{2}\log(4\pi D_x\Delta t)
\right]
+ \cdots
\]

SFI-style implementations are useful here because they are designed for drift + diffusion inference from noisy stochastic trajectories.

---

## 9. Measurement noise and why plain increments may fail

This is a major issue for microscopy trajectories.

If observed positions are
\[
x_i^{\text{obs}}(t_n) = x_i(t_n) + \xi_i^n,
\]
with localization noise \(\xi_i^n\), then naïve one-step increments are biased/noisy.

Consequences:
- drift estimation degrades
- diffusion is overestimated
- short-lag increments can be dominated by localization error

This is one of the strongest reasons to use SFI-inspired estimators or multi-lag finite differences.

### 9.1 Practical plan for this
We should implement the plain one-step estimator first for debugging, but the serious pipeline should include:
- multi-point finite-difference estimators
- lagged increments
- optional local temporal smoothing for state evaluation only
- comparison across lags to assess stability

---

## 10. Optional weak density-dependent correction

If pairwise kernels fit poorly, add a small mean-field term rather than jumping to a neural net.

For example, define local density around chromosome \(i\):
\[
\rho_i^{(x)} = \sum_{k\neq i} K_h(\|x_k - x_i\|),
\]
for some smoothing kernel \(K_h\).

Then add:
\[
F_i^{(x)} \leftarrow F_i^{(x)} + u_x(\rho_i^{(x)}) \, \nabla \rho_i^{(x)}
\]
or a simpler scalar density-dependent radial correction.

But this is **not** part of the initial model. It is only a second-stage extension if the pure pairwise model misses systematic collective behavior.

---

## 11. Biologically relevant extensions to keep in reserve

These should be added only after the first pipeline works.

### 11.1 Spindle-axis anisotropy
If interactions depend on the spindle axis \(e(t)\), allow kernels such as:
\[
f(r,\cos\theta)
\]
where \(\theta\) is the angle relative to the spindle axis.

### 11.2 Confinement / geometry terms
Add a smooth effective confinement term:
- toward metaphase plate
- away from cell boundary
- toward/away from spindle midzone

### 11.3 Type subclasses
Chromosomes may not all be equivalent if there are subclasses or attachment states. Do not start here unless labels exist.

### 11.4 Switching / hidden states
Eventually one could add hidden attachment states or phase-dependent kernels, but this should be clearly separated from the first implementation.

---

## 12. Validation strategy

Validation matters more than raw fit quality.

### 12.1 Synthetic-data validation first
Before real data, simulate trajectories from known kernels and known \(D\), then verify recovery.

Synthetic tests should include:
- no-noise ideal case
- realistic diffusion
- localization noise
- missing tracks / partial observations
- finite sampling interval

This is essential.

### 12.2 Real-data diagnostics
For real fits, check:
1. one-step prediction error
2. held-out timepoint likelihood / quasi-likelihood
3. stability of inferred kernels across cells
4. stability across lag choices
5. whether residual increments are approximately Gaussian and white
6. whether inferred kernels are biologically plausible and not oscillatory nonsense

### 12.3 Biological summaries to compare
Forward-simulate from the fitted model and compare:
- chromosome radial distribution
- spacing statistics
- centrosome-chromosome distance distributions
- collective alignment / congression summaries
- mean squared displacement over short lags
- segregation / clustering patterns if relevant

---

## 13. Implementation roadmap

## Stage 0: data representation
Build a clean data API.

Input:
- trajectories for chromosomes and centrosomes
- per-cell arrays
- frame times
- optional mask for missing observations
- optional metadata: mitotic phase, spindle axis, cell ID

Output structure:
- a per-cell trajectory object with methods for:
  - positions at time \(t_n\)
  - pairwise distances
  - observed increments
  - valid indices

Deliverable:
- robust trajectory loader and preprocessing module

---

## Stage 1: simplest working kernel regression
Implement:
- pairwise distance computation
- basis functions
- design matrix construction
- plain one-step increment regression
- ridge regularization
- residual-based diffusion estimates

Deliverable:
- a debug-friendly baseline that works on synthetic data

This stage is for correctness, not final inference quality.

---

## Stage 2: synthetic benchmark suite
Create simulation code for the same model family.

Capabilities:
- simulate overdamped trajectories
- choose known kernels
- add Gaussian localization noise
- subsample frames
- test recovery error as a function of:
  - \(\Delta t\)
  - track length
  - number of particles
  - localization noise
  - regularization strength

Deliverable:
- clear plots of recovered vs true kernels

This stage is mandatory before serious inference on microscopy data.

---

## Stage 3: SFI-style upgrades
Add the features that make the pipeline actually robust:
- multi-lag / multi-point derivative estimators
- quasi-likelihood treatment of drift and diffusion
- optional sparse model selection
- basis selection / information criterion
- uncertainty estimates for coefficients

Possible routes:
1. use parts of the existing SFI package directly, or
2. reproduce the relevant estimators ourselves in our feature language

Deliverable:
- robust inference under realistic microscopy noise

---

## Stage 4: domain-specific structure
After the core pipeline is working, add:
- spindle-axis anisotropy
- weak confinement term
- optional density correction
- phase dependence if needed

At each extension:
- test on synthetic data first
- add only one complexity axis at a time

---

## Stage 5: neural baseline later
Only after the above:
- build a GNN / NRI-style baseline
- compare predictive accuracy and inferred structure

Reason:
- neural nets may fit better if interactions are not really pairwise radial,
- but they are harder to interpret and easier to overfit,
- so they should be a comparison, not the first model.

---

## 14. Recommended coding structure

Suggested modules:

- `data.py`
  - load trajectories
  - masking / missingness
  - coordinate transforms
  - per-cell objects

- `basis.py`
  - spline / RBF basis
  - support intervals
  - roughness matrices

- `features.py`
  - pairwise distances
  - unit vectors
  - design matrix assembly

- `model.py`
  - coefficient containers
  - kernel evaluation
  - drift evaluation

- `fit.py`
  - ridge / penalized least squares
  - quasi-likelihood fit
  - diffusion estimation

- `simulate.py`
  - forward Euler-Maruyama simulator
  - synthetic benchmark generation
  - localization noise injection

- `validate.py`
  - held-out metrics
  - residual diagnostics
  - recovery metrics
  - forward-simulation comparisons

- `plotting.py`
  - learned kernels
  - confidence bands
  - residual plots
  - simulation-vs-data summaries

---

## 15. Initial modeling choices to use unless they clearly fail

Recommended first defaults:

- overdamped model
- pairwise radial kernels
- separate chromosome and centrosome diffusion constants
- 8–12 basis functions per kernel
- finite interaction cutoff
- ridge + roughness regularization
- synthetic validation before real fitting
- SFI-style noise handling in serious runs
- no neural nets yet
- no hidden states yet
- no anisotropy yet

---

## 16. Open questions we should decide early

1. Are trajectories 2D or 3D?
2. Are all chromosomes treated as identical particles?
3. Are centrosomes exactly two particles, always tracked?
4. Is there a known spindle axis from image analysis?
5. What is the frame interval \(\Delta t\)?
6. What is the estimated localization noise scale?
7. Are tracks long enough for per-cell inference, or should cells be pooled?
8. Do kernels vary systematically by mitotic phase?

These affect identifiability and the design of synthetic benchmarks.

---

## 17. Practical conclusion

The right first implementation is:

1. Build a transparent pairwise-kernel regression pipeline.
2. Validate it thoroughly on synthetic stochastic trajectories.
3. Upgrade to SFI-style estimators for noisy microscopy data.
4. Only then consider richer structure or neural baselines.

This gives the best balance of:
- interpretability,
- data efficiency,
- robustness to stochasticity,
- extensibility to more complex mitotic mechanisms later.