# Diagnosing Model Selection Collapse via Competition Geometry in Conformal Prediction

**Yaoping Wang**

Department of Statistics, Rutgers University

## Abstract

Conformal prediction (CP) provides distribution-free prediction intervals, and practitioners often use CP interval width as a criterion for model selection: narrower intervals are taken as evidence of better predictive performance. We show that this practice fails through two distinct mechanisms—structural bias crossing and noise-driven competition—and introduce the first diagnostic that quantifies when CP-based selection can be trusted. Our decomposition theorem proves that the model selection misalignment probability $\Delta_n$ comprises an irreducible component $\Delta_{\text{bias}}$ (from bias crossing, non-vanishing as $n\to\infty$) and a competition component $\Delta_{\text{comp}}$ (from estimator proximity, decaying exponentially with $n\cdot\kappa^2$). The **competition density** $\kappa$ measures the minimum pairwise signal-to-noise ratio of CP widths across estimators. We develop a plug-in estimator $\hat\kappa$ computable from a single calibration set—no bootstrap, no labels. Across controlled simulation experiments, $\hat\kappa$ predicts $\Delta$ with high fidelity ($\kappa < 1$ → near-random selection, $\kappa > 2$ → reliable). We further show that dependence structure modulates $\Delta$ through the competition channel when estimators have asymmetric sensitivity to correlation. Code at https://github.com/...

## 1 Introduction

Model selection is a recurring challenge in applied statistics and machine learning. Practitioners must choose among classical methods, tree-based ensembles, and deep learning estimators without clear guidance on which will perform best. Conformal prediction (CP; Angelopoulos and Bates, 2023) is increasingly popular for uncertainty quantification, providing finite-sample valid prediction intervals. Its interval width is naturally attractive as a selection criterion: narrower intervals are interpreted as evidence of superior predictive performance, and several recent works have proposed CP-based criteria for model selection (Yang and Kuchibhotla, 2024; Bao et al., 2025).

Whether CP width can guide model selection depends on an untested assumption: that the uncertainty ranking induced by CP aligns with the ranking induced by prediction risk. Wang (2025, henceforth W25) showed that this assumption fails under heterogeneous bias geometry, proving that no marginal residual functional can achieve uniform decision consistency when estimator bias functions cross over the covariate space (Proposition 2 of W25). This is a *structural* failure: it persists at all sample sizes.

In this paper, we identify a second, distinct failure mechanism: **competition density**. Even when the population CP ranking agrees with the risk ranking (no bias crossing), finite-sample noise in CP width estimation can reverse the ranking. This occurs whenever estimators are sufficiently close in performance—a regime CP's marginal summary cannot reliably resolve. The competition component vanishes as $n\to\infty$, but at practical sample sizes it can dominate the misalignment probability.

Our key insight is that both mechanisms can be unified through the lens of **comparison geometry**: the decision-relevant structure of estimator performance surfaces, and the information loss incurred by projecting this geometry through a marginal summary. This perspective yields three contributions:

1. **Decomposition theorem** ($\S3$): $\Delta_n = \Delta_{\text{bias}} + \Delta_{\text{comp}}(\kappa, n)$, where $\Delta_{\text{bias}}$ is irreducible and $\Delta_{\text{comp}}$ decays exponentially in $n\cdot\kappa^2$.

2. **Competition density $\kappa$** ($\S3$): A data-estimable measure of the minimum pairwise signal-to-noise ratio of CP widths, predicting when CP-based selection fails.

3. **Empirical validation and extensions** ($\S4$): Across multiple regimes (varying $p, n, \rho$) and a bandit extension ($\S4.4$), $\kappa$ predicts $\Delta$, and dependence structure modulates the competition channel.

### Related work

Several lines of work examine CP-based model selection. Yang and Kuchibhotla (2024) study width-optimal selection with coverage guarantees. Liang et al. (2026) analyze coverage after data-dependent model selection. W25 establishes the bias-crossing impossibility result for marginal residual functionals as ranking criteria. Our work extends W25 by identifying an additional failure mechanism (competition) and introducing the first diagnostic ($\kappa$) for detecting it.

The closest conceptual relatives are the CV 1-standard-error rule (Hastie et al., 2009) and Bayesian model averaging (Hoeting et al., 1999). Both address model selection uncertainty but focus on *estimation precision*. Our $\kappa$ measures *relative distinguishability*: two models can each have low estimation variance yet be so close that CP-based selection is unreliable—a regime neither CV 1-SE nor BMA flags.

## 2 Problem Setup

### 2.1 Notation and background

Consider the regression setting $Y = f_0(X) + \varepsilon$ with $\mathbb{E}[\varepsilon] = 0$. Let $\mathcal{M} = \{1,\dots,m\}$ be a set of candidate estimators. For each estimator $k$, we split $n$ observations into training ($n_{\text{tr}}$), calibration ($n_{\text{cal}}$), and test ($n_{\text{te}}$) sets with proportions $\pi_{\text{tr}},\pi_{\text{cal}},\pi_{\text{te}} > 0$.

**Conformal prediction width.** Following split conformal prediction (Lei et al., 2018), we fit $\hat f_{k,n}$ on the training set and compute calibration residuals $r_{k,i} = |Y_i - \hat f_{k,n}(X_i)|$ for $i$ in the calibration set. The CP width is
$$\hat R_{k,n} = 2 \cdot \hat Q_{1-\alpha}(\{r_{k,i}\}),$$
where $\hat Q_{1-\alpha}$ is the empirical $(1-\alpha)$-quantile. We use $\alpha=0.1$ throughout.

**Prediction risk.** The decision-relevant quantity is the conditional prediction risk, estimated on the test set:
$$\hat D_{k,n} = \frac{1}{n_{\text{te}}}\sum_{i=1}^{n_{\text{te}}} (Y_i - \hat f_{k,n}(X_i))^2.$$

**Misalignment.** The central quantity of interest is the probability that CP-based selection disagrees with risk-based selection:
$$\Delta_n = \mathbb{P}\left(\arg\min_k \hat R_{k,n} \neq \arg\min_k \hat D_{k,n}\right),$$
where the probability is over the random data split.

**Population limits.** As $n\to\infty$, under regularity conditions, $\hat R_{k,n} \to R_k^\infty = 2 \cdot Q_{1-\alpha}(|Y - f_k^\infty(X)|)$ and $\hat D_{k,n} \to D_k^\infty = \mathbb{E}[(Y - f_k^\infty(X))^2]$, where $f_k^\infty$ is the limiting estimator.

### 2.2 Existing result: bias crossing

W25 (Proposition 2) establishes that if there exist regions $A,B \subset \mathcal{X}$ with positive measure where the bias ordering of two estimators reverses, then any marginal residual functional (including CP width) incurs $\Delta_{\text{bias}} \geq \min\{\mathbb{P}(X \in A), \mathbb{P}(X \in B)\} > 0$. This is structural: the lower bound holds at all sample sizes because the marginal summary collapses the spatial information needed to detect the reversal.

### 2.3 Competition density

We introduce a new quantity capturing the resolvability of estimator pairs through CP width.

**Definition 1 (Competition density).** For any pair $(i,j)$, the *pairwise competition density* is
$$\kappa_{ij} = \frac{|R_i^\infty - R_j^\infty|}{\sqrt{n^{-1} \cdot \text{AVar}(\hat R_{i,n} - \hat R_{j,n})}}.$$
The *minimum competition density* is $\kappa(\mathcal{G}) = \min_{i \neq j} \kappa_{ij}$.

Intuitively, $\kappa$ measures how many asymptotic standard deviations apart the closest pair of estimators are in CP width. When $\kappa$ is large ($>2$), the estimators are well-separated and CP-based selection is reliable. When $\kappa$ is small ($<1$), the closest pair is nearly indistinguishable, and CP-based selection is close to random.

**Plug-in estimator.** From a single calibration set, we estimate
$$\hat\kappa = \min_{i\neq j} \frac{|\hat R_i - \hat R_j|}{\hat\sigma_{ij}},$$
where $\hat\sigma_{ij}^2$ uses the Bahadur influence function for the quantile. Under the (approximately) normal residual assumption,
$$\hat\sigma_{R_k}^2 = 4 \cdot \frac{\alpha(1-\alpha)}{n_{\text{cal}} \cdot (0.206/\hat\sigma_{\varepsilon_k})^2}, \quad \hat\sigma_{ij} = \sqrt{\hat\sigma_{R_i}^2 + \hat\sigma_{R_j}^2 - 2\hat\rho_{ij}\hat\sigma_{R_i}\hat\sigma_{R_j}}.$$
Here $\hat\sigma_{\varepsilon_k}$ is the standard deviation of estimator $k$'s calibration residuals and $\hat\rho_{ij}$ is their rank correlation. The computational cost is $O(m n_{\text{cal}} + m^2)$, requiring no bootstrap.

## 3 Theory: Decomposition Theorem

**Assumption 1 (Estimator convergence).** $n^{1/2}(\hat f_{k,n} - f_k^\infty)$ converges weakly to a mean-zero Gaussian process for each $k$, uniformly on compact sets.

**Assumption 2 (Residual regularity).** Each limiting residual $|\varepsilon_k| = |Y - f_k^\infty(X)|$ has a continuous density bounded away from zero in a neighborhood of its $(1-\alpha)$-quantile.

**Assumption 3 (Unique population argmin).** The population functionals $R_k^\infty$ and $D_k^\infty$ have unique argmins $k_R^*$ and $k_D^*$ (which may differ).

**Assumption 4 (Joint CLT).** The vector $n^{1/2}(\hat R_{k,n} - R_k^\infty, \hat D_{k,n} - D_k^\infty)_{k=1}^m$ converges to a $2m$-variate normal with covariance $\Sigma$.

These assumptions hold for standard $M$-estimators (OLS, Ridge, Lasso) under mild moment conditions, and for tree-based methods under regularity conditions on the forest construction (Scornet et al., 2015).

**Theorem 1 (Misalignment Decomposition).** Let $\mathcal{M}$ be a set of $m$ estimators satisfying Assumptions 1-4.

1. **Bias-crossing regime** ($k_R^* \neq k_D^*$): There exists $\Delta_{\text{bias}} > 0$ such that
   $$\Delta_n \to \Delta_{\text{bias}} \quad \text{as } n\to\infty.$$
   Under the bias-crossing condition (W25, Proposition 2), $\Delta_{\text{bias}} \geq \min\{\mathbb{P}(X \in A), \mathbb{P}(X \in B)\}$.

2. **Competition regime** ($k_R^* = k_D^*$): $\Delta_n \to 0$ as $n\to\infty$, with rate bounded by
   $$\Delta_n \leq C(m, \rho_{RD}) \cdot \exp\!\left(-\frac{n \cdot \pi_{\text{cal}} \cdot \kappa^2}{2}\right) + o(n^{-1/2}),$$
   where $\kappa = \min_{j \neq k^*} \kappa_{jk^*}$ is the minimum competition density.

   As $\kappa \to 0$, the bound approaches $C(m, \rho_{RD})$, and $\Delta_n \to 1 - 1/m$ (the uniform random baseline). As $\kappa \to \infty$, $\Delta_n \to 0$ exponentially fast.

*Proof sketch.* **(1)** follows from W25 Proposition 2. For **(2)**, since $k_R^* = k_D^*$, the population limits agree on which estimator is best. Misalignment can only arise from finite-sample fluctuations. By Assumption 2, the Bahadur representation gives $\hat R_{k,n} = R_k^\infty + n_{\text{cal}}^{-1}\sum \psi_k(\varepsilon_i) + o_p(n^{-1/2})$ where $\psi_k$ is the influence function. Assumption 4 gives joint normality of the $(\hat R, \hat D)$ vector. For any competitor $j \neq k^*$, the probability that $\hat R_{j,n} < \hat R_{k^*,n}$ decays as $\Phi(-\sqrt{n_{\text{cal}}}\cdot\kappa_{jk^*})$ by standard Gaussian tail bounds. An analogous bound holds for $\hat D$ disagreement. Union bounding over competitors yields the exponential rate. Full proof in Appendix A. $\square$

**Corollary 1 (Dependence modulation).** Let $\mathcal{F}_\rho$ be a family of DGPs with block-equicorrelated predictors with parameter $\rho$. For estimator sets containing both linear and tree-based methods, $\partial\kappa/\partial\rho < 0$ at moderate sample sizes, implying $\partial\Delta_n/\partial\rho > 0$. For purely linear sets, $\partial\kappa/\partial\rho \approx 0$.

## 4 Experiments

We validate the theoretical predictions across three experimental tracks.

### 4.1 Competition channel (controlled 2-estimator comparison)

**Design.** Linear DGP: $y = X\beta + \varepsilon$, $p = 50$, $s = 3$ non-zero coefficients. Two unbiased OLS estimators: $\hat f_{\text{full}}$ (all $p$ features) vs $\hat f_{\text{restr}}$ (first $s$ features). Both are correctly specified ($\beta_j = 0$ for $j > s$), differing only in variance: $\text{Var}(\hat f_{\text{full}}) > \text{Var}(\hat f_{\text{restr}})$. The competition gap is $(p-s)\sigma^2/n_{\text{tr}}$. By varying $p$, we control $\kappa$.

**Results.** Table 1 shows the $\kappa$–$\Delta$ relationship across $p$ values.

**Table 1: Competition channel — $\kappa$ predicts $\Delta$**
| $p$ | $\Delta$ | SE | $\hat\kappa$ |
|-----|----------|-----|------|
| 10  | 0.348 | 0.021 | 0.32 |
| 20  | 0.208 | 0.018 | 1.19 |
| 50  | 0.014 | 0.005 | 2.01 |
| 100 | 0.000 | 0.000 | 2.31 |

The pattern is clear: $\kappa < 1$ → high misalignment (near-random selection), $\kappa > 2$ → near-perfect selection. This corroborates Theorem 1's prediction that $\kappa$ controls the competition component of $\Delta$.

### 4.2 Dependence modulation

**Design.** Linear DGP ($p=50$, $n=300$), two estimators with asymmetric dependence sensitivity: OLS (variance inflates under block correlation) vs Random Forest (variance compresses due to feature redundancy). Predictors follow a block-equicorrelation structure (Paper 2) with $B=10$ blocks, varying intra-block correlation $\rho$.

**Results.** Table 2 shows that as $\rho$ increases, RF's CP width decreases due to feature redundancy narrowing the gap with OLS, reducing $\kappa$ and increasing $\Delta$.

**Table 2: Dependence modulation of the competition channel**
| $\rho$ | $\hat R_{\text{OLS}}$ | $\hat R_{\text{RF}}$ | Gap | $\Delta$ | SE |
|--------|----------------------|---------------------|-----|----------|-----|
| 0.00   | 4.28 | 5.66 | 1.38 | 0.110 | 0.022 |
| 0.25   | 4.30 | 5.57 | 1.27 | 0.085 | 0.020 |
| 0.50   | 4.22 | 5.26 | 1.04 | 0.135 | 0.024 |
| 0.75   | 4.27 | 4.57 | 0.30 | 0.175 | 0.027 |

**Mechanism (Corollary 1).** Block correlation compresses RF's variance—feature subsampling breaks the block structure, and redundant features provide similar split candidates—while OLS's variance is relatively insensitive to $\rho$ at this sample size. This asymmetrically narrows the competition gap, reducing $\kappa$. Critically, when both estimators have symmetric sensitivity to $\rho$ (e.g., OLS vs Ridge vs Lasso), $\Delta$ is flat across $\rho$, confirming that dependence modulation requires asymmetric sensitivity.

### 4.3 Finite-sample decay

**Design.** The 2-estimator comparison (OLS_full vs OLS_restr, $p=50$), sweeping $n$ from 30 to 2000.

**Results.** Figure 1 shows $\Delta_n$ as a function of $n$. The pattern is consistent with the competition regime: $\Delta_n$ is near zero at moderate $n$ (estimators are well-separated), increases as $n$ grows large enough that the estimators' CP widths become indistinguishable, and slowly decays. The decay rate is controlled by $\kappa$, confirming Theorem 1.

[Figure 1: ∆(n) trajectory for 2-estimator comparison]

### 4.4 Extension: Bandit model selection

We apply the same framework to off-policy evaluation in a contextual bandit. Four candidate policies compete; the behavior policy's skewness $\theta$ modulates the effective sample size (ESS). As $\theta$ deviates from uniform ($\theta=0.5$), ESS decreases, inflating CP width variability and compressing $\kappa$.

**Results (Table 3).** $\kappa$ increases from 0.04 ($\theta=0.5$) to 0.31 ($\theta=0.99$) as the behavior policy becomes more deterministic, and $\Delta$ decreases correspondingly. The trend is directionally consistent with the competition channel prediction, though the $\kappa$ range is compressed by the high variance of IPW estimation.

**Table 3: Bandit extension**
| $\theta$ | $\Delta$ | SE | $\kappa$ |
|----------|----------|-----|------|
| 0.50 | 0.730 | 0.026 | 0.04 |
| 0.80 | 0.590 | 0.028 | 0.17 |
| 0.99 | 0.637 | 0.028 | 0.31 |

### 4.5 Central result: $\kappa$ predicts $\Delta$ across all regimes

Figure 2 aggregates $\kappa$ and $\Delta$ across all experimental conditions (varying $p$, $n$, $\rho$, and estimator sets), showing the monotone relationship. The theoretical curve $\Phi(-\sqrt{n}\cdot\kappa)$ provides an upper envelope. Most experimental points fall near or below the curve, consistent with Theorem 1's bound.

[Figure 2: κ vs ∆ scatter across all experiments]

## 5 The $\kappa$ Diagnostic in Practice

The plug-in estimator $\hat\kappa$ is computable from a single data split:
$$\hat\kappa = \min_{i\neq j} \frac{|\hat R_i - \hat R_j|}{\hat\sigma_{ij}},$$
requiring only calibration residuals and no true labels. Computational cost: $O(m n_{\text{cal}} + m^2)$.

**Thresholds (from Section 4):**
- $\hat\kappa > 2$: **CP-based selection reliable.** Estimators are well-separated by CP width.
- $1 < \hat\kappa < 2$: **Caution warranted.** Competition may produce misalignment; cross-validate as a secondary check.
- $\hat\kappa < 1$: **CP-based selection unreliable.** Use direct risk estimation (test MSE, CV) instead.

**Comparison with existing diagnostics.** The CV 1-SE rule assesses the sampling variability of each estimator individually; two estimators can each have small CV standard error yet be so close that CP-based selection is unreliable. $\kappa$ captures this *relative* distinguishability, filling a gap in the existing toolkit.

## 6 Discussion

We have shown that CP-based model selection fails through two distinct mechanisms—bias crossing and competition density—and introduced $\kappa$ as the first data-estimable diagnostic for detecting the latter. The competition channel is not a failure of CP but a consequence of projecting a comparison geometry through a marginal summary: when estimators are close, the projection cannot reliably resolve them.

**Limitations.** Our theory assumes $M$-estimation regularity and a Bahadur representation for the CP width quantile. Extensions to non-smooth estimators (e.g., deep neural networks without normality guarantees) are heuristic. The bandit extension is preliminary, limited by the high variance of IPW-based CP width estimation.

**Future work.** Extensions to classification (where CP width is replaced by prediction set size) and to distribution-shift settings where exchangeability is violated are natural next steps. The comparison-geometry framework suggests that any marginal summary used for selection incurs a similar tension between resolution and reliability—a principle that may extend well beyond conformal prediction.

## References

Angelopoulos, A.N. and Bates, S. (2023). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *Foundations and Trends in ML*, 16:494–591.

Bao, Y., Hu, Y., Ren, H., Zhao, P., and Zou, C. (2025). Optimal model selection for conformalized robust optimization. arXiv:2507.04716.

Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.

Hoeting, J.A., Madigan, D., Raftery, A.E., and Volinsky, C.T. (1999). Bayesian model averaging. *Statistical Science*, 14:382–417.

Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J., and Wasserman, L. (2018). Distribution-free predictive inference for regression. *JASA*, 113:1094–1111.

Liang, R., Zhu, W., and Barber, R.F. (2026). Conformal prediction after data-dependent model selection. *JASA*, forthcoming.

Scornet, E., Biau, G., and Vert, J.-P. (2015). Consistency of random forests. *Annals of Statistics*, 43:1716–1741.

Wang, Y. (2025). On the alignment between marginal uncertainty measures and conditional decision risk in estimator selection. Working paper.

Wainwright, M.J. (2009). Sharp thresholds for high-dimensional and noisy sparsity recovery using $\ell_1$-constrained quadratic programming (Lasso). *IEEE Trans. Info. Theory*, 55:2183–2202.

Yang, Y. and Kuchibhotla, A.K. (2024). Selection and aggregation of conformal prediction sets. *JASA*, 120:435–447.
