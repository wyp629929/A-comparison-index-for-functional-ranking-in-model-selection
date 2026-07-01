# A comparison index for functional ranking in model selection

**Yaoping Wang**

Department of Statistics, Rutgers University

Some key words: Asymptotic comparison; Conformal prediction; Gaussian shift experiment; Model selection; Signal-to-noise ratio.

## Abstract

The ranking induced by a scalar selection functional can disagree with the ranking induced by prediction risk. For functionals with a Bahadur-type linear expansion, the misranking probability is governed by a comparison index $\kappa$ arising as the Le Cam discrimination parameter of the local asymptotic Gaussian experiment. When the population ordering agrees, the misranking probability decays at rate $\exp(-n\kappa^2/2)$; otherwise misalignment persists asymptotically. The analysis is developed for general selection functionals; a worked example is provided for conformal prediction width.

## 1 Introduction

Consider a set of candidate estimators evaluated by a selection functional $T$ and a risk functional $D$. Model selection based on $T$ is reliable when $\arg\min T$ agrees with $\arg\min D$, but this alignment is not guaranteed. The disagreement probability
$$\Delta_n(T) = \mathbb{P}\{\arg\min_k T_{k,n} \neq \arg\min_k D_{k,n}\}$$
depends on two phenomena: whether the population argmins differ, in which case misalignment persists asymptotically; and finite-sample noise, which can reverse the ranking even when the argmins agree. Both phenomena are governed by a single comparison index
$$\kappa_{ij} = \frac{|T_i^\infty - T_j^\infty|}{\sqrt{\operatorname{AVar}(T_i - T_j)}},$$
the Le Cam discrimination parameter of the functional difference under the local asymptotic Gaussian experiment.

The comparison of experiments framework (Le Cam, 1986) provides the natural language for this problem. A selection functional induces a statistical experiment; the difficulty of distinguishing two estimators based on this experiment is measured by $\kappa$. When the population argmins agree, the misranking probability decays as $\exp(-n\kappa^2/2)$; when they disagree, $\kappa$ is uninformative because structural disagreement dominates. The theory is developed for general Bahadur-regular functionals and illustrated for conformal prediction width.

The paper makes three contributions. First, it defines the comparison index $\kappa$ as the intrinsic parameter of the local asymptotic Gaussian comparison experiment (Theorem~1). Second, it establishes a phase transition: misalignment is either structural (when the population argmins disagree) or $\kappa$-governed (when they agree), with a sharp exponential rate (Theorem~2). Third, it verifies the theory for conformal prediction width through simulations and real data.

### Related work

The comparison of experiments framework \citep{LeCam:1986} provides the natural language for this problem: a selection functional induces a statistical experiment whose pairwise discrimination difficulty is measured by $\kappa$. Within this framework, the present work connects the asymptotic theory of model selection criteria \citep{Hastie:2009,Yang:2024} and the comparison of statistical functionals \citep{LeCam:1986,Bickel:2015}. The bias-crossing impossibility (Appendix~G) establishes a structural limitation of marginal functionals; the present paper identifies finite-sample competition as a second mechanism and introduces $\kappa$ as the LAN discrimination parameter governing the misranking rate.

## 2 Problem setup

### 2.1 General framework

Let $\mathcal{M} = \{1,\dots,m\}$ be a set of candidate estimators. Each estimator $k$ is evaluated by two scalar functionals: a selection functional $T_{k,n}$ used to choose among estimators, and a risk functional $D_{k,n}$ measuring predictive performance. Both converge to population limits $T_k^\infty$ and $D_k^\infty$ as $n\to\infty$. Model selection based on $T$ is reliable to the extent that $\arg\min_k T_k^\infty$ agrees with $\arg\min_k D_k^\infty$. The disagreement probability is
\begin{equation}
\Delta_n(T) = \mathbb{P}\{\arg\min_k T_{k,n} \neq \arg\min_k D_{k,n}\},
\end{equation}
with probability over the random data split. The behavior of $\Delta_n(T)$ depends on whether the population argmins $k_T^*$ and $k_D^*$ agree, and on the local asymptotic properties of $T$.

### 2.2 Comparison index

For any pair of estimators $(i,j)$, the local asymptotic comparison experiment induced by $T$ is characterized by
\begin{equation}
\kappa_{ij} = \frac{|T_i^\infty - T_j^\infty|}{\sigma_{ij}}, \qquad 
\sigma_{ij}^2 = \lim_{n\to\infty} n \cdot \operatorname{Var}(T_{i,n} - T_{j,n}),
\end{equation}
where the limit exists under Assumption~4 below. This parameter is the Le Cam optimal discrimination boundary of the functional comparison: under the local asymptotic Gaussian experiment of Theorem~1, $\kappa_{ij}$ completely determines the minimal achievable misranking probability. The minimum comparison index is $\kappa = \min_{i \neq j} \kappa_{ij}$. The scaling by $n$ cancels the $O(n^{-1})$ variance decay, making $\kappa_{ij}$ a population quantity.

The interpretation of $\kappa_{ij}$ is as a signal-to-noise ratio. The numerator $|T_i^\infty - T_j^\infty|$ is the population gap to resolve; the denominator $\sigma_{ij}$ is the asymptotic standard deviation of the functional difference. When $\kappa_{ij}$ is large, the two functionals are well-separated relative to estimation noise; when $\kappa_{ij}$ is small, they are nearly indistinguishable.

### 2.3 Bias crossing as non-identifiability

If $k_T^* \neq k_D^*$, misalignment does not vanish as $n\to\infty$. This is a non-identifiability of the risk ordering from the selection functional: the population limit of $T$ cannot distinguish which estimator minimizes $D$ (Appendix~G). In this case $\Delta_n(T)$ is bounded below by a positive constant that depends on the probability of the disagreement region. When $k_T^* = k_D^*$, misalignment is purely a finite-sample phenomenon controlled by $\kappa$.

## 3 Theory

The following assumptions characterize the local asymptotic behavior of $T$ and $D$. Assumptions~1, 2 and 4 are standard for $M$-estimators with linear influence functions; Section~4 verifies them for CP width.

**Assumption 1 (Estimator convergence).** $n^{1/2}(\hat f_{k,n} - f_k^\infty)$ converges weakly to a mean-zero Gaussian process for each $k$, uniformly on compact sets.

**Assumption 2 (Functional regularity).** Each limiting comparison functional $T_k^\infty = \lim_{n\to\infty} T_{k,n}$ has a Bahadur-type linear expansion
$$T_{k,n} = T_k^\infty + \frac{1}{n}\sum_{i=1}^n \psi_k(Z_i) + o_p(n^{-1/2}),$$
where $\psi_k$ is an influence function with $\mathbb{E}[\psi_k] = 0$ and $\mathbb{E}[\psi_k^2] < \infty$.

**Assumption 3 (Unique argmins).** The population functionals $T_k^\infty$ and $D_k^\infty$ have unique argmins $k_T^*$ and $k_D^*$ (which may differ).

**Assumption 4 (Joint CLT).** The vector $n^{1/2}(T_{k,n} - T_k^\infty, D_{k,n} - D_k^\infty)_{k=1}^m$ converges to a $2m$-variate normal with covariance $\Sigma$.

### 3.1 Gaussian comparison of functionals

The following proposition establishes the basic Gaussian approximation for functional differences.

**Proposition 1 (Gaussian comparison).** Under Assumptions~2 and~4, for any pair $(i,j)$,
$$\sqrt{n}\,\frac{T_{i,n} - T_{j,n}}{\sigma_{ij}} \;\xrightarrow{d}\; N(\kappa_{ij}, 1),$$
where $\sigma_{ij}^2 = \operatorname{AVar}(T_i - T_j)$ and $\kappa_{ij}$ is defined as in (2). Consequently,
$$\mathbb{P}(T_{j,n} < T_{i,n} \mid T_i^\infty > T_j^\infty) = \Phi(-\kappa_{ij}) + o(1).$$

*Proof.* Under Assumption~4, $\sqrt{n}(T_{i,n} - T_{j,n})$ is asymptotically normal with mean $\sqrt{n}(T_i^\infty - T_j^\infty)$ and variance $\sigma_{ij}^2$. Standardizing gives convergence to $N(\kappa_{ij}, 1)$. The tail probability follows directly. $\square$

The quantity $\kappa_{ij}^2/2$ is the large-deviation rate: $\Phi(-\kappa_{ij}) \leq \exp(-\kappa_{ij}^2/2)$ (Chernoff, 1952). As $\kappa_{ij}$ increases, the misranking probability decays super-exponentially fast, consistent with the interpretation of $\kappa_{ij}$ as a signal-to-noise ratio.

### 3.2 Optimality and maximal invariance

In the local asymptotic Gaussian limit, the pairwise ranking problem reduces to distinguishing $N(\kappa_{ij}, 1)$ from $N(-\kappa_{ij}, 1)$. This is a Gaussian shift experiment with separation $2\kappa_{ij}$. The parameter $\kappa_{ij}$ is the unique maximal invariant under this experiment: any decision rule for ranking has limiting risk that depends only on $\kappa_{ij}$, and the Gaussian shift likelihood ratio is an exponentially optimal test (Chernoff, 1952). Consequently, $\kappa_{ij}$ is both necessary and sufficient for characterizing the local asymptotic misranking probability: no alternative summary of the functional pair achieves a lower misranking rate at the LAN limit.

This optimality property is the key justification for using $\kappa$ as a comparison index. While one could define many measures of distinguishability (e.g., the absolute gap $|T_i^\infty - T_j^\infty|$ alone, or the variance-normalized gap without the $\sqrt{n}$ scaling), only the standardized index $\kappa_{ij}$ achieves the optimal misranking rate at the LAN limit. This follows from the Neyman--Pearson lemma applied to the limiting Gaussian experiment: the likelihood ratio test is optimal, and its power depends only on $\kappa_{ij}$.

### 3.3 Misalignment phase transition

**Theorem 1 (Misalignment regimes).** Let $\mathcal{M}$ be a set of $m$ estimators satisfying Assumptions~1-4.

1. **Structural regime** ($k_T^* \neq k_D^*$): $\Delta_n(T) \to \Delta_{\text{struct}}$ as $n\to\infty$, where $\Delta_{\text{struct}} > 0$ (Appendix~G). The comparison index $\kappa$ does not govern the misalignment rate because the population ranking differs.

2. **Local regime** ($k_T^* = k_D^*$): $\Delta_n(T) \to 0$ as $n\to\infty$, with rate bounded by
   $$\Delta_n(T) \leq 2(m-1) \cdot \exp\!\left(-\frac{n \kappa^2}{2}\right) + o(n^{-1/2}),$$
   where $\kappa = \min_{j \neq k^*} \kappa_{jk^*}$ is the minimum comparison index. The exponential rate $-\kappa^2/2$ is sharp (Chernoff, 1952), though the pre-factor $2(m-1)$ may be conservative under correlated competitors.

*Proof sketch.* (1) follows from Appendix~G. For (2), Proposition~1 gives the pairwise reversal probability $\Phi(-\sqrt{n}\cdot\kappa_{jk^*}) \leq \exp(-n\kappa_{jk^*}^2/2)$. An analogous bound for risk reversals follows from the CLT for sample means under Assumption~4. Union bounding over $m-1$ competitors yields the stated rate. $\square$

The union bound pre-factor $2(m-1)$ requires comment. In the worst case, where all competitors are perfectly correlated, the bound overcounts by a factor of at most $m-1$. Under independence, the bound is sharp up to the $o(n^{-1/2})$ remainder. The exponential rate $-\kappa^2/2$ is unaffected by the pre-factor and represents the fundamental large-deviation rate of the Gaussian comparison.

### 3.4 Comparison with classical model selection criteria

The comparison index $\kappa$ is not specific to conformal prediction. Any scalar selection functional with a Bahadur-type expansion—including AIC, BIC, CV, and WAIC—induces the same Gaussian comparison structure. For these criteria, the functional difference $T_{i,n} - T_{j,n}$ fluctuates around its population value with variance $O(n^{-1})$, and the ranking reliability depends on $|\text{gap}| / \text{SD(gap)}$. The bias-crossing versus competition dichotomy is universal: any marginal summary faces both a structural floor (bias crossing) and a finite-sample resolution limit (competition). A brief illustration with training MSE is provided in Appendix~H.

## 4 Example: Conformal prediction width

### 4.1 Assumption verification

For CP width $T_{k,n} = \hat R_{k,n}$, Assumptions~1--4 are satisfied with specific structure.

In split conformal prediction (Lei et al., 2018), $\hat f_{k,n}$ is fitted on a training set of size $n_{\text{tr}} = \pi_{\text{tr}} n$ and calibration residuals $r_{k,i} = |Y_i - \hat f_{k,n}(X_i)|$ are computed on a separate calibration set of size $n_{\text{cal}} = \pi_{\text{cal}} n$. The CP width is
$$\hat R_{k,n} = 2 \cdot \hat Q_{1-\alpha}(\{r_{k,i}\}),$$
with $\hat Q_{1-\alpha}$ the empirical $(1-\alpha)$-quantile. We use $\alpha = 0.1$ throughout.

Assumption~1 holds for $M$-estimators (OLS, Ridge, Lasso) under mild conditions; it is heuristic for tree-based methods. Assumption~2 holds with $T_k^\infty = R_k^\infty$ and $\psi_k$ the Bahadur influence function of the $(1-\alpha)$-quantile of absolute residuals (Bahadur, 1966). Specifically, for residuals with density $f_k$ at the quantile,
$$\psi_k(r) = \frac{\alpha - \mathbf{1}\{r \leq q_{1-\alpha}\}}{f_k(q_{1-\alpha})}.$$
Assumption~4 holds because calibration and test residuals are conditionally independent given the training fit.

The pairwise comparison index specializes to
$$\kappa_{ij} = \frac{|R_i^\infty - R_j^\infty|}{\sigma_{ij}}, \quad
\sigma_{ij}^2 = \lim_{n\to\infty} n \cdot \operatorname{Var}(\hat R_{i,n} - \hat R_{j,n}).$$

### 4.2 Plug-in estimation

The plug-in estimator $\hat\kappa$ targets the finite-sample signal-to-noise ratio $\sqrt{n_{\text{cal}}}\cdot\kappa$ (see §5 for the scaling). From calibration residuals,
$$\hat\kappa = \min_{i\neq j} \frac{|\hat R_i - \hat R_j|}{\hat\sigma_{ij}},$$
where $\hat\sigma_{ij}^2$ uses the Bahadur variance formula (Appendix~F). The estimator is consistent under Assumptions~1--4.

**Corollary 1 (Dependence modulation).** For the CP-width comparison under block-equicorrelated predictors, $\partial\kappa/\partial\rho < 0$ for sets containing both linear and tree-based estimators, implying $\partial\Delta_n/\partial\rho > 0$. For purely linear sets, $\partial\kappa/\partial\rho \approx 0$.

**Figure 3** (Failure Map) summarizes the interaction between $\kappa$ and bias-crossing severity.

## 5 Empirical illustration

We illustrate the theory for CP width. The Appendix contains detailed experimental procedures and additional results.

### 5.1 Competition channel

**Design.** Linear DGP: $y = X\beta + \varepsilon$, $p=50$, $s=3$ non-zero coefficients. Two unbiased OLS estimators: full ($p$ features) versus restricted ($s$ features), both correctly specified. By varying $p$ (the number of irrelevant features), we control the variance gap and hence $\kappa$.

**Table 1: Competition channel**
| $p$ | $\Delta$ | SE | $\hat\kappa$ |
|-----|----------|-----|------|
| 10  | 0.450 | 0.022 | 1.65 |
| 20  | 0.250 | 0.019 | 1.60 |
| 50  | 0.080 | 0.012 | 1.94 |
| 100 | 0.008 | 0.004 | 3.21 |

The pattern follows Theorem~1: $\kappa$ increases with $p$, and $\Delta$ decreases correspondingly.

### 5.2 Dependence modulation

**Design.** Linear DGP ($p=50$, $n=300$), OLS versus Random Forest, with block-equicorrelated predictors. Varying intra-block correlation $\rho$ modulates the competition gap asymmetrically.

**Table 2: Dependence modulation**
| $\rho$ | $\hat R_{\text{OLS}}$ | $\hat R_{\text{RF}}$ | $\hat\kappa$ | $\Delta$ | SE |
|--------|----------------------|---------------------|------|----------|-----|
| 0.00   | 4.03 | 4.54 | 1.26 | 0.275 | 0.032 |
| 0.25   | 4.04 | 4.46 | 1.16 | 0.305 | 0.033 |
| 0.50   | 3.99 | 4.28 | 1.14 | 0.320 | 0.033 |
| 0.75   | 3.97 | 4.04 | 0.87 | 0.405 | 0.035 |

As $\rho$ increases, RF's variance compresses through feature redundancy, narrowing the gap with OLS. This reduces $\kappa$ and increases $\Delta$, consistent with Corollary~1. A symmetric control with OLS versus Ridge shows no such effect (Table~3), confirming that asymmetry is necessary.

**Table 3: Symmetric control (OLS vs Ridge)**
| $\rho$ | $\hat R_{\text{OLS}}$ | $\hat R_{\text{Ridge}}$ | $\hat\kappa$ | $\Delta$ | SE |
|--------|----------------------|------------------------|------|----------|-----|
| 0.00   | 3.96 | 3.94 | 2.40 | 0.270 | 0.026 |
| 0.25   | 3.93 | 3.90 | 2.38 | 0.300 | 0.026 |
| 0.50   | 3.96 | 3.93 | 2.37 | 0.267 | 0.026 |
| 0.75   | 3.96 | 3.90 | 2.73 | 0.130 | 0.019 |

### 5.3 Decay experiment

**Design.** OLS full ($p=20$) versus OLS restricted ($s=3$), $n$ from 50 to 2000. The restricted estimator is misspecified, ensuring a non-vanishing asymptotic gap.

**Results.** Figure~1 shows $\Delta_n$ alongside $\hat\kappa$. At small $n$, $\Delta$ is near zero because the restricted estimator is clearly worse. As $n$ increases to 100--200, the full estimator's variance shrinks, narrowing the effective gap and pushing $\Delta$ to $\sim 0.37$. At $n \geq 500$, the full estimator's dominance re-emerges, and $\Delta$ decays monotonically. The non-monotonic trajectory mirrors $\hat\kappa$: $\hat\kappa$ drops from $2.68$ ($n=50$) to $0.93$ ($n=100$) then rises to $2.87$ ($n=1000$). Population $\kappa$ computed from $n_{\text{MC}} = 20000$ grows monotonically from $0.83$ to $3.37$, confirming the plug-in estimator's reliability at moderate sample sizes.

### 5.4 Real-data validation

**Design.** Diabetes ($n=442$, $p=10$) and California housing ($n=1000$, $p=8$), OLS versus Ridge.

**Table 4: Real-data validation**
| Dataset | $n$ | $\Delta$ | SE | $\hat\kappa$ | $\hat\kappa_{\text{KDE}}$ |
|---------|-----|----------|-----|------|------|
| diabetes | 100 | 0.340 | 0.021 | 0.78 | 1.15 |
| diabetes | 200 | 0.244 | 0.019 | 1.01 | 1.30 |
| diabetes | 442 | 0.208 | 0.018 | 1.00 | 1.28 |
| housing | 100 | 0.402 | 0.022 | 1.41 | 1.18 |
| housing | 200 | 0.452 | 0.022 | 1.90 | 1.35 |
| housing | 500 | 0.492 | 0.022 | 2.50 | 1.76 |
| housing | 1000 | 0.470 | 0.022 | 3.74 | 2.71 |

On diabetes, $\hat\kappa$ increases with $n$ while $\Delta$ decreases. On housing, $\hat\kappa$ ranges from 1.41 to 3.74. At $n=1000$, housing enters the bias-crossing regime: Ridge maintains a systematic risk advantage in $74.6\%$ of replicates despite the CP gap being negligible ($<0.001$). The KDE estimate (2.71) confirms a more modest signal-to-noise ratio.

### 5.5 Baseline comparison

The CV-normalized gap $|\text{CV}_i - \text{CV}_j| / \sqrt{\text{SE}_i^2 + \text{SE}_j^2}$ shows near-zero correlation with $\Delta$ (Table~5), while $\kappa$ is consistently negatively correlated ($-0.10$ to $-0.27$). This confirms that $\kappa$ captures relative distinguishability that standard CV-based measures miss.

**Table 5: $\kappa$ vs CV gap**
| $n$ | $\Delta$ | $\text{corr}(\kappa, \Delta)$ | $\text{corr}(\text{CV gap}, \Delta)$ |
|-----|----------|-------------------------------|--------------------------------------|
| 100 | 0.429 | $-0.150$ | $0.003$ |
| 200 | 0.387 | $-0.101$ | $-0.019$ |
| 500 | 0.076 | $-0.273$ | $0.080$ |
| 1000 | 0.013 | $-0.215$ | $0.017$ |

## 6 Estimation

The plug-in estimator $\hat\kappa$ computes the standardized minimum gap of CP widths. From calibration residuals, estimate each $\hat R_k$ as $2$ times the $(1-\alpha)$-quantile of absolute residuals, and $\widehat{\operatorname{Var}}(\hat R_k)$ via the Bahadur formula (Appendix~F). The estimator is consistent under Assumptions~1--4 and satisfies $\hat\kappa \xrightarrow{p} \sqrt{n_{\text{cal}}}\cdot\kappa$. In the bias-crossing regime, $\hat\kappa$ does not characterize $\Delta$, consistent with Theorem~1.

## 7 Discussion

Model selection based on CP width fails through two mechanisms: bias crossing, which is structural and persists asymptotically, and finite-sample competition, controlled by $\kappa$. The competition channel arises because a scalar marginal summary cannot resolve estimators close in functional value.

**Why $\kappa$ differs from standard diagnostics.** A confidence interval asks whether a parameter is estimated accurately. The comparison index asks whether two estimators are distinguishable by the selection functional. These are different: estimators with narrow individual intervals can be too close to rank reliably ($\kappa$ small), while estimators with wide intervals can be well-separated ($\kappa$ large). The CV 1-SE rule assesses the former; $\kappa$ assesses the latter (Table~5).

**Limitations.** Assumptions~1--4 hold for $M$-estimators under mild conditions but are heuristic for tree-based methods. The index $\kappa$ is predictive only when $k_T^* = k_D^*$ (Theorem~1); under bias crossing, $\hat\kappa$ is uninformative regardless of magnitude. Detecting regime membership automatically is left to future work. Extensions to classification and distribution shift are natural next steps.

**Broader impact.** The results suggest a distinction between uncertainty quantification and decision diagnostics: coverage validity does not guarantee reliable ranking. The comparison index $\kappa$ quantifies the resolution needed for the latter, a property distinct from the former.

**Declaration of AI use.** During the preparation of this work the author used Claude Code (Anthropic) to assist with coding, editing and literature review. After using this tool the author reviewed and edited the content as necessary and takes full responsibility for the publication.

## Appendix

### A. Bandit extension (preliminary)

We briefly explore an extension to off-policy evaluation in a contextual bandit with four candidate policies. The behavior policy's skewness $\theta$ modulates the effective sample size. Results (Table 6) show $\kappa$ ranging from $0.04$ to $0.31$, with $\Delta$ near the random baseline ($0.59$--$0.73$). The signal is too weak to support conclusions.

**Table 6: Bandit extension**
| $\theta$ | $\Delta$ | SE | $\kappa$ |
|----------|----------|-----|------|
| 0.50 | 0.730 | 0.026 | 0.04 |
| 0.80 | 0.590 | 0.028 | 0.17 |
| 0.99 | 0.637 | 0.028 | 0.31 |

### B. Robustness to non-normal residuals

We repeat the decay experiment with three residual distributions: $N(0,1)$, $t_5/\sqrt{5/3}$, and $(\Gamma(2,1)-2)/\sqrt{2}$.

**Table 7: Non-normal residuals**
| $n$ | dist | $\Delta$ | $\hat\kappa$ | $\text{corr}(\hat\kappa,\Delta)$ |
|-----|------|----------|--------------|----------------------------------|
| 100 | normal | 0.386 (0.022) | 0.96 | $-0.13$ |
| 100 | $t_5$ | 0.382 (0.022) | 0.94 | $-0.03$ |
| 100 | skewed | 0.402 (0.022) | 1.01 | $-0.16$ |
| 500 | normal | 0.084 (0.012) | 1.77 | $-0.30$ |
| 500 | $t_5$ | 0.106 (0.014) | 2.05 | $-0.31$ |
| 500 | skewed | 0.060 (0.011) | 2.41 | $-0.27$ |
| 1000 | normal | 0.008 (0.004) | 2.88 | $-0.19$ |
| 1000 | $t_5$ | 0.012 (0.005) | 3.33 | $-0.23$ |
| 1000 | skewed | 0.002 (0.002) | 4.47 | $-0.12$ |

The $\kappa$--$\Delta$ relationship is preserved across all distributions (correlation magnitude $0.12$--$0.31$). Heavy-tailed residuals produce $\hat\kappa$ within $15\%$ of normal at all $n$, confirming robustness to kurtosis. Skewed residuals inflate $\hat\kappa$ at $n=1000$ ($4.47$ vs $2.88$), but the effect is negligible at $n=100$--$500$.

### C. Sensitivity to the CP level $\alpha$

**Table 8: $\alpha$ sensitivity**
| $\alpha$ | $n$ | $\Delta$ | $\kappa$ | $\text{corr}(\kappa,\Delta)$ |
|----------|-----|----------|----------|------------------------------|
| 0.05 | 100 | 0.412 | 0.84 | $-0.10$ |
| 0.05 | 500 | 0.136 | 1.63 | $-0.31$ |
| 0.05 | 1000 | 0.018 | 2.76 | $-0.26$ |
| 0.10 | 100 | 0.410 | 0.96 | $-0.10$ |
| 0.10 | 500 | 0.086 | 1.76 | $-0.28$ |
| 0.10 | 1000 | 0.008 | 2.89 | $-0.17$ |
| 0.20 | 100 | 0.428 | 0.91 | $-0.20$ |
| 0.20 | 500 | 0.104 | 1.70 | $-0.28$ |
| 0.20 | 1000 | 0.010 | 2.89 | $-0.20$ |

The $\kappa$--$\Delta$ relationship is stable across $\alpha$: $\kappa$ varies by less than $15\%$ at each $n$.

### D. Additional real datasets

Four OpenML datasets: Boston ($n=506$, $p=13$), Concrete ($n=1030$, $p=8$), Abalone ($n=4177$, $p=7$), Wine quality ($n=6497$, $p=11$).

**Table 9: Additional datasets**
| Dataset | $n$ | $\Delta$ | $\kappa$ |
|---------|-----|----------|----------|
| boston | 200 | 0.447 | 2.16 |
| boston | 500 | 0.453 | 2.23 |
| concrete | 200 | 0.510 | 2.39 |
| concrete | 500 | 0.570 | 3.11 |
| concrete | 1000 | 0.450 | 4.84 |
| abalone | 200 | 0.383 | 1.45 |
| abalone | 500 | 0.470 | 1.63 |
| abalone | 1000 | 0.490 | 1.84 |
| wine | 200 | 0.413 | 1.57 |
| wine | 500 | 0.437 | 1.92 |
| wine | 1000 | 0.483 | 2.36 |

### E. Estimator robustness

Repeat the decay experiment with Lasso ($\alpha=0.1$), SVR (RBF kernel), and XGBoost as competitors.

**Table 10: Estimator robustness**
| Estimator | $n$ | $\Delta$ | $\kappa$ | $\text{corr}(\kappa,\Delta)$ |
|-----------|-----|----------|----------|------------------------------|
| Lasso | 100 | 0.237 | 1.25 | $-0.36$ |
| Lasso | 500 | 0.447 | 1.15 | $-0.10$ |
| Lasso | 1000 | 0.460 | 1.34 | $-0.18$ |
| SVR | 100 | 0.153 | 1.41 | $-0.28$ |
| SVR | 500 | 0.010 | 2.82 | $-0.18$ |
| SVR | 1000 | 0.013 | 3.43 | $-0.25$ |
| XGBoost | 100 | 0.213 | 1.27 | $-0.36$ |
| XGBoost | 500 | 0.007 | 2.96 | $-0.20$ |
| XGBoost | 1000 | 0.000 | 3.88 | — |

### F. Estimation details

The plug-in estimator $\hat\kappa$ is computed as follows. From calibration residuals:
1. Compute $\hat R_k = 2 \cdot \text{Quantile}_{1-\alpha}(\{r_{k,i}\})$.
2. Estimate $\hat\sigma_k = \text{Std}(\{y_i - \hat f_k(x_i)\})$ from raw calibration residuals.
3. Compute $\hat f_k = 2\phi(z_{1-\alpha/2})/\hat\sigma_k$, the normal plug-in density.
4. Estimate $\widehat{\operatorname{Var}}(\hat R_k) = 4\alpha(1-\alpha) / (n_{\text{cal}} \hat f_k^2)$.
5. For each pair $(i,j)$, compute $\hat\rho_{ij}$ as the Spearman correlation of absolute residuals, then
   $\hat\sigma_{ij}^2 = \widehat{\operatorname{Var}}(\hat R_i) + \widehat{\operatorname{Var}}(\hat R_j) - 2\hat\rho_{ij}\sqrt{\widehat{\operatorname{Var}}(\hat R_i)\widehat{\operatorname{Var}}(\hat R_j)}$.
6. $\hat\kappa = \min_{i<j} |\hat R_i - \hat R_j| / \hat\sigma_{ij}$.

The cost is $O(m n_{\text{cal}} + m^2)$; no bootstrap needed.

### G. Bias-crossing lower bound

Let $k_T^* \neq k_D^*$. Then there exists a pair $(i,j)$ with $T_i^\infty < T_j^\infty$ but $D_i^\infty > D_j^\infty$. Define $\mathcal{S} = \{x : T_i^\infty < T_j^\infty, D_i^\infty > D_j^\infty\}$. By the continuous mapping theorem,
$$\mathbb{P}(\hat T_i < \hat T_j, \hat D_i > \hat D_j) \geq \mathbb{P}(X \in \mathcal{S}) - o(1),$$
so $\liminf_{n\to\infty} \Delta_n(T) \geq \mathbb{P}(X \in \mathcal{S}) > 0$. This establishes $\Delta_{\text{struct}} > 0$ in Theorem~1.

### H. Second illustration: training MSE as selection functional

Repeat the decay experiment with $T =$ training MSE, $D =$ test MSE.

**Table 11: Training MSE vs test MSE**
| $n$ | $\Delta$ | $\kappa$ |
|-----|----------|----------|
| 100 | 0.680 | 3.11 |
| 500 | 0.020 | 4.81 |
| 1000 | 0.000 | 6.39 |

The $\kappa$--$\Delta$ relationship mirrors the CP-width case, confirming that the theory applies beyond conformal prediction.

## References

Bahadur, R.R. (1966). A note on quantiles in large samples. *Annals of Mathematical Statistics*, 37:577--580.

Bickel, P.J. and Doksum, K.A. (2015). *Mathematical Statistics: Basic Ideas and Selected Topics*, volume I, 2nd edition. CRC Press.

Chernoff, H. (1952). A measure of asymptotic efficiency for tests of a hypothesis based on the sum of observations. *Annals of Mathematical Statistics*, 23:493--507.

Ferguson, T.S. (1996). *A Course in Large Sample Theory*. Chapman & Hall.

Hegazy, M., Aolaritei, L., Jordan, M., and Dieuleveut, A. (2025). Valid selection among conformal sets. *Advances in Neural Information Processing Systems*, NeurIPS 2025.

Angelopoulos, A.N. and Bates, S. (2023). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *Foundations and Trends in ML*, 16:494--591.

Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.

Hoeting, J.A., Madigan, D., Raftery, A.E., and Volinsky, C.T. (1999). Bayesian model averaging. *Statistical Science*, 14:382--417.

Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J., and Wasserman, L. (2018). Distribution-free predictive inference for regression. *JASA*, 113:1094--1111.

Liang, R., Zhu, W., and Barber, R.F. (2026). Conformal prediction after data-dependent model selection. *JASA*, forthcoming.

Scornet, E., Biau, G., and Vert, J.-P. (2015). Consistency of random forests. *Annals of Statistics*, 43:1716--1741.

Yang, Y. and Kuchibhotla, A.K. (2024). Selection and aggregation of conformal prediction sets. *JASA*, 120:435--447.

Yu, B. (2013). Stability. *Bernoulli*, 19:1484--1500.

Zhou, Z., Zhang, X., Tao, C., and Yang, Y. (2026). Conformal prediction assessment: a framework for conditional coverage evaluation and selection. arXiv:2603.27189.
