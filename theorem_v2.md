# Theorem: Misalignment Decomposition (v2)

## Setup

Let $\mathcal{M} = \{1,\dots,m\}$ with estimators $\hat f_{k,n}$. Split $n$ into $n_{\text{tr}}/n_{\text{cal}}/n_{\text{te}}$ with fixed proportions $\pi_{\text{tr}},\pi_{\text{cal}},\pi_{\text{te}} > 0$.

**CP width:** $\hat R_{k,n} = 2 \cdot \hat Q_{1-\alpha}(\{|Y_i - \hat f_{k,n}(X_i)|\}_{i=1}^{n_{\text{cal}}})$.

**Risk:** $\hat D_{k,n} = n_{\text{te}}^{-1} \sum_{i=1}^{n_{\text{te}}} (Y_i - \hat f_{k,n}(X_i))^2$.

**Misalignment:** $\Delta_n = \mathbb{P}(\arg\min_k \hat R_{k,n} \neq \arg\min_k \hat D_{k,n})$.

### Assumptions

**(A1) Estimator convergence.** $n^{1/2}(\hat f_{k,n} - f_k^\infty) \rightsquigarrow \mathbb{G}_k$, a mean-zero Gaussian process, uniformly over compact $\mathcal{X}$.

**(A2) Residual density.** Each residual $\varepsilon_k = Y - f_k^\infty(X)$ has continuous density $f_{\varepsilon_k}$ in a neighborhood of $Q_{1-\alpha}(|\varepsilon_k|)$, bounded away from 0.

**(A3) Unique argmin.** The population functionals $R_k^\infty = 2 \cdot Q_{1-\alpha}(|\varepsilon_k|)$ and $D_k^\infty = \mathbb{E}[(Y - f_k^\infty(X))^2]$ have unique argmins $k_R^*, k_D^* \in \mathcal{M}$.

**(A4) Joint CLT.** $n^{1/2}(\hat R_{k,n} - R_k^\infty, \hat D_{k,n} - D_k^\infty)_{k=1}^m \xrightarrow{d} N(0, \Sigma)$, with $\Sigma$ positive definite.

### Competition Density

**Definition (Population $\kappa$).** For any pair $(i,j)$,

$$\kappa_{ij} = \frac{|R_i^\infty - R_j^\infty|}{\sqrt{n^{-1} \cdot \text{AVar}(\hat R_{i,n} - \hat R_{j,n})}} = \frac{|R_i^\infty - R_j^\infty|}{\sqrt{\Sigma_{ii} + \Sigma_{jj} - 2\Sigma_{ij}}},$$

where $\Sigma_{ij} = \lim_{n\to\infty} n \cdot \text{Cov}(\hat R_{i,n}, \hat R_{j,n})$. The *minimum competition density* is

$$\kappa(\mathcal{G}) = \min_{i \neq j} \kappa_{ij}.$$

---

## Main Result

**Theorem 1 (Misalignment Decomposition).** Under (A1)-(A4):

1. **Bias-crossing regime**: If $k_R^* \neq k_D^*$ (the population functionals disagree on which estimator is best), then
   $$\Delta_n \to \Delta_{\text{bias}} > 0 \quad \text{as } n \to \infty,$$
   where $\Delta_{\text{bias}} \geq \min\{\mathbb{P}(X \in A), \mathbb{P}(X \in B)\} > 0$ under the bias-crossing condition of Proposition 2 (Paper 1). The misalignment persists at all sample sizes.

2. **Competition regime**: If $k_R^* = k_D^* = k^*$ (no population-level disagreement), then
   $$\Delta_n \to 0 \quad \text{as } n \to \infty,$$
   with rate bounded by:
   $$\Delta_n \leq C \cdot \exp\!\left(-\frac{n \cdot \kappa^2 \cdot \pi_{\text{cal}}}{2}\right) + o(n^{-1/2}),$$
   where $C$ depends only on $m$ and the CP-Risk correlation $\rho_{RD}$, and $\kappa = \min_{j \neq k^*} \kappa_{jk^*}$ is the minimum competition density.

   **Corollary**: As $\kappa \to 0$, the bound approaches $C$, and $\Delta_n \to 1 - 1/m$ (the uniform random baseline for $m$ estimators). As $\kappa \to \infty$, $\Delta_n \to 0$ exponentially fast.

### Interpretation

- The **competition density** $\kappa$ controls the exponential decay rate of $\Delta_n$ in the competition regime.
- The bound is not tight for finite $n$ because the joint CP-Risk correlation also matters, but the qualitative prediction ($\kappa$ predicts $\Delta$, smaller $\kappa$ → larger $\Delta$) is robust.
- Empirical validation across all experimental conditions (Section 4.5, Figure 2) confirms the monotone $\kappa$–$\Delta$ relationship.

### General $m$ case

For $m > 2$, the same structure holds with a union bound over the $m-1$ competitors:

$$\Delta_n \leq \sum_{j \neq k^*} \Phi\!\left(-\sqrt{n} \cdot \kappa_{jk^*} \cdot \sqrt{\pi_{\text{cal}}}\right) + o(n^{-1/2}).$$

The dominant term is the closest competitor (smallest $\kappa_{jk^*}$), so the bound is driven by $\kappa = \min_{j \neq k^*} \kappa_{jk^*}$.

---

## Plug-in Estimator for $\kappa$

**Definition (Sample $\hat\kappa$).** Given a single data split with calibration set of size $n_{\text{cal}}$:

$$\hat\kappa = \min_{i\neq j} \frac{|\hat R_i - \hat R_j|}{\hat\sigma_{ij}},$$

where:

$$\hat\sigma_{ij}^2 = \hat\sigma_{R_i}^2 + \hat\sigma_{R_j}^2 - 2\hat\sigma_{R_iR_j},$$

and using the Bahadur influence function:

$$\hat\sigma_{R_k}^2 = \frac{\hat\alpha(1-\hat\alpha)}{n_{\text{cal}} \cdot \hat f_{|\varepsilon_k|}^2(\hat R_k/2)}, \quad \hat\alpha = 0.9.$$

The density $\hat f_{|\varepsilon_k|}$ is estimated by KDE (Gaussian kernel, bandwidth Silverman's rule) from the calibration residuals $\{|Y_i - \hat f_{k,n}(X_i)|\}_{i=1}^{n_{\text{cal}}}$.

The covariance $\hat\sigma_{R_iR_j}$ is:

$$\hat\sigma_{R_iR_j} = \frac{\hat\alpha(1-\hat\alpha)}{n_{\text{cal}} \cdot \hat f_{|\varepsilon_i|}(\hat R_i/2) \cdot \hat f_{|\varepsilon_j|}(\hat R_j/2)} \cdot \hat\rho_{ij},$$

where $\hat\rho_{ij}$ is the rank correlation of the two sets of calibration residuals.

**Computational cost:** Single pass over calibration data, $O(m n_{\text{cal}} + m^2)$. No bootstrap.
