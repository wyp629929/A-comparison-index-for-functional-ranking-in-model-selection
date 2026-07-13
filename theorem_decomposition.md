# Decomposition Theorem: Formal Statement and Proof

## 1 Setup and Notation

Let $\mathcal{M} = \{1, \dots, m\}$ be a set of candidate estimators. For each estimator $k \in \mathcal{M}$ and training sample $\mathcal{D}_n = \{(X_i, Y_i)\}_{i=1}^n$ of size $n$:

1. **Training**: Fit $\hat{f}_{k, n}$ on $n_{\text{tr}}$ observations.
2. **Calibration**: Compute conformity scores $S_{k,i} = |Y_i - \hat{f}_{k, n}(X_i)|$ on $n_{\text{cal}}$ held-out points.
3. **CP width**: $R_{k,n} = 2 \cdot Q_{1-\alpha}(\{S_{k,i}\}_{i=1}^{n_{\text{cal}}})$.
4. **Risk**: $D_{k,n} = \frac{1}{n_{\text{te}}} \sum_{i=1}^{n_{\text{te}}} (Y_i - \hat{f}_{k,n}(X_i))^2$ on $n_{\text{te}}$ test points.
5. **Misalignment**: $\Delta_n = \mathbb{P}\left( \arg\min_{k} R_{k,n} \neq \arg\min_{k} D_{k,n} \right)$, where probability is over all random splits.

We assume $n_{\text{tr}}, n_{\text{cal}}, n_{\text{te}} \to \infty$ with $n_{\text{cal}}/n \to c_{\text{cal}} \in (0,1)$ and $n_{\text{te}}/n \to c_{\text{te}} \in (0,1)$.

### 1.1 Assumptions

**(A1) Estimator convergence.** For each $k \in \mathcal{M}$, the estimator $\hat{f}_{k,n}$ converges in probability to a fixed limit $f_k^\infty$ uniformly on compact sets, with $n^{1/2}(\hat{f}_{k,n} - f_k^\infty)$ converging weakly to a mean-zero Gaussian process.

**(A2) Residual regularity.** For each $k$, the residuals $\varepsilon_{k,i} = Y_i - \hat{f}_{k,n}(X_i)$ have a continuous density $f_{\varepsilon_k}$ in a neighborhood of the $(1-\alpha)$-quantile of $|\varepsilon_k|$, uniformly over $n$ large enough.

**(A3) Uniqueness of population argmin.** The population limits
$$R_k^\infty = 2 \cdot Q_{1-\alpha}(|Y - f_k^\infty(X)|), \qquad D_k^\infty = \mathbb{E}[(Y - f_k^\infty(X))^2]$$
satisfy that $\arg\min_k R_k^\infty$ and $\arg\min_k D_k^\infty$ are unique, with gaps
$$\delta_R = \min_{k \neq \ell} |R_k^\infty - R_\ell^\infty| > 0, \qquad \delta_D = \min_{k \neq \ell} |D_k^\infty - D_\ell^\infty| > 0.$$

**(A4) Joint CLT.** The vector $(\sqrt{n_{\text{cal}}}(R_{k,n} - R_k^\infty), \sqrt{n_{\text{te}}}(D_{k,n} - D_k^\infty))_{k=1}^m$ converges in distribution to a mean-zero $2m$-variate Gaussian distribution.

### 1.2 Competition Density

For any pair $(i,j)$, define the **pairwise competition density**:

$$\kappa_{ij} = \frac{|R_i^\infty - R_j^\infty|}{\sqrt{\text{AVar}(R_{i,n} - R_{j,n})}},$$

where $\text{AVar}(X) = \lim_{n \to \infty} n \cdot \text{Var}(X)$ is the asymptotic variance. This measures how many asymptotic standard deviations apart the two estimators are in CP width.

The **minimum competition density** is:

$$\kappa(\mathcal{G}) = \min_{i \neq j} \kappa_{ij}.$$

If $\kappa(\mathcal{G})$ is large, the closest pair is well-separated in CP width → low competition → low misalignment. If $\kappa(\mathcal{G})$ is small, the closest pair is hard to distinguish → high competition → high misalignment.

---

## 2 Main Result

**Theorem 1 (Misalignment Decomposition).** Under Assumptions (A1)-(A4):

$$\Delta_n = \Delta_\infty + \frac{c(\mathcal{G})}{\sqrt{n}} + o\left(\frac{1}{\sqrt{n}}\right),$$

where:

1. **Irreducible component**: $\Delta_\infty = \lim_{n \to \infty} \Delta_n$ is non-zero iff the population limits satisfy $\arg\min R_k^\infty \neq \arg\min D_k^\infty$. This occurs only under bias crossing (Proposition 2 of [Paper 1]) or structural L1-L2 mismatch at the population level.

2. **Competition component**: When $\Delta_\infty = 0$ (no structural disagreement at the population level),

   $$\Delta_n = \frac{c(\mathcal{G})}{\sqrt{n}} + o\left(\frac{1}{\sqrt{n}}\right),$$

   where $c(\mathcal{G}) > 0$ depends only on the comparison geometry $\mathcal{G} = \{R_k^\infty, D_k^\infty, \Sigma\}$ (the vector of population limits and the $2m \times 2m$ covariance matrix of the joint limiting distribution).

3. **κ upper bound**: $c(\mathcal{G})$ satisfies:
   $$c(\mathcal{G}) \leq C \cdot \phi(\kappa(\mathcal{G})),$$
   where $C$ depends only on $m$ and $\alpha$, and $\phi(z) = (2\pi)^{-1/2} e^{-z^2/2}$ is the standard normal PDF. Equivalently, for any $\epsilon > 0$,
   $$\Delta_n \leq C \cdot \frac{\phi(\kappa(\mathcal{G}))}{\sqrt{n}} + \epsilon$$
   for all $n$ sufficiently large.

4. **κ monotonicity**: As $\kappa(\mathcal{G}) \to \infty$, $\Delta_n \to 0$ at rate $O(e^{-\kappa^2/2})$. As $\kappa(\mathcal{G}) \to 0$, $\Delta_n$ approaches the uniform random baseline $1 - 1/m$.

*Proof.* See below.

---

**Corollary 1 (Dependence Modulation).** Consider a family of DGPs $\{\mathcal{F}_\rho\}_{\rho \in [0,1]}$ with block dependence parameter $\rho$ and fixed bias structure. For an estimator set $\mathcal{M}$ containing both linear and tree-based methods:

$$\frac{\partial \kappa(\mathcal{F}_\rho)}{\partial \rho} < 0 \quad \text{for sufficiently large } \rho,$$

implying $\Delta_n(\mathcal{F}_\rho)$ increases with $\rho$ through the competition channel. For an estimator set $\mathcal{M}$ containing only linear methods:

$$\frac{\partial \kappa(\mathcal{F}_\rho)}{\partial \rho} \approx 0,$$

so $\Delta_n(\mathcal{F}_\rho)$ is insensitive to $\rho$.

---

## 3 Proof Sketch

### 3.1 Bahadur Representation

Under (A2), the quantile $Q_{1-\alpha}(\{|S_{k,i}|\})$ admits a Bahadur representation:

$$R_{k,n} = R_k^\infty + \frac{1}{n_{\text{cal}}} \sum_{i=1}^{n_{\text{cal}}} \frac{\alpha - \mathbf{1}\{|S_{k,i}| \leq R_k^\infty/2\}}{f_{|\varepsilon_k|}(R_k^\infty/2)} + o_p(n_{\text{cal}}^{-1/2}).$$

The leading term has mean zero and variance $\sigma_{R,k}^2 / n_{\text{cal}}$, where $\sigma_{R,k}^2 = \alpha(1-\alpha) / (f_{|\varepsilon_k|}(R_k^\infty/2))^2$.

### 3.2 Joint CLT

Stacking the $m$ estimators: by (A1), for any pair $(i,j)$:

$$\sqrt{n_{\text{cal}}} \begin{pmatrix} R_{i,n} - R_i^\infty \\ R_{j,n} - R_j^\infty \end{pmatrix} \xrightarrow{d} N\left(0, \begin{pmatrix} \sigma_{R,i}^2 & \sigma_{R,ij} \\ \sigma_{R,ij} & \sigma_{R,j}^2 \end{pmatrix}\right).$$

The risk estimates satisfy an analogous CLT with variance $\sigma_{D,k}^2/n_{\text{te}}$. Under (A4), the full $2m$-dimensional vector is jointly normal.

### 3.3 Argmin Disagreement

Let $k_R^* = \arg\min R_k^\infty$ and $k_D^* = \arg\min D_k^\infty$.

**Case $\Delta_\infty > 0$**: When $k_R^* \neq k_D^*$, the misalignment persists at all sample sizes because the population functionals disagree. This is the regime of bias crossing (Proposition 2 of Paper 1) or structural L1-L2 mismatch.

**Case $\Delta_\infty = 0$**: When $k_R^* = k_D^* = k^*$, misalignment can only arise from finite-sample noise. For any competitor $j \neq k^*$:

$$\mathbb{P}(R_{j,n} < R_{k^*,n}) = \Phi\left(-\sqrt{n_{\text{cal}}} \cdot \frac{R_j^\infty - R_{k^*}^\infty}{\sigma_{R,jk^*}}\right) + o(1),$$

where $\sigma_{R,jk^*}^2 = \text{AVar}(R_{j,n} - R_{k^*,n})$. The same holds for $D_{k,n}$. The total misalignment is bounded by:

$$\Delta_n \leq \sum_{j \neq k^*} \left[\mathbb{P}(R_{j,n} < R_{k^*,n}) + \mathbb{P}(D_{j,n} < D_{k^*,n})\right].$$

### 3.4 κ Bound

The dominant term is the pair with the smallest $\kappa_{jk^*}$. Define $\kappa = \min_{j \neq k^*} \kappa_{jk^*}$. For the worst-case pair:

$$\mathbb{P}(R_{j,n} < R_{k^*,n}) = \Phi(-\sqrt{n} \cdot \kappa \cdot \sqrt{c_{\text{cal}}}) + o(1).$$

Using the standard normal tail bound $\Phi(-x) \leq \phi(x)/x$ for $x > 0$:

$$\Delta_n \leq \frac{2(m-1)}{\sqrt{n} \cdot \kappa \cdot \sqrt{c_{\text{cal}}}} \phi(\sqrt{n} \cdot \kappa \cdot \sqrt{c_{\text{cal}}}) + o(n^{-1/2}).$$

Expanding: $\phi(\sqrt{n} \cdot \kappa) = (2\pi)^{-1/2} e^{-n \kappa^2 / 2}$. For large $n$, this is dominated by the exponential term. Thus:

$$c(\mathcal{G}) \leq C \cdot \phi(\kappa(\mathcal{G})),$$

where $C$ absorbs constants $m$, $c_{\text{cal}}$, and the $-1/2$ polynomial factor.

### 3.5 Discussion of the Bound

The bound has the right qualitative behavior:
- $\kappa \to \infty$ (well-separated estimators): $\phi(\kappa) \approx 0$ → $\Delta_n \approx 0$
- $\kappa \to 0$ (inseparable estimators): $\phi(0) = 1/\sqrt{2\pi}$ → bound approaches a positive constant consistent with the random baseline $1 - 1/m$

The bound is not tight for small $n$ due to the union bound and remainder terms, but it correctly captures the **mechanism**: misalignment from the competition channel decays as $e^{-n \kappa^2/2}$, while misalignment from the bias-crossing channel persists as $n \to \infty$.

---

## 4 Proof of Corollary 1 (Dependence Modulation)

Under block equicorrelation with parameter $\rho$, the covariance of $X$ is:

$$\Sigma_{ij} = \begin{cases} 1 & i = j \\ \rho & i,j \text{ in same block} \\ 0 & \text{otherwise} \end{cases}$$

**For linear estimators** (OLS, Ridge, Lasso): The variance of the prediction at a test point $x$ satisfies:

$$\text{Var}(\hat{f}_{\text{OLS}}(x)) = \sigma^2 x^{\top} (X_{\text{tr}}^{\top} X_{\text{tr}})^{-1} x.$$

Under block equicorrelation, $\mathbb{E}[X_{\text{tr}}^{\top} X_{\text{tr}}/n_{\text{tr}}] = \Sigma$. By standard results (Wishart concentration), the typical prediction variance inflates by a factor $\approx 1 + (m-1)\rho$ for coordinates within the same block, where $m = p/B$ is the block size.

For OLS (full model with $p$ features), all coordinates contribute to the variance. For the restricted model ($s$ features), only $s$ coordinates contribute. As $\rho$ increases, the variance of the full model inflates more than the restricted model:

$$\frac{\partial}{\partial \rho} \text{Var}(\hat{f}_{\text{OLS,full}}) > \frac{\partial}{\partial \rho} \text{Var}(\hat{f}_{\text{OLS,restr}}) > 0.$$

This changes $D_k^\infty$ and $R_k^\infty$ differentially, modulating $\kappa$ and thus $\Delta_n$.

**For tree-based estimators**: Feature subsampling breaks the block structure, reducing the effective inflation factor. The variance of $\hat{f}_{\text{RF}}$ is approximately:

$$\text{Var}(\hat{f}_{\text{RF}}(x)) \approx \frac{\sigma^2}{m_{\text{try}}} \sum_{j \in \mathcal{J}} \omega_j^2,$$

where $\mathcal{J}$ is the set of features considered at a split. Since $|\mathcal{J}| \ll p$ and features are sampled independently of block structure, the variance inflation from block correlation is approximately $1 + (m-1)\rho/B_{\text{eff}}$ where $B_{\text{eff}} \gg 1$ is the effective number of blocks sampled per split.

**Net effect**: As $\rho$ increases, the relative gap between OLS and RF CP widths narrows (RF becomes more competitive), reducing $\kappa$. This is the competition channel. A detailed calculation of the exact inflation factor is left to Appendix. The qualitative prediction is consistent with the empirical findings in Section 5.
