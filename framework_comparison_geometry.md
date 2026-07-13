# A Comparison-Geometric Framework for Marginal Summary Collapse

## 1 Framework

### 1.1 Decision-Relevant Comparison Geometry

Let $\mathcal{M} = \{1, \dots, m\}$ be a set of candidate entities (estimators, inference tasks, or models). For each entity $i \in \mathcal{M}$, define its **performance surface** as a function $g_i: \mathcal{X} \to \mathbb{R}$ that captures the decision-relevant characteristic at each point $x \in \mathcal{X}$ (the covariate space).

**Definition 1 (Comparison Geometry).** The *comparison geometry* of $\mathcal{M}$ is the set of performance surfaces $\mathcal{G} = \{g_1, \dots, g_m\}$, equipped with the pointwise comparison function

$$d(i, j \mid x) = \phi(g_i(x), g_j(x))$$

where $\phi: \mathbb{R} \times \mathbb{R} \to \mathbb{R}$ is a task-specific contrast function (e.g., $\phi(a, b) = a - b$ for ranking, $\phi(a, b) = |a - b|$ for discrimination).

The decision-relevant ordering is: $i \prec_x j$ if $d(i, j \mid x) < 0$, meaning entity $i$ is conditionally preferred to entity $j$ at $x$.

### 1.2 Summary Projection

A **summary functional** is a map $\Pi: \mathcal{G} \to \mathbb{R}^m$ that compresses each performance surface $g_i$ to a scalar statistic $R_i = \Pi(g_i)$ using only marginal information.

**Definition 2 (Marginal Summary).** $\Pi$ is a *marginal* summary if $R_i = \psi(\{r_{ik}\}_{k=1}^n)$ where $\psi$ is permutation-invariant and $r_{ik}$ depends on the data only through the $k$-th observation's contribution to estimating $g_i$. Examples: CP interval width (quantile of absolute residuals), cross-validated MSE (average of squared residuals), log(p)/n threshold (extremal noise scale).

The induced ordering is: $i \prec_\Pi j$ if $R_i < R_j$.

### 1.3 Collapse

**Definition 3 (Decision Sufficiency and Collapse).** $\Pi$ is *decision-sufficient* for $\mathcal{G}$ if for all $i, j \in \mathcal{M}$:

$$d(i, j \mid x) = 0 \ \forall x \in \mathcal{X} \ \Longrightarrow \ \Pi(g_i) = \Pi(g_j).$$

Equivalently, if two entities are indistinguishable in the full comparison geometry, their summaries should be equal. $\Pi$ is *decision-insufficient* if there exist $i, j$ and $x$ such that $d(i, j \mid x) \neq 0$ but $\Pi(g_i) = \Pi(g_j)$.

**Definition 4 (Collapse Regime).** A *collapse regime* is a class of comparison geometries $\mathcal{F}$ on which $\Pi$ is decision-insufficient. The *collapse magnitude* for a specific comparison geometry $\mathcal{G} \in \mathcal{F}$ is

$$\Delta(\mathcal{G}) = \mathbb{P}\left( \arg\min_i \Pi(g_i) \neq \arg\min_i \mathbb{E}_X[d(i, \cdot \mid X)] \right)$$

where the probability is over the finite sample used to compute $\Pi$, and $\mathbb{E}_X$ is the decision-maker's aggregation of pointwise comparisons.

### 1.4 The Nullspace Condition

The fundamental mechanism of collapse is a nullspace condition:

**Proposition 0 (Nullspace Collapse).** Let $N(\Pi) = \{(i,j): \Pi(g_i) = \Pi(g_j)\}$ be the equivalence relation induced by $\Pi$. Let $N(\mathcal{G}) = \{(i,j): d(i,j|x) = 0 \ \forall x\}$ be the decision-relevant equivalence. If $N(\Pi) \supsetneq N(\mathcal{G})$, then $\Pi$ is decision-insufficient and $\Delta(\mathcal{G}) > 0$ under any sampling distribution with positive probability on the discriminating region $\{x: d(i,j|x) \neq 0\}$.

*Proof.* Take $(i,j) \in N(\Pi) \setminus N(\mathcal{G})$. Since $\Pi(g_i) = \Pi(g_j)$, the summary does not distinguish $i$ from $j$, but the decision geometry does. Any decision rule based on $\Pi$ that must select between $i$ and $j$ incurs misalignment at least $\mathbb{P}(X \in \{x: d(i,j|x) \neq 0\}) / 2$.

---

## 2 Known Collapse Regimes

### 2.1 Regime B: Bias Crossing (Paper 1)

**Setup.** $\mathcal{M} = \{M_1, \dots, M_m\}$ are estimators. Performance surface is

$$g_i(x) = |\varepsilon + \text{bias}_i(x)|,$$

where $\text{bias}_i(x) = f_0(x) - \mathbb{E}[\hat{f}_i(x)]$. The summary projection is the CP interval width

$$\Pi(g_i) = 2 \cdot Q_{1-\alpha}(g_i(X)) \quad \text{(marginal absolute-residual quantile)}.$$

The decision contrast is the conditional risk

$$d(i,j|x) = \text{bias}_i^2(x) - \text{bias}_j^2(x) \quad \text{(up to common noise variance)}.$$

**Collapse condition (Proposition 2 of Paper 1).** If $\exists A, B \subset \mathcal{X}$ with positive measure such that $|\text{bias}_i(x)| < |\text{bias}_j(x)|$ on $A$ and the reverse on $B$, then $\Pi(g_i) = \Pi(g_j)$ is possible while $d(i,j|x)$ changes sign across $x$. This is a structural nullspace inclusion: $N(\Pi)$ identifies estimators with equal marginal absolute-residual distributions, but $N(\mathcal{G})$ requires pointwise risk equality. The collapse is irreducible: $\Delta \geq \min\{\mathbb{P}(X \in A), \mathbb{P}(X \in B)\} > 0$.

### 2.2 Regime D: Dependence Divergence (Paper 2)

**Setup.** $\mathcal{M} = \{\text{detection, selection, BH, spurious}\}$ are inference tasks. Performance surface is

$$g_i(B) = \mu_{50}^{(i)}(B) \quad \text{(signal strength at power 0.5 for task } i \text{ as a function of block count } B).$$

The summary projection is the log(p)/n scaling threshold

$$\Pi(g_i) = \sqrt{\log p / n} \quad \text{(scalar, independent of } B \text{ under independence)}.$$

The decision contrast is the task-specific phase-transition ordering

$$d(i,j|B) = \mu_{50}^{(i)}(B) - \mu_{50}^{(j)}(B).$$

**Collapse condition.** Under independence, $\Pi(g_i) = \Pi(g_j)$ for all $i,j$ (all tasks share the $\sqrt{\log p/n}$ threshold). However, under block dependence, $d(i,j|B)$ changes sign with $B$: detection's $\mu_{50}$ decreases with $B$, selection's increases, BH and spurious are flat. The marginal log(p)/n summary collapses four task-specific comparison geometries into a single scalar, losing the dependence structure. $N(\Pi)$ partitions all tasks as equivalent, while $N(\mathcal{G})$ distinguishes them.

### 2.3 Regime C: Competition Modulation (Week 0 Finding)

**Setup.** $\mathcal{M} = \{\text{OLS, Ridge, Lasso, RF}\}$ in a linear regression with block-dependent predictors. Performance surface is

$$g_i(x) = \mathbb{E}[(\hat{f}_i(x) - f_0(x))^2 \mid x] \quad \text{(conditional risk)}.$$

Under independent predictors (small $\rho$), the comparison geometry has large spread: OLS dominates, RF is far behind. Under dependent predictors (large $\rho$), RF's effective variance decreases (feature redundancy), compressing the comparison geometry. The same summary $\Pi$ (CP width) projects a now-denser geometry, increasing the probability of misranking.

**Collapse mechanism (new).** Unlike Regime B (structural, n→∞ persistent), this collapse is driven by **competition density**:

$$\kappa(\mathcal{G}) = \frac{\max_i \mathbb{E}[g_i(X)] - \min_i \mathbb{E}[g_i(X)]}{\sqrt{\max_i \text{Var}(\Pi(g_i))}}.$$

As $\kappa(\mathcal{G}) \to 0$, the comparison geometry is compressed relative to the noise level of the summary, and $\Delta$ increases. Dependence modulates $\kappa$ by differentially affecting estimator variances.

**Why this is distinct.** Regime B's collapse arises from bias crossing (the ordering of bias functions changes sign over $\mathcal{X}$). Regime C's collapse arises from insufficient spread in the comparison geometry (estimators are too close to reliably distinguish). The two mechanisms can coexist and interact.

---

## 3 Main Result: Decomposition Theorem

**Theorem 1 (Misalignment Decomposition).** Let $\mathcal{M}$ be a set of estimators and $\Pi$ be a marginal residual functional (Definition 2). Assume the DGP satisfies regularity conditions (finite variance of conformity scores, continuous distribution). Then:

$$\Delta_n(\mathcal{M}) = \Delta_\infty(\mathcal{M}) + \frac{c(\mathcal{M})}{\sqrt{n}} + o(n^{-1/2}),$$

where:

1. $\Delta_\infty(\mathcal{M})$ is the *irreducible misalignment*: the limit of $\Delta_n$ as $n \to \infty$, determined entirely by the comparison geometry $\mathcal{G}$ and not by estimation noise.

2. $\Delta_\infty(\mathcal{M})$ further decomposes as
   $$\Delta_\infty(\mathcal{M}) = \Delta_B(\mathcal{M}) + \Delta_C(\mathcal{M}),$$
   where $\Delta_B(\mathcal{M})$ is the bias-crossing component (exists if and only if the bias functions cross per Proposition 2 of Paper 1), and $\Delta_C(\mathcal{M})$ is the competition component (exists if and only if $\kappa(\mathcal{G}) < \kappa_0$ for a threshold $\kappa_0$ depending on $m$).

3. The bias-crossing component satisfies $\Delta_B(\mathcal{M}) \geq \min_A \mathbb{P}(X \in A)$ (structural lower bound, Paper 1), while $\Delta_C(\mathcal{M}) \to 0$ as the spread of the comparison geometry increases relative to the summary noise.

**Corollary 1 (Dependence Modulation).** For a family of DGPs $\{\mathcal{F}_\rho\}_{\rho \in [0,1]}$ with block dependence parameter $\rho$ and fixed bias structure:
- $\Delta_B(\mathcal{F}_\rho)$ is constant in $\rho$ (dependence does not create or destroy bias crossing in this class)
- $\Delta_C(\mathcal{F}_\rho)$ varies with $\rho$ through the compression of the comparison geometry
- Therefore, changes in $\Delta(\mathcal{F}_\rho)$ across $\rho$ isolate the competition channel

*Proof sketch of Theorem 1.* For (1), write $\Delta_n = \mathbb{P}(\arg\min \Pi_n \neq \arg\min D_n)$. As $n \to \infty$, $\Pi_n \to \Pi_\infty$ and $D_n \to D_\infty$ in probability (uniformly over $\mathcal{M}$ by Vapnik-Chervonenkis theory for the estimators considered). By the continuous mapping theorem, if the limit argmin is unique almost surely, $\Delta_n \to \Delta_\infty$. The $O(n^{-1/2})$ rate follows from the delta method applied to the quantile and mean functionals.

For (2), condition on the event that bias crossing occurs (i.e., $\exists i,j,x$ such that $|\text{bias}_i(x)| < |\text{bias}_j(x)|$ but $\Pi(g_i) = \Pi(g_j)$). The probability of this event is $\Delta_B$, independent of $n$ by Proposition 2. The remaining misalignment arises from estimators whose limiting summaries are separated by less than the noise scale; this probability scales with $1/\kappa(\mathcal{G})^2$.

For (3), Corollary 1 follows from the fact that block dependence in the predictors does not change the bias functions (which depend on the mean function $f_0$, not on the predictor covariance), but it changes estimator variances and thus the comparison geometry $\mathcal{G}$.

---

## 4 Generative Predictions

The framework predicts that ANY perturbation of the comparison geometry will modulate $\Delta$ through the competition channel, even without bias crossing or dependence. Testable predictions:

1. **Increasing $p$ (feature count)** at fixed $n$: Increases RF variance more than OLS variance → widens spread → decreases $\Delta_C$.
2. **Increasing $n$ (sample size)** at fixed $\mathcal{G}$: Reduces estimation noise → increases effective $\kappa$ → decreases $\Delta_C$.
3. **Identically performing estimators**: As $\kappa \to 0$, $\Delta_C \to 1-1/m$ (random selection baseline).
4. **Dependence sign asymmetry**: Negative within-block correlation increases OLS variance but decreases RF variance (different dependence sensitivity) → widens spread → decreases $\Delta$ (opposite to positive $\rho$).

---
