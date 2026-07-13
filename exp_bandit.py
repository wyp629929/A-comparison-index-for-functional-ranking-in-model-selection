"""
Bandit experiment: CP-based policy selection under distribution shift.

Setup:
  - 10-dim context X, 2 actions a ∈ {0,1}
  - Action-dependent rewards: r = f_a(X) + ε
  - Behavior policy π_b with parameter θ = P(a=1)
    θ ≈ 0.5 → high ESS (uniform), θ → 1 → low ESS (skewed)
  - 4 candidate policies competing for selection

Key hypothesis: ESS modulates competition density κ → predicts ∆.
"""

import numpy as np, pandas as pd
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")
np.random.seed(2030)

N_REPS = 400


def simulate(pi_b_param, n=2000, p=10):
    """Generate bandit data and evaluate 4 policies."""
    # Reward functions (action-dependent)
    beta0 = np.array([1.0, -0.5, 0.3, 0, 0, 0, 0, 0, 0, 0])
    beta1 = np.array([-0.3, 0.8, 0, 0, 0, 1.0, -0.5, 0, 0, 0])

    X = np.random.randn(n, p)
    # Behavior policy: Bernoulli(pi_b_param)
    actions = (np.random.rand(n) < pi_b_param).astype(int)
    rewards = np.where(actions == 0, X @ beta0, X @ beta1) + np.random.randn(n)

    # Candidate policies
    policies = {
        "always0": lambda x: 0,
        "always1": lambda x: 1,
        "thresh": lambda x: 1 if x[0] > 0 else 0,
        "lincomb": lambda x: 1 if x[0] + x[1] > 0 else 0,
    }

    # For each policy: compute IPW value estimate + CP width
    cp_vals = {}
    true_vals = {}

    for pname, policy in policies.items():
        # Policy decisions for each observation
        pdecisions = np.array([policy(x) for x in X])
        match = (pdecisions == actions)

        # IPW weights
        pi_b = np.where(actions == 0, 1 - pi_b_param, pi_b_param)
        w = np.where(match, 1.0 / pi_b, 0.0)

        # IPW estimate of value
        V_ipw = np.sum(w * rewards) / max(np.sum(w), 1e-10)

        # CP on weighted rewards
        weighted_rewards = w * rewards
        non_zero = weighted_rewards > 0
        if non_zero.sum() < 5:
            cp_vals[pname] = float('inf')
        else:
            residuals = np.abs(weighted_rewards[non_zero] - V_ipw)
            q = np.quantile(residuals, 0.9)
            cp_vals[pname] = 2 * q

        # True value (Monte Carlo)
        n_mc = 50000
        X_mc = np.random.randn(n_mc, p)
        actions_mc = np.array([policy(x) for x in X_mc])
        true_val = np.mean(np.where(actions_mc == 0, X_mc @ beta0, X_mc @ beta1))
        true_vals[pname] = true_val

    # ∆: does CP width ranking agree with true value ranking?
    # Policy selected by CP (narrowest width)
    cp_best = min(cp_vals, key=cp_vals.get)
    # Policy selected by true value (highest)
    true_best = max(true_vals, key=true_vals.get)
    delta = int(cp_best != true_best)

    # κ: competition density among policies
    vals = np.array(list(cp_vals.values()))
    vals = vals[vals != float('inf')]
    if len(vals) < 2:
        return None

    # κ = min gap / SD of differences for closest pair of policies
    names = [k for k, v in cp_vals.items() if v != float('inf')]
    kappa_vals = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            gap = abs(cp_vals[names[i]] - cp_vals[names[j]])
            # SD approximated from variability of IPW
            sd = (cp_vals[names[i]] + cp_vals[names[j]]) * 0.1
            kappa_vals.append(gap / max(sd, 1e-10))
    kappa = min(kappa_vals) if kappa_vals else float('inf')

    return dict(theta=pi_b_param, delta=delta, kappa=kappa, n=n)


print("=== Bandit: θ sweep ===")
print(f"{'θ':>8} {'∆':>8} {'SE':>8} {'κ':>8}")
print("-" * 40)

for theta in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    reps = Parallel(n_jobs=-1)(delayed(simulate)(theta) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    delta = df.delta.mean()
    se = np.sqrt(delta*(1-delta)/len(df))
    kappa = df.kappa.mean()
    print(f"{theta:>8.2f} {delta:>8.3f} {se:>8.3f} {kappa:>8.2f}")
