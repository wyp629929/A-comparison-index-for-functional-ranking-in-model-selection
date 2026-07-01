"""
Bandit experiment v2: Direct Method (reward model) + CP.

We fit a reward model on logged data from behavior policy π_b.
For each candidate policy π_k, we compute the predicted value V̂_k = avg( r̂(x, π_k(x)) ).
CP gives interval width around each V̂_k.

Distribution shift (π_b ≠ π_k) creates heteroscedastic prediction variance.
The "behavior gap" g = D_KL(π_b || uniform) modulates how much data we have
for each action → effective sample size → CP width → competition.

4 policies, vary behavior policy skewness.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")
np.random.seed(2032)

N_REPS = 300


def simulate(theta_prob1, n=5000, p=10):
    """θ_prob1: P(action=1) under behavior policy."""
    np.random.seed()
    beta0 = np.array([1.0, -0.5, 0.3, 0, 0, 0, 0, 0, 0, 0])
    beta1 = np.array([-0.3, 0.8, 0, 0, 0, 1.0, -0.5, 0, 0, 0])

    X = np.random.randn(n, p)
    actions = (np.random.rand(n) < theta_prob1).astype(int)
    rewards = np.where(actions == 0, X @ beta0, X @ beta1) + np.random.randn(n)

    # Fit reward model (shared across policies)
    # Simple approach: separate OLS for each action
    idx0 = actions == 0; idx1 = actions == 1
    if idx0.sum() < 5 or idx1.sum() < 5:
        return None

    model0 = Ridge(alpha=1.0).fit(X[idx0], rewards[idx0])
    model1 = Ridge(alpha=1.0).fit(X[idx1], rewards[idx1])

    # Candidate policies
    policies = {
        "always0": lambda x: 0,
        "always1": lambda x: 1,
        "thresh": lambda x: 1 if x[0] > 0 else 0,
        "lincomb": lambda x: 1 if x[0] + x[1] > 0 else 0,
    }

    # Evaluate each policy
    n_eval = 10000
    X_eval = np.random.randn(n_eval, p)

    cp_vals, true_vals, ess_vals = {}, {}, {}
    for pname, policy in policies.items():
        # Policy actions on evaluation set
        pi_actions = np.array([policy(x) for x in X_eval])

        # Predicted rewards (from reward model)
        pred_rewards = np.where(pi_actions == 0,
                                model0.predict(X_eval),
                                model1.predict(X_eval))

        # True rewards (known DGP — for oracle)
        true_rewards = np.where(pi_actions == 0,
                                X_eval @ beta0,
                                X_eval @ beta1)
        true_val = np.mean(true_rewards)
        true_vals[pname] = true_val

        # CP: resample train-test split of preds vs true to get interval
        # Use a held-out calibration set from the original data
        # For efficiency, calibrate on the evaluation set itself with a train/cal split
        # (the "training" is the reward model, already fit; calibration is on held-out predictions)

        _, X_cal, _, y_cal = train_test_split(X_eval, pred_rewards, test_size=0.3, random_state=None)
        # Actually use the original training data for calibration to mimic real bandit
        # Split the training data: 70% for reward model, 30% for calibration
        # But the reward model is already fit on ALL data — let me use a separate calibration fold

        # Simpler approach: use cross-fitting
        # 1. Fit model on 70% of data
        # 2. CP on 30% of data
        X_tr, X_cal, y_tr, y_cal, a_tr, a_cal = train_test_split(X, rewards, actions, test_size=0.3, random_state=None)

        idx0_tr = a_tr == 0; idx1_tr = a_tr == 1
        m0 = LinearRegression().fit(X_tr[idx0_tr], y_tr[idx0_tr])
        m1 = LinearRegression().fit(X_tr[idx1_tr], y_tr[idx1_tr])

        # Calibration: predict on held-out
        cal_actions = np.array([policy(x) for x in X_cal])
        cal_preds = np.where(cal_actions == 0, m0.predict(X_cal), m1.predict(X_cal))
        cal_resid = np.abs(y_cal - cal_preds)
        if len(cal_resid) < 5:
            cp_val = float('inf')
        else:
            q = np.quantile(cal_resid, 0.9)
            cp_val = 2 * q
        cp_vals[pname] = cp_val

        # Effective sample size: fraction of training data where policy matches behavior
        # This is a simple ESS metric
        match_rate = np.mean([policy(x) == a for x, a in zip(X_tr, a_tr)])
        ess_vals[pname] = match_rate * len(X_tr)

    # ∆
    valid_policies = {k: v for k, v in cp_vals.items() if v != float('inf')}
    if len(valid_policies) < 2:
        return None

    cp_best = min(valid_policies, key=valid_policies.get)
    true_best = max(true_vals, key=true_vals.get)
    delta = int(cp_best != true_best)

    # κ: min pairwise gap / noise
    names = list(valid_policies.keys())
    kappas = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            gap = abs(cp_vals[names[i]] - cp_vals[names[j]])
            noise = (cp_vals[names[i]] + cp_vals[names[j]]) * 0.1
            kappas.append(gap / max(noise, 1e-10))
    kappa = min(kappas) if kappas else float('inf')

    return dict(theta=theta_prob1, delta=delta, kappa=kappa)


print("=== Bandit v2: DM + CP ===")
print(f"{'θ(1)':>6} {'∆':>8} {'SE':>8} {'κ':>8}")
print("-" * 35)
for theta in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    reps = Parallel(n_jobs=-1)(delayed(simulate)(theta) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    delta = df.delta.mean()
    se = np.sqrt(delta*(1-delta)/len(df))
    kappa = df.kappa.mean()
    print(f"{theta:>6.2f} {delta:>8.3f} {se:>8.3f} {kappa:>8.2f}")
