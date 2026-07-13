"""
Intervention experiments v2 — corrected κ calculation.

κ = min_{i,j} |mean(R_i) - mean(R_j)| / SD(R_i - R_j)
This is the minimum competition density: how many SDs apart are the closest pair of estimators?

Experiment 1: Negative ρ (prediction: ∆ ↓ due to wider spread)
Experiment 2: Increasing p (prediction: ∆ ↓ due to OLS variance dominating)
Experiment 3: n effects on ∆ / κ relationship
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")
np.random.seed(2028)


def make_block_cov(p, B, rho):
    m = p // B
    if m <= 1:
        return np.eye(p)
    Sigma = np.eye(p)
    for b in range(B):
        idx = slice(b * m, (b + 1) * m)
        Sigma[idx, idx] = rho + (1 - rho) * np.eye(m)
    return Sigma


def simulate_one(rho, p, n=300, est_set="ridge_rf"):
    """Return CP widths and risks for all estimators."""
    s = 3
    beta = np.zeros(p)
    beta[:s] = 1.0

    B = max(p // 5, 1)
    Sigma = make_block_cov(p, B, rho)
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T
    y = X @ beta + np.random.randn(n)

    n_test = max(int(n * 0.2), 10)
    n_cal = max(int(n * 0.2), 10)
    n_train = n - n_test - n_cal
    if n_train < 5:
        return None

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=n_cal + n_test, random_state=None)
    X_ca, X_te, y_ca, y_te = train_test_split(X_tmp, y_tmp, test_size=n_test/(n_cal + n_test), random_state=None)

    ests = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RF": RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=None),
    }

    cp_w, risks = {}, {}
    for name, est in ests.items():
        est.fit(X_tr, y_tr)
        cp_w[name] = 2 * np.quantile(np.abs(y_ca - est.predict(X_ca)), 0.9)
        risks[name] = mean_squared_error(y_te, est.predict(X_te))

    return {"rho": rho, "p": p, "n": n,
            "cp_ols": cp_w["OLS"], "cp_ridge": cp_w["Ridge"], "cp_rf": cp_w["RF"],
            "risk_ols": risks["OLS"], "risk_ridge": risks["Ridge"], "risk_rf": risks["RF"]}


def analyze_one(df, verbose=False):
    """From a DataFrame of reps (all same condition), compute ∆ and κ."""
    n_reps = len(df)
    # ∆
    cols = [c for c in ["cp_ols", "cp_ridge", "cp_rf"] if c in df.columns]
    risk_cols = [c.replace("cp_", "risk_") for c in cols]
    cp_best = df[cols].idxmin(axis=1)
    risk_best = df[risk_cols].idxmin(axis=1)
    delta = (cp_best != risk_best).mean()

    if verbose:
        print(f"  CP win rates:  {cp_best.value_counts(normalize=True).to_dict()}")
        print(f"  Risk win rates: {risk_best.value_counts(normalize=True).to_dict()}")

    # κ: min pairwise gap / SD of difference
    names = cols
    kappa_vals = []
    for i in range(3):
        for j in range(i+1, 3):
            diff = df[names[i]] - df[names[j]]
            gap = abs(diff.mean())
            sd = diff.std()
            if sd > 1e-10:
                kappa_vals.append(gap / sd)

    kappa = min(kappa_vals) if kappa_vals else float('inf')

    return delta, kappa


N_REPS = 400

# — Experiment 1: ρ sweep —
print("=== Experiment 1: ρ sweep (p=20, n=300) ===")
print(f"{'ρ':>8} {'∆':>8} {'SE':>8} {'κ':>8}")
print("-" * 40)

rho_vals = [-0.15, -0.10, -0.05, 0.0, 0.25, 0.50, 0.75]
for rho in rho_vals[:2]:  # Just first 2 for debug
    reps = Parallel(n_jobs=-1)(delayed(simulate_one)(rho, p=20, n=300) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    delta, kappa = analyze_one(df, verbose=True)
    se = np.sqrt(delta * (1 - delta) / len(df))
    print(f"{rho:>8.2f} {delta:>8.3f} {se:>8.3f} {kappa:>8.2f}")

# — Experiment 2: p sweep (ρ=0.25) —
print("\n=== Experiment 2: p sweep (ρ=0.25, n=300) ===")
for p in [5, 10, 20, 50, 100]:
    reps = Parallel(n_jobs=-1)(delayed(simulate_one)(rho=0.25, p=p, n=300) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    delta, kappa = analyze_one(df)
    se = np.sqrt(delta * (1 - delta) / len(df))
    print(f"{p:>8d} {delta:>8.3f} {se:>8.3f} {kappa:>8.2f}")

# — Experiment 3: n effect on κ–∆ relationship (p=10 to handle small n) —
print("\n=== Experiment 3: n effect (ρ=0, p=10) ===")
for n in [30, 50, 100, 200, 500, 1000]:
    reps = Parallel(n_jobs=-1)(delayed(simulate_one)(rho=0.0, p=10, n=n) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    delta, kappa = analyze_one(df)
    se = np.sqrt(delta * (1 - delta) / len(df))
    print(f"{n:>5d} {delta:>8.3f} {se:>8.3f} {kappa:>8.2f}")
