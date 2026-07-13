"""
Week 0 v2: CP Model Selection Under Dependence — Multi-estimator competition.

Key insight from v1: With 2 estimators and linear DGP, OLS dominates → ∆=0.
Need competition. Use 5 estimators and Threshold DGP where bias crossing already exists.

Question: Does dependence (block-correlated X) increase ∆ beyond Paper 1's baseline?
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import warnings
import time

warnings.filterwarnings("ignore")
np.random.seed(2024)


def make_block_cov(p, B, rho):
    m = p // B
    Sigma = np.eye(p)
    for b in range(B):
        idx = slice(b * m, (b + 1) * m)
        Sigma[idx, idx] = rho + (1 - rho) * np.eye(m)
    return Sigma


def simulate_one(rho, B, dgp="linear", n=500, p=50, cal_ratio=0.3, test_ratio=0.3):
    if dgp == "linear":
        beta = np.zeros(p)
        signal_idx = np.array([0, 1, 2, 3, 4])
        beta[signal_idx] = [2.0, -1.5, 0.8, 0.5, -0.3]
        Sigma = make_block_cov(p, B, rho)
        L = np.linalg.cholesky(Sigma)
        X = np.random.randn(n, p) @ L.T
        f0 = X @ beta
        y = f0 + np.random.randn(n)

    elif dgp == "threshold":
        # Paper 1 Threshold DGP: y = 2·1(X1>0) + log(1+|X2|)·1(X3>0) - 1 + ε
        Sigma = make_block_cov(p, B, rho)
        L = np.linalg.cholesky(Sigma)
        X = np.random.randn(n, p) @ L.T
        f0 = 2.0 * (X[:, 0] > 0) + np.log(1 + np.abs(X[:, 1])) * (X[:, 2] > 0) - 1
        y = f0 + np.random.randn(n)

    elif dgp == "nonlinear":
        # Paper 1 Nonlinear DGP: y = sin(X1) + log(1+|X2|) + X3*X4 + ε
        Sigma = make_block_cov(p, B, rho)
        L = np.linalg.cholesky(Sigma)
        X = np.random.randn(n, p) @ L.T
        f0 = np.sin(X[:, 0]) + np.log(1 + np.abs(X[:, 1])) + X[:, 2] * X[:, 3]
        y = f0 + np.random.randn(n)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=cal_ratio + test_ratio, random_state=None
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio / (cal_ratio + test_ratio), random_state=None
    )

    # 5 estimators
    estimators = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=5000),
        "RF": RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=None),
        "XGB": None,  # skip to save time, use 4
    }

    cp_widths = {}
    risks = {}

    for name, est in estimators.items():
        if est is None:
            continue
        est.fit(X_train, y_train)

        # CP width
        cal_resid = np.abs(y_cal - est.predict(X_cal))
        q = np.quantile(cal_resid, 0.9)
        cp_widths[name] = 2 * q

        # Risk = test MSE (oracle for global selection)
        risks[name] = np.mean((y_test - est.predict(X_test)) ** 2)

    # Misalignment: does CP argmin differ from risk argmin?
    best_cp = min(cp_widths, key=cp_widths.get)
    best_risk = min(risks, key=risks.get)
    return int(best_cp != best_risk)


# --- Main experiment ---
print("=== Week 0 v2: CP Model Selection Under Dependence ===")
print("n=500, p=50, B=10, 4 estimators (OLS/Ridge/Lasso/RF)")
print()

for dgp_name in ["linear", "threshold", "nonlinear"]:
    print(f"--- DGP: {dgp_name} ---")
    print(f"{'ρ':>6} {'reps':>6} {'∆':>8} {'se':>8}  {'time':>8}")
    print("-" * 42)

    for rho in [0.0, 0.25, 0.5, 0.75]:
        t0 = time.time()
        n_reps = 100
        results = Parallel(n_jobs=-1)(
            delayed(simulate_one)(rho, B=10, dgp=dgp_name) for _ in range(n_reps)
        )
        delta = np.mean(results)
        se = np.sqrt(delta * (1 - delta) / n_reps)
        elapsed = time.time() - t0
        print(f"{rho:>6.2f} {n_reps:>6d} {delta:>8.3f} {se:>8.3f}  {elapsed:>7.0f}s")

    print()
