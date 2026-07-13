"""
Intervention experiments: Generative predictions of the comparison-geometry framework.

Experiment 1: Negative correlation (ρ < 0). Prediction: ∆ ↓ because
  negative ρ increases OLS variance more than RF variance → wider spread → larger κ.

Experiment 2: Increasing p. Prediction: ∆ ↓ because
  more noise features increase OLS variance → OLS becomes clearly worse → less competition.

Both use 3 estimators: OLS, Ridge, RF on the linear DGP (p=20 baseline).
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
np.random.seed(2027)


def make_block_cov(p, B, rho):
    m = p // B
    if m <= 1 or B <= 0:
        return np.eye(p)
    Sigma = np.eye(p)
    for b in range(B):
        idx = slice(b * m, (b + 1) * m)
        Sigma[idx, idx] = rho + (1 - rho) * np.eye(m)
    return Sigma


def simulate_one(rho, p, n=200):
    s = 3
    beta = np.zeros(p)
    beta[:s] = 1.0

    B = max(p // 5, 1)
    Sigma = make_block_cov(p, B, rho)
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return None

    X = np.random.randn(n, p) @ L.T
    y = X @ beta + np.random.randn(n)

    n_test = max(int(n * 0.25), 20)
    n_cal = max(int(n * 0.25), 20)
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

    best_cp = min(cp_w, key=cp_w.get)
    best_risk = min(risks, key=risks.get)

    # κ: competition density
    w = np.array(list(cp_w.values()))
    spread = w.max() - w.min()
    noise = np.std(list(cp_w.values()))
    kappa = spread / max(noise, 1e-10)

    return {"rho": rho, "p": p, "delta": int(best_cp != best_risk), "kappa": kappa}


# — Experiment 1: Negative ρ —
print("=== Experiment 1: Negative ρ ===")
print(f"{'ρ':>8} {'∆':>8} {'SE':>8} {'κ':>8}")
print("-" * 40)

for rho in [-0.15, -0.10, -0.05, 0.0, 0.25, 0.50, 0.75]:
    reps = Parallel(n_jobs=-1)(delayed(simulate_one)(rho, p=20) for _ in range(500))
    reps = [r for r in reps if r is not None]
    delta = np.mean([r["delta"] for r in reps])
    se = np.sqrt(delta * (1 - delta) / len(reps))
    kappa = np.mean([r["kappa"] for r in reps])
    print(f"{rho:>8.2f} {delta:>8.3f} {se:>8.3f} {kappa:>8.2f}")

# — Experiment 2: p effect —
print("\n=== Experiment 2: p effect (ρ=0.25) ===")
print(f"{'p':>8} {'∆':>8} {'SE':>8} {'κ':>8}")
print("-" * 40)

for p in [5, 10, 20, 50, 100]:
    reps = Parallel(n_jobs=-1)(delayed(simulate_one)(rho=0.25, p=p) for _ in range(500))
    reps = [r for r in reps if r is not None]
    delta = np.mean([r["delta"] for r in reps])
    se = np.sqrt(delta * (1 - delta) / len(reps))
    kappa = np.mean([r["kappa"] for r in reps])
    print(f"{p:>8d} {delta:>8.3f} {se:>8.3f} {kappa:>8.2f}")

# — Experiment 2b: p effect at ρ=0 (robustness) —
print("\n=== Experiment 2b: p effect (ρ=0) ===")
for p in [5, 10, 20, 50, 100]:
    reps = Parallel(n_jobs=-1)(delayed(simulate_one)(rho=0.0, p=p) for _ in range(500))
    reps = [r for r in reps if r is not None]
    delta = np.mean([r["delta"] for r in reps])
    se = np.sqrt(delta * (1 - delta) / len(reps))
    kappa = np.mean([r["kappa"] for r in reps])
    print(f"{p:>8d} {delta:>8.3f} {se:>8.3f} {kappa:>8.2f}")
