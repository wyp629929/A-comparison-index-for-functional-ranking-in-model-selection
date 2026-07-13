"""Analyze competition experiment v2 — deeper checks"""
import pandas as pd
import numpy as np

# Re-import from the saved experiment
# Actually let me just compute from scratch with targeted queries
import warnings
warnings.filterwarnings("ignore")
np.random.seed(2026)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed


def make_block_cov(p, B, rho):
    m = p // B
    Sigma = np.eye(p)
    for b in range(B):
        idx = slice(b * m, (b + 1) * m)
        Sigma[idx, idx] = rho + (1 - rho) * np.eye(m)
    return Sigma


def simulate_one(rho, n, p=20, s=3, B=4):
    beta = np.zeros(p)
    beta[:s] = 1.0

    Sigma = make_block_cov(p, B, rho)
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T
    y = X @ beta + np.random.randn(n)

    n_test = max(int(n * 0.3), 20)
    n_cal = max(int(n * 0.2), 20)
    n_train = n - n_test - n_cal
    if n_train < s + 2:
        return None

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=n_cal + n_test, random_state=None)
    X_ca, X_te, y_ca, y_te = train_test_split(X_tmp, y_tmp, test_size=n_test / (n_cal + n_test), random_state=None)

    ols_full = LinearRegression().fit(X_tr, y_tr)
    ols_rest = LinearRegression().fit(X_tr[:, :s], y_tr)

    cp_f = 2 * np.quantile(np.abs(y_ca - ols_full.predict(X_ca)), 0.9)
    cp_r = 2 * np.quantile(np.abs(y_ca - ols_rest.predict(X_ca[:, :s])), 0.9)
    risk_f = mean_squared_error(y_te, ols_full.predict(X_te))
    risk_r = mean_squared_error(y_te, ols_rest.predict(X_te[:, :s]))

    return {
        "rho": rho, "n": n,
        "delta": int((cp_f < cp_r) != (risk_f < risk_r)),
        "cp_full": cp_f, "cp_restr": cp_r,
        "risk_full": risk_f, "risk_restr": risk_r,
    }


# — Key cells —
print("=== Detail: actual CP widths and risks ===")
for n, rho in [(50, 0.0), (50, 0.75), (500, 0.0), (500, 0.75)]:
    reps = [simulate_one(rho, n) for _ in range(300)]
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    print(f"\nn={n}, ρ={rho}:")
    print(f"  ∆ = {df['delta'].mean():.3f}")
    print(f"  CP_full  mean={df['cp_full'].mean():.3f}  se={df['cp_full'].std()/len(df)**.5:.3f}")
    print(f"  CP_restr mean={df['cp_restr'].mean():.3f}  se={df['cp_restr'].std()/len(df)**.5:.3f}")
    print(f"  Risk_full  mean={df['risk_full'].mean():.3f}")
    print(f"  Risk_restr mean={df['risk_restr'].mean():.3f}")
    # Fraction where CP says full is better
    cp_says_full = (df['cp_full'] < df['cp_restr']).mean()
    risk_says_full = (df['risk_full'] < df['risk_restr']).mean()
    print(f"  CP prefers full: {cp_says_full:.3f}")
    print(f"  Risk prefers full: {risk_says_full:.3f}")
