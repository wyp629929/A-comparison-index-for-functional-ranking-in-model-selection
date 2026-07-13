"""
Two-estimator controlled competition.

Estimator A: OLS on all p=20 features (full, higher variance)
Estimator B: OLS on only the s=3 signal features (restricted, lower variance, but no bias if signal is only there)

Both are unbiased (signal only in the first 3 features, coefficients = 1.0).
Difference: A has extra (p-s)=17 noise features → excess variance = 17σ²/n.
So the gap = 17/n * σ².

At n=17, gap=σ² (large, B dominates).
At n=170, gap=0.1σ² (small, close competition).
As n→∞, both converge to σ² (tie).

Vary ρ ∈ {0, 0.5, 0.75} and see if ∆ changes.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")
np.random.seed(2026)


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

    # A: full OLS
    ols_full = LinearRegression().fit(X_tr, y_tr)
    # B: restricted OLS (first s features)
    ols_rest = LinearRegression().fit(X_tr[:, :s], y_tr)

    results = {}
    for name, est, X_ca_m, X_te_m in [
        ("full", ols_full, X_ca, X_te),
        ("restr", ols_rest, X_ca[:, :s], X_te[:, :s])
    ]:
        cal_resid = np.abs(y_ca - est.predict(X_ca_m))
        q = np.quantile(cal_resid, 0.9)
        cp = 2 * q
        risk = mean_squared_error(y_te, est.predict(X_te_m))
        results[name] = (cp, risk)

    cp_f, risk_f = results["full"]
    cp_r, risk_r = results["restr"]

    best_cp = "full" if cp_f < cp_r else "restr"
    best_risk = "full" if risk_f < risk_r else "restr"

    spread = abs(cp_f - cp_r)
    noise = (cp_f + cp_r) / 2 * 0.1  # approx 10% CV of CP width
    kappa = spread / max(noise, 1e-10)

    return {
        "rho": rho, "n": n,
        "delta": int(best_cp != best_risk),
        "kappa": kappa,
        "cp_full": cp_f, "risk_full": risk_f,
        "cp_restr": cp_r, "risk_restr": risk_r,
    }


# — Grid —
N_VALS = [20, 30, 50, 100, 200, 500]
RHOS = [0.0, 0.5, 0.75]
N_REPS = 500

print("=== Two-estimator Controlled Competition ===")
print(f"n ∈ {N_VALS}, ρ ∈ {RHOS}, {N_REPS} reps each")
print(f"{'n':>5} {'ρ':>5} {'∆':>8} {'SE':>8} {'κ':>8} {'A_wins':>8} {'B_wins':>8}")
print("-" * 65)

cells = [(rho, n) for n in N_VALS for rho in RHOS for _ in range(N_REPS)]
results = Parallel(n_jobs=-1)(
    delayed(simulate_one)(rho, n) for rho, n in cells
)
results = [r for r in results if r is not None]
df = pd.DataFrame(results)

for n in N_VALS:
    for rho in RHOS:
        sub = df[(df["n"] == n) & (df["rho"] == rho)]
        if len(sub) == 0:
            continue
        delta = sub["delta"].mean()
        se = np.sqrt(delta * (1 - delta) / len(sub))
        kappa = sub["kappa"].mean()
        a_wins_cp = (sub["cp_full"] < sub["cp_restr"]).mean()
        b_wins_cp = (sub["cp_restr"] < sub["cp_full"]).mean()
        print(f"{n:>5} {rho:>5.2f} {delta:>8.3f} {se:>8.3f} {kappa:>8.2f} {a_wins_cp:>8.3f} {b_wins_cp:>8.3f}")

# — ρ effect on ∆ for each n —
print(f"\n=== ρ effect on ∆ ===")
for n in N_VALS:
    sub0 = df[(df["n"] == n) & (df["rho"] == 0.0)]
    sub75 = df[(df["n"] == n) & (df["rho"] == 0.75)]
    if len(sub0) == 0 or len(sub75) == 0:
        continue
    d0 = sub0["delta"].mean()
    d75 = sub75["delta"].mean()
    diff = d75 - d0
    se_diff = np.sqrt(d0*(1-d0)/len(sub0) + d75*(1-d75)/len(sub75))
    print(f"n={n:>4}: ∆_ρ=0={d0:.3f}, ∆_ρ=0.75={d75:.3f}, diff={diff:+.3f} (SE={se_diff:.3f})")
