"""
Targeted experiment: Two-estimator competition with controlled gap.

Setup: OLS_full (p=50, all features) vs OLS_restricted (p=5, true features only).
- Under linear DGP, OLS_full is unbiased but higher variance (p > n_train possible)
- OLS_restricted is biased (omits weak signals) but lower variance
- By varying n, we control the competition gap:
  - Small n: restricted wins (variance dominates)
  - Large n: full wins (bias dominates)
  - Intermediate n: they're close → high competition

Vary ρ ∈ {0, 0.5, 0.75} to see if competition channel is modulated.

Key prediction: ∆(ρ) should be modulated only when estimators are close.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import warnings, os

warnings.filterwarnings("ignore")
np.random.seed(2026)

RESULT_PATH = "/Users/wangyaoping/Desktop/experiment_results"


def make_block_cov(p, B, rho):
    m = p // B
    Sigma = np.eye(p)
    for b in range(B):
        idx = slice(b * m, (b + 1) * m)
        Sigma[idx, idx] = rho + (1 - rho) * np.eye(m)
    return Sigma


def simulate_one(rho, n, p=50, s=5):
    """Return ∆ for OLS_full vs OLS_restricted."""
    beta = np.zeros(p)
    beta[:s] = np.random.uniform(0.5, 1.5, s) * np.random.choice([-1, 1], s)

    # Block-correlated X
    B = min(p // 5, 10)
    Sigma = make_block_cov(p, B, rho)
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return None
    X = np.random.randn(n, p) @ L.T
    y = X @ beta + np.random.randn(n)

    # Split
    n_test = max(int(n * 0.2), 50)
    n_cal = max(int(n * 0.2), 50)
    n_train = n - n_test - n_cal
    if n_train < 5:
        return None

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=n_cal + n_test, random_state=None)
    X_cal, X_test, y_cal, y_test = train_test_split(X_tmp, y_tmp, test_size=n_test / (n_cal + n_test), random_state=None)

    # Estimators
    ols_full = LinearRegression()
    ols_restricted = LinearRegression()

    ols_full.fit(X_train, y_train)
    ols_restricted.fit(X_train[:, :s], y_train)

    # CP width
    for name, (est, X_tr) in [("full", (ols_full, X_train)), ("restr", (ols_restricted, X_train[:, :s]))]:
        cal_resid = np.abs(y_cal - est.predict(X_cal if X_cal.shape[1] == X_tr.shape[1] else X_cal[:, :s]))
        q = np.quantile(cal_resid, 0.9)
        cp_width = 2 * q

        y_pred = est.predict(X_test if X_test.shape[1] == X_tr.shape[1] else X_test[:, :s])
        risk = mean_squared_error(y_test, y_pred)

        if name == "full":
            cp_f, risk_f = cp_width, risk
        else:
            cp_r, risk_r = cp_width, risk

    best_cp = "full" if cp_f < cp_r else "restr"
    best_risk = "full" if risk_f < risk_r else "restr"

    # κ: competition density
    spread = abs(cp_f - cp_r)
    noise = np.std([cp_f, cp_r]) / np.sqrt(2)
    kappa = spread / max(noise, 1e-10)

    return {
        "rho": rho, "n": n, "p": p, "s": s,
        "delta": int(best_cp != best_risk),
        "kappa": kappa,
        "cp_f": cp_f, "risk_f": risk_f,
        "cp_r": cp_r, "risk_r": risk_r,
    }


# — Experiment: n × ρ grid —
N_VALS = [50, 100, 200, 500, 1000]
RHOS = [0.0, 0.5, 0.75]
N_REPS = 300

cells = []
for rho in RHOS:
    for n in N_VALS:
        for rep in range(N_REPS):
            cells.append((rho, n))

print("=== Two-estimator competition experiment ===")
print(f"n ∈ {N_VALS}, ρ ∈ {RHOS}, {N_REPS} reps each")
print(f"{'n':>5} {'ρ':>5} {'∆':>8} {'SE':>8} {'κ':>8}  {'cp_gap':>8} {'risk_gap':>8}")
print("-" * 65)

results = Parallel(n_jobs=-1)(
    delayed(simulate_one)(rho, n) for rho, n in cells
)
results = [r for r in results if r is not None]
df = pd.DataFrame(results)

for n in N_VALS:
    for rho in RHOS:
        sub = df[(df["n"] == n) & (df["rho"] == rho)]
        delta = sub["delta"].mean()
        se = np.sqrt(delta * (1 - delta) / len(sub))
        kappa = sub["kappa"].mean()
        cp_gap = (sub["cp_f"] - sub["cp_r"]).mean()
        risk_gap = (sub["risk_f"] - sub["risk_r"]).mean()
        print(f"{n:>5} {rho:>5.2f} {delta:>8.3f} {se:>8.3f} {kappa:>8.2f}  {cp_gap:>+8.3f} {risk_gap:>+8.3f}")

# — Key analysis: ρ effect at competition peak —
print("\n=== ρ effect at n where competition is highest ===")
for rho in RHOS:
    sub = df[df["rho"] == rho]
    # Find n with highest ∆
    n_peak = sub.groupby("n")["delta"].mean().idxmax()
    peak_sub = sub[sub["n"] == n_peak]
    print(f"ρ={rho:.2f}: peak n={n_peak}, ∆={peak_sub['delta'].mean():.3f}")

print("\n=== ρ effect on gap between estimators ===")
for n in [100, 200, 500]:
    sub0 = df[(df["n"] == n) & (df["rho"] == 0.0)]
    sub75 = df[(df["n"] == n) & (df["rho"] == 0.75)]
    gap0 = (sub0["risk_f"] - sub0["risk_r"]).mean()
    gap75 = (sub75["risk_f"] - sub75["risk_r"]).mean()
    print(f"n={n}: risk_gap ρ=0: {gap0:.3f}, ρ=0.75: {gap75:.3f}, change={gap75-gap0:+.3f}")
