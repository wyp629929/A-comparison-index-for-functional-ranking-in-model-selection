"""
Factorial experiment: Decomposing ∆ into bias-crossing and competition channels.

Design:
  DGPs:      Linear (low bias crossing) vs Threshold (high bias crossing)
  ρ:         0, 0.5, 0.75
  Estimator sets:
    - Linear: {OLS, Ridge, Lasso}          — same bias family, no crossing
    - Mixed:  {OLS, Ridge, Lasso, RF, XGB} — crossing exists on Threshold

Predictions:
  1. On Linear DGP, Mixed set has higher ∆ than Linear set due to competition
  2. On Threshold DGP, Mixed set has even higher ∆ due to bias crossing + competition
  3. ρ increases → Mixed ∆ changes (competition modulated), Linear ∆ stays flat
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import warnings, time, os

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


def simulate_one(cell):
    dgp_name, est_names, rho, B, n = cell["dgp"], cell["ests"], cell["rho"], cell["B"], cell["n"]
    p = cell["p"]
    n_cal = int(n * 0.3)
    n_test = int(n * 0.3)
    n_train = n - n_cal - n_test

    # — DGP —
    if dgp_name == "linear":
        beta = np.zeros(p)
        beta[:5] = [2.0, -1.5, 0.8, 0.5, -0.3]
        Sigma = make_block_cov(p, B, rho)
        L = np.linalg.cholesky(Sigma)
        X = np.random.randn(n, p) @ L.T
        f0 = X @ beta
        y = f0 + np.random.randn(n)
    elif dgp_name == "threshold":
        Sigma = make_block_cov(p, B, rho)
        L = np.linalg.cholesky(Sigma)
        X = np.random.randn(n, p) @ L.T
        f0 = 2.0 * (X[:, 0] > 0) + np.log(1 + np.abs(X[:, 1])) * (X[:, 2] > 0) - 1
        y = f0 + np.random.randn(n)

    # — Split —
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=n_cal + n_test, random_state=None)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=n_test / (n_cal + n_test), random_state=None)

    # — Estimators —
    pool = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=5000),
        "RF": RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=None),
        "XGB": XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, verbosity=0, random_state=None),
    }
    estimators = {k: pool[k] for k in est_names}

    cp_widths, risks = {}, {}
    for name, est in estimators.items():
        est.fit(X_train, y_train)
        cal_resid = np.abs(y_cal - est.predict(X_cal))
        q = np.quantile(cal_resid, 0.9)
        cp_widths[name] = 2 * q
        risks[name] = mean_squared_error(y_test, est.predict(X_test))

    best_cp = min(cp_widths, key=cp_widths.get)
    best_risk = min(risks, key=risks.get)

    # κ: competition density = (max_cp - min_cp) / mean(se_cp)
    # where se_cp is bootstrap estimate of cp width variability
    widths_arr = np.array(list(cp_widths.values()))
    spread = widths_arr.max() - widths_arr.min()
    # Use calibration residual variance as a proxy for CP width variability
    noise = np.mean([(2 * np.std(np.abs(y_cal - estimators[k].predict(X_cal))) / np.sqrt(len(y_cal)))
                     for k in estimators])
    kappa = spread / max(noise, 1e-10)

    return {
        "dgp": dgp_name,
        "ests": "+".join(est_names),
        "rho": rho,
        "B": B,
        "n": n,
        "p": p,
        "delta": int(best_cp != best_risk),
        "kappa": kappa,
        "est_key": cell["est_key"],
        "cp_winner": best_cp,
        "risk_winner": best_risk,
    }


# — Design grid —
DGPS = ["linear", "threshold"]
RHOS = [0.0, 0.5, 0.75]
EST_SETS = {
    "Linear3": ["OLS", "Ridge", "Lasso"],
    "Mixed5": ["OLS", "Ridge", "Lasso", "RF", "XGB"],
}
B = 10
N = 500
P = 50
N_REPS = 200

cells = []
for dgp in DGPS:
    for est_key, est_list in EST_SETS.items():
        for rho in RHOS:
            for rep in range(N_REPS):
                cells.append({
                    "dgp": dgp, "ests": est_list, "est_key": est_key,
                    "rho": rho, "B": B, "n": N, "p": P, "rep": rep,
                })

print(f"=== Factorial Experiment: {len(cells)} cells ===")
print(f"DGPs: {DGPS}")
print(f"ρ: {RHOS}")
print(f"Estimator sets: {list(EST_SETS.keys())}")
print(f"Replications per cell: {N_REPS}")
print()

t0 = time.time()
results = Parallel(n_jobs=-1, verbose=1)(
    delayed(simulate_one)(c) for c in cells
)
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s")

# — Aggregate —
df = pd.DataFrame(results)
os.makedirs(RESULT_PATH, exist_ok=True)
df.to_csv(f"{RESULT_PATH}/raw_results.csv", index=False)

print("\n=== Results ===")
print(f"{'DGP':<12} {'EstSet':<10} {'ρ':<6} {'∆':<8} {'SE':<8} {'κ':<8}")
print("-" * 52)

for dgp in DGPS:
    for est_key in EST_SETS:
        for rho in RHOS:
            sub = df[(df["dgp"] == dgp) & (df["est_key"] == est_key) & (df["rho"] == rho)]
            delta = sub["delta"].mean()
            se = np.sqrt(delta * (1 - delta) / len(sub))
            kappa = sub["kappa"].mean()
            print(f"{dgp:<12} {est_key:<10} {rho:<6.2f} {delta:<8.3f} {se:<8.3f} {kappa:<8.2f}")
    print()

# — Winner analysis —
print("=== Winner rates (ρ=0.75 parens) ===")
for dgp in DGPS:
    for est_key in EST_SETS:
        sub = df[(df["dgp"] == dgp) & (df["est_key"] == est_key) & (df["rho"] == 0.0)]
        sub75 = df[(df["dgp"] == dgp) & (df["est_key"] == est_key) & (df["rho"] == 0.75)]
        print(f"{dgp:<12} {est_key:<10}  ρ=0 CP winners:  {sub['cp_winner'].value_counts().to_dict()}")
        print(f"{'':<12} {'':<10}  ρ=0 Risk winners: {sub['risk_winner'].value_counts().to_dict()}")
        print(f"{'':<12} {'':<10}  ρ=.75 CP winners: {sub75['cp_winner'].value_counts().to_dict()}")
        print(f"{'':<12} {'':<10}  ρ=.75 Risk win:   {sub75['risk_winner'].value_counts().to_dict()}")
    print()
