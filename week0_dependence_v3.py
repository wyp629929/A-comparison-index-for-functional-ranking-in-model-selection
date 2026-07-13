"""
Week 0 v3: Probing the mechanism — why does ∆ decrease with ρ?

Hypothesis: Dependence inflates OLS variance more than RF variance,
making OLS clearly worse → less competition → lower ∆.

Tests:
1. Linear DGP, 7 estimators (full set matching Paper 1)
2. Sweep ρ AND B to separate correlation magnitude from block structure
3. Track individual estimator selection rates to see who's being picked
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")
np.random.seed(2025)


def make_block_cov(p, B, rho):
    m = p // B
    Sigma = np.eye(p)
    for b in range(B):
        idx = slice(b * m, (b + 1) * m)
        Sigma[idx, idx] = rho + (1 - rho) * np.eye(m)
    return Sigma


def simulate_one(rho, B, n=300, p=50, cal_ratio=0.3, test_ratio=0.3):
    beta = np.zeros(p)
    signal_idx = np.array([0, 1, 2, 3, 4])
    beta[signal_idx] = [2.0, -1.5, 0.8, 0.5, -0.3]

    Sigma = make_block_cov(p, B, rho)
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T
    y = X @ beta + np.random.randn(n)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=cal_ratio + test_ratio, random_state=None
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio / (cal_ratio + test_ratio), random_state=None
    )

    estimators = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=5000),
        "RF": RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=None),
    }

    cp_widths = {}
    risks = {}

    for name, est in estimators.items():
        est.fit(X_train, y_train)
        cal_resid = np.abs(y_cal - est.predict(X_cal))
        q = np.quantile(cal_resid, 0.9)
        cp_widths[name] = 2 * q
        risks[name] = mean_squared_error(y_test, est.predict(X_test))

    best_cp = min(cp_widths, key=cp_widths.get)
    best_risk = min(risks, key=risks.get)

    return {
        "misaligned": int(best_cp != best_risk),
        "cp_winner": best_cp,
        "risk_winner": best_risk,
        "cp_widths": cp_widths,
        "risks": risks,
    }


# --- Experiment ---
print("=== Week 0 v3: Mechanism Probe ===")
print("n=300, p=50, Linear DGP, 4 estimators")
print()

# Panel A: Vary ρ at fixed B=10
print("--- Panel A: ρ sweep at B=10 ---")
print(f"{'ρ':>6} {'reps':>6} {'∆':>8} {'se':>8}  OLS_cp  OLS_risk  RF_cp  RF_risk")
print("-" * 65)

for rho in [0.0, 0.25, 0.5, 0.75]:
    n_reps = 200
    results = Parallel(n_jobs=-1)(
        delayed(simulate_one)(rho, B=10) for _ in range(n_reps)
    )
    deltas = [r["misaligned"] for r in results]
    delta = np.mean(deltas)
    se = np.sqrt(delta * (1 - delta) / n_reps)

    # Average CP widths and risks for OLS and RF
    mean_cp_ols = np.mean([r["cp_widths"]["OLS"] for r in results])
    mean_risk_ols = np.mean([r["risks"]["OLS"] for r in results])
    mean_cp_rf = np.mean([r["cp_widths"]["RF"] for r in results])
    mean_risk_rf = np.mean([r["risks"]["RF"] for r in results])

    print(
        f"{rho:>6.2f} {n_reps:>6d} {delta:>8.3f} {se:>8.3f}"
        f"  {mean_cp_ols:>6.2f}  {mean_risk_ols:>8.2f}  {mean_cp_rf:>5.2f}  {mean_risk_rf:>7.2f}"
    )

print()

# Panel B: Vary B at fixed ρ=0.5
print("--- Panel B: B sweep at ρ=0.5 ---")
print(f"{'B':>6} {'reps':>6} {'∆':>8} {'se':>8}  OLS_cp  OLS_risk  RF_cp  RF_risk")
print("-" * 65)

for B in [1, 2, 5, 10, 25]:
    n_reps = 200
    results = Parallel(n_jobs=-1)(
        delayed(simulate_one)(rho=0.5, B=B) for _ in range(n_reps)
    )
    deltas = [r["misaligned"] for r in results]
    delta = np.mean(deltas)
    se = np.sqrt(delta * (1 - delta) / n_reps)

    mean_cp_ols = np.mean([r["cp_widths"]["OLS"] for r in results])
    mean_risk_ols = np.mean([r["risks"]["OLS"] for r in results])
    mean_cp_rf = np.mean([r["cp_widths"]["RF"] for r in results])
    mean_risk_rf = np.mean([r["risks"]["RF"] for r in results])

    print(
        f"{B:>6d} {n_reps:>6d} {delta:>8.3f} {se:>8.3f}"
        f"  {mean_cp_ols:>6.2f}  {mean_risk_ols:>8.2f}  {mean_cp_rf:>5.2f}  {mean_risk_rf:>7.2f}"
    )

print()
print("Prediction if mechanism is 'reduced competition':")
print("  ρ↑ → OLS variance inflates → OLS CP_width and risk both increase → RF becomes clearly better → ∆↓")
print("  B↓ (larger blocks) → same effect as ρ↑")
