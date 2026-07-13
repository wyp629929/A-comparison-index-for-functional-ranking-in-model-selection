"""
Week 0 Reconnaissance: Does dependence increase misalignment in CP-based model selection?

Test: Linear DGP with block-equicorrelated X.
- OLS vs RF: do their CP width rankings agree with conditional risk rankings?
- Vary ρ ∈ {0, 0.25, 0.5} — does ∆ increase monotonically?

Paper 1 baseline: linear regime, independent X → ∆ ≈ 0.600 (7 estimators, 200 reps)
This test: 2 estimators with controlled block dependence.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")
np.random.seed(2024)

# --- DGP ---
def make_block_cov(p, B, rho):
    """Block equicorrelation covariance matrix."""
    m = p // B
    Sigma = np.eye(p)
    for b in range(B):
        idx = slice(b * m, (b + 1) * m)
        Sigma[idx, idx] = rho + (1 - rho) * np.eye(m)
    return Sigma

def simulate_one(rho, B, n=500, p=50, s=5, cal_ratio=0.3, test_ratio=0.3):
    """Single replication: return 1 if CP width ranking disagrees with risk ranking, else 0."""
    # Sparse coefficients
    beta = np.zeros(p)
    signal_idx = np.random.choice(p, s, replace=False)
    beta[signal_idx] = np.random.uniform(1.0, 2.0, s) * np.random.choice([-1, 1], s)

    # Block-correlated X
    Sigma = make_block_cov(p, B, rho)
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T
    y = X @ beta + np.random.randn(n)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=cal_ratio + test_ratio, random_state=None)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio / (cal_ratio + test_ratio), random_state=None)

    # Fit estimators
    estimators = {
        "OLS": LinearRegression(),
        "RF": RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=None),
    }

    results = {}
    for name, est in estimators.items():
        est.fit(X_train, y_train)

        # CP width: quantile of absolute residuals on calibration set
        cal_resid = np.abs(y_cal - est.predict(X_cal))
        q = np.quantile(cal_resid, 0.9)  # α = 0.1
        cp_width = 2 * q

        # Conditional risk oracle (known DGP): bias² + σ²
        # E[ŷ|x] for OLS is Xβ_hat (biased if misspecified, but here correctly specified)
        # For OLS: E[ŷ] = X @ β_hat where β_hat is OLS estimate
        # For RF: E[ŷ] approximated by prediction (since true RF expectation is complex)
        # We approximate conditional risk as (y_true - ŷ)² + noise variance
        # But that's not quite right. Let me use a simpler approach:
        # Since we know the true DGP, D(M|x) = (f0(x) - E[ŷ_M(x)])² + Var(ŷ_M(x))
        # We estimate E[ŷ_M(x)] by fitting on many MC samples

        y_pred = est.predict(X_test)
        f0 = X_test @ beta

        # Simple oracle: (f0 - ŷ)² (ignores variance of ŷ, but that's OK for comparison)
        # Actually, for the conditional risk oracle D(M|x):
        # D(M|x) = E[(Y - ŷ_M(X))² | X=x] = (f0(x) - E[ŷ_M(x)])² + Var(ŷ_M(x))
        # Under homoscedastic ε, Var(ŷ_M(x)) depends on the estimator's variance.
        # For OLS, Var(ŷ_OLS|x) = σ² x'(X_train'X_train)⁻¹x
        # For RF, it's more complex.
        # For this Week 0, I'll use a simpler approximation:
        # D(M|x) ≈ (y - ŷ)² averaged over test points (this is the test MSE)

        test_mse = np.mean((y_test - y_pred) ** 2)
        results[name] = {"cp_width": cp_width, "risk": test_mse}

    # Misalignment: does CP width ranking disagree with risk ranking?
    cp_ranking = "OLS" if results["OLS"]["cp_width"] < results["RF"]["cp_width"] else "RF"
    risk_ranking = "OLS" if results["OLS"]["risk"] < results["RF"]["risk"] else "RF"
    return int(cp_ranking != risk_ranking)


# --- Experiment ---
print("=== Week 0: CP Model Selection Under Dependence ===")
print(f"p=50, n=500, s=5, B=10 (blocks of size 5)")
print(f"Estimators: OLS vs RF")
print(f"{'ρ':>6} {'reps':>6} {'∆':>8} {'se':>8}")
print("-" * 30)

for rho in [0.0, 0.25, 0.5, 0.75]:
    B = 10
    n_reps = 200
    results = Parallel(n_jobs=-1)(
        delayed(simulate_one)(rho, B) for _ in range(n_reps)
    )
    delta = np.mean(results)
    se = np.sqrt(delta * (1 - delta) / n_reps)
    print(f"{rho:>6.2f} {n_reps:>6d} {delta:>8.3f} {se:>8.3f}")

print("\nPrediction: if dependence increases misalignment, ∆ should increase with ρ.")
