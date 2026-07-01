"""
Real data experiment: validate κ on UCI/sklearn datasets.
Compare OLS vs Ridge on diabetes and california housing.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes, fetch_california_housing
from joblib import Parallel, delayed
from scipy.stats import norm
F0 = 2 * norm.pdf(norm.ppf(0.95))  # ≈ 0.206
import warnings
warnings.filterwarnings("ignore")
np.random.seed(2038)

N_REPS = 500

def run_one_real(X_full, y_full, n, alpha=1.0):
    """Sample n rows, split, compute κ and ∆."""
    idx = np.random.choice(len(X_full), n, replace=False)
    X, y = X_full[idx], y_full[idx]

    nc = max(int(n * 0.25), 10); nt = nc; ntr = n - nc - nt
    if ntr < 10: return None

    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt, random_state=None)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt), random_state=None)

    ols = LinearRegression().fit(Xtr, ytr)
    ridge = Ridge(alpha=alpha).fit(Xtr, ytr)

    r_ols = np.abs(yca - ols.predict(Xca))
    r_ridge = np.abs(yca - ridge.predict(Xca))
    cp_ols = 2 * np.quantile(r_ols, 0.9)
    cp_ridge = 2 * np.quantile(r_ridge, 0.9)

    risk_ols = mean_squared_error(yte, ols.predict(Xte))
    risk_ridge = mean_squared_error(yte, ridge.predict(Xte))

    raw_ols = yca - ols.predict(Xca)
    sigma_ols = np.std(raw_ols, ddof=1)
    raw_ridge = yca - ridge.predict(Xca)
    sigma_ridge = np.std(raw_ridge, ddof=1)
    f_ols = F0 / max(sigma_ols, 1e-10)
    f_ridge = F0 / max(sigma_ridge, 1e-10)

    var_q_ols = 0.9 * 0.1 / (max(nc, 1) * f_ols**2)
    var_q_ridge = 0.9 * 0.1 / (max(nc, 1) * f_ridge**2)
    var_cp_ols = 4 * var_q_ols
    var_cp_ridge = 4 * var_q_ridge

    cor = np.corrcoef(r_ols, r_ridge)[0, 1] if len(r_ols) > 1 else 0
    if np.isnan(cor): cor = 0
    cov_cp = cor * np.sqrt(var_cp_ols * var_cp_ridge)
    sd_diff = np.sqrt(max(var_cp_ols + var_cp_ridge - 2*cov_cp, 1e-10))
    kappa = abs(cp_ols - cp_ridge) / max(sd_diff, 1e-10)

    delta = int((cp_ridge < cp_ols) != (risk_ridge < risk_ols))
    return dict(dataset='diabetes' if len(X_full) < 1000 else 'housing',
                n=n, delta=delta, kappa=kappa,
                cp_gap=cp_ols-cp_ridge, risk_gap=risk_ols-risk_ridge,
                nc=nc)


print("=== Real data experiments ===")
print(f"{'Dataset':>10} {'n':>5} {'∆':>8} {'SE':>8} {'κ':>8}")
print("-" * 45)

# Diabetes (n=442, p=10)
diabetes = load_diabetes()
X_dia, y_dia = diabetes.data, diabetes.target

# California housing (n≈20640, p=8)
housing = fetch_california_housing()
X_hou, y_hou = housing.data[:2000], housing.target[:2000]  # subsample for speed

for name, X, y, ns in [('diabetes', X_dia, y_dia, [100, 200, 442]),
                        ('housing', X_hou, y_hou, [100, 200, 500, 1000])]:
    for n in ns:
        reps = Parallel(n_jobs=-1)(delayed(run_one_real)(X, y, n) for _ in range(N_REPS))
        reps = [r for r in reps if r is not None]
        df = pd.DataFrame(reps)
        delta = df.delta.mean()
        se = np.sqrt(delta*(1-delta)/len(df))
        kappa = df.kappa.mean()
        print(f"{name:>10} {n:>5d} {delta:>8.4f} {se:>8.4f} {kappa:>8.3f}")

        # Save to all_data for central figure
        df.to_csv(f"/Users/wangyaoping/Desktop/paper3/experiment_results/{name}_n{n}.csv", index=False)

print("\nDone.")
