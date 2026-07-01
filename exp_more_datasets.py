"""
Additional real datasets (OpenML) for real-data validation.
Runs OLS vs Ridge on Boston, Concrete, Abalone, Wine Quality.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_openml
from joblib import Parallel, delayed
from scipy.stats import norm
import warnings; warnings.filterwarnings("ignore")
np.random.seed(2038)
F0 = 2 * norm.pdf(norm.ppf(0.95))

def run_one(X_full, y_full, n):
    idx = np.random.choice(len(X_full), n, replace=False)
    X, y = X_full[idx], y_full[idx]
    nc = max(int(n * 0.25), 10); nt = nc; ntr = n - nc - nt
    if ntr < 10: return None
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt, random_state=None)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt), random_state=None)
    ols = LinearRegression().fit(Xtr, ytr)
    ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
    cp_ols = 2 * np.quantile(np.abs(yca - ols.predict(Xca)), 0.9)
    cp_ridge = 2 * np.quantile(np.abs(yca - ridge.predict(Xca)), 0.9)
    risk_ols = mean_squared_error(yte, ols.predict(Xte))
    risk_ridge = mean_squared_error(yte, ridge.predict(Xte))
    delta = int((cp_ridge < cp_ols) != (risk_ridge < risk_ols))
    raw_ols = yca - ols.predict(Xca); raw_ridge = yca - ridge.predict(Xca)
    sigma_ols = np.std(raw_ols, ddof=1); sigma_ridge = np.std(raw_ridge, ddof=1)
    f_ols = F0 / max(sigma_ols, 1e-10); f_ridge = F0 / max(sigma_ridge, 1e-10)
    var_q_ols = 0.09 / (max(nc, 1) * f_ols**2)
    var_q_ridge = 0.09 / (max(nc, 1) * f_ridge**2)
    var_cp_ols = 4 * var_q_ols; var_cp_ridge = 4 * var_q_ridge
    cor = np.corrcoef(np.abs(raw_ols), np.abs(raw_ridge))[0, 1]
    if np.isnan(cor): cor = 0
    cov_cp = cor * np.sqrt(var_cp_ols * var_cp_ridge)
    sd_diff = np.sqrt(max(var_cp_ols + var_cp_ridge - 2*cov_cp, 1e-10))
    kappa = abs(cp_ols - cp_ridge) / max(sd_diff, 1e-10)
    return dict(ddelta=delta, kappa=kappa)

N_REPS = 300

datasets = {
    'boston': fetch_openml('boston', parser='auto'),
    'concrete': fetch_openml('Concrete_Data', parser='auto'),
    'abalone': fetch_openml('abalone', parser='auto'),
    'wine_quality': fetch_openml('wine_quality', parser='auto'),
}

print(f"{'Dataset':>15} {'n':>5} {'∆':>8} {'SE':>8} {'κ':>8}")
print("-" * 50)

for name, data in datasets.items():
    X_df = data.data if hasattr(data.data, 'iloc') else pd.DataFrame(data.data)
    # Drop non-numeric columns
    X_df = X_df.select_dtypes(include=['number'])
    if data.target is not None:
        y = data.target.values if hasattr(data.target, 'values') else data.target
    else:
        y = X_df.iloc[:, -1].values.astype(float)
        X_df = X_df.iloc[:, :-1]
    X = X_df.values.astype(float)
    y = y.astype(float)
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        nan_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[nan_mask]; y = y[nan_mask]
    n_total = len(X)
    # Convert classification targets to regression if needed
    if y.dtype in ('int', 'int64', 'object'):
        y = y.astype(float)
    ns = [min(200, n_total), min(500, n_total), min(1000, n_total)]
    ns = sorted(set(ns))
    for n in ns:
        reps = Parallel(n_jobs=-1)(delayed(run_one)(X, y, n) for _ in range(N_REPS))
        reps = [r for r in reps if r is not None]
        df = pd.DataFrame(reps)
        delta = df.ddelta.mean()
        se = np.sqrt(delta*(1-delta)/len(df))
        kappa = df.kappa.mean()
        print(f"{name:>15} {n:>5d} {delta:>8.3f} {se:>8.4f} {kappa:>8.2f}")
