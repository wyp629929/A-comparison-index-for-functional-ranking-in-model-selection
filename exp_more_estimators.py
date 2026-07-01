"""
Estimator robustness: repeat decay experiment with Lasso, SVR, XGBoost.
Compares a regularized linear method (Lasso), a kernel method (SVR),
and a tree-based method (XGBoost) against OLS_full.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from scipy.stats import norm
import warnings; warnings.filterwarnings("ignore")
np.random.seed(2040)
F0 = 2 * norm.pdf(norm.ppf(0.95))

def run_one(n, est_type='lasso'):
    p, s, n_sig_true = 20, 3, 5
    X = np.random.randn(n, p)
    beta = np.zeros(p); beta[:n_sig_true] = [1.0, 1.0, 1.0, 0.5, 0.3]
    y = X @ beta + np.random.randn(n)
    nc = max(int(n * 0.25), 15); nt = nc; ntr = n - nc - nt
    if ntr < 10: return None
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt))

    ols = LinearRegression().fit(Xtr, ytr)
    if est_type == 'lasso':
        other = Lasso(alpha=0.1, max_iter=10000).fit(Xtr, ytr)
    elif est_type == 'svr':
        other = SVR(kernel='rbf', gamma='scale', C=1.0).fit(Xtr, ytr)
    elif est_type == 'xgboost':
        import xgboost as xgb
        other = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=None).fit(Xtr, ytr)

    r_f = np.abs(yca - ols.predict(Xca))
    r_o = np.abs(yca - other.predict(Xca))
    cp_f = 2 * np.quantile(r_f, 0.9)
    cp_o = 2 * np.quantile(r_o, 0.9)
    risk_f = mean_squared_error(yte, ols.predict(Xte))
    risk_o = mean_squared_error(yte, other.predict(Xte))

    raw_f = yca - ols.predict(Xca)
    raw_o = yca - other.predict(Xca)
    sigma_f = np.std(raw_f, ddof=1); sigma_o = np.std(raw_o, ddof=1)
    f_f = F0 / max(sigma_f, 1e-10); f_o = F0 / max(sigma_o, 1e-10)
    var_q_f = 0.09 / (max(nc, 1) * f_f**2)
    var_q_o = 0.09 / (max(nc, 1) * f_o**2)
    var_cp_f = 4 * var_q_f; var_cp_o = 4 * var_q_o
    cor = np.corrcoef(r_f, r_o)[0, 1] if len(r_f) > 1 else 0
    if np.isnan(cor): cor = 0
    cov_cp = cor * np.sqrt(var_cp_f * var_cp_o)
    sd_diff = np.sqrt(max(var_cp_f + var_cp_o - 2*cov_cp, 1e-10))
    kappa = abs(cp_f - cp_o) / max(sd_diff, 1e-10)
    delta = int((cp_o < cp_f) != (risk_o < risk_f))
    return dict(n=n, delta=delta, kappa=kappa)

N_REPS = 300
print(f"{'Estimator':>12} {'n':>5} {'∆':>8} {'κ':>8} {'corr(κ,∆)':>10}")
print("-" * 50)
for est in ['lasso', 'svr', 'xgboost']:
    for n in [100, 500, 1000]:
        reps = Parallel(n_jobs=-1)(delayed(run_one)(n, est) for _ in range(N_REPS))
        reps = [r for r in reps if r is not None]
        df = pd.DataFrame(reps)
        delta = df.delta.mean()
        kappa = df.kappa.mean()
        corr = df.kappa.corr(df.delta)
        print(f"{est:>12} {n:>5d} {delta:>8.3f} {kappa:>8.2f} {corr:>10.3f}")
