"""
Second illustration for general theory: use test MSE as selection functional.
Same decay experiment design: OLS_full(p=20) vs OLS_restr(s=3).
Selection functional T = training MSE (in-sample error).
Risk functional D = test MSE (out-of-sample error).
Tests whether κ-Δ relationship holds for this non-CP selection functional.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from scipy.stats import norm
import warnings; warnings.filterwarnings("ignore")
np.random.seed(2040)

def run_one(n, p=20, s=3, n_sig_true=5):
    X = np.random.randn(n, p)
    beta = np.zeros(p); beta[:n_sig_true] = [1.0, 1.0, 1.0, 0.5, 0.3]
    y = X @ beta + np.random.randn(n)
    nc = max(int(n * 0.25), 15); nt = nc; ntr = n - nc - nt
    if ntr < 10: return None
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt))
    ols_f = LinearRegression().fit(Xtr, ytr)
    ols_r = LinearRegression().fit(Xtr[:, :s], ytr)

    # Selection functional: training MSE (in-sample)
    train_pred_f = ols_f.predict(Xtr); train_pred_r = ols_r.predict(Xtr[:, :s])
    T_f = mean_squared_error(ytr, train_pred_f)  # selection functional
    T_r = mean_squared_error(ytr, train_pred_r)   # selection functional

    # Risk functional: test MSE (out-of-sample)
    D_f = mean_squared_error(yte, ols_f.predict(Xte))
    D_r = mean_squared_error(yte, ols_r.predict(Xte[:, :s]))

    # κ for MSE: variance of MSE via sample variance of squared residuals
    res_f = (ytr - train_pred_f)**2; res_r = (ytr - train_pred_r)**2
    var_T_f = np.var(res_f, ddof=1) / len(ytr)
    var_T_r = np.var(res_r, ddof=1) / len(ytr)
    cor = np.corrcoef(res_f, res_r)[0, 1] if len(res_f) > 1 else 0
    if np.isnan(cor): cor = 0
    cov = cor * np.sqrt(var_T_f * var_T_r)
    sd_diff = np.sqrt(max(var_T_f + var_T_r - 2*cov, 1e-10))
    kappa = abs(T_f - T_r) / max(sd_diff, 1e-10)

    delta = int((T_r < T_f) != (D_r < D_f))
    return dict(n=n, delta=delta, kappa=kappa)

N_REPS = 500
print("=== Training MSE as selection functional ===")
print(f"{'n':>5} {'∆':>8} {'κ':>8} {'corr(κ,∆)':>10}")
for n in [100, 500, 1000]:
    reps = Parallel(n_jobs=-1)(delayed(run_one)(n) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    print(f"{n:>5d} {df.delta.mean():>8.3f} {df.kappa.mean():>8.2f} {df.kappa.corr(df.delta):>10.3f}")
