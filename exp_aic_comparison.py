"""
AIC comparison v2: cleaner design.
True model has s=3 non-zero coefficients.
Compare AIC for restricted (3 features) vs full (p features).
Sweep irrelevant features p to vary competition.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import warnings; warnings.filterwarnings("ignore")
np.random.seed(2040)

def run_one(p_full, n=500):
    s = 3
    X = np.random.randn(n, p_full)
    beta = np.zeros(p_full); beta[:s] = 1.0
    y = X @ beta + np.random.randn(n)

    nc = max(int(n * 0.25), 15); nt = nc; ntr = n - nc - nt
    if ntr < 10: return None

    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt))

    ols_f = LinearRegression().fit(Xtr, ytr)
    ols_r = LinearRegression().fit(Xtr[:, :s], ytr)
    n_tr = len(ytr)

    mse_f = mean_squared_error(ytr, ols_f.predict(Xtr))
    mse_r = mean_squared_error(ytr, ols_r.predict(Xtr[:, :s]))
    T_f = n_tr * np.log(mse_f) + 2 * p_full  # AIC
    T_r = n_tr * np.log(mse_r) + 2 * s       # AIC

    D_f = mean_squared_error(yte, ols_f.predict(Xte))
    D_r = mean_squared_error(yte, ols_r.predict(Xte[:, :s]))

    # κ via delta method: Var(AIC) ≈ n² * Var(MSE) / MSE²
    res_f = (ytr - ols_f.predict(Xtr))**2
    res_r = (ytr - ols_r.predict(Xtr[:, :s]))**2
    var_mse_f = np.var(res_f, ddof=1) / n_tr
    var_mse_r = np.var(res_r, ddof=1) / n_tr
    var_T_f = n_tr**2 * var_mse_f / (mse_f**2)
    var_T_r = n_tr**2 * var_mse_r / (mse_r**2)

    cor = np.corrcoef(res_f, res_r)[0, 1]
    if np.isnan(cor): cor = 0
    cov = cor * np.sqrt(var_T_f * var_T_r)
    sd_diff = np.sqrt(max(var_T_f + var_T_r - 2*cov, 1e-10))
    kappa = abs(T_f - T_r) / max(sd_diff, 1e-10)

    delta = int((T_r < T_f) != (D_r < D_f))
    return dict(n=n, p=p_full, delta=delta, kappa=kappa)

N_REPS = 500
print("=== AIC: restricted(s=3) vs full(p) ===")
print(f"{'p':>5} {'∆':>8} {'κ':>8} {'corr(κ,∆)':>10}")
for p in [10, 20, 50, 100]:
    reps = Parallel(n_jobs=-1)(delayed(run_one)(p, n=500) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    print(f"{p:>5d} {df.delta.mean():>8.3f} {df.kappa.mean():>8.2f} {df.kappa.corr(df.delta):>10.3f}")
