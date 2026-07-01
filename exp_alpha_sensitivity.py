"""
Sensitivity analysis: κ under different α values.
Tests whether κ-Δ relationship depends on the CP level.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from scipy.stats import norm
import warnings; warnings.filterwarnings("ignore")
np.random.seed(2040)
F0 = lambda alpha: 2 * norm.pdf(norm.ppf(1 - alpha/2))

def run_one(n, alpha=0.1):
    p, s, n_sig_true = 20, 3, 5
    X = np.random.randn(n, p)
    beta = np.zeros(p); beta[:n_sig_true] = [1.0, 1.0, 1.0, 0.5, 0.3]
    y = X @ beta + np.random.randn(n)
    nc = max(int(n * 0.25), 15); nt = nc; ntr = n - nc - nt
    if ntr < 10: return None
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt))
    ols_f = LinearRegression().fit(Xtr, ytr)
    ols_r = LinearRegression().fit(Xtr[:, :s], ytr)
    r_f = np.abs(yca - ols_f.predict(Xca))
    r_r = np.abs(yca - ols_r.predict(Xca[:, :s]))
    cp_f = 2 * np.quantile(r_f, 1 - alpha)
    cp_r = 2 * np.quantile(r_r, 1 - alpha)
    risk_f = mean_squared_error(yte, ols_f.predict(Xte))
    risk_r = mean_squared_error(yte, ols_r.predict(Xte[:, :s]))
    raw_f = yca - ols_f.predict(Xca)
    raw_r = yca - ols_r.predict(Xca[:, :s])
    sigma_f = np.std(raw_f, ddof=1); sigma_r = np.std(raw_r, ddof=1)
    f_f = F0(alpha) / max(sigma_f, 1e-10); f_r = F0(alpha) / max(sigma_r, 1e-10)
    a = alpha
    var_q_f = a * (1-a) / (max(nc, 1) * f_f**2)
    var_q_r = a * (1-a) / (max(nc, 1) * f_r**2)
    var_cp_f = 4 * var_q_f; var_cp_r = 4 * var_q_r
    cor = np.corrcoef(r_f, r_r)[0, 1] if len(r_f) > 1 else 0
    if np.isnan(cor): cor = 0
    cov_cp = cor * np.sqrt(var_cp_f * var_cp_r)
    sd_diff = np.sqrt(max(var_cp_f + var_cp_r - 2*cov_cp, 1e-10))
    kappa = abs(cp_f - cp_r) / max(sd_diff, 1e-10)
    delta = int((cp_f < cp_r) != (risk_f < risk_r))
    return dict(n=n, alpha=alpha, delta=delta, kappa=kappa)

N_REPS = 500
print("=== α sensitivity ===")
print(f"{'α':>6} {'n':>5} {'∆':>8} {'κ':>8} {'corr(κ,∆)':>10}")
for alpha in [0.05, 0.10, 0.20]:
    for n in [100, 500, 1000]:
        reps = Parallel(n_jobs=-1)(delayed(run_one)(n, alpha) for _ in range(N_REPS))
        reps = [r for r in reps if r is not None]
        df = pd.DataFrame(reps)
        delta = df.delta.mean()
        kappa = df.kappa.mean()
        corr = df.kappa.corr(df.delta)
        print(f"{alpha:>6.2f} {n:>5d} {delta:>8.4f} {kappa:>8.2f} {corr:>10.3f}")
