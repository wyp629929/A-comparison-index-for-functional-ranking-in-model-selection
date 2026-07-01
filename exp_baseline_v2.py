"""
Baseline comparison v2: κ vs CV-based distinguishability as ∆ predictors.
Hypothesis: κ correlates with ∆; CV gap (normalized) does not.
Design: OLS_full(p=20) vs OLS_restr(s=3), β has 5 non-zero coefs (misspecified restricted).
Sweep n to vary competition.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")
F0 = 2 * norm.pdf(norm.ppf(0.95))  # ≈ 0.206
np.random.seed(2042)

N_REPS = 1000

def run_one(n, p=20, s=3, n_sig_true=5):
    X = np.random.randn(n, p)
    beta = np.zeros(p)
    beta[:n_sig_true] = [1.0, 1.0, 1.0, 0.5, 0.3]
    y = X @ beta + np.random.randn(n)

    nc = max(int(n*0.25), 15); nt = nc; ntr = n - nc - nt
    if ntr < 10: return None
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt, random_state=None)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt), random_state=None)

    ols_f = LinearRegression().fit(Xtr, ytr)
    ols_r = LinearRegression().fit(Xtr[:, :s], ytr)

    # CP
    r_f = np.abs(yca - ols_f.predict(Xca))
    r_r = np.abs(yca - ols_r.predict(Xca[:, :s]))
    cp_f = 2 * np.quantile(r_f, 0.9)
    cp_r = 2 * np.quantile(r_r, 0.9)
    risk_f = mean_squared_error(yte, ols_f.predict(Xte))
    risk_r = mean_squared_error(yte, ols_r.predict(Xte[:, :s]))

    # κ
    raw_f = yca - ols_f.predict(Xca)
    sigma_f = np.std(raw_f, ddof=1)
    raw_r = yca - ols_r.predict(Xca[:, :s])
    sigma_r = np.std(raw_r, ddof=1)
    f_f = F0 / max(sigma_f, 1e-10)
    f_r = F0 / max(sigma_r, 1e-10)
    var_q_f = 0.9 * 0.1 / (max(nc, 1) * f_f**2)
    var_q_r = 0.9 * 0.1 / (max(nc, 1) * f_r**2)
    var_cp_f = 4 * var_q_f
    var_cp_r = 4 * var_q_r
    cor = np.corrcoef(r_f, r_r)[0, 1] if len(r_f) > 1 and len(r_r) > 1 else 0
    if np.isnan(cor): cor = 0
    cov_cp = cor * np.sqrt(var_cp_f * var_cp_r)
    sd_diff = np.sqrt(max(var_cp_f + var_cp_r - 2*cov_cp, 1e-10))
    kappa = abs(cp_f - cp_r) / max(sd_diff, 1e-10)

    # CV gap (normalized): how many CV SE apart are the two estimators?
    cv_f = cross_val_score(LinearRegression(), Xtr, ytr, cv=5, scoring='neg_mean_squared_error')
    cv_r = cross_val_score(LinearRegression(), Xtr[:, :s], ytr, cv=5, scoring='neg_mean_squared_error')
    cv_mean_f, cv_se_f = -cv_f.mean(), cv_f.std() / np.sqrt(5)
    cv_mean_r, cv_se_r = -cv_r.mean(), cv_r.std() / np.sqrt(5)
    cv_gap = abs(cv_mean_f - cv_mean_r) / max(np.sqrt(cv_se_f**2 + cv_se_r**2), 1e-10)

    delta = int((cp_f < cp_r) != (risk_f < risk_r))
    return dict(n=n, delta=delta, kappa=kappa, cv_gap=cv_gap)


print("=== Baseline v2: κ vs CV gap as ∆ predictors ===")
print(f"{'n':>5} {'∆':>8} {'κ_avg':>8} {'CV_gap':>8} {'corr(κ,∆)':>10} {'corr(CV,∆)':>10}")
print("-" * 55)

for n in [100, 200, 500, 1000]:
    reps = Parallel(n_jobs=-1)(delayed(run_one)(n) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    delta = df.delta.mean()
    kappa_avg = df.kappa.mean()
    cv_gap_avg = df.cv_gap.mean()
    corr_k = df.kappa.corr(df.delta)
    corr_cv = df.cv_gap.corr(df.delta)
    print(f"{n:>5d} {delta:>8.4f} {kappa_avg:>8.3f} {cv_gap_avg:>8.3f} {corr_k:>10.3f} {corr_cv:>10.3f}")

print("\nDone.")
