"""
Re-run competition channel (Table 1) and check Table 2 bias-crossing diagnostic.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from scipy.stats import norm
import warnings; warnings.filterwarnings("ignore")
np.random.seed(2031)
F0 = 2 * norm.pdf(norm.ppf(0.95))

def make_block_cov(p, B, rho):
    m = p // B
    if m <= 1: return np.eye(p)
    Sigma = np.eye(p)
    for b in range(B):
        idx = slice(b*m, (b+1)*m)
        Sigma[idx, idx] = rho + (1-rho)*np.eye(m)
    return Sigma

def run_one_comp(p, n=500):
    """Competition channel: ols vs ols_restr, correctly specified."""
    s = 3; beta = np.zeros(p); beta[:s] = 1.0
    X = np.random.randn(n, p)
    y = X @ beta + np.random.randn(n)
    nc = max(int(n * 0.25), 15); nt = nc; ntr = n - nc - nt
    if ntr < 5: return None
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt, random_state=None)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt), random_state=None)

    ols_full = LinearRegression().fit(Xtr, ytr)
    ols_restr = LinearRegression().fit(Xtr[:, :s], ytr)

    r_f = np.abs(yca - ols_full.predict(Xca))
    r_r = np.abs(yca - ols_restr.predict(Xca[:, :s]))
    cp_f = 2 * np.quantile(r_f, 0.9)
    cp_r = 2 * np.quantile(r_r, 0.9)
    risk_f = mean_squared_error(yte, ols_full.predict(Xte))
    risk_r = mean_squared_error(yte, ols_restr.predict(Xte[:, :s]))

    delta = int((cp_f < cp_r) != (risk_f < risk_r))

    # κ via Bahadur
    raw_f = yca - ols_full.predict(Xca)
    raw_r = yca - ols_restr.predict(Xca[:, :s])
    sigma_f = np.std(raw_f, ddof=1); sigma_r = np.std(raw_r, ddof=1)
    f_f = F0 / max(sigma_f, 1e-10); f_r = F0 / max(sigma_r, 1e-10)
    var_q_f = 0.09 / (max(nc, 1) * f_f**2); var_q_r = 0.09 / (max(nc, 1) * f_r**2)
    var_cp_f = 4 * var_q_f; var_cp_r = 4 * var_q_r
    cor = np.corrcoef(r_f, r_r)[0, 1]
    if np.isnan(cor): cor = 0
    cov_cp = cor * np.sqrt(var_cp_f * var_cp_r)
    sd_diff = np.sqrt(max(var_cp_f + var_cp_r - 2*cov_cp, 1e-10))
    kappa = abs(cp_f - cp_r) / max(sd_diff, 1e-10)
    return dict(delta=delta, kappa=kappa, cp_f=cp_f, cp_r=cp_r, risk_f=risk_f, risk_r=risk_r)

N_REPS = 500

print("=== Table 1: Competition channel (correctly specified) ===")
print(f"{'p':>5} {'∆':>8} {'SE':>8} {'κ':>8}")
for p in [10, 20, 50, 100]:
    reps = Parallel(n_jobs=-1)(delayed(run_one_comp)(p) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    delta = df.delta.mean(); se = np.sqrt(delta*(1-delta)/len(df))
    kappa = df.kappa.mean()
    # Wilson CI
    z = 1.96; d, nn = delta, len(df)
    denom = 1 + z**2/nn
    ci_low = (d + z**2/(2*nn) - z * np.sqrt(d*(1-d)/nn + z**2/(4*nn**2))) / denom
    ci_high = (d + z**2/(2*nn) + z * np.sqrt(d*(1-d)/nn + z**2/(4*nn**2))) / denom
    print(f"{p:>5d} {delta:>8.3f} {se:>8.4f} {kappa:>8.2f}  CI=[{ci_low:.3f}, {ci_high:.3f}]")

print()
print("=== Table 2 bias-crossing check: CP vs Risk agreement at ρ=0 ===")
from sklearn.ensemble import RandomForestRegressor

def run_one_rf_check(n=300):
    p = 50; s = 3; B = max(p // 5, 1)
    Sigma = make_block_cov(p, B, 0.0)
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T
    beta = np.zeros(p); beta[:s] = 1.0
    y = X @ beta + np.random.randn(n)
    nc = max(int(n * 0.25), 15); nt = nc; ntr = n - nc - nt
    if ntr < 5: return None
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt, random_state=None)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt), random_state=None)
    ols = LinearRegression().fit(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=None)
    rf.fit(Xtr, ytr)
    cp_ols = 2 * np.quantile(np.abs(yca - ols.predict(Xca)), 0.9)
    cp_rf = 2 * np.quantile(np.abs(yca - rf.predict(Xca)), 0.9)
    risk_ols = mean_squared_error(yte, ols.predict(Xte))
    risk_rf = mean_squared_error(yte, rf.predict(Xte))
    cp_best = 'ols' if cp_ols < cp_rf else 'rf'
    risk_best = 'ols' if risk_ols < risk_rf else 'rf'
    delta = int(cp_best != risk_best)
    return dict(delta=delta, cp_best=cp_best, risk_best=risk_best)

reps = Parallel(n_jobs=-1)(delayed(run_one_rf_check)() for _ in range(400))
reps = [r for r in reps if r is not None]
df = pd.DataFrame(reps)
cp_likes = df.cp_best.value_counts()
risk_likes = df.risk_best.value_counts()
agree = (df.cp_best == df.risk_best).mean()
print(f"CP prefers OLS: {cp_likes.get('ols', 0)} reps ({cp_likes.get('ols', 0)/len(df)*100:.1f}%)")
print(f"CP prefers RF:  {cp_likes.get('rf', 0)} reps ({cp_likes.get('rf', 0)/len(df)*100:.1f}%)")
print(f"Risk prefers OLS: {risk_likes.get('ols', 0)} reps ({risk_likes.get('ols', 0)/len(df)*100:.1f}%)")
print(f"Risk prefers RF:  {risk_likes.get('rf', 0)} reps ({risk_likes.get('rf', 0)/len(df)*100:.1f}%)")
print(f"CP & Risk agree: {agree*100:.1f}%")
print(f"∆ = {df.delta.mean():.3f}")
