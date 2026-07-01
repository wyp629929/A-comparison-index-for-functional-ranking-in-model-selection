"""
Fast re-run of Tables 2 & 3 with corrected κ formula (n=300 as in paper).
Uses lighter RF config for speed.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")
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

def run_one_rf(estimators, p=50, n=300, rho=0.0):
    s = 3; B = max(p // 5, 1)
    Sigma = make_block_cov(p, B, rho)
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T
    beta = np.zeros(p); beta[:s] = 1.0
    y = X @ beta + np.random.randn(n)
    nc = max(int(n * 0.25), 15); nt = nc; ntr = n - nc - nt
    if ntr < 5: return None
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt, random_state=None)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt), random_state=None)
    fitted = {}
    for est in estimators:
        if est == "ols":
            m = LinearRegression().fit(Xtr, ytr)
        elif est == "rf":
            m = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=None)
            m.fit(Xtr, ytr)
        elif est == "ridge":
            m = Ridge(alpha=1.0).fit(Xtr, ytr)
        fitted[est] = m
    cp_vals, risk_vals = {}, {}
    for est in estimators:
        m = fitted[est]
        cp_vals[est] = 2 * np.quantile(np.abs(yca - m.predict(Xca)), 0.9)
        risk_vals[est] = mean_squared_error(yte, m.predict(Xte))
    cp_best = min(cp_vals, key=cp_vals.get)
    risk_best = min(risk_vals, key=risk_vals.get)
    delta = int(cp_best != risk_best)
    raw_vals = {}
    for est in estimators:
        m = fitted[est]
        raw_vals[est] = yca - m.predict(Xca)
    names = list(cp_vals.keys())
    var_cp = {}
    for est in names:
        sigma = np.std(raw_vals[est], ddof=1)
        f_est = F0 / max(sigma, 1e-10)
        var_q = 0.9 * 0.1 / (max(nc, 1) * f_est**2)
        var_cp[est] = 4 * var_q
    kappas = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            r_i = np.abs(raw_vals[names[i]])
            r_j = np.abs(raw_vals[names[j]])
            cor = np.corrcoef(r_i, r_j)[0, 1]
            if np.isnan(cor): cor = 0
            cov = cor * np.sqrt(var_cp[names[i]] * var_cp[names[j]])
            sd = np.sqrt(max(var_cp[names[i]] + var_cp[names[j]] - 2*cov, 1e-10))
            gap = abs(cp_vals[names[i]] - cp_vals[names[j]])
            kappas.append(gap / max(sd, 1e-10))
    kappa = min(kappas)
    return dict(delta=delta, kappa=kappa, cp_vals=cp_vals)

N_RF = 200  # fewer reps for RF (lighter)
N_RIDGE = 300

print("=== Table 2: Dependence modulation (OLS vs RF, n=300) ===")
print(f"{'ρ':>5} {'κ':>8} {'∆':>8} {'SE':>8}")
for rho in [0.00, 0.25, 0.50, 0.75]:
    reps = Parallel(n_jobs=-1)(delayed(run_one_rf)(["ols", "rf"], n=300, rho=rho) for _ in range(N_RF))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    r_ols = np.mean([r['cp_vals']['ols'] for r in reps])
    r_rf = np.mean([r['cp_vals']['rf'] for r in reps])
    delta = df.delta.mean(); se = np.sqrt(delta*(1-delta)/len(df))
    kappa = df.kappa.mean()
    print(f"{rho:>5.2f} {kappa:>8.2f} {delta:>8.3f} {se:>8.3f}  (R_OLS={r_ols:.2f}, R_RF={r_rf:.2f}, gap={abs(r_ols-r_rf):.2f})")

print()
print("=== Table 3: Symmetric control (OLS vs Ridge, n=300) ===")
for rho in [0.00, 0.25, 0.50, 0.75]:
    reps = Parallel(n_jobs=-1)(delayed(run_one_rf)(["ols", "ridge"], n=300, rho=rho) for _ in range(N_RIDGE))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    r_ols = np.mean([r['cp_vals']['ols'] for r in reps])
    r_ridge = np.mean([r['cp_vals']['ridge'] for r in reps])
    delta = df.delta.mean(); se = np.sqrt(delta*(1-delta)/len(df))
    kappa = df.kappa.mean()
    print(f"{rho:>5.2f} {kappa:>8.2f} {delta:>8.3f} {se:>8.3f}  (R_OLS={r_ols:.2f}, R_Ridge={r_ridge:.2f}, gap={abs(r_ols-r_ridge):.4f})")
