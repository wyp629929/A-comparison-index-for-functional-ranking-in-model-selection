"""
3-channel experiment v2: Fix C1 (bias crossing) and C3 (dependence).
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")
np.random.seed(2033)

N_REPS = 500

def make_block_cov(p, B, rho):
    m = p // B
    if m <= 1: return np.eye(p)
    Sigma = np.eye(p)
    for b in range(B):
        idx = slice(b*m, (b+1)*m)
        Sigma[idx, idx] = rho + (1-rho)*np.eye(m)
    return Sigma

def run_one(ests, dgp, p=20, n=100, rho=0.0, n_sig=3):
    if "restr" in ests: s = n_sig
    else: s = n_sig
    B = max(p // 5, 1)
    Sigma = make_block_cov(p, B, rho)
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T
    if dgp == "linear":
        beta = np.zeros(p); beta[:n_sig] = 1.0
        y = X @ beta + np.random.randn(n)
    elif dgp == "threshold":
        y = 2.0*(X[:,0]>0) + np.log(1+np.abs(X[:,1]))*(X[:,2]>0) - 1 + np.random.randn(n)
    nc = max(int(n*0.25), 10); nt = nc; ntr = n - nc - nt
    if ntr < 5: return None
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt, random_state=None)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt), random_state=None)

    fitted = {}
    for e in ests:
        if e == "ols_f": fitted[e] = LinearRegression().fit(Xtr, ytr)
        elif e == "ols_r": fitted[e] = LinearRegression().fit(Xtr[:, :n_sig], ytr)
        elif e == "ridge": fitted[e] = Ridge(alpha=1.0).fit(Xtr, ytr)
        elif e == "rf": fitted[e] = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=None).fit(Xtr, ytr)
    cp, risk = {}, {}
    for e in ests:
        Xc = Xca[:, :n_sig] if e == "ols_r" else Xca
        Xt = Xte[:, :n_sig] if e == "ols_r" else Xte
        cp[e] = 2 * np.quantile(np.abs(yca - fitted[e].predict(Xc)), 0.9)
        risk[e] = mean_squared_error(yte, fitted[e].predict(Xt))

    return dict(ests="+".join(ests), dgp=dgp, rho=rho, n=n, p=p,
                delta=int(min(cp,key=cp.get)!=min(risk,key=risk.get)),
                **{f"cp_{k}":cp[k] for k in cp}, **{f"risk_{k}":risk[k] for k in risk})

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# C1: Bias crossing — OLS vs RF on threshold DGP (need competition, so add Ridge)
# Use low n to keep competition alive
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
print("=== C1: Bias crossing (OLS+Ridge+RF) ===")
for dgp in ["linear", "threshold"]:
    reps = Parallel(n_jobs=-1)(delayed(run_one)(["ols_f", "ridge", "rf"], dgp, n=80, p=10) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    delta = df.delta.mean(); se = np.sqrt(delta*(1-delta)/len(df))
    print(f"  {dgp:<10} ∆={delta:.3f} SE={se:.3f}")

# Difference = bias crossing contribution
sub_lin = pd.DataFrame(Parallel(n_jobs=-1)(delayed(run_one)(["ols_f", "ridge", "rf"], "linear", n=80, p=10) for _ in range(N_REPS)))
sub_thr = pd.DataFrame(Parallel(n_jobs=-1)(delayed(run_one)(["ols_f", "ridge", "rf"], "threshold", n=80, p=10) for _ in range(N_REPS)))
print(f"  C1 = ∆_threshold − ∆_linear = {sub_thr.delta.mean() - sub_lin.delta.mean():.3f}")

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# C2: Competition — OLS_full vs OLS_restricted, p sweep
# Using n=200
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
print("\n=== C2: Competition (OLS_full vs OLS_restr) ===")
for p in [10, 20, 50, 100]:
    reps = Parallel(n_jobs=-1)(delayed(run_one)(["ols_f", "ols_r"], "linear", n=200, p=p) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    print(f"  p={p:>3d} ∆={df.delta.mean():.3f} SE={np.sqrt(df.delta.mean()*(1-df.delta.mean())/len(df)):.3f}")

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# C3: Dependence modulation — OLS vs RF, ρ sweep at low n
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
print("\n=== C3: Dependence (OLS+RF, n=60, p=20) ===")
for rho in [0.0, 0.25, 0.50, 0.75]:
    reps = Parallel(n_jobs=-1)(delayed(run_one)(["ols_f", "rf"], "linear", n=60, p=20, rho=rho) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    delta = df.delta.mean(); se = np.sqrt(delta*(1-delta)/len(df))
    cp_gap = (df.cp_rf - df.cp_ols_f).mean()
    print(f"  ρ={rho:.2f} ∆={delta:.3f} SE={se:.3f} CP_gap(OLS-RF)={cp_gap:.3f}")
