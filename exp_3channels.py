"""
Three-channel experiment: Isolate bias crossing, competition, dependence.

Estimator sets:
  - S1 = {OLS_restricted} — low variance, well-specified (baseline, no competition)
  - S2 = {OLS_full} — high variance, well-specified
  - S3 = {OLS_full, OLS_restr} — pure competition
  - S4 = {OLS, RF} on threshold DGP — bias crossing
  - S5 = {OLS, RF} on linear DGP + ρ variation — dependence modulation

Channels:
  C1 (Bias crossing): compare ∆(S4, threshold) vs ∆(S4, linear) at ρ=0
  C2 (Competition): ∆(S3, linear) varies with p
  C3 (Dependence): ∆(S4, linear, ρ=0.75) - ∆(S4, linear, ρ=0)
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
F0 = 2 * norm.pdf(norm.ppf(0.95))  # ≈ 0.206

N_REPS = 400


def make_block_cov(p, B, rho):
    m = p // B
    if m <= 1: return np.eye(p)
    Sigma = np.eye(p)
    for b in range(B):
        idx = slice(b*m, (b+1)*m)
        Sigma[idx, idx] = rho + (1-rho)*np.eye(m)
    return Sigma


def run_one(estimators, dgp, p=50, n=200, rho=0.0):
    """
    estimators: list of names: "ols", "ols_restr", "rf"
    dgp: "linear" or "threshold"
    """
    s = 3
    B = max(p // 5, 1)
    Sigma = make_block_cov(p, B, rho)
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T

    if dgp == "linear":
        beta = np.zeros(p); beta[:s] = 1.0
        y = X @ beta + np.random.randn(n)
    elif dgp == "threshold":
        y = 2.0 * (X[:, 0] > 0) + np.log(1 + np.abs(X[:, 1])) * (X[:, 2] > 0) - 1
        y = y + np.random.randn(n)

    nc = max(int(n * 0.25), 15)
    nt = nc
    ntr = n - nc - nt
    if ntr < 5: return None

    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt, random_state=None)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt), random_state=None)

    # Fit estimators
    fitted = {}
    for est in estimators:
        if est == "ols":
            m = LinearRegression().fit(Xtr, ytr)
        elif est == "ols_restr":
            m = LinearRegression().fit(Xtr[:, :s], ytr)
        elif est == "rf":
            m = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=None)
            m.fit(Xtr, ytr)
        elif est == "ridge":
            m = Ridge(alpha=1.0).fit(Xtr, ytr)
        fitted[est] = m

    cp_vals, risk_vals = {}, {}
    for est in estimators:
        m = fitted[est]
        if est == "ols_restr":
            Xc = Xca[:, :s]; Xt = Xte[:, :s]
        else:
            Xc = Xca; Xt = Xte
        cp_vals[est] = 2 * np.quantile(np.abs(yca - m.predict(Xc)), 0.9)
        risk_vals[est] = mean_squared_error(yte, m.predict(Xt))

    # ∆ for this set
    cp_best = min(cp_vals, key=cp_vals.get)
    risk_best = min(risk_vals, key=risk_vals.get)
    delta = int(cp_best != risk_best)

    # κ via Bahadur variance (corrected formula)
    raw_vals = {}
    for est in estimators:
        m = fitted[est]
        if est == "ols_restr":
            Xc = Xca[:, :s]
        else:
            Xc = Xca
        raw_vals[est] = yca - m.predict(Xc)

    names = list(cp_vals.keys())
    if len(names) >= 2:
        # Compute variance of each CP width
        var_cp = {}
        for est in names:
            sigma = np.std(raw_vals[est], ddof=1)
            f_est = F0 / max(sigma, 1e-10)
            var_q = 0.9 * 0.1 / (max(nc, 1) * f_est**2)
            var_cp[est] = 4 * var_q

        kappas = []
        # Correlation between absolute residuals of each pair
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                r_i = np.abs(raw_vals[names[i]])
                r_j = np.abs(raw_vals[names[j]])
                cor = np.corrcoef(r_i, r_j)[0, 1] if len(r_i) > 1 else 0
                if np.isnan(cor): cor = 0
                cov = cor * np.sqrt(var_cp[names[i]] * var_cp[names[j]])
                sd = np.sqrt(max(var_cp[names[i]] + var_cp[names[j]] - 2*cov, 1e-10))
                gap = abs(cp_vals[names[i]] - cp_vals[names[j]])
                kappas.append(gap / max(sd, 1e-10))
        kappa = min(kappas)
    else:
        kappa = float('inf')

    return dict(dgp=dgp, est_set="+".join(estimators), rho=rho, delta=delta, kappa=kappa, cp_vals=cp_vals)


# — Settings —
settings = [
    # Pure competition channel (S3): vary p
    ("linear", ["ols", "ols_restr"], 0.0),
    # Bias crossing (S4 at threshold)
    ("threshold", ["ols", "rf"], 0.0),
    # Same estimators on linear (no bias crossing)
    ("linear", ["ols", "rf"], 0.0),
    # Dependence modulation: OLS vs RF at ρ=0.75
    ("linear", ["ols", "rf"], 0.75),
]

print("=== Three Channels ===")
print(f"{'DGP':<12} {'EstSet':<20} {'ρ':>5} {'∆':>8} {'SE':>8} {'κ':>8}")
print("-" * 65)

for dgp, ests, rho in settings:
    info = "+".join(ests)
    reps = Parallel(n_jobs=-1)(delayed(run_one)(ests, dgp, rho=rho) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    delta = df.delta.mean()
    se = np.sqrt(delta*(1-delta)/len(df))
    kappa = df.kappa.mean()
    print(f"{dgp:<12} {info:<20} {rho:>5.2f} {delta:>8.3f} {se:>8.3f} {kappa:>8.2f}")

# Channel 1 (Bias crossing): ∆(threshold, OLS+RF) - ∆(linear, OLS+RF) at ρ=0
print("\n=== Channel Decomposition ===")
sub_lin = pd.DataFrame(Parallel(n_jobs=-1)(delayed(run_one)(["ols", "rf"], "linear", rho=0.0) for _ in range(N_REPS)))
sub_thr = pd.DataFrame(Parallel(n_jobs=-1)(delayed(run_one)(["ols", "rf"], "threshold", rho=0.0) for _ in range(N_REPS)))
C1 = sub_thr.delta.mean() - sub_lin.delta.mean()
print(f"C1 (Bias crossing): ∆_threshold(OLS+RF) − ∆_linear(OLS+RF) = {C1:.3f}")

# Channel 2 (Competition): vary p
for p_val in [10, 50, 100]:
    reps = Parallel(n_jobs=-1)(delayed(run_one)(["ols", "ols_restr"], "linear", p=p_val) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    print(f"C2 (Competition at p={p_val}): ∆={df.delta.mean():.3f}, κ={df.kappa.mean():.2f}")

# Channel 3 (Dependence): OLS+RF, ρ=0 vs ρ=0.75
sub_rho0 = pd.DataFrame(Parallel(n_jobs=-1)(delayed(run_one)(["ols", "rf"], "linear", rho=0.0) for _ in range(N_REPS)))
sub_rho75 = pd.DataFrame(Parallel(n_jobs=-1)(delayed(run_one)(["ols", "rf"], "linear", rho=0.75) for _ in range(N_REPS)))
C3 = sub_rho75.delta.mean() - sub_rho0.delta.mean()
print(f"C3 (Dependence): ∆_ρ=0.75 − ∆_ρ=0 = {C3:.3f}")
