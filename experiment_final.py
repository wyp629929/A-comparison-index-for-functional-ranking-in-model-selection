"""
Final experiment: Two-estimator controlled competition.
Clean story with κ diagnostic + interventions.

Design: OLS_full (p=50) vs OLS_restricted (s=3 true features)
- Both unbiased, gap = (p-s)/n * σ²
- Vary n → control κ
- Vary ρ → modulate κ through block correlation
- Interventions: negative ρ, p variation
"""

import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")
np.random.seed(2029)


def make_block_cov(p, B, rho):
    m = p // B
    if m <= 1: return np.eye(p)
    Sigma = np.eye(p)
    for b in range(B):
        idx = slice(b*m, (b+1)*m)
        Sigma[idx, idx] = rho + (1-rho)*np.eye(m)
    return Sigma


def run_one(rho, p, n):
    s = 3
    beta = np.zeros(p); beta[:s] = 1.0
    B = max(p // 5, 1)
    Sigma = make_block_cov(p, B, rho)
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T
    y = X @ beta + np.random.randn(n)

    nc = max(int(n * 0.25), 15)
    nt = nc; ntr = n - nc - nt
    if ntr < 5 or nc < 5 or nt < 5: return None

    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt, random_state=None)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt), random_state=None)

    ols_full = LinearRegression().fit(Xtr, ytr)
    ols_restr = LinearRegression().fit(Xtr[:, :s], ytr)

    cp_f = 2 * np.quantile(np.abs(yca - ols_full.predict(Xca)), 0.9)
    cp_r = 2 * np.quantile(np.abs(yca - ols_restr.predict(Xca[:, :s])), 0.9)
    risk_f = mean_squared_error(yte, ols_full.predict(Xte))
    risk_r = mean_squared_error(yte, ols_restr.predict(Xte[:, :s]))

    return dict(rho=rho, p=p, n=n,
                cp_f=cp_f, cp_r=cp_r, risk_f=risk_f, risk_r=risk_r)


def analyze(df):
    """Compute ∆ and κ from a DataFrame."""
    delta = ((df.cp_f < df.cp_r) != (df.risk_f < df.risk_r)).mean()
    diff = df.cp_f - df.cp_r
    gap = abs(diff.mean())
    sd = diff.std()
    kappa = gap / sd if sd > 1e-10 else 0.0
    cp_says_full = (df.cp_f < df.cp_r).mean()
    risk_says_full = (df.risk_f < df.risk_r).mean()
    return delta, kappa, cp_says_full, risk_says_full


N_REPS = 500

# === Experiment 1: n × ρ (core) ===
print("=== E1: n × ρ (p=50) ===")
print(f"{'n':>5} {'ρ':>6} {'∆':>8} {'SE':>8} {'κ':>8} {'CP_full%':>8} {'Risk_full%':>9}")
print("-" * 60)
for n in [40, 60, 100, 200, 500, 1000]:
    for rho in [0.0, 0.5, 0.75]:
        reps = Parallel(n_jobs=-1)(delayed(run_one)(rho, 50, n) for _ in range(N_REPS))
        reps = [r for r in reps if r is not None]
        df = pd.DataFrame(reps)
        delta, kappa, cpf, riskf = analyze(df)
        se = np.sqrt(delta*(1-delta)/len(df))
        print(f"{n:>5} {rho:>6.2f} {delta:>8.3f} {se:>8.3f} {kappa:>8.3f} {cpf:>8.3f} {riskf:>9.3f}")

# === Experiment 2: Negative ρ ===
print("\n=== E2: Negative ρ (p=50, n=60) ===")
for rho in [-0.15, -0.10, -0.05, 0.0, 0.25, 0.50, 0.75]:
    reps = Parallel(n_jobs=-1)(delayed(run_one)(rho, 50, 60) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    delta, kappa, cpf, riskf = analyze(df)
    se = np.sqrt(delta*(1-delta)/len(df))
    print(f"{rho:>6.2f} {delta:>8.3f} {se:>8.3f} {kappa:>8.3f} {cpf:>8.3f} {riskf:>9.3f}")

# === Experiment 3: p effect ===
print("\n=== E3: p effect (ρ=0.25, n=100) ===")
for p in [5, 10, 20, 50, 100]:
    reps = Parallel(n_jobs=-1)(delayed(run_one)(0.25, p, 100) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    delta, kappa, cpf, riskf = analyze(df)
    se = np.sqrt(delta*(1-delta)/len(df))
    print(f"{p:>5d} {delta:>8.3f} {se:>8.3f} {kappa:>8.3f} {cpf:>8.3f} {riskf:>9.3f}")
