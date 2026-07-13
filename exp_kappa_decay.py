"""
Experiment: ∆(n) decay matches κ/√n prediction.
Design: OLS_full(p=50) vs OLS_restricted(s=3), sweep n from 30 to 2000.
Show: ∆_n ≈ Φ(-√n · κ · √π_cal).
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from scipy.stats import norm
F0 = 2 * norm.pdf(norm.ppf(0.95))  # ≈ 0.206
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")
np.random.seed(2035)

N_REPS = 500

def make_block_cov(p, B, rho):
    m = p // B
    if m <= 1: return np.eye(p)
    Sigma = np.eye(p)
    for b in range(B):
        idx = slice(b*m, (b+1)*m)
        Sigma[idx, idx] = rho + (1-rho)*np.eye(m)
    return Sigma

def run_one(n, p=50, s=3):
    rho = 0.0
    B = max(p // 5, 1)
    Sigma = make_block_cov(p, B, rho)
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T
    beta = np.zeros(p); beta[:s] = 1.0
    y = X @ beta + np.random.randn(n)

    nc = max(int(n * 0.25), 10); nt = nc; ntr = n - nc - nt
    if ntr < 5: return None
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt, random_state=None)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt), random_state=None)

    ols_full = LinearRegression().fit(Xtr, ytr)
    ols_restr = LinearRegression().fit(Xtr[:, :s], ytr)

    # CP widths
    r_f = np.abs(yca - ols_full.predict(Xca))
    r_r = np.abs(yca - ols_restr.predict(Xca[:, :s]))
    cp_f = 2 * np.quantile(r_f, 0.9)
    cp_r = 2 * np.quantile(r_r, 0.9)

    # Risks
    risk_f = mean_squared_error(yte, ols_full.predict(Xte))
    risk_r = mean_squared_error(yte, ols_restr.predict(Xte[:, :s]))

    # κ (plug-in, parametric normal)
    # For half-normal |ε|: f(q) = 2·φ(q/σ)/σ at q = Q_0.9(|ε|) = σ·z_half, z_half=1.645
    # Actually the half-normal density at its α-quantile:
    # f(z_α) = 2·φ(z_α)/σ where z_α = Φ^{-1}((1+α)/2) = 1.645 for α=0.9
    # So f(q) = 2·φ(1.645)/σ = 0.516/σ

    raw_f = yca - ols_full.predict(Xca)
    sigma_f = np.std(raw_f, ddof=1)
    raw_r = yca - ols_restr.predict(Xca[:, :s])
    sigma_r = np.std(raw_r, ddof=1)
    f_f = F0 / max(sigma_f, 1e-10)
    f_r = F0 / max(sigma_r, 1e-10)

    # Bahadur variance: Var(hat_q) = α(1-α)/(n·f²)
    var_q_f = 0.9 * 0.1 / (max(nc, 1) * f_f**2)
    var_q_r = 0.9 * 0.1 / (max(nc, 1) * f_r**2)

    # CP width = 2*q, so Var(CP) = 4*Var(q)
    var_cp_f = 4 * var_q_f
    var_cp_r = 4 * var_q_r

    # Covariance from residual correlation
    cor = np.corrcoef(r_f, r_r)[0, 1] if len(r_f) > 1 and len(r_r) > 1 else 0
    if np.isnan(cor): cor = 0
    cov_cp = cor * np.sqrt(var_cp_f * var_cp_r)
    sd_diff = np.sqrt(max(var_cp_f + var_cp_r - 2*cov_cp, 1e-10))
    kappa = abs(cp_f - cp_r) / max(sd_diff, 1e-10)

    return dict(n=n, delta=int((cp_f<cp_r) != (risk_f<risk_r)),
                kappa=kappa, cp_gap=cp_f-cp_r, risk_gap=risk_f-risk_r,
                nc=nc)


# — Sweep n —
print("=== ∆(n) decay vs κ/√n prediction ===")
print(f"{'n':>5} {'nc':>5} {'∆':>8} {'SE':>8} {'κ':>8} {'Φ(-√n·κ·√π_cal)':>20}")
print("-" * 60)

for n in [30, 50, 100, 200, 500, 1000, 2000]:
    reps = Parallel(n_jobs=-1)(delayed(run_one)(n) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    delta = df.delta.mean()
    se = np.sqrt(delta*(1-delta)/len(df))
    kappa = df.kappa.mean()
    nc_mean = df.nc.mean()

    # Predicted ∆ using plug-in κ
    pred = norm.cdf(-np.sqrt(nc_mean) * kappa) if kappa > 0 else 0.5
    print(f"{n:>5} {int(nc_mean):>5} {delta:>8.3f} {se:>8.3f} {kappa:>8.3f} {pred:>20.3f}")

# — Central figure data: κ vs ∆ across ALL reps —
print("\n=== Central figure data (saved) ===")
all_data = []
for n in [30, 50, 100, 200, 500, 1000, 2000]:
    reps = Parallel(n_jobs=-1)(delayed(run_one)(n) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    for r in reps:
        r['n_group'] = n
        all_data.append(r)

        df_all = pd.DataFrame(all_data)
        df_all.to_csv("/Users/wangyaoping/Desktop/experiment_results/kappa_delta_all.csv", index=False)

# Aggregate by κ bins for plotting
        df_all['kappa_bin'] = pd.cut(df_all.kappa, bins=np.linspace(0, 5, 21))
        agg = df_all.groupby('kappa_bin').agg(
    delta_mean=('delta', 'mean'),
    kappa_mean=('kappa', 'mean'),
    count=('delta', 'count')
    ).reset_index()
print(f"{'κ_bin':>10} {'κ_mean':>8} {'∆':>8} {'count':>6}")
for _, row in agg.iterrows():
    print(f"{str(row.kappa_bin):>10} {row.kappa_mean:>8.3f} {row.delta_mean:>8.3f} {int(row['count']):>6}")
