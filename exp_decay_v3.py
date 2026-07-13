"""
Experiment v3: ∆(n) decay with misspecified restricted model.
Design: True β has 5 non-zero coefs. OLS_full(all) vs OLS_restr(first 3).
Restricted is misspecified → non-vanishing gap → unique population argmin.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")
np.random.seed(2040)

N_REPS = 500
# Constant for Bahadur variance: 2 * φ(z_0.95) ≈ 0.206
F0 = 2 * norm.pdf(norm.ppf(0.95))

def run_one(n, p=20, s=3, n_sig_true=5):
    """s=3 features used by restricted, true model has n_sig_true=5 non-zero."""
    X = np.random.randn(n, p)
    beta = np.zeros(p)
    beta[:n_sig_true] = [1.0, 1.0, 1.0, 0.5, 0.3]  # 5 non-zero coefficients
    y = X @ beta + np.random.randn(n)

    nc = max(int(n * 0.25), 15); nt = nc; ntr = n - nc - nt
    if ntr < 10: return None
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

    # κ via Bahadur variance (normal plug-in: f(q) = 2·φ(z_0.95)/σ_raw)
    raw_f = yca - ols_full.predict(Xca)
    raw_r = yca - ols_restr.predict(Xca[:, :s])
    sigma_f = np.std(raw_f, ddof=1)
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

    delta = int((cp_f < cp_r) != (risk_f < risk_r))
    return dict(n=n, delta=delta, kappa=kappa, cp_gap=cp_f-cp_r, risk_gap=risk_f-risk_r, nc=nc)


print("=== Decay v3: misspecified restricted, ∆(n) ===")
print(f"{'n':>5} {'nc':>5} {'∆':>8} {'SE':>8} {'κ':>8} {'Φ(-√nc·κ)':>12}")
print("-" * 55)

all_data = []
for n in [50, 100, 200, 500, 1000, 2000]:
    reps = Parallel(n_jobs=-1)(delayed(run_one)(n) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    delta = df.delta.mean()
    se = np.sqrt(delta*(1-delta)/len(df))
    kappa = df.kappa.mean()
    nc_mean = df.nc.mean()
    pred = norm.cdf(-np.sqrt(nc_mean) * kappa) if kappa > 0 else 0.5
    print(f"{n:>5} {int(nc_mean):>5} {delta:>8.4f} {se:>8.4f} {kappa:>8.3f} {pred:>12.4f}")
    for _, r in df.iterrows():
        all_data.append(r.to_dict())

df_all = pd.DataFrame(all_data)
df_all.to_csv("/Users/wangyaoping/Desktop/paper3/experiment_results/decay_v3_all.csv", index=False)
print("\nSaved decay_v3_all.csv")
