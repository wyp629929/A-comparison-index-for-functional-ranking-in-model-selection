"""
Experiment v2: ∆(n) decay with biased vs unbiased estimator.
Design: OLS_full(p=50) vs Ridge(alpha=10), sweep n from 50 to 2000.
Ridge has non-vanishing bias → unique population argmin → competition regime applies.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from scipy.stats import norm
F0 = 2 * norm.pdf(norm.ppf(0.95))  # ≈ 0.206
import warnings
warnings.filterwarnings("ignore")
np.random.seed(2036)

N_REPS = 500

def run_one(n, p=50, alpha=10):
    X = np.random.randn(n, p)
    beta = np.zeros(p); beta[:3] = 1.0
    y = X @ beta + np.random.randn(n)

    nc = max(int(n * 0.25), 10); nt = nc; ntr = n - nc - nt
    if ntr < 10: return None
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt, random_state=None)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt), random_state=None)

    ols = LinearRegression().fit(Xtr, ytr)
    ridge = Ridge(alpha=alpha).fit(Xtr, ytr)

    # CP widths
    r_ols = np.abs(yca - ols.predict(Xca))
    r_ridge = np.abs(yca - ridge.predict(Xca))
    cp_ols = 2 * np.quantile(r_ols, 0.9)
    cp_ridge = 2 * np.quantile(r_ridge, 0.9)

    # Risks
    risk_ols = mean_squared_error(yte, ols.predict(Xte))
    risk_ridge = mean_squared_error(yte, ridge.predict(Xte))

    # κ via Bahadur variance
    raw_ols = yca - ols.predict(Xca)
    sigma_ols = np.std(raw_ols, ddof=1)
    raw_ridge = yca - ridge.predict(Xca)
    sigma_ridge = np.std(raw_ridge, ddof=1)
    f_ols = F0 / max(sigma_ols, 1e-10)
    f_ridge = F0 / max(sigma_ridge, 1e-10)

    var_q_ols = 0.9 * 0.1 / (max(nc, 1) * f_ols**2)
    var_q_ridge = 0.9 * 0.1 / (max(nc, 1) * f_ridge**2)

    var_cp_ols = 4 * var_q_ols
    var_cp_ridge = 4 * var_q_ridge

    cor = np.corrcoef(r_ols, r_ridge)[0, 1] if len(r_ols) > 1 and len(r_ridge) > 1 else 0
    if np.isnan(cor): cor = 0
    cov_cp = cor * np.sqrt(var_cp_ols * var_cp_ridge)
    sd_diff = np.sqrt(max(var_cp_ols + var_cp_ridge - 2*cov_cp, 1e-10))
    kappa = abs(cp_ols - cp_ridge) / max(sd_diff, 1e-10)

    # Which estimator is selected by each criterion?
    cp_selects_ridge = int(cp_ridge < cp_ols)
    risk_selects_ridge = int(risk_ridge < risk_ols)

    return dict(n=n, delta=int(cp_selects_ridge != risk_selects_ridge),
                kappa=kappa, cp_gap=cp_ols-cp_ridge, risk_gap=risk_ols-risk_ridge,
                nc=nc, cp_ols=cp_ols, cp_ridge=cp_ridge,
                risk_ols=risk_ols, risk_ridge=risk_ridge)


print("=== Decay v2: OLS vs Ridge, ∆(n) ===")
print(f"{'n':>5} {'nc':>5} {'∆':>8} {'SE':>8} {'κ':>8} {'Φ(-√nc·κ)':>12}")
print("-" * 50)

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
        df_all.to_csv("/Users/wangyaoping/Desktop/paper3/experiment_results/decay_v2_all.csv", index=False)
print("\nSaved decay_v2_all.csv")
