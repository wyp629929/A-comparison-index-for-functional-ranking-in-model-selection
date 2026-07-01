"""
Decay experiment v4: Clean test of κ-∆ relationship.
Uses population κ (computed from large MC) for validation.
Design: OLS_full(p=20) vs OLS_restr(s=3), true β has 5 non-zero coefficients.
Restricted is misspecified → non-vanishing gap → unique argmin.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")
np.random.seed(2041)

N_REPS = 500
F0 = 2 * norm.pdf(norm.ppf(0.95))  # ≈ 0.206

def compute_population_kappa(n, n_mc=20000):
    """Compute κ from population values (large MC)."""
    p, s, n_sig_true = 20, 3, 5
    kappas = []
    for _ in range(100):
        X_big = np.random.randn(n_mc, p)
        beta = np.zeros(p)
        beta[:n_sig_true] = [1.0, 1.0, 1.0, 0.5, 0.3]
        y_big = X_big @ beta + np.random.randn(n_mc)

        nc_big = max(int(n_mc * 0.25), 50)
        Xtr_big = X_big[:-nc_big]
        Xca_big = X_big[-nc_big:]
        ytr_big = y_big[:-nc_big]
        yca_big = y_big[-nc_big:]

        ols_f = LinearRegression().fit(Xtr_big, ytr_big)
        ols_r = LinearRegression().fit(Xtr_big[:, :s], ytr_big)

        r_f = np.abs(yca_big - ols_f.predict(Xca_big))
        r_r = np.abs(yca_big - ols_r.predict(Xca_big[:, :s]))

        raw_f = yca_big - ols_f.predict(Xca_big)
        raw_r = yca_big - ols_r.predict(Xca_big[:, :s])
        sigma_f = np.std(raw_f, ddof=1)
        sigma_r = np.std(raw_r, ddof=1)
        f_f = F0 / max(sigma_f, 1e-10)
        f_r = F0 / max(sigma_r, 1e-10)

        nc_eff = max(int(n * 0.25), 15)
        var_q_f = 0.9 * 0.1 / (nc_eff * f_f**2)
        var_q_r = 0.9 * 0.1 / (nc_eff * f_r**2)
        var_cp_f = 4 * var_q_f
        var_cp_r = 4 * var_q_r

        cor = np.corrcoef(r_f, r_r)[0, 1]
        if np.isnan(cor): cor = 0
        cov_cp = cor * np.sqrt(var_cp_f * var_cp_r)
        sd_diff = np.sqrt(max(var_cp_f + var_cp_r - 2*cov_cp, 1e-10))

        cp_f = 2 * np.quantile(r_f, 0.9)
        cp_r = 2 * np.quantile(r_r, 0.9)
        kappas.append(abs(cp_f - cp_r) / max(sd_diff, 1e-10))

    return np.median(kappas)


def run_one(n, p=20, s=3, n_sig_true=5):
    X = np.random.randn(n, p)
    beta = np.zeros(p)
    beta[:n_sig_true] = [1.0, 1.0, 1.0, 0.5, 0.3]
    y = X @ beta + np.random.randn(n)

    nc = max(int(n * 0.25), 15); nt = nc; ntr = n - nc - nt
    if ntr < 10: return None
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

    # Plug-in κ (Bahadur variance, normal plug-in density)
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


# Pre-compute population κ for each n
print("=== Decay v4: Misspecified restricted, ∆(n) ===")
print("Computing population κ values...")
ns = [50, 100, 200, 500, 1000]
pop_kappas = {}
for n in ns:
    pk = compute_population_kappa(n)
    pop_kappas[n] = pk
    print(f"  n={n:5d}  population κ = {pk:.3f}")

print(f"\n{'n':>5} {'nc':>5} {'∆':>8} {'SE':>8} {'κ_plug':>8} {'κ_pop':>8} {'Φ(-√nc·κ_pop)':>16}")
print("-" * 65)

# Separate the data generation into two phases to avoid confusion with the function
def generate_reps(n):
    return [run_one(n) for _ in range(N_REPS)]

for n in ns:
    reps = generate_reps(n)
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    delta = df.delta.mean()
    se = np.sqrt(delta*(1-delta)/len(df))
    kappa_plug = df.kappa.mean()
    kappa_pop = pop_kappas[n]
    nc_mean = df.nc.mean()
    pred = norm.cdf(-np.sqrt(nc_mean) * kappa_pop) if kappa_pop > 0 else 0.5
    print(f"{n:>5} {int(nc_mean):>5} {delta:>8.4f} {se:>8.4f} {kappa_plug:>8.3f} {kappa_pop:>8.3f} {pred:>16.4f}")
