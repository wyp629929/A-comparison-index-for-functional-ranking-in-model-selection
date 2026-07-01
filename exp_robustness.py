"""
Robustness analysis: κ̂ under non-normal residuals.
Compares normal plug-in vs KDE-based κ estimation for t5 (heavy-tailed)
and skewed residuals. Identifies regimes where plug-in κ̂ is reliable vs biased.

Design: same as exp_decay_v3 — OLS_full(p=20) vs OLS_restr(s=3),
true β has 5 non-zero coefs (misspecified restricted).
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from scipy.stats import norm, gaussian_kde
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(2040)

N_REPS = 500
N_VALS = [50, 100, 200, 500, 1000]

def run_one(n, p=20, s=3, n_sig_true=5, dist='normal'):
    X = np.random.randn(n, p)
    beta = np.zeros(p)
    beta[:n_sig_true] = [1.0, 1.0, 1.0, 0.5, 0.3]

    # Residual distributions with variance 1
    if dist == 'normal':
        eps = np.random.randn(n)
    elif dist == 't5':
        eps = np.random.standard_t(5, n) / np.sqrt(5/3)
    elif dist == 'skewed':
        eps = (np.random.gamma(2, 1, n) - 2) / np.sqrt(2)
    else:
        raise ValueError(f"Unknown dist: {dist}")

    y = X @ beta + eps

    nc = max(int(n * 0.25), 15); nt = nc; ntr = n - nc - nt
    if ntr < 10: return None
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt, random_state=None)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt), random_state=None)

    ols_full = LinearRegression().fit(Xtr, ytr)
    ols_restr = LinearRegression().fit(Xtr[:, :s], ytr)

    r_f = np.abs(yca - ols_full.predict(Xca))
    r_r = np.abs(yca - ols_restr.predict(Xca[:, :s]))
    q_f = np.quantile(r_f, 0.9)
    q_r = np.quantile(r_r, 0.9)
    cp_f = 2 * q_f
    cp_r = 2 * q_r

    risk_f = mean_squared_error(yte, ols_full.predict(Xte))
    risk_r = mean_squared_error(yte, ols_restr.predict(Xte[:, :s]))

    # === Normal plug-in κ (Bahadur variance, normal density at q_0.9) ===
    # f(q) = 2·φ(z_0.95) / σ_raw for absolute residuals of normal ε
    raw_f = yca - ols_full.predict(Xca)
    raw_r = yca - ols_restr.predict(Xca[:, :s])
    sigma_f = np.std(raw_f, ddof=1)
    sigma_r = np.std(raw_r, ddof=1)
    F0 = 2 * norm.pdf(norm.ppf(0.95))
    f_f = F0 / max(sigma_f, 1e-10)
    f_r = F0 / max(sigma_r, 1e-10)

    var_q_f = 0.9 * 0.1 / (max(nc, 1) * f_f**2)
    var_q_r = 0.9 * 0.1 / (max(nc, 1) * f_r**2)
    var_cp_f = 4 * var_q_f
    var_cp_r = 4 * var_q_r

    cor = np.corrcoef(r_f, r_r)[0, 1] if len(r_f) > 1 and len(r_r) > 1 else 0
    if np.isnan(cor): cor = 0
    cov_cp = cor * sqrt(var_cp_f * var_cp_r)
    sd_diff = sqrt(max(var_cp_f + var_cp_r - 2*cov_cp, 1e-10))
    kappa = abs(cp_f - cp_r) / max(sd_diff, 1e-10)

    # === KDE-corrected κ (nonparametric density at q_0.9) ===
    try:
        kde_f = gaussian_kde(r_f)
        kde_r = gaussian_kde(r_r)
        f_kde_f = kde_f.evaluate([q_f])[0]
        f_kde_r = kde_r.evaluate([q_r])[0]
        if f_kde_f > 1e-10 and f_kde_r > 1e-10:
            var_q_kde_f = 0.9 * 0.1 / (max(nc, 1) * f_kde_f**2)
            var_q_kde_r = 0.9 * 0.1 / (max(nc, 1) * f_kde_r**2)
            var_cp_kde_f = 4 * var_q_kde_f
            var_cp_kde_r = 4 * var_q_kde_r
            cov_cp_kde = cor * np.sqrt(var_cp_kde_f * var_cp_kde_r)
            sd_diff_kde = np.sqrt(max(var_cp_kde_f + var_cp_kde_r - 2*cov_cp_kde, 1e-10))
            kappa_kde = abs(cp_f - cp_r) / max(sd_diff_kde, 1e-10)
        else:
            kappa_kde = np.nan
    except Exception:
        kappa_kde = np.nan

    delta = int((cp_f < cp_r) != (risk_f < risk_r))
    return dict(n=n, delta=delta, kappa=kappa, kappa_kde=kappa_kde, dist=dist, nc=nc)


def run_dist(dist):
    print(f"\n=== dist={dist} ===")
    print(f"{'n':>5} {'nc':>5} {'∆':>8} {'SE':>8} {'κ̂':>8} {'κ̂_KDE':>8} {'corr(κ,∆)':>10} {'corr(KDE,∆)':>10} {'bias%':>8}")
    print("-" * 75)

    rows = []
    for n in N_VALS:
        reps = Parallel(n_jobs=-1)(delayed(run_one)(n, dist=dist) for _ in range(N_REPS))
        reps = [r for r in reps if r is not None]
        df = pd.DataFrame(reps)
        delta = df.delta.mean()
        se = np.sqrt(delta * (1 - delta) / len(df))
        kappa_mu = df.kappa.mean()
        kappa_kde_mu = df.kappa_kde.mean()
        corr_k = df.kappa.corr(df.delta)
        corr_kde = df.kappa_kde.corr(df.delta)

        # Bias of plug-in vs KDE
        bias = (kappa_mu - kappa_kde_mu) / max(kappa_kde_mu, 1e-10) * 100

        print(f"{n:>5d} {int(df.nc.iloc[0]):>5d} {delta:>8.4f} {se:>8.4f} {kappa_mu:>8.2f} "
              f"{kappa_kde_mu:>8.2f} {corr_k:>10.3f} {corr_kde:>10.3f} {bias:>8.0f}%")
        rows.append(dict(n=n, delta=delta, se=se, kappa=kappa_mu,
                         kappa_kde=kappa_kde_mu, corr_k=corr_k,
                         corr_kde=corr_kde, bias_pct=bias))
    return pd.DataFrame(rows)


print("=" * 75)
print("  Robustness Analysis: κ̂ under non-normal residuals")
print("=" * 75)
print()
print("Distributions (all scaled to variance 1):")
print("  normal — N(0,1) [baseline]")
print("  t5     — t(5) / sqrt(5/3) [heavy-tailed, excess kurtosis = 6]")
print("  skewed — (Gamma(2,1)-2) / sqrt(2) [right-skewed, skew = sqrt(2)]")
print()

results = {}
for dist in ['normal', 't5', 'skewed']:
    results[dist] = run_dist(dist)

print("\n" + "=" * 75)
print("  Summary: Bias of normal plug-in κ̂ relative to KDE correction")
print("=" * 75)
print(f"{'dist':>10} {'n':>5} {'κ̂':>8} {'κ̂_KDE':>8} {'bias%':>8} {'corr(κ̂,∆)':>10} {'corr(KDE,∆)':>10}")
print("-" * 55)
for dist in ['normal', 't5', 'skewed']:
    for _, r in results[dist].iterrows():
        print(f"{dist:>10} {int(r.n):>5d} {r.kappa:>8.2f} {r.kappa_kde:>8.2f} "
              f"{r.bias_pct:>8.0f}% {r.corr_k:>10.3f} {r.corr_kde:>10.3f}")
    print()

print("\nDone.")
