"""
Baseline comparison: κ vs CV 1-SE rule for predicting misalignment.
Compare κ diagnostic against CV 1-SE in terms of detecting unreliable CP selection.
Design: OLS_full(p=50) vs OLS_restricted(s=3), sweep p to vary competition.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from scipy.stats import norm
F0 = 2 * norm.pdf(norm.ppf(0.95))  # ≈ 0.206
import warnings
warnings.filterwarnings("ignore")
np.random.seed(2039)

N_REPS = 500

def run_one(p, n=200, s=3):
    X = np.random.randn(n, p)
    beta = np.zeros(p); beta[:s] = 1.0
    y = X @ beta + np.random.randn(n)

    nc = max(int(n*0.25), 10); nt = nc; ntr = n - nc - nt
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

    # κ
    raw_f = yca - ols_full.predict(Xca)
    sigma_f = np.std(raw_f, ddof=1)
    raw_r = yca - ols_restr.predict(Xca[:, :s])
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

    # CV 1-SE rule: compare 5-fold CV MSE of both estimators
    cv_f = cross_val_score(LinearRegression(), Xtr, ytr, cv=5, scoring='neg_mean_squared_error')
    cv_r = cross_val_score(LinearRegression(), Xtr[:, :s], ytr, cv=5, scoring='neg_mean_squared_error')
    cv_mean_f, cv_se_f = -cv_f.mean(), cv_f.std() / np.sqrt(5)
    cv_mean_r, cv_se_r = -cv_r.mean(), cv_r.std() / np.sqrt(5)
    # 1-SE rule: select restricted if its CV MSE is within 1 SE of the full's minimum
    cv_min = min(cv_mean_f, cv_mean_r)
    cv_se_at_min = cv_se_f if cv_mean_f == cv_min else cv_se_r
    # 1-SE rule selects the simpler model within 1 SE
    cv_1se_selects_restr = int(cv_mean_r <= cv_min + cv_se_at_min)
    cv_1se_selects_full = int(cv_mean_f <= cv_min + cv_se_at_min)
    if cv_1se_selects_restr and cv_1se_selects_full:
        # both within 1 SE: select simpler (restr)
        cv_1se_selects = 0  # 0 = restr
    elif cv_1se_selects_restr:
        cv_1se_selects = 0
    else:
        cv_1se_selects = 1  # 1 = full

    # Oracle selection (by test risk)
    oracle_selects_restr = int(risk_r < risk_f)

    delta = int((cp_r < cp_f) != (risk_r < risk_f))
    delta_cv = int(cv_1se_selects != oracle_selects_restr)

    return dict(p=p, n=n, delta=delta, delta_cv=delta_cv, kappa=kappa,
                cp_gap=cp_f-cp_r, risk_gap=risk_f-risk_r)


print("=== Baseline: κ vs CV 1-SE ===")
print(f"{'p':>4} {'∆_CP':>8} {'∆_CV':>8} {'κ':>8}")
print("-" * 35)

for p in [10, 20, 50, 100]:
    reps = Parallel(n_jobs=-1)(delayed(run_one)(p) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    delta = df.delta.mean()
    delta_cv = df.delta_cv.mean()
    kappa = df.kappa.mean()
    print(f"{p:>4d} {delta:>8.4f} {delta_cv:>8.4f} {kappa:>8.3f}")

print("\nDone.")
