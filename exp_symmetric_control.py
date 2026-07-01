"""
Symmetric control: OLS vs Ridge over ρ sweep.
Prediction: ∆ is flat across ρ (symmetric dependence sensitivity).
Compare with Table 2 (OLS vs RF, asymmetric → ∆ increases with ρ).
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")
np.random.seed(2037)

N_REPS = 500

def make_block_cov(p, B, rho):
    m = p // B
    if m <= 1: return np.eye(p)
    Sigma = np.eye(p)
    for b in range(B):
        idx = slice(b*m, (b+1)*m)
        Sigma[idx, idx] = rho + (1-rho)*np.eye(m)
    return Sigma

def run_one(rho, n=300, p=50, n_sig=3):
    B = max(p // 5, 1)
    Sigma = make_block_cov(p, B, rho)
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T
    beta = np.zeros(p); beta[:n_sig] = 1.0
    y = X @ beta + np.random.randn(n)

    nc = max(int(n*0.25), 10); nt = nc; ntr = n - nc - nt
    if ntr < 10: return None
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt, random_state=None)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt), random_state=None)

    ols = LinearRegression().fit(Xtr, ytr)
    ridge = Ridge(alpha=1.0).fit(Xtr, ytr)

    cp_ols = 2 * np.quantile(np.abs(yca - ols.predict(Xca)), 0.9)
    cp_ridge = 2 * np.quantile(np.abs(yca - ridge.predict(Xca)), 0.9)
    risk_ols = mean_squared_error(yte, ols.predict(Xte))
    risk_ridge = mean_squared_error(yte, ridge.predict(Xte))

    delta = int((cp_ridge < cp_ols) != (risk_ridge < risk_ols))
    return dict(rho=rho, n=n, p=p, delta=delta,
                cp_ols=cp_ols, cp_ridge=cp_ridge,
                risk_ols=risk_ols, risk_ridge=risk_ridge)


print("=== Symmetric control: OLS vs Ridge, ρ sweep ===")
print(f"{'ρ':>6} {'∆':>8} {'SE':>8} {'CP_OLS':>8} {'CP_Ridge':>8} {'Gap':>8}")
print("-" * 55)

all_data = []
for rho in [0.00, 0.25, 0.50, 0.75]:
    reps = Parallel(n_jobs=-1)(delayed(run_one)(rho) for _ in range(N_REPS))
    reps = [r for r in reps if r is not None]
    df = pd.DataFrame(reps)
    delta = df.delta.mean()
    se = np.sqrt(delta*(1-delta)/len(df))
    cp_o = df.cp_ols.mean()
    cp_r = df.cp_ridge.mean()
    print(f"{rho:>6.2f} {delta:>8.4f} {se:>8.4f} {cp_o:>8.2f} {cp_r:>8.2f} {cp_o-cp_r:>8.3f}")
    for _, r in df.iterrows():
        all_data.append(r.to_dict())

df_all = pd.DataFrame(all_data)
df_all.to_csv("/Users/wangyaoping/Desktop/paper3/experiment_results/symmetric_control.csv", index=False)
print("\nSaved symmetric_control.csv")
