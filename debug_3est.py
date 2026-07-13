"""Quick debug: 3-estimator case"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(2028)

def run(n=300, p=20, rho=0.0):
    s = 3; beta = np.zeros(p); beta[:s] = 1.0
    B = max(p // 5, 1); m = p // B
    Sigma = np.eye(p)
    for b in range(B):
        idx = slice(b*m, (b+1)*m)
        Sigma[idx, idx] = rho + (1-rho)*np.eye(m)
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T
    y = X @ beta + np.random.randn(n)
    nt = max(int(n*0.2), 10); nc = nt; ntr = n - nt - nc
    if ntr < 5: return None
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=nc+nt, random_state=None)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=nt/(nc+nt), random_state=None)

    ols = LinearRegression().fit(Xtr, ytr)
    ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=None)
    rf.fit(Xtr, ytr)

    res = {}
    for name, est, Xc, Xt in [("ols", ols, Xca, Xte), ("ridge", ridge, Xca, Xte), ("rf", rf, Xca, Xte)]:
        cp = 2 * np.quantile(np.abs(yca - est.predict(Xc)), 0.9)
        risk = mean_squared_error(yte, est.predict(Xt))
        res[f"cp_{name}"] = cp; res[f"risk_{name}"] = risk
    return res

reps = [run(300, 20, 0.0) for _ in range(100)]
reps = [r for r in reps if r is not None]
df = pd.DataFrame(reps)

print("Mean CP widths:", {k: df[k].mean() for k in ["cp_ols", "cp_ridge", "cp_rf"]})
print("Mean risks:", {k: df[k].mean() for k in ["risk_ols", "risk_ridge", "risk_rf"]})

cp_best = df[["cp_ols", "cp_ridge", "cp_rf"]].idxmin(axis=1)
risk_best = df[["risk_ols", "risk_ridge", "risk_rf"]].idxmin(axis=1)
print("\nCP win rates:", cp_best.value_counts(normalize=True).to_dict())
print("Risk win rates:", risk_best.value_counts(normalize=True).to_dict())
print("Delta:", (cp_best != risk_best).mean())
