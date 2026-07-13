"""Analyze experiment results."""
import pandas as pd
import numpy as np

df = pd.read_csv("/Users/wangyaoping/Desktop/experiment_results/raw_results.csv")

print("=== Main Results ===")
print(f"{'DGP':<12} {'EstSet':<10} {'ρ':<6} {'∆':<8} {'SE':<8} {'κ':<8}")
print("-" * 52)

for dgp in ["linear", "threshold"]:
    for est_key in ["Linear3", "Mixed5"]:
        for rho in [0.0, 0.5, 0.75]:
            sub = df[(df["dgp"] == dgp) & (df["est_key"] == est_key) & (df["rho"] == rho)]
            delta = sub["delta"].mean()
            se = np.sqrt(delta * (1 - delta) / len(sub))
            kappa = sub["kappa"].mean()
            print(f"{dgp:<12} {est_key:<10} {rho:<6.2f} {delta:<8.3f} {se:<8.3f} {kappa:<8.2f}")
    print()

# — ρ effect on ∆ (competition channel) —
print("=== ρ effect (∆_ρ=0.75 - ∆_ρ=0) ===")
for dgp in ["linear", "threshold"]:
    for est_key in ["Linear3", "Mixed5"]:
        sub0 = df[(df["dgp"] == dgp) & (df["est_key"] == est_key) & (df["rho"] == 0.0)]
        sub75 = df[(df["dgp"] == dgp) & (df["est_key"] == est_key) & (df["rho"] == 0.75)]
        d0 = sub0["delta"].mean()
        d75 = sub75["delta"].mean()
        diff = d75 - d0
        se_diff = np.sqrt(d0*(1-d0)/len(sub0) + d75*(1-d75)/len(sub75))
        print(f"{dgp:<12} {est_key:<10}  ∆_diff = {diff:+.3f} (SE={se_diff:.3f})")

print()

# — Winner analysis (key cells) —
for dgp in ["linear", "threshold"]:
    for est_key in ["Linear3", "Mixed5"]:
        for rho in [0.0, 0.75]:
            sub = df[(df["dgp"] == dgp) & (df["est_key"] == est_key) & (df["rho"] == rho)]
            cp_winners = sub["cp_winner"].value_counts(normalize=True).sort_index()
            risk_winners = sub["risk_winner"].value_counts(normalize=True).sort_index()
            print(f"{dgp:<10} {est_key:<8} ρ={rho}:")
            print(f"  CP winners:   {cp_winners.to_dict()}")
            print(f"  Risk winners: {risk_winners.to_dict()}")

print()

# — κ vs ∆ relationship —
print("=== κ vs ∆ (by group) ===")
for (dgp, est_key), grp in df.groupby(["dgp", "est_key"]):
    rho_list = []
    delta_list = []
    kappa_list = []
    for rho in [0.0, 0.5, 0.75]:
        sub = grp[grp["rho"] == rho]
        rho_list.append(rho)
        delta_list.append(sub["delta"].mean())
        kappa_list.append(sub["kappa"].mean())
    # Correlation between κ and ∆ across ρ
    if len(set(kappa_list)) > 1:
        corr = np.corrcoef(kappa_list, delta_list)[0, 1]
    else:
        corr = float('nan')
    print(f"{dgp:<10} {est_key:<8}  κ = {kappa_list}, ∆ = {[f'{d:.3f}' for d in delta_list]}, corr = {corr:.2f}")
