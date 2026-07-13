"""Generate central figure: κ vs ∆ across all experiments."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load κ-∆ data from decay experiment
df = pd.read_csv("/Users/wangyaoping/Desktop/experiment_results/kappa_delta_all.csv")

# Also load from factorial experiment
try:
    df2 = pd.read_csv("/Users/wangyaoping/Desktop/experiment_results/raw_results.csv")
    # Merge if possible
except:
    df2 = None

# Bin κ
df['kappa_bin'] = pd.cut(df['kappa'], bins=np.linspace(0, 5, 31))
agg = df.groupby('kappa_bin', observed=False).agg(
    delta_mean=('delta', 'mean'),
    kappa_mean=('kappa', 'mean'),
    count=('delta', 'count')
).reset_index()
agg = agg[agg['count'] >= 5].dropna()

# Theory curve: Φ(-√n·κ) for different n
n_cal_vals = [25, 50, 125, 250]
kappa_grid = np.linspace(0, 5, 200)

fig, ax = plt.subplots(figsize=(8, 5))

# Scatter: binned experimental data
ax.scatter(agg['kappa_mean'], agg['delta_mean'], s=agg['count'],
           alpha=0.7, c='steelblue', edgecolors='k', zorder=5)
ax.set_xlabel('Competition density κ', fontsize=13)
ax.set_ylabel('Misalignment probability Δ', fontsize=13)
ax.set_title('κ predicts Δ across experimental regimes', fontsize=14)

# Theoretical bounds for different n_cal
for nc in n_cal_vals:
    pred = norm.cdf(-np.sqrt(nc) * kappa_grid)
    ax.plot(kappa_grid, pred, '--', alpha=0.4, label=f'$\\Phi(-\\sqrt{{{nc}}}\\,\\kappa)$')

# Random baseline
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Random baseline (m=2)')

ax.legend(fontsize=10)
ax.set_xlim(0, 5)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("/Users/wangyaoping/Desktop/fig_central.png", dpi=150)
print("Saved fig_central.png")

# Also print summary statistics
print("\nSummary:")
print(f"κ < 1: mean ∆ = {df[df['kappa'] < 1]['delta'].mean():.3f} (n={len(df[df['kappa'] < 1])})")
print(f"1 ≤ κ < 2: mean ∆ = {df[(df['kappa'] >= 1) & (df['kappa'] < 2)]['delta'].mean():.3f} (n={len(df[(df['kappa'] >= 1) & (df['kappa'] < 2)])})")
print(f"κ ≥ 2: mean ∆ = {df[df['kappa'] >= 2]['delta'].mean():.3f} (n={len(df[df['kappa'] >= 2])})")
