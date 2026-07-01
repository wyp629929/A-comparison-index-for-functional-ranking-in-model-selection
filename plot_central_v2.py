"""
Unified central figure: κ vs ∆ across all experiments.
Multi-panel: (a) main scatter with error bars + LOESS, (b) proportion with κ < 2 vs ∆.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

# === Load individual-rep data ===
decay = pd.read_csv("experiment_results/decay_v3_all.csv")
decay['channel'] = 'decay'
decay['n_group'] = decay['n']

dia100 = pd.read_csv("experiment_results/diabetes_n100.csv")
dia200 = pd.read_csv("experiment_results/diabetes_n200.csv")
dia442 = pd.read_csv("experiment_results/diabetes_n442.csv")
dia = pd.concat([d.assign(channel='diabetes', n_group=d.n.iloc[0]) for d in [dia100, dia200, dia442]])

hou_files = [('housing_n100.csv', 100), ('housing_n200.csv', 200),
             ('housing_n500.csv', 500), ('housing_n1000.csv', 1000)]
hou = pd.concat([
    pd.read_csv(f"experiment_results/{f}").assign(channel='housing', n_group=n)
    for f, n in hou_files
])

all_data = pd.concat([decay, dia, hou], ignore_index=True)
all_data = all_data.dropna(subset=['kappa', 'delta'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# === Panel (a): Main scatter with LOESS and error bars ===
ax = axes[0]
colors = {'decay': '#2196F3', 'diabetes': '#FF9800', 'housing': '#4CAF50'}
markers = {'decay': 'o', 'diabetes': 's', 'housing': '^'}

# Background scatter (subsampled)
for ch, color in colors.items():
    sub = all_data[all_data['channel'] == ch]
    if len(sub) > 400:
        sub = sub.sample(400, random_state=42)
    ax.scatter(sub['kappa'], sub['delta'], c=color, alpha=0.06, s=5,
               marker=markers[ch])

# Per-channel binned aggregates with error bars
chan_agg = {}  # channel -> aggregated DataFrame
for ch, color in colors.items():
    sub = all_data[all_data['channel'] == ch].copy()
    bins = np.linspace(0, 20, 31)
    sub['bin'] = pd.cut(sub['kappa'], bins=bins)
    agg = sub.groupby('bin', observed=False).agg(
        k=('kappa', 'mean'), d=('delta', 'mean'), n=('delta', 'count')
    ).reset_index().dropna()
    agg = agg[agg['n'] >= 5]

    # Wilson score interval for binomial proportion (avoids [0,1] boundary issues)
    z = 1.96
    d, n = agg['d'].values, agg['n'].values
    denom = 1 + z**2 / n
    ci_low = (d + z**2/(2*n) - z * np.sqrt(d*(1-d)/n + z**2/(4*n**2))) / denom
    ci_high = (d + z**2/(2*n) + z * np.sqrt(d*(1-d)/n + z**2/(4*n**2))) / denom
    agg['ci_low'] = np.maximum(0, d - ci_low)
    agg['ci_high'] = np.maximum(0, ci_high - d)

    ax.errorbar(agg['k'], agg['d'], yerr=[agg['ci_low'], agg['ci_high']],
                fmt=markers[ch], color=color, capsize=2, markersize=5,
                alpha=0.8)
    # Binomial SE for LOESS inverse-variance weighting
    agg['se'] = np.sqrt(agg['d'] * (1 - agg['d']) / agg['n'])
    chan_agg[ch] = agg

# LOESS-like smooth per channel (solid to distinguish from dashed theoretical curves)
kappa_grid = np.linspace(0.1, 19.9, 200)
for ch, color in colors.items():
    agg = chan_agg[ch]
    if len(agg) < 8:
        continue
    try:
        spl = UnivariateSpline(agg['k'], agg['d'], w=1/agg['se'], s=8.0)
        ax.plot(kappa_grid, spl(kappa_grid), color=color, linewidth=1.5,
                linestyle='-', alpha=0.5)
    except Exception:
        pass

# Overall binned mean (black curve)
bins_all = np.linspace(0, 20, 41)
all_data['kappa_bin'] = pd.cut(all_data['kappa'], bins=bins_all)
agg_all = all_data.groupby('kappa_bin', observed=False).agg(
    delta_mean=('delta', 'mean'), kappa_mean=('kappa', 'mean'), cnt=('delta', 'count')
).reset_index().dropna()
agg_all = agg_all[agg_all['cnt'] >= 3]
ax.plot(agg_all['kappa_mean'], agg_all['delta_mean'], 'k-', linewidth=2, alpha=0.7,
        label='Overall binned mean')

# Theoretical curves (all gray — not per-channel — to distinguish from LOESS)
for nc in [25, 125, 250]:
    pred = norm.cdf(-np.sqrt(nc) * kappa_grid)
    ax.plot(kappa_grid, pred, '--', color='gray', alpha=0.5, linewidth=1,
            label=r'$\Phi(-\sqrt{' + str(nc) + r'}\,\kappa)$')

ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Random baseline')
ax.set_xlabel(r'Competition density $\kappa$', fontsize=13)
ax.set_ylabel(r'Misalignment probability $\Delta$', fontsize=13)
ax.set_title(r'(a) $\kappa$ predicts $\Delta$ across all regimes', fontsize=12)
ax.set_xlim(0, 20)
ax.set_ylim(-0.02, 1.02)
# Custom legend: each channel shows marker + LOESS line as one combined entry
channel_legend = [
    ('decay', '#2196F3', 'o'),
    ('diabetes', '#FF9800', 's'),
    ('housing', '#4CAF50', '^'),
]
combined = []
for ch, color, marker in channel_legend:
    m = Line2D([0], [0], color=color, marker=marker, linestyle='', markersize=5)
    l = Line2D([0], [0], color=color, linestyle='-', linewidth=1.5, alpha=0.5)
    combined.append(((m, l), ch))

# Auto-generated entries (theoretical curves, overall mean, baseline)
auto_handles, auto_labels = ax.get_legend_handles_labels()

all_handles = [c[0] for c in combined] + auto_handles
all_labels = [c[1] for c in combined] + auto_labels

ax.legend(handles=all_handles, labels=all_labels,
          handler_map={tuple: HandlerTuple(ndivide=None)},
          fontsize=7.5, loc='upper right', ncol=2)
ax.grid(alpha=0.15)

# === Panel (b): Proportion with κ < 2 vs ∆ ===
ax = axes[1]

for ch, color, marker in [('decay', '#2196F3', 'o'), ('diabetes', '#FF9800', 's'),
                           ('housing', '#4CAF50', '^')]:
    sub = all_data[all_data['channel'] == ch]
    if len(sub) < 20:
        continue
    stats = sub.groupby('n_group').agg(
        delta=('delta', 'mean'),
        prop_low_kappa=('kappa', lambda x: (x < 2).mean()),
        n=('delta', 'count')
    ).reset_index()

    ax.scatter(stats['prop_low_kappa'], stats['delta'], c=color, marker=marker,
               s=80, zorder=5, label=ch)
    # Wilson score CI for ∆ (per n_group)
    for _, row in stats.iterrows():
        d, nn = row['delta'], row['n']
        z = 1.96
        denom = 1 + z**2/nn
        ci_low = (d + z**2/(2*nn) - z * np.sqrt(d*(1-d)/nn + z**2/(4*nn**2))) / denom
        ci_high = (d + z**2/(2*nn) + z * np.sqrt(d*(1-d)/nn + z**2/(4*nn**2))) / denom
        yerr_low = max(0, d - ci_low)
        yerr_high = max(0, ci_high - d)
        ax.errorbar(row['prop_low_kappa'], row['delta'],
                    yerr=[[yerr_low], [yerr_high]],
                    fmt='none', color=color, alpha=0.4, capsize=2)
        ax.annotate(f"n={int(row['n_group'])}", (row['prop_low_kappa'], row['delta']),
                    xytext=(5, -8), textcoords='offset points', fontsize=7, color=color)

ax.set_xlabel(r'Proportion of reps with $\hat\kappa < 2$', fontsize=13)
ax.set_ylabel(r'$\Delta$', fontsize=13)
ax.set_title(r'(b) $\kappa < 2$ proportion predicts $\Delta$', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.15)

plt.tight_layout()
plt.savefig("fig_central.png", dpi=150)
print("Saved fig_central.png (multi-panel with error bars + LOESS)")

# Re-plot on axes, but also save a separate version with detailed caption in filename
print("\nFigure contains:")
print(f"  Total points: {len(all_data)}")
for ch in ['decay', 'diabetes', 'housing']:
    sub = all_data[all_data['channel'] == ch]
    print(f"  {ch}: {len(sub)} reps, mean κ={sub.kappa.mean():.2f}, mean ∆={sub.delta.mean():.3f}")

print("\nPanel (b) data points:")
for ch in ['decay', 'diabetes', 'housing']:
    sub = all_data[all_data['channel'] == ch]
    if len(sub) < 20: continue
    stats = sub.groupby('n_group').agg(
        delta=('delta', 'mean'),
        prop_low_kappa=('kappa', lambda x: (x < 2).mean()),
    ).reset_index()
    for _, r in stats.iterrows():
        print(f"  {ch} n={int(r['n_group']):5d}: prop(κ<2)={r['prop_low_kappa']:.3f}, ∆={r['delta']:.3f}")
