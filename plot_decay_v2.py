"""
Decay figure with population κ.
Shows ∆(n) decay alongside plug-in κ and population κ.
Designed as a single-panel figure with dual y-axes.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")
np.random.seed(2041)

# === Step 1: Compute population κ for each n ===
def compute_population_kappa(n, n_mc=20000):
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

        # Use raw residuals for correct Bahadur variance
        raw_f = yca_big - ols_f.predict(Xca_big)
        raw_r = yca_big - ols_r.predict(Xca_big[:, :s])
        sigma_f = np.std(raw_f, ddof=1)
        sigma_r = np.std(raw_r, ddof=1)
        f0 = 2 * norm.pdf(norm.ppf(0.95))
        f_f = f0 / max(sigma_f, 1e-10)
        f_r = f0 / max(sigma_r, 1e-10)

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

    return float(np.median(kappas))

ns = [50, 100, 200, 500, 1000]
print("Computing population κ...")
pop_kappa = {}
for n in ns:
    pk = compute_population_kappa(n)
    pop_kappa[n] = pk
    print(f"  n={n:5d}  κ_pop = {pk:.3f}")

# === Step 2: Load per-rep data ===
all_data = pd.read_csv("experiment_results/decay_v3_all.csv")

# Aggregate by n
agg = all_data.groupby('n').agg(
    delta=('delta', 'mean'),
    kappa_plug=('kappa', 'mean'),
    nc=('nc', 'first'),
    n_reps=('delta', 'count')
).reset_index()

# Wilson CI for ∆
z = 1.96
d, nn = agg['delta'].values, agg['n_reps'].values
denom = 1 + z**2 / nn
ci_low = (d + z**2/(2*nn) - z * np.sqrt(d*(1-d)/nn + z**2/(4*nn**2))) / denom
ci_high = (d + z**2/(2*nn) + z * np.sqrt(d*(1-d)/nn + z**2/(4*nn**2))) / denom
agg['ci_low'] = d - ci_low
agg['ci_high'] = ci_high - d

# Add population κ
agg['kappa_pop'] = agg['n'].map(pop_kappa)

# Theoretical prediction
nc_vals = agg['nc'].values
kappa_pop_vals = agg['kappa_pop'].values
agg['theoretical'] = [norm.cdf(-np.sqrt(nc) * kp) if kp > 0 else 0.5
                      for nc, kp in zip(nc_vals, kappa_pop_vals)]

print("\nAggregated data:")
print(f"{'n':>5} {'∆':>8} {'κ_plug':>8} {'κ_pop':>8} {'Φ(-√nc·κ_pop)':>16}")
print("-" * 45)
for _, r in agg.iterrows():
    pred = norm.cdf(-np.sqrt(r.nc) * r.kappa_pop) if r.kappa_pop > 0 else 0.5
    print(f"{int(r.n):>5d} {r.delta:>8.4f} {r.kappa_plug:>8.2f} {r.kappa_pop:>8.3f} {pred:>16.4f}")

# === Step 3: Create figure ===
fig, ax1 = plt.subplots(figsize=(8, 5))

# Colors
color_delta = '#D32F2F'
color_kappa = '#1565C0'
color_plug = '#FF8F00'

# --- ∆(n) with Wilson CI (left axis) ---
ax1.errorbar(agg['n'], agg['delta'],
             yerr=[agg['ci_low'], agg['ci_high']],
             color=color_delta, marker='o', markersize=8, capsize=4,
             linestyle='-', linewidth=1.5, label=r'$\Delta$ (misalignment)')
ax1.set_xlabel('Sample size $n$', fontsize=13)
ax1.set_ylabel(r'Misalignment probability $\Delta$', fontsize=13, color=color_delta)
ax1.tick_params(axis='y', labelcolor=color_delta)
ax1.set_xscale('log')
ax1.set_xticks(ns)
ax1.set_xticklabels([str(n) for n in ns])
ax1.set_xlim(40, 1200)
ax1.set_ylim(-0.02, 0.55)
ax1.grid(alpha=0.15)

# --- κ_pop and κ_plug (right axis) ---
ax2 = ax1.twinx()
ax2.plot(agg['n'], agg['kappa_pop'], 's-', color=color_kappa,
         markersize=7, linewidth=2, label=r'Population $\kappa$')
ax2.plot(agg['n'], agg['kappa_plug'], '^--', color=color_plug,
         markersize=7, linewidth=1.5, alpha=0.8, label=r'Plug-in $\hat\kappa$')
ax2.set_ylabel(r'Competition density $\kappa$', fontsize=13, color=color_kappa)
ax2.tick_params(axis='y', labelcolor=color_kappa)
ax2.set_ylim(0, 22)

# κ=2 threshold line
ax2.axhline(2.0, color='gray', linestyle=':', alpha=0.6, linewidth=1.2)
ax2.annotate(r'$\kappa=2$ threshold', xy=(550, 2.0), xytext=(550, 3.8),
             fontsize=9, color='gray', fontstyle='italic',
             arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=0.8))

# Combined legend (lower right to avoid overlapping n=1000 ∆ point)
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color=color_delta, marker='o', linestyle='-', linewidth=1.5),
    Line2D([0], [0], color=color_kappa, marker='s', linestyle='-', linewidth=2),
    Line2D([0], [0], color=color_plug, marker='^', linestyle='--', linewidth=1.5, alpha=0.8),
]
labels = [r'$\Delta$', r'Population $\kappa$', r'Plug-in $\hat\kappa$']
ax1.legend(handles=handles, labels=labels, fontsize=9, loc='lower right')

ax1.set_title(r'$\Delta(n)$ decay and competition density $\kappa$', fontsize=12)

plt.tight_layout()
plt.savefig("fig_decay.png", dpi=150)
print("\nSaved fig_decay.png")

# Report
print(f"\nAt κ=2 threshold:")
# Find n where κ_pop crosses 2
from scipy.interpolate import interp1d
f_kappa = interp1d(agg['kappa_pop'], agg['n'], bounds_error=False)
n_at_kappa2 = f_kappa(2.0)
print(f"  κ_pop = 2 at n ≈ {n_at_kappa2:.0f}")
print(f"  At n=50: κ_pop={pop_kappa[50]:.2f}, ∆={agg[agg.n==50].delta.values[0]:.3f}")
print(f"  At n=100: κ_pop={pop_kappa[100]:.2f}, ∆={agg[agg.n==100].delta.values[0]:.3f}")
