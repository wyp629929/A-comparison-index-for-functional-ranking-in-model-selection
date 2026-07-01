"""
Failure Map: 2D conceptual figure showing κ vs bias-crossing strength.
Four quadrants: Reliable, Competition-failure, Structural-failure, Dual-failure.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(7, 6))

# Draw quadrants
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)

# Shaded regions
# Reliable (top-right): high κ, low bias-crossing
ax.fill_between([0, 5], 0, 5, alpha=0.08, color='green', label='_nolegend_')
# Competition (bottom-right): high κ, high bias-crossing
ax.fill_between([0, 5], -5, 0, alpha=0.08, color='orange', label='_nolegend_')
# Structural failure (top-left): low κ, low bias-crossing
ax.fill_between([-5, 0], 0, 5, alpha=0.08, color='blue', label='_nolegend_')
# Dual failure (bottom-left): low κ, high bias-crossing
ax.fill_between([-5, 0], -5, 0, alpha=0.08, color='red', label='_nolegend_')

# Labels
ax.text(2.5, 3.5, 'Reliable', fontsize=14, ha='center', fontweight='bold', color='green')
ax.text(2.5, 0.5, 'Competition\nfailure', fontsize=12, ha='center', color='orange', fontweight='bold')
ax.text(-2.5, 3.5, 'κ powerless\n(bias-crossing\nregime)', fontsize=11, ha='center', color='blue', fontweight='bold')
ax.text(-2.5, -3, 'Dual\nfailure', fontsize=14, ha='center', fontweight='bold', color='darkred')

# Axis labels
ax.set_xlabel(r'Competition density $\kappa$ (log scale)', fontsize=13)
ax.set_ylabel('Bias-crossing severity', fontsize=13)
ax.set_xticks([-4, -2, 0, 2, 4])
ax.set_xticklabels(['0', '', '', '', ''])
ax.set_yticks([-4, -2, 0, 2, 4])
ax.set_yticklabels(['High', '', 'Low', '', ''])

# Arrow annotations
ax.annotate('κ increases →', xy=(3, -4.5), fontsize=11, ha='center', color='gray')
ax.annotate('← Bias crossing increases', xy=(-4, 3.5), fontsize=11, ha='center', rotation=90, color='gray')

# Data points from experiments (approximate positions)
# These are conceptual, not exact
experiments = [
    ('Decay n=1000', 1.5, 1.0, 'green'),
    ('Decay n=100', -0.5, 1.5, 'orange'),
    ('Housing n=1000', 1.8, -3.5, 'red'),
    ('Table 1 p=10', 0.8, 2.0, 'green'),
    ('Table 2 ρ=0.75', -0.3, 1.5, 'orange'),
]

for label, x, y, c in experiments:
    ax.scatter(x, y, s=80, c=c, zorder=5, edgecolors='black', linewidth=0.5)
    ax.annotate(label, (x, y), xytext=(8, 5), textcoords='offset points', fontsize=7)

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.grid(alpha=0.1)
ax.set_title('Failure Map: When does CP-based selection fail?', fontsize=13)

plt.tight_layout()
plt.savefig("fig_failure_map.png", dpi=150)
print("Saved fig_failure_map.png")
