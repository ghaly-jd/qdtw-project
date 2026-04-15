"""
Create K-Sweep Plot for Presentation
=====================================

Generates publication-quality plot showing PCA vs VQD accuracy across k values
with mean ± std error bars.

Author: VQD-DTW Research Team
Date: November 25, 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# Load results
results_file = Path(__file__).parent / "results" / "k_sweep_ci_results.json"
FIG_DIR = Path(__file__).parent / "figures"

with open(results_file) as f:
    data = json.load(f)

# Extract data
k_values = [6, 8, 10, 12]
pca_means = []
pca_stds = []
vqd_means = []
vqd_stds = []

for k in k_values:
    agg = data['aggregated'][str(k)]
    pca_means.append(agg['pca']['mean'] * 100)
    pca_stds.append(agg['pca']['std'] * 100)
    vqd_means.append(agg['vqd']['mean'] * 100)
    vqd_stds.append(agg['vqd']['std'] * 100)

# Convert to numpy arrays
pca_means = np.array(pca_means)
pca_stds = np.array(pca_stds)
vqd_means = np.array(vqd_means)
vqd_stds = np.array(vqd_stds)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 7))

# Plot with error bars
ax.errorbar(k_values, pca_means, yerr=pca_stds, 
           fmt='o-', linewidth=3, markersize=12,
           color='steelblue', ecolor='steelblue', 
           capsize=8, capthick=2.5, alpha=0.9,
           label='PCA')

ax.errorbar(k_values, vqd_means, yerr=vqd_stds,
           fmt='s-', linewidth=3, markersize=12,
           color='coral', ecolor='coral',
           capsize=8, capthick=2.5, alpha=0.9,
           label='VQD')

# Styling
ax.set_xlabel('Number of Components (k)', fontsize=18, fontweight='bold')
ax.set_ylabel('Classification Accuracy (%)', fontsize=18, fontweight='bold')
ax.set_title('VQD vs PCA: K-Sweep Results\n(Mean ± Std, n=5 seeds)', 
            fontsize=20, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax.set_axisbelow(True)

# Set x-axis ticks
ax.set_xticks(k_values)
ax.set_xticklabels(k_values, fontsize=16)

# Y-axis limits for better visibility
ax.set_ylim(68, 88)

# Legend
legend = ax.legend(loc='lower right', fontsize=16, framealpha=0.95,
                   edgecolor='black', fancybox=True, shadow=True)

# Add value labels on the points
for i, k in enumerate(k_values):
    # PCA labels
    ax.text(k, pca_means[i] - 2.5, 
           f'{pca_means[i]:.1f}±{pca_stds[i]:.1f}',
           ha='center', va='top', fontsize=11, 
           fontweight='bold', color='steelblue',
           bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='white', edgecolor='steelblue', alpha=0.8))
    
    # VQD labels
    ax.text(k, vqd_means[i] + 2.5,
           f'{vqd_means[i]:.1f}±{vqd_stds[i]:.1f}',
           ha='center', va='bottom', fontsize=11,
           fontweight='bold', color='coral',
           bbox=dict(boxstyle='round,pad=0.3',
                    facecolor='white', edgecolor='coral', alpha=0.8))

# Add shaded region to show VQD advantage
ax.fill_between(k_values, pca_means, vqd_means, 
               alpha=0.15, color='green', label='_nolegend_')

plt.tight_layout()

# Save
output_file = FIG_DIR / 'k_sweep_results.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")

# Also create a version without labels for cleaner look
fig2, ax2 = plt.subplots(figsize=(12, 7))

ax2.errorbar(k_values, pca_means, yerr=pca_stds, 
            fmt='o-', linewidth=4, markersize=14,
            color='steelblue', ecolor='steelblue', 
            capsize=10, capthick=3, alpha=0.9,
            label='PCA')

ax2.errorbar(k_values, vqd_means, yerr=vqd_stds,
            fmt='s-', linewidth=4, markersize=14,
            color='coral', ecolor='coral',
            capsize=10, capthick=3, alpha=0.9,
            label='VQD')

ax2.set_xlabel('Number of Components (k)', fontsize=18, fontweight='bold')
ax2.set_ylabel('Classification Accuracy (%)', fontsize=18, fontweight='bold')
ax2.set_title('VQD vs PCA: K-Sweep Results', 
             fontsize=20, fontweight='bold', pad=20)

ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax2.set_axisbelow(True)
ax2.set_xticks(k_values)
ax2.set_xticklabels(k_values, fontsize=16)
ax2.set_ylim(68, 88)

legend2 = ax2.legend(loc='lower right', fontsize=18, framealpha=0.95,
                    edgecolor='black', fancybox=True, shadow=True)

ax2.fill_between(k_values, pca_means, vqd_means, 
                alpha=0.15, color='green', label='_nolegend_')

plt.tight_layout()

output_file2 = FIG_DIR / 'k_sweep_results_clean.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file2}")

print("\n" + "="*70)
print("K-SWEEP PLOTS GENERATED!")
print("="*70)
print(f"\n1. k_sweep_results.png - With value labels")
print(f"2. k_sweep_results_clean.png - Clean version")
print(f"\nBoth saved to: {FIG_DIR}/")
print("\n✨ Ready for presentation! ✨")
