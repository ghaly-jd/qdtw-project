"""
Create Per-Class Delta Recall Bar Plot
=======================================

Generates bar plot showing VQD - PCA difference (Δ) for each action class.
Positive bars = VQD better, Negative bars = PCA better.

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
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'

# Load results
results_file = Path(__file__).parent / "results" / "by_class_results.json"
FIG_DIR = Path(__file__).parent / "figures"

with open(results_file) as f:
    data = json.load(f)

# Extract class names and accuracies
class_names = []
pca_accs = []
vqd_accs = []
deltas = []

for class_id, class_data in data['by_class'].items():
    class_names.append(class_data['class_name'])
    pca_accs.append(class_data['pca_accuracy'] * 100)
    vqd_accs.append(class_data['vqd_accuracy'] * 100)
    deltas.append(class_data['delta'] * 100)

# Convert to numpy array
deltas = np.array(deltas)

# Sort by delta for better visualization
sorted_indices = np.argsort(deltas)
class_names_sorted = [class_names[i] for i in sorted_indices]
deltas_sorted = deltas[sorted_indices]

# Color code: positive = VQD better (green), negative = PCA better (red)
colors = ['green' if d > 0 else 'red' for d in deltas_sorted]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 10))

y_pos = np.arange(len(class_names_sorted))
bars = ax.barh(y_pos, deltas_sorted, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add zero line
ax.axvline(x=0, color='black', linestyle='-', linewidth=2)

# Styling
ax.set_yticks(y_pos)
ax.set_yticklabels(class_names_sorted, fontsize=12)
ax.set_xlabel('Δ Recall (VQD - PCA) %', fontsize=16, fontweight='bold')
ax.set_title('Per-Class Performance Difference: VQD vs PCA\n(Positive = VQD Better, Negative = PCA Better)', 
            fontsize=16, fontweight='bold', pad=20)

# Add value labels on bars
for i, (bar, delta) in enumerate(zip(bars, deltas_sorted)):
    width = bar.get_width()
    label_x_pos = width + (1.5 if width > 0 else -1.5)
    ha = 'left' if width > 0 else 'right'
    
    ax.text(label_x_pos, bar.get_y() + bar.get_height()/2,
           f'{delta:.1f}%',
           ha=ha, va='center', fontsize=10, fontweight='bold',
           color='darkgreen' if width > 0 else 'darkred')

# Grid
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)
ax.set_axisbelow(True)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', alpha=0.7, edgecolor='black', label='VQD Better'),
                   Patch(facecolor='red', alpha=0.7, edgecolor='black', label='PCA Better')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=12, framealpha=0.95)

# Set x-axis limits for better visibility
max_abs = max(abs(deltas_sorted.min()), abs(deltas_sorted.max()))
ax.set_xlim(-max_abs - 10, max_abs + 10)

plt.tight_layout()

# Save
output_file = FIG_DIR / 'per_class_delta_recall.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")

# Create summary statistics
print("\n" + "="*70)
print("PER-CLASS DELTA RECALL STATISTICS")
print("="*70)

positive_deltas = deltas_sorted[deltas_sorted > 0]
negative_deltas = deltas_sorted[deltas_sorted < 0]

print(f"\nVQD wins: {len(positive_deltas)}/{len(deltas_sorted)} classes")
print(f"PCA wins: {len(negative_deltas)}/{len(deltas_sorted)} classes")
print(f"Ties: {np.sum(deltas_sorted == 0)}/{len(deltas_sorted)} classes")

print(f"\nVQD improvements:")
print(f"  Mean: +{positive_deltas.mean():.2f}%")
print(f"  Max:  +{positive_deltas.max():.2f}% ({class_names_sorted[np.argmax(deltas_sorted)]})")
print(f"  Min:  +{positive_deltas.min():.2f}%")

if len(negative_deltas) > 0:
    print(f"\nPCA improvements:")
    print(f"  Mean: {negative_deltas.mean():.2f}%")
    print(f"  Max:  {negative_deltas.max():.2f}%")
    print(f"  Min:  {negative_deltas.min():.2f}% ({class_names_sorted[np.argmin(deltas_sorted)]})")

print(f"\nOverall mean delta: {deltas_sorted.mean():.2f}%")
print(f"Overall median delta: {np.median(deltas_sorted):.2f}%")

# Top 5 VQD improvements
print(f"\nTop 5 VQD improvements:")
top5_indices = np.argsort(deltas_sorted)[-5:][::-1]
for idx in top5_indices:
    print(f"  {class_names_sorted[idx]}: +{deltas_sorted[idx]:.1f}%")

# Bottom 5 (worst VQD or best PCA)
print(f"\nBottom 5 (PCA better or VQD worst):")
bottom5_indices = np.argsort(deltas_sorted)[:5]
for idx in bottom5_indices:
    delta_sign = '+' if deltas_sorted[idx] >= 0 else ''
    print(f"  {class_names_sorted[idx]}: {delta_sign}{deltas_sorted[idx]:.1f}%")

print("\n✨ Per-class delta recall plot ready! ✨")
