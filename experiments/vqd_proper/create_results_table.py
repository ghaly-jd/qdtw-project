"""
Create Results Table with VQD Quality Metrics
==============================================

Generates publication-ready table showing:
- Method (PCA/VQD)
- K (number of components)
- Accuracy (mean ± std)
- VQD Quality (improvement over PCA)

Author: VQD-DTW Research Team
Date: November 25, 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'

# Load results
results_file = Path(__file__).parent / "results" / "k_sweep_ci_results.json"
FIG_DIR = Path(__file__).parent / "figures"

with open(results_file) as f:
    data = json.load(f)

# Prepare table data
table_data = []

for k in [6, 8, 10, 12]:
    agg = data['aggregated'][str(k)]
    
    # PCA row
    pca_mean = agg['pca']['mean'] * 100
    pca_std = agg['pca']['std'] * 100
    table_data.append({
        'Method': 'PCA',
        'K': k,
        'Accuracy (%)': f'{pca_mean:.1f} ± {pca_std:.1f}',
        'VQD Quality': '—'
    })
    
    # VQD row
    vqd_mean = agg['vqd']['mean'] * 100
    vqd_std = agg['vqd']['std'] * 100
    gap_mean = agg['gap']['mean'] * 100
    gap_std = agg['gap']['std'] * 100
    table_data.append({
        'Method': 'VQD',
        'K': k,
        'Accuracy (%)': f'{vqd_mean:.1f} ± {vqd_std:.1f}',
        'VQD Quality': f'+{gap_mean:.1f} ± {gap_std:.1f}'
    })

# Create DataFrame
df = pd.DataFrame(table_data)

# Create figure with table
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# Create table
table = ax.table(cellText=df.values, colLabels=df.columns,
                cellLoc='center', loc='center',
                colWidths=[0.20, 0.15, 0.35, 0.30])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.5)

# Header styling
for i in range(len(df.columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white', size=14)
    cell.set_height(0.08)

# Row styling - alternate colors
for i in range(1, len(df) + 1):
    method = df.iloc[i-1]['Method']
    
    if method == 'PCA':
        color = '#D9E2F3'  # Light blue
    else:  # VQD
        color = '#FCE4D6'  # Light coral
    
    for j in range(len(df.columns)):
        cell = table[(i, j)]
        cell.set_facecolor(color)
        cell.set_height(0.06)
        
        # Bold method names
        if j == 0:
            cell.set_text_props(weight='bold', size=13)
        
        # Color VQD Quality column
        if j == 3 and method == 'VQD':
            cell.set_text_props(color='darkgreen', weight='bold', size=13)

# Add borders
for key, cell in table.get_celld().items():
    cell.set_linewidth(1.5)
    cell.set_edgecolor('black')

# Title
plt.title('VQD vs PCA: Comprehensive Results Table\n(Mean ± Std across 5 seeds)', 
         fontsize=18, fontweight='bold', pad=20)

# Add footnote
fig.text(0.5, 0.02, 
        'VQD Quality: Improvement of VQD over PCA baseline (positive = VQD better)',
        ha='center', fontsize=10, style='italic', color='gray')

plt.tight_layout()

# Save
output_file = FIG_DIR / 'results_table.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")

# Also create a LaTeX table
print("\n" + "="*80)
print("LaTeX TABLE")
print("="*80)
print(r"""
\begin{table}[h]
\centering
\caption{VQD vs PCA Classification Results on MSR Action3D}
\label{tab:vqd_pca_results}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Method} & \textbf{K} & \textbf{Accuracy (\%)} & \textbf{VQD Quality} \\
\hline
""")

for _, row in df.iterrows():
    method = row['Method']
    k = row['K']
    acc = row['Accuracy (%)']
    qual = row['VQD Quality']
    
    if method == 'VQD':
        print(f"\\textbf{{{method}}} & {k} & {acc} & \\textcolor{{darkgreen}}{{{qual}}} \\\\")
    else:
        print(f"{method} & {k} & {acc} & {qual} \\\\")
    
    # Add horizontal line between k values
    if method == 'VQD' and k < 12:
        print("\\hline")

print(r"""\hline
\end{tabular}
\end{table}
""")

# Print text table
print("\n" + "="*80)
print("TEXT TABLE")
print("="*80)
print(df.to_string(index=False))

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

vqd_rows = df[df['Method'] == 'VQD']
print(f"\nVQD Improvements across all k:")
for _, row in vqd_rows.iterrows():
    print(f"  k={row['K']}: {row['VQD Quality']}")

# Extract numeric values for averages
gaps = []
for k in [6, 8, 10, 12]:
    gap = data['aggregated'][str(k)]['gap']['mean'] * 100
    gaps.append(gap)

print(f"\nAverage VQD improvement: +{np.mean(gaps):.2f}%")
print(f"Best improvement: +{np.max(gaps):.2f}% (k={[6,8,10,12][np.argmax(gaps)]})")
print(f"Min improvement: +{np.min(gaps):.2f}% (k={[6,8,10,12][np.argmin(gaps)]})")

print("\n✨ Results table generated! ✨")
