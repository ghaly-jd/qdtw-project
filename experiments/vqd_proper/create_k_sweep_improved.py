"""
Improved K-Sweep Plot
- Reads `results/k_sweep_ci_results.json`
- Handles NaN tokens robustly
- Produces high-resolution mean ± std errorbar plot (percent), and gap bar plot
- Saves PNG and PDF to `thesis_figures/03_k_sweep_improved.(png|pdf)`
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300

RESULTS_FILE = Path(__file__).parent / 'results' / 'k_sweep_ci_results.json'
OUT_DIR = Path(__file__).parent.parent / 'thesis_figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Robust JSON loader that replaces bare NaN with null
text = RESULTS_FILE.read_text()
# Replace 'NaN' tokens (unquoted) with null to make valid JSON
text = text.replace('NaN', 'null')

data = json.loads(text)

# Gather accuracies from by_seed structure
k_values = [6, 8, 10, 12]
vqd_accs = {k: [] for k in k_values}
pca_accs = {k: [] for k in k_values}

by_seed = data.get('by_seed', {})
for seed_key, seed_data in by_seed.items():
    for k_str, entry in seed_data.items():
        try:
            k = int(k_str)
        except Exception:
            continue
        if k not in k_values:
            continue
        pca_acc = entry.get('pca', {}).get('accuracy')
        vqd_acc = entry.get('vqd', {}).get('accuracy')
        if pca_acc is not None:
            pca_accs[k].append(pca_acc * 100.0)
        if vqd_acc is not None:
            vqd_accs[k].append(vqd_acc * 100.0)

# Compute aggregated stats
pca_means = np.array([np.mean(pca_accs[k]) for k in k_values])
pca_stds = np.array([np.std(pca_accs[k], ddof=1) for k in k_values])
vqd_means = np.array([np.mean(vqd_accs[k]) for k in k_values])
vqd_stds = np.array([np.std(vqd_accs[k], ddof=1) for k in k_values])
gap_means = vqd_means - pca_means

# Compute 95% CI using t-distribution
import scipy.stats as st
n = len(next(iter(pca_accs.values()))) if len(pca_accs)>0 else 1
ci_mult = st.t.ppf(0.975, df=max(n-1,1)) / np.sqrt(max(n,1))

pca_cis = ci_mult * pca_stds
vqd_cis = ci_mult * vqd_stds

# Create figure: left = errorbar, right = gap bar with CI
fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios':[2,1]})

ax = axes[0]
ax.errorbar(k_values, pca_means, yerr=pca_stds, fmt='o-', label='Classical PCA',
            color='#A23B72', ecolor='#A23B72', capsize=6, linewidth=2, markersize=8)
ax.errorbar(k_values, vqd_means, yerr=vqd_stds, fmt='s-', label='VQD',
            color='#2E86AB', ecolor='#2E86AB', capsize=6, linewidth=2, markersize=8)

ax.set_xlabel('Target dimension (k)', fontsize=12)
ax.set_ylabel('Classification accuracy (%)', fontsize=12)
ax.set_title('K-sweep: VQD vs PCA (mean ± std; n={})'.format(n), fontsize=13, weight='bold')
ax.set_xticks(k_values)
ax.grid(True, alpha=0.25, axis='y')
ax.legend()
ax.set_ylim(60, 92)

# Annotate mean values
for i, k in enumerate(k_values):
    ax.text(k, vqd_means[i]+1.2, f'{vqd_means[i]:.1f}%', ha='center', fontsize=9, color='#2E86AB')
    ax.text(k, pca_means[i]-1.8, f'{pca_means[i]:.1f}%', ha='center', fontsize=9, color='#A23B72')

# Right panel: gap bars with 95% CI computed from differences across seeds if available
ax2 = axes[1]
# If we have per-seed gaps, compute std of gaps
gaps_per_k = []
for k in k_values:
    # compute gap per seed by pairing lists
    p_list = pca_accs[k]
    v_list = vqd_accs[k]
    if len(p_list) == len(v_list) and len(p_list) > 0:
        gaps_per_k.append(np.array(v_list) - np.array(p_list))
    else:
        gaps_per_k.append(np.array(vqd_means[k_values.index(k)] - pca_means[k_values.index(k)]))

gap_means = np.array([np.mean(g) if hasattr(g, 'mean') else g for g in gaps_per_k])
# compute CIs for gaps
gap_stds = np.array([np.std(g, ddof=1) if hasattr(g, '__len__') and len(g)>1 else 0.0 for g in gaps_per_k])
ci_gap = ci_mult * gap_stds / np.sqrt(np.maximum(1, np.array([len(g) if hasattr(g,'__len__') else 1 for g in gaps_per_k])))

bars = ax2.bar(k_values, gap_means, color='#F18F01', edgecolor='black', alpha=0.9)
ax2.errorbar(k_values, gap_means, yerr=ci_gap, fmt='none', ecolor='black', capsize=6)
ax2.axhline(0, color='k', linestyle='--', alpha=0.6)
ax2.set_xlabel('k', fontsize=12)
ax2.set_ylabel('VQD advantage (%)', fontsize=12)
ax2.set_title('VQD Advantage (mean ± 95% CI)', fontsize=13, weight='bold')
ax2.set_xticks(k_values)
ax2.grid(True, alpha=0.25, axis='y')

# Highlight optimal k=8 if present
if 8 in k_values:
    idx = k_values.index(8)
    bars[idx].set_color('#D7263D')

# Label gaps
for bar, g in zip(bars, gap_means):
    h = bar.get_height()
    ax2.text(bar.get_x()+bar.get_width()/2., h + 0.25, f'{g:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

out_png = OUT_DIR / '03_k_sweep_improved.png'
out_pdf = OUT_DIR / '03_k_sweep_improved.pdf'
fig.savefig(out_png, bbox_inches='tight', dpi=300)
fig.savefig(out_pdf, bbox_inches='tight')
print(f'✓ Saved: {out_png}\n✓ Saved: {out_pdf}')

print('\nK-sweep improved plot created successfully.')
