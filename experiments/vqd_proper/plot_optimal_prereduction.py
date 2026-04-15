"""
Visualize Optimal Pre-Reduction Results
========================================

Creates publication-ready figures for thesis:
1. Main plot: Pre-reduction size vs Accuracy (PCA & VQD)
2. Gap plot: VQD advantage vs Pre-reduction size
3. Combined plot with variance explained
4. Results table (LaTeX-ready)

Expected pattern: U-shaped curve for VQD advantage
- Too small: Information loss
- Sweet spot: Best balance  
- Too large: Noise retained
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set publication style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def load_results():
    """Load results from JSON."""
    results_file = Path("results/optimal_prereduction_results.json")
    with open(results_file) as f:
        return json.load(f)

def plot_accuracy_vs_predim(data, output_dir):
    """
    Plot 1: Pre-reduction size vs Accuracy
    Shows PCA and VQD curves with error bars.
    """
    pre_dims = sorted([int(k) for k in data['aggregated'].keys()])
    
    pca_means = []
    pca_stds = []
    vqd_means = []
    vqd_stds = []
    
    for pre_dim in pre_dims:
        agg = data['aggregated'][str(pre_dim)]
        pca_means.append(agg['pca']['mean'] * 100)
        pca_stds.append(agg['pca']['std'] * 100)
        vqd_means.append(agg['vqd']['mean'] * 100)
        vqd_stds.append(agg['vqd']['std'] * 100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot PCA
    ax.errorbar(pre_dims, pca_means, yerr=pca_stds,
                marker='o', markersize=8, linewidth=2,
                capsize=5, capthick=2, label='Classical PCA',
                color='#2E86AB', alpha=0.8)
    
    # Plot VQD
    ax.errorbar(pre_dims, vqd_means, yerr=vqd_stds,
                marker='s', markersize=8, linewidth=2,
                capsize=5, capthick=2, label='VQD (Quantum-Inspired)',
                color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Pre-Reduction Dimension (60D → ?D)', fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%)', fontweight='bold')
    ax.set_title('Effect of Pre-Reduction Size on PCA vs VQD Performance', 
                 fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(pre_dims)
    
    # Add optimal marker
    best_idx = np.argmax(vqd_means)
    best_pre_dim = pre_dims[best_idx]
    best_acc = vqd_means[best_idx]
    ax.annotate(f'Optimal: {best_pre_dim}D\n{best_acc:.1f}%',
                xy=(best_pre_dim, best_acc),
                xytext=(best_pre_dim-3, best_acc+2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prereduction_accuracy_curve.png', bbox_inches='tight')
    plt.savefig(output_dir / 'prereduction_accuracy_curve.pdf', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'prereduction_accuracy_curve.png'}")
    plt.close()

def plot_gap_vs_predim(data, output_dir):
    """
    Plot 2: VQD Advantage vs Pre-reduction size
    Shows the VQD-PCA gap with error bars.
    Key figure showing U-shaped curve.
    """
    pre_dims = sorted([int(k) for k in data['aggregated'].keys()])
    
    gap_means = []
    gap_stds = []
    
    for pre_dim in pre_dims:
        agg = data['aggregated'][str(pre_dim)]
        gap_means.append(agg['gap']['mean'] * 100)
        gap_stds.append(agg['gap']['std'] * 100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot gap with shaded error region
    ax.plot(pre_dims, gap_means, marker='D', markersize=10,
            linewidth=2.5, color='#F18F01', label='VQD Advantage')
    ax.fill_between(pre_dims,
                     np.array(gap_means) - np.array(gap_stds),
                     np.array(gap_means) + np.array(gap_stds),
                     alpha=0.3, color='#F18F01')
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Shade regions
    ax.axhspan(-10, 0, alpha=0.1, color='red', label='VQD worse than PCA')
    ax.axhspan(0, 10, alpha=0.1, color='green', label='VQD better than PCA')
    
    ax.set_xlabel('Pre-Reduction Dimension (60D → ?D)', fontweight='bold')
    ax.set_ylabel('VQD Advantage over PCA (%)', fontweight='bold')
    ax.set_title('VQD Performance Gain vs Pre-Reduction Dimensionality',
                 fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(pre_dims)
    
    # Mark optimal
    best_idx = np.argmax(gap_means)
    best_pre_dim = pre_dims[best_idx]
    best_gap = gap_means[best_idx]
    ax.annotate(f'Peak: {best_pre_dim}D\n{best_gap:+.2f}%',
                xy=(best_pre_dim, best_gap),
                xytext=(best_pre_dim+2, best_gap-1),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                fontsize=10, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prereduction_vqd_advantage.png', bbox_inches='tight')
    plt.savefig(output_dir / 'prereduction_vqd_advantage.pdf', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'prereduction_vqd_advantage.png'}")
    plt.close()

def plot_combined_with_variance(data, output_dir):
    """
    Plot 3: Combined plot with dual y-axis
    Left: Accuracy, Right: Variance explained
    Shows trade-off between information retention and performance.
    """
    pre_dims = sorted([int(k) for k in data['aggregated'].keys()])
    
    pca_means = [data['aggregated'][str(d)]['pca']['mean'] * 100 for d in pre_dims]
    vqd_means = [data['aggregated'][str(d)]['vqd']['mean'] * 100 for d in pre_dims]
    variances = [data['aggregated'][str(d)]['variance_explained'] * 100 for d in pre_dims]
    gap_means = [data['aggregated'][str(d)]['gap']['mean'] * 100 for d in pre_dims]
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Left y-axis: Accuracy
    color1 = '#2E86AB'
    color2 = '#A23B72'
    ax1.set_xlabel('Pre-Reduction Dimension (60D → ?D)', fontweight='bold')
    ax1.set_ylabel('Classification Accuracy (%)', fontweight='bold', color='black')
    
    line1 = ax1.plot(pre_dims, pca_means, marker='o', markersize=8,
                     linewidth=2, label='PCA', color=color1, alpha=0.8)
    line2 = ax1.plot(pre_dims, vqd_means, marker='s', markersize=8,
                     linewidth=2, label='VQD', color=color2, alpha=0.8)
    
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(pre_dims)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Right y-axis: Variance explained
    ax2 = ax1.twinx()
    color3 = '#6B7A8F'
    ax2.set_ylabel('Variance Explained (%)', fontweight='bold', color=color3)
    line3 = ax2.plot(pre_dims, variances, marker='^', markersize=8,
                     linewidth=2, linestyle=':', label='Variance Explained',
                     color=color3, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color3)
    ax2.set_ylim([80, 100])
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', framealpha=0.95)
    
    ax1.set_title('Pre-Reduction Trade-off: Information vs Performance',
                  fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prereduction_combined_analysis.png', bbox_inches='tight')
    plt.savefig(output_dir / 'prereduction_combined_analysis.pdf', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'prereduction_combined_analysis.png'}")
    plt.close()

def plot_gap_with_annotations(data, output_dir):
    """
    Plot 4: VQD advantage with region annotations
    Clearly shows information loss vs noise retention zones.
    """
    pre_dims = sorted([int(k) for k in data['aggregated'].keys()])
    gap_means = [data['aggregated'][str(d)]['gap']['mean'] * 100 for d in pre_dims]
    gap_stds = [data['aggregated'][str(d)]['gap']['std'] * 100 for d in pre_dims]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot with error bars
    ax.errorbar(pre_dims, gap_means, yerr=gap_stds,
                marker='D', markersize=10, linewidth=2.5,
                capsize=5, capthick=2, color='#F18F01',
                ecolor='#F18F01', alpha=0.8)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Annotate regions
    mid_point = (max(pre_dims) + min(pre_dims)) / 2
    
    # Information loss region (left)
    ax.annotate('Information Loss Zone\n(Too aggressive reduction)',
                xy=(min(pre_dims), max(gap_means) * 0.8),
                fontsize=10, ha='left', style='italic',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE5E5', alpha=0.7))
    
    # Sweet spot (middle)
    best_idx = np.argmax(gap_means)
    best_pre_dim = pre_dims[best_idx]
    ax.annotate('Optimal Zone\n(Best balance)',
                xy=(best_pre_dim, gap_means[best_idx]),
                xytext=(best_pre_dim, gap_means[best_idx] + 1.5),
                fontsize=11, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#E5FFE5', alpha=0.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))
    
    # Noise retention (right)
    ax.annotate('Noise Retention Zone\n(Insufficient reduction)',
                xy=(max(pre_dims), max(gap_means) * 0.6),
                fontsize=10, ha='right', style='italic',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF5E5', alpha=0.7))
    
    ax.set_xlabel('Pre-Reduction Dimension', fontweight='bold', fontsize=13)
    ax.set_ylabel('VQD Advantage over PCA (%)', fontweight='bold', fontsize=13)
    ax.set_title('Understanding Pre-Reduction Trade-offs for Quantum-Inspired Subspace Learning',
                 fontweight='bold', pad=20, fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(pre_dims)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prereduction_annotated_analysis.png', bbox_inches='tight')
    plt.savefig(output_dir / 'prereduction_annotated_analysis.pdf', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'prereduction_annotated_analysis.png'}")
    plt.close()

def generate_latex_table(data, output_dir):
    """
    Generate LaTeX table for thesis.
    """
    pre_dims = sorted([int(k) for k in data['aggregated'].keys()])
    
    latex = []
    latex.append(r"\begin{table}[ht]")
    latex.append(r"\centering")
    latex.append(r"\caption{Effect of Pre-Reduction Dimensionality on VQD Performance}")
    latex.append(r"\label{tab:prereduction}")
    latex.append(r"\begin{tabular}{c c c c c c}")
    latex.append(r"\hline")
    latex.append(r"Pre-Dim & Variance & PCA Acc. & VQD Acc. & Gap & Best \\")
    latex.append(r"        & Explained & (\%)     & (\%)     & (\%) & \\")
    latex.append(r"\hline")
    
    best_idx = np.argmax([data['aggregated'][str(d)]['gap']['mean'] for d in pre_dims])
    
    for i, pre_dim in enumerate(pre_dims):
        agg = data['aggregated'][str(pre_dim)]
        
        var = agg['variance_explained'] * 100
        pca_m = agg['pca']['mean'] * 100
        pca_s = agg['pca']['std'] * 100
        vqd_m = agg['vqd']['mean'] * 100
        vqd_s = agg['vqd']['std'] * 100
        gap_m = agg['gap']['mean'] * 100
        gap_s = agg['gap']['std'] * 100
        
        best_marker = r"$\checkmark$" if i == best_idx else ""
        
        latex.append(f"{pre_dim} & {var:.1f} & "
                    f"{pca_m:.1f}$\\pm${pca_s:.1f} & "
                    f"{vqd_m:.1f}$\\pm${vqd_s:.1f} & "
                    f"{gap_m:+.1f}$\\pm${gap_s:.1f} & "
                    f"{best_marker} \\\\")
    
    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    latex_str = "\n".join(latex)
    
    output_file = output_dir / "prereduction_table.tex"
    with open(output_file, 'w') as f:
        f.write(latex_str)
    
    print(f"✓ Saved LaTeX table: {output_file}")
    return latex_str

def print_summary(data):
    """Print text summary."""
    pre_dims = sorted([int(k) for k in data['aggregated'].keys()])
    
    print("\n" + "="*80)
    print("OPTIMAL PRE-REDUCTION ANALYSIS SUMMARY")
    print("="*80)
    
    print("\nResults Table:")
    print("-"*80)
    print(f"{'Pre-Dim':<10} {'Variance':<12} {'PCA':<15} {'VQD':<15} {'Gap':<15}")
    print("-"*80)
    
    for pre_dim in pre_dims:
        agg = data['aggregated'][str(pre_dim)]
        var = agg['variance_explained'] * 100
        pca_m = agg['pca']['mean'] * 100
        pca_s = agg['pca']['std'] * 100
        vqd_m = agg['vqd']['mean'] * 100
        vqd_s = agg['vqd']['std'] * 100
        gap_m = agg['gap']['mean'] * 100
        gap_s = agg['gap']['std'] * 100
        
        print(f"{pre_dim:<10} {var:>5.1f}%      "
              f"{pca_m:.1f}±{pca_s:.1f}%     "
              f"{vqd_m:.1f}±{vqd_s:.1f}%     "
              f"{gap_m:+.1f}±{gap_s:.1f}%")
    
    print("-"*80)
    
    # Find optimal
    gaps = [data['aggregated'][str(d)]['gap']['mean'] * 100 for d in pre_dims]
    best_idx = np.argmax(gaps)
    best_pre_dim = pre_dims[best_idx]
    best_gap = gaps[best_idx]
    best_vqd = data['aggregated'][str(best_pre_dim)]['vqd']['mean'] * 100
    
    print(f"\n🏆 OPTIMAL: {best_pre_dim}D")
    print(f"   VQD Advantage: {best_gap:+.2f}%")
    print(f"   VQD Accuracy: {best_vqd:.2f}%")
    print(f"   Variance Retained: {data['aggregated'][str(best_pre_dim)]['variance_explained']*100:.2f}%")
    
    # Analysis
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    if best_pre_dim == min(pre_dims):
        print(f"\n⚠️  Optimal at minimum ({best_pre_dim}D) - may need to test smaller dims")
    elif best_pre_dim == max(pre_dims):
        print(f"\n⚠️  Optimal at maximum ({best_pre_dim}D) - may need to test larger dims")
    else:
        print(f"\n✓ Clear optimum found at {best_pre_dim}D (interior point)")
        print(f"  • Smaller dims ({min(pre_dims)}-{best_pre_dim-1}): Information loss")
        print(f"  • Larger dims ({best_pre_dim+1}-{max(pre_dims)}): Noise retention")
    
    print("\nRecommendation for thesis:")
    print(f"  Use 60D → {best_pre_dim}D → 8D pipeline for optimal VQD performance")
    print("="*80)

if __name__ == "__main__":
    print("Loading results...")
    data = load_results()
    
    # Create output directory
    output_dir = Path("figures/prereduction_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating figures...")
    print("-" * 50)
    
    # Generate all plots
    plot_accuracy_vs_predim(data, output_dir)
    plot_gap_vs_predim(data, output_dir)
    plot_combined_with_variance(data, output_dir)
    plot_gap_with_annotations(data, output_dir)
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    print("-" * 50)
    latex_table = generate_latex_table(data, output_dir)
    print("\nLaTeX Table:")
    print(latex_table)
    
    # Print summary
    print_summary(data)
    
    print("\n" + "="*80)
    print("✨ All figures and tables generated! ✨")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  • prereduction_accuracy_curve.png/pdf")
    print("  • prereduction_vqd_advantage.png/pdf")
    print("  • prereduction_combined_analysis.png/pdf")
    print("  • prereduction_annotated_analysis.png/pdf")
    print("  • prereduction_table.tex")
    print("\nReady for thesis! 🎓")
