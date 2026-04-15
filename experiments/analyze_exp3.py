"""
Visualization and Analysis for Experiment 3
Generate publication-ready plots and LaTeX tables
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd


def load_exp3_results(results_dir: str = "results/exp3_vqd_quantum_pca"):
    """Load Experiment 3 results"""
    results_path = Path(results_dir) / "results_summary.json"
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_pareto_accuracy_vs_runtime(results: dict, save_path: str = None):
    """
    Pareto plot: Accuracy vs Runtime
    Shows PCA and VQD tradeoffs
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pca_data = []
    vqd_data = []
    
    for key, res in results.items():
        if 'vqd' not in res or 'error' in res['vqd']:
            continue
        
        k = res['k']
        
        # PCA
        acc_pca = res['pca']['accuracy']['mean']
        time_pca = res['runtime_per_query_ms']['pca']['total']
        pca_data.append((k, time_pca, acc_pca))
        
        # VQD
        acc_vqd = res['vqd']['accuracy']['mean']
        time_vqd = res['runtime_per_query_ms']['vqd']['total']
        vqd_data.append((k, time_vqd, acc_vqd))
    
    pca_data = np.array(pca_data)
    vqd_data = np.array(vqd_data)
    
    # Sort by k
    pca_data = pca_data[pca_data[:, 0].argsort()]
    vqd_data = vqd_data[vqd_data[:, 0].argsort()]
    
    # Plot
    ax.plot(pca_data[:, 1], pca_data[:, 2], 'o-', 
            label='Classical PCA', markersize=8, linewidth=2, color='blue')
    ax.plot(vqd_data[:, 1], vqd_data[:, 2], 's-', 
            label='VQD Quantum PCA', markersize=8, linewidth=2, color='red')
    
    # Annotate k values
    for i, (k, time, acc) in enumerate(pca_data):
        ax.annotate(f'k={int(k)}', (time, acc), 
                   textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    ax.set_xlabel('Runtime per Query (ms)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Pareto Frontier: Accuracy vs Runtime\n(PCA vs VQD Quantum PCA)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    return fig


def plot_angles_vs_k(results: dict, save_path: str = None):
    """
    Plot principal angles (before and after Procrustes) vs k
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    k_values = []
    angles_before_mean = []
    angles_before_max = []
    angles_after_mean = []
    angles_after_max = []
    
    for key in sorted(results.keys()):
        res = results[key]
        if 'vqd' not in res or 'error' in res['vqd']:
            continue
        
        k = res['k']
        k_values.append(k)
        
        angles_before_mean.append(res['vqd']['angles_before_procrustes']['mean'])
        angles_before_max.append(res['vqd']['angles_before_procrustes']['max'])
        
        angles_after_mean.append(res['vqd_procrustes']['angles_after_procrustes']['mean'])
        angles_after_max.append(res['vqd_procrustes']['angles_after_procrustes']['max'])
    
    k_values = np.array(k_values)
    
    # Before Procrustes
    ax1.plot(k_values, angles_before_mean, 'o-', label='Mean', markersize=8, linewidth=2)
    ax1.plot(k_values, angles_before_max, 's-', label='Max', markersize=8, linewidth=2)
    ax1.set_xlabel('k (subspace dimension)', fontsize=12)
    ax1.set_ylabel('Principal Angle (degrees)', fontsize=12)
    ax1.set_title('Before Procrustes Alignment', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # After Procrustes
    ax2.plot(k_values, angles_after_mean, 'o-', label='Mean', markersize=8, linewidth=2)
    ax2.plot(k_values, angles_after_max, 's-', label='Max', markersize=8, linewidth=2)
    ax2.set_xlabel('k (subspace dimension)', fontsize=12)
    ax2.set_ylabel('Principal Angle (degrees)', fontsize=12)
    ax2.set_title('After Procrustes Alignment', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    return fig


def plot_procrustes_improvement(results: dict, save_path: str = None):
    """
    Plot Procrustes residual drop percentage vs k
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    k_values = []
    residual_drops = []
    
    for key in sorted(results.keys()):
        res = results[key]
        if 'vqd_procrustes' not in res:
            continue
        
        k = res['k']
        drop = res['vqd_procrustes']['residual_drop_pct']
        
        k_values.append(k)
        residual_drops.append(drop)
    
    ax.bar(k_values, residual_drops, color='steelblue', alpha=0.7)
    ax.set_xlabel('k (subspace dimension)', fontsize=12)
    ax.set_ylabel('Residual Drop (%)', fontsize=12)
    ax.set_title('Procrustes Alignment Improvement\n(Residual Drop Percentage)', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    return fig


def generate_latex_table(results: dict, save_path: str = None):
    """
    Generate LaTeX table for thesis
    """
    rows = []
    
    for key in sorted(results.keys()):
        res = results[key]
        if 'vqd' not in res or 'error' in res['vqd']:
            continue
        
        k = res['k']
        
        # Accuracy
        acc_pca = res['pca']['accuracy']
        acc_vqd = res['vqd']['accuracy']
        acc_proc = res['vqd_procrustes']['accuracy']
        
        # Angles
        angles_mean = res['vqd']['angles_before_procrustes']['mean']
        angles_max = res['vqd']['angles_before_procrustes']['max']
        
        # Procrustes
        proc_drop = res['vqd_procrustes']['residual_drop_pct']
        
        # Runtime
        time_pca = res['runtime_per_query_ms']['pca']
        time_vqd = res['runtime_per_query_ms']['vqd']
        
        # Speedup
        speedup_pca = res['speedup_vs_60D']['pca']
        speedup_vqd = res['speedup_vs_60D']['vqd']
        
        # McNemar
        mcnemar = res['statistical_tests']['mcnemar_vqd_vs_pca']
        sig = '*' if mcnemar['significant'] else ''
        
        row = {
            'k': k,
            'pca_acc': f"{acc_pca['mean']:.3f}",
            'pca_ci': f"[{acc_pca['ci_lower']:.2f}, {acc_pca['ci_upper']:.2f}]",
            'vqd_acc': f"{acc_vqd['mean']:.3f}",
            'vqd_ci': f"[{acc_vqd['ci_lower']:.2f}, {acc_vqd['ci_upper']:.2f}]",
            'proc_acc': f"{acc_proc['mean']:.3f}",
            'proc_ci': f"[{acc_proc['ci_lower']:.2f}, {acc_proc['ci_upper']:.2f}]",
            'angles': f"{angles_mean:.1f}/{angles_max:.1f}",
            'proc_drop': f"{proc_drop:.1f}",
            'pca_time': f"{time_pca['projection']:.2f}/{time_pca['dtw']:.2f}",
            'vqd_time': f"{time_vqd['projection']:.2f}/{time_vqd['dtw']:.2f}",
            'speedup_pca': f"{speedup_pca:.2f}",
            'speedup_vqd': f"{speedup_vqd:.2f}",
            'significant': sig
        }
        rows.append(row)
    
    # LaTeX table
    latex = r"""\begin{table}[htbp]
\centering
\caption{VQD Quantum PCA vs Classical PCA: Comprehensive Comparison}
\label{tab:vqd_vs_pca}
\begin{tabular}{c|ccc|cc|c|cc|cc}
\hline
\multirow{2}{*}{$k$} & \multicolumn{3}{c|}{Accuracy (mean [CI])} & \multicolumn{2}{c|}{Angles} & Proc & \multicolumn{2}{c|}{Time (ms)} & \multicolumn{2}{c}{Speedup} \\
& PCA & VQD & VQD+Proc & Mean/Max & Drop\% & & PCA & VQD & PCA & VQD \\
\hline
"""
    
    for row in rows:
        latex += f"{row['k']} & "
        latex += f"{row['pca_acc']} {row['pca_ci']} & "
        latex += f"{row['vqd_acc']}{row['significant']} {row['vqd_ci']} & "
        latex += f"{row['proc_acc']} {row['proc_ci']} & "
        latex += f"{row['angles']}° & "
        latex += f"{row['proc_drop']}\% & "
        latex += f" & "  # Empty column
        latex += f"{row['pca_time']} & "
        latex += f"{row['vqd_time']} & "
        latex += f"{row['speedup_pca']}× & "
        latex += f"{row['speedup_vqd']}× \\\\\n"
    
    latex += r"""\hline
\end{tabular}
\caption*{Time format: projection/DTW. * indicates significant difference (McNemar test, $p<0.05$).}
\end{table}
"""
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"✅ Saved LaTeX table: {save_path}")
    
    print("\n" + latex)
    
    return latex


def main():
    """Generate all visualizations for Experiment 3"""
    print("=" * 80)
    print("EXPERIMENT 3: VISUALIZATION & ANALYSIS")
    print("=" * 80)
    
    # Load results
    print("\n📊 Loading results...")
    results = load_exp3_results()
    
    # Create figures directory
    fig_dir = Path("results/exp3_vqd_quantum_pca/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n📈 Generating plots...")
    
    print("  1. Pareto: Accuracy vs Runtime")
    plot_pareto_accuracy_vs_runtime(
        results,
        save_path=fig_dir / "pareto_accuracy_vs_runtime.png"
    )
    
    print("  2. Principal Angles vs k")
    plot_angles_vs_k(
        results,
        save_path=fig_dir / "angles_vs_k.png"
    )
    
    print("  3. Procrustes Improvement")
    plot_procrustes_improvement(
        results,
        save_path=fig_dir / "procrustes_improvement.png"
    )
    
    # Generate LaTeX table
    print("\n📝 Generating LaTeX table...")
    generate_latex_table(
        results,
        save_path=fig_dir / "table_vqd_vs_pca.tex"
    )
    
    print(f"\n✅ All visualizations saved to {fig_dir}")


if __name__ == "__main__":
    main()
