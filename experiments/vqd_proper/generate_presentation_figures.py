"""
Generate Publication Figures for VQD-DTW Presentation
======================================================

Creates high-quality figures for PowerPoint presentation including:
1. PCA vs VQD comparison diagrams
2. Results visualization (accuracy bars, confidence intervals)
3. By-class performance heatmap
4. Quantum circuit diagram
5. Pipeline flowchart

Author: VQD-DTW Research Team
Date: November 25, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import json
from pathlib import Path
import seaborn as sns

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Output directory
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

def create_pca_diagram():
    """Create diagram showing Classical PCA concept."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    
    # Title
    ax.text(5, 5.5, 'Classical PCA: Eigenvectors of Covariance', 
            ha='center', fontsize=18, fontweight='bold')
    
    # Data matrix
    data_box = FancyBboxPatch((0.5, 3), 1.5, 1.5, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(data_box)
    ax.text(1.25, 3.75, 'Data\nX', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(1.25, 2.7, '(n × d)', ha='center', fontsize=10)
    
    # Arrow 1
    arrow1 = FancyArrowPatch((2.2, 3.75), (3.3, 3.75),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    ax.text(2.75, 4.0, '1. Compute', ha='center', fontsize=10)
    
    # Covariance matrix
    cov_box = FancyBboxPatch((3.5, 3), 1.5, 1.5,
                            boxstyle="round,pad=0.1",
                            edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(cov_box)
    ax.text(4.25, 3.75, 'Cov\nC', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(4.25, 2.7, r'$C = \frac{1}{n}X^TX$', ha='center', fontsize=10)
    
    # Arrow 2
    arrow2 = FancyArrowPatch((5.2, 3.75), (6.3, 3.75),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    ax.text(5.75, 4.0, '2. Eigendecomp', ha='center', fontsize=10)
    
    # Eigenvectors
    eigen_box = FancyBboxPatch((6.5, 3), 1.5, 1.5,
                              boxstyle="round,pad=0.1",
                              edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax.add_patch(eigen_box)
    ax.text(7.25, 3.75, 'PCs\nU', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(7.25, 2.7, r'$C = U\Lambda U^T$', ha='center', fontsize=10)
    
    # Properties box
    props_box = FancyBboxPatch((0.5, 0.5), 7.5, 1.8,
                              boxstyle="round,pad=0.1",
                              edgecolor='purple', facecolor='lavender', linewidth=2)
    ax.add_patch(props_box)
    
    ax.text(4.25, 2.0, 'Properties:', ha='center', fontsize=12, fontweight='bold')
    ax.text(1, 1.5, '• Maximize variance', ha='left', fontsize=11)
    ax.text(1, 1.1, '• Orthogonal components', ha='left', fontsize=11)
    ax.text(1, 0.7, '• Deterministic (closed-form)', ha='left', fontsize=11)
    
    ax.text(5.5, 1.5, '• Unsupervised', ha='left', fontsize=11)
    ax.text(5.5, 1.1, '• Optimal for Gaussian data', ha='left', fontsize=11)
    ax.text(5.5, 0.7, r'• Complexity: $O(d^3)$', ha='left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'pca_diagram.png', bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {FIG_DIR / 'pca_diagram.png'}")
    plt.close()

def create_vqd_diagram():
    """Create diagram showing VQD concept."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    
    # Title
    ax.text(5, 5.5, 'VQD: Variational Quantum Deflation', 
            ha='center', fontsize=18, fontweight='bold')
    
    # Data matrix
    data_box = FancyBboxPatch((0.5, 3), 1.5, 1.5,
                              boxstyle="round,pad=0.1",
                              edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(data_box)
    ax.text(1.25, 3.75, 'Data\nX', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(1.25, 2.7, '(n × d)', ha='center', fontsize=10)
    
    # Arrow 1
    arrow1 = FancyArrowPatch((2.2, 3.75), (3.3, 3.75),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    ax.text(2.75, 4.0, '1. Encode', ha='center', fontsize=10)
    
    # Hamiltonian
    ham_box = FancyBboxPatch((3.5, 3), 1.5, 1.5,
                            boxstyle="round,pad=0.1",
                            edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(ham_box)
    ax.text(4.25, 3.75, 'H\n', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(4.25, 2.7, r'$H = -C$', ha='center', fontsize=10)
    
    # Arrow 2
    arrow2 = FancyArrowPatch((5.2, 3.75), (6.3, 3.75),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    ax.text(5.75, 4.0, '2. VQD', ha='center', fontsize=10)
    
    # Quantum circuit
    circuit_box = FancyBboxPatch((6.5, 3), 1.5, 1.5,
                                boxstyle="round,pad=0.1",
                                edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax.add_patch(circuit_box)
    ax.text(7.25, 3.75, 'Circuit\n|ψ⟩', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(7.25, 2.7, r'$\theta$ params', ha='center', fontsize=10)
    
    # Arrow back (optimization)
    arrow3 = FancyArrowPatch((7.25, 2.8), (7.25, 2.2),
                            arrowstyle='<->', mutation_scale=20, linewidth=2, 
                            color='purple', linestyle='dashed')
    ax.add_patch(arrow3)
    ax.text(7.7, 2.5, 'Optimize', ha='left', fontsize=9, color='purple')
    
    # Properties box
    props_box = FancyBboxPatch((0.5, 0.5), 7.5, 1.8,
                              boxstyle="round,pad=0.1",
                              edgecolor='purple', facecolor='lavender', linewidth=2)
    ax.add_patch(props_box)
    
    ax.text(4.25, 2.0, 'Properties:', ha='center', fontsize=12, fontweight='bold')
    ax.text(1, 1.5, r'• Minimize $\langle\psi|H|\psi\rangle$', ha='left', fontsize=11)
    ax.text(1, 1.1, '• Orthogonality penalties', ha='left', fontsize=11)
    ax.text(1, 0.7, '• Stochastic (iterative)', ha='left', fontsize=11)
    
    ax.text(5.5, 1.5, '• Quantum circuit structure', ha='left', fontsize=11)
    ax.text(5.5, 1.1, '• Explores different subspaces', ha='left', fontsize=11)
    ax.text(5.5, 0.7, '• 4 qubits, depth 2', ha='left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'vqd_diagram.png', bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {FIG_DIR / 'vqd_diagram.png'}")
    plt.close()

def create_comparison_diagram():
    """Create side-by-side PCA vs VQD comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # PCA side
    ax1.axis('off')
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 6)
    ax1.text(2.5, 5.5, 'Classical PCA', ha='center', fontsize=16, fontweight='bold', color='blue')
    
    # PCA boxes
    pca_cov = FancyBboxPatch((1.25, 4), 2.5, 0.8, boxstyle="round,pad=0.05",
                             edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax1.add_patch(pca_cov)
    ax1.text(2.5, 4.4, r'$C = \frac{1}{n}X^TX$', ha='center', fontsize=12)
    
    pca_eigen = FancyBboxPatch((1.25, 2.8), 2.5, 0.8, boxstyle="round,pad=0.05",
                               edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax1.add_patch(pca_eigen)
    ax1.text(2.5, 3.2, r'$C = U\Lambda U^T$', ha='center', fontsize=12)
    
    pca_result = FancyBboxPatch((1.25, 1.6), 2.5, 0.8, boxstyle="round,pad=0.05",
                                edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax1.add_patch(pca_result)
    ax1.text(2.5, 2.0, 'Eigenvectors U', ha='center', fontsize=12)
    
    # PCA properties
    ax1.text(2.5, 1.0, '✓ Deterministic', ha='center', fontsize=10)
    ax1.text(2.5, 0.7, '✓ Optimal variance', ha='center', fontsize=10)
    ax1.text(2.5, 0.4, r'✓ Fast: $O(d^3)$', ha='center', fontsize=10)
    
    # Arrows
    ax1.arrow(2.5, 3.8, 0, -0.7, head_width=0.2, head_length=0.1, fc='blue', ec='blue')
    ax1.arrow(2.5, 2.6, 0, -0.7, head_width=0.2, head_length=0.1, fc='blue', ec='blue')
    
    # VQD side
    ax2.axis('off')
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 6)
    ax2.text(2.5, 5.5, 'VQD (Quantum)', ha='center', fontsize=16, fontweight='bold', color='red')
    
    # VQD boxes
    vqd_ham = FancyBboxPatch((1.25, 4), 2.5, 0.8, boxstyle="round,pad=0.05",
                             edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax2.add_patch(vqd_ham)
    ax2.text(2.5, 4.4, r'$H = -C$', ha='center', fontsize=12)
    
    vqd_circuit = FancyBboxPatch((1.25, 2.8), 2.5, 0.8, boxstyle="round,pad=0.05",
                                 edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax2.add_patch(vqd_circuit)
    ax2.text(2.5, 3.2, r'$|\psi(\theta)\rangle$ (4 qubits)', ha='center', fontsize=12)
    
    vqd_result = FancyBboxPatch((1.25, 1.6), 2.5, 0.8, boxstyle="round,pad=0.05",
                                edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax2.add_patch(vqd_result)
    ax2.text(2.5, 2.0, 'Eigenvectors U', ha='center', fontsize=12)
    
    # VQD properties
    ax2.text(2.5, 1.0, '✓ Stochastic', ha='center', fontsize=10)
    ax2.text(2.5, 0.7, '✓ Different subspaces', ha='center', fontsize=10)
    ax2.text(2.5, 0.4, '✓ Quantum exploration', ha='center', fontsize=10)
    
    # Arrows
    ax2.arrow(2.5, 3.8, 0, -0.7, head_width=0.2, head_length=0.1, fc='red', ec='red')
    ax2.arrow(2.5, 2.6, 0, -0.7, head_width=0.2, head_length=0.1, fc='red', ec='red')
    
    # Optimization loop
    ax2.annotate('', xy=(3.8, 3.2), xytext=(3.8, 2.4),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2, linestyle='dashed'))
    ax2.text(4.2, 2.8, 'Optimize\n200 iter', ha='left', fontsize=9, color='purple')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'pca_vqd_comparison.png', bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {FIG_DIR / 'pca_vqd_comparison.png'}")
    plt.close()

def create_results_barplot():
    """Create bar plot of k-sweep results with error bars."""
    # Load results
    results_file = Path(__file__).parent / "results" / "k_sweep_ci_results.json"
    with open(results_file) as f:
        data = json.load(f)
    
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
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(k_values))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pca_means, width, yerr=pca_stds,
                   label='PCA', color='steelblue', alpha=0.8,
                   capsize=5, error_kw={'linewidth': 2})
    bars2 = ax.bar(x + width/2, vqd_means, width, yerr=vqd_stds,
                   label='VQD', color='coral', alpha=0.8,
                   capsize=5, error_kw={'linewidth': 2})
    
    ax.set_xlabel('Dimension (k)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('VQD vs PCA: Accuracy Across Dimensions\n(Mean ± Std, 5 seeds)', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(65, 90)
    
    # Add value labels on bars
    for bars, means in [(bars1, pca_means), (bars2, vqd_means)]:
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{mean:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'accuracy_comparison.png', bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {FIG_DIR / 'accuracy_comparison.png'}")
    plt.close()

def create_gap_plot():
    """Create plot showing VQD-PCA gap with confidence intervals."""
    # Load results
    results_file = Path(__file__).parent / "results" / "k_sweep_ci_results.json"
    with open(results_file) as f:
        data = json.load(f)
    
    k_values = [6, 8, 10, 12]
    gaps = []
    ci95 = []
    
    for k in k_values:
        agg = data['aggregated'][str(k)]
        gaps.append(agg['gap']['mean'] * 100)
        ci95.append(agg['gap']['ci95'] * 100)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(k_values, gaps, yerr=ci95, fmt='o-', linewidth=3, markersize=10,
               color='green', ecolor='darkgreen', capsize=8, capthick=2,
               label='VQD Advantage (95% CI)')
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='No difference')
    ax.fill_between(k_values, 0, gaps, alpha=0.2, color='green')
    
    ax.set_xlabel('Dimension (k)', fontsize=14, fontweight='bold')
    ax.set_ylabel('VQD - PCA Gap (%)', fontsize=14, fontweight='bold')
    ax.set_title('VQD Advantage Over PCA\n(with 95% Confidence Intervals)', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    
    # Add value labels
    for k, gap, ci in zip(k_values, gaps, ci95):
        ax.text(k, gap + 0.5, f'+{gap:.1f}%\n±{ci:.1f}%', 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'vqd_advantage.png', bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {FIG_DIR / 'vqd_advantage.png'}")
    plt.close()

def create_circuit_diagram():
    """Create simplified quantum circuit diagram."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # Title
    ax.text(6, 4.5, 'VQD Quantum Circuit (4 qubits, depth 2)', 
           ha='center', fontsize=16, fontweight='bold')
    
    # Qubit lines
    qubit_labels = ['q₀:', 'q₁:', 'q₂:', 'q₃:']
    y_positions = [3.5, 2.5, 1.5, 0.5]
    
    for i, (label, y) in enumerate(zip(qubit_labels, y_positions)):
        ax.text(0.5, y, label, ha='right', va='center', fontsize=12)
        ax.plot([1, 11], [y, y], 'k-', linewidth=1)
    
    # Layer 1: RY gates
    x_pos = 2
    for i, y in enumerate(y_positions):
        rect = Rectangle((x_pos-0.3, y-0.2), 0.6, 0.4, 
                        facecolor='lightblue', edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
        ax.text(x_pos, y, f'Ry(θ{i})', ha='center', va='center', fontsize=10)
    
    # Layer 1: CNOT gates (even pairs)
    x_pos = 4
    # q0 -> q1
    ax.plot([x_pos, x_pos], [y_positions[0], y_positions[1]], 'b-', linewidth=2)
    ax.plot(x_pos, y_positions[0], 'ko', markersize=8)
    ax.add_patch(Circle((x_pos, y_positions[1]), 0.15, facecolor='white', edgecolor='blue', linewidth=2))
    ax.plot([x_pos-0.1, x_pos+0.1], [y_positions[1], y_positions[1]], 'b-', linewidth=2)
    ax.plot([x_pos, x_pos], [y_positions[1]-0.1, y_positions[1]+0.1], 'b-', linewidth=2)
    
    # q2 -> q3
    ax.plot([x_pos, x_pos], [y_positions[2], y_positions[3]], 'b-', linewidth=2)
    ax.plot(x_pos, y_positions[2], 'ko', markersize=8)
    ax.add_patch(Circle((x_pos, y_positions[3]), 0.15, facecolor='white', edgecolor='blue', linewidth=2))
    ax.plot([x_pos-0.1, x_pos+0.1], [y_positions[3], y_positions[3]], 'b-', linewidth=2)
    ax.plot([x_pos, x_pos], [y_positions[3]-0.1, y_positions[3]+0.1], 'b-', linewidth=2)
    
    # Layer 2: RY gates
    x_pos = 6
    for i, y in enumerate(y_positions):
        rect = Rectangle((x_pos-0.3, y-0.2), 0.6, 0.4,
                        facecolor='lightgreen', edgecolor='green', linewidth=2)
        ax.add_patch(rect)
        ax.text(x_pos, y, f'Ry(θ{i+4})', ha='center', va='center', fontsize=10)
    
    # Layer 2: CNOT gates (odd pairs)
    x_pos = 8
    # q1 -> q2
    ax.plot([x_pos, x_pos], [y_positions[1], y_positions[2]], 'b-', linewidth=2)
    ax.plot(x_pos, y_positions[1], 'ko', markersize=8)
    ax.add_patch(Circle((x_pos, y_positions[2]), 0.15, facecolor='white', edgecolor='blue', linewidth=2))
    ax.plot([x_pos-0.1, x_pos+0.1], [y_positions[2], y_positions[2]], 'b-', linewidth=2)
    ax.plot([x_pos, x_pos], [y_positions[2]-0.1, y_positions[2]+0.1], 'b-', linewidth=2)
    
    # Output state
    ax.text(10, 2, r'$|\psi(\theta)\rangle$', ha='center', va='center', 
           fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'quantum_circuit.png', bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {FIG_DIR / 'quantum_circuit.png'}")
    plt.close()

if __name__ == "__main__":
    print("="*70)
    print("GENERATING PRESENTATION FIGURES")
    print("="*70)
    
    print("\n1. Creating PCA diagram...")
    create_pca_diagram()
    
    print("\n2. Creating VQD diagram...")
    create_vqd_diagram()
    
    print("\n3. Creating comparison diagram...")
    create_comparison_diagram()
    
    print("\n4. Creating results bar plot...")
    create_results_barplot()
    
    print("\n5. Creating VQD advantage plot...")
    create_gap_plot()
    
    print("\n6. Creating quantum circuit diagram...")
    create_circuit_diagram()
    
    print("\n" + "="*70)
    print("ALL FIGURES GENERATED!")
    print("="*70)
    print(f"\nSaved to: {FIG_DIR}/")
    print("\nFigures created:")
    print("  1. pca_diagram.png - Classical PCA concept")
    print("  2. vqd_diagram.png - VQD concept")
    print("  3. pca_vqd_comparison.png - Side-by-side comparison")
    print("  4. accuracy_comparison.png - Bar plot with error bars")
    print("  5. vqd_advantage.png - Gap plot with 95% CI")
    print("  6. quantum_circuit.png - Circuit diagram")
    print("  7. by_class_comparison.png - Already created!")
    print("\n✨ Ready for PowerPoint! ✨")
