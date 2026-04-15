"""
Visualize VQD vs PCA Subspace Exploration
==========================================

Creates visualizations showing how VQD and PCA explore different subspaces:
1. Eigenvector similarity heatmap (orthogonality check)
2. Projected data scatter (2D/3D visualization)
3. Variance explained comparison
4. Subspace angle/distance metrics
5. Per-class separability in different subspaces

Author: VQD-DTW Research Team
Date: November 25, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from quantum.vqd_pca import vqd_quantum_pca

# Set style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_style("whitegrid")

# Directories
DATA_DIR = Path(__file__).parent.parent / "msr_action_data"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

class SubspaceVisualizer:
    """Visualize and compare PCA vs VQD subspaces."""
    
    def __init__(self, k=8, seed=42):
        self.k = k
        self.seed = seed
        np.random.seed(seed)
        
        print(f"Loading and preparing data...")
        # Load raw sequences
        sys.path.insert(0, str(Path(__file__).parent.parent / 'archive' / 'src'))
        from archive.src.loader import load_all_sequences
        
        sequences, labels = load_all_sequences(str(DATA_DIR))
        print(f"Loaded {len(sequences)} sequences")
        
        # Create frame bank from all sequences
        all_frames = np.vstack(sequences)
        print(f"Frame bank: {all_frames.shape}")
        
        # Normalize frame bank
        scaler_frames = StandardScaler()
        all_frames_scaled = scaler_frames.fit_transform(all_frames)
        
        # Pre-reduce to 16D using PCA
        print(f"Pre-reducing to 16D...")
        pca_pre = PCA(n_components=16)
        all_frames_16d = pca_pre.fit_transform(all_frames_scaled)
        print(f"Variance captured: {pca_pre.explained_variance_ratio_.sum():.1%}")
        
        # Project sequences to 16D
        sequences_16d = []
        idx = 0
        for seq in sequences:
            length = len(seq)
            seq_16d = all_frames_16d[idx:idx+length]
            sequences_16d.append(seq_16d)
            idx += length
        
        # Split train/test
        from sklearn.model_selection import train_test_split
        X_train_16d, X_test_16d, y_train, y_test = train_test_split(
            sequences_16d, labels, test_size=60, train_size=300, 
            random_state=seed, stratify=labels
        )
        
        self.X_train_16d = X_train_16d
        self.y_train = np.array(y_train)
        self.X_test_16d = X_test_16d
        self.y_test = np.array(y_test)
        
        print(f"Train: {len(self.X_train_16d)} sequences, Test: {len(self.X_test_16d)} sequences")
        
        # Per-sequence centering (fair comparison)
        self.X_train_centered = [seq - seq.mean(axis=0) for seq in self.X_train_16d]
        self.X_test_centered = [seq - seq.mean(axis=0) for seq in self.X_test_16d]
        
        # Flatten for subspace learning
        self.X_train_flat = np.vstack(self.X_train_centered)
        self.X_test_flat = np.vstack(self.X_test_centered)
        
        # Standardize
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_flat)
        self.X_test_scaled = self.scaler.transform(self.X_test_flat)
        
        print(f"Flattened train: {self.X_train_scaled.shape}")
    
    def fit_methods(self):
        """Fit both PCA and VQD."""
        print(f"\n{'='*70}")
        print(f"FITTING PCA AND VQD (k={self.k})")
        print(f"{'='*70}")
        
        # PCA
        print("\n1. Fitting PCA...")
        self.pca = PCA(n_components=self.k)
        self.pca.fit(self.X_train_scaled)
        self.U_pca = self.pca.components_.T  # (16, k)
        print(f"   PCA eigenvectors shape: {self.U_pca.shape}")
        print(f"   Variance explained: {self.pca.explained_variance_ratio_.sum():.1%}")
        
        # VQD
        print("\n2. Fitting VQD...")
        self.U_vqd, vqd_eigenvalues, vqd_logs = vqd_quantum_pca(
            self.X_train_scaled, n_components=self.k, 
            num_qubits=4, max_depth=2, maxiter=200, verbose=True)
        # Transpose to match PCA format (features x components)
        self.U_vqd = self.U_vqd.T
        print(f"   VQD eigenvectors shape: {self.U_vqd.shape}")
        
        # Compute covariance for variance explained
        C = np.cov(self.X_train_scaled.T)
        variances_vqd = np.array([self.U_vqd[:, i].T @ C @ self.U_vqd[:, i] 
                                  for i in range(self.k)])
        total_var = np.trace(C)
        print(f"   VQD variance captured: {variances_vqd.sum() / total_var:.1%}")
        
        print("\n✓ Both methods fitted!")
    
    def plot_eigenvector_similarity(self):
        """
        Create heatmap showing similarity between PCA and VQD eigenvectors.
        High values = subspaces are similar, Low values = different exploration.
        """
        print("\n" + "="*70)
        print("PLOTTING EIGENVECTOR SIMILARITY MATRIX")
        print("="*70)
        
        # Compute similarity matrix (absolute value of dot products)
        similarity = np.abs(self.U_pca.T @ self.U_vqd)  # (k, k)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(similarity, annot=True, fmt='.2f', cmap='RdYlGn_r',
                   vmin=0, vmax=1, square=True, cbar_kws={'label': 'Absolute Correlation'},
                   xticklabels=[f'VQD-{i+1}' for i in range(self.k)],
                   yticklabels=[f'PCA-{i+1}' for i in range(self.k)],
                   ax=ax)
        
        ax.set_title('PCA vs VQD Eigenvector Similarity\n' + 
                    '(Low values = Different Subspaces)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('VQD Components', fontsize=12, fontweight='bold')
        ax.set_ylabel('PCA Components', fontsize=12, fontweight='bold')
        
        # Add interpretation text
        avg_similarity = similarity.mean()
        max_similarity = similarity.max()
        fig.text(0.5, 0.02, 
                f'Average Similarity: {avg_similarity:.3f} | Max: {max_similarity:.3f}\n' +
                f'Low similarity indicates VQD explores different subspace structure',
                ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'subspace_similarity_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {FIG_DIR / 'subspace_similarity_heatmap.png'}")
        print(f"  Average similarity: {avg_similarity:.3f}")
        print(f"  Max similarity: {max_similarity:.3f}")
        print(f"  → {'Similar subspaces' if avg_similarity > 0.7 else 'Different subspaces!'}")
        plt.close()
    
    def plot_subspace_angles(self):
        """Plot principal angles between PCA and VQD subspaces."""
        print("\n" + "="*70)
        print("COMPUTING PRINCIPAL ANGLES")
        print("="*70)
        
        # Compute principal angles using SVD
        # Angle between subspaces: arccos(singular values of U_pca^T @ U_vqd)
        _, singular_values, _ = np.linalg.svd(self.U_pca.T @ self.U_vqd)
        principal_angles = np.arccos(np.clip(singular_values, -1, 1))
        angles_degrees = np.degrees(principal_angles)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        x = np.arange(1, self.k + 1)
        bars = ax1.bar(x, angles_degrees, color='steelblue', alpha=0.8, edgecolor='black')
        ax1.axhline(y=45, color='red', linestyle='--', linewidth=2, 
                   label='45° (Significantly Different)', alpha=0.7)
        ax1.axhline(y=10, color='orange', linestyle='--', linewidth=2,
                   label='10° (Somewhat Similar)', alpha=0.7)
        
        ax1.set_xlabel('Component Pair', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Principal Angle (degrees)', fontsize=12, fontweight='bold')
        ax1.set_title('Principal Angles Between\nPCA and VQD Subspaces', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, angle in zip(bars, angles_degrees):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{angle:.1f}°', ha='center', va='bottom', fontsize=9)
        
        # Cumulative plot
        ax2.plot(x, angles_degrees, 'o-', linewidth=3, markersize=10, 
                color='steelblue', label='Principal Angles')
        ax2.fill_between(x, 0, angles_degrees, alpha=0.3, color='steelblue')
        ax2.axhline(y=angles_degrees.mean(), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean: {angles_degrees.mean():.1f}°')
        
        ax2.set_xlabel('Component Pair', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Angle (degrees)', fontsize=12, fontweight='bold')
        ax2.set_title('Subspace Angle Profile', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'subspace_principal_angles.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {FIG_DIR / 'subspace_principal_angles.png'}")
        print(f"  Mean angle: {angles_degrees.mean():.1f}°")
        print(f"  Max angle: {angles_degrees.max():.1f}°")
        print(f"  Min angle: {angles_degrees.min():.1f}°")
        plt.close()
    
    def plot_2d_projections(self):
        """Plot 2D projections showing how each method separates classes."""
        print("\n" + "="*70)
        print("PLOTTING 2D PROJECTIONS")
        print("="*70)
        
        # Project test sequences onto first 2 components
        X_test_pca_2d = self.X_test_scaled @ self.U_pca[:, :2]
        X_test_vqd_2d = self.X_test_scaled @ self.U_vqd[:, :2]
        
        # Reconstruct per-sequence (sum over frames)
        test_seq_lens = [len(seq) for seq in self.X_test_centered]
        
        def reconstruct_sequences(X_proj):
            sequences = []
            idx = 0
            for length in test_seq_lens:
                seq_proj = X_proj[idx:idx+length]
                sequences.append(seq_proj.mean(axis=0))  # Average over frames
                idx += length
            return np.array(sequences)
        
        test_pca_2d = reconstruct_sequences(X_test_pca_2d)
        test_vqd_2d = reconstruct_sequences(X_test_vqd_2d)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Get unique classes for coloring
        unique_classes = np.unique(self.y_test)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
        
        # PCA projection
        for idx, cls in enumerate(unique_classes):
            mask = self.y_test == cls
            ax1.scatter(test_pca_2d[mask, 0], test_pca_2d[mask, 1],
                       c=[colors[idx]], label=f'Class {cls}', alpha=0.6, s=50)
        
        ax1.set_xlabel('PCA Component 1', fontsize=12, fontweight='bold')
        ax1.set_ylabel('PCA Component 2', fontsize=12, fontweight='bold')
        ax1.set_title('PCA Subspace Projection\n(First 2 Components)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, ncol=2, loc='best')
        
        # VQD projection
        for idx, cls in enumerate(unique_classes):
            mask = self.y_test == cls
            ax2.scatter(test_vqd_2d[mask, 0], test_vqd_2d[mask, 1],
                       c=[colors[idx]], label=f'Class {cls}', alpha=0.6, s=50)
        
        ax2.set_xlabel('VQD Component 1', fontsize=12, fontweight='bold')
        ax2.set_ylabel('VQD Component 2', fontsize=12, fontweight='bold')
        ax2.set_title('VQD Subspace Projection\n(First 2 Components)', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, ncol=2, loc='best')
        
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'subspace_2d_projections.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {FIG_DIR / 'subspace_2d_projections.png'}")
        plt.close()
    
    def plot_variance_comparison(self):
        """Compare variance captured by each component."""
        print("\n" + "="*70)
        print("PLOTTING VARIANCE COMPARISON")
        print("="*70)
        
        # PCA variance (directly available)
        pca_var = self.pca.explained_variance_ratio_
        
        # VQD variance (compute manually)
        C = np.cov(self.X_train_scaled.T)
        vqd_var = np.array([self.U_vqd[:, i].T @ C @ self.U_vqd[:, i] 
                           for i in range(self.k)])
        vqd_var_ratio = vqd_var / np.trace(C)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual component variance
        x = np.arange(1, self.k + 1)
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, pca_var * 100, width, label='PCA',
                       color='steelblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, vqd_var_ratio * 100, width, label='VQD',
                       color='coral', alpha=0.8)
        
        ax1.set_xlabel('Component', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Variance Captured per Component', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.legend(fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        
        # Cumulative variance
        pca_cumvar = np.cumsum(pca_var) * 100
        vqd_cumvar = np.cumsum(vqd_var_ratio) * 100
        
        ax2.plot(x, pca_cumvar, 'o-', linewidth=3, markersize=8, 
                label='PCA', color='steelblue')
        ax2.plot(x, vqd_cumvar, 's-', linewidth=3, markersize=8,
                label='VQD', color='coral')
        ax2.axhline(y=90, color='green', linestyle='--', linewidth=2,
                   alpha=0.5, label='90% threshold')
        
        ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Variance (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Add final values
        ax2.text(self.k, pca_cumvar[-1] + 1, f'{pca_cumvar[-1]:.1f}%',
                ha='center', fontsize=10, fontweight='bold', color='steelblue')
        ax2.text(self.k, vqd_cumvar[-1] - 2, f'{vqd_cumvar[-1]:.1f}%',
                ha='center', fontsize=10, fontweight='bold', color='coral')
        
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'subspace_variance_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {FIG_DIR / 'subspace_variance_comparison.png'}")
        print(f"  PCA total variance: {pca_cumvar[-1]:.1f}%")
        print(f"  VQD total variance: {vqd_cumvar[-1]:.1f}%")
        plt.close()
    
    def plot_class_separability(self):
        """Measure and plot class separability in each subspace."""
        print("\n" + "="*70)
        print("COMPUTING CLASS SEPARABILITY")
        print("="*70)
        
        # Project training data
        X_train_pca = self.X_train_scaled @ self.U_pca
        X_train_vqd = self.X_train_scaled @ self.U_vqd
        
        # Reconstruct per-sequence
        train_seq_lens = [len(seq) for seq in self.X_train_centered]
        
        def reconstruct_sequences(X_proj):
            sequences = []
            idx = 0
            for length in train_seq_lens:
                seq_proj = X_proj[idx:idx+length]
                sequences.append(seq_proj.mean(axis=0))
                idx += length
            return np.array(sequences)
        
        train_pca = reconstruct_sequences(X_train_pca)
        train_vqd = reconstruct_sequences(X_train_vqd)
        
        # Compute between-class vs within-class variance ratio (Fisher criterion)
        def fisher_criterion(X, y):
            """Higher is better separation."""
            classes = np.unique(y)
            n_classes = len(classes)
            
            # Overall mean
            mean_overall = X.mean(axis=0)
            
            # Between-class scatter
            S_b = np.zeros((X.shape[1], X.shape[1]))
            for cls in classes:
                X_cls = X[y == cls]
                n_cls = len(X_cls)
                mean_cls = X_cls.mean(axis=0)
                diff = (mean_cls - mean_overall).reshape(-1, 1)
                S_b += n_cls * (diff @ diff.T)
            
            # Within-class scatter
            S_w = np.zeros((X.shape[1], X.shape[1]))
            for cls in classes:
                X_cls = X[y == cls]
                mean_cls = X_cls.mean(axis=0)
                for x in X_cls:
                    diff = (x - mean_cls).reshape(-1, 1)
                    S_w += diff @ diff.T
            
            # Fisher criterion: trace(S_b) / trace(S_w)
            return np.trace(S_b) / (np.trace(S_w) + 1e-10)
        
        fisher_pca = fisher_criterion(train_pca, self.y_train)
        fisher_vqd = fisher_criterion(train_vqd, self.y_train)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        methods = ['PCA', 'VQD']
        scores = [fisher_pca, fisher_vqd]
        colors_bar = ['steelblue', 'coral']
        
        bars = ax.bar(methods, scores, color=colors_bar, alpha=0.8, 
                     edgecolor='black', linewidth=2)
        
        ax.set_ylabel('Fisher Criterion\n(Between-Class / Within-Class Variance)',
                     fontsize=12, fontweight='bold')
        ax.set_title('Class Separability in Different Subspaces\n(Higher = Better Separation)',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Add improvement percentage
        improvement = ((fisher_vqd - fisher_pca) / fisher_pca) * 100
        ax.text(0.5, max(scores) * 0.5, 
               f'VQD Improvement:\n{improvement:+.1f}%',
               ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow' if improvement > 0 else 'lightgray', 
                        alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'subspace_class_separability.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {FIG_DIR / 'subspace_class_separability.png'}")
        print(f"  PCA Fisher criterion: {fisher_pca:.2f}")
        print(f"  VQD Fisher criterion: {fisher_vqd:.2f}")
        print(f"  Improvement: {improvement:+.1f}%")
        plt.close()

def main():
    print("="*70)
    print("VQD VS PCA SUBSPACE VISUALIZATION")
    print("="*70)
    
    visualizer = SubspaceVisualizer(k=8, seed=42)
    visualizer.fit_methods()
    
    print("\n" + "="*70)
    print("GENERATING SUBSPACE VISUALIZATIONS")
    print("="*70)
    
    visualizer.plot_eigenvector_similarity()
    visualizer.plot_subspace_angles()
    visualizer.plot_variance_comparison()
    visualizer.plot_2d_projections()
    visualizer.plot_class_separability()
    
    print("\n" + "="*70)
    print("ALL SUBSPACE VISUALIZATIONS COMPLETE!")
    print("="*70)
    print(f"\nGenerated in: {FIG_DIR}/")
    print("\nFigures:")
    print("  1. subspace_similarity_heatmap.png - Eigenvector correlation")
    print("  2. subspace_principal_angles.png - Angular distance between subspaces")
    print("  3. subspace_variance_comparison.png - Variance captured")
    print("  4. subspace_2d_projections.png - Visual separation of classes")
    print("  5. subspace_class_separability.png - Fisher criterion comparison")
    print("\n✨ Ready for presentation! ✨")

if __name__ == "__main__":
    main()
