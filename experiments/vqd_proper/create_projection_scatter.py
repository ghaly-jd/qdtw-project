"""
Create Projection Equivalence Scatter Plot
===========================================

Visualizes how the SAME data points project differently under PCA vs VQD.
Shows that VQD explores a different subspace that better separates classes.

Author: VQD-DTW Research Team
Date: November 25, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from quantum.vqd_pca import vqd_quantum_pca
from archive.src.loader import load_all_sequences
from sklearn.model_selection import train_test_split

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'

# Paths
DATA_DIR = Path(__file__).parent.parent / "msr_action_data"
FIG_DIR = Path(__file__).parent / "figures"

print("="*70)
print("CREATING PROJECTION EQUIVALENCE SCATTER PLOT")
print("="*70)

# Load data
print("\n1. Loading MSR Action3D data...")
data_path = Path(__file__).parent.parent / "msr_action_data"
sequences, labels = load_all_sequences(str(data_path))
print(f"   Loaded: {len(sequences)} sequences")

# Split data
print("\n2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    sequences, labels,
    train_size=300,
    test_size=60,
    random_state=42,
    stratify=labels
)

print(f"   Train: {len(X_train)} sequences")
print(f"   Test: {len(X_test)} sequences")

# Convert to frame banks
print("\n3. Converting to frame banks...")
def to_frame_bank(sequences):
    return np.vstack([seq for seq in sequences])

X_train_frames = to_frame_bank(X_train)
X_test_frames = to_frame_bank(X_test)

print(f"   Train frames: {X_train_frames.shape}")
print(f"   Test frames: {X_test_frames.shape}")

# Standardize
print("\n4. Standardizing data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_frames)
X_test_scaled = scaler.transform(X_test_frames)

# Pre-reduce to 16D
print("\n5. Pre-reducing 60D → 16D with PCA...")
pre_pca = PCA(n_components=16)
X_train_16d = pre_pca.fit_transform(X_train_scaled)
X_test_16d = pre_pca.transform(X_test_scaled)

print(f"   Reduced to: {X_train_16d.shape}")

# Center per sequence for fair comparison
print("\n6. Per-sequence centering...")
def center_sequences(X_frames, X_sequences):
    """Center each sequence's frames separately."""
    X_centered = []
    idx = 0
    for seq in X_sequences:
        n_frames = len(seq)
        frames = X_frames[idx:idx+n_frames]
        frames_centered = frames - frames.mean(axis=0, keepdims=True)
        X_centered.append(frames_centered)
        idx += n_frames
    return np.vstack(X_centered)

X_train_centered = center_sequences(X_train_16d, X_train)
X_test_centered = center_sequences(X_test_16d, X_test)

# Fit PCA on 16D data
print("\n7. Fitting PCA (16D → 2D)...")
pca = PCA(n_components=2)
pca.fit(X_train_centered)
X_train_pca = pca.transform(X_train_centered)
X_test_pca = pca.transform(X_test_centered)

print(f"   PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

# Fit VQD on 16D data
print("\n8. Fitting VQD (16D → 2D)...")
print("   (This may take 2-3 minutes...)")
U_vqd, eigenvalues, logs = vqd_quantum_pca(
    X_train_centered,
    n_components=2,
    num_qubits=4,
    max_depth=2,
    maxiter=200,
    verbose=False
)

# U_vqd is (n_components, n_features), transpose to (n_features, n_components)
U_vqd = U_vqd.T

X_train_vqd = X_train_centered @ U_vqd
X_test_vqd = X_test_centered @ U_vqd

print(f"   VQD optimization complete!")
print(f"   Orthogonality error: {logs['orthogonality_error']:.6e}")

# Sample subset for visualization (to avoid overcrowding)
print("\n9. Sampling data for visualization...")
np.random.seed(42)
n_samples = 1000
sample_idx = np.random.choice(len(X_train_pca), size=min(n_samples, len(X_train_pca)), replace=False)

# Create figure with 2 subplots
print("\n10. Creating scatter plots...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Expand labels for all frames
y_train_frames = []
for seq, label in zip(X_train, y_train):
    y_train_frames.extend([label] * len(seq))
y_train_frames = np.array(y_train_frames)

# Get unique classes
unique_classes = np.unique(y_train_frames)
n_classes = len(unique_classes)

# Color map
cmap = plt.cm.get_cmap('tab20', n_classes)

# Plot PCA projection
ax1 = axes[0]
for i, class_id in enumerate(unique_classes):
    mask = y_train_frames[sample_idx] == class_id
    ax1.scatter(X_train_pca[sample_idx][mask, 0], 
               X_train_pca[sample_idx][mask, 1],
               c=[cmap(i)], s=20, alpha=0.6, edgecolors='none',
               label=f'Class {class_id}')

ax1.set_xlabel('PC 1', fontsize=14, fontweight='bold')
ax1.set_ylabel('PC 2', fontsize=14, fontweight='bold')
ax1.set_title('PCA Projection (Classical)', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot VQD projection
ax2 = axes[1]
for i, class_id in enumerate(unique_classes):
    mask = y_train_frames[sample_idx] == class_id
    ax2.scatter(X_train_vqd[sample_idx][mask, 0],
               X_train_vqd[sample_idx][mask, 1],
               c=[cmap(i)], s=20, alpha=0.6, edgecolors='none',
               label=f'Class {class_id}')

ax2.set_xlabel('VQD Component 1', fontsize=14, fontweight='bold')
ax2.set_ylabel('VQD Component 2', fontsize=14, fontweight='bold')
ax2.set_title('VQD Projection (Quantum)', fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Main title
fig.suptitle('Projection Equivalence: Same Data, Different Subspaces\n(Train set, 1000 random frames)', 
            fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout()

# Save
output_file = FIG_DIR / 'projection_equivalence_scatter.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_file}")
plt.close()

# Create version with legend showing top 5 classes only
print("\n11. Creating cleaner version (selected classes)...")

# Find top 5 most frequent classes
class_counts = [(c, np.sum(y_train_frames == c)) for c in unique_classes]
class_counts.sort(key=lambda x: x[1], reverse=True)
top_classes = [c for c, _ in class_counts[:5]]

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))

# Plot PCA with selected classes
ax1 = axes2[0]
for i, class_id in enumerate(top_classes):
    mask = y_train_frames[sample_idx] == class_id
    if mask.sum() > 0:
        ax1.scatter(X_train_pca[sample_idx][mask, 0],
                   X_train_pca[sample_idx][mask, 1],
                   s=30, alpha=0.7, edgecolors='white', linewidth=0.5,
                   label=f'Class {class_id}')

# Plot other classes in gray
other_mask = ~np.isin(y_train_frames[sample_idx], top_classes)
ax1.scatter(X_train_pca[sample_idx][other_mask, 0],
           X_train_pca[sample_idx][other_mask, 1],
           c='lightgray', s=10, alpha=0.3, edgecolors='none',
           label='Other classes')

ax1.set_xlabel('PC 1', fontsize=14, fontweight='bold')
ax1.set_ylabel('PC 2', fontsize=14, fontweight='bold')
ax1.set_title('PCA Projection', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=10, framealpha=0.9)

# Plot VQD with selected classes
ax2 = axes2[1]
for i, class_id in enumerate(top_classes):
    mask = y_train_frames[sample_idx] == class_id
    if mask.sum() > 0:
        ax2.scatter(X_train_vqd[sample_idx][mask, 0],
                   X_train_vqd[sample_idx][mask, 1],
                   s=30, alpha=0.7, edgecolors='white', linewidth=0.5,
                   label=f'Class {class_id}')

# Plot other classes in gray
ax2.scatter(X_train_vqd[sample_idx][other_mask, 0],
           X_train_vqd[sample_idx][other_mask, 1],
           c='lightgray', s=10, alpha=0.3, edgecolors='none',
           label='Other classes')

ax2.set_xlabel('VQD Component 1', fontsize=14, fontweight='bold')
ax2.set_ylabel('VQD Component 2', fontsize=14, fontweight='bold')
ax2.set_title('VQD Projection', fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', fontsize=10, framealpha=0.9)

# Main title
fig2.suptitle('Projection Equivalence: VQD Finds Different Discriminative Subspace', 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout()

# Save
output_file2 = FIG_DIR / 'projection_equivalence_clean.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_file2}")
plt.close()

# Compute class separability metrics
print("\n12. Computing class separability metrics...")

def compute_fisher_ratio(X, y):
    """Compute Fisher discriminant ratio (between-class / within-class variance)."""
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)
    
    # Between-class scatter
    between_scatter = 0
    for c in classes:
        X_c = X[y == c]
        n_c = len(X_c)
        mean_c = X_c.mean(axis=0)
        diff = (mean_c - overall_mean).reshape(-1, 1)
        between_scatter += n_c * (diff @ diff.T)
    
    # Within-class scatter
    within_scatter = 0
    for c in classes:
        X_c = X[y == c]
        mean_c = X_c.mean(axis=0)
        for x in X_c:
            diff = (x - mean_c).reshape(-1, 1)
            within_scatter += diff @ diff.T
    
    # Fisher ratio
    between_var = np.trace(between_scatter) / len(classes)
    within_var = np.trace(within_scatter) / len(X)
    
    return between_var / (within_var + 1e-10)

fisher_pca = compute_fisher_ratio(X_train_pca, y_train_frames)
fisher_vqd = compute_fisher_ratio(X_train_vqd, y_train_frames)

print(f"\n   Fisher Ratio (higher = better separation):")
print(f"   PCA: {fisher_pca:.4f}")
print(f"   VQD: {fisher_vqd:.4f}")
print(f"   VQD improvement: {((fisher_vqd/fisher_pca - 1) * 100):.1f}%")

print("\n" + "="*70)
print("PROJECTION EQUIVALENCE PLOTS GENERATED!")
print("="*70)
print(f"\nFiles created:")
print(f"1. projection_equivalence_scatter.png - All classes")
print(f"2. projection_equivalence_clean.png - Top 5 classes highlighted")
print(f"\nKey findings:")
print(f"- VQD explores a different subspace than PCA")
print(f"- VQD achieves {((fisher_vqd/fisher_pca - 1) * 100):.1f}% better class separability")
print(f"- Same input data projects differently → different basis vectors")
print("\n✨ Ready for presentation! ✨")
