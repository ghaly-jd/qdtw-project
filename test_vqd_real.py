"""Quick test of VQD PCA on real MSR data."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from quantum.vqd_pca import vqd_quantum_pca
from sklearn.decomposition import PCA

# Load real MSR data
X_real = np.load('data/frame_bank_std.npy')
print(f"Loaded data shape: {X_real.shape}")

# Subsample for testing
np.random.seed(42)
indices = np.random.choice(len(X_real), size=100, replace=False)
X_sub = X_real[indices]

# Reduce to 8 features first (3 qubits)
pca_pre = PCA(n_components=8)
X_reduced = pca_pre.fit_transform(X_sub)
print(f"Reduced data shape: {X_reduced.shape}")
print(f"Variance preserved: {pca_pre.explained_variance_ratio_.sum()*100:.1f}%")

# Run VQD PCA
print("\n" + "="*70)
print("Running VQD Quantum PCA on real MSR data")
print("="*70)

U_vqd, eigenvalues_vqd, logs = vqd_quantum_pca(
    X_reduced,
    n_components=4,
    num_qubits=3,
    max_depth=2,
    penalty_scale='auto',  # Use adaptive penalty
    maxiter=200,  # More iterations
    verbose=True,
    validate=True
)

print("\n" + "="*70)
print("✅ VQD PCA Complete!")
print("="*70)
print(f"Orthogonality error: {logs['orthogonality_error']:.6e}")
print(f"Mean principal angle: {np.mean(logs['principal_angles_deg']):.2f}°")
print(f"Max principal angle: {np.max(logs['principal_angles_deg']):.2f}°")
print(f"Mean eigenvalue error: {np.mean(logs['eigenvalue_errors']):.6f}")
print(f"Max eigenvalue relative error: {np.max(logs['eigenvalue_relative_errors'])*100:.2f}%")
