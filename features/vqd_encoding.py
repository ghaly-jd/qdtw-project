"""
VQD Quantum PCA Encoding for DTW Pipeline

Integrates Variational Quantum Deflation (VQD) PCA into the DTW classification pipeline.
Uses Procrustes-aligned basis for fair comparison with classical PCA.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum.vqd_pca import vqd_quantum_pca, procrustes_align


def vqd_pca_reduce(X_train, X_test, n_components=8, num_qubits=None, 
                   max_depth=2, penalty_scale='auto', use_procrustes=True,
                   verbose=True):
    """
    Apply VQD Quantum PCA dimensionality reduction for DTW pipeline.
    
    Parameters
    ----------
    X_train : ndarray, shape (n_train, n_features)
        Training data
    X_test : ndarray, shape (n_test, n_features)
        Test data
    n_components : int
        Number of principal components
    num_qubits : int or None
        Number of qubits (auto-determined if None)
    max_depth : int
        Circuit depth
    penalty_scale : float or 'auto'
        VQD orthogonality penalty
    use_procrustes : bool
        If True, use Procrustes-aligned basis for projection
    verbose : bool
        Print progress
        
    Returns
    -------
    X_train_reduced : ndarray, shape (n_train, n_components)
        Projected training data
    X_test_reduced : ndarray, shape (n_test, n_components)
        Projected test data
    info : dict
        VQD diagnostic information including Procrustes alignment
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"VQD Quantum PCA Feature Extraction")
        print(f"{'='*70}")
        print(f"Training data: {X_train.shape}")
        print(f"Test data: {X_test.shape}")
        print(f"Target components: {n_components}")
    
    # Fit VQD PCA on training data
    U_vqd, eigenvalues_vqd, logs = vqd_quantum_pca(
        X_train,
        n_components=n_components,
        num_qubits=num_qubits,
        max_depth=max_depth,
        penalty_scale=penalty_scale,
        maxiter=200,
        verbose=verbose,
        validate=True
    )
    
    # Use Procrustes-aligned basis if requested
    if use_procrustes and 'U_vqd_aligned' in logs:
        U_projection = logs['U_vqd_aligned']
        if verbose:
            print(f"\n✅ Using Procrustes-aligned basis")
            print(f"   Residual improvement: {logs['procrustes_improvement']*100:.1f}%")
    else:
        U_projection = U_vqd
        if verbose:
            print(f"\n⚠️  Using raw VQD basis (no Procrustes)")
    
    # Center data using training mean
    train_mean = np.mean(X_train, axis=0)
    X_train_centered = X_train - train_mean
    X_test_centered = X_test - train_mean
    
    # Project onto VQD basis
    X_train_reduced = X_train_centered @ U_projection.T
    X_test_reduced = X_test_centered @ U_projection.T
    
    if verbose:
        print(f"\nReduced shapes:")
        print(f"  Train: {X_train_reduced.shape}")
        print(f"  Test:  {X_test_reduced.shape}")
        print(f"{'='*70}\n")
    
    # Store info
    info = {
        'U_vqd': U_vqd,
        'U_projection': U_projection,
        'eigenvalues': eigenvalues_vqd,
        'train_mean': train_mean,
        'logs': logs,
        'use_procrustes': use_procrustes
    }
    
    return X_train_reduced, X_test_reduced, info


def main():
    """Test VQD encoding on MSR data."""
    
    print("\n" + "="*70)
    print("VQD Encoding Test")
    print("="*70)
    
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "frame_bank_std.npy"
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return
    
    X = np.load(data_path)
    print(f"Loaded data: {X.shape}")
    
    # Split into train/test
    np.random.seed(42)
    n_train = 100
    n_test = 30
    indices = np.random.permutation(len(X))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train+n_test]
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    # Apply classical PCA first to reduce to 8 features
    from sklearn.decomposition import PCA
    pca_pre = PCA(n_components=8)
    X_train_8d = pca_pre.fit_transform(X_train)
    X_test_8d = pca_pre.transform(X_test)
    
    print(f"\nPre-reduction with classical PCA:")
    print(f"  {X_train.shape[1]} → 8 features")
    print(f"  Variance preserved: {pca_pre.explained_variance_ratio_.sum()*100:.1f}%")
    
    # Apply VQD PCA
    print("\n" + "="*70)
    print("Testing VQD PCA (with Procrustes)")
    print("="*70)
    
    X_train_vqd, X_test_vqd, info = vqd_pca_reduce(
        X_train_8d,
        X_test_8d,
        n_components=4,
        num_qubits=3,
        max_depth=2,
        penalty_scale='auto',
        use_procrustes=True,
        verbose=True
    )
    
    print(f"\n✅ VQD encoding complete!")
    print(f"Final shapes: Train {X_train_vqd.shape}, Test {X_test_vqd.shape}")
    
    # Compare with classical PCA
    pca_classical = PCA(n_components=4)
    X_train_pca = pca_classical.fit_transform(X_train_8d)
    X_test_pca = pca_classical.transform(X_test_8d)
    
    # Compute projection difference
    proj_diff_train = np.linalg.norm(X_train_vqd - X_train_pca, 'fro')
    proj_diff_test = np.linalg.norm(X_test_vqd - X_test_pca, 'fro')
    
    print(f"\nProjection difference (VQD vs Classical PCA):")
    print(f"  Train: ||X_VQD - X_PCA||_F = {proj_diff_train:.2f}")
    print(f"  Test:  ||X_VQD - X_PCA||_F = {proj_diff_test:.2f}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
