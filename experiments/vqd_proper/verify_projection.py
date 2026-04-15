"""
Verify Projection Methodology Consistency

Check if PCA and VQD are using the same projection methodology.
Specifically verify centering and transformation steps.
"""

import numpy as np
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'archive' / 'src'))

from archive.src.loader import load_all_sequences
from quantum.vqd_pca import vqd_quantum_pca


def test_projection_consistency():
    """Test if PCA and VQD projections are consistent."""
    
    print("\n" + "="*80)
    print("PROJECTION METHODOLOGY VERIFICATION")
    print("="*80 + "\n")
    
    # Create synthetic data for testing
    print("1. Creating synthetic test data...")
    np.random.seed(42)
    n_samples = 100
    n_features = 16
    X = np.random.randn(n_samples, n_features)
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"   Data shape: {X_scaled.shape}")
    print(f"   Mean: {np.mean(X_scaled, axis=0)[:3]} ... (should be ~0)")
    print(f"   Std: {np.std(X_scaled, axis=0)[:3]} ... (should be ~1)")
    
    # Test PCA projection
    print("\n2. Testing PCA projection...")
    k = 4
    pca = PCA(n_components=k)
    pca.fit(X_scaled)
    
    # Manual projection (what we do in code)
    X_pca_manual = pca.transform(X_scaled)
    
    # Check PCA centering
    print(f"   PCA mean_: {pca.mean_[:3]} ...")
    print(f"   X_scaled mean: {np.mean(X_scaled, axis=0)[:3]} ...")
    print(f"   Are they same? {np.allclose(pca.mean_, np.mean(X_scaled, axis=0))}")
    
    print(f"   Projected shape: {X_pca_manual.shape}")
    print(f"   Projected mean: {np.mean(X_pca_manual, axis=0)} (should be ~0)")
    
    # Test VQD projection
    print("\n3. Testing VQD projection...")
    num_qubits = int(np.ceil(np.log2(n_features)))
    
    U_vqd, eigenvalues, logs = vqd_quantum_pca(
        X_scaled,
        n_components=k,
        num_qubits=num_qubits,
        max_depth=2,
        penalty_scale='auto',
        ramped_penalties=True,
        entanglement='alternating',
        maxiter=200,
        verbose=False,
        validate=True
    )
    
    # Our VQD projection (with per-sample centering)
    X_vqd_perseq = []
    for i in range(len(X_scaled)):
        sample = X_scaled[i:i+1]  # Single sample as 2D array
        mean = np.mean(sample, axis=0)
        proj = (sample - mean) @ U_vqd.T
        X_vqd_perseq.append(proj[0])
    X_vqd_perseq = np.array(X_vqd_perseq)
    
    # Alternative: Global centering (like PCA)
    X_vqd_global = (X_scaled - np.mean(X_scaled, axis=0)) @ U_vqd.T
    
    print(f"   VQD basis shape: {U_vqd.shape}")
    print(f"   Orthogonality: {logs['orthogonality_error']:.2e}")
    print(f"   Principal angles: mean={np.mean(logs['principal_angles_deg']):.1f}°, max={np.max(logs['principal_angles_deg']):.1f}°")
    
    print(f"\n   VQD projection (per-sample centering):")
    print(f"     Shape: {X_vqd_perseq.shape}")
    print(f"     Mean: {np.mean(X_vqd_perseq, axis=0)} (should be ~0)")
    
    print(f"\n   VQD projection (global centering, like PCA):")
    print(f"     Shape: {X_vqd_global.shape}")
    print(f"     Mean: {np.mean(X_vqd_global, axis=0)} (should be ~0)")
    
    # Compare the two approaches
    print("\n4. Comparing projection approaches...")
    diff_perseq = np.linalg.norm(X_vqd_perseq - X_pca_manual)
    diff_global = np.linalg.norm(X_vqd_global - X_pca_manual)
    
    print(f"   ||VQD_perseq - PCA||: {diff_perseq:.2f}")
    print(f"   ||VQD_global - PCA||: {diff_global:.2f}")
    
    # Check variance explained
    pca_var = np.var(X_pca_manual, axis=0).sum()
    vqd_perseq_var = np.var(X_vqd_perseq, axis=0).sum()
    vqd_global_var = np.var(X_vqd_global, axis=0).sum()
    
    print(f"\n   Total variance:")
    print(f"     PCA: {pca_var:.2f}")
    print(f"     VQD (per-seq): {vqd_perseq_var:.2f}")
    print(f"     VQD (global): {vqd_global_var:.2f}")
    
    # Test on actual sequence data
    print("\n5. Testing on actual sequence data...")
    data_path = Path(__file__).parent.parent / "msr_action_data"
    sequences, labels = load_all_sequences(str(data_path))
    
    # Take first sequence
    test_seq = sequences[0]  # Shape: (T, 60)
    print(f"   Test sequence shape: {test_seq.shape}")
    
    # Scale and pre-reduce
    seq_scaled = scaler.fit_transform(test_seq)
    pca_pre = PCA(n_components=16)
    pca_pre.fit(seq_scaled)
    seq_reduced = pca_pre.transform(seq_scaled)
    
    print(f"   After pre-reduction: {seq_reduced.shape}")
    
    # PCA projection
    pca_final = PCA(n_components=4)
    pca_final.fit(seq_reduced)
    seq_pca = pca_final.transform(seq_reduced)
    
    print(f"   PCA projection: {seq_pca.shape}")
    print(f"   PCA mean per frame: {np.mean(seq_pca):.3f}")
    
    # VQD projection (per-frame centering - our current approach)
    U_vqd_test, _, _ = vqd_quantum_pca(
        seq_reduced, n_components=4, num_qubits=4,
        max_depth=2, maxiter=100, verbose=False, validate=False
    )
    
    mean_seq = np.mean(seq_reduced, axis=0)
    seq_vqd = (seq_reduced - mean_seq) @ U_vqd_test.T
    
    print(f"   VQD projection: {seq_vqd.shape}")
    print(f"   VQD mean per frame: {np.mean(seq_vqd):.3f}")
    
    # Key finding
    print("\n" + "="*80)
    print("KEY FINDING:")
    print("="*80)
    print("""
Our VQD uses: (seq - mean(seq)) @ U_vqd.T  (per-sequence centering)
PCA uses:     pca.transform(seq)            (global centering from fit)

This difference means:
- Each VQD-projected sequence is centered around its own mean
- PCA projects all sequences relative to the training set global mean

This creates different feature spaces and may explain the accuracy differences!
    """)
    
    print("="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print("""
To make a fair comparison, we should align the centering approach:

Option 1: Make VQD use global centering (like PCA)
  seq_vqd = (seq - global_mean) @ U_vqd.T
  where global_mean = mean of all training frames

Option 2: Make PCA use per-sequence centering (like VQD)
  For each sequence:
    mean_seq = np.mean(seq)
    seq_pca = (seq - mean_seq) @ pca.components_.T

We should test both to see which is fair!
    """)
    print("="*80 + "\n")


def main():
    test_projection_consistency()


if __name__ == "__main__":
    main()
