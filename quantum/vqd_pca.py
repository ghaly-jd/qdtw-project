"""
Variational Quantum Deflation (VQD) for PCA

VQD finds multiple eigenvectors sequentially by adding orthogonality penalties.
For the r-th eigenvector |ψ(θ_r)⟩:

L_r(θ_r) = ⟨ψ|H|ψ⟩ + Σ_{j=1}^{r-1} λ_j |⟨ψ(θ_r)|ψ(θ_j*)⟩|²

Where H = -C to find largest eigenvectors of covariance matrix C.

This implementation uses a quantum-inspired variational approach where we encode
the classical eigenvector search into a parameterized quantum circuit.
"""

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer
import warnings
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def vqd_quantum_pca(X, n_components=4, num_qubits=None, max_depth=2, penalty_scale=10.0, 
                    maxiter=100, ramped_penalties=True, entanglement='ladder',
                    verbose=True, validate=True):
    """
    Perform PCA using Variational Quantum Deflation.
    
    Uses quantum circuits with amplitude encoding to represent eigenvectors.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data matrix
    n_components : int
        Number of principal components (eigenvectors) to find
    num_qubits : int or None
        Number of qubits (must satisfy 2^num_qubits >= n_features).
        If None, automatically determined.
    max_depth : int
        Maximum circuit depth (layers of rotations)
    penalty_scale : float or 'auto'
        Base penalty weight λ for orthogonality constraints (5-20 × spectral gap).
        If 'auto', automatically determined from spectral gap.
    maxiter : int
        Maximum optimization iterations per eigenvector
    ramped_penalties : bool
        If True, increase penalties progressively (×1.0, ×1.5, ×2.0...) to reduce late-mode mixing
    entanglement : str
        Entanglement pattern: 'ladder', 'full', or 'alternating'
    verbose : bool
        Print progress information
    validate : bool
        Validate results against classical PCA
        
    Returns
    -------
    components : ndarray, shape (n_components, n_features)
        Principal components (eigenvectors)
    eigenvalues : ndarray, shape (n_components,)
        Eigenvalues (variance explained)
    logs : dict
        Diagnostic information including angles, orthogonality errors, Rayleigh quotients
    """
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov = np.cov(X_centered.T)
    n_features = cov.shape[0]
    
    # Determine number of qubits needed
    if num_qubits is None:
        num_qubits = int(np.ceil(np.log2(n_features)))
    state_dim = 2**num_qubits
    
    if state_dim < n_features:
        raise ValueError(f"Need at least {int(np.ceil(np.log2(n_features)))} qubits for {n_features} features")
    
    # Pad covariance matrix to match quantum state dimension
    cov_padded = np.zeros((state_dim, state_dim))
    cov_padded[:n_features, :n_features] = cov
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"VQD Quantum PCA")
        print(f"{'='*70}")
        print(f"Data: {X.shape[0]} samples × {n_features} features")
        print(f"Qubits: {num_qubits} (state dimension: {state_dim})")
        print(f"Target: {n_components} principal components")
        print(f"Circuit depth: {max_depth}")
        print(f"Entanglement: {entanglement}")
        print(f"Penalty scale: {penalty_scale}")
        print(f"Ramped penalties: {ramped_penalties}")
    
    # Classical PCA for validation
    if validate:
        classical_eigenvalues, classical_eigenvectors = np.linalg.eigh(cov)
        # Sort in descending order (largest first)
        idx = np.argsort(classical_eigenvalues)[::-1]
        classical_eigenvalues = classical_eigenvalues[idx]
        classical_eigenvectors = classical_eigenvectors[:, idx]
        U_pca = classical_eigenvectors[:, :n_components].T
        
        if verbose:
            print(f"\nClassical PCA eigenvalues (top {n_components}):")
            print(f"  {classical_eigenvalues[:n_components]}")
    
    # H = -C (negative covariance) to find largest eigenvectors
    H = -cov_padded
    
    # Adaptive penalty scale based on spectral gap estimate
    if penalty_scale == 'auto':
        # Estimate spectral gap from classical eigenvalues
        if validate:
            spectral_gap = classical_eigenvalues[0] - classical_eigenvalues[min(n_components, len(classical_eigenvalues)-1)]
            penalty_scale = max(10.0, 10.0 * spectral_gap)
            if verbose:
                print(f"Adaptive penalty scale: {penalty_scale:.2f} (gap: {spectral_gap:.2f})")
        else:
            penalty_scale = 10.0
    
    
    # Store found eigenvectors and their statevectors
    found_eigenvectors = []  # Truncated to n_features
    found_eigenvalues = []
    found_statevectors = []  # Full state_dim for overlap calculations
    
    # VQD: Find eigenvectors sequentially
    for r in range(n_components):
        if verbose:
            print(f"\n{'─'*70}")
            print(f"Finding eigenvector {r+1}/{n_components}")
            print(f"{'─'*70}")
        
        # Initial random parameters (try multiple restarts for later eigenvectors)
        n_params = num_qubits * max_depth
        best_result = None
        best_eigenvalue = -np.inf
        
        n_restarts = 1 if r == 0 else 3  # More restarts for later eigenvectors
        
        for restart in range(n_restarts):
            theta_init = np.random.randn(n_params) * 0.1
            
            # Define objective function with orthogonality penalties
            def objective(theta):
                # Build quantum circuit
                qc = _build_quantum_ansatz(theta, num_qubits, max_depth, entanglement)
                statevector = Statevector(qc).data
                
                # Primary term: ⟨ψ|H|ψ⟩
                expectation = np.real(np.conj(statevector) @ H @ statevector)
                
                # Penalty terms: λ_j |⟨ψ|ψ_j⟩|²
                # With ramped penalties: λ increases with r
                penalty = 0.0
                for j, prev_state in enumerate(found_statevectors):
                    overlap = np.abs(np.vdot(prev_state, statevector))
                    if ramped_penalties:
                        # Ramp: ×1.0, ×1.5, ×2.0, ×2.5...
                        ramp_factor = 1.0 + 0.5 * r
                        effective_penalty = penalty_scale * ramp_factor
                    else:
                        effective_penalty = penalty_scale
                    penalty += effective_penalty * overlap**2
                
                return expectation + penalty
            
            # Optimize
            result = minimize(
                objective,
                theta_init,
                method='COBYLA',
                options={'maxiter': maxiter, 'disp': False}
            )
            
            # Track best result
            if best_result is None or result.fun < best_result.fun:
                best_result = result
                # Compute eigenvalue for this solution
                qc_test = _build_quantum_ansatz(result.x, num_qubits, max_depth)
                state_test = Statevector(qc_test).data[:n_features]
                state_test = state_test / np.linalg.norm(state_test)
                ev_test = np.real(state_test @ cov @ state_test)
                if ev_test > best_eigenvalue:
                    best_eigenvalue = ev_test
        
        result = best_result
        
        # Get optimized statevector
        qc_opt = _build_quantum_ansatz(result.x, num_qubits, max_depth, entanglement)
        statevector_opt = Statevector(qc_opt).data
        
        # Extract eigenvector (truncate to n_features and apply Gram-Schmidt)
        eigenvector = statevector_opt[:n_features].copy()
        
        # Apply Gram-Schmidt to ensure orthogonality
        for prev_vec in found_eigenvectors:
            eigenvector -= np.vdot(prev_vec, eigenvector) * prev_vec
        
        # Normalize
        norm = np.linalg.norm(eigenvector)
        if norm > 1e-10:
            eigenvector /= norm
        else:
            warnings.warn(f"Eigenvector {r+1} has near-zero norm after orthogonalization")
            eigenvector = np.random.randn(n_features)
            eigenvector /= np.linalg.norm(eigenvector)
        
        # Compute eigenvalue (Rayleigh quotient on original covariance)
        eigenvalue = np.real(eigenvector @ cov @ eigenvector)
        
        # Store results
        found_eigenvectors.append(eigenvector)
        found_eigenvalues.append(eigenvalue)
        found_statevectors.append(statevector_opt)
        
        if verbose:
            overlap_with_prev = 0.0
            if len(found_eigenvectors) > 1:
                overlap_with_prev = np.abs(np.vdot(found_eigenvectors[-2], eigenvector))
            
            print(f"  Optimization converged: {result.success}")
            print(f"  Iterations: {result.nfev}")
            print(f"  Final cost: {result.fun:.6f}")
            print(f"  Eigenvalue: {eigenvalue:.6f}")
            print(f"  Overlap with previous: {overlap_with_prev:.6e}")
    
    # Convert to array
    U_vqd = np.array(found_eigenvectors)
    eigenvalues_vqd = np.array(found_eigenvalues)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"VQD Complete")
        print(f"{'='*70}")
    
    # Compute diagnostic metrics
    logs = _compute_diagnostics(U_vqd, eigenvalues_vqd, cov, U_pca if validate else None, 
                                classical_eigenvalues[:n_components] if validate else None,
                                verbose)
    
    return U_vqd, eigenvalues_vqd, logs


def _build_quantum_ansatz(theta, num_qubits, depth, entanglement='ladder'):
    """
    Build hardware-efficient quantum ansatz for VQD.
    
    Uses:
    - R_Y rotations for each qubit (parameterized)
    - Entanglement layer (ladder, full, or alternating)
    - Repeated for 'depth' layers
    
    Parameters
    ----------
    theta : ndarray
        Parameters for the circuit
    num_qubits : int
        Number of qubits
    depth : int
        Number of layers
    entanglement : str
        'ladder': CNOT ladder (i, i+1)
        'full': All-to-all CNOT (every pair)
        'alternating': Alternating even-odd pairs
    """
    qc = QuantumCircuit(num_qubits)
    
    param_idx = 0
    for layer in range(depth):
        # R_Y rotations
        for qubit in range(num_qubits):
            qc.ry(theta[param_idx], qubit)
            param_idx += 1
        
        # Entangling layer
        if entanglement == 'ladder':
            # CNOT ladder: 0→1, 1→2, 2→3, ...
            for qubit in range(num_qubits - 1):
                qc.cx(qubit, qubit + 1)
                
        elif entanglement == 'full':
            # Full entanglement: all pairs (expensive but thorough)
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    qc.cx(i, j)
                    
        elif entanglement == 'alternating':
            # Alternating pattern: even pairs, then odd pairs
            if layer % 2 == 0:
                # Even layer: (0,1), (2,3), (4,5), ...
                for qubit in range(0, num_qubits - 1, 2):
                    qc.cx(qubit, qubit + 1)
            else:
                # Odd layer: (1,2), (3,4), (5,6), ...
                for qubit in range(1, num_qubits - 1, 2):
                    qc.cx(qubit, qubit + 1)
        else:
            raise ValueError(f"Unknown entanglement pattern: {entanglement}")
    
    return qc


def procrustes_align(U_vqd, U_pca):
    """
    Compute optimal orthogonal rotation R to align U_VQD with U_PCA.
    
    Solves: min_R ||U_VQD R - U_PCA||_F  s.t. R^T R = I
    
    Solution via SVD: U_VQD^T U_PCA = U Σ V^T, then R = U V^T
    
    Parameters
    ----------
    U_vqd : ndarray, shape (k, n_features)
        VQD basis vectors
    U_pca : ndarray, shape (k, n_features)
        PCA basis vectors
        
    Returns
    -------
    R : ndarray, shape (k, k)
        Optimal rotation matrix
    U_vqd_aligned : ndarray, shape (k, n_features)
        Rotated VQD basis: R^T U_VQD
    residual_before : float
        ||U_VQD - U_PCA||_F before alignment
    residual_after : float
        ||R^T U_VQD - U_PCA||_F after alignment
    """
    # Compute optimal rotation via Procrustes
    # We want: min_R ||R^T U_VQD - U_PCA||_F
    # Solution: M = U_VQD U_PCA^T, M = U Σ V^T, R = U V^T
    M = U_vqd @ U_pca.T  # (k, k)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    
    # Apply rotation: R^T @ U_VQD (each row is rotated)
    U_vqd_aligned = R.T @ U_vqd
    
    # Compute residuals
    residual_before = np.linalg.norm(U_vqd - U_pca, 'fro')
    residual_after = np.linalg.norm(U_vqd_aligned - U_pca, 'fro')
    
    return R, U_vqd_aligned, residual_before, residual_after


def _compute_diagnostics(U_vqd, eigenvalues_vqd, cov, U_pca=None, eigenvalues_pca=None, verbose=True):
    """
    Compute comprehensive diagnostic metrics.
    
    Returns
    -------
    logs : dict
        - orthogonality_error: ||U^T U - I||_F
        - rayleigh_errors: |⟨u_i|C|u_i⟩ - λ_i(PCA)|
        - principal_angles: angles between span(U_VQD) and span(U_PCA) (if U_pca provided)
        - eigenvalue_errors: |λ_i(VQD) - λ_i(PCA)| (if eigenvalues_pca provided)
        - procrustes_rotation: Optimal rotation R to align U_VQD with U_PCA
        - procrustes_residuals: (before, after) alignment residuals
        - U_vqd_aligned: Procrustes-aligned VQD basis
    """
    logs = {}
    
    # 1. Orthogonality error: ||U^T U - I||_F
    gram_matrix = U_vqd @ U_vqd.T
    identity = np.eye(len(U_vqd))
    orthogonality_error = np.linalg.norm(gram_matrix - identity, 'fro')
    logs['orthogonality_error'] = orthogonality_error
    
    # 2. Rayleigh quotient errors
    rayleigh_errors = []
    for i, u in enumerate(U_vqd):
        rayleigh = u @ cov @ u
        if eigenvalues_pca is not None:
            error = np.abs(rayleigh - eigenvalues_pca[i])
        else:
            error = 0.0
        rayleigh_errors.append(error)
    logs['rayleigh_errors'] = np.array(rayleigh_errors)
    logs['rayleigh_quotients'] = [u @ cov @ u for u in U_vqd]
    
    # 3. Principal angles (if classical PCA provided)
    if U_pca is not None:
        # Compute SVD of U_VQD^T @ U_PCA^T
        _, singular_values, _ = np.linalg.svd(U_vqd @ U_pca.T, full_matrices=False)
        # Clamp to [-1, 1] for numerical stability
        singular_values = np.clip(singular_values, -1, 1)
        principal_angles = np.arccos(singular_values)
        logs['principal_angles'] = principal_angles
        logs['principal_angles_deg'] = np.degrees(principal_angles)
        
        # Procrustes alignment
        R, U_vqd_aligned, res_before, res_after = procrustes_align(U_vqd, U_pca)
        logs['procrustes_rotation'] = R
        logs['procrustes_residual_before'] = res_before
        logs['procrustes_residual_after'] = res_after
        logs['procrustes_improvement'] = (res_before - res_after) / res_before
        logs['U_vqd_aligned'] = U_vqd_aligned
    
    # 4. Eigenvalue errors (if classical eigenvalues provided)
    if eigenvalues_pca is not None:
        eigenvalue_errors = np.abs(eigenvalues_vqd - eigenvalues_pca)
        logs['eigenvalue_errors'] = eigenvalue_errors
        logs['eigenvalue_relative_errors'] = eigenvalue_errors / eigenvalues_pca
    
    # Print summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"Diagnostics")
        print(f"{'='*70}")
        print(f"\n1. Orthogonality error ||U^T U - I||_F: {orthogonality_error:.6e}")
        
        print(f"\n2. Rayleigh quotients:")
        for i, rq in enumerate(logs['rayleigh_quotients']):
            print(f"   u_{i+1}: {rq:.6f}", end='')
            if eigenvalues_pca is not None:
                print(f"  (error: {rayleigh_errors[i]:.6e})")
            else:
                print()
        
        if 'principal_angles' in logs:
            print(f"\n3. Principal angles (degrees):")
            print(f"   {logs['principal_angles_deg']}")
            print(f"   Mean: {np.mean(logs['principal_angles_deg']):.2f}°")
            print(f"   Max:  {np.max(logs['principal_angles_deg']):.2f}°")
            
            # Procrustes analysis
            print(f"\n4. Procrustes alignment (subspace comparison):")
            print(f"   Residual before: ||U_VQD - U_PCA||_F = {logs['procrustes_residual_before']:.6f}")
            print(f"   Residual after:  ||U_VQD R - U_PCA||_F = {logs['procrustes_residual_after']:.6f}")
            print(f"   Improvement: {logs['procrustes_improvement']*100:.1f}%")
            if logs['procrustes_improvement'] > 0.5:
                print(f"   ✅ High improvement → span is similar, just rotated!")
            else:
                print(f"   ⚠️  Low improvement → subspaces differ in content")
        
        if 'eigenvalue_errors' in logs:
            print(f"\n5. Eigenvalue errors:")
            for i, err in enumerate(logs['eigenvalue_errors']):
                rel_err = logs['eigenvalue_relative_errors'][i]
                print(f"   λ_{i+1}: {err:.6f} ({rel_err*100:.2f}%)")
    
    return logs


def main():
    """Test VQD PCA on synthetic and real data."""
    
    print("\n" + "="*70)
    print("VQD Quantum PCA - Test Suite")
    print("="*70)
    
    # Test 1: Small synthetic data
    print("\n" + "="*70)
    print("Test 1: Synthetic data (20 samples × 4 features)")
    print("="*70)
    
    np.random.seed(42)
    n_samples = 20
    n_features = 4
    
    # Create data with known structure
    X = np.random.randn(n_samples, n_features)
    X[:, 0] = X[:, 0] * 3  # First component has high variance
    X[:, 1] = X[:, 1] * 2  # Second component
    
    U_vqd, eigenvalues_vqd, logs = vqd_quantum_pca(
        X, 
        n_components=3,
        max_depth=2,
        penalty_scale=10.0,
        maxiter=100,
        verbose=True,
        validate=True
    )
    
    # Test 2: Real MSR data (if available)
    print("\n\n" + "="*70)
    print("Test 2: Real MSR Action3D data")
    print("="*70)
    
    data_path = Path(__file__).parent.parent / "data" / "features.npy"
    if data_path.exists():
        X_real = np.load(data_path)
        print(f"Loaded data: {X_real.shape}")
        
        # Subsample to keep it tractable
        indices = np.random.choice(len(X_real), size=min(50, len(X_real)), replace=False)
        X_real_sub = X_real[indices]
        
        # Reduce dimensions if needed
        if X_real_sub.shape[1] > 8:
            print(f"Reducing from {X_real_sub.shape[1]} to 8 features using classical PCA first")
            from sklearn.decomposition import PCA
            pca_pre = PCA(n_components=8)
            X_real_sub = pca_pre.fit_transform(X_real_sub)
        
        U_vqd_real, eigenvalues_vqd_real, logs_real = vqd_quantum_pca(
            X_real_sub,
            n_components=4,
            max_depth=2,
            penalty_scale=15.0,
            maxiter=150,
            verbose=True,
            validate=True
        )
        
        print(f"\n✅ VQD PCA on real data complete!")
        print(f"   Orthogonality error: {logs_real['orthogonality_error']:.6e}")
        print(f"   Mean principal angle: {np.mean(logs_real['principal_angles_deg']):.2f}°")
        print(f"   Mean eigenvalue error: {np.mean(logs_real['eigenvalue_errors']):.6f}")
    else:
        print(f"⚠️  Data file not found: {data_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ All tests complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
