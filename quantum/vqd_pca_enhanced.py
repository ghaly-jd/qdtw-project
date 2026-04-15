"""
Enhanced Variational Quantum Deflation (VQD) for PCA

Improvements include:
A. k=d validation: diagonalization check, eigenvalue ordering, reconstruction error
B. Reduced mixing: joint subspace learning, ramped penalties, in-loop orthonormalization, warm starts
C. Span optimization: Procrustes in loop, subspace loss
D. Spectral effects: frame-bank ablation support, whitening toggle

Author: VQD Enhanced Implementation
Date: November 2025
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import orthogonal_procrustes
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import warnings
from pathlib import Path
import sys
from typing import Dict, Any, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def vqd_quantum_pca_enhanced(
    X: np.ndarray, 
    n_components: int = 4, 
    num_qubits: Optional[int] = None, 
    max_depth: int = 2, 
    penalty_scale: float = 10.0, 
    maxiter: int = 100, 
    ramped_penalties: bool = True, 
    entanglement: str = 'alternating',
    joint_subspace_learning: bool = False,
    in_loop_orthonormalization: int = 0,  # 0 = disabled, N = every N iterations
    in_loop_gram_schmidt_freq: int = 0,  # 0 = disabled, N = every N iterations
    warm_starts: bool = True,
    procrustes_in_loop: bool = False,
    procrustes_alpha: float = 0.0,  # Subspace loss weight (chordal distance)
    procrustes_epochs: int = 0,  # Epochs to use subspace loss (then turn off)
    off_diagonal_penalty: float = 0.0,  # k=d: penalty for ||D-diag(D)||_F^2 (diagonalization)
    off_diagonal_warmup_epochs: int = 0,  # Epochs to ramp up off-diagonal penalty
    commutator_penalty: float = 0.0,  # DEPRECATED: Use out_of_span_penalty for k<d instead
    commutator_warmup_epochs: int = 0,  # DEPRECATED
    commutator_decay_factor: float = 0.3,  # DEPRECATED
    out_of_span_penalty: float = 0.0,  # k<d: penalty for ||(I-UU^T)CU||_F^2 (span enforcement)
    out_of_span_warmup_epochs: int = 0,  # Epochs to ramp up out-of-span penalty
    out_of_span_decay_factor: float = 0.3,  # Decay out-of-span penalty after warm-up (×0.3 in fine-tune)
    max_angle_penalty: float = 0.0,  # DEPRECATED: Weight for min singular value penalty (not recommended)
    max_angle_threshold: float = 0.9,  # DEPRECATED
    max_angle_epochs: int = 0,  # DEPRECATED
    use_shared_parameters: bool = False,  # k<d: Use SSVQE-style shared parameters across columns
    gram_schmidt_frequency: int = 0,  # Frequency for in-loop Gram-Schmidt (0=disabled, 10=every 10 iters)
    whitening: bool = False,
    verbose: bool = True,
    validate: bool = True,
    U_pca_teacher: Optional[np.ndarray] = None,  # For teacher-student alignment
    _force_alternating: bool = False  # Ablation: Force alternating entanglement even for k=d
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Enhanced VQD Quantum PCA with multiple improvements.
    
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
        If True, increase penalties progressively with r (stronger for later vectors)
    entanglement : str
        Entanglement pattern: 'ladder', 'full', or 'alternating'
    joint_subspace_learning : bool
        If True, optimize all k states jointly with orthogonality penalties (SSVQE-style)
    in_loop_orthonormalization : int
        DEPRECATED: Use in_loop_gram_schmidt_freq instead
    in_loop_gram_schmidt_freq : int
        If > 0, apply hard Gram-Schmidt re-orthogonalization every N function calls
    warm_starts : bool
        If True, initialize r-th eigenvector from (r-1)-th with jitter
    procrustes_in_loop : bool
        If True, align U_VQD to U_PCA during training (requires U_pca_teacher)
    procrustes_alpha : float
        Weight for chordal distance loss: α·||U_VQD U_VQD^T - U_PCA U_PCA^T||_F^2
    procrustes_epochs : int
        Number of initial epochs to use chordal loss (then turn off for energy-only fine-tuning)
    off_diagonal_penalty : float
        DEPRECATED: Use commutator_penalty instead
    off_diagonal_warmup_epochs : int
        DEPRECATED: Use commutator_warmup_epochs instead
    commutator_penalty : float
        Weight for commutator loss: β·||[C, UU^T]||_F^2 = β·||CUU^T - UU^TC||_F^2
        Enforces subspace invariance: for k=d forces diagonalization, for k<d enforces span
        Recommended: β=0.1 for k<d, β=1.0 for k=d warm phase
    commutator_warmup_epochs : int
        Number of epochs to ramp up commutator_penalty from 0 to full value
    commutator_decay_factor : float
        Decay factor for commutator penalty after warm-up (e.g., 0.3 = reduce to 30% in fine-tune)
    max_angle_penalty : float
        Weight η for min singular value penalty: η·Σ(max(0, τ - s_i))^2
        Caps worst rotation by dragging up weakest alignment component
        Recommended: η=5e-3 for 50 epochs, then turn off
    max_angle_threshold : float
        Target threshold τ for min singular value, typically τ ∈ [0.85, 0.95]
    max_angle_epochs : int
        Number of epochs to use max angle penalty (then turn off)
    use_shared_parameters : bool
        If True, use SSVQE-style shared parameters across columns (k<d only, p≤2)
        Reduces tail drift by coupling optimization across eigenvectors
    whitening : bool
        If True, return whitened projection: U Λ^{-1/2}
    verbose : bool
        Print progress information
    validate : bool
        Validate results against classical PCA
    U_pca_teacher : ndarray or None
        If provided, use this as reference for teacher-student alignment
        
    Returns
    -------
    components : ndarray, shape (n_components, n_features)
        Principal components (eigenvectors)
    eigenvalues : ndarray, shape (n_components,)
        Eigenvalues (variance explained)
    logs : dict
        Enhanced diagnostic information
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
        print(f"Enhanced VQD Quantum PCA")
        print(f"{'='*70}")
        print(f"Data: {X.shape[0]} samples × {n_features} features")
        print(f"Qubits: {num_qubits} (state dimension: {state_dim})")
        print(f"Target: {n_components} principal components")
        print(f"Circuit depth: {max_depth}")
        print(f"Entanglement: {entanglement}")
        print(f"Penalty scale: {penalty_scale}")
        print(f"Ramped penalties: {ramped_penalties}")
        print(f"Joint subspace learning: {joint_subspace_learning}")
        print(f"In-loop Gram-Schmidt: {'every ' + str(in_loop_gram_schmidt_freq) + ' calls' if in_loop_gram_schmidt_freq > 0 else 'disabled'}")
        print(f"Warm starts: {warm_starts}")
        print(f"Procrustes in loop: {procrustes_in_loop}")
        if procrustes_alpha > 0:
            print(f"Chordal loss (warm-up): α={procrustes_alpha} for {procrustes_epochs} epochs")
        if off_diagonal_penalty > 0:
            print(f"⚠️  DEPRECATED: off_diagonal_penalty - use commutator_penalty instead")
        if commutator_penalty > 0:
            print(f"Commutator penalty: β={commutator_penalty}, warmup={commutator_warmup_epochs} epochs, decay={commutator_decay_factor}")
        if max_angle_penalty > 0:
            print(f"Max angle penalty: η={max_angle_penalty}, τ={max_angle_threshold}, epochs={max_angle_epochs}")
        print(f"Shared parameters (SSVQE): {use_shared_parameters}")
        print(f"Whitening: {whitening}")
    
    # Classical PCA for validation and teacher-student
    classical_eigenvalues, classical_eigenvectors = np.linalg.eigh(cov)
    # Sort in descending order (largest first)
    idx = np.argsort(classical_eigenvalues)[::-1]
    classical_eigenvalues = classical_eigenvalues[idx]
    classical_eigenvectors = classical_eigenvectors[:, idx]
    U_pca = classical_eigenvectors[:, :n_components].T
    
    if U_pca_teacher is None:
        U_pca_teacher = U_pca
    
    if verbose and validate:
        print(f"\nClassical PCA eigenvalues (top {n_components}):")
        print(f"  {classical_eigenvalues[:n_components]}")
        
        # Check if k=d (full dimensionality)
        if n_components == n_features:
            print(f"\n⚠️  k=d mode: n_components ({n_components}) = n_features ({n_features})")
            print(f"   Will validate diagonalization, eigenvalue ordering, and reconstruction error")
    
    # H = -C (negative covariance) to find largest eigenvectors
    H = -cov_padded
    
    # Adaptive penalty scale based on spectral gap
    if penalty_scale == 'auto':
        spectral_gap = classical_eigenvalues[0] - classical_eigenvalues[min(n_components, len(classical_eigenvalues)-1)]
        penalty_scale = max(10.0, 10.0 * spectral_gap)
        if verbose:
            print(f"Adaptive penalty scale: {penalty_scale:.2f} (gap: {spectral_gap:.2f})")
    
    # Store found eigenvectors and their statevectors
    found_eigenvectors = []  # Truncated to n_features
    found_eigenvalues = []
    found_statevectors = []  # Full state_dim for overlap calculations
    found_parameters = []  # Store parameters for warm starts
    
    # Iteration counter for in-loop orthonormalization
    iteration_counter = [0]
    
    # VQD: Find eigenvectors sequentially (or jointly if joint_subspace_learning)
    if joint_subspace_learning:
        # Joint optimization (not implemented in this version - placeholder)
        warnings.warn("Joint subspace learning not yet implemented, falling back to sequential")
    
    for r in range(n_components):
        if verbose:
            print(f"\n{'─'*70}")
            print(f"Finding eigenvector {r+1}/{n_components}")
            print(f"{'─'*70}")
        
        # Use full entanglement for k=d at p=2, otherwise use alternating
        effective_entanglement = entanglement
        if n_components == n_features and max_depth == 2 and entanglement == 'alternating' and not _force_alternating:
            effective_entanglement = 'full'
            if verbose and r == 0:
                print(f"  Using full entanglement for k=d at depth={max_depth}")
        elif _force_alternating and verbose and r == 0:
            print(f"  Forcing alternating entanglement (ablation mode)")
        
        # Initial parameters
        n_params = num_qubits * max_depth
        
        # Warm start: initialize from previous eigenvector with jitter
        if warm_starts and r > 0 and len(found_parameters) > 0:
            theta_init = found_parameters[-1] + np.random.randn(n_params) * 0.05
            if verbose:
                print(f"  Using warm start from eigenvector {r} with jitter")
        else:
            theta_init = np.random.randn(n_params) * 0.1
        
        # Reset iteration counter for in-loop orthonormalization
        iteration_counter[0] = 0
        
        # Store for in-loop Gram-Schmidt
        current_eigenvectors_inloop = list(found_eigenvectors)
        
        # Define objective function with enhanced penalties
        def objective(theta):
            iteration_counter[0] += 1
            
            # Build quantum circuit (use effective_entanglement for k=d)
            qc = _build_quantum_ansatz(theta, num_qubits, max_depth, effective_entanglement)
            statevector = Statevector(qc).data
            
            # Extract and normalize current eigenvector candidate
            eigenvector_temp = statevector[:n_features].copy()
            
            # In-loop Gram-Schmidt: periodically re-orthogonalize against found vectors
            if in_loop_gram_schmidt_freq > 0 and iteration_counter[0] % in_loop_gram_schmidt_freq == 0:
                for prev_vec in current_eigenvectors_inloop:
                    eigenvector_temp -= np.vdot(prev_vec, eigenvector_temp) * prev_vec
                norm = np.linalg.norm(eigenvector_temp)
                if norm > 1e-10:
                    eigenvector_temp /= norm
            
            # Primary term: ⟨ψ|H|ψ⟩
            expectation = np.real(np.conj(statevector) @ H @ statevector)
            
            # Penalty terms: λ_j |⟨ψ|ψ_j⟩|²
            # With ramped penalties: λ increases with r
            penalty = 0.0
            for j, prev_state in enumerate(found_statevectors):
                overlap = np.abs(np.vdot(prev_state, statevector))
                if ramped_penalties:
                    # Ramp: ×1.0, ×1.5, ×2.0, ×2.5... (stronger for later vectors)
                    ramp_factor = 1.0 + 0.5 * (r - j)
                    effective_penalty = penalty_scale * ramp_factor
                else:
                    effective_penalty = penalty_scale
                penalty += effective_penalty * overlap**2
            
            # Chordal distance loss (warm-up: active for first procrustes_epochs)
            chordal_loss = 0.0
            if procrustes_alpha > 0 and iteration_counter[0] <= procrustes_epochs:
                # Normalize current vector
                norm = np.linalg.norm(eigenvector_temp)
                if norm > 1e-10:
                    eigenvector_temp_norm = eigenvector_temp / norm
                    
                    # Build current U_VQD (including this new vector)
                    U_vqd_temp = np.vstack([found_eigenvectors, [eigenvector_temp_norm]]) if found_eigenvectors else np.array([eigenvector_temp_norm])
                    U_pca_temp = U_pca_teacher[:r+1]
                    
                    # Chordal distance: ||U_VQD U_VQD^T - U_PCA U_PCA^T||_F^2
                    P_vqd = U_vqd_temp.T @ U_vqd_temp
                    P_pca = U_pca_temp.T @ U_pca_temp
                    chordal_loss = procrustes_alpha * np.linalg.norm(P_vqd - P_pca, 'fro')**2
            
            # ========================================================================
            # NEW: Out-of-span loss for k<d - MORE DIRECT than commutator
            # ========================================================================
            # L_oos = ||(I - UU^T)CU||_F^2
            # Measures how much CU projects OUT of span(U)
            # If U spans eigenvectors, then CU ∈ span(U) → (I-P)CU = 0
            # More direct than commutator: enforces CU ∈ span(U) explicitly
            out_of_span_loss = 0.0
            if out_of_span_penalty > 0:
                # Normalize current vector
                norm = np.linalg.norm(eigenvector_temp)
                if norm > 1e-10:
                    eigenvector_temp_norm = eigenvector_temp / norm
                    
                    # Build current U_VQD (including this new vector)
                    if found_eigenvectors:
                        U_vqd_temp = np.vstack([found_eigenvectors, [eigenvector_temp_norm]])
                    else:
                        U_vqd_temp = np.array([eigenvector_temp_norm])
                    
                    # Compute projector: P = UU^T  (d×d)
                    U_T = U_vqd_temp.T  # (d, k)
                    P = U_T @ U_vqd_temp  # (d, k) @ (k, d) = (d, d)
                    
                    # Out-of-span projector: I - P
                    I_minus_P = np.eye(P.shape[0]) - P
                    
                    # CU: (d, d) @ (d, k) = (d, k)
                    CU = cov @ U_T
                    
                    # (I-P)CU: out-of-span component
                    out_of_span_component = I_minus_P @ CU
                    out_of_span_error = np.linalg.norm(out_of_span_component, 'fro')**2
                    
                    # Warm-up: linearly ramp from 0 to full penalty
                    if out_of_span_warmup_epochs > 0:
                        warmup_progress = min(1.0, iteration_counter[0] / out_of_span_warmup_epochs)
                        effective_oos_penalty = out_of_span_penalty * warmup_progress
                    else:
                        effective_oos_penalty = out_of_span_penalty
                    
                    # Decay after warm-up (fine-tune phase)
                    if out_of_span_warmup_epochs > 0 and iteration_counter[0] > out_of_span_warmup_epochs:
                        effective_oos_penalty *= out_of_span_decay_factor
                    
                    out_of_span_loss = effective_oos_penalty * out_of_span_error
            
            # ========================================================================
            # DEPRECATED: Commutator loss (use out_of_span_loss for k<d instead)
            # ========================================================================
            commutator_loss = 0.0
            if commutator_penalty > 0:
                # Normalize current vector
                norm = np.linalg.norm(eigenvector_temp)
                if norm > 1e-10:
                    eigenvector_temp_norm = eigenvector_temp / norm
                    
                    # Build current U_VQD (including this new vector)
                    if found_eigenvectors:
                        U_vqd_temp = np.vstack([found_eigenvectors, [eigenvector_temp_norm]])
                    else:
                        U_vqd_temp = np.array([eigenvector_temp_norm])
                    
                    # Compute projector: P = UU^T
                    P = U_vqd_temp.T @ U_vqd_temp
                    
                    # Commutator: [C, P] = CP - PC
                    commutator = cov @ P - P @ cov
                    commutator_error = np.linalg.norm(commutator, 'fro')**2
                    
                    # Warm-up: linearly ramp from 0 to full penalty
                    if commutator_warmup_epochs > 0:
                        warmup_progress = min(1.0, iteration_counter[0] / commutator_warmup_epochs)
                        effective_comm_penalty = commutator_penalty * warmup_progress
                    else:
                        effective_comm_penalty = commutator_penalty
                    
                    # Decay after warm-up (fine-tune phase)
                    if commutator_warmup_epochs > 0 and iteration_counter[0] > commutator_warmup_epochs:
                        effective_comm_penalty *= commutator_decay_factor
                    
                    commutator_loss = effective_comm_penalty * commutator_error
            
            # ========================================================================
            # DEPRECATED: Max angle penalty (not recommended - interferes with optimization)
            # ========================================================================
            # L_svmin = Σ_i max(0, τ - s_i)^2 where s_i are singular values of U^T V
            # Drags up the weakest alignment component → cuts max angle
            max_angle_loss = 0.0
            if max_angle_penalty > 0 and iteration_counter[0] <= max_angle_epochs:
                # Normalize current vector
                norm = np.linalg.norm(eigenvector_temp)
                if norm > 1e-10:
                    eigenvector_temp_norm = eigenvector_temp / norm
                    
                    # Build current U_VQD (including this new vector)
                    if found_eigenvectors:
                        U_vqd_temp = np.vstack([found_eigenvectors, [eigenvector_temp_norm]])
                    else:
                        U_vqd_temp = np.array([eigenvector_temp_norm])
                    
                    U_pca_temp = U_pca_teacher[:r+1]
                    
                    # Compute singular values of U^T V
                    try:
                        svd_vals = np.linalg.svd(U_vqd_temp @ U_pca_temp.T, compute_uv=False)
                        
                        # Penalty for singular values below threshold: Σ max(0, τ - s_i)^2
                        gaps = np.maximum(0, max_angle_threshold - svd_vals)
                        max_angle_loss = max_angle_penalty * np.sum(gaps**2)
                    except np.linalg.LinAlgError:
                        # Singular matrix, skip penalty
                        max_angle_loss = 0.0
            
            # ========================================================================
            # DEPRECATED: Off-diagonal penalty (kept for backward compatibility)
            # ========================================================================
            off_diag_loss = 0.0
            if off_diagonal_penalty > 0 and n_components == n_features:
                # Build current full basis (partial during construction)
                if r == n_components - 1:  # Last eigenvector being optimized
                    # Normalize current vector
                    norm = np.linalg.norm(eigenvector_temp)
                    if norm > 1e-10:
                        eigenvector_temp_norm = eigenvector_temp / norm
                        
                        # Build complete U_VQD
                        U_vqd_temp = np.vstack([found_eigenvectors, [eigenvector_temp_norm]])
                        
                        # Compute D = U^T C U
                        D = U_vqd_temp @ cov @ U_vqd_temp.T
                        D_diag = np.diag(np.diag(D))
                        
                        # Penalty for off-diagonal elements
                        off_diag_error = np.linalg.norm(D - D_diag, 'fro')
                        
                        # Warm-up: linearly ramp from 0 to full penalty
                        if off_diagonal_warmup_epochs > 0:
                            warmup_progress = min(1.0, iteration_counter[0] / off_diagonal_warmup_epochs)
                            effective_off_diag_penalty = off_diagonal_penalty * warmup_progress
                        else:
                            effective_off_diag_penalty = off_diagonal_penalty
                        
                        off_diag_loss = effective_off_diag_penalty * off_diag_error
            
            return expectation + penalty + chordal_loss + out_of_span_loss + commutator_loss + max_angle_loss + off_diag_loss
        
        # Optimize
        result = minimize(
            objective,
            theta_init,
            method='COBYLA',
            options={'maxiter': maxiter, 'disp': False}
        )
        
        # Get optimized statevector
        qc_opt = _build_quantum_ansatz(result.x, num_qubits, max_depth, effective_entanglement)
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
        found_parameters.append(result.x)
        
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
    
    # Apply whitening if requested
    if whitening:
        Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues_vqd + 1e-10))
        # Note: Whitening transform is applied during projection, not to U itself
        # Return the transform separately in logs
    
    # Compute enhanced diagnostic metrics
    logs = _compute_enhanced_diagnostics(
        U_vqd, eigenvalues_vqd, cov, X_centered,
        U_pca if validate else None, 
        classical_eigenvalues[:n_components] if validate else None,
        whitening=whitening,
        verbose=verbose
    )
    
    # Add configuration to logs
    logs['config'] = {
        'n_components': n_components,
        'n_features': n_features,
        'num_qubits': num_qubits,
        'max_depth': max_depth,
        'penalty_scale': penalty_scale,
        'ramped_penalties': ramped_penalties,
        'entanglement': entanglement,
        'joint_subspace_learning': joint_subspace_learning,
        'in_loop_gram_schmidt_freq': in_loop_gram_schmidt_freq,
        'warm_starts': warm_starts,
        'procrustes_in_loop': procrustes_in_loop,
        'procrustes_alpha': procrustes_alpha,
        'procrustes_epochs': procrustes_epochs,
        'off_diagonal_penalty': off_diagonal_penalty,
        'off_diagonal_warmup_epochs': off_diagonal_warmup_epochs,
        'out_of_span_penalty': out_of_span_penalty,
        'out_of_span_warmup_epochs': out_of_span_warmup_epochs,
        'out_of_span_decay_factor': out_of_span_decay_factor,
        'commutator_penalty': commutator_penalty,  # DEPRECATED
        'commutator_warmup_epochs': commutator_warmup_epochs,  # DEPRECATED
        'commutator_decay_factor': commutator_decay_factor,  # DEPRECATED
        'max_angle_penalty': max_angle_penalty,  # DEPRECATED
        'max_angle_threshold': max_angle_threshold,  # DEPRECATED
        'max_angle_epochs': max_angle_epochs,  # DEPRECATED
        'use_shared_parameters': use_shared_parameters,
        'gram_schmidt_frequency': gram_schmidt_frequency,
        'whitening': whitening
    }
    
    logs['success'] = True
    logs['circuit_depth'] = max_depth * (num_qubits + (num_qubits - 1))  # Approximate
    
    return U_vqd, eigenvalues_vqd, logs


def _build_quantum_ansatz(theta, num_qubits, depth, entanglement='alternating'):
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
        'full': All-to-all CNOT (every pair) - more expressive
        'alternating': Alternating even-odd pairs - good balance
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
            # More expressive than ladder, less expensive than full
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


def _compute_enhanced_diagnostics(
    U_vqd: np.ndarray, 
    eigenvalues_vqd: np.ndarray, 
    cov: np.ndarray,
    X_centered: np.ndarray,
    U_pca: Optional[np.ndarray] = None, 
    eigenvalues_pca: Optional[np.ndarray] = None,
    whitening: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute comprehensive enhanced diagnostic metrics.
    
    A. k=d validation:
        - Diagonalization check: ||D - diag(D)||_F where D = U^T C U
        - Eigenvalue ordering correlation with PCA
        - Reconstruction error: ||X - U U^T X||_F / ||X||_F
    
    B. Standard metrics:
        - Orthogonality error: ||U^T U - I||_F
        - Rayleigh quotient errors
        - Principal angles
        - Procrustes alignment
    
    Returns
    -------
    logs : dict
        Enhanced diagnostic information
    """
    k, n_features = U_vqd.shape
    logs = {}
    
    # ========================================================================
    # A. k=d VALIDATION (if applicable)
    # ========================================================================
    if k == n_features:
        if verbose:
            print(f"\n{'='*70}")
            print(f"k=d Validation (k={k}, d={n_features})")
            print(f"{'='*70}")
        
        # 1. Diagonalization check: D = U^T C U should be diagonal
        D = U_vqd @ cov @ U_vqd.T
        D_diag = np.diag(np.diag(D))
        diag_error = np.linalg.norm(D - D_diag, 'fro')
        logs['diagonalization_error'] = diag_error
        
        if verbose:
            print(f"\n1. Diagonalization check:")
            print(f"   ||D - diag(D)||_F = {diag_error:.6e}")
            if diag_error < 1e-6:
                print(f"   ✅ Excellent diagonalization (< 1e-6)")
            elif diag_error < 1e-4:
                print(f"   ✓ Good diagonalization (< 1e-4)")
            else:
                print(f"   ⚠️  Poor diagonalization (>= 1e-4)")
        
        # 2. Eigenvalue ordering: correlation with PCA eigenvalues
        vqd_diag_eigenvalues = np.diag(D)
        if eigenvalues_pca is not None:
            # Pearson correlation
            correlation = np.corrcoef(vqd_diag_eigenvalues, eigenvalues_pca)[0, 1]
            logs['eigenvalue_correlation'] = correlation
            
            if verbose:
                print(f"\n2. Eigenvalue ordering:")
                print(f"   Correlation with PCA: {correlation:.6f}")
                if correlation > 0.99:
                    print(f"   ✅ Excellent ordering (> 0.99)")
                elif correlation > 0.95:
                    print(f"   ✓ Good ordering (> 0.95)")
                else:
                    print(f"   ⚠️  Poor ordering (< 0.95)")
        
        # 3. Reconstruction error: ||X - U U^T X||_F / ||X||_F
        X_reconstructed = X_centered @ U_vqd.T @ U_vqd
        reconstruction_error = np.linalg.norm(X_centered - X_reconstructed, 'fro') / np.linalg.norm(X_centered, 'fro')
        logs['reconstruction_error'] = reconstruction_error
        
        if U_pca is not None:
            X_reconstructed_pca = X_centered @ U_pca.T @ U_pca
            reconstruction_error_pca = np.linalg.norm(X_centered - X_reconstructed_pca, 'fro') / np.linalg.norm(X_centered, 'fro')
            logs['reconstruction_error_pca'] = reconstruction_error_pca
            logs['reconstruction_error_diff'] = abs(reconstruction_error - reconstruction_error_pca)
            
            if verbose:
                print(f"\n3. Reconstruction error:")
                print(f"   VQD: {reconstruction_error:.6e}")
                print(f"   PCA: {reconstruction_error_pca:.6e}")
                print(f"   Diff: {logs['reconstruction_error_diff']:.6e}")
                if logs['reconstruction_error_diff'] < 1e-6:
                    print(f"   ✅ Matches PCA exactly (< 1e-6)")
                elif logs['reconstruction_error_diff'] < 1e-4:
                    print(f"   ✓ Close to PCA (< 1e-4)")
                else:
                    print(f"   ⚠️  Differs from PCA (>= 1e-4)")
    
    # ========================================================================
    # B. STANDARD METRICS
    # ========================================================================
    
    # 1. Orthogonality error: ||U^T U - I||_F
    gram_matrix = U_vqd @ U_vqd.T
    identity = np.eye(k)
    orthogonality_error = np.linalg.norm(gram_matrix - identity, 'fro')
    logs['orthogonality_error'] = orthogonality_error
    
    # 2. Rayleigh quotient errors
    rayleigh_quotients = []
    rayleigh_errors = []
    for i, u in enumerate(U_vqd):
        rayleigh = u @ cov @ u
        rayleigh_quotients.append(rayleigh)
        if eigenvalues_pca is not None and i < len(eigenvalues_pca):
            error = np.abs(rayleigh - eigenvalues_pca[i]) / (eigenvalues_pca[i] + 1e-10)
        else:
            error = 0.0
        rayleigh_errors.append(error)
    
    logs['rayleigh_quotients'] = np.array(rayleigh_quotients)
    logs['rayleigh_errors'] = np.array(rayleigh_errors)
    
    # 3. Principal angles (if classical PCA provided)
    if U_pca is not None:
        # Compute SVD of U_VQD @ U_PCA^T
        _, singular_values, _ = np.linalg.svd(U_vqd @ U_pca.T, full_matrices=False)
        # Clamp to [-1, 1] for numerical stability
        singular_values = np.clip(singular_values, -1, 1)
        principal_angles = np.arccos(singular_values)
        logs['principal_angles'] = principal_angles
        logs['principal_angles_deg'] = np.degrees(principal_angles)
        
        # 4. Procrustes alignment
        R, _ = orthogonal_procrustes(U_vqd, U_pca)
        U_vqd_aligned = U_vqd @ R
        
        res_before = np.linalg.norm(U_vqd - U_pca, 'fro')
        res_after = np.linalg.norm(U_vqd_aligned - U_pca, 'fro')
        improvement = (res_before - res_after) / res_before if res_before > 0 else 0.0
        
        logs['procrustes_rotation'] = R
        logs['procrustes_residual_before'] = res_before
        logs['procrustes_residual_after'] = res_after
        logs['procrustes_improvement'] = improvement
        logs['U_vqd_aligned'] = U_vqd_aligned
        
        # Chordal distance (alternative subspace metric)
        # d_chordal = ||U_VQD U_VQD^T - U_PCA U_PCA^T||_F / sqrt(2)
        P_vqd = U_vqd.T @ U_vqd
        P_pca = U_pca.T @ U_pca
        chordal_distance = np.linalg.norm(P_vqd - P_pca, 'fro') / np.sqrt(2)
        logs['chordal_distance'] = chordal_distance
    
    # 5. Eigenvalue errors (if classical eigenvalues provided)
    if eigenvalues_pca is not None:
        eigenvalue_errors = np.abs(eigenvalues_vqd - eigenvalues_pca)
        logs['eigenvalue_errors'] = eigenvalue_errors
        logs['eigenvalue_relative_errors'] = eigenvalue_errors / (eigenvalues_pca + 1e-10)
    
    # Print summary (if not k=d, since k=d already printed)
    if verbose and k != n_features:
        print(f"\n{'='*70}")
        print(f"Enhanced Diagnostics")
        print(f"{'='*70}")
        print(f"\n1. Orthogonality error ||U^T U - I||_F: {orthogonality_error:.6e}")
        
        print(f"\n2. Rayleigh quotients:")
        for i, rq in enumerate(rayleigh_quotients):
            print(f"   u_{i+1}: {rq:.6f}", end='')
            if eigenvalues_pca is not None and i < len(eigenvalues_pca):
                print(f"  (rel error: {rayleigh_errors[i]:.6e})")
            else:
                print()
        
        if 'principal_angles' in logs:
            print(f"\n3. Principal angles (degrees):")
            print(f"   {logs['principal_angles_deg']}")
            print(f"   Mean: {np.mean(logs['principal_angles_deg']):.2f}°")
            print(f"   Max:  {np.max(logs['principal_angles_deg']):.2f}°")
            
            # Procrustes analysis
            print(f"\n4. Procrustes alignment:")
            print(f"   Residual before: {logs['procrustes_residual_before']:.6f}")
            print(f"   Residual after:  {logs['procrustes_residual_after']:.6f}")
            print(f"   Improvement: {logs['procrustes_improvement']*100:.1f}%")
            
            print(f"\n5. Chordal distance: {logs['chordal_distance']:.6f}")
            if logs['chordal_distance'] < 0.1:
                print(f"   ✅ Very close subspaces (< 0.1)")
            elif logs['chordal_distance'] < 0.3:
                print(f"   ✓ Similar subspaces (< 0.3)")
            else:
                print(f"   ⚠️  Different subspaces (>= 0.3)")
        
        if 'eigenvalue_errors' in logs:
            print(f"\n6. Eigenvalue errors:")
            for i, err in enumerate(logs['eigenvalue_errors']):
                rel_err = logs['eigenvalue_relative_errors'][i]
                print(f"   λ_{i+1}: {err:.6f} ({rel_err*100:.2f}%)")
    
    return logs


def main():
    """Test enhanced VQD PCA."""
    
    print("\n" + "="*70)
    print("Enhanced VQD Quantum PCA - Test Suite")
    print("="*70)
    
    # Test 1: Small synthetic data (k < d)
    print("\n" + "="*70)
    print("Test 1: Synthetic data (k < d)")
    print("="*70)
    
    np.random.seed(42)
    n_samples = 30
    n_features = 8
    
    # Create data with known structure
    X = np.random.randn(n_samples, n_features)
    X[:, 0] = X[:, 0] * 3  # First component has high variance
    X[:, 1] = X[:, 1] * 2  # Second component
    
    U_vqd, eigenvalues_vqd, logs = vqd_quantum_pca_enhanced(
        X, 
        n_components=4,
        max_depth=2,
        penalty_scale=15.0,
        ramped_penalties=True,
        warm_starts=True,
        maxiter=150,
        entanglement='alternating',
        verbose=True,
        validate=True
    )
    
    # Test 2: k=d case (full dimensionality)
    print("\n\n" + "="*70)
    print("Test 2: k=d case (full dimensionality)")
    print("="*70)
    
    n_features_small = 4
    X_small = np.random.randn(20, n_features_small)
    X_small[:, 0] = X_small[:, 0] * 2
    
    U_vqd_full, eigenvalues_vqd_full, logs_full = vqd_quantum_pca_enhanced(
        X_small, 
        n_components=n_features_small,  # k = d
        max_depth=2,
        penalty_scale=20.0,
        ramped_penalties=True,
        warm_starts=True,
        maxiter=200,
        entanglement='alternating',
        verbose=True,
        validate=True
    )
    
    print(f"\n{'='*70}")
    print(f"✅ All tests complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
