"""
Experiment 3: VQD Quantum PCA vs Classical PCA
Comprehensive subspace learning comparison with statistical tests
"""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from experiments.base import (
    ExperimentBase,
    QuantumResourceTracker,
    load_msr_data
)
from experiments.dtw_parallel import DTWClassifierParallel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.linalg import orthogonal_procrustes
from scipy.stats import chi2
import time
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def compute_principal_angles(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """
    Compute principal angles between two subspaces
    
    Args:
        U1, U2: Orthonormal basis matrices (n, k)
    
    Returns:
        angles in degrees
    """
    # Compute SVD of U1^T @ U2
    _, s, _ = np.linalg.svd(U1.T @ U2, full_matrices=False)
    
    # Clamp to [-1, 1] for numerical stability
    s = np.clip(s, -1, 1)
    
    # Convert to angles
    angles = np.arccos(s) * 180 / np.pi
    
    return angles


def orthogonality_error(U: np.ndarray) -> float:
    """Compute Frobenius norm of U^T U - I"""
    k = U.shape[1]
    I = np.eye(k)
    return float(np.linalg.norm(U.T @ U - I, 'fro'))


def rayleigh_quotient_errors(X: np.ndarray, U: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute relative errors in eigenvalues via Rayleigh quotient
    
    Args:
        X: Data matrix (n, d) - CENTERED
        U: Eigenvectors (d, k)
        eigenvalues: Expected eigenvalues (k,)
    
    Returns:
        Relative errors (k,)
    """
    C = X.T @ X / (len(X) - 1)  # Covariance
    k = U.shape[1]
    errors = np.zeros(k)
    
    for i in range(k):
        u = U[:, i]
        rayleigh = u @ C @ u / (u @ u)
        errors[i] = abs(rayleigh - eigenvalues[i]) / (eigenvalues[i] + 1e-10)
    
    return errors


def mcnemar_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> Dict[str, float]:
    """
    McNemar test for paired predictions
    
    Returns:
        {'statistic': float, 'p_value': float, 'significant': bool}
    """
    # Contingency table
    # b: pred1 correct, pred2 wrong
    # c: pred1 wrong, pred2 correct
    b = np.sum((pred1 == y_true) & (pred2 != y_true))
    c = np.sum((pred1 != y_true) & (pred2 == y_true))
    
    # McNemar statistic with continuity correction
    if b + c == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(statistic, df=1)
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'n_only_pred1_correct': int(b),
        'n_only_pred2_correct': int(c)
    }


def vqd_quantum_pca_wrapper(
    X_train: np.ndarray,
    n_components: int,
    num_qubits: int = 3,
    max_depth: int = 2,
    ramped_penalties: bool = True,
    warm_starts: bool = True,
    entanglement: str = 'alternating',
    in_loop_gram_schmidt_freq: int = 0,
    off_diagonal_penalty: float = 0.0,
    off_diagonal_warmup_epochs: int = 100,
    out_of_span_penalty: float = 0.0,
    out_of_span_warmup_epochs: int = 100,
    out_of_span_decay_factor: float = 0.3,
    commutator_penalty: float = 0.0,  # DEPRECATED
    commutator_warmup_epochs: int = 0,  # DEPRECATED
    commutator_decay_factor: float = 0.3,  # DEPRECATED
    max_angle_penalty: float = 0.0,  # DEPRECATED
    max_angle_threshold: float = 0.9,  # DEPRECATED
    max_angle_epochs: int = 0,  # DEPRECATED
    use_shared_parameters: bool = False,
    chordal_loss_alpha: float = 0.0,
    chordal_loss_epochs: int = 0,
    maxiter: int = 200,
    use_enhanced: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Wrapper for VQD quantum PCA with error handling
    
    Returns:
        U_vqd, eigenvalues, logs
    """
    if use_enhanced:
        from quantum.vqd_pca_enhanced import vqd_quantum_pca_enhanced as vqd_func
    else:
        from quantum.vqd_pca import vqd_quantum_pca as vqd_func
    
    try:
        if use_enhanced:
            U_vqd, eigenvalues, logs = vqd_func(
                X_train,
                n_components=n_components,
                num_qubits=num_qubits,
                max_depth=max_depth,
                ramped_penalties=ramped_penalties,
                warm_starts=warm_starts,
                entanglement=entanglement,
                in_loop_gram_schmidt_freq=in_loop_gram_schmidt_freq,
                off_diagonal_penalty=off_diagonal_penalty,
                off_diagonal_warmup_epochs=off_diagonal_warmup_epochs,
                out_of_span_penalty=out_of_span_penalty,
                out_of_span_warmup_epochs=out_of_span_warmup_epochs,
                out_of_span_decay_factor=out_of_span_decay_factor,
                commutator_penalty=commutator_penalty,
                commutator_warmup_epochs=commutator_warmup_epochs,
                commutator_decay_factor=commutator_decay_factor,
                max_angle_penalty=max_angle_penalty,
                max_angle_threshold=max_angle_threshold,
                max_angle_epochs=max_angle_epochs,
                use_shared_parameters=use_shared_parameters,
                procrustes_alpha=chordal_loss_alpha,
                procrustes_epochs=chordal_loss_epochs,
                maxiter=maxiter,
                validate=True,
                verbose=verbose
            )
        else:
            U_vqd, eigenvalues, logs = vqd_func(
                X_train,
                n_components=n_components,
                num_qubits=num_qubits,
                max_depth=max_depth,
                ramped_penalties=ramped_penalties,
                entanglement=entanglement,
                maxiter=maxiter,
                validate=True,
                verbose=verbose
            )
        
        # VQD returns (k, d), but we need (d, k) for consistency with PCA
        U_vqd = U_vqd.T
        
        return U_vqd, eigenvalues, logs
    
    except Exception as e:
        print(f"  ⚠️  VQD failed: {e}")
        import traceback
        traceback.print_exc()
        # Return dummy values to allow experiment to continue
        d = X_train.shape[1]
        U_vqd = np.eye(d, n_components)
        eigenvalues = np.ones(n_components)
        logs = {'error': str(e), 'success': False}
        return U_vqd, eigenvalues, logs


def run_experiment_3(
    n_train: int = 100,
    n_test: int = 30,
    k_values: list = [6, 8, 10, 12, 16],
    frame_bank_sizes: list = [8],  # Can test [10, 20, 40]
    test_whitening: bool = False,
    use_enhanced_vqd: bool = True,
    vqd_params: Optional[Dict] = None
):
    """
    Experiment 3: VQD Quantum PCA vs Classical PCA
    
    Comprehensive subspace learning comparison with:
    - K-sweep evaluation
    - Principal angles & Procrustes
    - Orthogonality & Rayleigh quotient errors
    - Enhanced k=d validation (diagonalization, reconstruction)
    - McNemar statistical test
    - Runtime breakdown
    """
    print("=" * 80)
    print("EXPERIMENT 3: VQD QUANTUM PCA vs CLASSICAL PCA")
    if use_enhanced_vqd:
        print("(Using ENHANCED VQD with validation fixes)")
    print("=" * 80)
    
    # Initialize experiment
    exp = ExperimentBase(
        experiment_id="exp3_vqd_quantum_pca",
        results_dir="results"
    )
    
    quantum_tracker = QuantumResourceTracker()
    
    # VQD hyperparameters
    if vqd_params is None:
        # Default parameters with new enhancements
        vqd_params = {
            'num_qubits': 3,
            'max_depth': 2,
            'ramped_penalties': True,
            'warm_starts': True,
            'entanglement': 'alternating',
            'in_loop_gram_schmidt_freq': 20,  # Re-orthogonalize every 20 calls
            'off_diagonal_penalty': 5.0,  # For k=d: force diagonalization
            'off_diagonal_warmup_epochs': 100,  # Ramp up over 100 epochs
            'chordal_loss_alpha': 0.05,  # Chordal distance loss for warm-up
            'chordal_loss_epochs': 50,  # Use for first 50 epochs, then turn off
            'maxiter': 300,  # Increased for better convergence
            'use_enhanced': use_enhanced_vqd
        }
    else:
        # Ensure use_enhanced is set
        vqd_params['use_enhanced'] = use_enhanced_vqd
        # Set defaults for new parameters if not provided
        vqd_params.setdefault('in_loop_gram_schmidt_freq', 20)
        vqd_params.setdefault('off_diagonal_penalty', 5.0)
        vqd_params.setdefault('off_diagonal_warmup_epochs', 100)
        vqd_params.setdefault('chordal_loss_alpha', 0.05)
        vqd_params.setdefault('chordal_loss_epochs', 50)
    
    # Load data
    print("\n📊 Loading MSR Action3D data...")
    X, y = load_msr_data()
    
    # Get fixed split
    print(f"\n🔀 Loading fixed split (n_train={n_train}, n_test={n_test})...")
    train_idx, test_idx = exp.get_or_create_split(
        n_samples=len(X),
        n_train=n_train,
        n_test=n_test
    )
    
    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    # Store data info
    exp.manifest['data'] = {
        'dataset': 'MSR-Action3D',
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_train': n_train,
        'n_test': n_test,
        'split_file': str(exp.split_file),
        'split_seed': exp.seed
    }
    
    exp.manifest['vqd_params'] = vqd_params
    
    # Results storage
    all_results = {}
    
    # Loop over frame bank sizes
    for fb_size in frame_bank_sizes:
        print(f"\n{'=' * 80}")
        print(f"FRAME BANK SIZE: {fb_size}D")
        print('=' * 80)
        
        # Fit preprocessing pipeline (train-only)
        print(f"\n🔧 Fitting preprocessing pipeline (train-only)...")
        X_train_fb, X_test_fb, transform_info = exp.fit_transform_pipeline(
            X_train_raw,
            X_test_raw,
            scaler=True,
            frame_bank_dim=fb_size
        )
        
        print(f"  Original: {X_train_raw.shape[1]}D")
        if 'frame_bank' in transform_info:
            print(f"  Frame bank: {X_train_fb.shape[1]}D ({transform_info['frame_bank']['variance_explained']:.3f} variance)")
        else:
            print(f"  Frame bank: {X_train_fb.shape[1]}D (no reduction - using full features)")
        
        # Center data for subspace learning
        X_train_centered = X_train_fb - X_train_fb.mean(axis=0, keepdims=True)
        X_test_centered = X_test_fb - X_test_fb.mean(axis=0, keepdims=True)
        train_mean = X_train_fb.mean(axis=0)
        
        # Loop over k values
        for k in k_values:
            print(f"\n{'=' * 80}")
            print(f"K = {k} (Frame Bank = {fb_size}D)")
            print('=' * 80)
            
            if k > fb_size:
                print(f"  ⚠️  Skipping k={k} (exceeds frame bank {fb_size}D)")
                continue
            
            result_key = f"fb{fb_size}_k{k}"
            result = {
                'frame_bank_size': fb_size,
                'k': k,
                'variance_explained_fb': transform_info.get('frame_bank', {}).get('variance_explained', 1.0)
            }
            
            # ============================================================
            # Classical PCA
            # ============================================================
            print(f"\n📐 Classical PCA (k={k})...")
            
            t_start = time.perf_counter()
            pca = PCA(n_components=k)
            pca.fit(X_train_centered)
            U_pca = pca.components_.T  # (d, k)
            eigenvalues_pca = pca.explained_variance_
            t_pca_fit = (time.perf_counter() - t_start) * 1000
            
            # Project data
            @exp.time_component(f"pca_project_fb{fb_size}_k{k}")
            def project_pca():
                X_train_pca = X_train_centered @ U_pca
                X_test_pca = X_test_centered @ U_pca
                return X_train_pca, X_test_pca
            
            X_train_pca, X_test_pca = project_pca()
            
            # Whitening (optional)
            if test_whitening:
                Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues_pca + 1e-10))
                X_train_pca_white = X_train_pca @ Lambda_inv_sqrt
                X_test_pca_white = X_test_pca @ Lambda_inv_sqrt
            
            # DTW Classification
            print(f"  🔍 Running DTW 1-NN classification (PCA)...")
            
            @exp.time_component(f"dtw_pca_fb{fb_size}_k{k}")
            def classify_pca():
                clf = DTWClassifierParallel(n_jobs=-1)
                clf.fit(X_train_pca, y_train)
                return clf.predict(X_test_pca)
            
            pred_pca = classify_pca()
            acc_pca = float(np.mean(pred_pca == y_test))
            
            # Bootstrap CI for PCA accuracy
            def bootstrap_pca():
                idx = np.random.choice(len(X_test_pca), size=len(X_test_pca), replace=True)
                clf = DTWClassifierParallel(n_jobs=-1)
                clf.fit(X_train_pca, y_train)
                pred = clf.predict(X_test_pca[idx])
                return float(np.mean(pred == y_test[idx]))
            
            acc_pca_ci = exp.bootstrap_ci(bootstrap_pca, n_bootstrap=1000)
            
            # PCA quality metrics
            orth_error_pca = orthogonality_error(U_pca)
            rayleigh_errors_pca = rayleigh_quotient_errors(X_train_centered, U_pca, eigenvalues_pca)
            
            result['pca'] = {
                'accuracy': acc_pca_ci,
                'predictions': pred_pca.tolist(),
                'fit_time_ms': t_pca_fit,
                'variance_explained': float(pca.explained_variance_ratio_.sum()),
                'eigenvalues': eigenvalues_pca.tolist(),
                'orthogonality_error': orth_error_pca,
                'rayleigh_errors': {
                    'mean': float(rayleigh_errors_pca.mean()),
                    'max': float(rayleigh_errors_pca.max()),
                    'values': rayleigh_errors_pca.tolist()
                }
            }
            
            print(f"    Accuracy: {acc_pca_ci['mean']:.3f} [{acc_pca_ci['ci_lower']:.3f}, {acc_pca_ci['ci_upper']:.3f}]")
            print(f"    Variance: {result['pca']['variance_explained']:.3f}")
            print(f"    Fit time: {t_pca_fit:.2f} ms")
            print(f"    Orthogonality: {orth_error_pca:.2e}")
            print(f"    Rayleigh errors: mean={rayleigh_errors_pca.mean():.2e}, max={rayleigh_errors_pca.max():.2e}")
            
            # ============================================================
            # VQD Quantum PCA
            # ============================================================
            print(f"\n🌟 VQD Quantum PCA (k={k})...")
            
            # Set loss functions based on k=d or k<d
            if k == fb_size:
                # k=d: Use off-diagonal penalty for diagonalization
                off_diag_penalty_val = vqd_params.get('off_diagonal_penalty', 0.0)
                out_span_penalty_val = 0.0
                print(f"    k=d mode: off_diagonal_penalty={off_diag_penalty_val}")
            else:
                # k<d: Use out-of-span penalty for span enforcement
                off_diag_penalty_val = 0.0
                out_span_penalty_val = vqd_params.get('out_of_span_penalty', 0.0)
                print(f"    k<d mode: out_of_span_penalty={out_span_penalty_val}")
            
            t_start = time.perf_counter()
            U_vqd, eigenvalues_vqd, vqd_logs = vqd_quantum_pca_wrapper(
                X_train_centered,
                n_components=k,
                num_qubits=vqd_params['num_qubits'],
                max_depth=vqd_params['max_depth'],
                ramped_penalties=vqd_params['ramped_penalties'],
                warm_starts=vqd_params.get('warm_starts', True),
                entanglement=vqd_params.get('entanglement', 'alternating'),
                in_loop_gram_schmidt_freq=vqd_params.get('in_loop_gram_schmidt_freq', 0),
                off_diagonal_penalty=off_diag_penalty_val,
                off_diagonal_warmup_epochs=vqd_params.get('off_diagonal_warmup_epochs', 100),
                out_of_span_penalty=out_span_penalty_val,
                out_of_span_warmup_epochs=vqd_params.get('out_of_span_warmup_epochs', 100),
                out_of_span_decay_factor=vqd_params.get('out_of_span_decay_factor', 0.3),
                commutator_penalty=vqd_params.get('commutator_penalty', 0.0),  # DEPRECATED - keep for backward compat
                commutator_warmup_epochs=vqd_params.get('commutator_warmup_epochs', 100),
                commutator_decay_factor=vqd_params.get('commutator_decay_factor', 0.3),
                max_angle_penalty=vqd_params.get('max_angle_penalty', 0.0),  # DEPRECATED
                max_angle_threshold=vqd_params.get('max_angle_threshold', 0.9),
                max_angle_epochs=vqd_params.get('max_angle_epochs', 50),
                use_shared_parameters=vqd_params.get('use_shared_parameters', False),
                chordal_loss_alpha=vqd_params.get('procrustes_alpha', 0.0),
                chordal_loss_epochs=vqd_params.get('procrustes_epochs', 0),
                maxiter=vqd_params['maxiter'],
                use_enhanced=vqd_params.get('use_enhanced', True),
                verbose=False
            )
            t_vqd_fit = (time.perf_counter() - t_start) * 1000
            
            # Log quantum resources
            quantum_tracker.log_circuit(
                component=f"vqd_k{k}",
                qubits=vqd_params['num_qubits'],
                depth=vqd_logs.get('circuit_depth', 50),
                shots=None,
                backend='AerSimulator'
            )
            
            # Check if VQD succeeded
            if not vqd_logs.get('success', True):
                print(f"  ⚠️  VQD failed, skipping k={k}")
                result['vqd'] = {'error': vqd_logs.get('error', 'Unknown error')}
                all_results[result_key] = result
                continue
            
            # Project data
            @exp.time_component(f"vqd_project_fb{fb_size}_k{k}")
            def project_vqd():
                X_train_vqd = X_train_centered @ U_vqd
                X_test_vqd = X_test_centered @ U_vqd
                return X_train_vqd, X_test_vqd
            
            X_train_vqd, X_test_vqd = project_vqd()
            
            # DTW Classification
            print(f"  🔍 Running DTW 1-NN classification (VQD)...")
            
            @exp.time_component(f"dtw_vqd_fb{fb_size}_k{k}")
            def classify_vqd():
                clf = DTWClassifierParallel(n_jobs=-1)
                clf.fit(X_train_vqd, y_train)
                return clf.predict(X_test_vqd)
            
            pred_vqd = classify_vqd()
            acc_vqd = float(np.mean(pred_vqd == y_test))
            
            # Bootstrap CI for VQD accuracy
            def bootstrap_vqd():
                idx = np.random.choice(len(X_test_vqd), size=len(X_test_vqd), replace=True)
                clf = DTWClassifierParallel(n_jobs=-1)
                clf.fit(X_train_vqd, y_train)
                pred = clf.predict(X_test_vqd[idx])
                return float(np.mean(pred == y_test[idx]))
            
            acc_vqd_ci = exp.bootstrap_ci(bootstrap_vqd, n_bootstrap=1000)
            
            # VQD quality metrics
            orth_error_vqd = orthogonality_error(U_vqd)
            rayleigh_errors_vqd = rayleigh_quotient_errors(X_train_centered, U_vqd, eigenvalues_pca)
            
            # Principal angles BEFORE Procrustes
            angles_before = compute_principal_angles(U_pca, U_vqd)
            
            # Store enhanced diagnostics from VQD logs
            vqd_result = {
                'accuracy': acc_vqd_ci,
                'predictions': pred_vqd.tolist(),
                'fit_time_ms': t_vqd_fit,
                'eigenvalues': eigenvalues_vqd.tolist(),
                'orthogonality_error': orth_error_vqd,
                'rayleigh_errors': {
                    'mean': float(rayleigh_errors_vqd.mean()),
                    'max': float(rayleigh_errors_vqd.max()),
                    'values': rayleigh_errors_vqd.tolist()
                },
                'angles_before_procrustes': {
                    'mean': float(angles_before.mean()),
                    'max': float(angles_before.max()),
                    'values': angles_before.tolist()
                },
                'vqd_logs': vqd_logs
            }
            
            # Add enhanced k=d validation metrics if applicable
            if k == fb_size and 'diagonalization_error' in vqd_logs:
                vqd_result['kd_validation'] = {
                    'diagonalization_error': vqd_logs['diagonalization_error'],
                    'eigenvalue_correlation': vqd_logs.get('eigenvalue_correlation', None),
                    'reconstruction_error': vqd_logs.get('reconstruction_error', None),
                    'reconstruction_error_pca': vqd_logs.get('reconstruction_error_pca', None),
                    'reconstruction_error_diff': vqd_logs.get('reconstruction_error_diff', None)
                }
                
                print(f"\n  📊 k=d Validation Metrics:")
                print(f"    Diagonalization error: {vqd_logs['diagonalization_error']:.6e}")
                if 'eigenvalue_correlation' in vqd_logs:
                    print(f"    Eigenvalue correlation: {vqd_logs['eigenvalue_correlation']:.6f}")
                if 'reconstruction_error_diff' in vqd_logs:
                    print(f"    Reconstruction error diff: {vqd_logs['reconstruction_error_diff']:.6e}")
            
            # Add chordal distance if available
            if 'chordal_distance' in vqd_logs:
                vqd_result['chordal_distance'] = vqd_logs['chordal_distance']
            
            result['vqd'] = vqd_result
            
            print(f"    Accuracy: {acc_vqd_ci['mean']:.3f} [{acc_vqd_ci['ci_lower']:.3f}, {acc_vqd_ci['ci_upper']:.3f}]")
            print(f"    Fit time: {t_vqd_fit:.2f} ms ({t_vqd_fit/t_pca_fit:.1f}× PCA)")
            print(f"    Orthogonality: {orth_error_vqd:.2e}")
            print(f"    Rayleigh errors: mean={rayleigh_errors_vqd.mean():.2e}, max={rayleigh_errors_vqd.max():.2e}")
            print(f"    Angles (before Proc): mean={angles_before.mean():.1f}°, max={angles_before.max():.1f}°")
            
            # ============================================================
            # Orthogonal Procrustes Alignment
            # ============================================================
            print(f"\n🔄 Orthogonal Procrustes alignment...")
            
            # Find optimal rotation R: U_vqd @ R ≈ U_pca
            R, scale = orthogonal_procrustes(U_vqd, U_pca)
            U_vqd_aligned = U_vqd @ R
            
            # Residual before and after
            residual_before = np.linalg.norm(U_vqd - U_pca, 'fro')
            residual_after = np.linalg.norm(U_vqd_aligned - U_pca, 'fro')
            residual_drop_pct = (residual_before - residual_after) / residual_before * 100
            
            # Principal angles AFTER Procrustes
            angles_after = compute_principal_angles(U_pca, U_vqd_aligned)
            
            # Project with aligned basis
            X_train_vqd_proc = X_train_centered @ U_vqd_aligned
            X_test_vqd_proc = X_test_centered @ U_vqd_aligned
            
            # DTW Classification with Procrustes
            @exp.time_component(f"dtw_vqd_proc_fb{fb_size}_k{k}")
            def classify_vqd_proc():
                clf = DTWClassifierParallel(n_jobs=-1)
                clf.fit(X_train_vqd_proc, y_train)
                return clf.predict(X_test_vqd_proc)
            
            pred_vqd_proc = classify_vqd_proc()
            acc_vqd_proc = float(np.mean(pred_vqd_proc == y_test))
            
            # Bootstrap CI
            def bootstrap_vqd_proc():
                idx = np.random.choice(len(X_test_vqd_proc), size=len(X_test_vqd_proc), replace=True)
                clf = DTWClassifierParallel(n_jobs=-1)
                clf.fit(X_train_vqd_proc, y_train)
                pred = clf.predict(X_test_vqd_proc[idx])
                return float(np.mean(pred == y_test[idx]))
            
            acc_vqd_proc_ci = exp.bootstrap_ci(bootstrap_vqd_proc, n_bootstrap=1000)
            
            result['vqd_procrustes'] = {
                'accuracy': acc_vqd_proc_ci,
                'predictions': pred_vqd_proc.tolist(),
                'angles_after_procrustes': {
                    'mean': float(angles_after.mean()),
                    'max': float(angles_after.max()),
                    'values': angles_after.tolist()
                },
                'residual_before': float(residual_before),
                'residual_after': float(residual_after),
                'residual_drop_pct': float(residual_drop_pct),
                'rotation_matrix': R.tolist()
            }
            
            print(f"    Residual drop: {residual_drop_pct:.1f}% ({residual_before:.3f} → {residual_after:.3f})")
            print(f"    Angles (after Proc): mean={angles_after.mean():.1f}°, max={angles_after.max():.1f}°")
            print(f"    Accuracy (Proc): {acc_vqd_proc_ci['mean']:.3f} [{acc_vqd_proc_ci['ci_lower']:.3f}, {acc_vqd_proc_ci['ci_upper']:.3f}]")
            
            # ============================================================
            # Statistical Tests
            # ============================================================
            print(f"\n📊 Statistical tests...")
            
            # McNemar test: VQD vs PCA
            mcnemar_vqd_pca = mcnemar_test(y_test, pred_vqd, pred_pca)
            print(f"    McNemar (VQD vs PCA): stat={mcnemar_vqd_pca['statistic']:.2f}, "
                  f"p={mcnemar_vqd_pca['p_value']:.4f}, sig={mcnemar_vqd_pca['significant']}")
            
            # McNemar test: VQD+Proc vs PCA
            mcnemar_proc_pca = mcnemar_test(y_test, pred_vqd_proc, pred_pca)
            print(f"    McNemar (VQD+Proc vs PCA): stat={mcnemar_proc_pca['statistic']:.2f}, "
                  f"p={mcnemar_proc_pca['p_value']:.4f}, sig={mcnemar_proc_pca['significant']}")
            
            # Accuracy deltas
            delta_vqd = (acc_vqd - acc_pca) * 100  # percentage points
            delta_proc = (acc_vqd_proc - acc_pca) * 100
            
            result['statistical_tests'] = {
                'mcnemar_vqd_vs_pca': mcnemar_vqd_pca,
                'mcnemar_vqd_proc_vs_pca': mcnemar_proc_pca,
                'delta_vqd_vs_pca_pp': float(delta_vqd),
                'delta_vqd_proc_vs_pca_pp': float(delta_proc)
            }
            
            print(f"    Δ VQD vs PCA: {delta_vqd:+.2f} pp")
            print(f"    Δ VQD+Proc vs PCA: {delta_proc:+.2f} pp")
            
            # ============================================================
            # Runtime Summary
            # ============================================================
            timing = exp.get_timing_summary()
            
            proj_time_pca = timing.get(f'pca_project_fb{fb_size}_k{k}', 0.0) / n_test
            dtw_time_pca = timing.get(f'dtw_pca_fb{fb_size}_k{k}', 0.0) / n_test
            
            proj_time_vqd = timing.get(f'vqd_project_fb{fb_size}_k{k}', 0.0) / n_test
            dtw_time_vqd = timing.get(f'dtw_vqd_fb{fb_size}_k{k}', 0.0) / n_test
            
            result['runtime_per_query_ms'] = {
                'pca': {
                    'projection': proj_time_pca,
                    'dtw': dtw_time_pca,
                    'total': proj_time_pca + dtw_time_pca
                },
                'vqd': {
                    'projection': proj_time_vqd,
                    'dtw': dtw_time_vqd,
                    'total': proj_time_vqd + dtw_time_vqd
                }
            }
            
            # Speedup vs 60D baseline (from Experiment 1)
            # Assume 60D baseline is ~0.65 ms/query
            baseline_time = 0.65
            speedup_pca = baseline_time / result['runtime_per_query_ms']['pca']['total']
            speedup_vqd = baseline_time / result['runtime_per_query_ms']['vqd']['total']
            
            result['speedup_vs_60D'] = {
                'pca': float(speedup_pca),
                'vqd': float(speedup_vqd)
            }
            
            print(f"\n⏱️  Runtime per query:")
            print(f"    PCA: proj={proj_time_pca:.3f}ms + dtw={dtw_time_pca:.3f}ms = {proj_time_pca+dtw_time_pca:.3f}ms (speedup: {speedup_pca:.2f}×)")
            print(f"    VQD: proj={proj_time_vqd:.3f}ms + dtw={dtw_time_vqd:.3f}ms = {proj_time_vqd+dtw_time_vqd:.3f}ms (speedup: {speedup_vqd:.2f}×)")
            
            # Store result
            all_results[result_key] = result
    
    # ============================================================
    # Save Results
    # ============================================================
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print('=' * 80)
    
    exp.manifest['quantum_resources'] = quantum_tracker.get_summary()
    exp.manifest['results'] = all_results
    
    exp.write_manifest()
    exp.save_results(all_results)
    
    # Generate summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE")
    print('=' * 80)
    
    print(f"\n{'k':<4} {'FB':<4} {'PCA Acc':<20} {'VQD Acc':<20} {'VQD+Proc Acc':<20} "
          f"{'Angles (mean/max)':<20} {'Proc Drop %':<12} {'PCA Time':<10} {'VQD Time':<10}")
    print('-' * 140)
    
    for key in sorted(all_results.keys()):
        res = all_results[key]
        k = res['k']
        fb = res['frame_bank_size']
        
        if 'vqd' not in res or 'error' in res['vqd']:
            continue
        
        acc_pca = res['pca']['accuracy']
        acc_vqd = res['vqd']['accuracy']
        acc_proc = res['vqd_procrustes']['accuracy']
        
        angles_mean = res['vqd']['angles_before_procrustes']['mean']
        angles_max = res['vqd']['angles_before_procrustes']['max']
        
        proc_drop = res['vqd_procrustes']['residual_drop_pct']
        
        time_pca = res['runtime_per_query_ms']['pca']['total']
        time_vqd = res['runtime_per_query_ms']['vqd']['total']
        
        print(f"{k:<4} {fb:<4} "
              f"{acc_pca['mean']:.3f} [{acc_pca['ci_lower']:.2f},{acc_pca['ci_upper']:.2f}] "
              f"{acc_vqd['mean']:.3f} [{acc_vqd['ci_lower']:.2f},{acc_vqd['ci_upper']:.2f}] "
              f"{acc_proc['mean']:.3f} [{acc_proc['ci_lower']:.2f},{acc_proc['ci_upper']:.2f}] "
              f"{angles_mean:.1f}°/{angles_max:.1f}°       "
              f"{proc_drop:<12.1f} "
              f"{time_pca:<10.2f} "
              f"{time_vqd:<10.2f}")
    
    print(f"\n✅ Experiment 3 complete! Results saved to {exp.exp_dir}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment 3: VQD Quantum PCA")
    parser.add_argument("--n-train", type=int, default=100, help="Number of training samples")
    parser.add_argument("--n-test", type=int, default=100, help="Number of test samples (≥100 recommended for stable CIs)")
    parser.add_argument("--k-values", type=int, nargs="+", default=[6, 8, 10, 12],
                        help="K values to test")
    parser.add_argument("--frame-bank-sizes", type=int, nargs="+", default=[8],
                        help="Frame bank sizes to test")
    parser.add_argument("--test-whitening", action="store_true",
                        help="Test whitening transform")
    parser.add_argument("--vqd-depth", type=int, default=2,
                        help="VQD circuit depth")
    parser.add_argument("--vqd-qubits", type=int, default=3,
                        help="VQD number of qubits")
    parser.add_argument("--vqd-maxiter", type=int, default=300,
                        help="VQD max iterations per eigenvector (300+ recommended)")
    parser.add_argument("--use-enhanced", action="store_true", default=True,
                        help="Use enhanced VQD implementation with validation fixes")
    parser.add_argument("--no-enhanced", dest="use_enhanced", action="store_false",
                        help="Use original VQD implementation")
    parser.add_argument("--warm-starts", action="store_true", default=True,
                        help="Use warm starts for VQD (initialize from previous eigenvector)")
    parser.add_argument("--no-warm-starts", dest="warm_starts", action="store_false",
                        help="Disable warm starts")
    parser.add_argument("--off-diagonal-penalty", type=float, default=0.0,
                        help="DEPRECATED: Use --commutator-penalty instead")
    parser.add_argument("--out-of-span-penalty", type=float, default=0.0,
                        help="Out-of-span penalty ||(I-UU^T)CU||_F^2 for k<d (0.05-0.1 recommended)")
    parser.add_argument("--out-of-span-warmup", type=int, default=100,
                        help="Out-of-span penalty warmup epochs (100-200)")
    parser.add_argument("--out-of-span-decay", type=float, default=0.3,
                        help="Out-of-span decay factor for fine-tune phase (0.3)")
    parser.add_argument("--commutator-penalty", type=float, default=None,
                        help="DEPRECATED: Use --out-of-span-penalty for k<d instead")
    parser.add_argument("--commutator-warmup", type=int, default=100,
                        help="DEPRECATED: Commutator penalty warmup epochs (100-200)")
    parser.add_argument("--commutator-decay", type=float, default=0.3,
                        help="DEPRECATED: Commutator decay factor for fine-tune phase (0.3)")
    parser.add_argument("--max-angle-penalty", type=float, default=0.0,
                        help="DEPRECATED: Max angle penalty (not recommended)")
    parser.add_argument("--max-angle-threshold", type=float, default=0.9,
                        help="DEPRECATED: Max angle threshold τ for min singular value (0.85-0.95)")
    parser.add_argument("--max-angle-epochs", type=int, default=50,
                        help="DEPRECATED: Max angle penalty epochs (50)")
    parser.add_argument("--use-shared-params", action="store_true", default=False,
                        help="Use SSVQE-style shared parameters for k<d (TODO: not yet implemented)")
    parser.add_argument("--chordal-loss", type=float, default=0.05,
                        help="Chordal distance loss for warm-up (0.01-0.1, turns off after chordal-loss-epochs)")
    parser.add_argument("--chordal-loss-epochs", type=int, default=50,
                        help="Epochs to use chordal loss before turning off (50 recommended)")
    parser.add_argument("--gram-schmidt-freq", type=int, default=10,
                        help="In-loop Gram-Schmidt frequency (0=disabled, 10 recommended)")
    
    # Entanglement control for ablation study
    parser.add_argument("--force-alternating-entanglement", action="store_true", default=False,
                        help="Force alternating entanglement even for k=d (disable auto-switch)")
    parser.add_argument("--allow-full-entanglement", action="store_true", default=True,
                        help="Allow full entanglement for k=d at depth=2 (default: True)")
    
    args = parser.parse_args()
    
    # Auto-select commutator penalty based on k=d or k<d
    if args.commutator_penalty is None:
        # Will be set per k value in experiment
        commutator_penalty_kd = 1.0  # k=d case
        commutator_penalty_kless = 0.1  # k<d case
    else:
        commutator_penalty_kd = args.commutator_penalty
        commutator_penalty_kless = args.commutator_penalty
    
    # Determine entanglement mode for ablation control
    if args.force_alternating_entanglement:
        entanglement_mode = 'alternating'  # Force alternating, disable auto-switch
    elif args.allow_full_entanglement:
        entanglement_mode = 'alternating'  # Will auto-switch to 'full' for k=d in vqd_pca_enhanced.py
    else:
        entanglement_mode = 'alternating'  # Default
    
    vqd_params = {
        'num_qubits': args.vqd_qubits,
        'max_depth': args.vqd_depth,
        'ramped_penalties': True,
        'warm_starts': args.warm_starts,
        'entanglement': entanglement_mode,
        'in_loop_gram_schmidt_freq': args.gram_schmidt_freq,
        'off_diagonal_penalty': args.off_diagonal_penalty,
        'off_diagonal_warmup_epochs': 100,
        'out_of_span_penalty': args.out_of_span_penalty,
        'out_of_span_warmup_epochs': args.out_of_span_warmup,
        'out_of_span_decay_factor': args.out_of_span_decay,
        'commutator_penalty': args.commutator_penalty if args.commutator_penalty else 0.0,  # DEPRECATED
        'commutator_warmup_epochs': args.commutator_warmup,  # DEPRECATED
        'commutator_decay_factor': args.commutator_decay,  # DEPRECATED
        'max_angle_penalty': args.max_angle_penalty,  # DEPRECATED
        'max_angle_threshold': args.max_angle_threshold,  # DEPRECATED
        'max_angle_epochs': args.max_angle_epochs,  # DEPRECATED
        'use_shared_parameters': args.use_shared_params,
        'procrustes_alpha': args.chordal_loss,
        'procrustes_epochs': args.chordal_loss_epochs,
        'maxiter': args.vqd_maxiter,
        '_force_alternating': args.force_alternating_entanglement  # Pass to VQD for k=d override
    }
    
    results = run_experiment_3(
        n_train=args.n_train,
        n_test=args.n_test,
        k_values=args.k_values,
        frame_bank_sizes=args.frame_bank_sizes,
        test_whitening=args.test_whitening,
        use_enhanced_vqd=args.use_enhanced,
        vqd_params=vqd_params
    )
