"""
Whitening Toggle Experiment
============================

Compares standard projection (U) vs whitened projection (U Λ^{-1/2}).

Whitening transforms to unit variance along each principal component,
which can stabilize DTW distances especially for k=8-12 where lower
eigenvalues become small.

Author: VQD-DTW Research Team
Date: November 24, 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time

# Import our modules
from archive.src.loader import load_all_sequences
from quantum.vqd_pca import vqd_quantum_pca
from dtw.dtw_runner import one_nn

class WhiteningExperiment:
    """Test whitening effect on PCA and VQD."""
    
    def __init__(self, k_values=[6, 8, 10, 12], seed=42, n_train=300, n_test=60, pre_k=16):
        self.k_values = k_values
        self.seed = seed
        self.n_train = n_train
        self.n_test = n_test
        self.pre_k = pre_k
        
        self.results = {
            'config': {
                'k_values': k_values,
                'seed': seed,
                'n_train': n_train,
                'n_test': n_test,
                'pre_k': pre_k,
                'date': datetime.now().isoformat()
            },
            'by_k': {}  # k -> method -> whitening -> metrics
        }
    
    def load_and_prepare_data(self):
        """Load data and split."""
        print(f"\n{'='*70}")
        print(f"Loading data with seed={self.seed}")
        print(f"{'='*70}")
        
        # Data is in parent directory
        data_path = Path(__file__).parent.parent / "msr_action_data"
        sequences, labels = load_all_sequences(str(data_path))
        print(f"Loaded {len(sequences)} sequences")
        
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels,
            train_size=self.n_train,
            test_size=self.n_test,
            random_state=self.seed,
            stratify=labels
        )
        
        print(f"Split: {len(X_train)} train, {len(X_test)} test")
        return X_train, X_test, y_train, y_test
    
    def build_frame_bank(self, X_train):
        """Build and pre-reduce frame bank."""
        print("\nBuilding frame bank...")
        frame_bank = np.vstack([seq for seq in X_train])
        
        self.scaler = StandardScaler()
        frame_bank_scaled = self.scaler.fit_transform(frame_bank)
        
        self.pca_pre = PCA(n_components=self.pre_k)
        frame_bank_reduced = self.pca_pre.fit_transform(frame_bank_scaled)
        print(f"Frame bank shape: {frame_bank_reduced.shape}")
        
        return frame_bank_reduced
    
    def project_sequence(self, seq, U_proj, eigenvalues=None, whiten=False):
        """
        Project sequence with optional whitening.
        
        Args:
            seq: input sequence (T, 60)
            U_proj: projection matrix (k, 16)
            eigenvalues: optional eigenvalues for whitening (k,)
            whiten: if True, apply U Λ^{-1/2} instead of U
        """
        # Standard preprocessing
        seq_norm = self.scaler.transform(seq)
        seq_reduced = self.pca_pre.transform(seq_norm)
        mean = np.mean(seq_reduced, axis=0)
        seq_centered = seq_reduced - mean
        
        # Project
        seq_proj = seq_centered @ U_proj.T  # (T, k)
        
        # Apply whitening if requested
        if whiten and eigenvalues is not None:
            # Λ^{-1/2}: scale by inverse square root of eigenvalues
            inv_sqrt_lambda = 1.0 / np.sqrt(eigenvalues + 1e-8)
            seq_proj = seq_proj * inv_sqrt_lambda  # Broadcasting
        
        return seq_proj
    
    def evaluate_method(self, X_train, X_test, y_train, y_test, 
                       U_proj, eigenvalues, method_name, k, whiten):
        """Evaluate with or without whitening."""
        whiten_str = "whitened" if whiten else "standard"
        print(f"\n--- {method_name} k={k} ({whiten_str}) ---")
        
        # Project all sequences
        X_train_proj = [self.project_sequence(seq, U_proj, eigenvalues, whiten) 
                       for seq in X_train]
        X_test_proj = [self.project_sequence(seq, U_proj, eigenvalues, whiten) 
                      for seq in X_test]
        
        # DTW 1-NN - loop through test sequences
        start = time.time()
        y_pred = []
        y_train_arr = np.array(y_train)
        
        for test_seq in X_test_proj:
            pred, _ = one_nn(X_train_proj, y_train_arr, test_seq)
            y_pred.append(pred)
        
        y_pred = np.array(y_pred)
        elapsed = time.time() - start
        
        accuracy = np.mean(y_pred == y_test)
        print(f"{method_name} ({whiten_str}) Accuracy: {accuracy*100:.1f}%")
        
        # Compute variance of projected data (useful diagnostic)
        all_frames_proj = np.vstack(X_train_proj)
        variances = np.var(all_frames_proj, axis=0)
        print(f"Projected variances: min={variances.min():.2e}, max={variances.max():.2e}, "
              f"mean={variances.mean():.2e}")
        
        return {
            'accuracy': accuracy,
            'time': elapsed,
            'projected_variances': variances.tolist()
        }
    
    def run_single_k(self, k, X_train, X_test, y_train, y_test, frame_bank_reduced):
        """Run PCA and VQD with/without whitening for single k."""
        print(f"\n{'='*70}")
        print(f"Testing k={k}")
        print(f"{'='*70}")
        
        k_results = {}
        
        # ========== PCA ==========
        print("\n### PCA ###")
        pca = PCA(n_components=k)
        pca.fit(frame_bank_reduced)
        
        U_pca = pca.components_
        eigenvalues_pca = pca.explained_variance_
        
        print(f"PCA eigenvalues: {eigenvalues_pca}")
        print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Standard projection
        pca_standard = self.evaluate_method(
            X_train, X_test, y_train, y_test,
            U_pca, eigenvalues_pca, "PCA", k, whiten=False
        )
        
        # Whitened projection
        pca_whitened = self.evaluate_method(
            X_train, X_test, y_train, y_test,
            U_pca, eigenvalues_pca, "PCA", k, whiten=True
        )
        
        k_results['pca'] = {
            'standard': pca_standard,
            'whitened': pca_whitened,
            'delta': pca_whitened['accuracy'] - pca_standard['accuracy'],
            'eigenvalues': eigenvalues_pca.tolist()
        }
        
        # ========== VQD ==========
        print("\n### VQD ###")
        U_vqd, eigenvalues_vqd, logs = vqd_quantum_pca(
            frame_bank_reduced,
            n_components=k,
            num_qubits=4,
            max_depth=2,
            penalty_scale='auto',
            ramped_penalties=True,
            entanglement='alternating',
            maxiter=200,
            validate=True
        )
        
        # Use aligned basis
        if 'U_vqd_aligned' in logs:
            U_proj = logs['U_vqd_aligned']
        else:
            U_proj = U_vqd
        
        print(f"VQD eigenvalues: {eigenvalues_vqd}")
        print(f"VQD orthogonality: {logs.get('mean_orthogonality_error', np.nan):.2e}")
        
        # Standard projection
        vqd_standard = self.evaluate_method(
            X_train, X_test, y_train, y_test,
            U_proj, eigenvalues_vqd, "VQD", k, whiten=False
        )
        
        # Whitened projection
        vqd_whitened = self.evaluate_method(
            X_train, X_test, y_train, y_test,
            U_proj, eigenvalues_vqd, "VQD", k, whiten=True
        )
        
        k_results['vqd'] = {
            'standard': vqd_standard,
            'whitened': vqd_whitened,
            'delta': vqd_whitened['accuracy'] - vqd_standard['accuracy'],
            'eigenvalues': eigenvalues_vqd.tolist(),
            'orthogonality_error': float(logs.get('mean_orthogonality_error', np.nan)),
            'max_angle': float(logs.get('max_principal_angle', np.nan))
        }
        
        # ========== Summary ==========
        print(f"\n{'='*70}")
        print(f"k={k} Summary:")
        print(f"{'='*70}")
        print(f"PCA standard:  {pca_standard['accuracy']*100:.1f}%")
        print(f"PCA whitened:  {pca_whitened['accuracy']*100:.1f}% "
              f"(Δ={k_results['pca']['delta']*100:+.1f}%)")
        print(f"VQD standard:  {vqd_standard['accuracy']*100:.1f}%")
        print(f"VQD whitened:  {vqd_whitened['accuracy']*100:.1f}% "
              f"(Δ={k_results['vqd']['delta']*100:+.1f}%)")
        print(f"\nVQD-PCA gap (standard): {(vqd_standard['accuracy']-pca_standard['accuracy'])*100:+.1f}%")
        print(f"VQD-PCA gap (whitened): {(vqd_whitened['accuracy']-pca_whitened['accuracy'])*100:+.1f}%")
        
        return k_results
    
    def run(self):
        """Run full whitening experiment."""
        print("="*70)
        print("WHITENING TOGGLE EXPERIMENT")
        print("="*70)
        print(f"Comparing U vs U Λ^{{-1/2}} for k={self.k_values}")
        print("="*70)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_and_prepare_data()
        frame_bank_reduced = self.build_frame_bank(X_train)
        
        # Run for each k
        for k in self.k_values:
            k_results = self.run_single_k(k, X_train, X_test, y_train, y_test, 
                                         frame_bank_reduced)
            self.results['by_k'][k] = k_results
            
            # Save incremental
            self.save_results()
        
        # Print final summary table
        self.print_summary_table()
        
        print("\n" + "="*70)
        print("WHITENING EXPERIMENT COMPLETE!")
        print(f"Results saved to: results/whitening_results.json")
        print("="*70)
    
    def print_summary_table(self):
        """Print comparison table."""
        print(f"\n{'='*70}")
        print("WHITENING COMPARISON TABLE")
        print(f"{'='*70}\n")
        
        print(f"{'k':<4} {'Method':<6} {'Standard':<10} {'Whitened':<10} {'Delta':<10}")
        print("-" * 50)
        
        for k in self.k_values:
            if k in self.results['by_k']:
                kr = self.results['by_k'][k]
                
                # PCA
                pca_std = kr['pca']['standard']['accuracy'] * 100
                pca_wht = kr['pca']['whitened']['accuracy'] * 100
                pca_dlt = kr['pca']['delta'] * 100
                print(f"{k:<4} {'PCA':<6} {pca_std:>8.1f}%  {pca_wht:>8.1f}%  {pca_dlt:>+8.1f}%")
                
                # VQD
                vqd_std = kr['vqd']['standard']['accuracy'] * 100
                vqd_wht = kr['vqd']['whitened']['accuracy'] * 100
                vqd_dlt = kr['vqd']['delta'] * 100
                print(f"{k:<4} {'VQD':<6} {vqd_std:>8.1f}%  {vqd_wht:>8.1f}%  {vqd_dlt:>+8.1f}%")
                print()
    
    def save_results(self):
        """Save results to JSON."""
        output_path = Path(__file__).parent / "results" / "whitening_results.json"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

if __name__ == "__main__":
    experiment = WhiteningExperiment(
        k_values=[6, 8, 10, 12],
        seed=42,
        n_train=300,
        n_test=60,
        pre_k=16
    )
    
    experiment.run()
