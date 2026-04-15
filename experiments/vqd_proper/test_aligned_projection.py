"""
VQD-DTW with ALIGNED Projection Methods

Tests both centering approaches to ensure fair comparison:
1. Global centering (both PCA and VQD use training set mean)
2. Per-sequence centering (both PCA and VQD center each sequence)

This addresses the projection inconsistency found in verification.
"""

import numpy as np
from pathlib import Path
import sys
import time
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'archive' / 'src'))

from archive.src.loader import load_all_sequences
from quantum.vqd_pca import vqd_quantum_pca
from dtw.dtw_runner import one_nn


class AlignedProjectionPipeline:
    """Pipeline with consistent projection methods."""
    
    def __init__(self, k_values=[4, 8, 12], centering='global'):
        """
        Parameters
        ----------
        k_values : list
            Target dimensions
        centering : str
            'global' - use training set mean (standard PCA approach)
            'per_seq' - center each sequence independently
        """
        self.k_values = k_values
        self.centering = centering
        self.n_train = 300
        self.n_test = 60
        self.pre_k = 16
        self.random_state = 42
        self.results = {}
        
        print(f"\n{'='*80}")
        print(f"ALIGNED PROJECTION: {centering.upper()} CENTERING")
        print(f"{'='*80}")
        print(f"Using {centering} centering for both PCA and VQD")
        print(f"{'='*80}\n")
    
    def load_data(self):
        """Load data."""
        data_path = Path(__file__).parent.parent / "msr_action_data"
        sequences, labels = load_all_sequences(str(data_path))
        
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels,
            train_size=self.n_train,
            test_size=self.n_test,
            random_state=self.random_state,
            stratify=labels
        )
        
        self.X_train_raw = X_train
        self.X_test_raw = X_test
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        
        print(f"✅ Data loaded: {len(y_train)} train, {len(y_test)} test")
        return self
    
    def build_frame_bank(self):
        """Build frame bank."""
        self.frame_bank = np.vstack([seq for seq in self.X_train_raw])
        
        self.scaler = StandardScaler()
        self.frame_bank_scaled = self.scaler.fit_transform(self.frame_bank)
        
        self.pca_pre = PCA(n_components=self.pre_k)
        self.frame_bank_reduced = self.pca_pre.fit_transform(self.frame_bank_scaled)
        
        # Store global mean for global centering approach
        self.global_mean = np.mean(self.frame_bank_reduced, axis=0)
        
        print(f"✅ Frame bank: {self.frame_bank_reduced.shape}")
        print(f"✅ Centering mode: {self.centering}")
        
        return self
    
    def run_comparison(self):
        """Run comparison for all k values."""
        print(f"\n{'='*80}")
        print(f"RUNNING COMPARISON")
        print(f"{'='*80}\n")
        
        # Baseline
        print(f"Baseline (60D)...", end=" ")
        X_train_norm = [self.scaler.transform(seq) for seq in self.X_train_raw]
        X_test_norm = [self.scaler.transform(seq) for seq in self.X_test_raw]
        baseline_acc, _ = self._dtw_classify(X_train_norm, X_test_norm)
        print(f"{baseline_acc*100:.1f}%")
        
        self.results['baseline'] = baseline_acc
        
        # For each k
        for k in self.k_values:
            print(f"\nk = {k}")
            print(f"{'─'*40}")
            
            # PCA
            print(f"  PCA...", end=" ")
            pca = PCA(n_components=k)
            pca.fit(self.frame_bank_reduced)
            
            X_train_pca = self._project_pca(self.X_train_raw, pca)
            X_test_pca = self._project_pca(self.X_test_raw, pca)
            
            pca_acc, _ = self._dtw_classify(X_train_pca, X_test_pca)
            print(f"{pca_acc*100:.1f}%")
            
            # VQD
            print(f"  VQD...", end=" ")
            num_qubits = int(np.ceil(np.log2(self.pre_k)))
            
            U_vqd, _, logs = vqd_quantum_pca(
                self.frame_bank_reduced,
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
            
            U_proj = logs.get('U_vqd_aligned', U_vqd)
            
            X_train_vqd = self._project_vqd(self.X_train_raw, U_proj)
            X_test_vqd = self._project_vqd(self.X_test_raw, U_proj)
            
            vqd_acc, _ = self._dtw_classify(X_train_vqd, X_test_vqd)
            print(f"{vqd_acc*100:.1f}% (angle={np.max(logs['principal_angles_deg']):.0f}°)")
            
            self.results[f'k{k}'] = {
                'pca': pca_acc,
                'vqd': vqd_acc,
                'gap': vqd_acc - pca_acc,
                'angle': np.max(logs['principal_angles_deg'])
            }
        
        return self
    
    def _project_pca(self, sequences, pca):
        """Project sequences with PCA using chosen centering."""
        projected = []
        
        for seq in sequences:
            seq_norm = self.scaler.transform(seq)
            seq_reduced = self.pca_pre.transform(seq_norm)
            
            if self.centering == 'global':
                # Standard PCA: use global mean from training
                seq_proj = pca.transform(seq_reduced)
            else:  # per_seq
                # Per-sequence centering
                seq_mean = np.mean(seq_reduced, axis=0)
                seq_proj = (seq_reduced - seq_mean) @ pca.components_.T
            
            projected.append(seq_proj)
        
        return projected
    
    def _project_vqd(self, sequences, U_vqd):
        """Project sequences with VQD using chosen centering."""
        projected = []
        
        for seq in sequences:
            seq_norm = self.scaler.transform(seq)
            seq_reduced = self.pca_pre.transform(seq_norm)
            
            if self.centering == 'global':
                # Global centering: use training set mean
                seq_proj = (seq_reduced - self.global_mean) @ U_vqd.T
            else:  # per_seq
                # Per-sequence centering
                seq_mean = np.mean(seq_reduced, axis=0)
                seq_proj = (seq_reduced - seq_mean) @ U_vqd.T
            
            projected.append(seq_proj)
        
        return projected
    
    def _dtw_classify(self, X_train, X_test):
        """DTW classification."""
        y_pred = []
        times = []
        
        for test_seq in X_test:
            start = time.time()
            pred, _ = one_nn(X_train, self.y_train, test_seq, metric='euclidean')
            times.append(time.time() - start)
            y_pred.append(pred)
        
        accuracy = np.mean(np.array(y_pred) == self.y_test)
        return accuracy, times
    
    def print_summary(self):
        """Print results."""
        print(f"\n{'='*80}")
        print(f"SUMMARY: {self.centering.upper()} CENTERING")
        print(f"{'='*80}\n")
        
        print(f"Baseline: {self.results['baseline']*100:.1f}%\n")
        
        print(f"{'k':<5} {'PCA':<10} {'VQD':<10} {'Gap':<10} {'Angle':<10}")
        print(f"{'-'*45}")
        
        for k in self.k_values:
            r = self.results[f'k{k}']
            print(f"{k:<5} {r['pca']*100:>5.1f}%     {r['vqd']*100:>5.1f}%     {r['gap']*100:>+5.1f}%     {r['angle']:>5.0f}°")
        
        print(f"\n{'='*80}\n")
        
        return self


def main():
    """Run both centering approaches."""
    
    print("\n" + "="*80)
    print("VQD-DTW: ALIGNED PROJECTION COMPARISON")
    print("="*80)
    print("Testing both global and per-sequence centering")
    print("="*80 + "\n")
    
    results = {}
    
    # Test 1: Global centering (standard approach)
    print("\n" + "="*80)
    print("TEST 1: GLOBAL CENTERING")
    print("="*80)
    
    pipeline_global = AlignedProjectionPipeline(
        k_values=[4, 8, 12],
        centering='global'
    )
    
    (pipeline_global
     .load_data()
     .build_frame_bank()
     .run_comparison()
     .print_summary())
    
    results['global'] = pipeline_global.results
    
    # Test 2: Per-sequence centering
    print("\n" + "="*80)
    print("TEST 2: PER-SEQUENCE CENTERING")
    print("="*80)
    
    pipeline_perseq = AlignedProjectionPipeline(
        k_values=[4, 8, 12],
        centering='per_seq'
    )
    
    (pipeline_perseq
     .load_data()
     .build_frame_bank()
     .run_comparison()
     .print_summary())
    
    results['per_seq'] = pipeline_perseq.results
    
    # Compare both approaches
    print("\n" + "="*80)
    print("COMPARISON: GLOBAL vs PER-SEQUENCE")
    print("="*80 + "\n")
    
    print(f"{'Method':<20} {'k':<5} {'PCA':<10} {'VQD':<10} {'Gap':<10}")
    print(f"{'-'*55}")
    
    for k in [4, 8, 12]:
        r_global = results['global'][f'k{k}']
        r_perseq = results['per_seq'][f'k{k}']
        
        print(f"Global centering     {k:<5} {r_global['pca']*100:>5.1f}%     {r_global['vqd']*100:>5.1f}%     {r_global['gap']*100:>+5.1f}%")
        print(f"Per-seq centering    {k:<5} {r_perseq['pca']*100:>5.1f}%     {r_perseq['vqd']*100:>5.1f}%     {r_perseq['gap']*100:>+5.1f}%")
        print()
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    with open('results/aligned_projection_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Results saved to results/aligned_projection_results.json")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    # Check if VQD still wins with global centering
    global_gaps = [results['global'][f'k{k}']['gap'] for k in [4, 8, 12]]
    avg_global_gap = np.mean(global_gaps)
    
    print(f"\nWith GLOBAL centering (fair comparison):")
    print(f"  Average VQD-PCA gap: {avg_global_gap*100:+.1f}%")
    
    if avg_global_gap > 0.05:
        print(f"  ✅ VQD still significantly outperforms PCA!")
    elif avg_global_gap > 0:
        print(f"  ⚠️  VQD slightly outperforms PCA")
    else:
        print(f"  ❌ VQD advantage disappears with fair comparison")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
