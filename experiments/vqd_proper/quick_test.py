"""
VQD-DTW Pipeline - QUICK TEST VERSION (k=4 only)

Quick validation run to verify pipeline works before full experiment.
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

from loader import load_all_sequences
from quantum.vqd_pca import vqd_quantum_pca
from dtw.dtw_runner import one_nn


class QuickTestPipeline:
    """Quick test version - k=4 only."""
    
    def __init__(self):
        self.k = 4  # Single k value for quick test
        self.n_train = 300
        self.n_test = 60
        self.pre_k = 16
        self.random_state = 42
        self.results = {}
        
    def load_data(self):
        """Load data."""
        print(f"\n{'='*60}")
        print(f"LOADING DATA")
        print(f"{'='*60}")
        
        data_path = Path(__file__).parent.parent / "msr_action_data"
        sequences, labels = load_all_sequences(str(data_path))
        print(f"✅ {len(sequences)} sequences")
        
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
        
        print(f"✅ Train: {len(y_train)}, Test: {len(y_test)}")
        return self
    
    def build_frame_bank(self):
        """Build frame bank."""
        print(f"\n{'─'*60}")
        print(f"FRAME BANK")
        print(f"{'─'*60}")
        
        self.frame_bank = np.vstack([seq for seq in self.X_train_raw])
        print(f"✅ Shape: {self.frame_bank.shape}")
        
        self.scaler = StandardScaler()
        self.frame_bank_scaled = self.scaler.fit_transform(self.frame_bank)
        
        self.pca_pre = PCA(n_components=self.pre_k)
        self.frame_bank_reduced = self.pca_pre.fit_transform(self.frame_bank_scaled)
        print(f"✅ Pre-reduced: 60D → {self.pre_k}D")
        
        return self
    
    def evaluate_baseline(self):
        """Baseline: DTW on raw 60D."""
        print(f"\n{'='*60}")
        print(f"BASELINE (60D)")
        print(f"{'='*60}")
        
        X_train_norm = [self.scaler.transform(seq) for seq in self.X_train_raw]
        X_test_norm = [self.scaler.transform(seq) for seq in self.X_test_raw]
        
        acc, times = self._dtw_classify(X_train_norm, X_test_norm)
        self.baseline_time = np.mean(times)
        
        print(f"✅ Accuracy: {acc*100:.1f}%")
        print(f"✅ Time/query: {np.mean(times):.2f}s")
        
        self.results['baseline'] = {
            'k': 60,
            'accuracy': acc,
            'time': np.mean(times)
        }
        
        return self
    
    def evaluate_pca(self):
        """Classical PCA k=4."""
        print(f"\n{'─'*60}")
        print(f"PCA k={self.k}")
        print(f"{'─'*60}")
        
        pca = PCA(n_components=self.k)
        pca.fit(self.frame_bank_reduced)
        
        X_train_proj = self._project_sequences(self.X_train_raw, pca)
        X_test_proj = self._project_sequences(self.X_test_raw, pca)
        
        acc, times = self._dtw_classify(X_train_proj, X_test_proj)
        speedup = self.baseline_time / np.mean(times)
        
        print(f"✅ Accuracy: {acc*100:.1f}%")
        print(f"✅ Speedup: {speedup:.2f}×")
        
        self.results['pca'] = {
            'k': self.k,
            'accuracy': acc,
            'speedup': speedup
        }
        
        return self
    
    def evaluate_vqd(self):
        """VQD Quantum PCA k=4."""
        print(f"\n{'─'*60}")
        print(f"VQD k={self.k}")
        print(f"{'─'*60}")
        
        num_qubits = int(np.ceil(np.log2(self.pre_k)))
        print(f"Computing VQD ({num_qubits} qubits)...")
        
        U_vqd, eigenvalues, logs = vqd_quantum_pca(
            self.frame_bank_reduced,
            n_components=self.k,
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
        
        X_train_proj = self._project_sequences_vqd(self.X_train_raw, U_proj)
        X_test_proj = self._project_sequences_vqd(self.X_test_raw, U_proj)
        
        acc, times = self._dtw_classify(X_train_proj, X_test_proj)
        speedup = self.baseline_time / np.mean(times)
        
        ortho = logs['orthogonality_error']
        angles = logs['principal_angles_deg']
        
        print(f"✅ Orthogonality: {ortho:.2e}")
        print(f"✅ Angles: mean={np.mean(angles):.1f}°, max={np.max(angles):.1f}°")
        print(f"✅ Accuracy: {acc*100:.1f}%")
        print(f"✅ Speedup: {speedup:.2f}×")
        
        self.results['vqd'] = {
            'k': self.k,
            'accuracy': acc,
            'speedup': speedup,
            'orthogonality': ortho,
            'mean_angle': np.mean(angles),
            'max_angle': np.max(angles)
        }
        
        return self
    
    def _project_sequences(self, sequences, pca):
        """Project sequences with PCA."""
        projected = []
        for seq in sequences:
            seq_norm = self.scaler.transform(seq)
            seq_reduced = self.pca_pre.transform(seq_norm)
            seq_proj = pca.transform(seq_reduced)
            projected.append(seq_proj)
        return projected
    
    def _project_sequences_vqd(self, sequences, U_vqd):
        """Project sequences with VQD."""
        projected = []
        for seq in sequences:
            seq_norm = self.scaler.transform(seq)
            seq_reduced = self.pca_pre.transform(seq_norm)
            mean = np.mean(seq_reduced, axis=0)
            seq_proj = (seq_reduced - mean) @ U_vqd.T
            projected.append(seq_proj)
        return projected
    
    def _dtw_classify(self, X_train, X_test):
        """DTW 1-NN classification."""
        y_pred = []
        times = []
        
        for i, test_seq in enumerate(X_test):
            if (i+1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(X_test)}", end='\r')
            
            start = time.time()
            pred, _ = one_nn(X_train, self.y_train, test_seq, metric='euclidean')
            times.append(time.time() - start)
            y_pred.append(pred)
        
        if len(X_test) >= 10:
            print()
        
        accuracy = np.mean(np.array(y_pred) == self.y_test)
        return accuracy, times
    
    def print_summary(self):
        """Print summary."""
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}\n")
        
        print(f"Method          k    Accuracy    Speedup    Notes")
        print(f"{'-'*60}")
        
        r = self.results['baseline']
        print(f"Baseline       {r['k']:2d}    {r['accuracy']*100:5.1f}%       1.0×")
        
        r = self.results['pca']
        print(f"PCA            {r['k']:2d}    {r['accuracy']*100:5.1f}%      {r['speedup']:4.1f}×")
        
        r = self.results['vqd']
        print(f"VQD            {r['k']:2d}    {r['accuracy']*100:5.1f}%      {r['speedup']:4.1f}×      angle={r['max_angle']:.0f}°")
        
        print(f"\n{'='*60}")
        
        # Check success criteria
        print("\n✅ SUCCESS CHECKS:")
        baseline_ok = self.results['baseline']['accuracy'] >= 0.70
        ortho_ok = self.results['vqd']['orthogonality'] < 1e-6
        angle_ok = self.results['vqd']['max_angle'] < 45
        gap = abs(self.results['pca']['accuracy'] - self.results['vqd']['accuracy'])
        gap_ok = gap <= 0.05
        
        print(f"  • Baseline ≥ 70%: {baseline_ok} ({self.results['baseline']['accuracy']*100:.1f}%)")
        print(f"  • Orthogonality < 1e-6: {ortho_ok} ({self.results['vqd']['orthogonality']:.2e})")
        print(f"  • Max angle < 45°: {angle_ok} ({self.results['vqd']['max_angle']:.1f}°)")
        print(f"  • VQD-PCA gap ≤ 5%: {gap_ok} ({gap*100:.1f}%)")
        
        if all([baseline_ok, ortho_ok, angle_ok, gap_ok]):
            print("\n✅ ALL CHECKS PASSED - Ready for full run!")
        else:
            print("\n⚠️  SOME CHECKS FAILED - Review before full run")
        
        print(f"{'='*60}\n")
        
        return self


def main():
    """Run quick test."""
    
    print("\n" + "="*60)
    print("VQD-DTW QUICK TEST (k=4 only)")
    print("="*60)
    print("Verify pipeline works before full run")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    pipeline = QuickTestPipeline()
    (pipeline
     .load_data()
     .build_frame_bank()
     .evaluate_baseline()
     .evaluate_pca()
     .evaluate_vqd()
     .print_summary())
    
    elapsed = time.time() - start_time
    
    print(f"Test completed in {elapsed/60:.1f} minutes\n")


if __name__ == "__main__":
    main()
