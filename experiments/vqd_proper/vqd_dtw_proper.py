"""
VQD-DTW Pipeline - PROPER IMPLEMENTATION

Uses FULL temporal sequences (not single frames) with proper DTW classification.
Compares VQD quantum PCA vs classical PCA for dimensionality reduction.

Key features:
1. Full sequences (13-255 frames each)
2. 300 train / 60 test (~15 samples per class)
3. Frame bank for learning subspace (all train frames)
4. Project full sequences to subspace
5. DTW 1-NN classification on projected sequences
6. Compare VQD vs classical PCA

Target: Match 75% DTW baseline accuracy
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


class ProperVQDDTWPipeline:
    """VQD-DTW Pipeline using full temporal sequences."""
    
    def __init__(self, k_values=[4, 6, 8], n_train=300, n_test=60, 
                 pre_k=16, random_state=42):
        """
        Initialize pipeline.
        
        Parameters
        ----------
        k_values : list
            Target dimensions for VQD/PCA projection
        n_train : int
            Number of training sequences (need ~15 per class)
        n_test : int
            Number of test sequences
        pre_k : int
            Intermediate dimension for frame bank reduction
        random_state : int
            Random seed
        """
        self.k_values = k_values
        self.n_train = n_train
        self.n_test = n_test
        self.pre_k = pre_k
        self.random_state = random_state
        self.results = {}
        
    def load_data(self):
        """Load full temporal sequences from MSR Action3D."""
        print(f"\n{'='*80}")
        print(f"LOADING MSR ACTION3D SEQUENCES")
        print(f"{'='*80}")
        
        data_path = Path(__file__).parent.parent / "msr_action_data"
        sequences, labels = load_all_sequences(str(data_path))
        print(f"✅ Loaded {len(sequences)} sequences")
        print(f"✅ Classes: {len(set(labels))} (range: {min(labels)}-{max(labels)})")
        
        # Show sequence statistics
        lens = [len(seq) for seq in sequences]
        print(f"✅ Sequence lengths: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.1f} frames")
        print(f"✅ Feature dimension: {sequences[0].shape[1]}D")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels,
            train_size=self.n_train,
            test_size=self.n_test,
            random_state=self.random_state,
            stratify=labels  # Ensure balanced classes
        )
        
        self.X_train_raw = X_train
        self.X_test_raw = X_test
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        
        print(f"\n✅ Train: {len(y_train)} sequences")
        print(f"✅ Test: {len(y_test)} sequences")
        
        # Check class distribution
        from collections import Counter
        train_counts = Counter(y_train)
        print(f"✅ Samples per class: min={min(train_counts.values())}, max={max(train_counts.values())}")
        
        return self
    
    def build_frame_bank(self):
        """Build frame bank from all training frames for learning subspace."""
        print(f"\n{'─'*80}")
        print(f"BUILDING FRAME BANK")
        print(f"{'─'*80}")
        
        # Collect all frames from all training sequences
        all_frames = []
        for seq in self.X_train_raw:
            all_frames.append(seq)  # seq is (T_i, 60)
        
        self.frame_bank = np.vstack(all_frames)  # Shape: (total_frames, 60)
        print(f"Frame bank: {self.frame_bank.shape} (all frames from {len(self.X_train_raw)} train sequences)")
        
        # Normalize frame bank (train-only statistics)
        self.scaler = StandardScaler()
        self.frame_bank_scaled = self.scaler.fit_transform(self.frame_bank)
        print(f"✅ Normalized with train-only statistics")
        
        # Pre-reduce with classical PCA (60D → pre_k)
        self.pca_pre = PCA(n_components=self.pre_k)
        self.frame_bank_reduced = self.pca_pre.fit_transform(self.frame_bank_scaled)
        variance = self.pca_pre.explained_variance_ratio_.sum()
        print(f"✅ Pre-reduced: 60D → {self.pre_k}D (variance: {variance*100:.1f}%)")
        
        return self
    
    def evaluate_baseline(self):
        """Evaluate baseline: DTW on raw 60D sequences."""
        print(f"\n{'='*80}")
        print(f"BASELINE: DTW on Raw 60D Sequences")
        print(f"{'='*80}")
        
        # Normalize test sequences using train statistics
        X_train_norm = [self.scaler.transform(seq) for seq in self.X_train_raw]
        X_test_norm = [self.scaler.transform(seq) for seq in self.X_test_raw]
        
        # DTW 1-NN classification
        acc, times = self._dtw_classify(X_train_norm, X_test_norm)
        
        self.results['baseline'] = {
            'method': 'raw',
            'k': 60,
            'accuracy': acc,
            'time_per_query': np.mean(times),
            'speedup': 1.0
        }
        
        self.baseline_time = np.mean(times)
        
        print(f"\nAccuracy: {acc*100:.1f}%")
        print(f"Time/query: {np.mean(times):.2f}s")
        
        return self
    
    def evaluate_pca(self, k):
        """Evaluate classical PCA projection + DTW."""
        print(f"\n{'─'*40}")
        print(f"Classical PCA {k}D")
        print(f"{'─'*40}")
        
        # Learn PCA on frame bank
        pca = PCA(n_components=k)
        pca.fit(self.frame_bank_reduced)
        
        # Project all sequences
        X_train_proj = self._project_sequences(self.X_train_raw, pca)
        X_test_proj = self._project_sequences(self.X_test_raw, pca)
        
        # DTW 1-NN classification
        acc, times = self._dtw_classify(X_train_proj, X_test_proj)
        
        speedup = self.baseline_time / np.mean(times)
        
        print(f"  Accuracy: {acc*100:.1f}%")
        print(f"  Time/query: {np.mean(times):.2f}s")
        print(f"  Speedup: {speedup:.2f}×")
        
        return {
            'method': 'pca',
            'k': k,
            'accuracy': acc,
            'time_per_query': np.mean(times),
            'speedup': speedup
        }
    
    def evaluate_vqd(self, k):
        """Evaluate VQD quantum PCA projection + DTW."""
        print(f"\n{'─'*40}")
        print(f"VQD Quantum PCA {k}D")
        print(f"{'─'*40}")
        
        # Learn VQD on frame bank
        num_qubits = int(np.ceil(np.log2(self.pre_k)))
        print(f"  Computing VQD ({num_qubits} qubits, k={k})...")
        
        U_vqd, eigenvalues, logs = vqd_quantum_pca(
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
        
        # Use Procrustes-aligned basis if available
        if 'U_vqd_aligned' in logs:
            U_proj = logs['U_vqd_aligned']
            basis_name = 'procrustes'
        else:
            U_proj = U_vqd
            basis_name = 'raw'
        
        # Project all sequences
        X_train_proj = self._project_sequences_vqd(self.X_train_raw, U_proj)
        X_test_proj = self._project_sequences_vqd(self.X_test_raw, U_proj)
        
        # DTW 1-NN classification
        acc, times = self._dtw_classify(X_train_proj, X_test_proj)
        
        speedup = self.baseline_time / np.mean(times)
        
        # VQD quality metrics
        mean_angle = np.mean(logs['principal_angles_deg'])
        max_angle = np.max(logs['principal_angles_deg'])
        ortho_error = logs['orthogonality_error']
        
        print(f"  VQD Quality:")
        print(f"    Orthogonality: {ortho_error:.2e}")
        print(f"    Angles: mean={mean_angle:.1f}°, max={max_angle:.1f}°")
        print(f"  Classification ({basis_name} basis):")
        print(f"    Accuracy: {acc*100:.1f}%")
        print(f"    Time/query: {np.mean(times):.2f}s")
        print(f"    Speedup: {speedup:.2f}×")
        
        return {
            'method': 'vqd',
            'k': k,
            'basis': basis_name,
            'accuracy': acc,
            'time_per_query': np.mean(times),
            'speedup': speedup,
            'orthogonality_error': ortho_error,
            'mean_angle': mean_angle,
            'max_angle': max_angle
        }
    
    def run_k_sweep(self):
        """Evaluate both PCA and VQD for all k values."""
        print(f"\n{'='*80}")
        print(f"K-SWEEP: PCA vs VQD")
        print(f"{'='*80}")
        
        for k in self.k_values:
            print(f"\n{'='*80}")
            print(f"k = {k}")
            print(f"{'='*80}")
            
            # Classical PCA
            pca_results = self.evaluate_pca(k)
            self.results[f'pca_k{k}'] = pca_results
            
            # VQD Quantum PCA
            vqd_results = self.evaluate_vqd(k)
            self.results[f'vqd_k{k}'] = vqd_results
        
        return self
    
    def _project_sequences(self, sequences, pca):
        """Project sequences using classical PCA."""
        projected = []
        for seq in sequences:
            # Normalize and pre-reduce
            seq_norm = self.scaler.transform(seq)
            seq_reduced = self.pca_pre.transform(seq_norm)
            # Project with PCA
            seq_proj = pca.transform(seq_reduced)
            projected.append(seq_proj)
        return projected
    
    def _project_sequences_vqd(self, sequences, U_vqd):
        """Project sequences using VQD basis."""
        projected = []
        for seq in sequences:
            # Normalize and pre-reduce
            seq_norm = self.scaler.transform(seq)
            seq_reduced = self.pca_pre.transform(seq_norm)
            # Center and project with VQD
            mean = np.mean(seq_reduced, axis=0)
            seq_proj = (seq_reduced - mean) @ U_vqd.T
            projected.append(seq_proj)
        return projected
    
    def _dtw_classify(self, X_train, X_test):
        """DTW 1-NN classification on sequences."""
        y_pred = []
        times = []
        
        total = len(X_test)
        for i, test_seq in enumerate(X_test):
            if (i+1) % 10 == 0:
                print(f"    Progress: {i+1}/{total}", end='\r')
            
            start = time.time()
            pred, _ = one_nn(X_train, self.y_train, test_seq, metric='euclidean')
            times.append(time.time() - start)
            y_pred.append(pred)
        
        if total >= 10:
            print()  # Clear progress line
        
        y_pred = np.array(y_pred)
        accuracy = np.mean(y_pred == self.y_test)
        
        return accuracy, times
    
    def save_results(self, output_path='results/vqd_dtw_proper_results.json'):
        """Save all results to JSON."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'k_values': self.k_values,
                'n_train': self.n_train,
                'n_test': self.n_test,
                'pre_k': self.pre_k,
                'random_state': self.random_state
            },
            'results': self.results
        }
        
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")
        return self
    
    def print_summary(self):
        """Print summary table."""
        print(f"\n{'='*80}")
        print(f"SUMMARY TABLE")
        print(f"{'='*80}\n")
        
        print(f"{'Method':<20} {'k':<5} {'Accuracy':<12} {'Speedup':<10} {'Time (s)':<12} {'Notes':<20}")
        print(f"{'-'*20} {'-'*5} {'-'*12} {'-'*10} {'-'*12} {'-'*20}")
        
        # Baseline
        r = self.results['baseline']
        print(f"{'Baseline (60D)':<20} {r['k']:<5} {r['accuracy']*100:>6.1f}%     {r['speedup']:>6.2f}×    {r['time_per_query']:>8.2f}      {'Full sequences':<20}")
        
        # PCA and VQD
        for k in self.k_values:
            pca_key = f'pca_k{k}'
            vqd_key = f'vqd_k{k}'
            
            if pca_key in self.results:
                r = self.results[pca_key]
                print(f"{f'PCA {k}D':<20} {r['k']:<5} {r['accuracy']*100:>6.1f}%     {r['speedup']:>6.2f}×    {r['time_per_query']:>8.2f}      {'':<20}")
            
            if vqd_key in self.results:
                r = self.results[vqd_key]
                angle_note = f"angle={r['max_angle']:.0f}°"
                print(f"{f'VQD {k}D':<20} {r['k']:<5} {r['accuracy']*100:>6.1f}%     {r['speedup']:>6.2f}×    {r['time_per_query']:>8.2f}      {angle_note:<20}")
        
        print(f"\n{'='*80}\n")
        
        return self


def main():
    """Run proper VQD-DTW pipeline."""
    
    print("\n" + "="*80)
    print("VQD-DTW PIPELINE - PROPER IMPLEMENTATION")
    print("="*80)
    print("Using FULL temporal sequences (not single frames)")
    print("Target: Match 75% DTW baseline")
    print("="*80 + "\n")
    
    # Initialize pipeline
    pipeline = ProperVQDDTWPipeline(
        k_values=[4, 6, 8, 10, 12],
        n_train=300,
        n_test=60,
        pre_k=16,
        random_state=42
    )
    
    # Run pipeline
    start_time = time.time()
    
    (pipeline
     .load_data()
     .build_frame_bank()
     .evaluate_baseline()
     .run_k_sweep()
     .save_results('results/vqd_dtw_proper_results.json')
     .print_summary())
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"\n✅ Proper VQD-DTW pipeline complete!")
    print(f"✅ Results saved to: results/vqd_dtw_proper_results.json")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
