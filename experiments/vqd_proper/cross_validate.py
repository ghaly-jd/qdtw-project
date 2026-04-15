"""
VQD-DTW Cross-Validation

Run the experiment with multiple random seeds to validate findings.
Tests if VQD's superior performance is consistent or seed-dependent.
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


class CrossValidationPipeline:
    """Cross-validation for VQD-DTW experiment."""
    
    def __init__(self, k_values=[4, 8, 12], n_splits=5, n_train=300, n_test=60, pre_k=16):
        """
        Parameters
        ----------
        k_values : list
            Dimensions to test
        n_splits : int
            Number of different random seeds to test
        n_train, n_test : int
            Train/test sizes
        pre_k : int
            Pre-reduction dimension
        """
        self.k_values = k_values
        self.n_splits = n_splits
        self.n_train = n_train
        self.n_test = n_test
        self.pre_k = pre_k
        self.all_results = []
        
    def run_single_split(self, random_state, k):
        """Run one split for one k value."""
        print(f"\n  Seed {random_state}, k={k}...", end=" ")
        
        # Load and split data
        data_path = Path(__file__).parent.parent / "msr_action_data"
        sequences, labels = load_all_sequences(str(data_path))
        
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels,
            train_size=self.n_train,
            test_size=self.n_test,
            random_state=random_state,
            stratify=labels
        )
        
        # Build frame bank
        frame_bank = np.vstack([seq for seq in X_train])
        scaler = StandardScaler()
        frame_bank_scaled = scaler.fit_transform(frame_bank)
        
        pca_pre = PCA(n_components=self.pre_k)
        frame_bank_reduced = pca_pre.fit_transform(frame_bank_scaled)
        
        # Classical PCA
        pca = PCA(n_components=k)
        pca.fit(frame_bank_reduced)
        
        X_train_pca = self._project_sequences_pca(X_train, scaler, pca_pre, pca)
        X_test_pca = self._project_sequences_pca(X_test, scaler, pca_pre, pca)
        
        pca_acc = self._dtw_classify(X_train_pca, X_test_pca, y_train, y_test)
        
        # VQD Quantum PCA
        num_qubits = int(np.ceil(np.log2(self.pre_k)))
        
        U_vqd, eigenvalues, logs = vqd_quantum_pca(
            frame_bank_reduced,
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
        
        X_train_vqd = self._project_sequences_vqd(X_train, scaler, pca_pre, U_proj)
        X_test_vqd = self._project_sequences_vqd(X_test, scaler, pca_pre, U_proj)
        
        vqd_acc = self._dtw_classify(X_train_vqd, X_test_vqd, y_train, y_test)
        
        # Baseline (60D)
        X_train_norm = [scaler.transform(seq) for seq in X_train]
        X_test_norm = [scaler.transform(seq) for seq in X_test]
        baseline_acc = self._dtw_classify(X_train_norm, X_test_norm, y_train, y_test)
        
        print(f"Baseline={baseline_acc*100:.1f}%, PCA={pca_acc*100:.1f}%, VQD={vqd_acc*100:.1f}%")
        
        return {
            'seed': random_state,
            'k': k,
            'baseline_acc': baseline_acc,
            'pca_acc': pca_acc,
            'vqd_acc': vqd_acc,
            'vqd_pca_gap': vqd_acc - pca_acc,
            'vqd_baseline_gap': vqd_acc - baseline_acc,
            'orthogonality': logs['orthogonality_error'],
            'mean_angle': np.mean(logs['principal_angles_deg']),
            'max_angle': np.max(logs['principal_angles_deg'])
        }
    
    def _project_sequences_pca(self, sequences, scaler, pca_pre, pca):
        """Project sequences using PCA."""
        projected = []
        for seq in sequences:
            seq_norm = scaler.transform(seq)
            seq_reduced = pca_pre.transform(seq_norm)
            seq_proj = pca.transform(seq_reduced)
            projected.append(seq_proj)
        return projected
    
    def _project_sequences_vqd(self, sequences, scaler, pca_pre, U_vqd):
        """Project sequences using VQD."""
        projected = []
        for seq in sequences:
            seq_norm = scaler.transform(seq)
            seq_reduced = pca_pre.transform(seq_norm)
            mean = np.mean(seq_reduced, axis=0)
            seq_proj = (seq_reduced - mean) @ U_vqd.T
            projected.append(seq_proj)
        return projected
    
    def _dtw_classify(self, X_train, X_test, y_train, y_test):
        """DTW 1-NN classification."""
        y_pred = []
        for test_seq in X_test:
            pred, _ = one_nn(X_train, y_train, test_seq, metric='euclidean')
            y_pred.append(pred)
        
        accuracy = np.mean(np.array(y_pred) == y_test)
        return accuracy
    
    def run_cross_validation(self):
        """Run full cross-validation."""
        print(f"\n{'='*80}")
        print(f"CROSS-VALIDATION: {self.n_splits} splits × {len(self.k_values)} k values")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        for k in self.k_values:
            print(f"\n{'─'*80}")
            print(f"k = {k}")
            print(f"{'─'*80}")
            
            for split_idx in range(self.n_splits):
                seed = 42 + split_idx
                result = self.run_single_split(seed, k)
                self.all_results.append(result)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"Cross-validation complete in {elapsed/60:.1f} minutes")
        print(f"{'='*80}\n")
        
        return self
    
    def analyze_results(self):
        """Analyze cross-validation results."""
        print(f"\n{'='*80}")
        print(f"CROSS-VALIDATION ANALYSIS")
        print(f"{'='*80}\n")
        
        for k in self.k_values:
            k_results = [r for r in self.all_results if r['k'] == k]
            
            baseline_accs = [r['baseline_acc'] for r in k_results]
            pca_accs = [r['pca_acc'] for r in k_results]
            vqd_accs = [r['vqd_acc'] for r in k_results]
            vqd_pca_gaps = [r['vqd_pca_gap'] for r in k_results]
            vqd_baseline_gaps = [r['vqd_baseline_gap'] for r in k_results]
            angles = [r['mean_angle'] for r in k_results]
            
            print(f"k = {k}")
            print(f"{'─'*80}")
            print(f"  Baseline:  {np.mean(baseline_accs)*100:.1f}% ± {np.std(baseline_accs)*100:.1f}%")
            print(f"  PCA:       {np.mean(pca_accs)*100:.1f}% ± {np.std(pca_accs)*100:.1f}%")
            print(f"  VQD:       {np.mean(vqd_accs)*100:.1f}% ± {np.std(vqd_accs)*100:.1f}%")
            print(f"  VQD-PCA gap:      {np.mean(vqd_pca_gaps)*100:+.1f}% ± {np.std(vqd_pca_gaps)*100:.1f}%")
            print(f"  VQD-Baseline gap: {np.mean(vqd_baseline_gaps)*100:+.1f}% ± {np.std(vqd_baseline_gaps)*100:.1f}%")
            print(f"  Mean angles: {np.mean(angles):.1f}° ± {np.std(angles):.1f}°")
            print()
        
        # Overall summary
        print(f"{'='*80}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*80}\n")
        
        all_vqd_pca_gaps = [r['vqd_pca_gap'] for r in self.all_results]
        all_vqd_baseline_gaps = [r['vqd_baseline_gap'] for r in self.all_results]
        
        print(f"Across all {len(self.all_results)} runs:")
        print(f"  VQD vs PCA:      {np.mean(all_vqd_pca_gaps)*100:+.1f}% ± {np.std(all_vqd_pca_gaps)*100:.1f}%")
        print(f"  VQD vs Baseline: {np.mean(all_vqd_baseline_gaps)*100:+.1f}% ± {np.std(all_vqd_baseline_gaps)*100:.1f}%")
        
        # Statistical significance
        vqd_wins_pca = sum(1 for g in all_vqd_pca_gaps if g > 0)
        vqd_wins_baseline = sum(1 for g in all_vqd_baseline_gaps if g > 0)
        
        print(f"\n  VQD beats PCA:      {vqd_wins_pca}/{len(self.all_results)} times ({vqd_wins_pca/len(self.all_results)*100:.0f}%)")
        print(f"  VQD beats Baseline: {vqd_wins_baseline}/{len(self.all_results)} times ({vqd_wins_baseline/len(self.all_results)*100:.0f}%)")
        
        # Check consistency
        print(f"\n✅ VALIDATION CHECKS:")
        consistent_pca = vqd_wins_pca / len(self.all_results) >= 0.8
        consistent_baseline = vqd_wins_baseline / len(self.all_results) >= 0.8
        low_variance = np.std(all_vqd_pca_gaps) < 0.10
        
        print(f"  • VQD consistently beats PCA (≥80%): {consistent_pca}")
        print(f"  • VQD consistently beats baseline (≥80%): {consistent_baseline}")
        print(f"  • Low variance across seeds (std<10%): {low_variance}")
        
        if consistent_pca and consistent_baseline and low_variance:
            print(f"\n🎉 FINDINGS VALIDATED: Results are consistent across different random seeds!")
        else:
            print(f"\n⚠️  CAUTION: Results show high variance or inconsistency across seeds")
        
        print(f"\n{'='*80}\n")
        
        return self
    
    def save_results(self, output_path='results/cross_validation_results.json'):
        """Save cross-validation results."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'k_values': self.k_values,
                'n_splits': self.n_splits,
                'n_train': self.n_train,
                'n_test': self.n_test,
                'pre_k': self.pre_k
            },
            'results': self.all_results,
            'summary': self._compute_summary()
        }
        
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✅ Cross-validation results saved to {output_path}\n")
        return self
    
    def _compute_summary(self):
        """Compute summary statistics."""
        summary = {}
        
        for k in self.k_values:
            k_results = [r for r in self.all_results if r['k'] == k]
            
            summary[f'k{k}'] = {
                'baseline_mean': float(np.mean([r['baseline_acc'] for r in k_results])),
                'baseline_std': float(np.std([r['baseline_acc'] for r in k_results])),
                'pca_mean': float(np.mean([r['pca_acc'] for r in k_results])),
                'pca_std': float(np.std([r['pca_acc'] for r in k_results])),
                'vqd_mean': float(np.mean([r['vqd_acc'] for r in k_results])),
                'vqd_std': float(np.std([r['vqd_acc'] for r in k_results])),
                'vqd_pca_gap_mean': float(np.mean([r['vqd_pca_gap'] for r in k_results])),
                'vqd_pca_gap_std': float(np.std([r['vqd_pca_gap'] for r in k_results])),
                'vqd_baseline_gap_mean': float(np.mean([r['vqd_baseline_gap'] for r in k_results])),
                'vqd_baseline_gap_std': float(np.std([r['vqd_baseline_gap'] for r in k_results])),
            }
        
        return summary


def main():
    """Run cross-validation."""
    
    print("\n" + "="*80)
    print("VQD-DTW CROSS-VALIDATION")
    print("="*80)
    print("Validating findings with multiple random seeds")
    print("="*80 + "\n")
    
    # Run with 3 k values and 5 splits for reasonable runtime
    pipeline = CrossValidationPipeline(
        k_values=[4, 8, 12],  # Test key k values
        n_splits=5,           # 5 different random seeds
        n_train=300,
        n_test=60,
        pre_k=16
    )
    
    (pipeline
     .run_cross_validation()
     .analyze_results()
     .save_results('results/cross_validation_results.json'))
    
    print("="*80)
    print("✅ Cross-validation complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
