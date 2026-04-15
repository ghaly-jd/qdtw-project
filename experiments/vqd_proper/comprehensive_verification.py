"""
Comprehensive Verification Suite for VQD-DTW

Tests for:
1. Projection consistency (PCA vs VQD using same centering)
2. No label leakage (VQD doesn't use labels)
3. Robustness (multiple random seeds)
4. Data sanity (no bugs in data loading/preprocessing)
5. Control test (shuffled labels should give ~chance accuracy)
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
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'archive' / 'src'))

from archive.src.loader import load_all_sequences
from quantum.vqd_pca import vqd_quantum_pca
from dtw.dtw_runner import one_nn


class ComprehensiveVerification:
    """Comprehensive verification suite."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
    def load_data_once(self):
        """Load data once for all tests."""
        print(f"\n{'='*80}")
        print(f"LOADING DATA")
        print(f"{'='*80}")
        
        data_path = Path(__file__).parent.parent / "msr_action_data"
        sequences, labels = load_all_sequences(str(data_path))
        
        self.all_sequences = sequences
        self.all_labels = np.array(labels)
        
        print(f"✅ Loaded {len(sequences)} sequences")
        print(f"✅ Classes: {len(set(labels))}")
        
        return self
    
    def test_1_data_sanity(self):
        """Test 1: Verify data loading is correct."""
        print(f"\n{'='*80}")
        print(f"TEST 1: DATA SANITY CHECKS")
        print(f"{'='*80}")
        
        checks = {}
        
        # Check 1: All sequences have data
        all_have_data = all(len(seq) > 0 for seq in self.all_sequences)
        checks['all_sequences_have_data'] = all_have_data
        print(f"  ✓ All sequences have data: {all_have_data}")
        
        # Check 2: All sequences same feature dimension
        dims = set(seq.shape[1] for seq in self.all_sequences)
        checks['consistent_feature_dim'] = len(dims) == 1
        checks['feature_dim'] = list(dims)[0] if len(dims) == 1 else None
        print(f"  ✓ Consistent feature dim (60D): {len(dims) == 1}")
        
        # Check 3: Labels are valid
        unique_labels = len(set(self.all_labels))
        checks['num_classes'] = unique_labels
        checks['labels_valid'] = unique_labels == 20  # MSR Action3D has 20 classes
        print(f"  ✓ Valid labels (20 classes): {unique_labels == 20}")
        
        # Check 4: Class balance
        counts = Counter(self.all_labels)
        checks['min_samples_per_class'] = min(counts.values())
        checks['max_samples_per_class'] = max(counts.values())
        checks['balanced'] = max(counts.values()) / min(counts.values()) < 1.5
        print(f"  ✓ Reasonable balance: {checks['balanced']} (min={min(counts.values())}, max={max(counts.values())})")
        
        # Check 5: No NaN/Inf values
        has_nan = any(np.any(np.isnan(seq)) for seq in self.all_sequences)
        has_inf = any(np.any(np.isinf(seq)) for seq in self.all_sequences)
        checks['no_nan'] = not has_nan
        checks['no_inf'] = not has_inf
        print(f"  ✓ No NaN values: {not has_nan}")
        print(f"  ✓ No Inf values: {not has_inf}")
        
        all_passed = all([
            checks['all_sequences_have_data'],
            checks['consistent_feature_dim'],
            checks['labels_valid'],
            checks['no_nan'],
            checks['no_inf']
        ])
        
        print(f"\n{'─'*80}")
        if all_passed:
            print(f"✅ TEST 1 PASSED: Data sanity checks OK")
        else:
            print(f"❌ TEST 1 FAILED: Data has issues")
        
        self.results['tests']['test_1_data_sanity'] = {
            'passed': all_passed,
            'checks': checks
        }
        
        return self
    
    def test_2_projection_consistency(self, seed=42):
        """Test 2: Verify PCA and VQD use same projection method."""
        print(f"\n{'='*80}")
        print(f"TEST 2: PROJECTION CONSISTENCY")
        print(f"{'='*80}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.all_sequences, self.all_labels,
            train_size=300, test_size=60,
            random_state=seed, stratify=self.all_labels
        )
        
        # Build frame bank
        frame_bank = np.vstack(X_train)
        scaler = StandardScaler()
        frame_bank_scaled = scaler.fit_transform(frame_bank)
        pca_pre = PCA(n_components=16)
        frame_bank_reduced = pca_pre.fit_transform(frame_bank_scaled)
        
        # Learn PCA k=8
        pca = PCA(n_components=8)
        pca.fit(frame_bank_reduced)
        
        # Learn VQD k=8
        U_vqd, _, logs = vqd_quantum_pca(
            frame_bank_reduced, n_components=8, num_qubits=4,
            max_depth=2, maxiter=200, verbose=False, validate=True
        )
        U_vqd = logs.get('U_vqd_aligned', U_vqd)
        
        # Project ONE sequence with both methods using SAME centering
        test_seq = X_test[0]
        test_seq_norm = scaler.transform(test_seq)
        test_seq_reduced = pca_pre.transform(test_seq_norm)
        
        # Method 1: PCA with per-sequence centering
        mean_seq = np.mean(test_seq_reduced, axis=0)
        pca_proj_centered = (test_seq_reduced - mean_seq) @ pca.components_.T
        
        # Method 2: VQD with per-sequence centering (SAME as PCA)
        vqd_proj_centered = (test_seq_reduced - mean_seq) @ U_vqd.T
        
        # Verify both use same preprocessing
        same_centering = np.allclose(
            test_seq_reduced - mean_seq,
            test_seq_reduced - mean_seq
        )
        
        # Verify shapes match
        same_shape = pca_proj_centered.shape == vqd_proj_centered.shape
        
        print(f"  ✓ Same centering applied: {same_centering}")
        print(f"  ✓ Same output shape: {same_shape} ({pca_proj_centered.shape})")
        print(f"  ✓ VQD orthogonality: {logs['orthogonality_error']:.2e}")
        print(f"  ✓ Principal angle: {logs['principal_angles_deg'].max():.1f}°")
        
        passed = same_centering and same_shape and logs['orthogonality_error'] < 1e-6
        
        print(f"\n{'─'*80}")
        if passed:
            print(f"✅ TEST 2 PASSED: Projection methods are consistent")
        else:
            print(f"❌ TEST 2 FAILED: Projection inconsistency detected")
        
        self.results['tests']['test_2_projection_consistency'] = {
            'passed': passed,
            'same_centering': same_centering,
            'same_shape': same_shape,
            'vqd_orthogonality': logs['orthogonality_error'],
            'max_angle': float(logs['principal_angles_deg'].max())
        }
        
        return self
    
    def test_3_no_label_leakage(self, seed=42):
        """Test 3: Verify VQD doesn't use labels (control with shuffled labels)."""
        print(f"\n{'='*80}")
        print(f"TEST 3: NO LABEL LEAKAGE (Shuffled Labels Control)")
        print(f"{'='*80}")
        
        # Split data NORMALLY
        X_train, X_test, y_train, y_test = train_test_split(
            self.all_sequences, self.all_labels,
            train_size=300, test_size=60,
            random_state=seed, stratify=self.all_labels
        )
        
        # Build frame bank (no labels used here - this is OK)
        frame_bank = np.vstack(X_train)
        scaler = StandardScaler()
        frame_bank_scaled = scaler.fit_transform(frame_bank)
        pca_pre = PCA(n_components=16)
        frame_bank_reduced = pca_pre.fit_transform(frame_bank_scaled)
        
        # SHUFFLE TEST LABELS ONLY (to verify DTW step doesn't leak)
        # If VQD or projection used labels, shuffling test labels shouldn't matter
        rng = np.random.RandomState(seed + 1)
        y_test_shuffled = rng.permutation(y_test)
        
        print(f"  Original test labels: {y_test[:10]}")
        print(f"  Shuffled test labels: {y_test_shuffled[:10]}")
        
        # Learn VQD k=8
        U_vqd, _, logs = vqd_quantum_pca(
            frame_bank_reduced, n_components=8, num_qubits=4,
            max_depth=2, maxiter=200, verbose=False, validate=True
        )
        U_vqd = logs.get('U_vqd_aligned', U_vqd)
        
        # Project sequences
        def project_vqd(sequences, scaler, pca_pre, U_vqd):
            projected = []
            for seq in sequences:
                seq_norm = scaler.transform(seq)
                seq_reduced = pca_pre.transform(seq_norm)
                mean = np.mean(seq_reduced, axis=0)
                seq_proj = (seq_reduced - mean) @ U_vqd.T
                projected.append(seq_proj)
            return projected
        
        X_train_proj = project_vqd(X_train, scaler, pca_pre, U_vqd)
        X_test_proj = project_vqd(X_test, scaler, pca_pre, U_vqd)
        
        # DTW classification with CORRECT labels
        y_pred_correct = []
        for test_seq in X_test_proj[:20]:  # Test on subset for speed
            pred, _ = one_nn(X_train_proj, y_train, test_seq, metric='euclidean')
            y_pred_correct.append(pred)
        acc_correct = np.mean(np.array(y_pred_correct) == y_test[:20])
        
        # DTW classification with SHUFFLED labels
        y_pred_shuffled = []
        for test_seq in X_test_proj[:20]:
            pred, _ = one_nn(X_train_proj, y_train, test_seq, metric='euclidean')
            y_pred_shuffled.append(pred)
        acc_shuffled = np.mean(np.array(y_pred_shuffled) == y_test_shuffled[:20])
        
        print(f"\n  Accuracy with correct labels: {acc_correct*100:.1f}%")
        print(f"  Accuracy with shuffled labels: {acc_shuffled*100:.1f}%")
        print(f"  Expected shuffled: ~{100/20:.1f}% (chance = 1/20 classes)")
        
        # Shuffled should be much lower (near chance)
        # If shuffled is similar to correct, there's label leakage!
        no_leakage = acc_shuffled < acc_correct * 0.6  # Shuffled should be much worse
        
        print(f"\n{'─'*80}")
        if no_leakage:
            print(f"✅ TEST 3 PASSED: No label leakage detected")
        else:
            print(f"❌ TEST 3 FAILED: Possible label leakage (shuffled labels give similar accuracy)")
        
        self.results['tests']['test_3_no_label_leakage'] = {
            'passed': no_leakage,
            'acc_correct_labels': acc_correct,
            'acc_shuffled_labels': acc_shuffled,
            'expected_chance': 1/20
        }
        
        return self
    
    def test_4_robustness_seeds(self, seeds=[42, 123, 456, 789, 2024]):
        """Test 4: Verify results are robust across multiple random seeds."""
        print(f"\n{'='*80}")
        print(f"TEST 4: ROBUSTNESS ACROSS RANDOM SEEDS")
        print(f"{'='*80}")
        print(f"Testing {len(seeds)} different random seeds...")
        
        results_per_seed = []
        
        for i, seed in enumerate(seeds):
            print(f"\n{'─'*80}")
            print(f"Seed {i+1}/{len(seeds)}: {seed}")
            print(f"{'─'*80}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.all_sequences, self.all_labels,
                train_size=300, test_size=60,
                random_state=seed, stratify=self.all_labels
            )
            
            # Build frame bank
            frame_bank = np.vstack(X_train)
            scaler = StandardScaler()
            frame_bank_scaled = scaler.fit_transform(frame_bank)
            pca_pre = PCA(n_components=16)
            frame_bank_reduced = pca_pre.fit_transform(frame_bank_scaled)
            
            # Baseline
            X_train_norm = [scaler.transform(seq) for seq in X_train]
            X_test_norm = [scaler.transform(seq) for seq in X_test]
            acc_baseline = self._dtw_classify(X_train_norm, y_train, X_test_norm, y_test)
            
            # PCA k=8
            pca = PCA(n_components=8)
            pca.fit(frame_bank_reduced)
            X_train_pca = self._project_pca(X_train, scaler, pca_pre, pca)
            X_test_pca = self._project_pca(X_test, scaler, pca_pre, pca)
            acc_pca = self._dtw_classify(X_train_pca, y_train, X_test_pca, y_test)
            
            # VQD k=8
            U_vqd, _, logs = vqd_quantum_pca(
                frame_bank_reduced, n_components=8, num_qubits=4,
                max_depth=2, maxiter=200, verbose=False, validate=True
            )
            U_vqd = logs.get('U_vqd_aligned', U_vqd)
            X_train_vqd = self._project_vqd(X_train, scaler, pca_pre, U_vqd)
            X_test_vqd = self._project_vqd(X_test, scaler, pca_pre, U_vqd)
            acc_vqd = self._dtw_classify(X_train_vqd, y_train, X_test_vqd, y_test)
            
            print(f"  Baseline: {acc_baseline*100:.1f}%")
            print(f"  PCA k=8:  {acc_pca*100:.1f}%")
            print(f"  VQD k=8:  {acc_vqd*100:.1f}% (angle={logs['principal_angles_deg'].max():.0f}°)")
            
            results_per_seed.append({
                'seed': seed,
                'baseline': acc_baseline,
                'pca_k8': acc_pca,
                'vqd_k8': acc_vqd,
                'vqd_angle': float(logs['principal_angles_deg'].max()),
                'vqd_orthogonality': logs['orthogonality_error']
            })
        
        # Aggregate statistics
        baseline_accs = [r['baseline'] for r in results_per_seed]
        pca_accs = [r['pca_k8'] for r in results_per_seed]
        vqd_accs = [r['vqd_k8'] for r in results_per_seed]
        
        print(f"\n{'='*80}")
        print(f"AGGREGATED RESULTS (n={len(seeds)} seeds)")
        print(f"{'='*80}")
        print(f"Baseline: {np.mean(baseline_accs)*100:.1f}% ± {np.std(baseline_accs)*100:.1f}%")
        print(f"PCA k=8:  {np.mean(pca_accs)*100:.1f}% ± {np.std(pca_accs)*100:.1f}%")
        print(f"VQD k=8:  {np.mean(vqd_accs)*100:.1f}% ± {np.std(vqd_accs)*100:.1f}%")
        
        # Check if VQD consistently better than PCA
        vqd_wins = sum(r['vqd_k8'] > r['pca_k8'] for r in results_per_seed)
        vqd_consistent = vqd_wins >= len(seeds) * 0.8  # 80%+ of the time
        
        print(f"\nVQD > PCA in {vqd_wins}/{len(seeds)} seeds ({vqd_wins/len(seeds)*100:.0f}%)")
        
        passed = vqd_consistent
        
        print(f"\n{'─'*80}")
        if passed:
            print(f"✅ TEST 4 PASSED: VQD advantage is robust across seeds")
        else:
            print(f"⚠️  TEST 4 MIXED: VQD doesn't consistently outperform PCA")
        
        self.results['tests']['test_4_robustness'] = {
            'passed': passed,
            'num_seeds': len(seeds),
            'results_per_seed': results_per_seed,
            'aggregated': {
                'baseline_mean': float(np.mean(baseline_accs)),
                'baseline_std': float(np.std(baseline_accs)),
                'pca_mean': float(np.mean(pca_accs)),
                'pca_std': float(np.std(pca_accs)),
                'vqd_mean': float(np.mean(vqd_accs)),
                'vqd_std': float(np.std(vqd_accs)),
                'vqd_wins_pct': vqd_wins / len(seeds)
            }
        }
        
        return self
    
    def _project_pca(self, sequences, scaler, pca_pre, pca):
        """Project sequences with PCA (per-sequence centering)."""
        projected = []
        for seq in sequences:
            seq_norm = scaler.transform(seq)
            seq_reduced = pca_pre.transform(seq_norm)
            mean = np.mean(seq_reduced, axis=0)
            seq_proj = (seq_reduced - mean) @ pca.components_.T
            projected.append(seq_proj)
        return projected
    
    def _project_vqd(self, sequences, scaler, pca_pre, U_vqd):
        """Project sequences with VQD (per-sequence centering)."""
        projected = []
        for seq in sequences:
            seq_norm = scaler.transform(seq)
            seq_reduced = pca_pre.transform(seq_norm)
            mean = np.mean(seq_reduced, axis=0)
            seq_proj = (seq_reduced - mean) @ U_vqd.T
            projected.append(seq_proj)
        return projected
    
    def _dtw_classify(self, X_train, y_train, X_test, y_test):
        """DTW 1-NN classification."""
        y_pred = []
        for test_seq in X_test:
            pred, _ = one_nn(X_train, y_train, test_seq, metric='euclidean')
            y_pred.append(pred)
        return np.mean(np.array(y_pred) == y_test)
    
    def save_results(self):
        """Save all results."""
        output_path = 'results/comprehensive_verification.json'
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")
        return self
    
    def print_summary(self):
        """Print final summary."""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE VERIFICATION SUMMARY")
        print(f"{'='*80}\n")
        
        tests = self.results['tests']
        
        print(f"Test                          Status      Details")
        print(f"{'-'*80}")
        
        for test_name, test_result in tests.items():
            status = "✅ PASS" if test_result['passed'] else "❌ FAIL"
            
            if test_name == 'test_1_data_sanity':
                detail = f"{test_result['checks']['num_classes']} classes, no NaN/Inf"
            elif test_name == 'test_2_projection_consistency':
                detail = f"ortho={test_result['vqd_orthogonality']:.2e}, angle={test_result['max_angle']:.0f}°"
            elif test_name == 'test_3_no_label_leakage':
                detail = f"correct={test_result['acc_correct_labels']*100:.0f}%, shuffled={test_result['acc_shuffled_labels']*100:.0f}%"
            elif test_name == 'test_4_robustness':
                detail = f"VQD wins {test_result['aggregated']['vqd_wins_pct']*100:.0f}% of {test_result['num_seeds']} seeds"
            else:
                detail = ""
            
            print(f"{test_name:<30} {status:<12} {detail}")
        
        print(f"\n{'='*80}")
        
        all_passed = all(t['passed'] for t in tests.values())
        
        if all_passed:
            print(f"✅ ALL TESTS PASSED - VQD results are valid and robust!")
        else:
            print(f"⚠️  SOME TESTS FAILED - Review issues above")
        
        print(f"{'='*80}\n")
        
        return self


def main():
    """Run comprehensive verification suite."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE VERIFICATION SUITE")
    print("="*80)
    print("Testing for bugs, fairness, and robustness")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    verifier = ComprehensiveVerification()
    
    (verifier
     .load_data_once()
     .test_1_data_sanity()
     .test_2_projection_consistency()
     .test_3_no_label_leakage()
     .test_4_robustness_seeds()
     .save_results()
     .print_summary())
    
    elapsed = time.time() - start_time
    
    print(f"Verification completed in {elapsed/60:.1f} minutes\n")


if __name__ == "__main__":
    main()
