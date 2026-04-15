"""
No Pre-Reduction Experiment: Direct 60D → kD
==============================================

Tests PCA and VQD directly on 60D data without the 60D→16D pre-reduction step.

Comparison:
- Current pipeline: 60D → 16D (PCA) → kD (PCA/VQD) → DTW
- New pipeline: 60D → kD (PCA/VQD) → DTW

This will reveal:
1. Does pre-reduction help or hurt final accuracy?
2. Can VQD handle 60D directly? (needs 6 qubits)
3. What's the true VQD advantage without intermediate PCA?

Author: VQD-DTW Research Team
Date: November 25, 2025
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

class NoPreReductionExperiment:
    """Compare PCA vs VQD directly on 60D data (no pre-reduction)."""
    
    def __init__(self, k_values=[6, 8, 10, 12], seeds=[42], n_train=300, n_test=60):
        self.k_values = k_values
        self.seeds = seeds
        self.n_train = n_train
        self.n_test = n_test
        
        self.results = {
            'config': {
                'k_values': k_values,
                'seeds': seeds,
                'n_train': n_train,
                'n_test': n_test,
                'pre_reduction': None,  # No pre-reduction!
                'date': datetime.now().isoformat()
            },
            'by_seed': {},
            'aggregated': {}
        }
    
    def load_and_prepare_data(self, seed):
        """Load data with given seed."""
        print(f"\n{'='*70}")
        print(f"Loading data with seed={seed}")
        print(f"{'='*70}")
        
        data_path = Path(__file__).parent.parent / "msr_action_data"
        sequences, labels = load_all_sequences(str(data_path))
        print(f"Loaded {len(sequences)} sequences")
        
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels,
            train_size=self.n_train,
            test_size=self.n_test,
            random_state=seed,
            stratify=labels
        )
        
        return X_train, X_test, np.array(y_train), np.array(y_test)
    
    def build_frame_bank(self, X_train):
        """Build frame bank - NO pre-reduction, stay at 60D."""
        print("\nBuilding frame bank (60D - no pre-reduction)...")
        frame_bank = np.vstack([seq for seq in X_train])
        
        # Only normalize, no PCA pre-reduction
        self.scaler = StandardScaler()
        frame_bank_scaled = self.scaler.fit_transform(frame_bank)
        
        print(f"Frame bank shape: {frame_bank_scaled.shape}")
        print(f"Keeping full 60D features!")
        
        return frame_bank_scaled
    
    def project_sequence(self, seq, U_proj):
        """
        Project a single sequence using per-sequence centering.
        
        Steps:
        1. Normalize with train statistics
        2. Center per-sequence
        3. Project with U_proj
        """
        # Normalize
        seq_norm = self.scaler.transform(seq)
        
        # Center per-sequence
        mean = np.mean(seq_norm, axis=0)
        seq_centered = seq_norm - mean
        
        # Project
        seq_proj = seq_centered @ U_proj.T
        
        return seq_proj
    
    def run_single_config(self, seed, k):
        """Run PCA and VQD for a single (seed, k) configuration."""
        print(f"\n{'='*70}")
        print(f"SEED={seed}, K={k}")
        print(f"{'='*70}")
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_and_prepare_data(seed)
        
        # Build frame bank (60D - no pre-reduction)
        frame_bank = self.build_frame_bank(X_train)
        
        results = {'seed': seed, 'k': k}
        
        # ===== PCA: 60D → kD =====
        print(f"\n--- PCA: 60D → {k}D ---")
        pca = PCA(n_components=k)
        pca.fit(frame_bank)
        print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Project sequences using proper method
        X_train_pca = [self.project_sequence(seq, pca.components_) for seq in X_train]
        X_test_pca = [self.project_sequence(seq, pca.components_) for seq in X_test]
        
        # DTW classification
        print(f"Running DTW on {len(X_test_pca)} test sequences...")
        pca_preds = []
        y_train_arr = np.array(y_train)
        
        for i, test_seq in enumerate(X_test_pca):
            pred, _ = one_nn(X_train_pca, y_train_arr, test_seq)
            pca_preds.append(pred)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(X_test_pca)} test sequences")
        
        pca_preds = np.array(pca_preds)
        pca_acc = np.mean(pca_preds == y_test)
        print(f"PCA Accuracy: {pca_acc*100:.2f}%")
        
        results['pca_accuracy'] = float(pca_acc)
        results['pca_variance_explained'] = float(pca.explained_variance_ratio_.sum())
        
        # ===== VQD: 60D → kD =====
        print(f"\n--- VQD: 60D → {k}D ---")
        print(f"Note: 60D requires 6 qubits (2^6 = 64)")
        print("This will take longer than 16D version...")
        
        start_time = time.time()
        
        try:
            U_vqd, eigenvalues, logs = vqd_quantum_pca(
                frame_bank,
                n_components=k,
                num_qubits=6,  # 2^6 = 64 >= 60
                max_depth=2,
                maxiter=200,
                verbose=True
            )
            
            vqd_time = time.time() - start_time
            print(f"VQD training time: {vqd_time/60:.2f} minutes")
            
            # Project sequences using proper method
            X_train_vqd = [self.project_sequence(seq, U_vqd) for seq in X_train]
            X_test_vqd = [self.project_sequence(seq, U_vqd) for seq in X_test]
            
            # DTW classification
            print(f"Running DTW on {len(X_test_vqd)} test sequences...")
            vqd_preds = []
            
            for i, test_seq in enumerate(X_test_vqd):
                pred, _ = one_nn(X_train_vqd, y_train_arr, test_seq)
                vqd_preds.append(pred)
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(X_test_vqd)} test sequences")
            
            vqd_preds = np.array(vqd_preds)
            vqd_acc = np.mean(vqd_preds == y_test)
            print(f"VQD Accuracy: {vqd_acc*100:.2f}%")
            
            results['vqd_accuracy'] = float(vqd_acc)
            results['vqd_training_time'] = float(vqd_time)
            results['vqd_orthogonality_error'] = float(logs['orthogonality_error'])
            
            if 'principal_angles_deg' in logs:
                results['vqd_principal_angles'] = logs['principal_angles_deg'].tolist()
            
        except Exception as e:
            print(f"VQD failed: {e}")
            results['vqd_accuracy'] = None
            results['vqd_error'] = str(e)
        
        # Compute gap
        if results.get('vqd_accuracy') is not None:
            results['gap'] = float(results['vqd_accuracy'] - results['pca_accuracy'])
            print(f"\nVQD - PCA Gap: {results['gap']*100:+.2f}%")
        
        return results
    
    def run_all(self):
        """Run all configurations."""
        print("="*70)
        print("NO PRE-REDUCTION EXPERIMENT")
        print("Testing: 60D → kD directly (no 60D→16D step)")
        print("="*70)
        
        for seed in self.seeds:
            seed_key = f"seed_{seed}"
            self.results['by_seed'][seed_key] = {}
            
            for k in self.k_values:
                print(f"\n{'#'*70}")
                print(f"# Running: seed={seed}, k={k}")
                print(f"{'#'*70}")
                
                result = self.run_single_config(seed, k)
                self.results['by_seed'][seed_key][f"k_{k}"] = result
                
                # Save after each run
                self.save_results()
        
        # Compute aggregated statistics
        self.compute_aggregated_stats()
        self.save_results()
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE!")
        print("="*70)
        self.print_summary()
    
    def compute_aggregated_stats(self):
        """Compute mean/std across seeds for each k."""
        print("\nComputing aggregated statistics...")
        
        for k in self.k_values:
            pca_accs = []
            vqd_accs = []
            gaps = []
            
            for seed in self.seeds:
                seed_key = f"seed_{seed}"
                k_key = f"k_{k}"
                result = self.results['by_seed'][seed_key][k_key]
                
                pca_accs.append(result['pca_accuracy'])
                if result.get('vqd_accuracy') is not None:
                    vqd_accs.append(result['vqd_accuracy'])
                    gaps.append(result['gap'])
            
            self.results['aggregated'][str(k)] = {
                'pca': {
                    'mean': float(np.mean(pca_accs)),
                    'std': float(np.std(pca_accs)),
                    'values': pca_accs
                },
                'vqd': {
                    'mean': float(np.mean(vqd_accs)) if vqd_accs else None,
                    'std': float(np.std(vqd_accs)) if vqd_accs else None,
                    'values': vqd_accs
                },
                'gap': {
                    'mean': float(np.mean(gaps)) if gaps else None,
                    'std': float(np.std(gaps)) if gaps else None,
                    'values': gaps
                }
            }
    
    def save_results(self):
        """Save results to JSON."""
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "no_prereduction_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved: {output_file}")
    
    def print_summary(self):
        """Print summary table."""
        print("\nSUMMARY TABLE:")
        print("="*70)
        print(f"{'K':<5} {'PCA Mean':<12} {'VQD Mean':<12} {'Gap':<12}")
        print("-"*70)
        
        for k in self.k_values:
            agg = self.results['aggregated'][str(k)]
            pca_mean = agg['pca']['mean'] * 100
            pca_std = agg['pca']['std'] * 100
            
            if agg['vqd']['mean'] is not None:
                vqd_mean = agg['vqd']['mean'] * 100
                vqd_std = agg['vqd']['std'] * 100
                gap_mean = agg['gap']['mean'] * 100
                
                print(f"{k:<5} {pca_mean:.1f}±{pca_std:.1f}%    {vqd_mean:.1f}±{vqd_std:.1f}%    {gap_mean:+.1f}%")
            else:
                print(f"{k:<5} {pca_mean:.1f}±{pca_std:.1f}%    VQD FAILED    N/A")
        
        print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='No Pre-Reduction Experiment')
    parser.add_argument('--k-values', nargs='+', type=int, default=[6, 8, 10, 12],
                       help='K values to test')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                       help='Random seeds')
    parser.add_argument('--n-train', type=int, default=300,
                       help='Number of training sequences')
    parser.add_argument('--n-test', type=int, default=60,
                       help='Number of test sequences')
    
    args = parser.parse_args()
    
    print("Configuration:")
    print(f"  K values: {args.k_values}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Train: {args.n_train}, Test: {args.n_test}")
    print(f"  Pipeline: 60D → kD (NO pre-reduction!)")
    
    experiment = NoPreReductionExperiment(
        k_values=args.k_values,
        seeds=args.seeds,
        n_train=args.n_train,
        n_test=args.n_test
    )
    
    experiment.run_all()
    
    print("\n✨ No pre-reduction experiment complete! ✨")
    print("\nCompare with pre-reduction results:")
    print("  With pre-reduction: results/k_sweep_ci_results.json")
    print("  Without pre-reduction: results/no_prereduction_results.json")
