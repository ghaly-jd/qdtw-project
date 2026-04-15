"""
K-Sweep with Confidence Intervals
==================================

Runs k={6,8,10,12} with 5 random seeds to compute mean±std for both PCA and VQD.
Uses per-sequence centering (the fair setting we validated).

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

class KSweepCIExperiment:
    """K-sweep with confidence intervals across multiple seeds."""
    
    def __init__(self, k_values=[6, 8, 10, 12], seeds=[42, 123, 456, 789, 2024],
                 n_train=300, n_test=60, pre_k=16):
        self.k_values = k_values
        self.seeds = seeds
        self.n_train = n_train
        self.n_test = n_test
        self.pre_k = pre_k
        
        # Results storage
        self.results = {
            'config': {
                'k_values': k_values,
                'seeds': seeds,
                'n_train': n_train,
                'n_test': n_test,
                'pre_k': pre_k,
                'centering': 'per-sequence',
                'date': datetime.now().isoformat()
            },
            'by_seed': {},  # seed -> k -> method -> metrics
            'aggregated': {}  # k -> method -> {mean, std, ci95}
        }
        
    def load_and_prepare_data(self, seed):
        """Load data and create train/test split for given seed."""
        print(f"\n{'='*70}")
        print(f"Loading data with seed={seed}")
        print(f"{'='*70}")
        
        # Load sequences (data is in parent directory)
        data_path = Path(__file__).parent.parent / "msr_action_data"
        sequences, labels = load_all_sequences(str(data_path))
        print(f"Loaded {len(sequences)} sequences, {len(np.unique(labels))} classes")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels,
            train_size=self.n_train,
            test_size=self.n_test,
            random_state=seed,
            stratify=labels
        )
        
        print(f"Split: {len(X_train)} train, {len(X_test)} test")
        
        return X_train, X_test, y_train, y_test
    
    def build_frame_bank(self, X_train):
        """Build frame bank and pre-reduce to 16D."""
        print("\nBuilding frame bank...")
        frame_bank = np.vstack([seq for seq in X_train])
        print(f"Frame bank shape: {frame_bank.shape}")
        
        # Normalize
        self.scaler = StandardScaler()
        frame_bank_scaled = self.scaler.fit_transform(frame_bank)
        
        # Pre-reduce 60D -> 16D
        self.pca_pre = PCA(n_components=self.pre_k)
        frame_bank_reduced = self.pca_pre.fit_transform(frame_bank_scaled)
        print(f"Pre-reduced to {self.pre_k}D, variance: {self.pca_pre.explained_variance_ratio_.sum():.3f}")
        
        return frame_bank_reduced
    
    def project_sequence(self, seq, U_proj):
        """
        Project a single sequence using per-sequence centering.
        
        Steps:
        1. Normalize with train statistics
        2. Pre-reduce 60D -> 16D
        3. Center per-sequence
        4. Project with U_proj
        """
        # Normalize
        seq_norm = self.scaler.transform(seq)
        
        # Pre-reduce
        seq_reduced = self.pca_pre.transform(seq_norm)
        
        # Center per-sequence
        mean = np.mean(seq_reduced, axis=0)
        seq_centered = seq_reduced - mean
        
        # Project
        seq_proj = seq_centered @ U_proj.T
        
        return seq_proj
    
    def evaluate_pca(self, X_train, X_test, y_train, y_test, frame_bank_reduced, k):
        """Evaluate classical PCA for given k."""
        print(f"\n--- PCA k={k} ---")
        
        # Learn PCA on frame bank
        pca = PCA(n_components=k)
        pca.fit(frame_bank_reduced)
        print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Project all sequences
        X_train_proj = [self.project_sequence(seq, pca.components_) for seq in X_train]
        X_test_proj = [self.project_sequence(seq, pca.components_) for seq in X_test]
        
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
        print(f"PCA Accuracy: {accuracy*100:.1f}%")
        
        return {
            'accuracy': accuracy,
            'time': elapsed,
            'variance_explained': float(pca.explained_variance_ratio_.sum())
        }
    
    def evaluate_vqd(self, X_train, X_test, y_train, y_test, frame_bank_reduced, k):
        """Evaluate VQD for given k."""
        print(f"\n--- VQD k={k} ---")
        
        # Learn VQD on frame bank
        U_vqd, eigenvalues, logs = vqd_quantum_pca(
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
        
        # Use aligned basis if available
        if 'U_vqd_aligned' in logs:
            U_proj = logs['U_vqd_aligned']
            print("Using Procrustes-aligned basis")
        else:
            U_proj = U_vqd
        
        # Quality metrics
        orth_error = logs.get('mean_orthogonality_error', np.nan)
        max_angle = logs.get('max_principal_angle', np.nan)
        print(f"VQD orthogonality error: {orth_error:.2e}")
        print(f"VQD max principal angle: {max_angle:.1f}°")
        
        # Project all sequences
        X_train_proj = [self.project_sequence(seq, U_proj) for seq in X_train]
        X_test_proj = [self.project_sequence(seq, U_proj) for seq in X_test]
        
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
        print(f"VQD Accuracy: {accuracy*100:.1f}%")
        
        return {
            'accuracy': accuracy,
            'time': elapsed,
            'orthogonality_error': float(orth_error),
            'max_angle': float(max_angle),
            'eigenvalues': eigenvalues.tolist()
        }
    
    def run_single_seed(self, seed):
        """Run all k values for a single seed."""
        print(f"\n{'#'*70}")
        print(f"# SEED = {seed}")
        print(f"{'#'*70}")
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_and_prepare_data(seed)
        
        # Build frame bank
        frame_bank_reduced = self.build_frame_bank(X_train)
        
        # Store results for this seed
        seed_results = {}
        
        # Run for each k
        for k in self.k_values:
            print(f"\n{'='*70}")
            print(f"Testing k={k}")
            print(f"{'='*70}")
            
            # PCA
            pca_metrics = self.evaluate_pca(X_train, X_test, y_train, y_test, 
                                           frame_bank_reduced, k)
            
            # VQD
            vqd_metrics = self.evaluate_vqd(X_train, X_test, y_train, y_test,
                                           frame_bank_reduced, k)
            
            # Store
            seed_results[k] = {
                'pca': pca_metrics,
                'vqd': vqd_metrics,
                'gap': vqd_metrics['accuracy'] - pca_metrics['accuracy']
            }
            
            print(f"\nk={k} Summary:")
            print(f"  PCA: {pca_metrics['accuracy']*100:.1f}%")
            print(f"  VQD: {vqd_metrics['accuracy']*100:.1f}%")
            print(f"  Gap: {seed_results[k]['gap']*100:+.1f}%")
        
        return seed_results
    
    def compute_statistics(self):
        """Compute mean, std, and 95% CI for each k and method."""
        print(f"\n{'='*70}")
        print("Computing aggregate statistics...")
        print(f"{'='*70}")
        
        for k in self.k_values:
            # Collect accuracies across seeds
            pca_accs = []
            vqd_accs = []
            gaps = []
            
            for seed in self.seeds:
                if seed in self.results['by_seed']:
                    pca_accs.append(self.results['by_seed'][seed][k]['pca']['accuracy'])
                    vqd_accs.append(self.results['by_seed'][seed][k]['vqd']['accuracy'])
                    gaps.append(self.results['by_seed'][seed][k]['gap'])
            
            # Convert to numpy
            pca_accs = np.array(pca_accs)
            vqd_accs = np.array(vqd_accs)
            gaps = np.array(gaps)
            
            # Compute statistics
            self.results['aggregated'][k] = {
                'pca': {
                    'mean': float(np.mean(pca_accs)),
                    'std': float(np.std(pca_accs, ddof=1)),
                    'ci95': float(1.96 * np.std(pca_accs, ddof=1) / np.sqrt(len(pca_accs)))
                },
                'vqd': {
                    'mean': float(np.mean(vqd_accs)),
                    'std': float(np.std(vqd_accs, ddof=1)),
                    'ci95': float(1.96 * np.std(vqd_accs, ddof=1) / np.sqrt(len(vqd_accs)))
                },
                'gap': {
                    'mean': float(np.mean(gaps)),
                    'std': float(np.std(gaps, ddof=1)),
                    'ci95': float(1.96 * np.std(gaps, ddof=1) / np.sqrt(len(gaps)))
                }
            }
            
            print(f"\nk={k}:")
            print(f"  PCA: {self.results['aggregated'][k]['pca']['mean']*100:.1f}% "
                  f"± {self.results['aggregated'][k]['pca']['std']*100:.1f}% "
                  f"(95% CI: ±{self.results['aggregated'][k]['pca']['ci95']*100:.1f}%)")
            print(f"  VQD: {self.results['aggregated'][k]['vqd']['mean']*100:.1f}% "
                  f"± {self.results['aggregated'][k]['vqd']['std']*100:.1f}% "
                  f"(95% CI: ±{self.results['aggregated'][k]['vqd']['ci95']*100:.1f}%)")
            print(f"  Gap: {self.results['aggregated'][k]['gap']['mean']*100:+.1f}% "
                  f"± {self.results['aggregated'][k]['gap']['std']*100:.1f}% "
                  f"(95% CI: ±{self.results['aggregated'][k]['gap']['ci95']*100:.1f}%)")
    
    def print_summary_table(self):
        """Print LaTeX-ready table."""
        print(f"\n{'='*70}")
        print("SUMMARY TABLE (LaTeX format)")
        print(f"{'='*70}\n")
        
        print("k & PCA Accuracy & VQD Accuracy & Gap \\\\")
        print("\\hline")
        
        for k in self.k_values:
            if k in self.results['aggregated']:
                agg = self.results['aggregated'][k]
                pca_mean = agg['pca']['mean'] * 100
                pca_std = agg['pca']['std'] * 100
                vqd_mean = agg['vqd']['mean'] * 100
                vqd_std = agg['vqd']['std'] * 100
                gap_mean = agg['gap']['mean'] * 100
                gap_std = agg['gap']['std'] * 100
                
                print(f"{k} & ${pca_mean:.1f} \\pm {pca_std:.1f}$ & "
                      f"${vqd_mean:.1f} \\pm {vqd_std:.1f}$ & "
                      f"${gap_mean:+.1f} \\pm {gap_std:.1f}$ \\\\")
    
    def run(self):
        """Run full experiment."""
        print("="*70)
        print("K-SWEEP WITH CONFIDENCE INTERVALS")
        print("="*70)
        print(f"k values: {self.k_values}")
        print(f"Seeds: {self.seeds}")
        print(f"Train/test: {self.n_train}/{self.n_test}")
        print(f"Centering: per-sequence (fair setting)")
        print("="*70)
        
        # Run each seed
        for seed in self.seeds:
            seed_results = self.run_single_seed(seed)
            self.results['by_seed'][seed] = seed_results
            
            # Save incremental results
            self.save_results()
        
        # Compute statistics
        self.compute_statistics()
        
        # Print summary
        self.print_summary_table()
        
        # Final save
        self.save_results()
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE!")
        print(f"Results saved to: results/k_sweep_ci_results.json")
        print("="*70)
    
    def save_results(self):
        """Save results to JSON."""
        output_path = Path(__file__).parent / "results" / "k_sweep_ci_results.json"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

if __name__ == "__main__":
    # Run experiment
    experiment = KSweepCIExperiment(
        k_values=[6, 8, 10, 12],
        seeds=[42, 123, 456, 789, 2024],
        n_train=300,
        n_test=60,
        pre_k=16
    )
    
    experiment.run()
