"""
Optimal Pre-Reduction Dimensionality Experiment
================================================

Research Question: What is the optimal pre-reduction size for VQD?

Pipeline: 60D → {8, 12, 16, 20, 24, 32}D → kD (PCA/VQD) → DTW

This experiment answers:
1. Is 16D the optimal pre-reduction size?
2. What's the trade-off: info loss (too small) vs noise (too large)?
3. How does pre-reduction size affect VQD advantage?

Expected: U-shaped curve for VQD advantage
- Too small (8D): Information loss dominates
- Sweet spot (16D): Best balance
- Too large (32D): Noise retained, VQD struggles

Statistical validation: 5 seeds for each configuration
Target k: 8 (known optimal from previous experiments)

Author: VQD-DTW Research Team
Date: December 24, 2025
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

class OptimalPreReductionExperiment:
    """Test different pre-reduction sizes to find optimal."""
    
    def __init__(self, 
                 pre_dims=[8, 12, 16, 20, 24, 32],
                 k=8,
                 seeds=[42, 123, 456, 789, 2024],
                 n_train=300,
                 n_test=60):
        """
        Args:
            pre_dims: List of pre-reduction dimensions to test
            k: Target dimension after VQD (fixed at 8 based on previous results)
            seeds: Random seeds for statistical validation
            n_train: Training sequences
            n_test: Test sequences
        """
        self.pre_dims = pre_dims
        self.k = k
        self.seeds = seeds
        self.n_train = n_train
        self.n_test = n_test
        
        # Validate configurations
        for pre_dim in pre_dims:
            if pre_dim < k:
                raise ValueError(f"Pre-reduction dim {pre_dim} must be >= k={k}")
        
        self.results = {
            'config': {
                'pre_dims': pre_dims,
                'k': k,
                'seeds': seeds,
                'n_train': n_train,
                'n_test': n_test,
                'date': datetime.now().isoformat()
            },
            'by_seed': {},
            'aggregated': {}
        }
    
    def load_and_prepare_data(self, seed):
        """Load MSR Action3D and split."""
        print(f"\nLoading MSR Action3D dataset...")
        
        # Load sequences (data is in parent directory)
        data_path = Path(__file__).parent.parent / "msr_action_data"
        sequences, labels = load_all_sequences(str(data_path))
        print(f"Loaded {len(sequences)} sequences, {len(np.unique(labels))} classes")
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels,
            train_size=self.n_train,
            test_size=self.n_test,
            stratify=labels,
            random_state=seed
        )
        
        print(f"Split: {len(X_train)} train, {len(X_test)} test")
        return X_train, X_test, y_train, y_test
    
    def build_frame_bank(self, sequences, pre_dim):
        """
        Build frame bank with specific pre-reduction size.
        
        Pipeline: Stack all frames → Normalize → Pre-reduce to pre_dim
        
        Args:
            sequences: List of training sequences
            pre_dim: Target pre-reduction dimension
        
        Returns:
            frame_bank_reduced: (n_frames, pre_dim) array
            scaler: Fitted StandardScaler
            pca_pre: Fitted PCA reducer
        """
        print(f"\nBuilding frame bank with pre-reduction to {pre_dim}D...")
        
        # Stack all frames from all sequences
        all_frames = np.vstack([seq for seq in sequences])
        print(f"  Stacked {all_frames.shape[0]} frames, {all_frames.shape[1]}D each")
        
        # Normalize
        scaler = StandardScaler()
        frames_normalized = scaler.fit_transform(all_frames)
        
        # Pre-reduce to pre_dim
        pca_pre = PCA(n_components=pre_dim)
        frame_bank_reduced = pca_pre.fit_transform(frames_normalized)
        
        variance_explained = pca_pre.explained_variance_ratio_.sum()
        print(f"  Pre-reduced 60D → {pre_dim}D")
        print(f"  Variance explained: {variance_explained:.4f}")
        
        return frame_bank_reduced, scaler, pca_pre
    
    def project_sequence(self, seq, scaler, pca_pre, U_proj):
        """
        Project a sequence through the full pipeline.
        
        Steps:
        1. Normalize with training statistics
        2. Pre-reduce 60D → pre_dim
        3. Center per-sequence (mean subtraction)
        4. Project to k-dimensional subspace
        
        Args:
            seq: (n_frames, 60) sequence
            scaler: Fitted StandardScaler
            pca_pre: Fitted PCA pre-reducer
            U_proj: (k, pre_dim) projection matrix
        
        Returns:
            seq_proj: (n_frames, k) projected sequence
        """
        # Normalize
        seq_norm = scaler.transform(seq)
        
        # Pre-reduce
        seq_reduced = pca_pre.transform(seq_norm)
        
        # Center per-sequence
        mean = np.mean(seq_reduced, axis=0)
        seq_centered = seq_reduced - mean
        
        # Project
        seq_proj = seq_centered @ U_proj.T
        
        return seq_proj
    
    def run_single_config(self, seed, pre_dim):
        """Run PCA and VQD for a single (seed, pre_dim) configuration."""
        print(f"\n{'='*70}")
        print(f"SEED={seed}, PRE_DIM={pre_dim}, K={self.k}")
        print(f"{'='*70}")
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_and_prepare_data(seed)
        
        # Build frame bank with specific pre-reduction
        frame_bank, scaler, pca_pre = self.build_frame_bank(X_train, pre_dim)
        
        results = {
            'seed': seed,
            'pre_dim': pre_dim,
            'k': self.k,
            'variance_explained_pre': float(pca_pre.explained_variance_ratio_.sum())
        }
        
        # ===== PCA: pre_dim → kD =====
        print(f"\n--- PCA: {pre_dim}D → {self.k}D ---")
        pca_start = time.time()
        
        pca = PCA(n_components=self.k)
        pca.fit(frame_bank)
        
        pca_variance = pca.explained_variance_ratio_.sum()
        print(f"PCA variance explained: {pca_variance:.4f}")
        
        # Project sequences
        X_train_pca = [self.project_sequence(seq, scaler, pca_pre, pca.components_) 
                       for seq in X_train]
        X_test_pca = [self.project_sequence(seq, scaler, pca_pre, pca.components_) 
                      for seq in X_test]
        
        # DTW classification
        print(f"Running DTW on {len(X_test_pca)} test sequences...")
        pca_preds = []
        y_train_arr = np.array(y_train)
        
        for test_seq in X_test_pca:
            pred, _ = one_nn(X_train_pca, y_train_arr, test_seq)
            pca_preds.append(pred)
        
        pca_preds = np.array(pca_preds)
        pca_acc = np.mean(pca_preds == y_test)
        pca_time = time.time() - pca_start
        
        print(f"PCA Accuracy: {pca_acc*100:.2f}%")
        print(f"PCA Time: {pca_time:.2f}s")
        
        results['pca_accuracy'] = float(pca_acc)
        results['pca_time'] = float(pca_time)
        results['pca_variance_explained'] = float(pca_variance)
        
        # ===== VQD: pre_dim → kD =====
        print(f"\n--- VQD: {pre_dim}D → {self.k}D ---")
        
        # Determine number of qubits needed
        num_qubits = int(np.ceil(np.log2(pre_dim)))
        print(f"VQD using {num_qubits} qubits (2^{num_qubits} = {2**num_qubits} >= {pre_dim})")
        
        vqd_start = time.time()
        
        try:
            U_vqd, eigenvalues, logs = vqd_quantum_pca(
                frame_bank,
                n_components=self.k,
                num_qubits=num_qubits,
                max_depth=2,
                penalty_scale='auto',
                ramped_penalties=True,
                entanglement='alternating',
                maxiter=200,
                validate=True
            )
            
            vqd_time = time.time() - vqd_start
            print(f"VQD training time: {vqd_time/60:.2f} minutes")
            
            # Use aligned basis if available
            if 'U_vqd_aligned' in logs:
                U_proj = logs['U_vqd_aligned']
                print("Using Procrustes-aligned basis")
            else:
                U_proj = U_vqd
            
            # Quality metrics
            orth_error = logs.get('mean_orthogonality_error', np.nan)
            max_angle = logs.get('max_principal_angle', np.nan)
            mean_angle = logs.get('mean_principal_angle', np.nan)
            
            print(f"VQD orthogonality error: {orth_error:.2e}")
            print(f"VQD mean principal angle: {mean_angle:.1f}°")
            print(f"VQD max principal angle: {max_angle:.1f}°")
            
            # Project sequences
            X_train_vqd = [self.project_sequence(seq, scaler, pca_pre, U_proj) 
                          for seq in X_train]
            X_test_vqd = [self.project_sequence(seq, scaler, pca_pre, U_proj) 
                         for seq in X_test]
            
            # DTW classification
            print(f"Running DTW on {len(X_test_vqd)} test sequences...")
            vqd_preds = []
            
            for test_seq in X_test_vqd:
                pred, _ = one_nn(X_train_vqd, y_train_arr, test_seq)
                vqd_preds.append(pred)
            
            vqd_preds = np.array(vqd_preds)
            vqd_acc = np.mean(vqd_preds == y_test)
            
            print(f"VQD Accuracy: {vqd_acc*100:.2f}%")
            
            # Calculate gap
            gap = vqd_acc - pca_acc
            print(f"\nVQD - PCA Gap: {gap*100:+.2f}%")
            
            results['vqd_accuracy'] = float(vqd_acc)
            results['vqd_time'] = float(vqd_time)
            results['vqd_orthogonality_error'] = float(orth_error)
            results['vqd_mean_principal_angle'] = float(mean_angle)
            results['vqd_max_principal_angle'] = float(max_angle)
            results['gap'] = float(gap)
            results['num_qubits'] = int(num_qubits)
            results['success'] = True
            
        except Exception as e:
            print(f"VQD FAILED: {e}")
            results['vqd_accuracy'] = None
            results['vqd_time'] = None
            results['gap'] = None
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def run_all(self):
        """Run all configurations."""
        print("\n" + "="*70)
        print("OPTIMAL PRE-REDUCTION EXPERIMENT")
        print("="*70)
        print(f"Pre-reduction dimensions: {self.pre_dims}")
        print(f"Target dimension k: {self.k}")
        print(f"Seeds: {self.seeds}")
        print(f"Total runs: {len(self.pre_dims)} × {len(self.seeds)} = {len(self.pre_dims) * len(self.seeds)}")
        print("="*70)
        
        for seed in self.seeds:
            seed_key = f"seed_{seed}"
            self.results['by_seed'][seed_key] = {}
            
            for pre_dim in self.pre_dims:
                print(f"\n{'#'*70}")
                print(f"# Running: seed={seed}, pre_dim={pre_dim}, k={self.k}")
                print(f"{'#'*70}")
                
                result = self.run_single_config(seed, pre_dim)
                self.results['by_seed'][seed_key][f"pre_{pre_dim}"] = result
                
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
        """Compute mean/std across seeds for each pre_dim."""
        print("\nComputing aggregated statistics...")
        
        for pre_dim in self.pre_dims:
            pca_accs = []
            vqd_accs = []
            gaps = []
            pca_times = []
            vqd_times = []
            variances = []
            
            for seed in self.seeds:
                seed_key = f"seed_{seed}"
                pre_key = f"pre_{pre_dim}"
                result = self.results['by_seed'][seed_key][pre_key]
                
                if result.get('success', False):
                    pca_accs.append(result['pca_accuracy'])
                    vqd_accs.append(result['vqd_accuracy'])
                    gaps.append(result['gap'])
                    pca_times.append(result['pca_time'])
                    vqd_times.append(result['vqd_time'])
                    variances.append(result['variance_explained_pre'])
            
            self.results['aggregated'][str(pre_dim)] = {
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
                },
                'pca_time': {
                    'mean': float(np.mean(pca_times)),
                    'std': float(np.std(pca_times))
                },
                'vqd_time': {
                    'mean': float(np.mean(vqd_times)) if vqd_times else None,
                    'std': float(np.std(vqd_times)) if vqd_times else None
                },
                'variance_explained': float(np.mean(variances))
            }
    
    def save_results(self):
        """Save results to JSON."""
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "optimal_prereduction_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved: {output_file}")
    
    def print_summary(self):
        """Print summary table."""
        print("\nSUMMARY TABLE:")
        print("="*90)
        print(f"{'Pre-Dim':<10} {'Variance':<12} {'PCA Mean':<12} {'VQD Mean':<12} {'Gap':<15} {'Advantage'}")
        print("-"*90)
        
        for pre_dim in self.pre_dims:
            agg = self.results['aggregated'][str(pre_dim)]
            
            variance = agg['variance_explained'] * 100
            pca_mean = agg['pca']['mean'] * 100
            pca_std = agg['pca']['std'] * 100
            
            if agg['vqd']['mean'] is not None:
                vqd_mean = agg['vqd']['mean'] * 100
                vqd_std = agg['vqd']['std'] * 100
                gap_mean = agg['gap']['mean'] * 100
                gap_std = agg['gap']['std'] * 100
                
                advantage = "✓ YES" if gap_mean > 1.0 else "✗ NO"
                
                print(f"{pre_dim:<10} {variance:>5.1f}%      "
                      f"{pca_mean:.1f}±{pca_std:.1f}%    "
                      f"{vqd_mean:.1f}±{vqd_std:.1f}%    "
                      f"{gap_mean:+.1f}±{gap_std:.1f}%     {advantage}")
            else:
                print(f"{pre_dim:<10} {variance:>5.1f}%      "
                      f"{pca_mean:.1f}±{pca_std:.1f}%    "
                      f"VQD FAILED      N/A            ✗")
        
        print("="*90)
        
        # Find optimal
        best_pre_dim = max(self.pre_dims, 
                          key=lambda d: self.results['aggregated'][str(d)]['gap']['mean'] or -1)
        best_gap = self.results['aggregated'][str(best_pre_dim)]['gap']['mean'] * 100
        
        print(f"\n🏆 OPTIMAL PRE-REDUCTION: {best_pre_dim}D (gap = {best_gap:+.2f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimal Pre-Reduction Experiment')
    parser.add_argument('--pre-dims', nargs='+', type=int, 
                       default=[8, 12, 16, 20, 24, 32],
                       help='Pre-reduction dimensions to test')
    parser.add_argument('--k', type=int, default=8,
                       help='Target dimension after VQD')
    parser.add_argument('--seeds', nargs='+', type=int, 
                       default=[42, 123, 456, 789, 2024],
                       help='Random seeds')
    parser.add_argument('--n-train', type=int, default=300,
                       help='Number of training sequences')
    parser.add_argument('--n-test', type=int, default=60,
                       help='Number of test sequences')
    
    args = parser.parse_args()
    
    print("Configuration:")
    print(f"  Pre-reduction dimensions: {args.pre_dims}")
    print(f"  Target k: {args.k}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Train: {args.n_train}, Test: {args.n_test}")
    print(f"  Pipeline: 60D → {{8,12,16,20,24,32}}D → {args.k}D")
    
    experiment = OptimalPreReductionExperiment(
        pre_dims=args.pre_dims,
        k=args.k,
        seeds=args.seeds,
        n_train=args.n_train,
        n_test=args.n_test
    )
    
    experiment.run_all()
    
    print("\n✨ Optimal pre-reduction experiment complete! ✨")
    print("\nNext steps:")
    print("  1. Visualize: python plot_optimal_prereduction.py")
    print("  2. Check results: results/optimal_prereduction_results.json")
