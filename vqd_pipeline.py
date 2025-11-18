"""
VQD First-Class Subspace Pipeline

Frozen pipeline with comprehensive evaluation:
1. Train-only z-score normalization
2. Frame bank construction
3. VQD quantum PCA (k sweep: 4, 6, 8, 10, 12)
4. Optional Orthogonal Procrustes alignment
5. Project sequences
6. DTW-1NN classification (Euclidean)

Logs all metrics for each run and produces Pareto analysis.
"""

import numpy as np
from pathlib import Path
import sys
import time
import json
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from quantum.vqd_pca import vqd_quantum_pca
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dtw.dtw_runner import one_nn


class VQDPipeline:
    """First-class VQD subspace pipeline with comprehensive logging."""
    
    def __init__(self, k_values=[4, 6, 8, 10, 12], use_procrustes=True, 
                 n_train=100, n_test=30, random_seed=42):
        """
        Initialize VQD pipeline.
        
        Parameters
        ----------
        k_values : list
            Component counts to evaluate
        use_procrustes : bool
            Whether to use Procrustes-aligned basis
        n_train : int
            Number of training samples
        n_test : int
            Number of test samples
        random_seed : int
            Random seed for reproducibility
        """
        self.k_values = k_values
        self.use_procrustes = use_procrustes
        self.n_train = n_train
        self.n_test = n_test
        self.random_seed = random_seed
        self.results = {}
        
    def load_and_split_data(self, data_path):
        """Load data and create train/test split."""
        print(f"\n{'='*80}")
        print(f"DATA LOADING & SPLITTING")
        print(f"{'='*80}")
        
        X = np.load(data_path)
        y = np.arange(len(X)) % 20  # Action classes
        
        print(f"Dataset: {X.shape}")
        print(f"Classes: {len(np.unique(y))}")
        
        # Fixed split for reproducibility
        np.random.seed(self.random_seed)
        indices = np.random.permutation(len(X))
        train_idx = indices[:self.n_train]
        test_idx = indices[self.n_train:self.n_train+self.n_test]
        
        self.X_train_raw = X[train_idx]
        self.X_test_raw = X[test_idx]
        self.y_train = y[train_idx]
        self.y_test = y[test_idx]
        
        print(f"Train: {self.n_train} samples")
        print(f"Test: {self.n_test} samples")
        
        return self
    
    def preprocess(self):
        """Apply train-only z-score normalization."""
        print(f"\n{'─'*80}")
        print(f"PREPROCESSING: Train-only z-score")
        print(f"{'─'*80}")
        
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_raw)
        self.X_test_scaled = self.scaler.transform(self.X_test_raw)
        
        print(f"✅ StandardScaler fit on train, applied to test")
        print(f"   Train mean: {np.mean(self.X_train_scaled):.2e} (should be ~0)")
        print(f"   Train std: {np.std(self.X_train_scaled):.2e} (should be ~1)")
        
        return self
    
    def build_frame_bank(self, pre_k=8):
        """Build frame bank with pre-reduction via classical PCA."""
        print(f"\n{'─'*80}")
        print(f"FRAME BANK: Pre-reduction {self.X_train_raw.shape[1]}D → {pre_k}D")
        print(f"{'─'*80}")
        
        self.pre_k = pre_k
        self.pca_pre = PCA(n_components=pre_k)
        self.X_train_bank = self.pca_pre.fit_transform(self.X_train_scaled)
        self.X_test_bank = self.pca_pre.transform(self.X_test_scaled)
        
        variance = self.pca_pre.explained_variance_ratio_.sum()
        print(f"Variance preserved: {variance*100:.1f}%")
        print(f"Frame bank shape: Train {self.X_train_bank.shape}, Test {self.X_test_bank.shape}")
        
        return self
    
    def evaluate_baseline(self):
        """Evaluate baseline (raw 60D, no reduction)."""
        print(f"\n{'='*80}")
        print(f"BASELINE: Raw 60D (no reduction)")
        print(f"{'='*80}")
        
        acc, ypred, time_per_query = self._classify(
            self.X_train_scaled, self.X_test_scaled
        )
        
        mean_acc, ci_lower, ci_upper = self._bootstrap_ci(self.y_test, ypred)
        
        self.results['baseline'] = {
            'k': 60,
            'method': 'raw',
            'accuracy': acc,
            'accuracy_mean': mean_acc,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'time_per_query': time_per_query,
            'speedup': 1.0,
            'dimensionality': 60
        }
        
        self.baseline_time = time_per_query
        
        print(f"Accuracy: {acc*100:.1f}% [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")
        print(f"Time/query: {time_per_query*1000:.2f} ms")
        
        return self
    
    def evaluate_k_sweep(self):
        """Evaluate VQD and PCA across multiple k values."""
        print(f"\n{'='*80}")
        print(f"K-SWEEP EVALUATION")
        print(f"{'='*80}")
        
        for k in self.k_values:
            print(f"\n{'─'*80}")
            print(f"k = {k}")
            print(f"{'─'*80}")
            
            # Classical PCA
            print(f"\n[PCA {k}D]")
            pca_results = self._evaluate_pca(k)
            self.results[f'pca_k{k}'] = pca_results
            
            # VQD Quantum PCA
            print(f"\n[VQD {k}D]")
            vqd_results = self._evaluate_vqd(k)
            self.results[f'vqd_k{k}'] = vqd_results
            
        return self
    
    def _evaluate_pca(self, k):
        """Evaluate classical PCA for given k."""
        pca = PCA(n_components=k)
        X_train_proj = pca.fit_transform(self.X_train_bank)
        X_test_proj = pca.transform(self.X_test_bank)
        
        acc, ypred, time_per_query = self._classify(X_train_proj, X_test_proj)
        mean_acc, ci_lower, ci_upper = self._bootstrap_ci(self.y_test, ypred)
        speedup = self.baseline_time / time_per_query
        
        print(f"  Accuracy: {acc*100:.1f}% [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")
        print(f"  Time/query: {time_per_query*1000:.2f} ms")
        print(f"  Speedup: {speedup:.2f}×")
        
        return {
            'k': k,
            'method': 'pca',
            'accuracy': acc,
            'accuracy_mean': mean_acc,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'time_per_query': time_per_query,
            'speedup': speedup,
            'dimensionality': k,
            'eigenvalues': pca.explained_variance_.tolist()
        }
    
    def _evaluate_vqd(self, k):
        """Evaluate VQD quantum PCA for given k."""
        # Compute VQD
        U_vqd, eigenvalues_vqd, logs = vqd_quantum_pca(
            self.X_train_bank,
            n_components=k,
            num_qubits=int(np.ceil(np.log2(self.pre_k))),
            max_depth=2,
            penalty_scale='auto',
            ramped_penalties=True,
            entanglement='alternating',
            maxiter=200,
            verbose=False,
            validate=True
        )
        
        # Choose basis (raw or Procrustes-aligned)
        if self.use_procrustes and 'U_vqd_aligned' in logs:
            U_proj = logs['U_vqd_aligned']
            basis_name = 'procrustes'
        else:
            U_proj = U_vqd
            basis_name = 'raw'
        
        # Project data
        train_mean = np.mean(self.X_train_bank, axis=0)
        X_train_proj = (self.X_train_bank - train_mean) @ U_proj.T
        X_test_proj = (self.X_test_bank - train_mean) @ U_proj.T
        
        # Classify
        acc, ypred, time_per_query = self._classify(X_train_proj, X_test_proj)
        mean_acc, ci_lower, ci_upper = self._bootstrap_ci(self.y_test, ypred)
        speedup = self.baseline_time / time_per_query
        
        # Log metrics
        mean_angle = np.mean(logs['principal_angles_deg'])
        max_angle = np.max(logs['principal_angles_deg'])
        procrustes_residual = logs.get('procrustes_residual_after', None)
        procrustes_improvement = logs.get('procrustes_improvement', None)
        orthogonality_error = logs['orthogonality_error']
        
        print(f"  VQD Metrics:")
        print(f"    Orthogonality ||U^T U - I||_F: {orthogonality_error:.2e}")
        print(f"    Principal angles: mean {mean_angle:.1f}°, max {max_angle:.1f}°")
        if procrustes_residual is not None:
            print(f"    Procrustes: residual {procrustes_residual:.4f}, improvement {procrustes_improvement*100:.1f}%")
        print(f"    Rayleigh errors: {np.mean(logs['rayleigh_errors']):.2e} (mean)")
        
        print(f"  Classification ({basis_name} basis):")
        print(f"    Accuracy: {acc*100:.1f}% [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")
        print(f"    Time/query: {time_per_query*1000:.2f} ms")
        print(f"    Speedup: {speedup:.2f}×")
        
        return {
            'k': k,
            'method': 'vqd',
            'basis': basis_name,
            'accuracy': acc,
            'accuracy_mean': mean_acc,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'time_per_query': time_per_query,
            'speedup': speedup,
            'dimensionality': k,
            'eigenvalues': eigenvalues_vqd.tolist(),
            'orthogonality_error': orthogonality_error,
            'mean_principal_angle': mean_angle,
            'max_principal_angle': max_angle,
            'procrustes_residual': procrustes_residual,
            'procrustes_improvement': procrustes_improvement,
            'mean_rayleigh_error': float(np.mean(logs['rayleigh_errors'])),
            'eigenvalue_relative_errors': logs['eigenvalue_relative_errors'].tolist()
        }
    
    def _classify(self, X_train, X_test):
        """DTW 1-NN classification with timing."""
        train_seqs = [X_train[i:i+1] for i in range(len(X_train))]
        train_labels = list(self.y_train)
        
        y_pred = []
        times = []
        
        for i in range(len(X_test)):
            test_seq = X_test[i:i+1]
            start = time.time()
            pred, _ = one_nn(train_seqs, train_labels, test_seq, metric='euclidean')
            times.append(time.time() - start)
            y_pred.append(pred)
        
        y_pred = np.array(y_pred)
        accuracy = np.mean(y_pred == self.y_test)
        mean_time = np.mean(times)
        
        return accuracy, y_pred, mean_time
    
    def _bootstrap_ci(self, y_true, y_pred, n_bootstrap=1000, confidence=0.95):
        """Compute bootstrap confidence interval."""
        accuracies = []
        n = len(y_true)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            acc = np.mean(y_true[indices] == y_pred[indices])
            accuracies.append(acc)
        
        alpha = 1 - confidence
        lower = np.percentile(accuracies, 100 * alpha / 2)
        upper = np.percentile(accuracies, 100 * (1 - alpha / 2))
        mean_acc = np.mean(accuracies)
        
        return mean_acc, lower, upper
    
    def save_results(self, output_path='vqd_pipeline_results.json'):
        """Save results to JSON."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'k_values': self.k_values,
                'use_procrustes': self.use_procrustes,
                'n_train': self.n_train,
                'n_test': self.n_test,
                'random_seed': self.random_seed,
                'pre_k': self.pre_k
            },
            'results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")
        
        return self
    
    def generate_pareto_plots(self, output_dir='figures'):
        """Generate Pareto analysis plots."""
        print(f"\n{'='*80}")
        print(f"GENERATING PARETO PLOTS")
        print(f"{'='*80}")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Extract data
        pca_data = [(r['k'], r['accuracy']*100, r['speedup'], r['time_per_query']*1000) 
                    for key, r in self.results.items() if key.startswith('pca_k')]
        vqd_data = [(r['k'], r['accuracy']*100, r['speedup'], r['time_per_query']*1000,
                     r['max_principal_angle']) 
                    for key, r in self.results.items() if key.startswith('vqd_k')]
        
        if not pca_data or not vqd_data:
            print("⚠️  Insufficient data for plots")
            return self
        
        pca_k, pca_acc, pca_speedup, pca_time = zip(*pca_data)
        vqd_k, vqd_acc, vqd_speedup, vqd_time, vqd_angle = zip(*vqd_data)
        
        # Plot 1: Accuracy vs k
        plt.figure(figsize=(10, 6))
        plt.plot(pca_k, pca_acc, 'o-', label='Classical PCA', linewidth=2, markersize=8)
        plt.plot(vqd_k, vqd_acc, 's-', label='VQD Quantum PCA', linewidth=2, markersize=8)
        plt.xlabel('Number of Components (k)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Classification Accuracy vs Dimensionality', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pareto_accuracy_vs_k.png', dpi=300)
        print(f"  Saved: {output_dir}/pareto_accuracy_vs_k.png")
        plt.close()
        
        # Plot 2: Accuracy vs Speedup
        plt.figure(figsize=(10, 6))
        plt.scatter(pca_speedup, pca_acc, s=100, label='Classical PCA', alpha=0.7)
        plt.scatter(vqd_speedup, vqd_acc, s=100, label='VQD Quantum PCA', alpha=0.7, marker='s')
        for i, k in enumerate(pca_k):
            plt.annotate(f'k={k}', (pca_speedup[i], pca_acc[i]), fontsize=9, alpha=0.7)
        for i, k in enumerate(vqd_k):
            plt.annotate(f'k={k}', (vqd_speedup[i], vqd_acc[i]), fontsize=9, alpha=0.7)
        plt.xlabel('Speedup vs 60D (×)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Pareto Frontier: Accuracy vs Speedup', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pareto_accuracy_vs_speedup.png', dpi=300)
        print(f"  Saved: {output_dir}/pareto_accuracy_vs_speedup.png")
        plt.close()
        
        # Plot 3: Principal Angle vs Accuracy (VQD only)
        plt.figure(figsize=(10, 6))
        plt.scatter(vqd_angle, vqd_acc, s=100, c=list(vqd_k), cmap='viridis', alpha=0.7)
        for i, k in enumerate(vqd_k):
            plt.annotate(f'k={k}', (vqd_angle[i], vqd_acc[i]), fontsize=9)
        plt.colorbar(label='k')
        plt.xlabel('Max Principal Angle (°)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('VQD: Principal Angle vs Classification Accuracy', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/vqd_angle_vs_accuracy.png', dpi=300)
        print(f"  Saved: {output_dir}/vqd_angle_vs_accuracy.png")
        plt.close()
        
        print(f"\n✅ Generated 3 Pareto plots in {output_dir}/")
        
        return self
    
    def print_summary(self):
        """Print final summary table."""
        print(f"\n{'='*80}")
        print(f"SUMMARY TABLE")
        print(f"{'='*80}\n")
        
        print(f"{'Method':<20} {'k':<5} {'Accuracy':<20} {'Speedup':<10} {'Time (ms)':<12} {'Max Angle':<12}")
        print(f"{'-'*20} {'-'*5} {'-'*20} {'-'*10} {'-'*12} {'-'*12}")
        
        # Baseline
        r = self.results['baseline']
        acc_str = f"{r['accuracy']*100:.1f}%"
        speedup_str = f"{r['speedup']:.2f}×"
        time_str = f"{r['time_per_query']*1000:.2f}"
        print(f"{'Baseline (60D)':<20} {r['k']:<5} {acc_str:<20} {speedup_str:<10} {time_str:<12} {'-':<12}")
        
        # PCA and VQD for each k
        for k in self.k_values:
            pca_key = f'pca_k{k}'
            vqd_key = f'vqd_k{k}'
            
            if pca_key in self.results:
                r = self.results[pca_key]
                method_str = f'PCA {k}D'
                acc_str = f"{r['accuracy']*100:.1f}%"
                speedup_str = f"{r['speedup']:.2f}×"
                time_str = f"{r['time_per_query']*1000:.2f}"
                print(f"{method_str:<20} {r['k']:<5} {acc_str:<20} {speedup_str:<10} {time_str:<12} {'-':<12}")
            
            if vqd_key in self.results:
                r = self.results[vqd_key]
                method_str = f'VQD {k}D'
                acc_str = f"{r['accuracy']*100:.1f}%"
                speedup_str = f"{r['speedup']:.2f}×"
                time_str = f"{r['time_per_query']*1000:.2f}"
                angle = r.get('max_principal_angle', 0)
                angle_str = f"{angle:.1f}°"
                print(f"{method_str:<20} {r['k']:<5} {acc_str:<20} {speedup_str:<10} {time_str:<12} {angle_str:<12}")
        
        print(f"\n{'='*80}\n")
        
        return self


def main():
    """Run VQD first-class pipeline."""
    
    print("\n" + "="*80)
    print("VQD FIRST-CLASS SUBSPACE PIPELINE")
    print("="*80)
    
    # Initialize pipeline
    pipeline = VQDPipeline(
        k_values=[4, 6, 8],  # Start with smaller sweep for speed
        use_procrustes=True,
        n_train=100,
        n_test=30,
        random_seed=42
    )
    
    # Run pipeline
    data_path = Path("data/frame_bank_std.npy")
    
    (pipeline
     .load_and_split_data(data_path)
     .preprocess()
     .build_frame_bank(pre_k=8)
     .evaluate_baseline()
     .evaluate_k_sweep()
     .save_results('results/vqd_pipeline_results.json')
     .generate_pareto_plots('figures')
     .print_summary())
    
    print("\n✅ Pipeline complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
