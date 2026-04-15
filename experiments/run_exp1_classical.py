"""Experiment 1: Classical Baseline Sweep"""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from experiments.base import (
    ExperimentBase,
    DTWClassifier,
    load_msr_data
)
from sklearn.decomposition import PCA


def run_experiment_1(
    n_train: int = 100,
    n_test: int = 30,
    k_values: list = [60, 8, 6, 4, 2]
):
    """
    Experiment 1: Classical Baseline Sweep
    
    Test all dimensionalities with classical PCA and Euclidean distance
    to establish performance baseline.
    """
    print("=" * 80)
    print("EXPERIMENT 1: CLASSICAL BASELINE SWEEP")
    print("=" * 80)
    
    # Initialize experiment
    exp = ExperimentBase(
        experiment_id="exp1_classical_baseline",
        results_dir="results"
    )
    
    # Load data
    print("\n📊 Loading MSR Action3D data...")
    X, y = load_msr_data()
    
    # Get fixed split
    print(f"\n🔀 Creating/Loading fixed split (n_train={n_train}, n_test={n_test})...")
    train_idx, test_idx = exp.get_or_create_split(
        n_samples=len(X),
        n_train=n_train,
        n_test=n_test
    )
    
    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    # Fit preprocessing pipeline (train-only)
    print("\n🔧 Fitting preprocessing pipeline (train-only)...")
    X_train_fb, X_test_fb, transform_info = exp.fit_transform_pipeline(
        X_train_raw,
        X_test_raw,
        scaler=True,
        frame_bank_dim=8  # Reduce to 8D frame bank
    )
    
    print(f"  Original: {X_train_raw.shape[1]}D")
    print(f"  Frame bank: {X_train_fb.shape[1]}D ({transform_info['frame_bank']['variance_explained']:.3f} variance)")
    
    # Store data info in manifest
    exp.manifest['data'] = {
        'dataset': 'MSR-Action3D',
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_train': n_train,
        'n_test': n_test,
        'split_file': str(exp.split_file),
        'split_seed': exp.seed
    }
    
    exp.manifest['preprocessing'] = {
        'scaler': 'StandardScaler',
        'frame_bank_dim': 8,
        'frame_bank_variance': transform_info['frame_bank']['variance_explained']
    }
    
    # Run experiments for each k
    results_by_k = {}
    
    for k in k_values:
        print(f"\n{'=' * 80}")
        print(f"Testing k = {k}")
        print('=' * 80)
        
        # Project to k dimensions (train-only fit)
        if k == 60:
            # Baseline: no dimensionality reduction
            X_train_k = X_train_raw
            X_test_k = X_test_raw
            X_train_k, X_test_k, _ = exp.fit_transform_pipeline(
                X_train_k, X_test_k,
                scaler=True,
                frame_bank_dim=None  # No frame bank
            )
            variance_explained = 1.0
        elif k == 8:
            # Frame bank only
            X_train_k = X_train_fb
            X_test_k = X_test_fb
            variance_explained = transform_info['frame_bank']['variance_explained']
        else:
            # Further reduce with PCA
            print(f"  🔧 Fitting PCA (8D → {k}D) on train...")
            pca = PCA(n_components=k)
            X_train_k = pca.fit_transform(X_train_fb)
            X_test_k = pca.transform(X_test_fb)
            variance_explained = float(pca.explained_variance_ratio_.sum())
            print(f"    Variance explained: {variance_explained:.3f}")
        
        # DTW Classification with timing
        print(f"  🔍 Running DTW 1-NN classification...")
        
        @exp.time_component(f"projection_k{k}")
        def project():
            return X_test_k  # Already projected
        
        @exp.time_component(f"dtw_k{k}")
        def classify():
            clf = DTWClassifier()
            clf.fit(X_train_k, y_train)
            return clf.score(X_test_k, y_test)
        
        project()
        accuracy = classify()
        
        # Bootstrap CI for accuracy
        print(f"  📊 Computing bootstrap 95% CI...")
        
        def bootstrap_accuracy():
            # Resample test set
            idx = np.random.choice(len(X_test_k), size=len(X_test_k), replace=True)
            X_bootstrap = X_test_k[idx]
            y_bootstrap = y_test[idx]
            
            clf = DTWClassifier()
            clf.fit(X_train_k, y_train)
            return clf.score(X_bootstrap, y_bootstrap)
        
        accuracy_ci = exp.bootstrap_ci(bootstrap_accuracy, n_bootstrap=1000)
        
        # Get timing summary for this k
        timing = exp.get_timing_summary()
        
        # Store results
        results_by_k[f"k{k}"] = {
            'k': k,
            'variance_explained': variance_explained,
            'accuracy': accuracy_ci,
            'runtime_per_query_ms': {
                'projection': timing.get(f'projection_k{k}', 0.0) / n_test,
                'dtw': timing.get(f'dtw_k{k}', 0.0) / n_test,
                'quantum': 0.0,
                'total': (timing.get(f'projection_k{k}', 0.0) + timing.get(f'dtw_k{k}', 0.0)) / n_test
            }
        }
        
        print(f"\n  ✅ Results for k={k}:")
        print(f"     Accuracy: {accuracy_ci['mean']:.3f} ± [{accuracy_ci['ci_lower']:.3f}, {accuracy_ci['ci_upper']:.3f}]")
        print(f"     Variance: {variance_explained:.3f}")
        print(f"     Time/query: {results_by_k[f'k{k}']['runtime_per_query_ms']['total']:.2f} ms")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print('=' * 80)
    print(f"\n{'k':<5} {'Var':<8} {'Accuracy':<20} {'Time (ms)':<10}")
    print('-' * 50)
    for k in k_values:
        res = results_by_k[f"k{k}"]
        acc = res['accuracy']
        print(f"{k:<5} {res['variance_explained']:<8.3f} "
              f"{acc['mean']:.3f} [{acc['ci_lower']:.3f}, {acc['ci_upper']:.3f}]  "
              f"{res['runtime_per_query_ms']['total']:<10.2f}")
    
    # Save results
    exp.manifest['hyperparameters'] = {
        'method': 'pca',
        'k_values': k_values,
        'distance': 'euclidean'
    }
    
    exp.manifest['results'] = results_by_k
    
    exp.write_manifest()
    exp.save_results(results_by_k)
    
    print(f"\n✅ Experiment 1 complete! Results saved to {exp.exp_dir}")
    
    return results_by_k


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment 1: Classical Baseline")
    parser.add_argument("--n-train", type=int, default=100, help="Number of training samples")
    parser.add_argument("--n-test", type=int, default=30, help="Number of test samples")
    parser.add_argument("--k-values", type=int, nargs="+", default=[60, 8, 6, 4, 2],
                        help="Dimensionalities to test")
    
    args = parser.parse_args()
    
    results = run_experiment_1(
        n_train=args.n_train,
        n_test=args.n_test,
        k_values=args.k_values
    )
