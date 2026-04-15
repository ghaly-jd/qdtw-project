"""
Enhanced Analysis Script for VQD Quantum PCA Results

Analyzes the enhanced validation metrics including:
- k=d validation (diagonalization, eigenvalue correlation, reconstruction error)
- Mixing reduction (principal angles, chordal distance)
- Span optimization (Procrustes improvement)
- Comparison with original results
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any
import sys

sys.path.append(str(Path(__file__).parent.parent))


def load_results(results_dir: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    results_path = Path(results_dir) / "results_summary.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def analyze_kd_validation(results: Dict[str, Any]) -> None:
    """
    Analyze k=d validation metrics.
    
    Checks:
    1. Diagonalization error: ||D - diag(D)||_F (should be ~0)
    2. Eigenvalue correlation with PCA (should be ~1)
    3. Reconstruction error match with PCA (should be ~0 difference)
    """
    print("\n" + "=" * 80)
    print("A. k=d VALIDATION ANALYSIS")
    print("=" * 80)
    
    kd_results = {}
    for key, res in results.items():
        # Check if k=d (k == frame_bank_size)
        if res['k'] == res['frame_bank_size']:
            if 'vqd' in res and 'kd_validation' in res['vqd']:
                kd_results[key] = res
    
    if not kd_results:
        print("⚠️  No k=d results found in the dataset")
        return
    
    print(f"\nFound {len(kd_results)} k=d configurations:")
    print(f"{'Config':<15} {'k':<5} {'Diag Error':<15} {'Eigen Corr':<15} {'Recon Error Diff':<20} {'Status':<10}")
    print("-" * 90)
    
    for key, res in kd_results.items():
        k = res['k']
        kd_val = res['vqd']['kd_validation']
        
        diag_err = kd_val.get('diagonalization_error', None)
        eigen_corr = kd_val.get('eigenvalue_correlation', None)
        recon_diff = kd_val.get('reconstruction_error_diff', None)
        
        # Determine status
        status = "✅ PASS"
        if diag_err is not None and diag_err > 1e-4:
            status = "⚠️  WARN"
        if diag_err is not None and diag_err > 1e-2:
            status = "❌ FAIL"
        if eigen_corr is not None and eigen_corr < 0.95:
            status = "⚠️  WARN"
        if recon_diff is not None and recon_diff > 1e-4:
            status = "⚠️  WARN"
        
        print(f"{key:<15} {k:<5} {diag_err if diag_err else 'N/A':<15.6e} "
              f"{eigen_corr if eigen_corr else 'N/A':<15.6f} "
              f"{recon_diff if recon_diff else 'N/A':<20.6e} {status:<10}")
    
    print("\nInterpretation:")
    print("  - Diagonalization error < 1e-6: Excellent")
    print("  - Diagonalization error < 1e-4: Good")
    print("  - Eigenvalue correlation > 0.99: Excellent ordering")
    print("  - Eigenvalue correlation > 0.95: Good ordering")
    print("  - Reconstruction error diff < 1e-4: Matches PCA")


def analyze_mixing_reduction(results: Dict[str, Any]) -> None:
    """
    Analyze mixing reduction for k < d.
    
    Metrics:
    1. Principal angles (mean and max) - should be small
    2. Chordal distance - should be small
    3. Orthogonality error - should be near zero
    """
    print("\n" + "=" * 80)
    print("B. MIXING REDUCTION ANALYSIS (k < d)")
    print("=" * 80)
    
    print(f"\n{'Config':<15} {'k/d':<8} {'Angles (mean)':<15} {'Angles (max)':<15} "
          f"{'Chordal Dist':<15} {'Orth Error':<15} {'Status':<10}")
    print("-" * 105)
    
    for key in sorted(results.keys()):
        res = results[key]
        k = res['k']
        d = res['frame_bank_size']
        
        if 'vqd' not in res or 'error' in res['vqd']:
            continue
        
        vqd = res['vqd']
        
        angles_mean = vqd['angles_before_procrustes']['mean']
        angles_max = vqd['angles_before_procrustes']['max']
        chordal = vqd.get('chordal_distance', None)
        orth_err = vqd['orthogonality_error']
        
        # Determine status
        status = "✅ GOOD"
        if angles_mean > 30.0:
            status = "⚠️  HIGH"
        if angles_max > 60.0:
            status = "⚠️  HIGH"
        if chordal is not None and chordal > 0.3:
            status = "⚠️  HIGH"
        
        print(f"{key:<15} {k}/{d:<6} {angles_mean:<15.2f} {angles_max:<15.2f} "
              f"{chordal if chordal else 'N/A':<15.6f} {orth_err:<15.6e} {status:<10}")
    
    print("\nInterpretation:")
    print("  - Mean angle < 15°: Excellent alignment")
    print("  - Mean angle < 30°: Good alignment")
    print("  - Max angle < 30°: Excellent alignment")
    print("  - Max angle < 60°: Acceptable alignment")
    print("  - Chordal distance < 0.1: Very close subspaces")
    print("  - Chordal distance < 0.3: Similar subspaces")
    print("  - Orthogonality error < 1e-6: Excellent orthogonality")


def analyze_procrustes_improvement(results: Dict[str, Any]) -> None:
    """
    Analyze Procrustes alignment improvement.
    
    Metrics:
    1. Residual drop percentage
    2. Angles before vs after Procrustes
    3. Accuracy improvement with Procrustes
    """
    print("\n" + "=" * 80)
    print("C. PROCRUSTES ALIGNMENT IMPROVEMENT")
    print("=" * 80)
    
    print(f"\n{'Config':<15} {'k':<5} {'Res Drop %':<15} {'Angles Before':<15} "
          f"{'Angles After':<15} {'Acc PCA':<12} {'Acc VQD':<12} {'Acc VQD+P':<12} {'Δ VQD+P':<10}")
    print("-" * 120)
    
    for key in sorted(results.keys()):
        res = results[key]
        k = res['k']
        
        if 'vqd' not in res or 'error' in res['vqd']:
            continue
        
        if 'vqd_procrustes' not in res:
            continue
        
        res_drop = res['vqd_procrustes']['residual_drop_pct']
        
        angles_before = res['vqd']['angles_before_procrustes']['mean']
        angles_after = res['vqd_procrustes']['angles_after_procrustes']['mean']
        
        acc_pca = res['pca']['accuracy']['mean']
        acc_vqd = res['vqd']['accuracy']['mean']
        acc_vqd_proc = res['vqd_procrustes']['accuracy']['mean']
        
        delta_proc = (acc_vqd_proc - acc_pca) * 100
        
        print(f"{key:<15} {k:<5} {res_drop:<15.1f} {angles_before:<15.2f} "
              f"{angles_after:<15.2f} {acc_pca:<12.3f} {acc_vqd:<12.3f} "
              f"{acc_vqd_proc:<12.3f} {delta_proc:+10.1f}")
    
    print("\nInterpretation:")
    print("  - Residual drop > 50%: Span is similar, just rotated")
    print("  - Residual drop < 20%: Subspaces differ in content")
    print("  - Angles after < Angles before: Alignment helping")
    print("  - Δ VQD+P positive: Procrustes improves accuracy")


def analyze_k_sweep(results: Dict[str, Any]) -> None:
    """
    Analyze trends across k values.
    """
    print("\n" + "=" * 80)
    print("D. K-SWEEP ANALYSIS")
    print("=" * 80)
    
    # Group by frame bank size
    fb_groups = {}
    for key, res in results.items():
        fb = res['frame_bank_size']
        if fb not in fb_groups:
            fb_groups[fb] = []
        fb_groups[fb].append(res)
    
    for fb, group in sorted(fb_groups.items()):
        print(f"\nFrame Bank Size: {fb}D")
        print(f"{'k':<5} {'PCA Acc':<12} {'VQD Acc':<12} {'VQD+P Acc':<12} "
              f"{'Angles':<15} {'Speedup PCA':<15} {'Speedup VQD':<15}")
        print("-" * 95)
        
        for res in sorted(group, key=lambda x: x['k']):
            k = res['k']
            
            if 'vqd' not in res or 'error' in res['vqd']:
                continue
            
            acc_pca = res['pca']['accuracy']['mean']
            acc_vqd = res['vqd']['accuracy']['mean']
            acc_vqd_proc = res['vqd_procrustes']['accuracy']['mean'] if 'vqd_procrustes' in res else None
            
            angles = res['vqd']['angles_before_procrustes']['mean']
            
            speedup_pca = res['speedup_vs_60D']['pca']
            speedup_vqd = res['speedup_vs_60D']['vqd']
            
            print(f"{k:<5} {acc_pca:<12.3f} {acc_vqd:<12.3f} "
                  f"{acc_vqd_proc if acc_vqd_proc else 'N/A':<12} "
                  f"{angles:<15.2f} {speedup_pca:<15.2f} {speedup_vqd:<15.2f}")


def compare_with_original(results_new: Dict[str, Any], results_old_path: str) -> None:
    """
    Compare enhanced results with original results (if available).
    """
    print("\n" + "=" * 80)
    print("E. COMPARISON WITH ORIGINAL RESULTS")
    print("=" * 80)
    
    results_old_file = Path(results_old_path)
    if not results_old_file.exists():
        print(f"⚠️  Original results not found: {results_old_path}")
        print("   Skipping comparison")
        return
    
    with open(results_old_file, 'r') as f:
        results_old = json.load(f)
    
    print(f"\n{'Config':<15} {'k':<5} {'Angles Old':<15} {'Angles New':<15} "
          f"{'Improvement':<15} {'Acc Old':<12} {'Acc New':<12} {'Δ Acc':<10}")
    print("-" * 105)
    
    for key in sorted(results_new.keys()):
        if key not in results_old:
            continue
        
        res_new = results_new[key]
        res_old = results_old[key]
        
        if 'vqd' not in res_new or 'error' in res_new['vqd']:
            continue
        if 'vqd' not in res_old or 'error' in res_old['vqd']:
            continue
        
        k = res_new['k']
        
        angles_old = res_old['vqd']['angles_before_procrustes']['mean']
        angles_new = res_new['vqd']['angles_before_procrustes']['mean']
        improvement = angles_old - angles_new
        
        acc_old = res_old['vqd_procrustes']['accuracy']['mean'] if 'vqd_procrustes' in res_old else res_old['vqd']['accuracy']['mean']
        acc_new = res_new['vqd_procrustes']['accuracy']['mean'] if 'vqd_procrustes' in res_new else res_new['vqd']['accuracy']['mean']
        delta_acc = (acc_new - acc_old) * 100
        
        print(f"{key:<15} {k:<5} {angles_old:<15.2f} {angles_new:<15.2f} "
              f"{improvement:+15.2f} {acc_old:<12.3f} {acc_new:<12.3f} {delta_acc:+10.1f}")
    
    print("\nInterpretation:")
    print("  - Positive improvement: Angles reduced (better alignment)")
    print("  - Positive Δ Acc: Accuracy improved")


def generate_summary_report(results: Dict[str, Any], output_file: str = "analysis_summary.txt") -> None:
    """
    Generate a comprehensive summary report.
    """
    print(f"\n{'=' * 80}")
    print("GENERATING SUMMARY REPORT")
    print('=' * 80)
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ENHANCED VQD QUANTUM PCA - ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        total_configs = len(results)
        successful_configs = sum(1 for res in results.values() if 'vqd' in res and 'error' not in res['vqd'])
        kd_configs = sum(1 for res in results.values() if res['k'] == res['frame_bank_size'])
        
        f.write(f"Total configurations: {total_configs}\n")
        f.write(f"Successful VQD runs: {successful_configs}\n")
        f.write(f"k=d configurations: {kd_configs}\n\n")
        
        # Best results
        f.write("=" * 80 + "\n")
        f.write("BEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        best_acc_vqd = max(
            [(key, res['vqd']['accuracy']['mean']) for key, res in results.items() 
             if 'vqd' in res and 'error' not in res['vqd']],
            key=lambda x: x[1]
        )
        f.write(f"Best VQD accuracy: {best_acc_vqd[0]} = {best_acc_vqd[1]:.3f}\n")
        
        best_acc_proc = max(
            [(key, res['vqd_procrustes']['accuracy']['mean']) for key, res in results.items() 
             if 'vqd_procrustes' in res],
            key=lambda x: x[1]
        )
        f.write(f"Best VQD+Procrustes accuracy: {best_acc_proc[0]} = {best_acc_proc[1]:.3f}\n")
        
        best_angles = min(
            [(key, res['vqd']['angles_before_procrustes']['mean']) for key, res in results.items() 
             if 'vqd' in res and 'error' not in res['vqd']],
            key=lambda x: x[1]
        )
        f.write(f"Best principal angles: {best_angles[0]} = {best_angles[1]:.2f}°\n\n")
        
        # Recommendations
        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        # Check if improvements are needed
        avg_angles = np.mean([res['vqd']['angles_before_procrustes']['mean'] 
                              for res in results.values() 
                              if 'vqd' in res and 'error' not in res['vqd']])
        
        if avg_angles > 30:
            f.write("⚠️  HIGH AVERAGE ANGLES ({:.2f}°)\n".format(avg_angles))
            f.write("   Consider:\n")
            f.write("   - Increasing penalty_scale\n")
            f.write("   - Using in-loop orthonormalization\n")
            f.write("   - Enabling subspace loss (procrustes_alpha > 0)\n\n")
        else:
            f.write("✅ GOOD AVERAGE ANGLES ({:.2f}°)\n".format(avg_angles))
            f.write("   Current settings are working well\n\n")
        
        # Check k=d validation
        if kd_configs > 0:
            kd_results = [res for res in results.values() 
                         if res['k'] == res['frame_bank_size'] and 'vqd' in res and 'kd_validation' in res['vqd']]
            if kd_results:
                avg_diag_err = np.mean([res['vqd']['kd_validation']['diagonalization_error'] 
                                       for res in kd_results])
                if avg_diag_err < 1e-4:
                    f.write("✅ GOOD k=d DIAGONALIZATION (avg {:.2e})\n".format(avg_diag_err))
                else:
                    f.write("⚠️  POOR k=d DIAGONALIZATION (avg {:.2e})\n".format(avg_diag_err))
                    f.write("   Consider:\n")
                    f.write("   - Increasing maxiter\n")
                    f.write("   - Using stronger ramped penalties\n\n")
    
    print(f"✅ Summary report saved to: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze enhanced VQD PCA results")
    parser.add_argument("--results-dir", type=str, 
                       default="results/exp3_vqd_quantum_pca",
                       help="Path to results directory")
    parser.add_argument("--compare-with", type=str, default=None,
                       help="Path to original results JSON for comparison")
    parser.add_argument("--output", type=str, default="analysis_summary.txt",
                       help="Output file for summary report")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ENHANCED VQD QUANTUM PCA - RESULTS ANALYSIS")
    print("=" * 80)
    
    # Load results
    print(f"\nLoading results from: {args.results_dir}")
    results = load_results(args.results_dir)
    print(f"✅ Loaded {len(results)} configurations")
    
    # Run analyses
    analyze_kd_validation(results)
    analyze_mixing_reduction(results)
    analyze_procrustes_improvement(results)
    analyze_k_sweep(results)
    
    if args.compare_with:
        compare_with_original(results, args.compare_with)
    
    # Generate summary report
    generate_summary_report(results, args.output)
    
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print('=' * 80)


if __name__ == "__main__":
    main()
