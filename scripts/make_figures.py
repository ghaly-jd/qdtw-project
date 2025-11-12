#!/usr/bin/env python
"""
Generate all evaluation figures and summary report.

This script loads metrics from CSV files, creates visualizations,
and generates a summary README with best results.

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --results-dir results --output-dir figures
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.aggregate import (  # noqa: E402
    get_best_k_per_method,
    load_and_merge_metrics,
    summarize_metrics,
)
from eval.plotting import create_all_plots  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_figures_readme(
    metrics: dict,
    output_path: Path = Path('figures/README.md')
) -> None:
    """
    Generate README.md with summary and best results table.

    Args:
        metrics: Dictionary from load_and_merge_metrics()
        output_path: Path to save README
    """
    lines = ["# QDTW Evaluation Results\n"]

    # Summary paragraph
    lines.append("## Summary\n")
    lines.append(
        "This directory contains visualizations comparing the performance of "
        "Quantum PCA (Uq) and Classical PCA (Uc) for action recognition using "
        "Dynamic Time Warping (DTW). The analysis evaluates both accuracy and "
        "computational efficiency across different dimensionality reductions "
        "(k=5, 8, 10) and distance metrics (cosine, euclidean, fidelity). "
        "Results demonstrate the trade-offs between dimensionality reduction, "
        "classification accuracy, and query time. Lower k values significantly "
        "reduce computational cost while maintaining competitive accuracy, "
        "particularly for the euclidean metric. The Pareto frontier plot "
        "reveals optimal configurations balancing accuracy and speed.\n"
    )

    # Figures section
    lines.append("## Figures\n")
    lines.append("- `accuracy_vs_k.png`: Classification accuracy vs number of principal components\n")
    lines.append("- `time_vs_k.png`: Average query time vs number of principal components\n")
    lines.append("- `pareto_accuracy_time.png`: Pareto frontier showing accuracy-time trade-offs\n")

    # Best results table
    lines.append("\n## Best Results by Method and Metric\n")

    if 'all_subspace' in metrics and not metrics['all_subspace'].empty:
        subspace = metrics['all_subspace']

        # Create table for each metric
        for metric_name in ['euclidean', 'cosine', 'fidelity']:
            lines.append(f"\n### {metric_name.capitalize()} Distance\n")

            best_k = get_best_k_per_method(subspace, metric_name)

            if not best_k.empty:
                lines.append("| Method | Best k | Accuracy | Avg Time (ms) |")
                lines.append("|--------|--------|----------|---------------|")

                for _, row in best_k.iterrows():
                    lines.append(
                        f"| {row['method']:6s} | "
                        f"{int(row['k']):6d} | "
                        f"{row['accuracy']:8.4f} | "
                        f"{row['time_ms']:13.1f} |"
                    )
            else:
                lines.append(f"*No results found for {metric_name} metric*\n")

    # Add baseline comparison if available
    if 'baseline' in metrics and not metrics['baseline'].empty:
        lines.append("\n### Baseline (60D Full Space)\n")
        lines.append("| Metric | Accuracy | Avg Time (ms) |")
        lines.append("|--------|----------|---------------|")

        baseline = metrics['baseline']
        for _, row in baseline.iterrows():
            lines.append(
                f"| {row['metric']:10s} | "
                f"{row['accuracy']:8.4f} | "
                f"{row['time_ms']:13.1f} |"
            )

    # Key findings
    lines.append("\n## Key Findings\n")

    if 'all_subspace' in metrics and not metrics['all_subspace'].empty:
        subspace = metrics['all_subspace']

        # Find overall best configuration
        best_overall = subspace.loc[subspace['accuracy'].idxmax()]
        fastest = subspace.loc[subspace['time_ms'].idxmin()]

        lines.append(f"- **Highest Accuracy**: {best_overall['method']} with k={int(best_overall['k'])}, "
                     f"{best_overall['metric']} metric achieved {best_overall['accuracy']:.4f} accuracy\n")
        lines.append(f"- **Fastest Query**: {fastest['method']} with k={int(fastest['k'])}, "
                     f"{fastest['metric']} metric at {fastest['time_ms']:.1f}ms per query\n")

        # Dimensionality reduction benefit
        if 'baseline' in metrics and not metrics['baseline'].empty:
            baseline = metrics['baseline']
            euclidean_baseline = baseline[baseline['metric'] == 'euclidean']

            if not euclidean_baseline.empty:
                baseline_time = euclidean_baseline.iloc[0]['time_ms']
                best_subspace_time = subspace[subspace['metric'] == 'euclidean']['time_ms'].min()
                speedup = baseline_time / best_subspace_time

                lines.append(f"- **Speedup**: Up to {speedup:.1f}x faster than 60D baseline "
                             f"with minimal accuracy loss\n")

        # Method comparison
        uq_data = subspace[subspace['method'] == 'Uq']
        uc_data = subspace[subspace['method'] == 'Uc']

        if not uq_data.empty and not uc_data.empty:
            uq_mean_acc = uq_data['accuracy'].mean()
            uc_mean_acc = uc_data['accuracy'].mean()

            if uc_mean_acc > uq_mean_acc:
                diff = (uc_mean_acc - uq_mean_acc) * 100
                lines.append(f"- **Method Comparison**: Classical PCA (Uc) achieves {diff:.2f}% "
                             f"higher average accuracy than Quantum PCA (Uq)\n")
            else:
                diff = (uq_mean_acc - uc_mean_acc) * 100
                lines.append(f"- **Method Comparison**: Quantum PCA (Uq) achieves {diff:.2f}% "
                             f"higher average accuracy than Classical PCA (Uc)\n")

    # Generated timestamp
    from datetime import datetime
    lines.append(f"\n---\n*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f"Generated figures README: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate evaluation figures and summary'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory containing metrics CSV files (default: results)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures',
        help='Directory to save figures (default: figures)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Figure resolution (default: 300)'
    )
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create sample metrics if CSV files do not exist'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    logger.info("=" * 70)
    logger.info("GENERATING QDTW EVALUATION FIGURES")
    logger.info("=" * 70)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Check if metrics exist, create samples if needed
    required_files = [
        results_dir / 'metrics_baseline.csv',
        results_dir / 'metrics_subspace_Uq.csv',
        results_dir / 'metrics_subspace_Uc.csv'
    ]

    missing_files = [f for f in required_files if not f.exists()]

    if missing_files and args.create_sample:
        logger.warning(f"Missing {len(missing_files)} metric files, creating samples...")
        from eval.aggregate import create_sample_metrics
        create_sample_metrics(results_dir)
    elif missing_files:
        logger.warning(f"Missing {len(missing_files)} metric files:")
        for f in missing_files:
            logger.warning(f"  - {f}")
        logger.warning("Use --create-sample to generate sample data for testing")

    # Load metrics
    logger.info("\nLoading metrics...")
    metrics = load_and_merge_metrics(results_dir)

    if not metrics:
        logger.error("No metrics loaded. Cannot generate figures.")
        return 1

    # Print summary
    print("\n" + summarize_metrics(metrics))

    # Create plots
    logger.info("\nGenerating plots...")
    create_all_plots(
        uq_df=metrics.get('Uq'),
        uc_df=metrics.get('Uc'),
        baseline_df=metrics.get('baseline'),
        output_dir=output_dir,
        dpi=args.dpi
    )

    # Generate README
    logger.info("\nGenerating README...")
    generate_figures_readme(metrics, output_dir / 'README.md')

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Figures saved to: {output_dir}")
    logger.info(f"  - accuracy_vs_k.png")
    logger.info(f"  - time_vs_k.png")
    logger.info(f"  - pareto_accuracy_time.png")
    logger.info(f"  - README.md")
    logger.info("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
