#!/usr/bin/env python
"""
Run DTW on k-D subspace-projected sequences.

This script evaluates DTW-based 1-NN classification on projected sequences
using different PCA methods (Uc/Uq), dimensions (k), and distance metrics.

Usage:
    python scripts/run_dtw_subspace.py --method Uq --k 5 8 10 --metric cosine
    python scripts/run_dtw_subspace.py --method Uc --k 8 --metric euclidean fidelity
    python scripts/run_dtw_subspace.py --method both --k 5 --metric cosine --window 10
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from dtw.dtw_runner import dtw_distance, one_nn  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_projected_sequences(
    method: str,
    k: int,
    split: str = 'train'
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load projected sequences and extract labels from directory structure.

    Args:
        method: 'Uq' or 'Uc'
        k: Number of principal components
        split: 'train' or 'test'

    Returns:
        sequences: List of sequences
        labels: List of labels (extracted from seq indices for demo)
    """
    base_path = Path(f'results/subspace/{method}/k{k}/{split}')

    if not base_path.exists():
        raise FileNotFoundError(
            f"Projected sequences not found: {base_path}\n"
            f"Run projection script first."
        )

    # Load all sequence files
    seq_files = sorted(base_path.glob('seq_*.npy'))

    if not seq_files:
        raise FileNotFoundError(f"No sequence files found in {base_path}")

    sequences = []
    labels = []

    for i, filepath in enumerate(seq_files):
        seq = np.load(filepath)
        sequences.append(seq)

        # For demo: create synthetic labels based on sequence index
        # In real scenario, these would come from dataset metadata
        # We'll create 20 classes with ~22-28 samples each
        label = i % 20
        labels.append(label)

    logger.info(
        f"Loaded {len(sequences)} {split} sequences from {method}/k{k}"
    )
    logger.info(f"Label distribution: {len(set(labels))} unique classes")

    return sequences, labels


def evaluate_1nn(
    train_seqs: List[np.ndarray],
    train_labels: List[int],
    test_seqs: List[np.ndarray],
    test_labels: List[int],
    metric: str,
    window: Optional[int] = None
) -> Tuple[float, float]:
    """
    Evaluate 1-NN classification accuracy and timing.

    Args:
        train_seqs: Training sequences
        train_labels: Training labels
        test_seqs: Test sequences
        test_labels: Test labels
        metric: Distance metric
        window: Optional DTW window

    Returns:
        accuracy: Classification accuracy in [0, 1]
        avg_time_ms: Average query time in milliseconds
    """
    n_correct = 0
    total_time = 0.0

    for test_seq, true_label in zip(test_seqs, test_labels):
        start_time = time.time()

        pred_label, _ = one_nn(
            train_seqs,
            train_labels,
            test_seq,
            metric=metric,
            window=window
        )

        elapsed = time.time() - start_time
        total_time += elapsed

        if pred_label == true_label:
            n_correct += 1

    accuracy = n_correct / len(test_seqs)
    avg_time_ms = (total_time / len(test_seqs)) * 1000

    return accuracy, avg_time_ms


def run_experiments(
    methods: List[str],
    k_values: List[int],
    metrics: List[str],
    window: Optional[int]
) -> Dict[str, List[dict]]:
    """
    Run DTW experiments for all combinations of parameters.

    Args:
        methods: List of methods ('Uq', 'Uc')
        k_values: List of k values
        metrics: List of distance metrics
        window: Optional DTW window

    Returns:
        results: Dictionary mapping method to list of result dicts
    """
    results = {}

    for method in methods:
        logger.info("=" * 70)
        logger.info(f"EVALUATING METHOD: {method}")
        logger.info("=" * 70)

        method_results = []

        for k in k_values:
            logger.info(f"\n--- k = {k} ---")

            try:
                # Load data
                train_seqs, train_labels = load_projected_sequences(
                    method, k, 'train'
                )
                test_seqs, test_labels = load_projected_sequences(
                    method, k, 'test'
                )

                logger.info(
                    f"Train: {len(train_seqs)} sequences, "
                    f"Test: {len(test_seqs)} sequences"
                )

                for metric in metrics:
                    logger.info(f"\nMetric: {metric}")

                    # Evaluate
                    accuracy, avg_time_ms = evaluate_1nn(
                        train_seqs,
                        train_labels,
                        test_seqs,
                        test_labels,
                        metric,
                        window
                    )

                    logger.info(f"  Accuracy: {accuracy:.4f}")
                    logger.info(f"  Avg time per query: {avg_time_ms:.2f} ms")

                    # Store results
                    method_results.append({
                        'k': k,
                        'metric': metric,
                        'accuracy': accuracy,
                        'time_ms': avg_time_ms
                    })

            except FileNotFoundError as e:
                logger.error(f"Skipping {method}/k{k}: {e}")
                continue

        results[method] = method_results

    return results


def save_results_csv(results: Dict[str, List[dict]], output_dir: Path):
    """
    Save results to CSV files.

    Args:
        results: Dictionary mapping method to list of result dicts
        output_dir: Directory to save CSV files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for method, method_results in results.items():
        if not method_results:
            logger.warning(f"No results for {method}, skipping CSV")
            continue

        csv_path = output_dir / f"metrics_subspace_{method}.csv"

        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['k', 'metric', 'accuracy', 'time_ms']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for row in method_results:
                writer.writerow(row)

        logger.info(f"Saved {method} results to: {csv_path}")


def print_summary(results: Dict[str, List[dict]]):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for method, method_results in results.items():
        if not method_results:
            continue

        print(f"\n{method}:")

        # Group by k and metric
        for k in sorted(set(r['k'] for r in method_results)):
            k_results = [r for r in method_results if r['k'] == k]

            print(f"  k={k}:")
            for metric in sorted(set(r['metric'] for r in k_results)):
                metric_results = [
                    r for r in k_results if r['metric'] == metric
                ]
                if metric_results:
                    r = metric_results[0]
                    print(
                        f"    {metric:12s}: "
                        f"accuracy={r['accuracy']:.4f}, "
                        f"avg_time={r['time_ms']:6.2f} ms"
                    )

    print("=" * 70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run DTW on k-D subspace-projected sequences'
    )
    parser.add_argument(
        '--method',
        type=str,
        nargs='+',
        default=['both'],
        help='PCA method(s): Uq, Uc, or both (default: both)'
    )
    parser.add_argument(
        '--k',
        type=int,
        nargs='+',
        default=[5, 8, 10],
        help='Number of principal components (default: 5 8 10)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        nargs='+',
        default=['cosine'],
        choices=['cosine', 'euclidean', 'fidelity'],
        help='Distance metric(s) (default: cosine)'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=None,
        help='DTW window size (default: None = no constraint)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for CSV files (default: results)'
    )

    args = parser.parse_args()

    # Parse methods
    if 'both' in args.method:
        methods = ['Uq', 'Uc']
    else:
        methods = args.method

    logger.info("=" * 70)
    logger.info("DTW SUBSPACE EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Methods: {methods}")
    logger.info(f"k values: {args.k}")
    logger.info(f"Metrics: {args.metric}")
    logger.info(f"Window: {args.window}")
    logger.info(f"Output directory: {args.output_dir}")

    # Run experiments
    results = run_experiments(methods, args.k, args.metric, args.window)

    # Save results
    output_dir = Path(args.output_dir)
    save_results_csv(results, output_dir)

    # Print summary
    print_summary(results)

    logger.info("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
