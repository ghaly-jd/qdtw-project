#!/usr/bin/env python
"""
Quick demo of DTW on subspace-projected sequences.

This is a smaller demo that evaluates DTW on a subset of the data
to demonstrate the pipeline works. For full evaluation, use
run_dtw_subspace.py.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dtw.dtw_runner import one_nn  # noqa: E402


def load_subset(method: str, k: int, n_train: int = 30, n_test: int = 10):
    """Load a small subset of sequences for quick demo."""
    base_dir = Path(__file__).parent.parent  # Project root
    train_path = base_dir / 'results' / 'subspace' / method / f'k{k}' / 'train'
    test_path = base_dir / 'results' / 'subspace' / method / f'k{k}' / 'test'

    # Load train subset
    train_files = sorted(train_path.glob('seq_*.npy'))[:n_train]
    train_seqs = [np.load(f) for f in train_files]
    train_labels = [i % 20 for i in range(len(train_seqs))]

    # Load test subset
    test_files = sorted(test_path.glob('seq_*.npy'))[:n_test]
    test_seqs = [np.load(f) for f in test_files]
    test_labels = [i % 20 for i in range(len(test_seqs))]

    return train_seqs, train_labels, test_seqs, test_labels


def main():
    """Run quick demo."""
    print("=" * 70)
    print("QUICK DTW DEMO")
    print("=" * 70)

    methods = ['Uq', 'Uc']
    k_values = [5, 8, 10]
    metrics = ['cosine', 'euclidean', 'fidelity']

    for method in methods:
        print(f"\n{method}:")

        for k in k_values:
            print(f"  k={k}:")

            # Load subset
            train_seqs, train_labels, test_seqs, test_labels = load_subset(
                method, k, n_train=30, n_test=10
            )

            for metric in metrics:
                # Evaluate
                start_time = time.time()
                n_correct = 0

                for test_seq, true_label in zip(test_seqs, test_labels):
                    pred_label, _ = one_nn(
                        train_seqs, train_labels, test_seq, metric=metric
                    )
                    if pred_label == true_label:
                        n_correct += 1

                elapsed = time.time() - start_time
                accuracy = n_correct / len(test_seqs)
                avg_time = (elapsed / len(test_seqs)) * 1000

                print(
                    f"    {metric:10s}: "
                    f"acc={accuracy:.2f} ({n_correct}/{len(test_seqs)}), "
                    f"avg_time={avg_time:6.1f}ms"
                )

    print("\n" + "=" * 70)
    print("Demo complete! For full evaluation, run:")
    print("  python scripts/run_dtw_subspace.py --method both --k 5 8 10")
    print("=" * 70)


if __name__ == '__main__':
    main()
