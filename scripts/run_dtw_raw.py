#!/usr/bin/env python
"""
Run DTW classification on RAW 60-D skeleton data (no encoding, no PCA).

This script provides a baseline to compare against the encoded+PCA pipeline.
It should achieve 60-80% accuracy if the encoding is the problem.

Usage:
    python scripts/run_dtw_raw.py --n-train 454 --n-test 113
    python scripts/run_dtw_raw.py --n-train 100 --n-test 30 --quick
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.loader import load_all_sequences, flatten_sequence
from dtw.dtw_runner import one_nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simple_train_test_split(sequences, labels, test_fraction=0.2, seed=42):
    """Simple train/test split."""
    np.random.seed(seed)
    n = len(sequences)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    n_test = int(n * test_fraction)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    train_seqs = [sequences[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_seqs = [sequences[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return train_seqs, train_labels, test_seqs, test_labels


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run DTW classification on raw 60-D skeleton data'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='msr_action_data',
        help='Directory containing skeleton files'
    )
    parser.add_argument(
        '--n-train',
        type=int,
        default=None,
        help='Number of training samples (None = use all)'
    )
    parser.add_argument(
        '--n-test',
        type=int,
        default=None,
        help='Number of test samples (None = use all)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='euclidean',
        choices=['euclidean', 'cosine', 'fidelity'],
        help='Distance metric for DTW'
    )
    parser.add_argument(
        '--test-fraction',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (sets n-train=100, n-test=30)'
    )

    args = parser.parse_args()

    if args.quick:
        args.n_train = 100
        args.n_test = 30

    logger.info("=" * 70)
    logger.info("RAW DATA DTW CLASSIFICATION (60-D, NO ENCODING)")
    logger.info("=" * 70)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Distance metric: {args.metric}")
    logger.info(f"Test fraction: {args.test_fraction}")
    logger.info(f"Random seed: {args.seed}")

    # Load sequences
    logger.info("\nLoading MSR Action3D sequences...")
    sequences, labels = load_all_sequences(args.data_dir)
    logger.info(f"Loaded {len(sequences)} sequences")

    # Check that sequences are flattened (T, 60)
    for i, seq in enumerate(sequences[:5]):
        logger.debug(f"Sequence {i}: shape {seq.shape}, dtype {seq.dtype}")

    # Split into train/test
    logger.info("\nSplitting into train/test...")
    train_seqs, train_labels, test_seqs, test_labels = simple_train_test_split(
        sequences, labels, 
        test_fraction=args.test_fraction, 
        seed=args.seed
    )

    # Limit samples if requested
    if args.n_train is not None:
        train_seqs = train_seqs[:args.n_train]
        train_labels = train_labels[:args.n_train]

    if args.n_test is not None:
        test_seqs = test_seqs[:args.n_test]
        test_labels = test_labels[:args.n_test]

    logger.info(f"Training samples: {len(train_seqs)}")
    logger.info(f"Test samples: {len(test_seqs)}")

    # Check class distribution
    train_dist = Counter(train_labels)
    test_dist = Counter(test_labels)
    logger.info(f"\nTraining classes: {len(train_dist)} unique actions")
    logger.info(f"Test classes: {len(test_dist)} unique actions")
    logger.info(f"Train distribution (top 5): {train_dist.most_common(5)}")
    logger.info(f"Test distribution (top 5): {test_dist.most_common(5)}")

    # Sanity checks
    logger.info("\n" + "=" * 70)
    logger.info("SANITY CHECKS")
    logger.info("=" * 70)

    # Check data dimensions
    sample_seq = train_seqs[0]
    logger.info(f"Sample sequence shape: {sample_seq.shape}")
    logger.info(f"Expected: (T, 60) where T is number of frames")

    if sample_seq.shape[1] != 60:
        logger.error(f"❌ Wrong dimension! Expected 60, got {sample_seq.shape[1]}")
        return 1

    logger.info("✅ Sequences have correct dimension (60-D)")

    # Check data range (should be in mm, not normalized)
    sample_mean = np.mean(sample_seq)
    sample_norm = np.linalg.norm(sample_seq[0])
    logger.info(f"Sample frame mean: {sample_mean:.2f}")
    logger.info(f"Sample frame L2 norm: {sample_norm:.2f}")

    if sample_norm < 10:
        logger.warning("⚠️  Data appears normalized (norm < 10)")
        logger.warning("    Expected raw coordinates with norm ~3000")
    else:
        logger.info("✅ Data appears to be raw coordinates (not normalized)")

    # Check labels (0-indexed in load_all_sequences: 0-19)
    assert all(0 <= lbl <= 19 for lbl in train_labels), "Invalid train labels"
    assert all(0 <= lbl <= 19 for lbl in test_labels), "Invalid test labels"
    logger.info("✅ Labels are valid MSR Action3D IDs (0-19, zero-indexed)")

    logger.info("=" * 70)

    # Run classification
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING 1-NN DTW CLASSIFICATION")
    logger.info("=" * 70)
    logger.info(f"Classifying {len(test_seqs)} test samples...")
    logger.info(f"Using {len(train_seqs)} training samples")
    logger.info(f"Distance metric: {args.metric}")
    logger.info(f"Estimated comparisons: {len(test_seqs) * len(train_seqs)}")
    logger.info("")

    n_correct = 0
    total_time = 0.0
    predictions = []
    distances = []

    start_time = time.time()

    for i, (test_seq, true_label) in enumerate(zip(test_seqs, test_labels)):
        # Classify using 1-NN
        pred_label, min_dist = one_nn(
            train_seqs,
            train_labels,
            test_seq,
            metric=args.metric,
            window=None
        )

        predictions.append(pred_label)
        distances.append(min_dist)

        if pred_label == true_label:
            n_correct += 1

        # Progress logging
        if (i + 1) % 10 == 0 or (i + 1) == len(test_seqs):
            elapsed = time.time() - start_time
            progress = (i + 1) / len(test_seqs) * 100
            current_acc = n_correct / (i + 1) * 100
            logger.info(
                f"  Progress: {i+1}/{len(test_seqs)} ({progress:.1f}%) | "
                f"Accuracy: {current_acc:.2f}% | "
                f"Time: {elapsed:.1f}s"
            )

    total_time = time.time() - start_time

    # Calculate metrics
    accuracy = n_correct / len(test_seqs)
    avg_time_per_sample = total_time / len(test_seqs)

    # Results
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Accuracy: {accuracy * 100:.2f}% ({n_correct}/{len(test_seqs)})")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Avg time per sample: {avg_time_per_sample * 1000:.1f}ms")
    logger.info(f"Mean DTW distance: {np.mean(distances):.2f}")
    logger.info(f"Std DTW distance: {np.std(distances):.2f}")
    logger.info("")

    # Interpretation
    logger.info("=" * 70)
    logger.info("INTERPRETATION")
    logger.info("=" * 70)

    if accuracy >= 0.60:
        logger.info("✅ EXCELLENT: Raw data achieves good accuracy (>60%)")
        logger.info("   This confirms that encoding destroys class information.")
        logger.info("   Solution: Skip encoding OR redesign to preserve magnitude.")
    elif accuracy >= 0.40:
        logger.info("⚠️  MODERATE: Raw data achieves moderate accuracy (40-60%)")
        logger.info("   Encoding may not be the only issue.")
        logger.info("   Check: Data quality, train/test split, distance metric.")
    elif accuracy >= 0.20:
        logger.info("⚠️  POOR: Raw data achieves poor accuracy (20-40%)")
        logger.info("   Multiple issues likely present:")
        logger.info("   - Check data loading and preprocessing")
        logger.info("   - Verify label correctness")
        logger.info("   - Consider different distance metrics")
    else:
        logger.info("❌ RANDOM: Accuracy near random chance (<20% for 20 classes)")
        logger.info("   Critical issues with pipeline:")
        logger.info("   - Labels may be incorrect")
        logger.info("   - Data may be corrupted")
        logger.info("   - Train/test split may be broken")

    logger.info("")
    logger.info("Compare to encoded+PCA results:")
    logger.info("  - Encoded 8-D: ~5% accuracy (random)")
    logger.info("  - Raw 60-D: {:.2f}% accuracy".format(accuracy * 100))
    logger.info("")

    if accuracy > 0.05:
        improvement = (accuracy - 0.05) / 0.05 * 100
        logger.info(f"Improvement: {improvement:.0f}x better than encoded approach")

    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
