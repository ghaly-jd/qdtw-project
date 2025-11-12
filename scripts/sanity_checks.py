#!/usr/bin/env python
"""
Sanity checks for QDTW encoding pipeline.

This script runs comprehensive validation tests to ensure the encoding
approach preserves class-discriminative information.

Checks:
1. Same-clip identity: train == test for clips → expect 100% accuracy
2. Twin retrieval: time-warped copies → nearest neighbor must be twin
3. Class separability: intra vs inter distance ratio should be >> 1

Usage:
    python scripts/sanity_checks.py
    python scripts/sanity_checks.py --encoded-dir results/subspace/Uc/k8
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.loader import load_all_sequences
from features.amplitude_encoding import batch_encode_unit_vectors
from dtw.dtw_runner import dtw_distance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def add_temporal_jitter(seq: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Create a time-warped copy of a sequence by adding small temporal jitter.
    
    Args:
        seq: Input sequence of shape [T, D]
        seed: Random seed
        
    Returns:
        Jittered sequence with slightly different length
    """
    np.random.seed(seed)
    T, D = seq.shape
    
    # Random drop 10% of frames
    keep_rate = 0.9
    n_keep = int(T * keep_rate)
    indices = np.sort(np.random.choice(T, size=n_keep, replace=False))
    
    return seq[indices].copy()


def check_same_clip_identity(sequences, n_samples=20, metric='euclidean'):
    """
    Sanity Check 1: Same-clip identity test.
    
    If train == test (same exact sequence), we should get 100% accuracy.
    
    Args:
        sequences: List of sequences
        n_samples: Number of clips to test
        metric: DTW distance metric
        
    Returns:
        pass_rate: Fraction of correctly identified clips
    """
    logger.info("\n" + "="*70)
    logger.info("SANITY CHECK 1: Same-Clip Identity Test")
    logger.info("="*70)
    logger.info("Setup: Use same sequences for train and test")
    logger.info("Expected: 100% accuracy (each clip matches itself)")
    logger.info(f"Testing {n_samples} clips with {metric} distance")
    logger.info("")
    
    # Use first n_samples sequences
    test_seqs = sequences[:n_samples]
    train_seqs = sequences[:n_samples]
    
    n_correct = 0
    
    for i, test_seq in enumerate(test_seqs):
        # Compute distances to all training sequences
        distances = []
        for train_seq in train_seqs:
            dist = dtw_distance(test_seq, train_seq, metric=metric)
            distances.append(dist)
        
        # Find nearest neighbor
        nearest_idx = np.argmin(distances)
        
        if nearest_idx == i:
            n_correct += 1
        else:
            logger.warning(f"  Clip {i}: Matched to {nearest_idx} (expected {i})")
    
    pass_rate = n_correct / n_samples
    
    logger.info(f"\nResults: {n_correct}/{n_samples} correct ({pass_rate*100:.1f}%)")
    
    if pass_rate == 1.0:
        logger.info("✅ PASS: All clips correctly matched to themselves")
        return True
    else:
        logger.error(f"❌ FAIL: Only {pass_rate*100:.1f}% correct (expected 100%)")
        return False


def check_twin_retrieval(sequences, n_samples=20, metric='euclidean'):
    """
    Sanity Check 2: Twin retrieval test.
    
    Add time-warped copies of training sequences. Each test sequence should
    match its twin (not other clips).
    
    Args:
        sequences: List of sequences
        n_samples: Number of clips to test
        metric: DTW distance metric
        
    Returns:
        pass_rate: Fraction of correctly matched twins
    """
    logger.info("\n" + "="*70)
    logger.info("SANITY CHECK 2: Twin Retrieval Test")
    logger.info("="*70)
    logger.info("Setup: Create time-warped copies (drop 10% of frames)")
    logger.info("Expected: Each copy matches its original (nearest neighbor)")
    logger.info(f"Testing {n_samples} clip pairs with {metric} distance")
    logger.info("")
    
    # Use first n_samples sequences as training
    train_seqs = sequences[:n_samples]
    
    # Create jittered copies as test sequences
    test_seqs = [add_temporal_jitter(seq, seed=i) for i, seq in enumerate(train_seqs)]
    
    n_correct = 0
    
    for i, test_seq in enumerate(test_seqs):
        # Compute distances to all training sequences
        distances = []
        for train_seq in train_seqs:
            dist = dtw_distance(test_seq, train_seq, metric=metric)
            distances.append(dist)
        
        # Find nearest neighbor
        nearest_idx = np.argmin(distances)
        min_dist = distances[nearest_idx]
        
        if nearest_idx == i:
            n_correct += 1
        else:
            logger.warning(
                f"  Twin {i}: Matched to {nearest_idx} (expected {i}), "
                f"dist={min_dist:.2f} vs correct={distances[i]:.2f}"
            )
    
    pass_rate = n_correct / n_samples
    
    logger.info(f"\nResults: {n_correct}/{n_samples} twins correctly matched ({pass_rate*100:.1f}%)")
    
    if pass_rate >= 0.90:
        logger.info("✅ PASS: ≥90% of twins correctly matched")
        return True
    else:
        logger.error(f"❌ FAIL: Only {pass_rate*100:.1f}% correct (expected ≥90%)")
        return False


def check_class_separability(sequences, labels, n_classes=5, n_per_class=3, metric='euclidean'):
    """
    Sanity Check 3: Class separability test.
    
    Compute intra-class vs inter-class DTW distances.
    Ratio should be >> 1 (aim for ≥1.5x).
    
    Args:
        sequences: List of sequences
        labels: List of integer labels
        n_classes: Number of classes to test
        n_per_class: Samples per class
        metric: DTW distance metric
        
    Returns:
        separation_ratio: Inter-class distance / intra-class distance
    """
    logger.info("\n" + "="*70)
    logger.info("SANITY CHECK 3: Class Separability Test")
    logger.info("="*70)
    logger.info("Setup: Compute intra-class vs inter-class distances")
    logger.info("Expected: Inter-class >> Intra-class (ratio ≥1.5x)")
    logger.info(f"Testing {n_classes} classes × {n_per_class} samples with {metric} distance")
    logger.info("")
    
    # Convert labels to list if needed
    labels = list(labels)
    
    # Group sequences by class
    sequences_by_class = {}
    for seq, label in zip(sequences, labels):
        if label not in sequences_by_class:
            sequences_by_class[label] = []
        sequences_by_class[label].append(seq)
    
    # Select n_classes with enough samples, preferring diverse actions
    valid_classes = [
        (cls, len(seqs)) for cls, seqs in sequences_by_class.items()
        if len(seqs) >= n_per_class
    ]
    
    if len(valid_classes) < n_classes:
        logger.warning(f"Only {len(valid_classes)} classes with ≥{n_per_class} samples")
        n_classes = min(len(valid_classes), n_classes)
    
    # Sort by class ID and take first n_classes
    valid_classes.sort(key=lambda x: x[0])
    selected_classes = [cls for cls, count in valid_classes[:n_classes]]
    
    logger.info(f"Selected classes: {selected_classes}")
    logger.info(f"Samples per class: {[len(sequences_by_class[cls]) for cls in selected_classes]}")
    
    # Compute intra-class distances
    intra_distances = []
    
    logger.info("\nComputing intra-class distances...")
    for cls in selected_classes:
        class_seqs = sequences_by_class[cls][:n_per_class]
        
        for i in range(len(class_seqs)):
            for j in range(i+1, len(class_seqs)):
                dist = dtw_distance(class_seqs[i], class_seqs[j], metric=metric)
                intra_distances.append(dist)
                logger.debug(f"  Class {cls}: seq {i} vs {j} = {dist:.2f}")
    
    intra_mean = np.mean(intra_distances)
    intra_std = np.std(intra_distances)
    
    logger.info(f"  Intra-class: {intra_mean:.2f} ± {intra_std:.2f} (n={len(intra_distances)})")
    
    # Compute inter-class distances
    inter_distances = []
    
    logger.info("Computing inter-class distances...")
    for i, cls1 in enumerate(selected_classes):
        for cls2 in selected_classes[i+1:]:
            seqs1 = sequences_by_class[cls1][:n_per_class]
            seqs2 = sequences_by_class[cls2][:n_per_class]
            
            for seq1 in seqs1:
                for seq2 in seqs2:
                    dist = dtw_distance(seq1, seq2, metric=metric)
                    inter_distances.append(dist)
                    logger.debug(f"  Class {cls1} vs {cls2}: {dist:.2f}")
    
    inter_mean = np.mean(inter_distances)
    inter_std = np.std(inter_distances)
    
    logger.info(f"  Inter-class: {inter_mean:.2f} ± {inter_std:.2f} (n={len(inter_distances)})")
    
    # Compute separation ratio
    separation_ratio = inter_mean / intra_mean
    
    logger.info(f"\nSeparation ratio: {separation_ratio:.2f}x")
    
    if separation_ratio >= 1.5:
        logger.info(f"✅ PASS: Separation ratio {separation_ratio:.2f}x ≥ 1.5x")
        return True, separation_ratio
    elif separation_ratio >= 1.2:
        logger.warning(f"⚠️  MARGINAL: Separation ratio {separation_ratio:.2f}x (aim for ≥1.5x)")
        return False, separation_ratio
    else:
        logger.error(f"❌ FAIL: Separation ratio {separation_ratio:.2f}x < 1.2x")
        logger.error("   Classes are not well-separated!")
        return False, separation_ratio


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run sanity checks on QDTW encoding'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='msr_action_data',
        help='Directory containing raw skeleton files'
    )
    parser.add_argument(
        '--encoded-dir',
        type=str,
        default=None,
        help='Directory containing encoded sequences (optional)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='euclidean',
        choices=['euclidean', 'cosine', 'fidelity'],
        help='Distance metric for DTW'
    )
    parser.add_argument(
        '--test-raw',
        action='store_true',
        help='Test raw data (before encoding)'
    )
    parser.add_argument(
        '--test-encoded',
        action='store_true',
        help='Test encoded data (after standardization)'
    )
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("QDTW ENCODING SANITY CHECKS")
    logger.info("="*70)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Distance metric: {args.metric}")
    
    # Load sequences
    logger.info("\nLoading sequences...")
    sequences, labels = load_all_sequences(args.data_dir)
    logger.info(f"Loaded {len(sequences)} sequences with {len(set(labels))} unique classes")
    
    # Run tests on raw data
    if args.test_raw or not args.test_encoded:
        logger.info("\n" + "="*70)
        logger.info("TESTING RAW DATA (60-D skeleton coordinates)")
        logger.info("="*70)
        
        results = {}
        
        # Check 1: Same-clip identity
        results['identity'] = check_same_clip_identity(
            sequences[:50], n_samples=20, metric=args.metric
        )
        
        # Check 2: Twin retrieval
        results['twins'] = check_twin_retrieval(
            sequences[:50], n_samples=20, metric=args.metric
        )
        
        # Check 3: Class separability
        results['separability'], ratio_raw = check_class_separability(
            sequences, labels, n_classes=5, n_per_class=3, metric=args.metric
        )
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("RAW DATA SUMMARY")
        logger.info("="*70)
        logger.info(f"Same-clip identity: {'✅ PASS' if results['identity'] else '❌ FAIL'}")
        logger.info(f"Twin retrieval: {'✅ PASS' if results['twins'] else '❌ FAIL'}")
        logger.info(f"Class separability: {'✅ PASS' if results['separability'] else '❌ FAIL'} (ratio: {ratio_raw:.2f}x)")
        
        if all(results.values()):
            logger.info("\n✅ ALL RAW DATA CHECKS PASSED")
        else:
            logger.warning("\n⚠️  SOME RAW DATA CHECKS FAILED")
    
    # Run tests on encoded data
    if args.test_encoded:
        logger.info("\n" + "="*70)
        logger.info("TESTING ENCODED DATA (Standardization)")
        logger.info("="*70)
        
        # Apply encoding
        logger.info("\nApplying standardization encoding...")
        encoded_sequences = []
        for seq in sequences:
            # Encode each sequence independently
            seq_encoded = batch_encode_unit_vectors(seq)
            encoded_sequences.append(seq_encoded)
        
        logger.info(f"Encoded {len(encoded_sequences)} sequences")
        
        # Verify encoding
        sample_encoded = encoded_sequences[0]
        logger.info(f"\nEncoded sample statistics:")
        logger.info(f"  Shape: {sample_encoded.shape}")
        logger.info(f"  Mean: {np.mean(sample_encoded):.6f}")
        logger.info(f"  Std: {np.std(sample_encoded):.6f}")
        logger.info(f"  Range: [{np.min(sample_encoded):.2f}, {np.max(sample_encoded):.2f}]")
        
        results = {}
        
        # Check 1: Same-clip identity
        results['identity'] = check_same_clip_identity(
            encoded_sequences[:50], n_samples=20, metric=args.metric
        )
        
        # Check 2: Twin retrieval
        results['twins'] = check_twin_retrieval(
            encoded_sequences[:50], n_samples=20, metric=args.metric
        )
        
        # Check 3: Class separability
        results['separability'], ratio_encoded = check_class_separability(
            encoded_sequences, labels, n_classes=5, n_per_class=3, metric=args.metric
        )
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("ENCODED DATA SUMMARY")
        logger.info("="*70)
        logger.info(f"Same-clip identity: {'✅ PASS' if results['identity'] else '❌ FAIL'}")
        logger.info(f"Twin retrieval: {'✅ PASS' if results['twins'] else '❌ FAIL'}")
        logger.info(f"Class separability: {'✅ PASS' if results['separability'] else '❌ FAIL'} (ratio: {ratio_encoded:.2f}x)")
        
        if all(results.values()):
            logger.info("\n✅ ALL ENCODED DATA CHECKS PASSED")
            logger.info("   Standardization preserves discriminative information!")
        else:
            logger.error("\n❌ SOME ENCODED DATA CHECKS FAILED")
            logger.error("   Encoding may still be destroying class structure")
    
    logger.info("\n" + "="*70)


if __name__ == "__main__":
    main()
