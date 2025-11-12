"""
Build a frame bank by sampling frames from training sequences.

This script creates a representative set of normalized skeleton frames
for quantum PCA analysis.
"""

import numpy as np
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from features.amplitude_encoding import batch_encode_unit_vectors  # noqa: E402
from src.loader import load_all_sequences  # noqa: E402


def simple_train_test_split(sequences, labels, test_size=0.3, random_state=42):
    """
    Simple train/test split to avoid sklearn dependency issues.

    Args:
        sequences: List of sequences
        labels: List of labels
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    n = len(sequences)
    indices = np.arange(n)
    np.random.shuffle(indices)

    split_idx = int(n * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train = [sequences[i] for i in train_indices]
    X_test = [sequences[i] for i in test_indices]
    y_train = [labels[i] for i in train_indices]
    y_test = [labels[i] for i in test_indices]

    return X_train, X_test, y_train, y_test


def build_frame_bank(train_sequences, per_sequence: int = 20, seed: int = 42) -> np.ndarray:
    """
    Sample frames from training sequences and normalize them.

    Args:
        train_sequences: List of sequences, each of shape [T, 60]
        per_sequence: Number of frames to sample per sequence
        seed: Random seed for reproducibility

    Returns:
        frame_bank: np.ndarray of shape [M, 60] where M = len(train_sequences) * per_sequence
                   All rows are L2-normalized (unit vectors)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    all_frames = []

    for seq in train_sequences:
        T = seq.shape[0]  # Number of frames in sequence

        # Sample frames uniformly across the sequence
        if T <= per_sequence:
            # If sequence is short, take all frames
            sampled_indices = list(range(T))
        else:
            # Sample uniformly across the sequence
            sampled_indices = np.linspace(0, T - 1, per_sequence, dtype=int)

        sampled_frames = seq[sampled_indices]
        all_frames.append(sampled_frames)

    # Concatenate all sampled frames
    frame_bank = np.vstack(all_frames)  # Shape: [M, 60]

    # Encode frames using standardization (NOT unit normalization)
    # This preserves relative magnitude differences between features
    frame_bank_normalized = batch_encode_unit_vectors(frame_bank)

    return frame_bank_normalized


def main():
    """Build and save the frame bank."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Build frame bank from training sequences'
    )
    parser.add_argument(
        '--per-seq',
        type=int,
        default=20,
        help='Number of frames to sample per sequence (default: 20)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/frame_bank.npy',
        help='Output path for frame bank (default: data/frame_bank.npy)'
    )

    args = parser.parse_args()

    print("="*60)
    print("BUILDING FRAME BANK FOR QUANTUM PCA")
    print("="*60)
    print("Parameters:")
    print(f"  Frames per sequence: {args.per_seq}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output path: {args.output}")

    # Load dataset
    print("\nLoading MSR Action dataset...")
    sequences, labels = load_all_sequences("msr_action_data")
    print(f"Loaded {len(sequences)} sequences")

    # Split into train/test
    X_train, X_test, y_train, y_test = simple_train_test_split(
        sequences, labels, test_size=0.3, random_state=args.seed)

    print(f"Training sequences: {len(X_train)}")
    print(f"Test sequences: {len(X_test)}")

    # Build frame bank
    print("\nSampling frames from training sequences...")
    frame_bank = build_frame_bank(X_train, per_sequence=args.per_seq, seed=args.seed)

    print(f"\nFrame bank shape: {frame_bank.shape}")
    print(f"Total frames: {frame_bank.shape[0]}")
    print(f"Expected: ~{len(X_train) * args.per_seq}")

    # Verify standardization (not L2 normalization)
    means = np.mean(frame_bank, axis=0)
    stds = np.std(frame_bank, axis=0)
    print("\nStandardization check:")
    print(f"  Mean per feature: {np.mean(means):.6f} (expect ~0)")
    print(f"  Std per feature: {np.mean(stds):.6f} (expect ~1)")
    print(f"  Mean range: [{np.min(means):.4f}, {np.max(means):.4f}]")
    print(f"  Std range: [{np.min(stds):.4f}, {np.max(stds):.4f}]")
    print(f"  Properly standardized: {np.allclose(means, 0, atol=0.1) and np.allclose(stds, 1, atol=0.2)}")

    # Save frame bank
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, frame_bank)
    print(f"\n✅ Frame bank saved to: {args.output}")

    # Statistics
    print("\nFrame bank statistics:")
    print(f"  Mean per dimension: {np.mean(frame_bank, axis=0).mean():.4f}")
    print(f"  Std per dimension: {np.std(frame_bank, axis=0).mean():.4f}")
    print(f"  Min value: {np.min(frame_bank):.4f}")
    print(f"  Max value: {np.max(frame_bank):.4f}")


if __name__ == "__main__":
    main()
