#!/usr/bin/env python
"""
Project sequences into k-D subspaces using PCA components.

This script loads sequences, projects them using classical (Uc) and/or
quantum (Uq) PCA components, and saves the projected sequences to disk.

Usage:
    python scripts/project_sequences.py --k 8 --method both
    python scripts/project_sequences.py --k 10 --method Uq
    python scripts/project_sequences.py --k 5 --method Uc
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup to avoid import errors
from quantum.project import project_sequence  # noqa: E402
from quantum.classical_pca import load_pca_components  # noqa: E402
from quantum.qpca import load_qpca_components  # noqa: E402
from src.loader import load_all_sequences  # noqa: E402
from features.amplitude_encoding import batch_encode_unit_vectors  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simple_train_test_split(sequences, test_fraction=0.2, seed=42):
    """
    Simple train/test split function (copied from build_frame_bank.py).

    Args:
        sequences: List of sequences
        test_fraction: Fraction of data to use for testing (default: 0.2)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        train_sequences, test_sequences: Lists of sequences
    """
    np.random.seed(seed)
    n_total = len(sequences)
    n_test = int(n_total * test_fraction)

    indices = np.arange(n_total)
    np.random.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_sequences = [sequences[i] for i in train_indices]
    test_sequences = [sequences[i] for i in test_indices]

    return train_sequences, test_sequences


def load_sequences_from_msr(data_dir: Path) -> list[np.ndarray]:
    """
    Load sequences from MSR Action3D dataset using proper loader.
    Applies standardization encoding.

    Args:
        data_dir: Path to directory containing skeleton files

    Returns:
        sequences: List of sequences, each of shape [T, 60], standardized
    """
    logger.info(f"Loading sequences from {data_dir}")

    # Use the proper loader that handles MSR skeleton format
    sequences, labels = load_all_sequences(str(data_dir))
    
    logger.info(f"Loaded {len(sequences)} raw sequences")
    
    # Apply standardization encoding to each sequence
    logger.info("Applying standardization encoding...")
    encoded_sequences = []
    for seq in sequences:
        # Encode each sequence independently using standardization
        seq_encoded = batch_encode_unit_vectors(seq)
        encoded_sequences.append(seq_encoded)
    
    logger.info(f"Encoded {len(encoded_sequences)} sequences")
    
    return encoded_sequences, labels


def project_and_save(
    sequences: list[np.ndarray],
    labels: list[int],
    U: np.ndarray,
    output_dir: Path,
    split_name: str,
    k: int
) -> None:
    """
    Project sequences and save to disk with metadata.

    Args:
        sequences: List of sequences to project
        labels: List of labels for sequences
        U: Principal components matrix [D, k]
        output_dir: Base output directory
        split_name: 'train' or 'test'
        k: Number of components
    """
    logger.info(
        f"Projecting {len(sequences)} {split_name} sequences into {k}D"
    )

    # Create output directory
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Project and save each sequence
    for i, seq in enumerate(sequences):
        projected = project_sequence(seq, U, normalize_rows=False)

        # Generate filename (preserve original naming if possible)
        filename = f"seq_{i:04d}.npy"
        filepath = split_dir / filename

        np.save(filepath, projected)

        if (i + 1) % 100 == 0:
            logger.info(f"  Saved {i + 1}/{len(sequences)} sequences")

    # Save metadata with labels
    metadata_path = split_dir / 'metadata.npz'
    np.savez(
        metadata_path,
        labels=np.array(labels),
        filenames=np.array([f"seq_{i:04d}.npy" for i in range(len(sequences))])
    )
    
    logger.info(
        f"Saved {len(sequences)} {split_name} sequences to {split_dir}"
    )
    logger.info(f"Saved metadata to {metadata_path}")


def main():
    """Main function for projecting sequences."""
    parser = argparse.ArgumentParser(
        description='Project sequences into k-D subspaces using PCA components'
    )
    parser.add_argument(
        '--k',
        type=int,
        required=True,
        help='Number of principal components (e.g., 5, 8, 10)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['both', 'Uq', 'Uc'],
        default='both',
        help='Which PCA method(s) to use: both, Uq (quantum), or Uc (classical)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='msr_action_data',
        help='Directory containing skeleton data files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/subspace',
        help='Base output directory for projected sequences'
    )
    parser.add_argument(
        '--test-fraction',
        type=float,
        default=0.2,
        help='Fraction of data to use for testing (default: 0.2)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/test split (default: 42)'
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("SEQUENCE PROJECTION")
    logger.info("=" * 70)
    logger.info(f"k = {args.k}")
    logger.info(f"Method = {args.method}")
    logger.info(f"Data directory = {args.data_dir}")
    logger.info(f"Output directory = {args.output_dir}")

    # Convert paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Load sequences (with encoding)
    sequences, labels = load_sequences_from_msr(data_dir)

    if not sequences:
        logger.error("No sequences loaded. Exiting.")
        return 1

    # Split into train/test
    train_sequences, test_sequences = simple_train_test_split(
        sequences,
        test_fraction=args.test_fraction,
        seed=args.seed
    )
    
    # Split labels too
    train_labels, test_labels = simple_train_test_split(
        labels,
        test_fraction=args.test_fraction,
        seed=args.seed
    )

    logger.info(f"Train sequences: {len(train_sequences)}")
    logger.info(f"Test sequences: {len(test_sequences)}")

    # Determine which methods to use
    methods = []
    if args.method == 'both':
        methods = ['Uq', 'Uc']
    else:
        methods = [args.method]

    # Process each method
    for method in methods:
        logger.info("=" * 70)
        logger.info(f"Processing method: {method}")
        logger.info("=" * 70)

        # Load appropriate PCA components
        try:
            if method == 'Uq':
                # Load from standardized PCA file
                pca_file = Path('results') / f'Uq_k{args.k}_std.npz'
                if not pca_file.exists():
                    pca_file = Path('results') / f'Uq_k{args.k}.npz'
                data = np.load(pca_file)
                U = data['U']
                logger.info(f"Loaded quantum PCA components from {pca_file}: shape {U.shape}")
            else:  # Uc
                # Load from standardized PCA file
                pca_file = Path('results') / f'Uc_k{args.k}_std.npz'
                if not pca_file.exists():
                    pca_file = Path('results') / f'Uc_k{args.k}.npz'
                data = np.load(pca_file)
                U = data['U']
                logger.info(f"Loaded classical PCA components from {pca_file}: shape {U.shape}")
        except FileNotFoundError as e:
            logger.error(f"Failed to load {method} components: {e}")
            logger.error(
                f"Please run classical_pca first to generate "
                f"{method}_k{args.k}_std.npz"
            )
            continue

        # Create output directories
        method_output_dir = output_dir / method / f"k{args.k}"

        # Project train sequences
        project_and_save(
            train_sequences,
            train_labels,
            U,
            method_output_dir,
            'train',
            args.k
        )

        # Project test sequences
        project_and_save(
            test_sequences,
            test_labels,
            U,
            method_output_dir,
            'test',
            args.k
        )

        logger.info(f"Completed {method} projection")

    logger.info("=" * 70)
    logger.info("PROJECTION COMPLETE")
    logger.info("=" * 70)

    # Print summary
    for method in methods:
        for k_val in [args.k]:
            for split in ['train', 'test']:
                path = output_dir / method / f"k{k_val}" / split
                if path.exists():
                    n_files = len(list(path.glob("*.npy")))
                    logger.info(f"{method}/k{k_val}/{split}: {n_files} files")

    return 0


if __name__ == '__main__':
    sys.exit(main())
