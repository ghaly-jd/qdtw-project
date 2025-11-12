"""
Create label metadata for projected sequences.

This script creates a mapping from sequence indices to action labels
based on the original MSR Action3D filenames.
"""

import logging
import re
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_action_id_from_filename(filename: str) -> int:
    """
    Extract action ID from MSR Action3D filename.
    
    Args:
        filename: Filename like 'a01_s01_e01_skeleton.txt'
    
    Returns:
        action_id: Action ID (1-20)
    """
    match = re.match(r'a(\d+)_s(\d+)_e(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not parse action ID from filename: {filename}")


def create_metadata_for_split(
    data_dir: Path,
    output_dir: Path,
    split: str
) -> None:
    """
    Create metadata file for a specific train/test split.
    
    Args:
        data_dir: Path to MSR Action3D data directory
        output_dir: Path to projected sequences directory (e.g., results/subspace/Uq/k8/train)
        split: 'train' or 'test'
    """
    # Get list of skeleton files in the same order as project_sequences.py
    skeleton_files = sorted(data_dir.glob("*.txt"))
    
    # Load sequences to get the same train/test split
    # (Using same logic as project_sequences.py)
    n_total = len(skeleton_files)
    n_test = int(n_total * 0.2)
    
    np.random.seed(42)  # Same seed as project_sequences.py
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    
    if split == 'test':
        split_indices = indices[:n_test]
    else:  # train
        split_indices = indices[n_test:]
    
    # Create mapping: seq_idx -> action_id
    labels = []
    filenames = []
    
    for i, idx in enumerate(sorted(split_indices)):
        filename = skeleton_files[idx].name
        action_id = extract_action_id_from_filename(filename)
        labels.append(action_id)
        filenames.append(filename)
    
    # Save metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / 'metadata.npz'
    
    np.savez(
        metadata_path,
        labels=np.array(labels),
        filenames=np.array(filenames)
    )
    
    logger.info(f"Saved metadata to {metadata_path}")
    logger.info(f"  {len(labels)} sequences")
    logger.info(f"  {len(set(labels))} unique action classes")
    logger.info(f"  Action IDs: {sorted(set(labels))}")


def main():
    """Main function."""
    data_dir = Path('msr_action_data')
    
    # Create metadata for all method/k/split combinations
    methods = ['Uq', 'Uc']
    k_values = [3, 5, 8, 10, 12, 16]
    splits = ['train', 'test']
    
    for method in methods:
        for k in k_values:
            for split in splits:
                output_dir = Path(f'results/subspace/{method}/k{k}/{split}')
                
                if output_dir.exists():
                    logger.info(f"\nProcessing {method} k={k} {split}...")
                    create_metadata_for_split(data_dir, output_dir, split)
                else:
                    logger.warning(f"Directory not found: {output_dir}")
    
    logger.info("\nMetadata creation complete!")


if __name__ == '__main__':
    main()
