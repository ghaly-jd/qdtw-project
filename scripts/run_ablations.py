#!/usr/bin/env python
"""
Run ablation studies for QDTW evaluation.

This script runs comprehensive ablation experiments to analyze:
- Distance metric choice
- k value sweep
- Frame sampling strategies
- Robustness to noise and temporal jitter

Usage:
    python scripts/run_ablations.py --all
    python scripts/run_ablations.py --distance --k-sweep
    python scripts/run_ablations.py --sampling --robustness
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.ablations import (  # noqa: E402
    plot_distance_choice_ablation,
    plot_k_sweep_ablation,
    plot_robustness_ablation,
    plot_sampling_strategy_ablation,
    run_distance_choice_ablation,
    run_k_sweep_ablation,
    run_robustness_ablation,
    run_sampling_strategy_ablation,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sequences_subset(
    method: str,
    k: int,
    split: str,
    n_samples: int = 30
) -> tuple:
    """Load a subset of sequences for ablation studies."""
    # Try standardized data first (new encoding), fall back to old
    base_path = Path(f'results/subspace_std/{method}/k{k}/{split}')
    if not base_path.exists():
        base_path = Path(f'results/subspace/{method}/k{k}/{split}')

    if not base_path.exists():
        logger.warning(f"Path not found: {base_path}")
        return [], []

    seq_files = sorted(base_path.glob('seq_*.npy'))[:n_samples]

    sequences = []
    labels = []

    # Load label mapping from metadata if available
    label_map_path = base_path / 'metadata.npz'
    metadata_labels = None
    if label_map_path.exists():
        metadata = np.load(label_map_path, allow_pickle=True)
        if 'labels' in metadata:
            metadata_labels = metadata['labels']

    for i, filepath in enumerate(seq_files):
        seq = np.load(filepath)
        sequences.append(seq)
        
        # Extract sequence index from filename (e.g., seq_0041.npy -> 41)
        seq_idx = int(filepath.stem.split('_')[1])
        
        # Use seq_idx (from filename) to get the label from metadata
        # The metadata labels array is indexed by the sequence number in the filename
        if metadata_labels is not None and seq_idx < len(metadata_labels):
            labels.append(int(metadata_labels[seq_idx]))
        else:
            # Fall back to extracting action ID from original filename pattern
            # For MSR Action3D, action ID is in range 1-20
            # We'll use a simple modulo for now if metadata not available
            logger.warning(f"No metadata label for seq_{seq_idx:04d}, using fallback")
            labels.append((seq_idx % 20) + 1)  # Actions are 1-20, not 0-19

    return sequences, labels


def create_sample_sequences(n_train: int = 30, n_test: int = 10):
    """Create sample sequences for testing when real data not available."""
    logger.info("Creating sample sequences...")

    # Generate synthetic sequences
    train_seqs = [np.random.randn(np.random.randint(80, 120), 8) for _ in range(n_train)]
    train_labels = [i % 20 for i in range(n_train)]

    test_seqs = [np.random.randn(np.random.randint(80, 120), 8) for _ in range(n_test)]
    test_labels = [i % 20 for i in range(n_test)]

    return train_seqs, train_labels, test_seqs, test_labels


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run QDTW ablation studies'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all ablation experiments'
    )
    parser.add_argument(
        '--distance',
        action='store_true',
        help='Run distance choice ablation'
    )
    parser.add_argument(
        '--k-sweep',
        action='store_true',
        help='Run k sweep ablation'
    )
    parser.add_argument(
        '--sampling',
        action='store_true',
        help='Run sampling strategy ablation'
    )
    parser.add_argument(
        '--robustness',
        action='store_true',
        help='Run robustness ablation'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for CSV (default: results)'
    )
    parser.add_argument(
        '--figures-dir',
        type=str,
        default='figures',
        help='Output directory for figures (default: figures)'
    )
    parser.add_argument(
        '--use-sample-data',
        action='store_true',
        help='Use synthetic sample data for testing'
    )
    parser.add_argument(
        '--n-train',
        type=int,
        default=30,
        help='Number of training samples (default: 30)'
    )
    parser.add_argument(
        '--n-test',
        type=int,
        default=10,
        help='Number of test samples (default: 10)'
    )

    args = parser.parse_args()

    # If --all specified, enable all experiments
    if args.all:
        args.distance = True
        args.k_sweep = True
        args.sampling = True
        args.robustness = True

    # If no experiments specified, show help
    if not (args.distance or args.k_sweep or args.sampling or args.robustness):
        parser.print_help()
        return 1

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)

    logger.info("=" * 70)
    logger.info("QDTW ABLATION STUDIES")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Figures directory: {figures_dir}")

    all_results = []

    # Load or create data
    if args.use_sample_data:
        logger.info("\nUsing synthetic sample data...")
        train_seqs, train_labels, test_seqs, test_labels = create_sample_sequences(
            args.n_train, args.n_test
        )
    else:
        logger.info("\nLoading sequences from projected data...")
        # Try to load from Uq k=8 as default
        train_seqs, train_labels = load_sequences_subset('Uq', 8, 'train', args.n_train)
        test_seqs, test_labels = load_sequences_subset('Uq', 8, 'test', args.n_test)

        if not train_seqs:
            logger.warning("No real data found, falling back to sample data...")
            train_seqs, train_labels, test_seqs, test_labels = create_sample_sequences(
                args.n_train, args.n_test
            )

    logger.info(f"Loaded {len(train_seqs)} train, {len(test_seqs)} test sequences")

    # ===== SANITY CHECKS =====
    logger.info("\n" + "=" * 70)
    logger.info("SANITY CHECKS")
    logger.info("=" * 70)
    
    train_classes = sorted(set(train_labels))
    test_classes = sorted(set(test_labels))
    
    logger.info(f"Train classes: {train_classes}")
    logger.info(f"Test classes: {test_classes}")
    logger.info(f"Number of train classes: {len(train_classes)}")
    logger.info(f"Number of test classes: {len(test_classes)}")
    
    # Check for valid MSR Action3D labels (0-19 or 1-20 depending on encoding)
    if not args.use_sample_data:
        # Support both 0-indexed and 1-indexed labels
        min_label = min(train_labels + test_labels)
        max_label = max(train_labels + test_labels)
        if min_label >= 0 and max_label <= 19:
            # 0-indexed labels (new standardized data)
            assert all(0 <= lbl <= 19 for lbl in train_labels), \
                f"Invalid train labels: {[lbl for lbl in train_labels if not (0 <= lbl <= 19)][:10]}"
            assert all(0 <= lbl <= 19 for lbl in test_labels), \
                f"Invalid test labels: {[lbl for lbl in test_labels if not (0 <= lbl <= 19)][:10]}"
        else:
            # 1-indexed labels (old data)
            assert all(1 <= lbl <= 20 for lbl in train_labels), \
                f"Invalid train labels: {[lbl for lbl in train_labels if not (1 <= lbl <= 20)][:10]}"
            assert all(1 <= lbl <= 20 for lbl in test_labels), \
                f"Invalid test labels: {[lbl for lbl in test_labels if not (1 <= lbl <= 20)][:10]}"
        
        # Warn if too few classes (expected for small samples)
        if len(train_classes) < 10:
            logger.warning(f"⚠️  Only {len(train_classes)} train classes (expected ~20 for full dataset)")
            logger.warning(f"   Consider using --n-train 454 for complete class coverage")
        else:
            logger.info(f"✅ Good class diversity: {len(train_classes)}/20 actions represented")
        
        logger.info("✅ Labels are valid MSR Action3D IDs (1-20)")
    
    # Print class distribution
    from collections import Counter
    train_dist = Counter(train_labels)
    logger.info(f"Train class distribution (top 5): {train_dist.most_common(5)}")
    test_dist = Counter(test_labels)
    logger.info(f"Test class distribution (top 5): {test_dist.most_common(5)}")
    
    logger.info("=" * 70)

    # Track total progress
    total_experiments = 0
    if args.distance: total_experiments += 1
    if args.k_sweep: total_experiments += 1
    if args.sampling: total_experiments += 1
    if args.robustness: total_experiments += 1
    
    current_experiment = 0
    overall_start = time.time()

    # 1. Distance Choice Ablation
    if args.distance:
        current_experiment += 1
        exp_start = time.time()
        
        logger.info("\n" + "=" * 70)
        logger.info(f"ABLATION {current_experiment}/{total_experiments}: Distance Choice")
        logger.info("=" * 70)
        logger.info(f"Testing 6 configurations (Uq/Uc × cosine/euclidean/fidelity)")
        logger.info(f"Each config: {len(test_seqs)} test samples × {len(train_seqs)} train samples")
        logger.info(f"Estimated time: ~{len(test_seqs) * len(train_seqs) * 6 * 0.001:.1f}s")
        logger.info(f"Progress: Experiment {current_experiment}/{total_experiments}")

        if args.use_sample_data:
            # For sample data, create pseudo Uc data
            results_df = run_distance_choice_ablation(
                train_seqs,
                train_labels,
                test_seqs,
                test_labels,
                methods=['Uq', 'Uc'],
                metrics=['cosine', 'euclidean', 'fidelity']
            )
        else:
            # Load both Uq and Uc
            uq_train, uq_train_labels = load_sequences_subset('Uq', 8, 'train', args.n_train)
            uq_test, uq_test_labels = load_sequences_subset('Uq', 8, 'test', args.n_test)
            uc_train, uc_train_labels = load_sequences_subset('Uc', 8, 'train', args.n_train)
            uc_test, uc_test_labels = load_sequences_subset('Uc', 8, 'test', args.n_test)

            # Run ablation for each method
            results_list = []
            if uq_train:
                uq_results = run_distance_choice_ablation(
                    uq_train, uq_train_labels, uq_test, uq_test_labels,
                    methods=['Uq'], metrics=['cosine', 'euclidean', 'fidelity']
                )
                results_list.append(uq_results)

            if uc_train:
                uc_results = run_distance_choice_ablation(
                    uc_train, uc_train_labels, uc_test, uc_test_labels,
                    methods=['Uc'], metrics=['cosine', 'euclidean', 'fidelity']
                )
                results_list.append(uc_results)

            if results_list:
                results_df = pd.concat(results_list, ignore_index=True)
            else:
                results_df = pd.DataFrame()

        if not results_df.empty:
            all_results.append(results_df)
            
            exp_time = time.time() - exp_start
            logger.info(f"\n✅ Distance Choice completed in {exp_time:.1f}s ({exp_time/60:.1f}m)")
            # Convert to native Python types to avoid numpy compatibility issues
            best_acc = float(results_df['accuracy'].max())
            logger.info(f"Best accuracy: {best_acc*100:.2f}%")
            best_idx = int(results_df['accuracy'].idxmax())
            logger.info(f"Best metric: {results_df.loc[best_idx, 'metric']}")
            
            plot_distance_choice_ablation(
                results_df,
                figures_dir / 'ablations_distance.png'
            )
            logger.info(f"Saved figure: {figures_dir / 'ablations_distance.png'}")
            
            # Estimate remaining time
            if current_experiment < total_experiments:
                avg_time_per_exp = (time.time() - overall_start) / current_experiment
                remaining_time = avg_time_per_exp * (total_experiments - current_experiment)
                eta = datetime.now() + timedelta(seconds=remaining_time)
                logger.info(f"Estimated time remaining: {remaining_time/60:.1f}m (ETA: {eta.strftime('%H:%M:%S')})")

    # 2. k Sweep Ablation
    if args.k_sweep:
        current_experiment += 1
        exp_start = time.time()
        
        logger.info("\n" + "=" * 70)
        logger.info(f"ABLATION {current_experiment}/{total_experiments}: k Sweep")
        logger.info("=" * 70)

        k_values = [3, 5, 8, 10, 12, 16]
        logger.info(f"Testing {len(k_values)} k values: {k_values}")
        logger.info(f"Each k: {len(test_seqs)} test samples × {len(train_seqs)} train samples")
        logger.info(f"Estimated time: ~{len(test_seqs) * len(train_seqs) * len(k_values) * 0.001:.1f}s")
        logger.info(f"Progress: Experiment {current_experiment}/{total_experiments}")

        if args.use_sample_data:
            # Create sample data at different k values
            sequences_by_k = {}
            test_sequences_by_k = {}

            for k in k_values:
                # Simulate different k by slicing dimensions
                k_actual = min(k, 8)  # Limit to available dimensions
                train_k = [seq[:, :k_actual] for seq in train_seqs]
                test_k = [seq[:, :k_actual] for seq in test_seqs]
                sequences_by_k[k] = (train_k, train_labels)
                test_sequences_by_k[k] = (test_k, test_labels)

            results_df = run_k_sweep_ablation(
                sequences_by_k,
                test_sequences_by_k,
                methods=['Uq'],
                k_values=k_values,
                metric='euclidean'
            )
        else:
            # Load sequences at different k values
            sequences_by_k_uq = {}
            test_sequences_by_k_uq = {}
            sequences_by_k_uc = {}
            test_sequences_by_k_uc = {}

            for k in k_values:
                uq_train, uq_train_labels = load_sequences_subset('Uq', k, 'train', args.n_train)
                uq_test, uq_test_labels = load_sequences_subset('Uq', k, 'test', args.n_test)
                if uq_train:
                    sequences_by_k_uq[k] = (uq_train, uq_train_labels)
                    test_sequences_by_k_uq[k] = (uq_test, uq_test_labels)

                uc_train, uc_train_labels = load_sequences_subset('Uc', k, 'train', args.n_train)
                uc_test, uc_test_labels = load_sequences_subset('Uc', k, 'test', args.n_test)
                if uc_train:
                    sequences_by_k_uc[k] = (uc_train, uc_train_labels)
                    test_sequences_by_k_uc[k] = (uc_test, uc_test_labels)

            results_list = []
            if sequences_by_k_uq:
                uq_results = run_k_sweep_ablation(
                    sequences_by_k_uq, test_sequences_by_k_uq,
                    methods=['Uq'], k_values=k_values, metric='euclidean'
                )
                results_list.append(uq_results)

            if sequences_by_k_uc:
                uc_results = run_k_sweep_ablation(
                    sequences_by_k_uc, test_sequences_by_k_uc,
                    methods=['Uc'], k_values=k_values, metric='euclidean'
                )
                results_list.append(uc_results)

            if results_list:
                results_df = pd.concat(results_list, ignore_index=True)
            else:
                results_df = pd.DataFrame()

        if not results_df.empty:
            all_results.append(results_df)
            
            exp_time = time.time() - exp_start
            logger.info(f"\n✅ K Sweep completed in {exp_time:.1f}s ({exp_time/60:.1f}m)")
            logger.info(f"Best accuracy: {results_df['accuracy'].max()*100:.2f}% at k={int(results_df.loc[results_df['accuracy'].idxmax(), 'k'])}")
            logger.info(f"Worst accuracy: {results_df['accuracy'].min()*100:.2f}% at k={int(results_df.loc[results_df['accuracy'].idxmin(), 'k'])}")
            
            plot_k_sweep_ablation(
                results_df,
                figures_dir / 'ablations_k_sweep.png'
            )
            logger.info(f"Saved figure: {figures_dir / 'ablations_k_sweep.png'}")
            
            # Estimate remaining time
            if current_experiment < total_experiments:
                avg_time_per_exp = (time.time() - overall_start) / current_experiment
                remaining_time = avg_time_per_exp * (total_experiments - current_experiment)
                eta = datetime.now() + timedelta(seconds=remaining_time)
                logger.info(f"Estimated time remaining: {remaining_time/60:.1f}m (ETA: {eta.strftime('%H:%M:%S')})")

    # 3. Sampling Strategy Ablation
    if args.sampling:
        current_experiment += 1
        exp_start = time.time()
        
        logger.info("\n" + "=" * 70)
        logger.info(f"ABLATION {current_experiment}/{total_experiments}: Sampling Strategy")
        logger.info("=" * 70)
        logger.info(f"Testing 2 strategies: uniform vs energy-based")
        logger.info(f"Each strategy: {len(test_seqs)} test samples × {len(train_seqs)} train samples")
        logger.info(f"Estimated time: ~{len(test_seqs) * len(train_seqs) * 2 * 0.001:.1f}s")
        logger.info(f"Progress: Experiment {current_experiment}/{total_experiments}")

        results_df = run_sampling_strategy_ablation(
            train_seqs,
            train_labels,
            test_seqs,
            test_labels,
            n_samples=50,
            metric='euclidean'
        )

        if not results_df.empty:
            all_results.append(results_df)
            
            exp_time = time.time() - exp_start
            logger.info(f"\n✅ Sampling Strategy completed in {exp_time:.1f}s ({exp_time/60:.1f}m)")
            best_row = results_df.loc[results_df['accuracy'].idxmax()]
            logger.info(f"Best strategy: {best_row['setting']} ({best_row['accuracy']*100:.2f}%)")
            logger.info(f"Uniform: {results_df[results_df['setting']=='uniform']['accuracy'].values[0]*100:.2f}%")
            logger.info(f"Energy: {results_df[results_df['setting']=='energy']['accuracy'].values[0]*100:.2f}%")
            
            plot_sampling_strategy_ablation(
                results_df,
                figures_dir / 'ablations_sampling.png'
            )
            logger.info(f"Saved figure: {figures_dir / 'ablations_sampling.png'}")
            
            # Estimate remaining time
            if current_experiment < total_experiments:
                avg_time_per_exp = (time.time() - overall_start) / current_experiment
                remaining_time = avg_time_per_exp * (total_experiments - current_experiment)
                eta = datetime.now() + timedelta(seconds=remaining_time)
                logger.info(f"Estimated time remaining: {remaining_time/60:.1f}m (ETA: {eta.strftime('%H:%M:%S')})")

    # 4. Robustness Ablation
    if args.robustness:
        current_experiment += 1
        exp_start = time.time()
        
        logger.info("\n" + "=" * 70)
        logger.info(f"ABLATION {current_experiment}/{total_experiments}: Robustness")
        logger.info("=" * 70)
        
        noise_sigmas = [0.0, 0.01, 0.02]
        jitter_rates = [0.0, 0.05]
        total_tests = len(noise_sigmas) + len(jitter_rates)
        
        logger.info(f"Testing {len(noise_sigmas)} noise levels: {noise_sigmas}")
        logger.info(f"Testing {len(jitter_rates)} jitter rates: {jitter_rates}")
        logger.info(f"Total: {total_tests} robustness tests")
        logger.info(f"Each test: {len(test_seqs)} test samples × {len(train_seqs)} train samples")
        logger.info(f"Estimated time: ~{len(test_seqs) * len(train_seqs) * total_tests * 0.001:.1f}s")
        logger.info(f"Progress: Experiment {current_experiment}/{total_experiments}")

        results_df = run_robustness_ablation(
            train_seqs,
            train_labels,
            test_seqs,
            test_labels,
            noise_sigmas=noise_sigmas,
            jitter_rates=jitter_rates,
            metric='euclidean'
        )

        if not results_df.empty:
            all_results.append(results_df)
            
            exp_time = time.time() - exp_start
            logger.info(f"\n✅ Robustness completed in {exp_time:.1f}s ({exp_time/60:.1f}m)")
            
            noise_df = results_df[results_df['exp'] == 'robustness_noise']
            jitter_df = results_df[results_df['exp'] == 'robustness_jitter']
            
            if not noise_df.empty:
                logger.info(f"Noise robustness: {noise_df['accuracy'].min()*100:.2f}% - {noise_df['accuracy'].max()*100:.2f}%")
            if not jitter_df.empty:
                logger.info(f"Jitter robustness: {jitter_df['accuracy'].min()*100:.2f}% - {jitter_df['accuracy'].max()*100:.2f}%")
            
            plot_robustness_ablation(
                results_df,
                figures_dir / 'ablations_robustness.png'
            )
            logger.info(f"Saved figure: {figures_dir / 'ablations_robustness.png'}")

    # Save combined results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        output_path = output_dir / 'ablations.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        
        total_time = time.time() - overall_start
        
        logger.info("\n" + "=" * 70)
        logger.info("ABLATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total experiments run: {current_experiment}")
        logger.info(f"Total configurations tested: {len(combined_df)}")
        logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        logger.info(f"Average time per experiment: {total_time/current_experiment:.1f}s")
        logger.info("")

        for exp in combined_df['exp'].unique():
            exp_data = combined_df[combined_df['exp'] == exp]
            logger.info(f"\n{exp}:")
            logger.info(f"  Settings tested: {len(exp_data)}")
            logger.info(f"  Best accuracy: {exp_data['accuracy'].max():.4f}")
            logger.info(f"  Worst accuracy: {exp_data['accuracy'].min():.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE!")
    logger.info("=" * 70)
    if all_results:
        logger.info(f"Results saved to: {output_dir}/ablations.csv")
        logger.info(f"Figures saved to: {figures_dir}/ablations_*.png")
    logger.info("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
