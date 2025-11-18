"""
Ablation studies for QDTW evaluation.

This module implements various ablation experiments to analyze:
1. Distance metric choice (cosine, euclidean, fidelity)
2. k value sweep (dimensionality reduction)
3. Frame sampling strategy (uniform vs energy-based)
4. Robustness to noise and temporal jitter
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_temporal_jitter(
    sequence: np.ndarray,
    drop_rate: float = 0.05,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add temporal jitter by randomly dropping frames and interpolating.

    Args:
        sequence: Input sequence of shape [T, D]
        drop_rate: Fraction of frames to drop (default: 0.05 = 5%)
        seed: Random seed for reproducibility

    Returns:
        Jittered sequence with same shape [T, D]
    """
    if seed is not None:
        np.random.seed(seed)

    T, D = sequence.shape
    n_drop = int(T * drop_rate)

    if n_drop == 0 or T <= 2:
        return sequence.copy()

    # Select frames to keep (random indices)
    indices = np.arange(T)
    drop_indices = np.random.choice(T, size=n_drop, replace=False)
    keep_indices = np.setdiff1d(indices, drop_indices)
    keep_indices = np.sort(keep_indices)

    if len(keep_indices) < 2:
        return sequence.copy()

    # Interpolate back to original length
    jittered = np.zeros_like(sequence)
    for d in range(D):
        jittered[:, d] = np.interp(
            indices,
            keep_indices,
            sequence[keep_indices, d]
        )

    return jittered


def add_joint_noise(
    sequence: np.ndarray,
    sigma: float = 0.01,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Gaussian noise to joint coordinates.

    Args:
        sequence: Input sequence of shape [T, D]
        sigma: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        Noisy sequence of shape [T, D]
    """
    if seed is not None:
        np.random.seed(seed)

    noise = np.random.randn(*sequence.shape) * sigma
    return sequence + noise


def sample_frames_uniform(
    sequence: np.ndarray,
    n_samples: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Sample frames uniformly over time.

    Args:
        sequence: Input sequence of shape [T, D]
        n_samples: Number of frames to sample
        seed: Random seed

    Returns:
        Sampled frames of shape [n_samples, D]
    """
    if seed is not None:
        np.random.seed(seed)

    T = sequence.shape[0]
    if n_samples >= T:
        return sequence.copy()

    # Uniform sampling
    indices = np.linspace(0, T - 1, n_samples, dtype=int)
    return sequence[indices]


def sample_frames_energy(
    sequence: np.ndarray,
    n_samples: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Sample frames based on energy (L2 norm).

    Selects frames with highest L2 norm, which tend to be
    more informative (e.g., peak motion moments).

    Args:
        sequence: Input sequence of shape [T, D]
        n_samples: Number of frames to sample
        seed: Random seed (for consistency, not randomness)

    Returns:
        Sampled frames of shape [n_samples, D]
    """
    T = sequence.shape[0]
    if n_samples >= T:
        return sequence.copy()

    # Compute L2 norm per frame
    energies = np.linalg.norm(sequence, axis=1)

    # Select top-k energetic frames
    top_indices = np.argsort(energies)[-n_samples:]
    top_indices = np.sort(top_indices)  # Maintain temporal order

    return sequence[top_indices]


def run_distance_choice_ablation(
    train_seqs: List[np.ndarray],
    train_labels: List[int],
    test_seqs: List[np.ndarray],
    test_labels: List[int],
    methods: List[str] = ['Uq', 'Uc'],
    metrics: List[str] = ['cosine', 'euclidean', 'fidelity', 'quantum_fidelity'],
    quantum_shots: int = 256
) -> pd.DataFrame:
    """
    Ablation: Compare different distance metrics.

    Args:
        train_seqs: Training sequences
        train_labels: Training labels
        test_seqs: Test sequences
        test_labels: Test labels
        methods: PCA methods to test
        metrics: Distance metrics to test (includes 'quantum_fidelity' for real quantum)
        quantum_shots: Number of shots for quantum measurements (default: 256, lower for speed)

    Returns:
        DataFrame with columns: exp, method, metric, accuracy, time_ms
    """
    from dtw.dtw_runner import one_nn
    import time

    logger.info("Running distance choice ablation...")

    results = []

    for method in methods:
        for metric in metrics:
            logger.info(f"  Testing {method} with {metric} metric...")

            n_correct = 0
            total_time = 0.0

            for test_seq, true_label in zip(test_seqs, test_labels):
                start_time = time.time()
                pred_label, _ = one_nn(
                    train_seqs,
                    train_labels,
                    test_seq,
                    metric=metric,
                    window=None,
                    quantum_shots=quantum_shots
                )
                elapsed = time.time() - start_time
                total_time += elapsed

                if pred_label == true_label:
                    n_correct += 1

            accuracy = n_correct / len(test_seqs)
            avg_time_ms = (total_time / len(test_seqs)) * 1000

            results.append({
                'exp': 'distance_choice',
                'method': method,
                'k': None,
                'metric': metric,
                'setting': metric,
                'accuracy': accuracy,
                'time_ms': avg_time_ms
            })

            logger.info(f"    Accuracy: {accuracy:.4f}, Time: {avg_time_ms:.1f}ms")

    return pd.DataFrame(results)


def run_k_sweep_ablation(
    sequences_by_k: Dict[int, Tuple[List[np.ndarray], List[int]]],
    test_sequences_by_k: Dict[int, Tuple[List[np.ndarray], List[int]]],
    methods: List[str] = ['Uq', 'Uc'],
    k_values: List[int] = [3, 5, 8, 10, 12, 16],
    metric: str = 'euclidean'
) -> pd.DataFrame:
    """
    Ablation: Sweep over different k values.

    Args:
        sequences_by_k: Dict mapping k -> (train_seqs, train_labels)
        test_sequences_by_k: Dict mapping k -> (test_seqs, test_labels)
        methods: PCA methods to test
        k_values: k values to sweep
        metric: Distance metric to use

    Returns:
        DataFrame with columns: exp, method, k, metric, setting, accuracy, time_ms
    """
    from dtw.dtw_runner import one_nn
    import time

    logger.info("Running k sweep ablation...")

    results = []

    for method in methods:
        for k in k_values:
            if k not in sequences_by_k:
                logger.warning(f"  k={k} not available for {method}, skipping")
                continue

            logger.info(f"  Testing {method} with k={k}...")

            train_seqs, train_labels = sequences_by_k[k]
            test_seqs, test_labels = test_sequences_by_k[k]

            n_correct = 0
            total_time = 0.0

            for test_seq, true_label in zip(test_seqs, test_labels):
                start_time = time.time()
                pred_label, _ = one_nn(
                    train_seqs,
                    train_labels,
                    test_seq,
                    metric=metric,
                    window=None
                )
                elapsed = time.time() - start_time
                total_time += elapsed

                if pred_label == true_label:
                    n_correct += 1

            accuracy = n_correct / len(test_seqs)
            avg_time_ms = (total_time / len(test_seqs)) * 1000

            results.append({
                'exp': 'k_sweep',
                'method': method,
                'k': k,
                'metric': metric,
                'setting': f'k={k}',
                'accuracy': accuracy,
                'time_ms': avg_time_ms
            })

            logger.info(f"    Accuracy: {accuracy:.4f}, Time: {avg_time_ms:.1f}ms")

    return pd.DataFrame(results)


def run_sampling_strategy_ablation(
    original_seqs: List[np.ndarray],
    labels: List[int],
    test_seqs: List[np.ndarray],
    test_labels: List[int],
    n_samples: int = 50,
    metric: str = 'euclidean'
) -> pd.DataFrame:
    """
    Ablation: Compare uniform vs energy-based sampling.

    Args:
        original_seqs: Full sequences for sampling
        labels: Training labels
        test_seqs: Test sequences (also to be sampled)
        test_labels: Test labels
        n_samples: Number of frames to sample
        metric: Distance metric to use

    Returns:
        DataFrame with columns: exp, method, k, metric, setting, accuracy, time_ms
    """
    from dtw.dtw_runner import one_nn
    import time

    logger.info("Running sampling strategy ablation...")

    strategies = {
        'uniform': sample_frames_uniform,
        'energy': sample_frames_energy
    }

    results = []

    for strategy_name, sample_func in strategies.items():
        logger.info(f"  Testing {strategy_name} sampling...")

        # Sample training sequences
        train_sampled = [
            sample_func(seq, n_samples, seed=42)
            for seq in original_seqs
        ]

        # Sample test sequences
        test_sampled = [
            sample_func(seq, n_samples, seed=42)
            for seq in test_seqs
        ]

        n_correct = 0
        total_time = 0.0

        for test_seq, true_label in zip(test_sampled, test_labels):
            start_time = time.time()
            pred_label, _ = one_nn(
                train_sampled,
                labels,
                test_seq,
                metric=metric,
                window=None
            )
            elapsed = time.time() - start_time
            total_time += elapsed

            if pred_label == true_label:
                n_correct += 1

        accuracy = n_correct / len(test_sampled)
        avg_time_ms = (total_time / len(test_sampled)) * 1000

        results.append({
            'exp': 'sampling_strategy',
            'method': None,
            'k': None,
            'metric': metric,
            'setting': strategy_name,
            'accuracy': accuracy,
            'time_ms': avg_time_ms
        })

        logger.info(f"    Accuracy: {accuracy:.4f}, Time: {avg_time_ms:.1f}ms")

    return pd.DataFrame(results)


def run_robustness_ablation(
    train_seqs: List[np.ndarray],
    train_labels: List[int],
    test_seqs: List[np.ndarray],
    test_labels: List[int],
    noise_sigmas: List[float] = [0.0, 0.01, 0.02],
    jitter_rates: List[float] = [0.0, 0.05],
    metric: str = 'euclidean'
) -> pd.DataFrame:
    """
    Ablation: Test robustness to noise and temporal jitter.

    Args:
        train_seqs: Training sequences
        train_labels: Training labels
        test_seqs: Test sequences (to be perturbed)
        test_labels: Test labels
        noise_sigmas: Gaussian noise levels to test
        jitter_rates: Frame drop rates to test
        metric: Distance metric to use

    Returns:
        DataFrame with columns: exp, method, k, metric, setting, accuracy, time_ms
    """
    from dtw.dtw_runner import one_nn
    import time

    logger.info("Running robustness ablation...")

    results = []

    # Test noise robustness
    for sigma in noise_sigmas:
        logger.info(f"  Testing noise sigma={sigma}...")

        # Add noise to test sequences
        test_noisy = [
            add_joint_noise(seq, sigma=sigma, seed=42)
            for seq in test_seqs
        ]

        n_correct = 0
        total_time = 0.0

        for test_seq, true_label in zip(test_noisy, test_labels):
            start_time = time.time()
            pred_label, _ = one_nn(
                train_seqs,
                train_labels,
                test_seq,
                metric=metric,
                window=None
            )
            elapsed = time.time() - start_time
            total_time += elapsed

            if pred_label == true_label:
                n_correct += 1

        accuracy = n_correct / len(test_noisy)
        avg_time_ms = (total_time / len(test_noisy)) * 1000

        results.append({
            'exp': 'robustness_noise',
            'method': None,
            'k': None,
            'metric': metric,
            'setting': f'sigma={sigma}',
            'accuracy': accuracy,
            'time_ms': avg_time_ms
        })

        logger.info(f"    Accuracy: {accuracy:.4f}, Time: {avg_time_ms:.1f}ms")

    # Test temporal jitter robustness
    for jitter_rate in jitter_rates:
        logger.info(f"  Testing jitter rate={jitter_rate}...")

        # Add jitter to test sequences
        test_jittered = [
            add_temporal_jitter(seq, drop_rate=jitter_rate, seed=42)
            for seq in test_seqs
        ]

        n_correct = 0
        total_time = 0.0

        for test_seq, true_label in zip(test_jittered, test_labels):
            start_time = time.time()
            pred_label, _ = one_nn(
                train_seqs,
                train_labels,
                test_seq,
                metric=metric,
                window=None
            )
            elapsed = time.time() - start_time
            total_time += elapsed

            if pred_label == true_label:
                n_correct += 1

        accuracy = n_correct / len(test_jittered)
        avg_time_ms = (total_time / len(test_jittered)) * 1000

        results.append({
            'exp': 'robustness_jitter',
            'method': None,
            'k': None,
            'metric': metric,
            'setting': f'drop={jitter_rate}',
            'accuracy': accuracy,
            'time_ms': avg_time_ms
        })

        logger.info(f"    Accuracy: {accuracy:.4f}, Time: {avg_time_ms:.1f}ms")

    return pd.DataFrame(results)


def plot_distance_choice_ablation(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300
) -> None:
    """
    Plot distance choice ablation results.

    Args:
        df: Results DataFrame
        output_path: Output path for figure
        dpi: Figure resolution
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    methods = df['method'].unique()
    metrics = df['metric'].unique()
    x = np.arange(len(metrics))
    width = 0.35

    # Accuracy comparison
    for i, method in enumerate(methods):
        method_data = df[df['method'] == method]
        accuracies = [
            method_data[method_data['metric'] == m]['accuracy'].values[0]
            for m in metrics
        ]
        ax1.bar(x + i * width, accuracies, width, label=method)

    ax1.set_xlabel('Distance Metric', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Distance Metric Comparison: Accuracy', fontweight='bold')
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Time comparison
    for i, method in enumerate(methods):
        method_data = df[df['method'] == method]
        times = [
            method_data[method_data['metric'] == m]['time_ms'].values[0]
            for m in metrics
        ]
        ax2.bar(x + i * width, times, width, label=method)

    ax2.set_xlabel('Distance Metric', fontweight='bold')
    ax2.set_ylabel('Query Time (ms)', fontweight='bold')
    ax2.set_title('Distance Metric Comparison: Speed', fontweight='bold')
    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved distance choice plot: {output_path}")


def plot_k_sweep_ablation(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300
) -> None:
    """
    Plot k sweep ablation results.

    Args:
        df: Results DataFrame
        output_path: Output path for figure
        dpi: Figure resolution
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    methods = df['method'].unique()
    colors = {'Uq': '#E63946', 'Uc': '#457B9D'}

    # Accuracy vs k
    for method in methods:
        method_data = df[df['method'] == method].sort_values('k')
        ax1.plot(
            method_data['k'],
            method_data['accuracy'],
            marker='o',
            label=method,
            color=colors.get(method, 'gray'),
            linewidth=2.5,
            markersize=9
        )

    ax1.set_xlabel('k (number of principal components)', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('k Sweep: Accuracy vs Dimensionality', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Time vs k
    for method in methods:
        method_data = df[df['method'] == method].sort_values('k')
        ax2.plot(
            method_data['k'],
            method_data['time_ms'],
            marker='s',
            label=method,
            color=colors.get(method, 'gray'),
            linewidth=2.5,
            markersize=9
        )

    ax2.set_xlabel('k (number of principal components)', fontweight='bold')
    ax2.set_ylabel('Query Time (ms)', fontweight='bold')
    ax2.set_title('k Sweep: Time vs Dimensionality', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved k sweep plot: {output_path}")


def plot_sampling_strategy_ablation(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300
) -> None:
    """
    Plot sampling strategy ablation results.

    Args:
        df: Results DataFrame
        output_path: Output path for figure
        dpi: Figure resolution
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    strategies = df['setting'].values
    accuracies = df['accuracy'].values
    times = df['time_ms'].values

    # Accuracy comparison
    bars1 = ax1.bar(strategies, accuracies, color=['#2A9D8F', '#E76F51'])
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Sampling Strategy: Accuracy', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    # Time comparison
    bars2 = ax2.bar(strategies, times, color=['#2A9D8F', '#E76F51'])
    ax2.set_ylabel('Query Time (ms)', fontweight='bold')
    ax2.set_title('Sampling Strategy: Speed', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.1f}ms',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved sampling strategy plot: {output_path}")


def plot_robustness_ablation(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300
) -> None:
    """
    Plot robustness ablation results.

    Args:
        df: Results DataFrame
        output_path: Output path for figure
        dpi: Figure resolution
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Noise robustness
    noise_df = df[df['exp'] == 'robustness_noise']
    if not noise_df.empty:
        settings = noise_df['setting'].values
        accuracies = noise_df['accuracy'].values

        ax1.plot(
            range(len(settings)),
            accuracies,
            marker='o',
            color='#E63946',
            linewidth=2.5,
            markersize=9
        )
        ax1.set_xticks(range(len(settings)))
        ax1.set_xticklabels(settings, rotation=0)
        ax1.set_xlabel('Noise Level (σ)', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Robustness to Gaussian Noise', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for i, acc in enumerate(accuracies):
            ax1.text(i, acc, f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

    # Jitter robustness
    jitter_df = df[df['exp'] == 'robustness_jitter']
    if not jitter_df.empty:
        settings = jitter_df['setting'].values
        accuracies = jitter_df['accuracy'].values

        ax2.plot(
            range(len(settings)),
            accuracies,
            marker='s',
            color='#457B9D',
            linewidth=2.5,
            markersize=9
        )
        ax2.set_xticks(range(len(settings)))
        ax2.set_xticklabels(settings, rotation=0)
        ax2.set_xlabel('Frame Drop Rate', fontweight='bold')
        ax2.set_ylabel('Accuracy', fontweight='bold')
        ax2.set_title('Robustness to Temporal Jitter', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for i, acc in enumerate(accuracies):
            ax2.text(i, acc, f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved robustness plot: {output_path}")


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test utility functions
    print("Testing ablation utilities...")

    # Test temporal jitter
    seq = np.random.randn(100, 5)
    jittered = add_temporal_jitter(seq, drop_rate=0.05, seed=42)
    print(f"Original shape: {seq.shape}, Jittered shape: {jittered.shape}")

    # Test noise
    noisy = add_joint_noise(seq, sigma=0.01, seed=42)
    print(f"Noise RMS: {np.sqrt(np.mean((noisy - seq) ** 2)):.4f}")

    # Test sampling
    uniform = sample_frames_uniform(seq, n_samples=20, seed=42)
    energy = sample_frames_energy(seq, n_samples=20, seed=42)
    print(f"Uniform shape: {uniform.shape}, Energy shape: {energy.shape}")

    print("\nAblation utilities ready!")
