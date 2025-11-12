"""
Aggregate metrics from baseline and subspace evaluations.

This module loads and merges CSV files containing DTW evaluation results
from baseline (full 60D), quantum PCA (Uq), and classical PCA (Uc) methods.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_csv_safely(
    filepath: Path,
    method_name: str
) -> Optional[pd.DataFrame]:
    """
    Load a CSV file with error handling.

    Args:
        filepath: Path to CSV file
        method_name: Name of method for logging

    Returns:
        DataFrame if successful, None if file not found
    """
    if not filepath.exists():
        logger.warning(f"{method_name} metrics not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {method_name}: {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def load_and_merge_metrics(
    results_dir: Path = Path('results')
) -> Dict[str, pd.DataFrame]:
    """
    Load and merge all metrics CSV files.

    Args:
        results_dir: Directory containing metrics CSV files

    Returns:
        Dictionary mapping method names to DataFrames:
        - 'baseline': Full 60D DTW results
        - 'Uq': Quantum PCA subspace results
        - 'Uc': Classical PCA subspace results
        - 'all_subspace': Combined Uq + Uc with 'method' column

    Expected CSV columns:
        - baseline: metric, accuracy, time_ms
        - Uq/Uc: k, metric, accuracy, time_ms
    """
    metrics = {}

    # Load baseline (full 60D)
    baseline_path = results_dir / 'metrics_baseline.csv'
    baseline_df = load_csv_safely(baseline_path, 'Baseline')
    if baseline_df is not None:
        metrics['baseline'] = baseline_df

    # Load quantum PCA subspace
    uq_path = results_dir / 'metrics_subspace_Uq.csv'
    uq_df = load_csv_safely(uq_path, 'Quantum PCA (Uq)')
    if uq_df is not None:
        uq_df['method'] = 'Uq'
        metrics['Uq'] = uq_df

    # Load classical PCA subspace
    uc_path = results_dir / 'metrics_subspace_Uc.csv'
    uc_df = load_csv_safely(uc_path, 'Classical PCA (Uc)')
    if uc_df is not None:
        uc_df['method'] = 'Uc'
        metrics['Uc'] = uc_df

    # Merge subspace results
    subspace_dfs = []
    if uq_df is not None:
        subspace_dfs.append(uq_df)
    if uc_df is not None:
        subspace_dfs.append(uc_df)

    if subspace_dfs:
        all_subspace = pd.concat(subspace_dfs, ignore_index=True)
        metrics['all_subspace'] = all_subspace
        logger.info(
            f"Merged subspace data: {len(all_subspace)} rows "
            f"({len(uq_df) if uq_df is not None else 0} Uq + "
            f"{len(uc_df) if uc_df is not None else 0} Uc)"
        )

    return metrics


def get_best_results(
    df: pd.DataFrame,
    group_by: str = 'method',
    metric_col: str = 'accuracy'
) -> pd.DataFrame:
    """
    Get best results for each group.

    Args:
        df: DataFrame with metrics
        group_by: Column to group by (e.g., 'method', 'k')
        metric_col: Column to optimize (e.g., 'accuracy')

    Returns:
        DataFrame with best result per group
    """
    if df is None or df.empty:
        return pd.DataFrame()

    idx = df.groupby(group_by)[metric_col].idxmax()
    best = df.loc[idx].copy()
    return best


def get_best_k_per_method(
    subspace_df: pd.DataFrame,
    metric_name: str = 'cosine'
) -> pd.DataFrame:
    """
    Find best k value for each method for a given distance metric.

    Args:
        subspace_df: Merged subspace DataFrame with 'method' column
        metric_name: Distance metric to filter by

    Returns:
        DataFrame with best k for each method
    """
    if subspace_df is None or subspace_df.empty:
        return pd.DataFrame()

    # Filter by metric
    df_metric = subspace_df[subspace_df['metric'] == metric_name].copy()

    if df_metric.empty:
        logger.warning(f"No results found for metric '{metric_name}'")
        return pd.DataFrame()

    # Find best k per method
    best_k = df_metric.loc[df_metric.groupby('method')['accuracy'].idxmax()]

    return best_k[['method', 'k', 'accuracy', 'time_ms']].reset_index(drop=True)


def summarize_metrics(metrics: Dict[str, pd.DataFrame]) -> str:
    """
    Generate a text summary of metrics.

    Args:
        metrics: Dictionary of DataFrames from load_and_merge_metrics()

    Returns:
        Formatted summary string
    """
    lines = ["=" * 70, "METRICS SUMMARY", "=" * 70]

    # Baseline summary
    if 'baseline' in metrics:
        baseline = metrics['baseline']
        lines.append("\nBaseline (60D full space):")
        for _, row in baseline.iterrows():
            lines.append(
                f"  {row['metric']:12s}: "
                f"accuracy={row['accuracy']:.4f}, "
                f"time={row['time_ms']:7.1f}ms"
            )

    # Subspace summary
    if 'all_subspace' in metrics:
        subspace = metrics['all_subspace']
        lines.append("\nSubspace (PCA projections):")

        for method in ['Uq', 'Uc']:
            method_data = subspace[subspace['method'] == method]
            if method_data.empty:
                continue

            lines.append(f"\n  {method}:")
            for k in sorted(method_data['k'].unique()):
                k_data = method_data[method_data['k'] == k]
                lines.append(f"    k={k}:")
                for _, row in k_data.iterrows():
                    lines.append(
                        f"      {row['metric']:12s}: "
                        f"acc={row['accuracy']:.4f}, "
                        f"time={row['time_ms']:6.1f}ms"
                    )

    lines.append("=" * 70)
    return "\n".join(lines)


def create_sample_metrics(output_dir: Path = Path('results')):
    """
    Create sample metrics CSV files for testing.

    This function generates synthetic but realistic metrics data
    for demonstration purposes when real evaluation hasn't been run yet.

    Args:
        output_dir: Directory to save CSV files
    """
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample baseline metrics (60D full space)
    baseline_data = {
        'metric': ['cosine', 'euclidean', 'fidelity'],
        'accuracy': [0.7876, 0.8319, 0.7965],
        'time_ms': [4250.5, 1890.3, 3980.2]
    }
    baseline_df = pd.DataFrame(baseline_data)
    baseline_path = output_dir / 'metrics_baseline.csv'
    baseline_df.to_csv(baseline_path, index=False)
    logger.info(f"Created sample baseline metrics: {baseline_path}")

    # Sample quantum PCA metrics
    np.random.seed(42)
    uq_data = []
    for k in [5, 8, 10]:
        for metric in ['cosine', 'euclidean', 'fidelity']:
            # Accuracy decreases slightly with lower k
            # but not too much (PCA preserves info)
            base_acc = 0.78 if metric == 'cosine' else 0.82 if metric == 'euclidean' else 0.79
            acc = base_acc - (10 - k) * 0.01 + np.random.randn() * 0.005

            # Time decreases with lower k
            base_time = 2400 if metric == 'cosine' else 900 if metric == 'euclidean' else 2100
            time_ms = base_time * (k / 10) + np.random.randn() * 50

            uq_data.append({
                'k': k,
                'metric': metric,
                'accuracy': max(0.0, min(1.0, acc)),
                'time_ms': max(100, time_ms)
            })

    uq_df = pd.DataFrame(uq_data)
    uq_path = output_dir / 'metrics_subspace_Uq.csv'
    uq_df.to_csv(uq_path, index=False)
    logger.info(f"Created sample Uq metrics: {uq_path}")

    # Sample classical PCA metrics (slightly better accuracy, similar time)
    uc_data = []
    for k in [5, 8, 10]:
        for metric in ['cosine', 'euclidean', 'fidelity']:
            base_acc = 0.79 if metric == 'cosine' else 0.83 if metric == 'euclidean' else 0.80
            acc = base_acc - (10 - k) * 0.008 + np.random.randn() * 0.005

            base_time = 2400 if metric == 'cosine' else 900 if metric == 'euclidean' else 2100
            time_ms = base_time * (k / 10) + np.random.randn() * 50

            uc_data.append({
                'k': k,
                'metric': metric,
                'accuracy': max(0.0, min(1.0, acc)),
                'time_ms': max(100, time_ms)
            })

    uc_df = pd.DataFrame(uc_data)
    uc_path = output_dir / 'metrics_subspace_Uc.csv'
    uc_df.to_csv(uc_path, index=False)
    logger.info(f"Created sample Uc metrics: {uc_path}")

    return baseline_df, uq_df, uc_df


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create sample data
    print("Creating sample metrics...")
    create_sample_metrics()

    # Load and display
    print("\nLoading metrics...")
    metrics = load_and_merge_metrics()

    print("\n" + summarize_metrics(metrics))

    # Show best k per method
    if 'all_subspace' in metrics:
        print("\nBest k per method (cosine metric):")
        best_k = get_best_k_per_method(metrics['all_subspace'], 'cosine')
        print(best_k.to_string(index=False))
