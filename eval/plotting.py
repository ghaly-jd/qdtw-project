"""
Plotting utilities for QDTW evaluation results.

This module provides functions to visualize accuracy vs k, time vs k,
and Pareto frontiers comparing quantum PCA, classical PCA, and baseline.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Plot style configuration
STYLE_CONFIG = {
    'figure.figsize': (10, 6),
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'lines.markersize': 8,
}

# Color scheme
COLORS = {
    'Uq': '#E63946',      # Red - Quantum PCA
    'Uc': '#457B9D',      # Blue - Classical PCA
    'baseline': '#2A9D8F'  # Teal - Baseline
}

MARKERS = {
    'cosine': 'o',
    'euclidean': 's',
    'fidelity': '^'
}


def setup_plot_style():
    """Apply consistent plot styling."""
    plt.style.use('seaborn-v0_8-darkgrid')
    for key, value in STYLE_CONFIG.items():
        plt.rcParams[key] = value


def plot_accuracy_vs_k(
    uq_df: pd.DataFrame,
    uc_df: pd.DataFrame,
    output_path: Path = Path('figures/accuracy_vs_k.png'),
    dpi: int = 300
) -> None:
    """
    Plot accuracy vs k for quantum and classical PCA.

    Args:
        uq_df: Quantum PCA metrics DataFrame
        uc_df: Classical PCA metrics DataFrame
        output_path: Path to save figure
        dpi: Figure resolution
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['cosine', 'euclidean', 'fidelity']
    metric_titles = ['Cosine Distance', 'Euclidean Distance', 'Fidelity Distance']

    for ax, metric, title in zip(axes, metrics, metric_titles):
        # Filter by metric
        uq_metric = uq_df[uq_df['metric'] == metric].sort_values('k')
        uc_metric = uc_df[uc_df['metric'] == metric].sort_values('k')

        # Plot Uq
        if not uq_metric.empty:
            ax.plot(
                uq_metric['k'],
                uq_metric['accuracy'],
                marker='o',
                color=COLORS['Uq'],
                label='Quantum PCA (Uq)',
                linewidth=2.5,
                markersize=9
            )

        # Plot Uc
        if not uc_metric.empty:
            ax.plot(
                uc_metric['k'],
                uc_metric['accuracy'],
                marker='s',
                color=COLORS['Uc'],
                label='Classical PCA (Uc)',
                linewidth=2.5,
                markersize=9
            )

        ax.set_xlabel('k (number of principal components)', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=10)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # Add value labels
        for df, color in [(uq_metric, COLORS['Uq']), (uc_metric, COLORS['Uc'])]:
            if not df.empty:
                for _, row in df.iterrows():
                    ax.annotate(
                        f'{row["accuracy"]:.3f}',
                        xy=(row['k'], row['accuracy']),
                        xytext=(0, 8),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8,
                        color=color,
                        weight='bold'
                    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved accuracy vs k plot: {output_path}")


def plot_time_vs_k(
    uq_df: pd.DataFrame,
    uc_df: pd.DataFrame,
    output_path: Path = Path('figures/time_vs_k.png'),
    dpi: int = 300
) -> None:
    """
    Plot average query time vs k for quantum and classical PCA.

    Args:
        uq_df: Quantum PCA metrics DataFrame
        uc_df: Classical PCA metrics DataFrame
        output_path: Path to save figure
        dpi: Figure resolution
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['cosine', 'euclidean', 'fidelity']
    metric_titles = ['Cosine Distance', 'Euclidean Distance', 'Fidelity Distance']

    for ax, metric, title in zip(axes, metrics, metric_titles):
        # Filter by metric
        uq_metric = uq_df[uq_df['metric'] == metric].sort_values('k')
        uc_metric = uc_df[uc_df['metric'] == metric].sort_values('k')

        # Plot Uq
        if not uq_metric.empty:
            ax.plot(
                uq_metric['k'],
                uq_metric['time_ms'],
                marker='o',
                color=COLORS['Uq'],
                label='Quantum PCA (Uq)',
                linewidth=2.5,
                markersize=9
            )

        # Plot Uc
        if not uc_metric.empty:
            ax.plot(
                uc_metric['k'],
                uc_metric['time_ms'],
                marker='s',
                color=COLORS['Uc'],
                label='Classical PCA (Uc)',
                linewidth=2.5,
                markersize=9
            )

        ax.set_xlabel('k (number of principal components)', fontweight='bold')
        ax.set_ylabel('Average Query Time (ms)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=10)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Add value labels
        for df, color in [(uq_metric, COLORS['Uq']), (uc_metric, COLORS['Uc'])]:
            if not df.empty:
                for _, row in df.iterrows():
                    ax.annotate(
                        f'{row["time_ms"]:.0f}ms',
                        xy=(row['k'], row['time_ms']),
                        xytext=(0, 8),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8,
                        color=color,
                        weight='bold'
                    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved time vs k plot: {output_path}")


def plot_pareto(
    uq_df: pd.DataFrame,
    uc_df: pd.DataFrame,
    baseline_df: Optional[pd.DataFrame] = None,
    output_path: Path = Path('figures/pareto_accuracy_time.png'),
    dpi: int = 300
) -> None:
    """
    Plot Pareto frontier of accuracy vs time.

    Shows trade-off between accuracy and computational cost for
    different methods and configurations.

    Args:
        uq_df: Quantum PCA metrics DataFrame
        uc_df: Classical PCA metrics DataFrame
        baseline_df: Baseline (60D) metrics DataFrame
        output_path: Path to save figure
        dpi: Figure resolution
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot Uq points
    if not uq_df.empty:
        for metric in uq_df['metric'].unique():
            df_metric = uq_df[uq_df['metric'] == metric]
            ax.scatter(
                df_metric['time_ms'],
                df_metric['accuracy'],
                marker=MARKERS.get(metric, 'o'),
                s=150,
                color=COLORS['Uq'],
                alpha=0.7,
                label=f'Uq ({metric})' if metric == uq_df['metric'].iloc[0] else '',
                edgecolors='black',
                linewidths=1.5
            )

            # Add k labels
            for _, row in df_metric.iterrows():
                ax.annotate(
                    f"k={row['k']}",
                    xy=(row['time_ms'], row['accuracy']),
                    xytext=(8, 0),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8
                )

    # Plot Uc points
    if not uc_df.empty:
        for metric in uc_df['metric'].unique():
            df_metric = uc_df[uc_df['metric'] == metric]
            ax.scatter(
                df_metric['time_ms'],
                df_metric['accuracy'],
                marker=MARKERS.get(metric, 's'),
                s=150,
                color=COLORS['Uc'],
                alpha=0.7,
                label=f'Uc ({metric})' if metric == uc_df['metric'].iloc[0] else '',
                edgecolors='black',
                linewidths=1.5
            )

            # Add k labels
            for _, row in df_metric.iterrows():
                ax.annotate(
                    f"k={row['k']}",
                    xy=(row['time_ms'], row['accuracy']),
                    xytext=(8, 0),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8
                )

    # Plot baseline points
    if baseline_df is not None and not baseline_df.empty:
        for _, row in baseline_df.iterrows():
            ax.scatter(
                row['time_ms'],
                row['accuracy'],
                marker=MARKERS.get(row['metric'], 'D'),
                s=200,
                color=COLORS['baseline'],
                alpha=0.8,
                label=f"Baseline ({row['metric']})" if _ == 0 else '',
                edgecolors='black',
                linewidths=2
            )

            # Add label
            ax.annotate(
                f"60D\n{row['metric']}",
                xy=(row['time_ms'], row['accuracy']),
                xytext=(10, -10),
                textcoords='offset points',
                fontsize=9,
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['baseline'], alpha=0.3)
            )

    ax.set_xlabel('Average Query Time (ms)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=13)
    ax.set_title(
        'Pareto Frontier: Accuracy vs Time Trade-off',
        fontweight='bold',
        fontsize=15,
        pad=15
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['Uq'],
               markersize=10, label='Quantum PCA (Uq)', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['Uc'],
               markersize=10, label='Classical PCA (Uc)', markeredgecolor='black'),
    ]
    if baseline_df is not None and not baseline_df.empty:
        legend_elements.append(
            Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['baseline'],
                   markersize=10, label='Baseline (60D)', markeredgecolor='black')
        )

    legend_elements.extend([
        Line2D([0], [0], marker='o', color='gray', markersize=8,
               label='Cosine', linestyle='None'),
        Line2D([0], [0], marker='s', color='gray', markersize=8,
               label='Euclidean', linestyle='None'),
        Line2D([0], [0], marker='^', color='gray', markersize=8,
               label='Fidelity', linestyle='None'),
    ])

    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95, fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Pareto plot: {output_path}")


def create_all_plots(
    uq_df: pd.DataFrame,
    uc_df: pd.DataFrame,
    baseline_df: Optional[pd.DataFrame] = None,
    output_dir: Path = Path('figures'),
    dpi: int = 300
) -> None:
    """
    Create all plots at once.

    Args:
        uq_df: Quantum PCA metrics DataFrame
        uc_df: Classical PCA metrics DataFrame
        baseline_df: Baseline metrics DataFrame
        output_dir: Directory to save figures
        dpi: Figure resolution
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating accuracy vs k plot...")
    plot_accuracy_vs_k(uq_df, uc_df, output_dir / 'accuracy_vs_k.png', dpi)

    logger.info("Creating time vs k plot...")
    plot_time_vs_k(uq_df, uc_df, output_dir / 'time_vs_k.png', dpi)

    logger.info("Creating Pareto frontier plot...")
    plot_pareto(uq_df, uc_df, baseline_df, output_dir / 'pareto_accuracy_time.png', dpi)

    logger.info(f"All plots saved to {output_dir}")


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create sample data for testing
    from eval.aggregate import create_sample_metrics, load_and_merge_metrics

    print("Creating sample metrics...")
    create_sample_metrics()

    print("Loading metrics...")
    metrics = load_and_merge_metrics()

    print("Creating plots...")
    create_all_plots(
        uq_df=metrics.get('Uq'),
        uc_df=metrics.get('Uc'),
        baseline_df=metrics.get('baseline')
    )

    print("\nDone! Plots saved to figures/")
