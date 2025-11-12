"""
Evaluation module for aggregating and analyzing QDTW metrics.
"""

from eval.aggregate import load_and_merge_metrics
from eval.plotting import (
    plot_accuracy_vs_k,
    plot_pareto,
    plot_time_vs_k,
)

__all__ = [
    'load_and_merge_metrics',
    'plot_accuracy_vs_k',
    'plot_time_vs_k',
    'plot_pareto',
]
