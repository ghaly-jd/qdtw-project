"""
Tests for evaluation aggregation module.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from eval.aggregate import (
    create_sample_metrics,
    get_best_k_per_method,
    get_best_results,
    load_and_merge_metrics,
    load_csv_safely,
    summarize_metrics,
)


class TestLoadCSVSafely:
    """Test CSV loading with error handling."""

    def test_load_existing_csv(self, tmp_path):
        """Test loading an existing CSV file."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.to_csv(csv_path, index=False)

        loaded = load_csv_safely(csv_path, "test")
        assert loaded is not None
        assert len(loaded) == 3
        assert list(loaded.columns) == ['a', 'b']

    def test_load_nonexistent_csv(self, tmp_path):
        """Test loading a non-existent CSV file."""
        csv_path = tmp_path / "nonexistent.csv"
        loaded = load_csv_safely(csv_path, "test")
        assert loaded is None


class TestLoadAndMergeMetrics:
    """Test loading and merging metrics from CSV files."""

    def test_load_all_metrics(self, tmp_path):
        """Test loading all three metric files."""
        # Create sample CSVs
        baseline_df = pd.DataFrame({
            'metric': ['cosine', 'euclidean'],
            'accuracy': [0.8, 0.85],
            'time_ms': [1000, 500]
        })
        baseline_df.to_csv(tmp_path / 'metrics_baseline.csv', index=False)

        uq_df = pd.DataFrame({
            'k': [5, 8],
            'metric': ['cosine', 'cosine'],
            'accuracy': [0.75, 0.78],
            'time_ms': [800, 900]
        })
        uq_df.to_csv(tmp_path / 'metrics_subspace_Uq.csv', index=False)

        uc_df = pd.DataFrame({
            'k': [5, 8],
            'metric': ['cosine', 'cosine'],
            'accuracy': [0.76, 0.79],
            'time_ms': [800, 900]
        })
        uc_df.to_csv(tmp_path / 'metrics_subspace_Uc.csv', index=False)

        # Load metrics
        metrics = load_and_merge_metrics(tmp_path)

        assert 'baseline' in metrics
        assert 'Uq' in metrics
        assert 'Uc' in metrics
        assert 'all_subspace' in metrics

        assert len(metrics['baseline']) == 2
        assert len(metrics['Uq']) == 2
        assert len(metrics['Uc']) == 2
        assert len(metrics['all_subspace']) == 4  # 2 Uq + 2 Uc

    def test_load_partial_metrics(self, tmp_path):
        """Test loading when some files are missing."""
        # Only create Uq file
        uq_df = pd.DataFrame({
            'k': [5],
            'metric': ['cosine'],
            'accuracy': [0.75],
            'time_ms': [800]
        })
        uq_df.to_csv(tmp_path / 'metrics_subspace_Uq.csv', index=False)

        metrics = load_and_merge_metrics(tmp_path)

        assert 'Uq' in metrics
        assert 'baseline' not in metrics
        assert 'Uc' not in metrics

    def test_method_column_added(self, tmp_path):
        """Test that 'method' column is added to subspace DataFrames."""
        uq_df = pd.DataFrame({
            'k': [5],
            'metric': ['cosine'],
            'accuracy': [0.75],
            'time_ms': [800]
        })
        uq_df.to_csv(tmp_path / 'metrics_subspace_Uq.csv', index=False)

        uc_df = pd.DataFrame({
            'k': [5],
            'metric': ['cosine'],
            'accuracy': [0.76],
            'time_ms': [800]
        })
        uc_df.to_csv(tmp_path / 'metrics_subspace_Uc.csv', index=False)

        metrics = load_and_merge_metrics(tmp_path)

        assert 'method' in metrics['Uq'].columns
        assert 'method' in metrics['Uc'].columns
        assert metrics['Uq']['method'].iloc[0] == 'Uq'
        assert metrics['Uc']['method'].iloc[0] == 'Uc'


class TestGetBestResults:
    """Test finding best results."""

    def test_get_best_by_method(self):
        """Test getting best result per method."""
        df = pd.DataFrame({
            'method': ['Uq', 'Uq', 'Uc', 'Uc'],
            'k': [5, 8, 5, 8],
            'accuracy': [0.75, 0.80, 0.76, 0.82]
        })

        best = get_best_results(df, group_by='method', metric_col='accuracy')

        assert len(best) == 2
        assert best[best['method'] == 'Uq']['k'].iloc[0] == 8
        assert best[best['method'] == 'Uc']['k'].iloc[0] == 8

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        best = get_best_results(df)
        assert best.empty


class TestGetBestKPerMethod:
    """Test finding best k value per method."""

    def test_best_k_cosine(self):
        """Test finding best k for cosine metric."""
        df = pd.DataFrame({
            'method': ['Uq', 'Uq', 'Uc', 'Uc'],
            'k': [5, 8, 5, 8],
            'metric': ['cosine', 'cosine', 'cosine', 'cosine'],
            'accuracy': [0.75, 0.80, 0.76, 0.82],
            'time_ms': [800, 900, 800, 900]
        })

        best_k = get_best_k_per_method(df, 'cosine')

        assert len(best_k) == 2
        assert best_k[best_k['method'] == 'Uq']['k'].iloc[0] == 8
        assert best_k[best_k['method'] == 'Uc']['k'].iloc[0] == 8

    def test_empty_for_missing_metric(self):
        """Test with metric that doesn't exist."""
        df = pd.DataFrame({
            'method': ['Uq'],
            'k': [5],
            'metric': ['cosine'],
            'accuracy': [0.75],
            'time_ms': [800]
        })

        best_k = get_best_k_per_method(df, 'nonexistent')
        assert best_k.empty


class TestSummarizeMetrics:
    """Test metrics summary generation."""

    def test_summary_with_all_data(self):
        """Test summary with complete metrics."""
        baseline = pd.DataFrame({
            'metric': ['cosine'],
            'accuracy': [0.80],
            'time_ms': [1000]
        })

        subspace = pd.DataFrame({
            'method': ['Uq', 'Uc'],
            'k': [5, 5],
            'metric': ['cosine', 'cosine'],
            'accuracy': [0.75, 0.76],
            'time_ms': [800, 800]
        })

        metrics = {
            'baseline': baseline,
            'all_subspace': subspace
        }

        summary = summarize_metrics(metrics)

        assert 'METRICS SUMMARY' in summary
        assert 'Baseline' in summary
        assert 'Subspace' in summary
        assert 'cosine' in summary
        assert '0.80' in summary  # baseline accuracy
        assert '0.75' in summary  # Uq accuracy

    def test_summary_with_empty_dict(self):
        """Test summary with empty metrics."""
        summary = summarize_metrics({})
        assert 'METRICS SUMMARY' in summary


class TestCreateSampleMetrics:
    """Test sample metrics generation."""

    def test_creates_all_files(self, tmp_path):
        """Test that all sample files are created."""
        baseline, uq, uc = create_sample_metrics(tmp_path)

        assert (tmp_path / 'metrics_baseline.csv').exists()
        assert (tmp_path / 'metrics_subspace_Uq.csv').exists()
        assert (tmp_path / 'metrics_subspace_Uc.csv').exists()

    def test_baseline_has_correct_structure(self, tmp_path):
        """Test baseline metrics structure."""
        baseline, _, _ = create_sample_metrics(tmp_path)

        assert 'metric' in baseline.columns
        assert 'accuracy' in baseline.columns
        assert 'time_ms' in baseline.columns
        assert len(baseline) == 3  # cosine, euclidean, fidelity

    def test_subspace_has_correct_structure(self, tmp_path):
        """Test subspace metrics structure."""
        _, uq, uc = create_sample_metrics(tmp_path)

        for df in [uq, uc]:
            assert 'k' in df.columns
            assert 'metric' in df.columns
            assert 'accuracy' in df.columns
            assert 'time_ms' in df.columns
            assert len(df) == 9  # 3 k values × 3 metrics

    def test_accuracy_in_valid_range(self, tmp_path):
        """Test that accuracies are in [0, 1]."""
        baseline, uq, uc = create_sample_metrics(tmp_path)

        for df in [baseline, uq, uc]:
            assert (df['accuracy'] >= 0).all()
            assert (df['accuracy'] <= 1).all()

    def test_time_is_positive(self, tmp_path):
        """Test that times are positive."""
        baseline, uq, uc = create_sample_metrics(tmp_path)

        for df in [baseline, uq, uc]:
            assert (df['time_ms'] > 0).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
