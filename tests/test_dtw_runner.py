"""
Tests for dtw_runner module.

This module tests DTW distance computation and 1-NN classification
with various distance metrics: cosine, euclidean, and fidelity.
"""

import numpy as np
import pytest

from dtw.dtw_runner import (
    cosine_distance,
    dtw_distance,
    euclidean_distance,
    fidelity_distance,
    one_nn,
)


class TestDistanceMetrics:
    """Test individual distance metric functions."""

    def test_cosine_distance_identical(self):
        """Test cosine distance between identical vectors is zero."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        dist = cosine_distance(a, b)
        assert abs(dist) < 1e-10

    def test_cosine_distance_orthogonal(self):
        """Test cosine distance between orthogonal vectors is 1."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        dist = cosine_distance(a, b)
        assert abs(dist - 1.0) < 1e-10

    def test_cosine_distance_opposite(self):
        """Test cosine distance between opposite vectors is 2."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        dist = cosine_distance(a, b)
        assert abs(dist - 2.0) < 1e-10

    def test_cosine_distance_scaled(self):
        """Test cosine distance is scale-invariant."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])  # 2x scaled
        dist = cosine_distance(a, b)
        assert abs(dist) < 1e-10

    def test_euclidean_distance_identical(self):
        """Test euclidean distance between identical vectors is zero."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        dist = euclidean_distance(a, b)
        assert abs(dist) < 1e-10

    def test_euclidean_distance_unit_offset(self):
        """Test euclidean distance for unit offset."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        dist = euclidean_distance(a, b)
        assert abs(dist - 1.0) < 1e-10

    def test_euclidean_distance_diagonal(self):
        """Test euclidean distance for diagonal."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 1.0, 1.0])
        dist = euclidean_distance(a, b)
        expected = np.sqrt(3.0)
        assert abs(dist - expected) < 1e-10

    def test_fidelity_distance_identical(self):
        """Test fidelity distance between identical vectors is zero."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        dist = fidelity_distance(a, b)
        assert abs(dist) < 1e-10

    def test_fidelity_distance_orthogonal(self):
        """Test fidelity distance between orthogonal vectors is 1."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        dist = fidelity_distance(a, b)
        assert abs(dist - 1.0) < 1e-10

    def test_fidelity_distance_opposite(self):
        """Test fidelity distance treats opposite vectors as same direction."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        dist = fidelity_distance(a, b)
        # Both point along same axis, so |<â,b̂>|² = 1
        assert abs(dist) < 1e-10

    def test_fidelity_distance_45_degrees(self):
        """Test fidelity distance at 45 degrees."""
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 1.0])
        dist = fidelity_distance(a, b)
        # |<â,b̂>|² = (1/√2)² = 0.5
        expected = 1.0 - 0.5
        assert abs(dist - expected) < 1e-10

    def test_distance_metrics_zero_vector(self):
        """Test distance metrics handle zero vectors gracefully."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])

        # Euclidean should work
        dist_euc = euclidean_distance(a, b)
        assert abs(dist_euc - 1.0) < 1e-10

        # Cosine and fidelity should handle zero vectors
        # (implementation returns 0.0 for zero norm)
        dist_cos = cosine_distance(a, b)
        dist_fid = fidelity_distance(a, b)
        assert dist_cos >= 0.0
        assert dist_fid >= 0.0


class TestDTWDistance:
    """Test DTW distance computation."""

    def test_dtw_identical_sequences(self):
        """Test DTW distance between identical sequences is zero."""
        seqA = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        seqB = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        for metric in ['cosine', 'euclidean', 'fidelity']:
            dist = dtw_distance(seqA, seqB, metric=metric)
            assert abs(dist) < 1e-8, f"Failed for {metric}"

    def test_dtw_single_frame_sequences(self):
        """Test DTW distance for single-frame sequences."""
        seqA = np.array([[1.0, 2.0, 3.0]])
        seqB = np.array([[1.0, 2.0, 3.0]])

        dist = dtw_distance(seqA, seqB, metric='euclidean')
        assert abs(dist) < 1e-10

    def test_dtw_different_lengths(self):
        """Test DTW handles sequences of different lengths."""
        seqA = np.array([[1.0], [2.0], [3.0]])
        seqB = np.array([[1.0], [2.0]])

        dist = dtw_distance(seqA, seqB, metric='euclidean')
        # Distance should be positive and finite
        assert dist > 0
        assert np.isfinite(dist)

    def test_dtw_time_warped_sin_wave(self):
        """
        Test DTW with time-warped sinusoid.

        A time-warped version of a sin wave should match better
        than a completely unrelated sequence.
        """
        # Original sin wave
        t = np.linspace(0, 4 * np.pi, 50)
        sin_wave = np.sin(t).reshape(-1, 1)

        # Time-warped version (stretched middle, compressed ends)
        t_warped = np.concatenate([
            np.linspace(0, np.pi, 10),
            np.linspace(np.pi, 3 * np.pi, 30),  # Stretched
            np.linspace(3 * np.pi, 4 * np.pi, 10)
        ])
        sin_wave_warped = np.sin(t_warped).reshape(-1, 1)

        # Unrelated sequence (random)
        np.random.seed(42)
        random_seq = np.random.randn(50, 1)

        # DTW distance
        dist_warped = dtw_distance(sin_wave, sin_wave_warped, metric='euclidean')
        dist_random = dtw_distance(sin_wave, random_seq, metric='euclidean')

        # Warped sin wave should be closer than random
        assert dist_warped < dist_random, (
            f"Warped sin wave should match better: "
            f"dist_warped={dist_warped:.4f}, dist_random={dist_random:.4f}"
        )

    def test_dtw_with_window_constraint(self):
        """Test DTW with Sakoe-Chiba window constraint."""
        seqA = np.array([[i] for i in range(10)], dtype=float)
        seqB = np.array([[i] for i in range(10)], dtype=float)

        dist_no_window = dtw_distance(seqA, seqB, metric='euclidean', window=None)
        dist_with_window = dtw_distance(seqA, seqB, metric='euclidean', window=5)

        # For identical sequences, both should be zero
        assert abs(dist_no_window) < 1e-10
        assert abs(dist_with_window) < 1e-10

    def test_dtw_window_affects_distance(self):
        """Test that window constraint affects distance computation."""
        # Create sequences that would benefit from large warping
        seqA = np.array([[i] for i in [0, 1, 2, 3, 4, 5]], dtype=float)
        seqB = np.array([[i] for i in [0, 0, 1, 2, 3, 4, 5, 5]], dtype=float)

        dist_no_window = dtw_distance(seqA, seqB, metric='euclidean', window=None)
        dist_small_window = dtw_distance(seqA, seqB, metric='euclidean', window=1)

        # Small window should constrain alignment, potentially increasing distance
        assert dist_small_window >= dist_no_window

    def test_dtw_all_metrics(self):
        """Test DTW works with all supported metrics."""
        seqA = np.random.randn(10, 3)
        seqB = np.random.randn(12, 3)

        for metric in ['cosine', 'euclidean', 'fidelity']:
            dist = dtw_distance(seqA, seqB, metric=metric)
            assert dist >= 0, f"Distance should be non-negative for {metric}"
            assert np.isfinite(dist), f"Distance should be finite for {metric}"

    def test_dtw_invalid_metric(self):
        """Test DTW raises error for invalid metric."""
        seqA = np.array([[1.0]])
        seqB = np.array([[2.0]])

        with pytest.raises(ValueError, match="Unknown metric"):
            dtw_distance(seqA, seqB, metric='invalid_metric')


class TestOneNN:
    """Test 1-Nearest Neighbor classification."""

    def test_one_nn_perfect_match(self):
        """Test 1-NN finds exact match."""
        train_seqs = [
            np.array([[1.0], [2.0], [3.0]]),
            np.array([[4.0], [5.0], [6.0]]),
            np.array([[7.0], [8.0], [9.0]])
        ]
        train_labels = [0, 1, 2]
        test_seq = np.array([[4.0], [5.0], [6.0]])  # Exact match to class 1

        pred_label, pred_dist = one_nn(
            train_seqs, train_labels, test_seq, metric='euclidean'
        )

        assert pred_label == 1
        assert abs(pred_dist) < 1e-10

    def test_one_nn_closest_match(self):
        """Test 1-NN finds closest match."""
        train_seqs = [
            np.array([[0.0]]),
            np.array([[10.0]]),
            np.array([[100.0]])
        ]
        train_labels = [0, 1, 2]
        test_seq = np.array([[9.0]])  # Closest to 10.0 (class 1)

        pred_label, pred_dist = one_nn(
            train_seqs, train_labels, test_seq, metric='euclidean'
        )

        assert pred_label == 1
        assert abs(pred_dist - 1.0) < 1e-10

    def test_one_nn_different_lengths(self):
        """Test 1-NN with sequences of different lengths."""
        train_seqs = [
            np.array([[1.0], [2.0]]),
            np.array([[1.0], [2.0], [3.0]]),
            np.array([[1.0]])
        ]
        train_labels = [0, 1, 2]
        test_seq = np.array([[1.0], [2.0], [3.0]])

        pred_label, pred_dist = one_nn(
            train_seqs, train_labels, test_seq, metric='euclidean'
        )

        # Should match class 1 (identical)
        assert pred_label == 1
        assert abs(pred_dist) < 1e-10

    def test_one_nn_all_metrics(self):
        """Test 1-NN works with all metrics."""
        np.random.seed(42)
        train_seqs = [np.random.randn(10, 3) for _ in range(5)]
        train_labels = [0, 1, 2, 3, 4]
        test_seq = np.random.randn(10, 3)

        for metric in ['cosine', 'euclidean', 'fidelity']:
            pred_label, pred_dist = one_nn(
                train_seqs, train_labels, test_seq, metric=metric
            )

            assert pred_label in train_labels
            assert pred_dist >= 0
            assert np.isfinite(pred_dist)

    def test_one_nn_with_window(self):
        """Test 1-NN with DTW window constraint."""
        train_seqs = [
            np.array([[i] for i in range(10)], dtype=float),
            np.array([[i + 10] for i in range(10)], dtype=float)
        ]
        train_labels = [0, 1]
        test_seq = np.array([[i + 0.1] for i in range(10)], dtype=float)

        pred_label, pred_dist = one_nn(
            train_seqs, train_labels, test_seq, metric='euclidean', window=5
        )

        # Should match class 0 (closer values)
        assert pred_label == 0

    def test_one_nn_single_training_sample(self):
        """Test 1-NN with single training sample."""
        train_seqs = [np.array([[1.0], [2.0], [3.0]])]
        train_labels = [0]
        test_seq = np.array([[1.0], [2.0], [3.0]])

        pred_label, pred_dist = one_nn(
            train_seqs, train_labels, test_seq, metric='cosine'
        )

        assert pred_label == 0

    def test_one_nn_multiclass(self):
        """Test 1-NN with multiple classes."""
        # Create 3 clusters in 2D space
        np.random.seed(42)
        train_seqs = (
            [np.array([[0.0, 0.0]]) + np.random.randn(1, 2) * 0.1 for _ in range(3)] +
            [np.array([[5.0, 0.0]]) + np.random.randn(1, 2) * 0.1 for _ in range(3)] +
            [np.array([[0.0, 5.0]]) + np.random.randn(1, 2) * 0.1 for _ in range(3)]
        )
        train_labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]

        # Test samples near each cluster
        test_cases = [
            (np.array([[0.0, 0.0]]), 0),
            (np.array([[5.0, 0.0]]), 1),
            (np.array([[0.0, 5.0]]), 2)
        ]

        for test_seq, expected_label in test_cases:
            pred_label, _ = one_nn(
                train_seqs, train_labels, test_seq, metric='euclidean'
            )
            assert pred_label == expected_label


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_sequences(self):
        """Test handling of empty sequences."""
        # DTW should raise error for empty sequences
        seqA = np.array([]).reshape(0, 2)
        seqB = np.array([[1.0, 2.0]])

        with pytest.raises(ValueError, match="empty sequences"):
            dtw_distance(seqA, seqB, metric='euclidean')

    def test_mismatched_dimensions(self):
        """Test handling of mismatched frame dimensions."""
        seqA = np.array([[1.0, 2.0]])
        seqB = np.array([[1.0, 2.0, 3.0]])  # Different dimension

        with pytest.raises((ValueError, IndexError)):
            dtw_distance(seqA, seqB, metric='euclidean')

    def test_large_window(self):
        """Test DTW with window larger than sequence length."""
        seqA = np.array([[i] for i in range(5)], dtype=float)
        seqB = np.array([[i] for i in range(5)], dtype=float)

        # Window larger than sequence should behave like no window
        dist = dtw_distance(seqA, seqB, metric='euclidean', window=100)
        assert abs(dist) < 1e-10

    def test_negative_window(self):
        """Test DTW with negative window is treated as None."""
        seqA = np.array([[1.0], [2.0]])
        seqB = np.array([[1.0], [2.0]])

        # Should not crash
        dist = dtw_distance(seqA, seqB, metric='euclidean', window=-1)
        assert np.isfinite(dist)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
