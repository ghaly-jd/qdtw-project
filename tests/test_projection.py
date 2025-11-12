"""
Tests for sequence projection functionality.

Tests verify:
1. Shape correctness of projections
2. Norm preservation for orthonormal projections
3. Row normalization when requested
4. Batch processing
5. Edge cases
"""

import numpy as np
import pytest

from quantum.project import (
    project_sequence,
    project_sequences_batch,
    verify_projection_properties,
)


class TestProjectSequence:
    """Test project_sequence function."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        seq = np.random.randn(100, 60).astype(np.float32)
        U = np.linalg.qr(np.random.randn(60, 8))[0]

        projected = project_sequence(seq, U)

        assert projected.shape == (100, 8), \
            f"Expected shape (100, 8), got {projected.shape}"

    def test_preserves_dtype_float32(self):
        """Test that float32 dtype is preserved."""
        seq = np.random.randn(100, 60).astype(np.float32)
        U = np.random.randn(60, 8).astype(np.float64)

        projected = project_sequence(seq, U)

        assert projected.dtype == np.float32, \
            f"Expected float32, got {projected.dtype}"

    def test_preserves_dtype_float64(self):
        """Test that float64 dtype is preserved."""
        seq = np.random.randn(100, 60).astype(np.float64)
        U = np.random.randn(60, 8)

        projected = project_sequence(seq, U)

        assert projected.dtype == np.float64, \
            f"Expected float64, got {projected.dtype}"

    def test_different_k_values(self):
        """Test projection with different k values."""
        seq = np.random.randn(100, 60)

        for k in [1, 5, 8, 10, 20, 30, 60]:
            U = np.linalg.qr(np.random.randn(60, k))[0]
            projected = project_sequence(seq, U)
            assert projected.shape == (100, k), \
                f"Wrong shape for k={k}"

    def test_different_sequence_lengths(self):
        """Test projection with different sequence lengths."""
        U = np.linalg.qr(np.random.randn(60, 8))[0]

        for T in [10, 50, 100, 500]:
            seq = np.random.randn(T, 60)
            projected = project_sequence(seq, U)
            assert projected.shape == (T, 8), \
                f"Wrong shape for T={T}"

    def test_normalize_rows_option(self):
        """Test that normalize_rows creates unit vectors."""
        seq = np.random.randn(100, 60)
        U = np.linalg.qr(np.random.randn(60, 8))[0]

        projected = project_sequence(seq, U, normalize_rows=True)

        # Check that all rows have unit norm
        norms = np.linalg.norm(projected, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6), \
            f"Rows not normalized: norms range [{norms.min():.6f}, " \
            f"{norms.max():.6f}]"

    def test_normalize_rows_preserves_shape(self):
        """Test that normalization doesn't change shape."""
        seq = np.random.randn(100, 60)
        U = np.linalg.qr(np.random.randn(60, 8))[0]

        projected = project_sequence(seq, U, normalize_rows=True)

        assert projected.shape == (100, 8)

    def test_invalid_seq_shape_raises_error(self):
        """Test that invalid seq shapes raise errors."""
        U = np.random.randn(60, 8)

        # 1D array
        seq_1d = np.random.randn(60)
        with pytest.raises(ValueError, match="must be 2D"):
            project_sequence(seq_1d, U)

        # 3D array
        seq_3d = np.random.randn(10, 60, 3)
        with pytest.raises(ValueError, match="must be 2D"):
            project_sequence(seq_3d, U)

    def test_invalid_u_shape_raises_error(self):
        """Test that invalid U shapes raise errors."""
        seq = np.random.randn(100, 60)

        # 1D array
        U_1d = np.random.randn(60)
        with pytest.raises(ValueError, match="must be 2D"):
            project_sequence(seq, U_1d)

        # 3D array
        U_3d = np.random.randn(60, 8, 2)
        with pytest.raises(ValueError, match="must be 2D"):
            project_sequence(seq, U_3d)

    def test_dimension_mismatch_raises_error(self):
        """Test that dimension mismatch raises error."""
        seq = np.random.randn(100, 60)
        U = np.random.randn(50, 8)  # Wrong dimension

        with pytest.raises(ValueError, match="Incompatible dimensions"):
            project_sequence(seq, U)

    def test_invalid_k_raises_error(self):
        """Test that invalid k raises error."""
        seq = np.random.randn(100, 60)

        # k = 0
        U_zero = np.random.randn(60, 0)
        with pytest.raises(ValueError, match="k must be in range"):
            project_sequence(seq, U_zero)

        # k > D (should be caught in validation)
        # This would fail during array creation, so skip


class TestNormPreservation:
    """Test norm preservation properties."""

    def test_norm_preservation_orthonormal_u(self):
        """
        Test that projection with orthonormal U preserves or reduces norms.

        For orthonormal U, ||Z||_2 <= ||X||_2 where Z = X @ U.
        """
        seq = np.random.randn(100, 60)
        U = np.linalg.qr(np.random.randn(60, 8))[0]  # Orthonormal

        projected = project_sequence(seq, U)

        # Compute average norms
        norms_original = np.linalg.norm(seq, axis=1)
        norms_projected = np.linalg.norm(projected, axis=1)

        avg_norm_original = np.mean(norms_original)
        avg_norm_projected = np.mean(norms_projected)

        tolerance = 1e-6
        assert avg_norm_projected <= avg_norm_original + tolerance, \
            f"Norms not preserved: original={avg_norm_original:.6f}, " \
            f"projected={avg_norm_projected:.6f}"

    def test_rowwise_norm_preservation(self):
        """Test that each row's norm is preserved or reduced."""
        seq = np.random.randn(100, 60)
        U = np.linalg.qr(np.random.randn(60, 8))[0]

        projected = project_sequence(seq, U)

        norms_original = np.linalg.norm(seq, axis=1)
        norms_projected = np.linalg.norm(projected, axis=1)

        # Each projected norm should be <= original norm
        tolerance = 1e-6
        assert np.all(norms_projected <= norms_original + tolerance), \
            "Some projected norms exceed original norms"

    def test_norm_exact_for_full_rank(self):
        """Test that full-rank projection preserves norms exactly."""
        seq = np.random.randn(50, 60)
        U = np.eye(60)  # Identity = full rank projection

        projected = project_sequence(seq, U)

        # Should be identical
        np.testing.assert_allclose(projected, seq, rtol=1e-10)

    def test_norm_zero_for_zero_sequence(self):
        """Test projection of zero sequence."""
        seq = np.zeros((100, 60))
        U = np.linalg.qr(np.random.randn(60, 8))[0]

        projected = project_sequence(seq, U)

        assert np.allclose(projected, 0.0), \
            "Projection of zero should be zero"


class TestBatchProcessing:
    """Test batch projection of multiple sequences."""

    def test_batch_same_as_individual(self):
        """Test that batch processing gives same results as individual."""
        sequences = [
            np.random.randn(100, 60),
            np.random.randn(150, 60),
            np.random.randn(80, 60),
        ]
        U = np.linalg.qr(np.random.randn(60, 8))[0]

        # Batch processing
        batch_result = project_sequences_batch(sequences, U)

        # Individual processing
        individual_result = [
            project_sequence(seq, U) for seq in sequences
        ]

        # Compare
        for batch, individual in zip(batch_result, individual_result):
            np.testing.assert_allclose(batch, individual, rtol=1e-10)

    def test_batch_different_lengths(self):
        """Test batch processing with different sequence lengths."""
        sequences = [
            np.random.randn(50, 60),
            np.random.randn(100, 60),
            np.random.randn(200, 60),
        ]
        U = np.linalg.qr(np.random.randn(60, 8))[0]

        projected = project_sequences_batch(sequences, U)

        assert len(projected) == 3
        assert projected[0].shape == (50, 8)
        assert projected[1].shape == (100, 8)
        assert projected[2].shape == (200, 8)

    def test_batch_with_normalization(self):
        """Test batch processing with row normalization."""
        sequences = [
            np.random.randn(100, 60),
            np.random.randn(100, 60),
        ]
        U = np.linalg.qr(np.random.randn(60, 8))[0]

        projected = project_sequences_batch(
            sequences, U, normalize_rows=True
        )

        # Check all rows are normalized
        for seq in projected:
            norms = np.linalg.norm(seq, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-6), \
                "Not all rows normalized in batch processing"

    def test_batch_empty_list(self):
        """Test batch processing with empty list."""
        sequences = []
        U = np.linalg.qr(np.random.randn(60, 8))[0]

        projected = project_sequences_batch(sequences, U)

        assert projected == []


class TestVerifyProjectionProperties:
    """Test verification function."""

    def test_verify_correct_projection(self):
        """Test verification of correct projection."""
        seq = np.random.randn(100, 60)
        U = np.linalg.qr(np.random.randn(60, 8))[0]
        projected = project_sequence(seq, U)

        results = verify_projection_properties(seq, projected, U)

        assert results['shape_ok'] is True
        assert results['is_orthonormal'] is True
        assert bool(results['norm_preserved']) is True
        assert results['avg_norm_original'] > 0
        assert results['avg_norm_projected'] > 0

    def test_verify_shape_mismatch(self):
        """Test verification catches shape mismatch."""
        seq = np.random.randn(100, 60)
        U = np.linalg.qr(np.random.randn(60, 8))[0]
        projected = np.random.randn(99, 8)  # Wrong T

        results = verify_projection_properties(seq, projected, U)

        assert results['shape_ok'] is False

    def test_verify_non_orthonormal_u(self):
        """Test verification with non-orthonormal U."""
        seq = np.random.randn(100, 60)
        U = np.random.randn(60, 8)  # Not orthonormal
        projected = seq @ U

        results = verify_projection_properties(seq, projected, U)

        assert results['is_orthonormal'] is False
        assert results['norm_preserved'] is None  # Can't verify

    def test_verify_normalized_rows(self):
        """Test verification of normalized projection."""
        seq = np.random.randn(100, 60)
        U = np.linalg.qr(np.random.randn(60, 8))[0]
        projected = project_sequence(seq, U, normalize_rows=True)

        # Check that all rows have unit norm
        norms = np.linalg.norm(projected, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)


class TestEdgeCases:
    """Test edge cases."""

    def test_single_frame_sequence(self):
        """Test projection of single-frame sequence."""
        seq = np.random.randn(1, 60)
        U = np.linalg.qr(np.random.randn(60, 8))[0]

        projected = project_sequence(seq, U)

        assert projected.shape == (1, 8)

    def test_k_equals_1(self):
        """Test projection to 1D subspace."""
        seq = np.random.randn(100, 60)
        U = np.random.randn(60, 1)
        U = U / np.linalg.norm(U)  # Normalize

        projected = project_sequence(seq, U)

        assert projected.shape == (100, 1)

    def test_k_equals_d(self):
        """Test projection to full-dimensional space."""
        seq = np.random.randn(100, 60)
        U = np.eye(60)

        projected = project_sequence(seq, U)

        assert projected.shape == (100, 60)
        np.testing.assert_allclose(projected, seq, rtol=1e-10)

    def test_very_long_sequence(self):
        """Test projection of very long sequence."""
        seq = np.random.randn(10000, 60).astype(np.float32)
        U = np.linalg.qr(np.random.randn(60, 8))[0].astype(np.float32)

        projected = project_sequence(seq, U)

        assert projected.shape == (10000, 8)
        assert projected.dtype == np.float32

    def test_zero_rows_in_sequence(self):
        """Test projection with some zero rows."""
        seq = np.random.randn(100, 60)
        seq[10:20, :] = 0  # Zero out some rows
        U = np.linalg.qr(np.random.randn(60, 8))[0]

        projected = project_sequence(seq, U)

        # Zero rows should project to zero
        assert np.allclose(projected[10:20, :], 0.0)

    def test_normalize_zero_rows(self):
        """Test normalization with zero rows (should handle gracefully)."""
        seq = np.random.randn(100, 60)
        seq[50, :] = 0  # One zero row
        U = np.linalg.qr(np.random.randn(60, 8))[0]

        # Should not crash
        projected = project_sequence(seq, U, normalize_rows=True)

        # Non-zero rows should be normalized
        norms = np.linalg.norm(projected, axis=1)
        non_zero_norms = norms[norms > 0.1]
        assert np.allclose(non_zero_norms, 1.0, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
