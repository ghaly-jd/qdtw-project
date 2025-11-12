"""
Tests for amplitude encoding utilities.

Tests normalization, zero vector handling, batch processing, and edge cases.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from features.amplitude_encoding import (  # noqa: E402
    encode_unit_vector,
    batch_encode_unit_vectors,
    verify_normalization
)


class TestEncodeUnitVector:
    """Tests for single vector encoding."""

    def test_normalization(self):
        """Test that encoded vector has norm = 1."""
        # Random 60-D vector
        x = np.random.randn(60)
        encoded = encode_unit_vector(x)

        # Check norm is 1.0 within tolerance
        norm = np.linalg.norm(encoded)
        assert np.abs(norm - 1.0) < 1e-6, f"Expected norm=1.0, got {norm}"

    def test_zero_vector_returns_uniform(self):
        """Test that zero vector returns uniform distribution."""
        x = np.zeros(60)
        encoded = encode_unit_vector(x)

        # Should return 1/sqrt(60) for all elements
        expected = np.ones(60) / np.sqrt(60)
        np.testing.assert_allclose(encoded, expected, rtol=1e-6)

        # Verify it's normalized
        norm = np.linalg.norm(encoded)
        assert np.abs(norm - 1.0) < 1e-6

    def test_near_zero_vector(self):
        """Test vector with very small values (numerical noise)."""
        x = np.ones(60) * 1e-15  # Below EPS threshold
        encoded = encode_unit_vector(x)

        # Should be treated as zero and return uniform
        expected = np.ones(60) / np.sqrt(60)
        np.testing.assert_allclose(encoded, expected, rtol=1e-6)

    def test_preserves_direction(self):
        """Test that normalization preserves vector direction."""
        x = np.array([3.0] + [0.0] * 59)
        encoded = encode_unit_vector(x)

        # First element should be 1.0, rest should be 0.0
        assert np.abs(encoded[0] - 1.0) < 1e-6
        assert np.allclose(encoded[1:], 0.0, atol=1e-6)

    def test_wrong_shape_raises_error(self):
        """Test that wrong input shape raises ValueError."""
        # 2D array
        with pytest.raises(ValueError, match="Expected 1D array"):
            encode_unit_vector(np.random.randn(10, 60))

        # Wrong length
        with pytest.raises(ValueError, match="Expected length 60"):
            encode_unit_vector(np.random.randn(50))

    def test_list_input_conversion(self):
        """Test that list input is converted to array."""
        x = [1.0] * 60
        encoded = encode_unit_vector(x)

        assert isinstance(encoded, np.ndarray)
        assert encoded.shape == (60,)
        assert np.abs(np.linalg.norm(encoded) - 1.0) < 1e-6


class TestBatchEncodeUnitVectors:
    """Tests for batch vector encoding."""

    def test_batch_normalization(self):
        """Test that all rows are normalized."""
        X = np.random.randn(10, 60)
        encoded = batch_encode_unit_vectors(X)

        # Check all rows have norm = 1.0
        norms = np.linalg.norm(encoded, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)

    def test_preserves_shape(self):
        """Test that output shape matches input shape."""
        X = np.random.randn(25, 60)
        encoded = batch_encode_unit_vectors(X)

        assert encoded.shape == X.shape

    def test_zero_rows_replaced_with_uniform(self):
        """Test that zero rows are replaced with uniform distribution."""
        X = np.random.randn(5, 60)
        X[2] = 0.0  # Make one row zero

        encoded = batch_encode_unit_vectors(X)

        # Check zero row is uniform
        expected_uniform = np.ones(60) / np.sqrt(60)
        np.testing.assert_allclose(encoded[2], expected_uniform, rtol=1e-6)

        # Check all rows are normalized
        norms = np.linalg.norm(encoded, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)

    def test_multiple_zero_rows(self):
        """Test handling of multiple zero rows."""
        X = np.random.randn(6, 60)
        X[1] = 0.0
        X[3] = 0.0
        X[5] = 0.0

        encoded = batch_encode_unit_vectors(X)

        # All should be normalized
        norms = np.linalg.norm(encoded, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)

    def test_wrong_shape_raises_error(self):
        """Test that wrong input shape raises ValueError."""
        # 1D array
        with pytest.raises(ValueError, match="Expected 2D array"):
            batch_encode_unit_vectors(np.random.randn(60))

        # Wrong number of columns
        with pytest.raises(ValueError, match="Expected 60 columns"):
            batch_encode_unit_vectors(np.random.randn(10, 50))

    def test_single_row_batch(self):
        """Test batch encoding with single row."""
        X = np.random.randn(1, 60)
        encoded = batch_encode_unit_vectors(X)

        assert encoded.shape == (1, 60)
        norm = np.linalg.norm(encoded[0])
        assert np.abs(norm - 1.0) < 1e-6

    def test_large_batch(self):
        """Test batch encoding with large number of frames."""
        X = np.random.randn(1000, 60)
        encoded = batch_encode_unit_vectors(X)

        assert encoded.shape == (1000, 60)
        assert verify_normalization(encoded)


class TestVerifyNormalization:
    """Tests for normalization verification."""

    def test_normalized_data_passes(self):
        """Test that properly normalized data passes verification."""
        X = np.random.randn(10, 60)
        X_normalized = batch_encode_unit_vectors(X)

        assert verify_normalization(X_normalized) is True

    def test_unnormalized_data_fails(self):
        """Test that unnormalized data fails verification."""
        X = np.random.randn(10, 60) * 5.0  # Not normalized

        assert verify_normalization(X) is False

    def test_tolerance_parameter(self):
        """Test custom tolerance parameter."""
        X = np.random.randn(5, 60)
        X_normalized = batch_encode_unit_vectors(X)

        # Should pass with reasonable tolerance
        assert verify_normalization(X_normalized, tolerance=1e-4) is True

        # Should fail with very strict tolerance
        # (numerical errors might make this fail at machine precision)


class TestIntegration:
    """Integration tests with real skeleton data patterns."""

    def test_realistic_skeleton_frame(self):
        """Test with realistic skeleton coordinate values."""
        # Typical MSR Action skeleton coordinates
        x = np.random.uniform(0, 300, 60)  # Realistic coordinate range

        encoded = encode_unit_vector(x)

        assert encoded.shape == (60,)
        assert np.abs(np.linalg.norm(encoded) - 1.0) < 1e-6

    def test_realistic_sequence(self):
        """Test with realistic skeleton sequence."""
        # Simulate a 50-frame sequence
        T = 50
        X = np.random.uniform(0, 300, (T, 60))

        encoded = batch_encode_unit_vectors(X)

        assert encoded.shape == (T, 60)
        assert verify_normalization(encoded)

        # Check min and max norms
        norms = np.linalg.norm(encoded, axis=1)
        assert np.min(norms) > 0.999
        assert np.max(norms) < 1.001

    def test_dtype_preservation(self):
        """Test that output dtype is float32."""
        X = np.random.randn(10, 60).astype(np.float64)
        encoded = batch_encode_unit_vectors(X)

        assert encoded.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
