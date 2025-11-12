"""
Tests for classical PCA implementation.

Tests orthonormality, explained variance ratio, saving/loading, and edge cases.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tempfile  # noqa: E402
import shutil  # noqa: E402

from quantum.classical_pca import (  # noqa: E402
    classical_pca,
    save_pca_components,
    load_pca_components,
    compute_reconstruction_error
)


class TestClassicalPCA:
    """Tests for classical_pca function."""

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        # Create random data
        N, D, k = 100, 60, 10
        X = np.random.randn(N, D)

        U, evr = classical_pca(X, k=k)

        # Check shapes
        assert U.shape == (D, k), f"Expected U.shape=(60, 10), got {U.shape}"
        assert evr.shape == (k,), f"Expected evr.shape=(10,), got {evr.shape}"

    def test_columns_are_orthonormal(self):
        """Test that columns of U are orthonormal within 1e-6 tolerance."""
        # Create random data
        N, D, k = 200, 60, 15
        X = np.random.randn(N, D)

        U, evr = classical_pca(X, k=k)

        # Compute U.T @ U (should be identity matrix)
        UTU = U.T @ U
        I_k = np.eye(k)

        # Check orthonormality
        assert np.allclose(UTU, I_k, atol=1e-6), \
            f"Columns not orthonormal. Max deviation: {np.max(np.abs(UTU - I_k))}"

    def test_evr_sums_to_fraction_of_variance(self):
        """Test that EVR sums to approximately the fraction of total variance for k components."""
        # Create data with known structure
        N, D, k = 500, 60, 10
        X = np.random.randn(N, D)

        U, evr = classical_pca(X, k=k)

        # EVR should sum to a value between 0 and 1
        evr_sum = np.sum(evr)
        assert 0 <= evr_sum <= 1, f"EVR sum {evr_sum} not in [0, 1]"

        # Each component should be between 0 and 1
        assert np.all(evr >= 0) and np.all(evr <= 1), \
            "EVR components not in [0, 1]"

        # Components should be in decreasing order
        assert np.all(evr[:-1] >= evr[1:]), \
            "EVR components not in decreasing order"

    def test_evr_decreasing_order(self):
        """Test that explained variance ratios are in decreasing order."""
        X = np.random.randn(300, 60)
        U, evr = classical_pca(X, k=20)

        # Check decreasing order
        for i in range(len(evr) - 1):
            assert evr[i] >= evr[i + 1], \
                f"EVR not decreasing: evr[{i}]={evr[i]}, evr[{i+1}]={evr[i+1]}"

    def test_different_k_values(self):
        """Test with different k values."""
        X = np.random.randn(200, 60)

        for k in [1, 5, 10, 20, 30]:
            U, evr = classical_pca(X, k=k)

            assert U.shape == (60, k)
            assert evr.shape == (k,)

            # Check orthonormality
            UTU = U.T @ U
            assert np.allclose(UTU, np.eye(k), atol=1e-6)

    def test_k_equals_min_dimension(self):
        """Test with k equal to minimum dimension."""
        N, D = 50, 60
        X = np.random.randn(N, D)

        # k = min(N, D) = 50
        k = min(N, D)
        U, evr = classical_pca(X, k=k)

        assert U.shape == (D, k)
        assert evr.shape == (k,)

    def test_invalid_k_raises_error(self):
        """Test that invalid k values raise errors."""
        X = np.random.randn(100, 60)

        # k too large
        with pytest.raises(ValueError, match="must be <="):
            classical_pca(X, k=200)

        # k too small
        with pytest.raises(ValueError, match="must be >= 1"):
            classical_pca(X, k=0)

        with pytest.raises(ValueError, match="must be >= 1"):
            classical_pca(X, k=-5)

    def test_invalid_input_shape_raises_error(self):
        """Test that invalid input shapes raise errors."""
        # 1D array
        with pytest.raises(ValueError, match="Expected 2D array"):
            classical_pca(np.random.randn(100), k=5)

        # 3D array
        with pytest.raises(ValueError, match="Expected 2D array"):
            classical_pca(np.random.randn(10, 60, 5), k=5)

    def test_small_synthetic_example(self):
        """Small synthetic test for acceptance criteria."""
        # Create simple 2D data that lies mostly along one direction
        np.random.seed(42)
        t = np.linspace(0, 10, 100)
        X = np.column_stack([
            t + np.random.randn(100) * 0.1,  # Strong variance
            np.random.randn(100) * 0.01       # Weak variance
        ])

        U, evr = classical_pca(X, k=2)

        # Should have 2 components
        assert U.shape == (2, 2)
        assert evr.shape == (2,)

        # First component should explain most variance
        assert evr[0] > 0.9, f"First component explains {evr[0]:.2%}, expected > 90%"

        # Orthonormality check
        UTU = U.T @ U
        assert np.allclose(UTU, np.eye(2), atol=1e-6)

        # Sum of EVR should be close to 1 (we used all components)
        assert np.abs(np.sum(evr) - 1.0) < 0.01


class TestSavingAndLoading:
    """Tests for saving and loading PCA components."""

    def setup_method(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_components(self):
        """Test saving and loading PCA components."""
        # Create random components
        D, k = 60, 10
        U_original = np.random.randn(D, k)

        # Orthonormalize using QR decomposition
        U_original, _ = np.linalg.qr(U_original)

        # Save
        output_path = save_pca_components(U_original, k=k, output_dir=self.temp_dir)

        # Check file exists
        assert os.path.exists(output_path)

        # Load
        U_loaded = load_pca_components(k=k, output_dir=self.temp_dir)

        # Check equality
        np.testing.assert_array_almost_equal(U_original, U_loaded)

    def test_save_creates_directory(self):
        """Test that save creates output directory if it doesn't exist."""
        new_dir = os.path.join(self.temp_dir, "nested", "path")

        U = np.random.randn(60, 10)
        output_path = save_pca_components(U, k=10, output_dir=new_dir)

        assert os.path.exists(output_path)
        assert os.path.exists(new_dir)

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_pca_components(k=999, output_dir=self.temp_dir)

    def test_filename_format(self):
        """Test that filename follows Uc_k{K}.npy format."""
        U = np.random.randn(60, 15)

        for k in [5, 10, 20]:
            output_path = save_pca_components(U[:, :k], k=k, output_dir=self.temp_dir)
            expected_filename = f"Uc_k{k}.npy"
            assert os.path.basename(output_path) == expected_filename

    def test_integration_pca_and_save(self):
        """Integration test: PCA → save → load."""
        # Generate data
        X = np.random.randn(200, 60)
        k = 15

        # Compute PCA
        U, evr = classical_pca(X, k=k)

        # Save
        save_pca_components(U, k=k, output_dir=self.temp_dir)

        # Load
        U_loaded = load_pca_components(k=k, output_dir=self.temp_dir)

        # Verify
        np.testing.assert_array_almost_equal(U, U_loaded)

        # Verify orthonormality is preserved
        UTU = U_loaded.T @ U_loaded
        assert np.allclose(UTU, np.eye(k), atol=1e-6)


class TestReconstructionError:
    """Tests for reconstruction error computation."""

    def test_perfect_reconstruction(self):
        """Test reconstruction with all components (k=D)."""
        N, D = 100, 20
        X = np.random.randn(N, D)

        # Use all components
        U, evr = classical_pca(X, k=D)

        error = compute_reconstruction_error(X, U)

        # Error should be very small (numerical precision)
        assert error < 1e-10, f"Perfect reconstruction error: {error}"

    def test_partial_reconstruction(self):
        """Test reconstruction with partial components."""
        N, D = 200, 60
        X = np.random.randn(N, D)

        # Use only some components
        k = 10
        U, evr = classical_pca(X, k=k)

        error = compute_reconstruction_error(X, U)

        # Error should be positive (lost information)
        assert error > 0

    def test_more_components_less_error(self):
        """Test that more components → less reconstruction error."""
        X = np.random.randn(150, 60)

        errors = []
        for k in [5, 10, 20, 40]:
            U, evr = classical_pca(X, k=k)
            error = compute_reconstruction_error(X, U)
            errors.append(error)

        # Errors should be decreasing
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1], \
                f"Error increased: k={5*(i+1)} → k={5*(i+2)}"


class TestEdgeCases:
    """Edge case tests."""

    def test_single_component(self):
        """Test with k=1."""
        X = np.random.randn(100, 60)
        U, evr = classical_pca(X, k=1)

        assert U.shape == (60, 1)
        assert evr.shape == (1,)

        # Single vector should be unit norm
        assert np.abs(np.linalg.norm(U[:, 0]) - 1.0) < 1e-6

    def test_wide_matrix(self):
        """Test with more features than samples (N < D)."""
        N, D = 30, 60
        X = np.random.randn(N, D)

        k = 20  # k < N < D
        U, evr = classical_pca(X, k=k)

        assert U.shape == (D, k)
        assert evr.shape == (k,)

        # Check orthonormality
        UTU = U.T @ U
        assert np.allclose(UTU, np.eye(k), atol=1e-6)

    def test_tall_matrix(self):
        """Test with more samples than features (N > D)."""
        N, D = 500, 60
        X = np.random.randn(N, D)

        k = 30
        U, evr = classical_pca(X, k=k)

        assert U.shape == (D, k)
        assert evr.shape == (k,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
