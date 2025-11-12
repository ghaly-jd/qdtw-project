"""
Tests for Quantum PCA (qPCA) implementation.

Tests verify:
1. Consistency between qPCA and classical PCA on low-noise data
2. Orthonormality of principal components
3. Density matrix properties
4. State preparation circuits
5. Save/load functionality
6. Edge cases
"""

import numpy as np
import pytest
from pathlib import Path

from quantum.qpca import (
    qpca_directions,
    build_state_prep_circuit,
    save_qpca_components,
    load_qpca_components,
    compute_principal_angles,
)
from quantum.classical_pca import classical_pca


class TestQPCADirections:
    """Test qpca_directions function for correctness."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        X = np.random.randn(100, 60)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        U = qpca_directions(X, k=10)

        assert U.shape == (60, 10), f"Expected shape (60, 10), got {U.shape}"

    def test_columns_orthonormal(self):
        """Test that U columns are orthonormal within tolerance."""
        X = np.random.randn(100, 60)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        U = qpca_directions(X, k=10)

        # Check U.T @ U = I
        ortho_check = U.T @ U
        identity = np.eye(10)
        max_deviation = np.max(np.abs(ortho_check - identity))

        assert max_deviation < 1e-6, \
            f"Columns not orthonormal: max deviation = {max_deviation:.2e}"

    def test_different_k_values(self):
        """Test with different k values."""
        X = np.random.randn(200, 60)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        for k in [1, 5, 10, 20, 30]:
            U = qpca_directions(X, k=k)
            assert U.shape == (60, k), f"Wrong shape for k={k}"

            # Check orthonormality
            ortho_check = U.T @ U
            identity = np.eye(k)
            max_dev = np.max(np.abs(ortho_check - identity))
            assert max_dev < 1e-6, f"Not orthonormal for k={k}"

    def test_density_matrix_trace(self):
        """Test that density matrix has trace ≈ 1."""
        X = np.random.randn(50, 60)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # Build density matrix manually
        M = X.shape[0]
        rho = (X.T @ X) / M

        trace_val = np.trace(rho)
        assert np.isclose(trace_val, 1.0, atol=1e-3), \
            f"Density matrix trace = {trace_val:.6f}, expected ≈1.0"

    def test_density_matrix_hermitian(self):
        """Test that density matrix is Hermitian (symmetric for real data)."""
        X = np.random.randn(50, 60)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        M = X.shape[0]
        rho = (X.T @ X) / M

        assert np.allclose(rho, rho.T, atol=1e-10), \
            "Density matrix is not Hermitian"

    def test_eigenvalues_positive(self):
        """Test that density matrix eigenvalues are non-negative."""
        X = np.random.randn(50, 60)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        M = X.shape[0]
        rho = (X.T @ X) / M

        eigenvalues = np.linalg.eigvalsh(rho)
        assert np.all(eigenvalues >= -1e-10), \
            f"Negative eigenvalues found: min = {np.min(eigenvalues):.2e}"

    def test_requires_normalized_input(self):
        """Test that function requires normalized input."""
        X = np.random.randn(50, 60)
        # Don't normalize

        with pytest.raises(ValueError, match="unit vectors"):
            qpca_directions(X, k=10)

    def test_invalid_k_raises_error(self):
        """Test that invalid k values raise errors."""
        X = np.random.randn(50, 60)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # k too large
        with pytest.raises(ValueError, match="k must be in range"):
            qpca_directions(X, k=100)

        # k negative
        with pytest.raises(ValueError, match="k must be in range"):
            qpca_directions(X, k=-1)

        # k zero
        with pytest.raises(ValueError, match="k must be in range"):
            qpca_directions(X, k=0)

    def test_invalid_shape_raises_error(self):
        """Test that invalid X shapes raise errors."""
        # 1D array
        X_1d = np.random.randn(60)
        with pytest.raises(ValueError, match="must be 2D"):
            qpca_directions(X_1d, k=5)

        # 3D array
        X_3d = np.random.randn(10, 60, 3)
        with pytest.raises(ValueError, match="must be 2D"):
            qpca_directions(X_3d, k=5)


class TestConsistencyWithClassicalPCA:
    """Test consistency between qPCA and classical PCA."""

    def test_principal_angles_low_noise_data(self):
        """
        Test that principal angles < 15° on low-noise synthetic data.

        This verifies that qPCA recovers similar principal directions
        as classical PCA on clean data when BOTH use the same normalization.

        NOTE: Classical PCA on centered data vs qPCA on normalized data
        can give different results. For consistency, we compare classical
        PCA on normalized data.
        """
        # Generate synthetic data with clear principal directions
        np.random.seed(42)
        n_samples = 500

        # Create data with dominant directions
        # First component: high variance
        component1 = np.random.randn(n_samples, 1) * 10.0
        # Second component: medium variance
        component2 = np.random.randn(n_samples, 1) * 5.0
        # Remaining components: low variance (noise)
        noise = np.random.randn(n_samples, 58) * 0.5

        # Random rotation to mix components
        random_rotation = np.linalg.qr(np.random.randn(60, 60))[0]
        X_raw = np.hstack([component1, component2, noise])
        X_rotated = X_raw @ random_rotation.T

        # Normalize for BOTH methods (fair comparison)
        X_normalized = X_rotated / np.linalg.norm(
            X_rotated, axis=1, keepdims=True
        )

        # Run both methods on NORMALIZED data
        k = 10
        U_classical, _ = classical_pca(X_normalized, k=k)
        U_quantum = qpca_directions(X_normalized, k=k)

        # Compute principal angles
        angles = compute_principal_angles(U_classical, U_quantum)

        max_angle = np.max(angles)
        print(f"\nPrincipal angles: {angles}")
        print(f"Max principal angle: {max_angle:.2f}°")

        assert max_angle < 15.0, \
            f"Max principal angle {max_angle:.2f}° exceeds 15° threshold"

    def test_similar_reconstruction_quality(self):
        """
        Test that both methods have similar reconstruction quality.

        NOTE: We compare both methods on normalized data for fair comparison.
        """
        np.random.seed(123)
        X = np.random.randn(200, 60)
        X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)

        k = 10
        U_classical, _ = classical_pca(X_normalized, k=k)
        U_quantum = qpca_directions(X_normalized, k=k)

        # Compute reconstruction errors on SAME data
        X_norm_centered = (
            X_normalized - np.mean(X_normalized, axis=0, keepdims=True)
        )

        # Classical reconstruction
        X_recon_classical = (X_norm_centered @ U_classical) @ U_classical.T
        error_classical = np.mean((X_norm_centered - X_recon_classical) ** 2)

        # Quantum reconstruction
        X_recon_quantum = (X_norm_centered @ U_quantum) @ U_quantum.T
        error_quantum = np.mean((X_norm_centered - X_recon_quantum) ** 2)

        print(f"\nClassical reconstruction error: {error_classical:.6f}")
        print(f"Quantum reconstruction error: {error_quantum:.6f}")
        print(f"Ratio: {error_quantum / error_classical:.3f}")

        # Errors should be in same ballpark (within factor of 2)
        assert 0.3 < (error_quantum / error_classical) < 3.0, \
            "Reconstruction errors differ too much"

    def test_span_overlap_high_on_low_rank_data(self):
        """Test high subspace overlap on low-rank data."""
        # Create low-rank data (rank 5)
        np.random.seed(999)
        n_samples = 300
        true_rank = 5

        # Generate low-rank matrix
        A = np.random.randn(n_samples, true_rank)
        B = np.random.randn(true_rank, 60)
        X = A @ B

        X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)

        k = 5  # Extract true rank
        U_classical, _ = classical_pca(X, k=k)
        U_quantum = qpca_directions(X_normalized, k=k)

        angles = compute_principal_angles(U_classical, U_quantum)
        max_angle = np.max(angles)

        print(f"\nLow-rank data max angle: {max_angle:.2f}°")

        # Should be very small for low-rank data
        assert max_angle < 10.0, \
            f"Max angle {max_angle:.2f}° too large for low-rank data"


class TestStatePreparationCircuit:
    """Test quantum state preparation circuit."""

    def test_circuit_creation(self):
        """Test that state prep circuit is created successfully."""
        x = np.random.randn(60)
        x = x / np.linalg.norm(x)

        qc = build_state_prep_circuit(x)

        # Should have 6 qubits (2^6 = 64 >= 60)
        assert qc.num_qubits == 6, f"Expected 6 qubits, got {qc.num_qubits}"

    def test_circuit_has_gates(self):
        """Test that circuit has non-zero depth."""
        x = np.random.randn(60)
        x = x / np.linalg.norm(x)

        qc = build_state_prep_circuit(x)

        assert qc.depth() > 0, "Circuit should have non-zero depth"
        assert qc.size() > 0, "Circuit should have gates"

    def test_circuit_metadata(self):
        """Test that circuit has proper metadata."""
        x = np.random.randn(60)
        x = x / np.linalg.norm(x)

        qc = build_state_prep_circuit(x)

        assert hasattr(qc, 'metadata'), "Circuit should have metadata"
        assert 'original_dim' in qc.metadata
        assert 'padded_dim' in qc.metadata
        assert 'n_qubits' in qc.metadata

        assert qc.metadata['original_dim'] == 60
        assert qc.metadata['padded_dim'] == 64
        assert qc.metadata['n_qubits'] == 6

    def test_auto_normalization(self):
        """Test that unnormalized vectors are normalized automatically."""
        x = np.random.randn(60) * 5.0  # Not normalized

        # Should not raise error, just warn
        qc = build_state_prep_circuit(x)

        assert qc.num_qubits == 6

    def test_different_dimensions(self):
        """Test state prep for different dimensions."""
        for D in [10, 32, 60, 100]:
            x = np.random.randn(D)
            x = x / np.linalg.norm(x)

            qc = build_state_prep_circuit(x)

            expected_qubits = int(np.ceil(np.log2(D)))
            assert qc.num_qubits == expected_qubits, \
                f"For D={D}, expected {expected_qubits} qubits, got {qc.num_qubits}"

    def test_invalid_input_raises_error(self):
        """Test that invalid inputs raise errors."""
        # 2D array
        X = np.random.randn(10, 60)
        with pytest.raises(ValueError, match="must be 1D"):
            build_state_prep_circuit(X)


class TestSavingAndLoading:
    """Test saving and loading qPCA components."""

    def test_save_and_load(self):
        """Test that components can be saved and loaded."""
        X = np.random.randn(100, 60)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        U = qpca_directions(X, k=10)

        # Save
        path = save_qpca_components(U, k=10)
        assert Path(path).exists(), f"File not created: {path}"

        # Load
        U_loaded = load_qpca_components(k=10)

        # Verify equality
        np.testing.assert_array_equal(U, U_loaded)

    def test_load_nonexistent_raises_error(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_qpca_components(k=999)

    def test_save_creates_directory(self):
        """Test that save creates output directory if needed."""
        X = np.random.randn(50, 60)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        U = qpca_directions(X, k=5)
        path = save_qpca_components(U, k=5)

        assert Path(path).parent.exists(), "Output directory not created"

    def test_filename_format(self):
        """Test that filename follows Uq_k{K}.npy format."""
        X = np.random.randn(50, 60)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        U = qpca_directions(X, k=7)
        path = save_qpca_components(U, k=7)

        assert "Uq_k7.npy" in path, f"Unexpected filename format: {path}"

    def test_integration_qpca_save_load(self):
        """Test full pipeline: qPCA → save → load."""
        X = np.random.randn(100, 60)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # Run qPCA
        U_original = qpca_directions(X, k=15)

        # Save
        save_qpca_components(U_original, k=15)

        # Load
        U_loaded = load_qpca_components(k=15)

        # Verify
        np.testing.assert_allclose(U_original, U_loaded, rtol=1e-10)


class TestPrincipalAngles:
    """Test principal angles computation."""

    def test_identical_subspaces_zero_angles(self):
        """Test that identical subspaces have zero principal angles."""
        U = np.linalg.qr(np.random.randn(60, 10))[0]

        angles = compute_principal_angles(U, U)

        # Use looser tolerance due to numerical precision
        assert np.allclose(angles, 0.0, atol=1e-5), \
            f"Identical subspaces should have zero angles, got {angles}"

    def test_orthogonal_subspaces_90_degrees(self):
        """Test that orthogonal subspaces have 90° angles."""
        # Create two orthogonal subspaces
        Q = np.linalg.qr(np.random.randn(60, 20))[0]
        U1 = Q[:, :10]   # First 10 columns
        U2 = Q[:, 10:20]  # Next 10 (orthogonal to first)

        angles = compute_principal_angles(U1, U2)

        assert np.allclose(angles, 90.0, atol=1.0), \
            f"Orthogonal subspaces should have 90° angles, got {angles}"

    def test_angles_in_valid_range(self):
        """Test that angles are in [0, 90] degrees."""
        U1 = np.linalg.qr(np.random.randn(60, 10))[0]
        U2 = np.linalg.qr(np.random.randn(60, 10))[0]

        angles = compute_principal_angles(U1, U2)

        assert np.all(angles >= 0.0), \
            f"Angles should be >= 0, got min {np.min(angles)}"
        assert np.all(angles <= 90.0), \
            f"Angles should be <= 90, got max {np.max(angles)}"

    def test_different_dimensions(self):
        """Test principal angles with different subspace dimensions."""
        U1 = np.linalg.qr(np.random.randn(60, 5))[0]
        U2 = np.linalg.qr(np.random.randn(60, 10))[0]

        angles = compute_principal_angles(U1, U2)

        # Should return min(k1, k2) angles
        assert len(angles) == 5, f"Expected 5 angles, got {len(angles)}"


class TestEdgeCases:
    """Test edge cases for qPCA."""

    def test_single_component(self):
        """Test extracting single component (k=1)."""
        X = np.random.randn(50, 60)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        U = qpca_directions(X, k=1)

        assert U.shape == (60, 1)
        assert np.isclose(np.linalg.norm(U), 1.0, atol=1e-6)

    def test_wide_matrix(self):
        """Test with wide matrix (M < D)."""
        X = np.random.randn(30, 60)  # 30 samples, 60 features
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # Can extract at most M components
        U = qpca_directions(X, k=20)

        assert U.shape == (60, 20)

        # Check orthonormality
        ortho_check = U.T @ U
        identity = np.eye(20)
        assert np.allclose(ortho_check, identity, atol=1e-6)

    def test_tall_matrix(self):
        """Test with tall matrix (M > D)."""
        X = np.random.randn(500, 60)  # 500 samples, 60 features
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        U = qpca_directions(X, k=30)

        assert U.shape == (60, 30)

        # Check orthonormality
        ortho_check = U.T @ U
        identity = np.eye(30)
        assert np.allclose(ortho_check, identity, atol=1e-6)

    def test_k_equals_min_dimension(self):
        """Test with k = min(M, D)."""
        M, D = 100, 60
        X = np.random.randn(M, D)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        k = min(M, D)
        U = qpca_directions(X, k=k)

        assert U.shape == (D, k)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
