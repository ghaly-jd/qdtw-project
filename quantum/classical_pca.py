"""
Classical PCA implementation for baseline comparison.

This module provides classical Principal Component Analysis
to produce top-k principal directions for comparison with quantum methods.
"""

import numpy as np
import os
from typing import Tuple


def classical_pca(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute classical PCA to extract top-k principal components.

    Uses SVD-based implementation equivalent to sklearn's PCA with svd_solver='full'.

    Args:
        X: Data matrix of shape [N, D] where N is number of samples, D is dimensionality
           For skeleton frames: [N, 60] where N is number of frames
        k: Number of principal components to extract

    Returns:
        U: Principal component matrix of shape [D, k] with orthonormal columns
           Each column is a principal direction (eigenvector)
        evr: Explained variance ratio of shape [k]
             Fraction of total variance explained by each component

    Notes:
        - U columns are orthonormal: U.T @ U = I_k
        - Components are ordered by decreasing explained variance
        - evr[i] is in range [0, 1] and sum(evr) ≤ 1

    Example:
        >>> X = np.random.randn(1000, 60)
        >>> U, evr = classical_pca(X, k=10)
        >>> U.shape
        (60, 10)
        >>> evr.shape
        (10,)
        >>> np.allclose(U.T @ U, np.eye(10))
        True
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")

    if k > min(X.shape):
        raise ValueError(f"k={k} must be <= min(X.shape)={min(X.shape)}")

    if k < 1:
        raise ValueError(f"k={k} must be >= 1")

    # Center the data
    X_centered = X - np.mean(X, axis=0, keepdims=True)

    # Compute SVD: X_centered = U_svd @ S @ Vt
    # For PCA, we want V (right singular vectors)
    # which are the principal directions
    U_svd, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Principal components are rows of Vt (or columns of V)
    # We want shape [D, k]
    V = Vt.T  # Shape: [D, min(N,D)]
    U = V[:, :k]  # Shape: [D, k]

    # Compute explained variance ratio
    # Variance explained by each component = (singular_value^2) / (N-1)
    explained_variance = (S ** 2) / (X.shape[0] - 1)
    total_variance = np.sum(explained_variance)

    # Explained variance ratio for top k components
    evr = explained_variance[:k] / total_variance

    return U, evr


def save_pca_components(U: np.ndarray, evr: np.ndarray, filepath: str) -> str:
    """
    Save PCA components to disk as NPZ file.

    Args:
        U: Principal component matrix of shape [D, k]
        evr: Explained variance ratio of shape [k]
        filepath: Full path to output file (should end in .npz)

    Returns:
        filepath: Path where file was saved

    Example:
        >>> U = np.random.randn(60, 10)
        >>> evr = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        >>> save_pca_components(U, evr, 'results/Uc_k10.npz')
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save as compressed numpy archive
    np.savez(filepath, U=U, explained_variance_ratio=evr)

    return filepath


def load_pca_components(k: int, output_dir: str = "quantum/outputs") -> np.ndarray:
    """
    Load PCA components from disk.

    Args:
        k: Number of components (used in filename)
        output_dir: Directory containing the file

    Returns:
        U: Principal component matrix of shape [D, k]

    Raises:
        FileNotFoundError: If the file doesn't exist

    Example:
        >>> U = load_pca_components(k=10)
        >>> U.shape
        (60, 10)
    """
    filename = f"Uc_k{k}.npy"
    filepath = os.path.join(output_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PCA components file not found: {filepath}")

    U = np.load(filepath)
    return U


def compute_reconstruction_error(X: np.ndarray, U: np.ndarray) -> float:
    """
    Compute reconstruction error using PCA components.

    Args:
        X: Data matrix of shape [N, D]
        U: Principal components of shape [D, k]

    Returns:
        error: Mean squared reconstruction error

    Notes:
        - Projects data onto principal subspace and reconstructs
        - Lower error indicates better representation
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0, keepdims=True)

    # Project onto principal subspace
    # Z = X_centered @ U  # Shape: [N, k]
    # X_reconstructed = Z @ U.T  # Shape: [N, D]

    # Equivalent one-liner
    X_reconstructed = (X_centered @ U) @ U.T

    # Compute mean squared error
    error = np.mean((X_centered - X_reconstructed) ** 2)

    return error


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute classical PCA')
    parser.add_argument('--frames', type=str, required=True, help='Path to frame bank .npy file')
    parser.add_argument('--k', type=int, required=True, help='Number of principal components')
    parser.add_argument('--output', type=str, required=True, help='Output path for PCA .npz file')
    
    args = parser.parse_args()
    
    print(f"Loading frame bank from {args.frames}...")
    X = np.load(args.frames)
    print(f"Frame bank shape: {X.shape}")
    
    print(f"\nComputing PCA with k={args.k}...")
    U, evr = classical_pca(X, args.k)
    
    print(f"\nPCA computed:")
    print(f"  U shape: {U.shape}")
    print(f"  Explained variance ratio: {evr}")
    print(f"  Total variance explained: {np.sum(evr):.4f}")
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_pca_components(U, evr, args.output)
    print(f"\n✅ Saved to {args.output}")

