"""
Sequence projection into k-D subspaces.

This module provides utilities for projecting time-series sequences from
high-dimensional space (60D) into low-dimensional subspaces (k-D) using
principal components from PCA or qPCA.
"""

import logging

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def project_sequence(
    seq: np.ndarray,
    U: np.ndarray,
    normalize_rows: bool = False
) -> np.ndarray:
    """
    Project a sequence into k-D subspace defined by principal components U.

    Given a sequence of T frames in D-dimensional space and principal
    components U, this function computes the projection:
        Z = X @ U
    where X is shape [T, D] and U is shape [D, k], resulting in Z of
    shape [T, k].

    Optionally, each projected row can be normalized to unit length.

    Args:
        seq: Input sequence of shape [T, D] where:
             T = number of time steps
             D = original dimension (typically 60)
             Can be float32 or float64
        U: Principal components matrix of shape [D, k] where:
           k = target subspace dimension
           Columns should be orthonormal for proper projection
        normalize_rows: If True, normalize each row of output to unit length
                       Default: False

    Returns:
        projected: Projected sequence of shape [T, k]
                  Preserves input dtype (float32 or float64)
                  If normalize_rows=True, each row has unit L2 norm

    Raises:
        ValueError: If shapes are incompatible or invalid

    Example:
        >>> seq = np.random.randn(100, 60).astype(np.float32)  # 100 frames
        >>> U = np.random.randn(60, 8)  # 8-D subspace
        >>> U = np.linalg.qr(U)[0]  # Orthonormalize
        >>> z = project_sequence(seq, U)
        >>> print(z.shape)  # (100, 8)
        >>> print(z.dtype)  # float32
    """
    # Validate inputs
    if seq.ndim != 2:
        raise ValueError(f"seq must be 2D array, got shape {seq.shape}")

    if U.ndim != 2:
        raise ValueError(f"U must be 2D array, got shape {U.shape}")

    T, D = seq.shape
    D_U, k = U.shape

    if D != D_U:
        raise ValueError(
            f"Incompatible dimensions: seq has D={D}, U has D={D_U}"
        )

    if k <= 0 or k > D:
        raise ValueError(f"k must be in range [1, {D}], got {k}")

    # Preserve input dtype
    input_dtype = seq.dtype
    logger.debug(
        f"Projecting sequence: {seq.shape} @ {U.shape} -> ({T}, {k})"
    )
    logger.debug(f"Input dtype: {input_dtype}")

    # Project: Z = X @ U
    projected = seq @ U

    # Normalize rows if requested
    if normalize_rows:
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms > 1e-12, norms, 1.0)
        projected = projected / norms
        logger.debug("Normalized projected rows to unit length")

    # Ensure output dtype matches input dtype
    if projected.dtype != input_dtype:
        projected = projected.astype(input_dtype)

    return projected


def project_sequences_batch(
    sequences: list[np.ndarray],
    U: np.ndarray,
    normalize_rows: bool = False
) -> list[np.ndarray]:
    """
    Project multiple sequences into k-D subspace.

    Convenience function for projecting a list of sequences using the
    same principal components U.

    Args:
        sequences: List of sequences, each of shape [T_i, D]
                  Can have different lengths T_i
        U: Principal components matrix of shape [D, k]
        normalize_rows: If True, normalize rows of each projected sequence

    Returns:
        projected_sequences: List of projected sequences, each shape [T_i, k]

    Example:
        >>> seqs = [np.random.randn(100, 60), np.random.randn(150, 60)]
        >>> U = np.linalg.qr(np.random.randn(60, 8))[0]
        >>> z_seqs = project_sequences_batch(seqs, U)
        >>> print([z.shape for z in z_seqs])  # [(100, 8), (150, 8)]
    """
    projected = []

    for i, seq in enumerate(sequences):
        z = project_sequence(seq, U, normalize_rows=normalize_rows)
        projected.append(z)

        if (i + 1) % 100 == 0:
            logger.info(f"Projected {i + 1}/{len(sequences)} sequences")

    return projected


def verify_projection_properties(
    seq: np.ndarray,
    projected: np.ndarray,
    U: np.ndarray,
    tolerance: float = 1e-6
) -> dict:
    """
    Verify mathematical properties of projection.

    Checks:
    1. Shape consistency
    2. Norm preservation (for orthonormal U)
    3. Row normalization (if applicable)

    Args:
        seq: Original sequence of shape [T, D]
        projected: Projected sequence of shape [T, k]
        U: Principal components of shape [D, k]
        tolerance: Tolerance for numerical checks

    Returns:
        results: Dictionary with verification results:
            - 'shape_ok': bool
            - 'avg_norm_original': float
            - 'avg_norm_projected': float
            - 'norm_preserved': bool (for orthonormal U)
            - 'is_orthonormal': bool (check if U is orthonormal)

    Example:
        >>> seq = np.random.randn(100, 60)
        >>> U = np.linalg.qr(np.random.randn(60, 8))[0]
        >>> z = project_sequence(seq, U)
        >>> results = verify_projection_properties(seq, z, U)
        >>> print(results['norm_preserved'])  # True
    """
    T, D = seq.shape
    T_z, k = projected.shape

    # Check shapes
    shape_ok = (T == T_z) and (U.shape == (D, k))

    # Check if U is orthonormal
    U_inner = U.T @ U
    identity = np.eye(k)
    is_orthonormal = np.allclose(U_inner, identity, atol=tolerance)

    # Compute average norms
    norms_original = np.linalg.norm(seq, axis=1)
    norms_projected = np.linalg.norm(projected, axis=1)

    avg_norm_original = np.mean(norms_original)
    avg_norm_projected = np.mean(norms_projected)

    # For orthonormal U, projection should preserve or reduce norms
    # ||Z||_2 <= ||X||_2 (Pythagorean theorem in subspace)
    if is_orthonormal:
        # Projected norms should be <= original norms
        norm_preserved = avg_norm_projected <= avg_norm_original + tolerance
    else:
        # Can't make guarantees without orthonormality
        norm_preserved = None

    results = {
        'shape_ok': shape_ok,
        'avg_norm_original': float(avg_norm_original),
        'avg_norm_projected': float(avg_norm_projected),
        'norm_preserved': norm_preserved,
        'is_orthonormal': is_orthonormal,
    }

    return results
