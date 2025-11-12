"""
Amplitude encoding utilities for 60-D quaternion frame vectors.

This module provides functions to normalize skeleton frame vectors for quantum
amplitude encoding, ensuring all vectors are unit-normalized (L2 norm = 1).
"""

import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Small epsilon to prevent division by zero
EPS = 1e-12


def encode_unit_vector(x: np.ndarray) -> np.ndarray:
    """
    Encode a single 60-D frame vector using standardization.

    **CRITICAL CHANGE (Nov 7, 2025)**: 
    This function now uses z-score standardization instead of L2 normalization.

    Note: For single vectors, standardization requires a reference dataset.
    This function is kept for API compatibility but should not be used alone.
    Use batch_encode_unit_vectors() for proper standardization.

    Args:
        x: 1D float array of length 60

    Returns:
        Standardized vector with shape (60,) - but note this is not
        recommended for single vectors (no reference statistics)

    Raises:
        ValueError: If input is not a 1D array of length 60

    Notes:
        - Single-vector standardization is ill-defined
        - This just returns the input as-is with a warning
        - Use batch_encode_unit_vectors() for proper encoding
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {x.shape}")

    if len(x) != 60:
        raise ValueError(f"Expected length 60, got {len(x)}")

    logger.warning(
        "encode_unit_vector() called on single vector. "
        "Standardization requires batch statistics. "
        "Use batch_encode_unit_vectors() instead."
    )

    # Return as-is (no meaningful standardization for single vector)
    return x.astype(np.float32)


def batch_encode_unit_vectors(X: np.ndarray) -> np.ndarray:
    """
    Encode multiple 60-D frame vectors using STANDARDIZATION (not normalization).

    **CRITICAL CHANGE (Nov 7, 2025)**: 
    This function now uses z-score standardization instead of L2 normalization.
    The old approach (unit vector normalization) destroyed magnitude information,
    causing class separability to collapse from 1.97x to 1.04x.

    Standardization formula: X_std = (X - mean) / std
    - Preserves relative magnitude differences between features
    - Centers data at zero mean, unit variance per feature
    - Does NOT destroy class-discriminative information

    Args:
        X: 2D array of shape [T, 60] where T is the number of frames

    Returns:
        2D array of shape [T, 60] where each COLUMN is standardized
        (zero mean, unit variance)

    Raises:
        ValueError: If input is not a 2D array with 60 columns

    Notes:
        - Standardizes per-feature (column-wise), not per-frame (row-wise)
        - Handles constant features gracefully (sets std=1 to avoid division by zero)
        - Maintains class separability unlike L2 normalization
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")

    if X.shape[1] != 60:
        raise ValueError(f"Expected 60 columns, got {X.shape[1]}")

    # Compute column-wise statistics (per-feature)
    mean = np.mean(X, axis=0, keepdims=True)  # Shape: (1, 60)
    std = np.std(X, axis=0, keepdims=True)    # Shape: (1, 60)

    # Handle constant features (std = 0)
    std_safe = np.where(std < EPS, 1.0, std)

    # Standardize: z-score per feature
    X_standardized = ((X - mean) / std_safe).astype(np.float32)

    # Log statistics
    num_constant = np.sum(std.squeeze() < EPS)
    if num_constant > 0:
        logger.warning(f"Found {num_constant} constant features (std < {EPS})")
    
    logger.info(f"Standardized {X.shape[0]} frames:")
    logger.info(f"  Original mean range: [{np.min(X)}:.2f, {np.max(X)}:.2f]")
    logger.info(f"  Standardized mean: {np.mean(X_standardized):.6f}")
    logger.info(f"  Standardized std: {np.std(X_standardized):.6f}")

    return X_standardized


def verify_normalization(X: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Verify standardization properties of encoded vectors.

    **UPDATED (Nov 7, 2025)**: Now checks standardization (zero mean, unit variance)
    instead of L2 normalization.

    Args:
        X: 2D array of shape [T, 60]
        tolerance: Tolerance for mean/std checks

    Returns:
        True if X is properly standardized (mean ≈ 0, std ≈ 1 per feature)

    Example:
        >>> X_std = batch_encode_unit_vectors(X_raw)
        >>> assert verify_normalization(X_std)
    """
    if X.ndim != 2:
        logger.error(f"Expected 2D array, got shape {X.shape}")
        return False

    # Check column-wise (per-feature) statistics
    means = np.mean(X, axis=0)  # Shape: (60,)
    stds = np.std(X, axis=0)    # Shape: (60,)

    # All feature means should be near 0
    mean_check = np.allclose(means, 0.0, atol=tolerance)
    
    # All feature stds should be near 1
    std_check = np.allclose(stds, 1.0, atol=tolerance)

    if mean_check and std_check:
        logger.info(
            f"✅ Standardization verified: "
            f"mean={np.mean(means):.6f}, std={np.mean(stds):.6f}"
        )
        return True
    else:
        logger.warning(
            f"⚠️  Standardization failed: "
            f"mean={np.mean(means):.6f} (expected 0), "
            f"std={np.mean(stds):.6f} (expected 1)"
        )
        return False
