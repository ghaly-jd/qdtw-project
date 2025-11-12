"""
Dynamic Time Warping (DTW) for k-D sequences.

This module provides DTW distance computation with multiple metrics:
- Cosine distance
- Euclidean distance  
- Fidelity distance

Supports:
- Windowing constraints for efficiency
- 1-NN classification
- Multiple distance metrics
"""

import logging
from typing import Optional, Tuple

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine distance between two vectors.

    Cosine distance = 1 - cosine_similarity
    where cosine_similarity = (a · b) / (||a|| ||b||)

    Args:
        a: Vector of shape [k]
        b: Vector of shape [k]

    Returns:
        distance: Cosine distance in [0, 2]
                 0 = identical direction
                 1 = orthogonal
                 2 = opposite direction

    Example:
        >>> a = np.array([1, 0, 0])
        >>> b = np.array([1, 0, 0])
        >>> cosine_distance(a, b)
        0.0
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Handle zero vectors
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0  # Orthogonal as default

    cos_sim = np.dot(a, b) / (norm_a * norm_b)
    # Clamp to [-1, 1] to avoid numerical issues
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    return 1.0 - cos_sim


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean (L2) distance between two vectors.

    Args:
        a: Vector of shape [k]
        b: Vector of shape [k]

    Returns:
        distance: L2 distance >= 0

    Example:
        >>> a = np.array([0, 0])
        >>> b = np.array([3, 4])
        >>> euclidean_distance(a, b)
        5.0
    """
    return np.linalg.norm(a - b)


def fidelity_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute fidelity-based distance between two vectors.

    Fidelity distance = 1 - |<a_hat, b_hat>|²
    where a_hat = a / ||a||, b_hat = b / ||b|| (L2-normalized)

    This is inspired by quantum state fidelity, measuring
    overlap of normalized states.

    Args:
        a: Vector of shape [k]
        b: Vector of shape [k]

    Returns:
        distance: Fidelity distance in [0, 1]
                 0 = identical direction
                 1 = orthogonal

    Example:
        >>> a = np.array([2, 0])
        >>> b = np.array([3, 0])
        >>> fidelity_distance(a, b)  # Same direction
        0.0
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Handle zero vectors
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0  # Maximum distance

    a_hat = a / norm_a
    b_hat = b / norm_b

    overlap = np.abs(np.dot(a_hat, b_hat))
    fidelity = overlap ** 2

    return 1.0 - fidelity


def dtw_distance(
    seqA: np.ndarray,
    seqB: np.ndarray,
    metric: str = "cosine",
    window: Optional[int] = None
) -> float:
    """
    Compute Dynamic Time Warping distance between two sequences.

    DTW finds the optimal alignment between two time series by
    allowing flexible warping of the time axis.

    Args:
        seqA: First sequence of shape [T1, k]
        seqB: Second sequence of shape [T2, k]
        metric: Distance metric, one of:
                - "cosine": 1 - cosine_similarity
                - "euclidean": L2 distance
                - "fidelity": 1 - |<a_hat, b_hat>|²
        window: Optional Sakoe-Chiba window constraint.
                If provided, only consider alignments within
                |i-j| <= window. None = no constraint.

    Returns:
        distance: DTW distance >= 0

    Algorithm:
        Standard DTW using dynamic programming.
        Cost[i, j] = frame_dist(A[i], B[j]) + min(
            Cost[i-1, j],    # insertion
            Cost[i, j-1],    # deletion
            Cost[i-1, j-1]   # match
        )

    Example:
        >>> seqA = np.random.randn(100, 8)
        >>> seqB = np.random.randn(120, 8)
        >>> dist = dtw_distance(seqA, seqB, metric="euclidean")
        >>> print(f"DTW distance: {dist:.2f}")
    """
    # Select distance function
    if metric == "cosine":
        dist_func = cosine_distance
    elif metric == "euclidean":
        dist_func = euclidean_distance
    elif metric == "fidelity":
        dist_func = fidelity_distance
    else:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Choose from: cosine, euclidean, fidelity"
        )

    T1, k1 = seqA.shape
    T2, k2 = seqB.shape

    if k1 != k2:
        raise ValueError(
            f"Sequences must have same dimension. "
            f"Got seqA: {k1}, seqB: {k2}"
        )

    # Handle edge cases
    if T1 == 0 or T2 == 0:
        raise ValueError("Cannot compute DTW for empty sequences")

    # Ensure window is valid
    if window is not None and window < 0:
        window = None  # Treat negative window as no constraint

    # Initialize cost matrix
    # Use float32 for memory efficiency
    cost = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0

    # Fill cost matrix with DP
    for i in range(1, T1 + 1):
        # Determine valid j range based on window
        if window is None:
            j_start = 1
            j_end = T2 + 1
        else:
            # Sakoe-Chiba band: |i-j| <= window
            j_start = max(1, i - window)
            j_end = min(T2 + 1, i + window + 1)

        for j in range(j_start, j_end):
            # Frame distance
            frame_dist = dist_func(seqA[i - 1], seqB[j - 1])

            # DTW recurrence
            cost[i, j] = frame_dist + min(
                cost[i - 1, j],      # insertion
                cost[i, j - 1],      # deletion
                cost[i - 1, j - 1]   # match
            )

    return float(cost[T1, T2])


def one_nn(
    train_seqs: list[np.ndarray],
    train_labels: list[int],
    test_seq: np.ndarray,
    metric: str = "cosine",
    window: Optional[int] = None
) -> Tuple[int, float]:
    """
    Perform 1-Nearest Neighbor classification using DTW distance.

    Finds the training sequence with minimum DTW distance to the
    test sequence and returns its label.

    Args:
        train_seqs: List of training sequences, each shape [T_i, k]
        train_labels: List of integer labels, one per training sequence
        test_seq: Test sequence of shape [T_test, k]
        metric: Distance metric (cosine, euclidean, or fidelity)
        window: Optional DTW window constraint

    Returns:
        predicted_label: Label of nearest neighbor
        min_distance: DTW distance to nearest neighbor

    Example:
        >>> train_seqs = [seq1, seq2, seq3]
        >>> train_labels = [0, 1, 1]
        >>> test_seq = seq_new
        >>> pred, dist = one_nn(train_seqs, train_labels, test_seq)
        >>> print(f"Predicted: {pred}, Distance: {dist:.2f}")
    """
    if len(train_seqs) != len(train_labels):
        raise ValueError(
            f"Number of sequences ({len(train_seqs)}) must match "
            f"number of labels ({len(train_labels)})"
        )

    if len(train_seqs) == 0:
        raise ValueError("Training set is empty")

    min_distance = np.inf
    predicted_label = -1

    for seq, label in zip(train_seqs, train_labels):
        dist = dtw_distance(test_seq, seq, metric=metric, window=window)

        if dist < min_distance:
            min_distance = dist
            predicted_label = label

    return predicted_label, float(min_distance)
