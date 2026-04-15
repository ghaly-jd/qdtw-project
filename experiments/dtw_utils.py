"""Experiment utilities: DTW runner with timing"""

import numpy as np
from typing import Callable


def dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    distance_fn: Callable[[np.ndarray, np.ndarray], float] = None
) -> float:
    """
    Compute DTW distance between two sequences
    
    Args:
        seq1: First sequence (n1, d)
        seq2: Second sequence (n2, d)
        distance_fn: Point-wise distance function (default: Euclidean)
    
    Returns:
        DTW distance (scalar)
    """
    if distance_fn is None:
        distance_fn = lambda x, y: np.linalg.norm(x - y)
    
    n1, n2 = len(seq1), len(seq2)
    
    # Initialize cost matrix
    cost = np.full((n1 + 1, n2 + 1), np.inf)
    cost[0, 0] = 0
    
    # Fill cost matrix
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            d = distance_fn(seq1[i-1], seq2[j-1])
            cost[i, j] = d + min(
                cost[i-1, j],     # insertion
                cost[i, j-1],     # deletion
                cost[i-1, j-1]    # match
            )
    
    return cost[n1, n2]


def dtw_path(
    seq1: np.ndarray,
    seq2: np.ndarray,
    distance_fn: Callable[[np.ndarray, np.ndarray], float] = None
) -> tuple:
    """
    Compute DTW distance and alignment path
    
    Returns:
        distance, path (list of (i,j) tuples)
    """
    if distance_fn is None:
        distance_fn = lambda x, y: np.linalg.norm(x - y)
    
    n1, n2 = len(seq1), len(seq2)
    
    # Initialize cost matrix
    cost = np.full((n1 + 1, n2 + 1), np.inf)
    cost[0, 0] = 0
    
    # Fill cost matrix
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            d = distance_fn(seq1[i-1], seq2[j-1])
            cost[i, j] = d + min(
                cost[i-1, j],     # insertion
                cost[i, j-1],     # deletion
                cost[i-1, j-1]    # match
            )
    
    # Backtrack to find path
    path = []
    i, j = n1, n2
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        
        # Find minimum predecessor
        candidates = [
            (cost[i-1, j-1], (i-1, j-1)),
            (cost[i-1, j], (i-1, j)),
            (cost[i, j-1], (i, j-1))
        ]
        _, (i, j) = min(candidates, key=lambda x: x[0])
    
    path.reverse()
    
    return cost[n1, n2], path
