import cupy as cp
import numpy as np

def _dtw_distance_gpu_classical(seq1_gpu, seq2_gpu):
    """
    Classical DTW implementation fully on GPU (fair comparison)
    """
    n, m = len(seq1_gpu), len(seq2_gpu)
    
    # Compute distance matrix on GPU
    cost_matrix = cp.sqrt(cp.sum((seq1_gpu[:, None, :] - seq2_gpu[None, :, :])**2, axis=-1))
    
    # DTW computation on GPU
    dtw = cp.full((n + 1, m + 1), cp.inf, dtype=cp.float32)
    dtw[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_matrix[i - 1, j - 1]
            # Use GPU operations for minimum
            dtw[i, j] = cost + cp.minimum(cp.minimum(dtw[i-1, j], dtw[i, j-1]), dtw[i-1, j-1])
    
    return dtw[n, m]

def dtw_distance_gpu(seq1, seq2):
    """
    Pure GPU classical DTW distance computation.
    """
    seq1_gpu = cp.array(seq1, dtype=cp.float32)
    seq2_gpu = cp.array(seq2, dtype=cp.float32)
    
    result = _dtw_distance_gpu_classical(seq1_gpu, seq2_gpu)
    return float(result.get())