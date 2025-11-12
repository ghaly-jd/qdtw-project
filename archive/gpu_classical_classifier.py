import cupy as cp
import numpy as np
from scipy.spatial.distance import cdist

def _dtw_distance_gpu_fair(seq1_gpu, seq2_gpu):
    """
    GPU DTW using same optimization level as quantum (scipy cdist)
    """
    # Convert to CPU for scipy (same as quantum method)
    seq1 = seq1_gpu.get()
    seq2 = seq2_gpu.get()
    
    # Use scipy's optimized distance matrix (same as quantum)
    cost_matrix = cdist(seq1, seq2, metric='euclidean')
    
    n, m = cost_matrix.shape
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw[0, 0] = 0
    
    # Same DTW computation as quantum
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_matrix[i - 1, j - 1]
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return dtw[n, m]

def classify_knn_classical_gpu(test_seq, train_seqs_gpu, train_labels, k=1):
    """
    Fair classical classifier using same optimization level as quantum
    """
    test_seq_gpu = cp.array(test_seq, dtype=cp.float32)
    
    min_dist = float('inf')
    best_label = None
    
    for train_seq_gpu, label in zip(train_seqs_gpu, train_labels):
        dist = _dtw_distance_gpu_fair(test_seq_gpu, train_seq_gpu)  # Same optimization level
        
        if dist < min_dist:
            min_dist = dist
            best_label = label
    
    return best_label