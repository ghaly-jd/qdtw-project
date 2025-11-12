# src/dtw.py
import numpy as np
from scipy.spatial.distance import euclidean

def dtw_distance(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = euclidean(seq1[i-1], seq2[j-1])
            dtw[i, j] = cost + min(
                dtw[i-1, j],
                dtw[i, j-1],
                dtw[i-1, j-1]
            )
    return dtw[n, m]
