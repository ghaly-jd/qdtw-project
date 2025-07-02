import numpy as np
from qdtw import qdtw_distance

def classify_knn_quantum(test_seq, train_seqs, train_labels, k=1):
    """
    Classifies a test sequence using k-Nearest Neighbors with quantum DTW distance.
    
    Parameters:
    ----------
    test_seq : array-like
        The test sequence to classify
    train_seqs : list of array-like
        List of training sequences
    train_labels : list
        Labels corresponding to training sequences
    k : int, default=1
        Number of neighbors to consider
        
    Returns:
    -------
    predicted_label : 
        The predicted class label
    """
    # Calculate distances using quantum DTW
    distances = [qdtw_distance(test_seq, train_seq) for train_seq in train_seqs]
    
    # Find k nearest neighbors
    if k == 1:
        nearest_idx = np.argmin(distances)
        return train_labels[nearest_idx]
    else:
        # For k > 1, find k smallest distances and take majority vote
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = [train_labels[i] for i in nearest_indices]
        # Return most common label (majority vote)
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]