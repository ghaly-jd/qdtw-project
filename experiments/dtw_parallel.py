"""
Parallelized CPU DTW for Experiment 3
Uses multiprocessing to speed up DTW computations
"""

import numpy as np
from functools import partial
from multiprocessing import Pool, cpu_count


def _dtw_distance(seq1, seq2):
    """Fast DTW distance computation"""
    n, m = len(seq1), len(seq2)
    
    # Flatten if needed
    if seq1.ndim > 1:
        seq1 = seq1.flatten()
    if seq2.ndim > 1:
        seq2 = seq2.flatten()
    
    # Compute cost matrix
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.abs(seq1[i-1] - seq2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return dtw[n, m]


def _compute_single_prediction(test_sample, X_train, y_train):
    """Compute prediction for a single test sample"""
    min_dist = float('inf')
    min_label = None
    
    for x_train, y_train_i in zip(X_train, y_train):
        dist = _dtw_distance(test_sample, x_train)
        if dist < min_dist:
            min_dist = dist
            min_label = y_train_i
    
    return min_label


class DTWClassifierParallel:
    """DTW 1-NN classifier with multiprocessing parallelization"""
    
    def __init__(self, n_jobs=-1):
        """
        Args:
            n_jobs: Number of parallel jobs (-1 uses all CPUs)
        """
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()
        self.X_train = None
        self.y_train = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the classifier (just store training data)"""
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict using 1-NN with DTW in parallel"""
        # Create partial function with fixed training data
        predict_fn = partial(_compute_single_prediction, 
                            X_train=self.X_train, 
                            y_train=self.y_train)
        
        # Parallelize across test samples
        with Pool(processes=self.n_jobs) as pool:
            predictions = pool.map(predict_fn, X_test)
        
        return np.array(predictions)
    
    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Return accuracy"""
        predictions = self.predict(X_test)
        return float(np.mean(predictions == y_test))


if __name__ == "__main__":
    import time
    
    print(f"🧪 Testing Parallel DTW classifier with {cpu_count()} CPUs...")
    
    # Generate dummy data
    np.random.seed(42)
    X_train = np.random.randn(100, 16).astype(np.float32)
    y_train = np.random.randint(0, 20, 100)
    X_test = np.random.randn(30, 16).astype(np.float32)
    y_test = np.random.randint(0, 20, 30)
    
    # Test parallel classifier
    clf = DTWClassifierParallel(n_jobs=-1)
    clf.fit(X_train, y_train)
    
    start = time.time()
    acc = clf.score(X_test, y_test)
    elapsed = time.time() - start
    
    print(f"✅ Parallel DTW test complete")
    print(f"   Accuracy: {acc:.3f}")
    print(f"   Time: {elapsed:.2f} s")
    print(f"   Speed: {len(X_test) * len(X_train) / elapsed:.0f} comparisons/sec")
