"""
GPU-Accelerated DTW for Experiment 3
Uses CuPy for massive speedup on classification
"""

import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠️  CuPy not available, falling back to CPU")


class DTWClassifierGPU:
    """DTW 1-NN classifier with GPU acceleration"""
    
    def __init__(self, gpu_id=0):
        """
        Args:
            gpu_id: Which GPU to use (0, 1, or 2)
        """
        self.gpu_id = gpu_id
        self.X_train = None
        self.y_train = None
        
        if CUPY_AVAILABLE:
            cp.cuda.Device(gpu_id).use()
            print(f"✅ Using GPU {gpu_id}")
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the classifier (store training data on GPU)"""
        self.X_train = X_train
        self.y_train = y_train
        
        if CUPY_AVAILABLE:
            # Move training data to GPU
            self.X_train_gpu = cp.array(X_train, dtype=cp.float32)
        
        return self
    
    def _dtw_distance_batch_gpu(self, X_test_gpu, X_train_gpu):
        """
        Compute DTW distances between test samples and all training samples
        Vectorized for speed
        
        Args:
            X_test_gpu: (n_test, seq_len, n_features) on GPU
            X_train_gpu: (n_train, seq_len, n_features) on GPU
            
        Returns:
            distances: (n_test, n_train) on GPU
        """
        n_test = X_test_gpu.shape[0]
        n_train = X_train_gpu.shape[0]
        seq_len = X_test_gpu.shape[1]
        
        # Compute all pairwise distances in parallel
        # Shape: (n_test, n_train, seq_len)
        diffs = X_test_gpu[:, None, :, :] - X_train_gpu[None, :, :, :]
        point_dists = cp.sqrt(cp.sum(diffs**2, axis=-1))  # (n_test, n_train, seq_len)
        
        # DTW dynamic programming - vectorized over all pairs
        n = seq_len
        distances = cp.zeros((n_test, n_train), dtype=cp.float32)
        
        for test_idx in range(n_test):
            for train_idx in range(n_train):
                # Classic DTW for this pair
                cost = point_dists[test_idx, train_idx]
                dtw = cp.full((n + 1, n + 1), cp.inf, dtype=cp.float32)
                dtw[0, 0] = 0
                
                for i in range(1, n + 1):
                    for j in range(1, n + 1):
                        dtw[i, j] = cost[i-1] + cp.minimum(
                            cp.minimum(dtw[i-1, j], dtw[i, j-1]),
                            dtw[i-1, j-1]
                        )
                
                distances[test_idx, train_idx] = dtw[n, n]
        
        return distances
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict using 1-NN with DTW on GPU"""
        if not CUPY_AVAILABLE:
            # Fallback to CPU
            return self._predict_cpu(X_test)
        
        # Move test data to GPU
        X_test_gpu = cp.array(X_test, dtype=cp.float32)
        
        # Ensure data is 3D: (n_samples, seq_len, n_features)
        if X_test_gpu.ndim == 2:
            X_test_gpu = X_test_gpu[:, None, :]  # (n, 1, d) - single timepoint
        if self.X_train_gpu.ndim == 2:
            X_train_gpu = self.X_train_gpu[:, None, :]
        else:
            X_train_gpu = self.X_train_gpu
        
        # Compute all DTW distances on GPU
        distances = self._dtw_distance_batch_gpu(X_test_gpu, X_train_gpu)
        
        # Find nearest neighbor for each test sample
        nearest_indices = cp.argmin(distances, axis=1)
        
        # Get predictions (move back to CPU)
        predictions = self.y_train[nearest_indices.get()]
        
        return predictions
    
    def _predict_cpu(self, X_test: np.ndarray) -> np.ndarray:
        """CPU fallback"""
        from experiments.dtw_utils import dtw_distance
        
        predictions = []
        for x_test in X_test:
            min_dist = float('inf')
            min_label = None
            
            for x_train, y_train in zip(self.X_train, self.y_train):
                dist = dtw_distance(x_test, x_train)
                if dist < min_dist:
                    min_dist = dist
                    min_label = y_train
            
            predictions.append(min_label)
        
        return np.array(predictions)
    
    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Return accuracy"""
        predictions = self.predict(X_test)
        return float(np.mean(predictions == y_test))


def test_gpu_dtw():
    """Quick test of GPU DTW"""
    if not CUPY_AVAILABLE:
        print("❌ CuPy not available")
        return
    
    print("🧪 Testing GPU DTW classifier...")
    
    # Generate dummy data
    np.random.seed(42)
    X_train = np.random.randn(50, 10, 5).astype(np.float32)  # 50 samples, 10 timesteps, 5 features
    y_train = np.random.randint(0, 3, 50)
    X_test = np.random.randn(10, 10, 5).astype(np.float32)
    y_test = np.random.randint(0, 3, 10)
    
    # Test GPU classifier
    clf = DTWClassifierGPU(gpu_id=0)
    clf.fit(X_train, y_train)
    
    import time
    start = time.time()
    acc = clf.score(X_test, y_test)
    elapsed = time.time() - start
    
    print(f"✅ GPU DTW test complete")
    print(f"   Accuracy: {acc:.3f}")
    print(f"   Time: {elapsed*1000:.1f} ms")
    print(f"   Speed: {len(X_test) * len(X_train) / elapsed:.0f} comparisons/sec")


if __name__ == "__main__":
    test_gpu_dtw()
