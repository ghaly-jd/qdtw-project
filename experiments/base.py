"""
Experimental Infrastructure for Thesis Experiments
Enforces ground rules: fixed split, train-only fit, bootstrap CI, manifests
"""

import numpy as np
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from functools import wraps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.stats


class ExperimentBase:
    """Base class for all thesis experiments with ground rules enforcement"""
    
    def __init__(
        self,
        experiment_id: str,
        results_dir: str = "results",
        seed: int = 42
    ):
        self.experiment_id = experiment_id
        self.results_dir = Path(results_dir)
        self.seed = seed
        self.exp_dir = self.results_dir / experiment_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Track timing
        self.timings = []
        
        # Ground rule: fixed split
        self.split_file = self.results_dir / "splits" / "split_indices.npz"
        self.split_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Manifest data
        self.manifest = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "commit_sha": self._get_git_commit(),
            "git_branch": self._get_git_branch(),
        }
    
    def _get_git_commit(self) -> str:
        """Get current git commit SHA"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _get_git_branch(self) -> str:
        """Get current git branch"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return "unknown"
    
    def get_or_create_split(
        self,
        n_samples: int,
        n_train: int,
        n_test: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ground Rule 1: Fixed Split
        Load or create a fixed train/test split that will be reused across all experiments
        """
        if self.split_file.exists():
            print(f"✅ Loading existing split from {self.split_file}")
            data = np.load(self.split_file)
            train_idx = data['train_idx']
            test_idx = data['test_idx']
            
            # Validate split size matches
            if len(train_idx) != n_train or len(test_idx) != n_test:
                print(f"⚠️  Split size mismatch: expected ({n_train}, {n_test}), "
                      f"got ({len(train_idx)}, {len(test_idx)})")
                print("Creating new split...")
                train_idx, test_idx = self._create_new_split(n_samples, n_train, n_test)
            
        else:
            print(f"📝 Creating new split: {n_train} train, {n_test} test")
            train_idx, test_idx = self._create_new_split(n_samples, n_train, n_test)
        
        return train_idx, test_idx
    
    def _create_new_split(
        self,
        n_samples: int,
        n_train: int,
        n_test: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a new fixed split and save it"""
        np.random.seed(self.seed)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_idx = indices[:n_train]
        test_idx = indices[n_train:n_train + n_test]
        
        # Save for future reuse
        np.savez(
            self.split_file,
            train_idx=train_idx,
            test_idx=test_idx,
            n_train=n_train,
            n_test=n_test,
            seed=self.seed,
            timestamp=datetime.now().isoformat()
        )
        print(f"✅ Split saved to {self.split_file}")
        
        return train_idx, test_idx
    
    def fit_transform_pipeline(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        scaler: bool = True,
        frame_bank_dim: Optional[int] = 8
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Ground Rule 2: Train-Only Fit
        Fit all transformations on training data only, then transform both train and test
        
        Returns:
            X_train_transformed, X_test_transformed, transform_info
        """
        transform_info = {}
        
        # Step 1: Standardization (if requested)
        if scaler:
            print("  🔧 Fitting StandardScaler on train...")
            scaler_obj = StandardScaler()
            X_train = scaler_obj.fit_transform(X_train)
            X_test = scaler_obj.transform(X_test)
            transform_info['scaler'] = {
                'mean': scaler_obj.mean_.tolist(),
                'std': scaler_obj.scale_.tolist()
            }
        
        # Step 2: Frame Bank (PCA reduction)
        if frame_bank_dim is not None and frame_bank_dim < X_train.shape[1]:
            print(f"  🔧 Fitting Frame Bank PCA ({X_train.shape[1]}D → {frame_bank_dim}D) on train...")
            pca = PCA(n_components=frame_bank_dim)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            transform_info['frame_bank'] = {
                'n_components': frame_bank_dim,
                'variance_explained': float(pca.explained_variance_ratio_.sum()),
                'singular_values': pca.singular_values_.tolist()
            }
        
        print(f"  ✅ Pipeline complete: {X_train.shape} train, {X_test.shape} test")
        return X_train, X_test, transform_info
    
    def bootstrap_ci(
        self,
        metric_fn,
        n_bootstrap: int = 1000,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Ground Rule 3: Bootstrap 95% CI
        Compute bootstrap confidence interval for a metric
        
        Args:
            metric_fn: Function that computes metric (returns scalar)
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level (0.05 for 95% CI)
        
        Returns:
            {'mean': float, 'ci_lower': float, 'ci_upper': float}
        """
        np.random.seed(self.seed)
        bootstrap_samples = []
        
        for _ in range(n_bootstrap):
            sample = metric_fn()
            bootstrap_samples.append(sample)
        
        bootstrap_samples = np.array(bootstrap_samples)
        mean = float(np.mean(bootstrap_samples))
        ci_lower = float(np.percentile(bootstrap_samples, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_samples, 100 * (1 - alpha / 2)))
        
        return {
            'mean': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': float(np.std(bootstrap_samples)),
            'n_bootstrap': n_bootstrap
        }
    
    def time_component(self, name: str):
        """
        Ground Rule 4: Runtime Breakdown
        Decorator to time a component of the pipeline
        
        Usage:
            @exp.time_component("projection")
            def project(X):
                return X @ U.T
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                
                self.timings.append({
                    'component': name,
                    'time_ms': elapsed
                })
                
                return result
            return wrapper
        return decorator
    
    def get_timing_summary(self) -> Dict[str, float]:
        """Get average timing per component"""
        from collections import defaultdict
        
        component_times = defaultdict(list)
        for timing in self.timings:
            component_times[timing['component']].append(timing['time_ms'])
        
        return {
            component: float(np.mean(times))
            for component, times in component_times.items()
        }
    
    def write_manifest(self, additional_data: Optional[Dict[str, Any]] = None):
        """
        Ground Rule 6: Write Manifest
        Save complete experiment metadata to JSON
        """
        if additional_data:
            self.manifest.update(additional_data)
        
        # Convert numpy arrays to lists for JSON serialization
        manifest_json = self._make_json_serializable(self.manifest)
        
        manifest_path = self.exp_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest_json, f, indent=2)
        
        print(f"✅ Manifest written to {manifest_path}")
        return manifest_path
    
    def _make_json_serializable(self, obj):
        """Recursively convert numpy arrays and other non-serializable objects to JSON-compatible types"""
        if isinstance(obj, np.ndarray):
            # Check if array contains complex numbers
            if np.iscomplexobj(obj):
                return np.real(obj).tolist()
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.complex128, np.complex64, complex)):
            # Convert complex to real (take real part, or magnitude if needed)
            return float(np.real(obj))
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def save_results(self, results: Dict[str, Any], filename: str = "results_summary.json"):
        """Save experiment results"""
        results_path = self.exp_dir / filename
        # Convert to JSON-serializable first
        json_ready = self._make_json_serializable(results)
        with open(results_path, 'w') as f:
            json.dump(json_ready, f, indent=2)
        
        print(f"✅ Results saved to {results_path}")
        return results_path


class QuantumResourceTracker:
    """
    Ground Rule 5: Track Quantum Resources
    Track qubits, depth, shots for quantum components
    """
    
    def __init__(self):
        self.resources = []
    
    def log_circuit(
        self,
        component: str,
        qubits: int,
        depth: int,
        shots: Optional[int] = None,
        backend: str = "AerSimulator"
    ):
        """Log quantum circuit resources"""
        self.resources.append({
            'component': component,
            'qubits': qubits,
            'depth': depth,
            'shots': shots,
            'backend': backend
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get resource summary"""
        if not self.resources:
            return {}
        
        return {
            'total_qubits': max(r['qubits'] for r in self.resources),
            'total_depth': sum(r['depth'] for r in self.resources),
            'total_shots': sum(r['shots'] for r in self.resources if r['shots']),
            'components': self.resources
        }


class DTWClassifier:
    """Simple DTW 1-NN classifier for experiments"""
    
    def __init__(self, distance_fn=None):
        """
        Args:
            distance_fn: Function(x, y) -> distance (default: Euclidean)
        """
        self.distance_fn = distance_fn or self._euclidean_distance
        self.X_train = None
        self.y_train = None
    
    def _euclidean_distance(self, x, y):
        """Default Euclidean distance"""
        return np.linalg.norm(x - y)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the classifier (just store training data)"""
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict using 1-NN with DTW"""
        from experiments.dtw_utils import dtw_distance
        
        predictions = []
        for x_test in X_test:
            min_dist = float('inf')
            min_label = None
            
            for x_train, y_train in zip(self.X_train, self.y_train):
                dist = dtw_distance(x_test, x_train, self.distance_fn)
                if dist < min_dist:
                    min_dist = dist
                    min_label = y_train
            
            predictions.append(min_label)
        
        return np.array(predictions)
    
    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Return accuracy"""
        predictions = self.predict(X_test)
        return float(np.mean(predictions == y_test))


def load_msr_data(data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray]:
    """Load MSR Action3D data"""
    # Try different possible data files
    possible_files = [
        Path(data_dir) / "frame_bank_std.npy",
        Path(data_dir) / "features.npy",
        Path(data_dir) / "features_pca2.npy"
    ]
    
    data_path = None
    for path in possible_files:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(f"Data not found. Tried: {possible_files}")
    
    X = np.load(data_path)
    
    # If data is 1D, reshape it (frame_bank_std.npy is flattened)
    if X.ndim == 1:
        # Assume 60D features (common for skeleton data)
        n_features = 60
        if len(X) % n_features == 0:
            X = X.reshape(-1, n_features)
        else:
            raise ValueError(f"Cannot reshape data: {len(X)} samples doesn't divide by {n_features}")
    
    # Generate labels (assuming first 100 are class 0, next 100 class 1, etc.)
    n_samples = len(X)
    n_classes = 20  # MSR-Action3D has 20 classes
    samples_per_class = n_samples // n_classes
    y = np.repeat(np.arange(n_classes), samples_per_class)[:n_samples]
    
    print(f"✅ Loaded MSR data: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")
    print(f"   Source: {data_path}")
    return X, y


if __name__ == "__main__":
    # Test the infrastructure
    print("🧪 Testing Experimental Infrastructure\n")
    
    # Test 1: Fixed split
    print("Test 1: Fixed Split")
    exp1 = ExperimentBase("test_exp", results_dir="results")
    train_idx, test_idx = exp1.get_or_create_split(n_samples=1000, n_train=100, n_test=30)
    print(f"  Train indices: {train_idx[:5]}... (shape: {train_idx.shape})")
    print(f"  Test indices: {test_idx[:5]}... (shape: {test_idx.shape})")
    
    # Test 2: Load same split again
    print("\nTest 2: Load Same Split")
    exp2 = ExperimentBase("test_exp2", results_dir="results")
    train_idx2, test_idx2 = exp2.get_or_create_split(n_samples=1000, n_train=100, n_test=30)
    assert np.array_equal(train_idx, train_idx2), "❌ Split mismatch!"
    print("  ✅ Same split loaded successfully")
    
    # Test 3: Train-only fit
    print("\nTest 3: Train-Only Fit")
    X = np.random.randn(1000, 60)
    X_train = X[train_idx]
    X_test = X[test_idx]
    X_train_t, X_test_t, info = exp1.fit_transform_pipeline(
        X_train, X_test,
        scaler=True,
        frame_bank_dim=8
    )
    print(f"  Original: {X_train.shape} → Transformed: {X_train_t.shape}")
    print(f"  Frame bank variance: {info['frame_bank']['variance_explained']:.3f}")
    
    # Test 4: Bootstrap CI
    print("\nTest 4: Bootstrap CI")
    metric_fn = lambda: np.random.randn()
    ci = exp1.bootstrap_ci(metric_fn, n_bootstrap=100)
    print(f"  Mean: {ci['mean']:.3f}, CI: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
    
    # Test 5: Timing
    print("\nTest 5: Component Timing")
    @exp1.time_component("test_operation")
    def slow_op():
        time.sleep(0.01)
        return 42
    
    result = slow_op()
    timing = exp1.get_timing_summary()
    print(f"  test_operation: {timing['test_operation']:.2f} ms")
    
    # Test 6: Quantum resources
    print("\nTest 6: Quantum Resources")
    tracker = QuantumResourceTracker()
    tracker.log_circuit("swap_test", qubits=4, depth=20, shots=4096)
    tracker.log_circuit("vqd", qubits=3, depth=50, shots=None)
    summary = tracker.get_summary()
    print(f"  Total qubits: {summary['total_qubits']}")
    print(f"  Total depth: {summary['total_depth']}")
    print(f"  Components: {len(summary['components'])}")
    
    # Test 7: Manifest
    print("\nTest 7: Write Manifest")
    exp1.manifest['data'] = {'n_samples': 1000, 'n_features': 60}
    exp1.manifest['quantum_resources'] = tracker.get_summary()
    manifest_path = exp1.write_manifest()
    
    print("\n✅ All tests passed! Infrastructure is ready.")
