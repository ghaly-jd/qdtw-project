# 10 - Experimental Setup

**File:** `10_EXPERIMENTAL_SETUP.md`  
**Purpose:** Complete experimental protocol and hyperparameters  
**For Thesis:** Experiments chapter - reproducibility

---

## 10.1 Dataset Configuration

**Dataset:** MSR Action3D  
**Sequences:** 567 total (510 train, 57 test)  
**Split:** Cross-subject (Subject 5 held out)  
**Classes:** 20 action classes  
**Raw dimensionality:** 60D (20 joints × 3 coordinates)

---

## 10.2 Hyperparameters

### 10.2.1 Data Preprocessing
```python
# Normalization
normalization = 'StandardScaler'  # Z-score: (x - μ) / σ

# Pre-reduction
pre_dim = 20  # Optimal (Section 11)
variance_threshold = 0.99  # 99% variance retained

# Per-sequence centering
per_sequence_centering = True  # Essential (+3.3%)
```

### 10.2.2 VQD Configuration
```python
# Target dimensionality
n_components = 8  # Optimal k (Section 12)

# Quantum circuit
num_qubits = 5  # ceil(log2(20)) = 5
circuit_depth = 2  # Layers
entanglement = 'alternating'  # Best pattern

# Optimization
optimizer = 'COBYLA'
maxiter = 200  # Per eigenvector
penalty_scale = 10.0  # Orthogonality penalty
ramped_penalties = True  # Increase with r
n_restarts = 1  # First eigenvector
n_restarts_later = 3  # Later eigenvectors
```

### 10.2.3 DTW Classification
```python
# Distance metric
dtw_metric = 'cosine'  # 1 - cos(a, b)

# Classification
method = '1-NN'  # Nearest neighbor
```

---

## 10.3 Statistical Validation

### 10.3.1 Multiple Seeds
```python
seeds = [42, 123, 456, 789, 2024]  # 5 independent runs
```

**Purpose:** Assess stability and significance.

### 10.3.2 Metrics Reported
- Mean accuracy ± standard deviation
- 95% confidence intervals
- Paired t-test p-values (VQD vs PCA)

---

## 10.4 Experiments Conducted

| Experiment | Purpose | Configs | Seeds | Total Runs |
|------------|---------|---------|-------|------------|
| **Pre-reduction sweep** | Find optimal pre-dim | 6 | 5 | 30 |
| **K-sweep** | Find optimal target k | 4 | 5 | 20 |
| **No pre-reduction** | Validate necessity | 1 | 5 | 5 |
| **Ablations** | Component validation | 3 | 5 | 15 |
| **Total** | — | — | — | **70** |

---

## 10.5 Computational Environment

```python
hardware = {
    'CPU': 'Intel i7-10700K @ 3.8GHz (8 cores)',
    'RAM': '32 GB DDR4',
    'GPU': 'None (not needed for statevector simulation)',
    'OS': 'Ubuntu 22.04 LTS'
}

software = {
    'Python': '3.10.12',
    'NumPy': '1.24.3',
    'scikit-learn': '1.3.0',
    'Qiskit': '0.45.1',
    'Qiskit Aer': '0.13.1',
    'SciPy': '1.11.3'
}
```

**Training time:** ~10 minutes per VQD configuration  
**Total experimental time:** ~12 hours (70 runs × 10 min)

---

## 10.6 Reproducibility

All experiments use **deterministic** settings:
```python
np.random.seed(seed)  # NumPy RNG
# Qiskit statevector is deterministic (no shots)
```

**Results are exactly reproducible** given same seed.

---

**Next:** [11_PREREDUCTION_OPTIMIZATION.md](./11_PREREDUCTION_OPTIMIZATION.md) →

---

**Navigation:**
- [← 09_DTW_CLASSIFICATION.md](./09_DTW_CLASSIFICATION.md)
- [→ 11_PREREDUCTION_OPTIMIZATION.md](./11_PREREDUCTION_OPTIMIZATION.md)
- [↑ Index](./README.md)
