# 08 - Sequence Projection

**File:** `08_SEQUENCE_PROJECTION.md`  
**Purpose:** Applying learned transforms to individual sequences  
**For Thesis:** Methodology - inference pipeline

---

## 8.1 Overview

After learning transforms (scaler, pre-reduction PCA, VQD), we must **project** individual sequences through the full pipeline.

**Key insight:** Per-sequence centering is **critical** (+3.3% improvement).

---

## 8.2 Full Projection Pipeline

### 8.2.1 Four-Stage Transform

```python
def project_sequence(sequence, pipeline_params):
    """
    Project sequence through full pipeline.
    
    Pipeline: Raw → Normalize → Pre-reduce → Center → VQD
    
    Parameters
    ----------
    sequence : ndarray, shape (T, 60)
        Raw sequence
    pipeline_params : dict
        {'scaler': StandardScaler,
         'pca_pre': PCA,
         'vqd_basis': ndarray}
    
    Returns
    -------
    sequence_vqd : ndarray, shape (T, k)
        Final reduced sequence
    """
    # Stage 1: Normalize (global statistics)
    sequence_norm = pipeline_params['scaler'].transform(sequence)
    
    # Stage 2: Pre-reduce (classical PCA)
    sequence_pre = pipeline_params['pca_pre'].transform(sequence_norm)
    
    # Stage 3: Per-sequence centering ★ CRITICAL ★
    sequence_mean = np.mean(sequence_pre, axis=0)
    sequence_centered = sequence_pre - sequence_mean
    
    # Stage 4: VQD projection
    U_vqd = pipeline_params['vqd_basis']
    sequence_vqd = sequence_centered @ U_vqd.T
    
    return sequence_vqd
```

---

## 8.3 Per-Sequence Centering (Critical!)

### 8.3.1 Why It's Necessary

**Problem without per-sequence centering:**

Different sequences have different mean poses:
- Subject A (tall): Mean pose at (0, 500, 2000)
- Subject B (short): Mean pose at (0, 400, 1800)

**Result:** First principal component captures **position differences**, not motion!

**Solution:** Center each sequence at origin before projection.

### 8.3.2 Mathematical Formulation

For sequence $\mathbf{S} \in \mathbb{R}^{T \times d}$:

$$\bar{\mathbf{s}} = \frac{1}{T} \sum_{t=1}^T \mathbf{s}_t$$

$$\mathbf{S}_{\text{centered}} = \mathbf{S} - \mathbf{1}_T \bar{\mathbf{s}}^T$$

Where $\mathbf{1}_T$ is column vector of ones.

**Interpretation:** Translate sequence so temporal mean = origin.

### 8.3.3 Implementation Details

```python
# Compute mean pose
sequence_mean = np.mean(sequence_pre, axis=0, keepdims=True)  # (1, 20)

# Broadcast subtraction
sequence_centered = sequence_pre - sequence_mean  # (T, 20) - (1, 20) → (T, 20)

# Verify centering
assert np.allclose(np.mean(sequence_centered, axis=0), 0, atol=1e-10)
```

### 8.3.4 Ablation Study Result

| Configuration | Accuracy | Notes |
|---------------|----------|-------|
| **With per-seq centering** | **83.4%** | ✓ Optimal |
| Without per-seq centering | 80.1% | -3.3% drop |
| Global centering only | 80.1% | Same as above |

**Conclusion:** Per-sequence centering adds **+3.3% absolute** improvement.

---

## 8.4 When to Center: Global vs Per-Sequence

### 8.4.1 Two Centering Stages

**1. Global centering (during normalization):**
```python
# Removes dataset-wide mean
scaler = StandardScaler()  # Does: x' = (x - μ_global) / σ
frame_bank_norm = scaler.fit_transform(frame_bank)
```

**Purpose:** Remove absolute position bias across entire dataset.

**2. Per-sequence centering (during projection):**
```python
# Removes sequence-specific mean
sequence_centered = sequence_pre - np.mean(sequence_pre, axis=0)
```

**Purpose:** Remove sequence-specific position bias.

### 8.4.2 Why Both?

**Global centering alone:**
- Removes dataset mean (~zero-centered on average)
- But individual sequences still have different means
- Example: After global centering, Subject A mean = (+50, +20, +10), Subject B mean = (-30, -15, -8)

**Per-sequence centering:**
- Removes remaining sequence-specific bias
- Every sequence centered at exact origin
- Only relative motion remains

**Analogy:**
- Global: "Remove population average height"
- Per-sequence: "Make each person start at ground level"

---

## 8.5 Complete Projection Functions

### 8.5.1 For Training Sequences

```python
def project_train_sequences(sequences, pipeline_params):
    """
    Project all training sequences.
    
    Parameters
    ----------
    sequences : list of ndarray
        Raw training sequences
    pipeline_params : dict
        Learned transforms
    
    Returns
    -------
    sequences_vqd : list of ndarray
        Projected sequences
    """
    sequences_vqd = []
    
    for seq in sequences:
        seq_vqd = project_sequence(seq, pipeline_params)
        sequences_vqd.append(seq_vqd)
    
    return sequences_vqd
```

### 8.5.2 For Test Sequences

```python
def project_test_sequences(sequences, pipeline_params):
    """
    Project test sequences (same as training).
    
    Note: Uses SAME learned transforms, no refitting!
    """
    return project_train_sequences(sequences, pipeline_params)
```

**Key:** Test projection is identical to training projection (just different data).

---

## 8.6 Projection Mathematics

### 8.6.1 Full Transform Equation

Starting from raw sequence $\mathbf{S}_{\text{raw}} \in \mathbb{R}^{T \times 60}$:

$$\mathbf{S}_{\text{vqd}} = \left(\left(\mathbf{S}_{\text{raw}} - \mu_{\text{global}}\right) \cdot \mathbf{U}_{\text{pca}}^T - \bar{\mathbf{s}}_{\text{pre}}\right) \cdot \mathbf{U}_{\text{vqd}}^T$$

Where:
- $\mu_{\text{global}} \in \mathbb{R}^{60}$: Global mean from StandardScaler
- $\mathbf{U}_{\text{pca}} \in \mathbb{R}^{60 \times 20}$: Pre-reduction basis
- $\bar{\mathbf{s}}_{\text{pre}} \in \mathbb{R}^{20}$: Per-sequence mean (after pre-reduction)
- $\mathbf{U}_{\text{vqd}} \in \mathbb{R}^{8 \times 20}$: VQD basis

**Result:** $\mathbf{S}_{\text{vqd}} \in \mathbb{R}^{T \times 8}$

### 8.6.2 Step-by-Step Example

```python
# Example sequence: 42 frames × 60 features
S_raw = load_sequence('a05_s03_e01_skeleton.txt')  # (42, 60)

# Step 1: Normalize
S_norm = scaler.transform(S_raw)  # (42, 60)
# Each feature: zero mean, unit variance (globally)

# Step 2: Pre-reduce
S_pre = pca_pre.transform(S_norm)  # (42, 20)
# Keeps 99% variance, removes noise

# Step 3: Per-sequence center
S_mean = np.mean(S_pre, axis=0)  # (20,)
S_centered = S_pre - S_mean  # (42, 20)
# Sequence mean now at origin

# Step 4: VQD project
S_vqd = S_centered @ U_vqd.T  # (42, 8)
# Final 8D representation

# Verify shapes
assert S_vqd.shape == (42, 8)
```

---

## 8.7 Batch vs Sequential Projection

### 8.7.1 Sequential (One at a Time)

```python
# Project sequences one by one
sequences_vqd = []
for seq in sequences:
    seq_vqd = project_sequence(seq, pipeline_params)
    sequences_vqd.append(seq_vqd)
```

**Pros:** Simple, low memory  
**Cons:** Slower (no vectorization)

### 8.7.2 Batch Processing

```python
def project_sequences_batch(sequences, pipeline_params, batch_size=32):
    """
    Project sequences in batches for efficiency.
    """
    sequences_vqd = []
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        
        # Project batch
        batch_vqd = [project_sequence(s, pipeline_params) for s in batch]
        sequences_vqd.extend(batch_vqd)
    
    return sequences_vqd
```

**Pros:** Can leverage parallelization  
**Cons:** More complex, higher memory

**Our choice:** Sequential (sequences have different lengths, hard to batch).

---

## 8.8 Computational Cost

### 8.8.1 Per-Sequence Projection Time

For sequence of length $T$:

1. **Normalize:** $O(T \times 60)$ (matrix subtraction + division)
2. **Pre-reduce:** $O(T \times 60 \times 20)$ (matrix multiplication)
3. **Center:** $O(T \times 20)$ (mean + subtraction)
4. **VQD project:** $O(T \times 20 \times 8)$ (matrix multiplication)

**Total:** $O(T \times 60 \times 20) \approx O(1200 T)$ operations

**Example:** $T=42$ frames → ~50,000 ops → **<0.01 seconds on CPU**

### 8.8.2 Full Dataset Projection

For 567 sequences (avg length 42 frames):
- Total: 567 × 0.01 sec ≈ **6 seconds**
- Negligible compared to VQD training (9 minutes)

---

## 8.9 Error Handling

### 8.9.1 Shape Validation

```python
def project_sequence_safe(sequence, pipeline_params):
    """
    Project sequence with error checking.
    """
    # Check input shape
    if sequence.ndim != 2:
        raise ValueError(f"Expected 2D array, got {sequence.ndim}D")
    
    if sequence.shape[1] != 60:
        raise ValueError(f"Expected 60 features, got {sequence.shape[1]}")
    
    # Check for invalid values
    if not np.all(np.isfinite(sequence)):
        raise ValueError("Sequence contains NaN or Inf")
    
    # Project
    try:
        sequence_vqd = project_sequence(sequence, pipeline_params)
    except Exception as e:
        print(f"Projection failed: {e}")
        raise
    
    # Validate output
    assert sequence_vqd.shape[0] == sequence.shape[0], "Frame count mismatch"
    assert sequence_vqd.shape[1] == pipeline_params['vqd_basis'].shape[0], "Dimension mismatch"
    
    return sequence_vqd
```

---

## 8.10 Visualization of Projection

### 8.10.1 Dimensionality Reduction Visualization

```python
import matplotlib.pyplot as plt

# Project example sequence through stages
seq_raw = sequences[0]  # (T, 60)

seq_norm = scaler.transform(seq_raw)  # (T, 60)
seq_pre = pca_pre.transform(seq_norm)  # (T, 20)
seq_centered = seq_pre - np.mean(seq_pre, axis=0)  # (T, 20)
seq_vqd = seq_centered @ U_vqd.T  # (T, 8)

# Plot first 2 dimensions at each stage
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(seq_raw[:, 0], seq_raw[:, 1])
axes[0, 0].set_title('Raw (60D) - First 2 features')

axes[0, 1].plot(seq_norm[:, 0], seq_norm[:, 1])
axes[0, 1].set_title('Normalized (60D)')

axes[1, 0].plot(seq_pre[:, 0], seq_pre[:, 1])
axes[1, 0].set_title('Pre-reduced (20D)')

axes[1, 1].plot(seq_vqd[:, 0], seq_vqd[:, 1])
axes[1, 1].set_title('VQD (8D) ★')

for ax in axes.flat:
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/projection_stages.png', dpi=300)
```

---

## 8.11 Key Takeaways

**Critical insights:**

1. ✅ **Per-sequence centering is essential**
   - +3.3% accuracy improvement
   - Makes features position-invariant
   - Zero computational overhead

2. ✅ **Apply learned transforms, never refit**
   - Use training scaler, PCA, VQD for all data
   - Consistent feature space across train/test
   - Realistic evaluation

3. ✅ **Four-stage pipeline**
   - Normalize → Pre-reduce → Center → VQD
   - Each stage serves specific purpose
   - All stages necessary (ablation confirmed)

4. ✅ **Fast inference**
   - <0.01 sec per sequence
   - No iterative optimization at test time
   - Real-time capable

**For thesis defense:**

Q: *"Why center each sequence separately?"*  
A: Different subjects have different body sizes and camera distances. Per-sequence centering removes position bias, keeping only relative motion. Without it: -3.3% accuracy (Section 18.5).

Q: *"Doesn't centering lose information?"*  
A: We lose absolute position (which varies with subject/camera). We keep relative motion (which is discriminative for actions). This is intentional and beneficial.

Q: *"Can you project sequences in parallel?"*  
A: Yes, projection is embarrassingly parallel (no inter-sequence dependencies). Could use multiprocessing for large datasets.

---

**Next:** [09_DTW_CLASSIFICATION.md](./09_DTW_CLASSIFICATION.md) - Dynamic Time Warping →

---

**Navigation:**
- [← 07_VQD_QUANTUM_PCA.md](./07_VQD_QUANTUM_PCA.md)
- [→ 09_DTW_CLASSIFICATION.md](./09_DTW_CLASSIFICATION.md)
- [↑ Index](./README.md)
