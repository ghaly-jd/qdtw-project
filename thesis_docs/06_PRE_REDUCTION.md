# 06 - Pre-Reduction (Classical PCA)

**File:** `06_PRE_REDUCTION.md`  
**Purpose:** Classical PCA for noise removal before VQD  
**For Thesis:** Methodology - critical preprocessing step

---

## 6.1 Role of Pre-Reduction

**Purpose:** Reduce dimensionality from 60D → 20D using classical PCA **before** applying VQD.

**Why necessary:**
1. **Noise removal:** Discard small eigenvalues (1% variance = noise)
2. **Improve VQD optimization:** Cleaner input → better convergence
3. **Computational efficiency:** 5 qubits (20D) vs 6 qubits (60D)

**Key finding:** Pre-reduction is **essential** for VQD advantage (Section 11, 18.2).

---

## 6.2 Principal Component Analysis (PCA)

### 6.2.1 Mathematical Foundation

Given normalized data $\mathbf{X} \in \mathbb{R}^{M \times 60}$, PCA finds:

$$\mathbf{C} = \frac{1}{M-1} \mathbf{X}^T \mathbf{X}$$

Eigendecomposition:
$$\mathbf{C} = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^T$$

Where:
- $\mathbf{U} \in \mathbb{R}^{60 \times 60}$: Orthonormal eigenvectors
- $\mathbf{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_{60})$: Eigenvalues (sorted: $\lambda_1 \geq \lambda_2 \geq \cdots$)

**Projection:** Take first $d$ eigenvectors:
$$\mathbf{X}_{\text{reduced}} = \mathbf{X} \cdot \mathbf{U}_{:d}$$

---

## 6.3 Implementation

```python
from sklearn.decomposition import PCA
import numpy as np

def apply_pre_reduction(frame_bank_norm, n_components=20, verbose=True):
    """
    Apply classical PCA pre-reduction.
    
    Parameters
    ----------
    frame_bank_norm : ndarray, shape (M, 60)
        Normalized frame bank
    n_components : int
        Target dimensionality (default: 20)
    
    Returns
    -------
    frame_bank_pre : ndarray, shape (M, n_components)
        Pre-reduced frame bank
    pca : PCA
        Fitted PCA model
    """
    pca = PCA(n_components=n_components)
    frame_bank_pre = pca.fit_transform(frame_bank_norm)
    
    if verbose:
        cumulative_var = np.cumsum(pca.explained_variance_ratio_)
        print(f"\nPre-reduction: 60D → {n_components}D")
        print(f"  Variance retained: {cumulative_var[-1]:.1%}")
        print(f"  Top 5 eigenvalues: {pca.explained_variance_[:5]}")
    
    return frame_bank_pre, pca
```

---

## 6.4 Optimal Dimensionality: 20D

### 6.4.1 Why 20D?

**Empirical sweep:** Tested {8, 12, 16, 20, 24, 32}D (Section 11).

| Pre-Dim | Variance | VQD Gap | Conclusion |
|---------|----------|---------|------------|
| 8D      | 94.2%    | 0.0%    | Too small (info loss) |
| 16D     | 98.1%    | +4.3%   | Good but suboptimal |
| **20D** | **99.0%** | **+5.7%** | ✓ **Optimal** |
| 32D     | 99.6%    | +1.8%   | Too large (noise) |

**20D balances:**
- Signal preservation: 99% variance
- Noise removal: 1% discarded
- VQD advantage: Maximum at +5.7%

### 6.4.2 Eigenvalue Spectrum

```python
# Analyze full spectrum
pca_full = PCA(n_components=60).fit(frame_bank_norm)
eigenvalues = pca_full.explained_variance_
cumulative = np.cumsum(pca_full.explained_variance_ratio_)

# Plot
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Scree plot
axes[0].plot(range(1, 61), eigenvalues, 'o-')
axes[0].axvline(20, color='red', linestyle='--', label='20D (optimal)')
axes[0].set_xlabel('Component')
axes[0].set_ylabel('Eigenvalue')
axes[0].set_title('Scree Plot')
axes[0].set_yscale('log')
axes[0].legend()

# Cumulative variance
axes[1].plot(range(1, 61), cumulative * 100)
axes[1].axhline(99, color='green', linestyle='--', label='99% threshold')
axes[1].axvline(20, color='red', linestyle='--', label='20D')
axes[1].set_xlabel('Components')
axes[1].set_ylabel('Cumulative Variance (%)')
axes[1].set_title('Variance Retention')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/eigenvalue_spectrum.png', dpi=300)
```

**Key observations:**
- Sharp drop after component 20 ("elbow")
- Components 21-60: Each <0.5% variance
- 20D: 99.0% variance (sweet spot)

---

## 6.5 Signal vs Noise Interpretation

### 6.5.1 Large Eigenvalues = Signal

**Components 1-20** (99% variance):
- Capture global body motion
- Consistent across subjects
- Discriminative for action recognition

**Examples:**
- PC1: Overall body translation (X,Y,Z movement)
- PC2: Vertical motion (jumping, bending)
- PC3: Arm extension patterns
- PC4-20: Finer motion details

### 6.5.2 Small Eigenvalues = Noise

**Components 21-60** (1% variance):
- Sensor noise (Kinect tracking errors)
- Subject-specific jitter
- Non-reproducible variations

**Evidence:**
- Keeping 32D vs 20D: **Worse** VQD accuracy (-3.9%)
- Extra dimensions hurt, don't help
- Confirms small eigenvalues = noise

---

## 6.6 Comparison: With vs Without Pre-Reduction

### 6.6.1 Direct VQD (No Pre-Reduction)

**Pipeline:** 60D → 8D (VQD only)

```python
# NO pre-reduction
U_vqd_direct, _ = vqd_quantum_pca(frame_bank_norm, n_components=8)
acc_vqd = evaluate(U_vqd_direct)  # 77.7%
acc_pca = evaluate(U_pca_direct)  # 77.7%
# Gap: 0.0% (VQD = PCA)
```

### 6.6.2 With Pre-Reduction

**Pipeline:** 60D → 20D (PCA) → 8D (VQD)

```python
# WITH pre-reduction
frame_bank_pre, pca_pre = apply_pre_reduction(frame_bank_norm, n_components=20)
U_vqd, _ = vqd_quantum_pca(frame_bank_pre, n_components=8)
acc_vqd = evaluate(U_vqd)  # 83.4%
acc_pca = evaluate(U_pca)  # 77.7%
# Gap: +5.7% (VQD advantage!)
```

**Conclusion:** Pre-reduction **enables** VQD advantage.

---

## 6.7 Projecting New Sequences

### 6.7.1 Training Sequences

```python
def project_train_sequence_prereduction(sequence, scaler, pca_pre):
    """
    Project training sequence through pre-reduction.
    
    Parameters
    ----------
    sequence : ndarray, shape (T, 60)
        Raw sequence
    scaler : StandardScaler
        Fitted normalizer
    pca_pre : PCA
        Fitted pre-reduction PCA
    
    Returns
    -------
    sequence_pre : ndarray, shape (T, 20)
        Pre-reduced sequence
    """
    # 1. Normalize
    sequence_norm = scaler.transform(sequence)
    
    # 2. Pre-reduce
    sequence_pre = pca_pre.transform(sequence_norm)
    
    return sequence_pre
```

### 6.7.2 Test Sequences

**Critical:** Use training PCA (don't refit!)

```python
# CORRECT: Use training PCA
test_seq_pre = pca_pre.transform(test_seq_norm)

# WRONG: Don't do this!
pca_test = PCA(n_components=20).fit(test_seq_norm)  # ❌
test_seq_pre = pca_test.transform(test_seq_norm)
```

---

## 6.8 Visualization of Pre-Reduction

### 6.8.1 Feature Space Visualization

```python
from sklearn.manifold import TSNE

# Project to 2D for visualization
frame_bank_2d = TSNE(n_components=2).fit_transform(frame_bank_pre)

# Color by action class
plt.figure(figsize=(10, 8))
scatter = plt.scatter(frame_bank_2d[:, 0], frame_bank_2d[:, 1],
                     c=frame_labels, cmap='tab20', alpha=0.5, s=1)
plt.colorbar(scatter, label='Action Class')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('Pre-Reduced Feature Space (20D → 2D via t-SNE)')
plt.tight_layout()
plt.savefig('figures/prereduction_tsne.png', dpi=300)
```

**Expected:** Some class clustering visible (but not perfect - that's where VQD helps!).

---

## 6.9 Key Takeaways

**Main insights:**

1. ✅ **Pre-reduction is essential**
   - Without it: 0% VQD advantage
   - With it: +5.7% advantage

2. ✅ **20D is optimal**
   - 99% variance retained
   - 1% noise removed
   - Maximum VQD benefit

3. ✅ **Classical PCA as denoiser**
   - Not competing with VQD
   - Complementary: cleans input for VQD
   - Hybrid classical-quantum pipeline

4. ✅ **Use training PCA for all data**
   - Fit once on training
   - Apply to train and test
   - Never refit on test

**For thesis defense:**

Q: *"Why not skip PCA and use VQD directly?"*  
A: Tried that (Section 18.2) - 0% advantage. Noise confounds VQD optimization. Pre-reduction removes noise, enabling VQD to find better subspace.

Q: *"Is 20D arbitrary?"*  
A: No - systematically tested 6 dimensions (Section 11). 20D is data-driven optimal (99% variance, maximum VQD gap).

Q: *"Could you learn pre-reduction end-to-end?"*  
A: Future work (Section 20.6.2) - trainable pre-reduction with gradient-based VQD. Currently infeasible (VQD non-differentiable).

---

**Next:** [07_VQD_QUANTUM_PCA.md](./07_VQD_QUANTUM_PCA.md) - Quantum dimensionality reduction →

---

**Navigation:**
- [← 05_NORMALIZATION.md](./05_NORMALIZATION.md)
- [→ 07_VQD_QUANTUM_PCA.md](./07_VQD_QUANTUM_PCA.md)
- [↑ Index](./README.md)
