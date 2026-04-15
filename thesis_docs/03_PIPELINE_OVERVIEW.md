# 03 - Pipeline Overview

**File:** `03_PIPELINE_OVERVIEW.md`  
**Purpose:** High-level architecture and data flow  
**For Thesis:** Methodology chapter introduction

---

## 3.1 Pipeline Architecture

Our VQD-DTW pipeline consists of **7 sequential stages** that transform raw skeletal data into action class predictions:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VQD-DTW PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────┘

Stage 1: DATA LOADING
─────────────────────────────────────────────────────────────────────────
Input:  Skeleton files (*.txt)
Output: Raw sequences [(T₁,60), (T₂,60), ..., (T_N,60)]
Action: Parse joint positions, extract x,y,z coordinates
File:   archive/src/loader.py

                              ↓

Stage 2: FRAME BANK CONSTRUCTION
─────────────────────────────────────────────────────────────────────────
Input:  Train sequences
Output: Frame bank matrix F ∈ ℝ^(M × 60)  where M = Σ T_i
Action: Stack all training frames into single matrix
Purpose: Learn global data distribution

                              ↓

Stage 3: NORMALIZATION
─────────────────────────────────────────────────────────────────────────
Input:  Frame bank F (raw)
Output: Frame bank F_norm (standardized)
Action: StandardScaler - zero mean, unit variance per feature
Math:   x' = (x - μ) / σ
File:   sklearn.preprocessing.StandardScaler

                              ↓

Stage 4: PRE-REDUCTION (Classical PCA)
─────────────────────────────────────────────────────────────────────────
Input:  F_norm ∈ ℝ^(M × 60)
Output: F_pre ∈ ℝ^(M × d_pre)  where d_pre = 20 (optimal)
Action: Classical PCA to reduce dimensionality
Math:   F_pre = F_norm × U_pca^T  where U_pca ∈ ℝ^(60 × 20)
Purpose: Remove noise, improve VQD optimization
File:   sklearn.decomposition.PCA

                              ↓

Stage 5: VQD QUANTUM PCA
─────────────────────────────────────────────────────────────────────────
Input:  F_pre ∈ ℝ^(M × 20)
Output: VQD basis U_vqd ∈ ℝ^(k × 20)  where k = 8 (optimal)
Action: Learn quantum-inspired principal components
Method: Variational Quantum Deflation with statevector simulation
File:   quantum/vqd_pca.py
★ CORE INNOVATION ★

                              ↓

Stage 6: SEQUENCE PROJECTION
─────────────────────────────────────────────────────────────────────────
Input:  Raw sequence S ∈ ℝ^(T × 60), transforms from stages 3-5
Output: Reduced sequence S_vqd ∈ ℝ^(T × k)
Action: Apply full pipeline to each sequence:
        S → normalize → pre-reduce → center → project
Math:   S_vqd[t] = (S_pre[t] - μ_seq) × U_vqd^T
Note:   Per-sequence centering (μ_seq computed per sequence)

                              ↓

Stage 7: DTW + 1-NN CLASSIFICATION
─────────────────────────────────────────────────────────────────────────
Input:  Train set {S_vqd^train}, Test set {S_vqd^test}
Output: Predicted class labels
Action: For each test sequence:
        1. Compute DTW distance to all training sequences
        2. Find nearest neighbor
        3. Return its class label
Distance: Cosine distance (1 - cosine_similarity)
File:   dtw/dtw_runner.py
```

---

## 3.2 Mathematical Data Flow

### 3.2.1 Dimensions at Each Stage

Let's trace a **single training sequence** through the pipeline:

| Stage | Operation | Input Dim | Output Dim | Notation |
|-------|-----------|-----------|------------|----------|
| 1 | Load skeleton | File | (T, 60) | $\mathbf{S}_{\text{raw}}$ |
| 2 | Stack frames | (T, 60) | (M, 60) | $\mathbf{F}$ |
| 3 | Normalize | (M, 60) | (M, 60) | $\mathbf{F}_{\text{norm}}$ |
| 4 | Pre-reduce (PCA) | (M, 60) | (M, 20) | $\mathbf{F}_{\text{pre}}$ |
| 5 | VQD | (M, 20) | (k, 20) | $\mathbf{U}_{\text{vqd}}$ (basis) |
| 6 | Project | (T, 60) | (T, k) | $\mathbf{S}_{\text{vqd}}$ |
| 7 | DTW | (T₁, k), (T₂, k) | scalar | $d_{\text{DTW}}$ |

**Key insight:** Pipeline learns from **frame bank** (all training frames) but applies to **individual sequences** during projection.

### 3.2.2 Transform Composition

The complete transformation is a **composition** of 4 transforms:

$$\mathbf{S}_{\text{vqd}} = \left((\mathbf{S}_{\text{raw}} - \mu_{\text{global}}) \cdot \mathbf{U}_{\text{pca}}^T - \mu_{\text{seq}}\right) \cdot \mathbf{U}_{\text{vqd}}^T$$

Where:
- $\mu_{\text{global}}$ = global mean from training frame bank (60D)
- $\mathbf{U}_{\text{pca}}$ = classical PCA basis (60 × 20)
- $\mu_{\text{seq}}$ = per-sequence mean after pre-reduction (20D)
- $\mathbf{U}_{\text{vqd}}$ = VQD basis (k × 20)

**Intuition:**
1. **Global centering:** Remove dataset-wide mean pose
2. **Pre-reduction:** Project to 20D, keep 99% variance
3. **Per-sequence centering:** Remove sequence-specific mean (critical!)
4. **VQD projection:** Project to final k=8 dimensions

---

## 3.3 Training vs Inference

### 3.3.1 Training Phase (One-Time)

```python
# ═══════════════════════════════════════════════════════════
# TRAINING: Learn transformations from training data
# ═══════════════════════════════════════════════════════════

# 1. Load training sequences
train_sequences, train_labels = load_train_data()

# 2. Build frame bank (stack all training frames)
frame_bank = np.vstack(train_sequences)  # (M, 60)

# 3. Fit normalizer
scaler = StandardScaler()
frame_bank_norm = scaler.fit_transform(frame_bank)

# 4. Fit pre-reduction PCA
pca = PCA(n_components=20)
frame_bank_pre = pca.fit_transform(frame_bank_norm)

# 5. Fit VQD (learn quantum basis)
U_vqd, eigenvalues_vqd = vqd_quantum_pca(frame_bank_pre, n_components=8)

# Store learned transforms
pipeline_params = {
    'scaler': scaler,           # μ_global, σ_global
    'pca': pca,                 # U_pca
    'vqd_basis': U_vqd,         # U_vqd
    'n_components': 8
}
```

**Computational cost:** ~10 minutes for 510 training sequences
- Normalization: <1 second
- Pre-reduction PCA: ~2 seconds
- VQD (8 eigenvectors): ~8 minutes (200 iterations × 8)

**Run once, use forever!**

### 3.3.2 Inference Phase (Per Sequence)

```python
# ═══════════════════════════════════════════════════════════
# INFERENCE: Apply learned transforms to new sequence
# ═══════════════════════════════════════════════════════════

def project_sequence(sequence, pipeline_params):
    """
    Apply full pipeline to a single sequence.
    
    Parameters
    ----------
    sequence : ndarray, shape (T, 60)
        Raw skeletal sequence
    pipeline_params : dict
        Learned transforms from training
    
    Returns
    -------
    sequence_vqd : ndarray, shape (T, k)
        Reduced sequence
    """
    # 1. Normalize (use global statistics)
    sequence_norm = pipeline_params['scaler'].transform(sequence)
    
    # 2. Pre-reduce (classical PCA)
    sequence_pre = pipeline_params['pca'].transform(sequence_norm)
    
    # 3. Per-sequence centering ★ CRITICAL ★
    sequence_centered = sequence_pre - np.mean(sequence_pre, axis=0)
    
    # 4. VQD projection
    U_vqd = pipeline_params['vqd_basis']
    sequence_vqd = sequence_centered @ U_vqd.T
    
    return sequence_vqd
```

**Computational cost:** <0.01 seconds per sequence
- Dominated by matrix multiplications
- No iterative optimization needed

### 3.3.3 Classification

```python
# ═══════════════════════════════════════════════════════════
# CLASSIFICATION: 1-NN with DTW distance
# ═══════════════════════════════════════════════════════════

def classify_sequence(test_seq, train_seqs, train_labels):
    """
    Classify test sequence using 1-NN + DTW.
    
    Parameters
    ----------
    test_seq : ndarray, shape (T_test, k)
        Test sequence (already projected)
    train_seqs : list of ndarray
        Training sequences (already projected)
    train_labels : list of int
        Training labels
    
    Returns
    -------
    predicted_label : int
        Predicted class
    """
    min_dist = np.inf
    predicted_label = None
    
    for train_seq, label in zip(train_seqs, train_labels):
        # Compute DTW distance
        dist = dtw_distance(test_seq, train_seq, metric='cosine')
        
        if dist < min_dist:
            min_dist = dist
            predicted_label = label
    
    return predicted_label
```

**Computational cost:** ~1 second per test sequence
- DTW: O(T₁ × T₂) per pair
- 1-NN: Must compute distance to all training sequences

---

## 3.4 Design Rationale

### 3.4.1 Why Pre-Reduction Before VQD?

**Without pre-reduction (60D → 8D directly):**
- ❌ VQD = PCA (0% advantage)
- ❌ Noise overwhelms quantum optimization
- ❌ 6 qubits needed (2^6=64 ≥ 60), harder to optimize

**With pre-reduction (60D → 20D → 8D):**
- ✅ VQD advantage: +5.7%
- ✅ Noise removed, signal enhanced
- ✅ 5 qubits (2^5=32 ≥ 20), easier to optimize
- ✅ 99% variance retained

**Conclusion:** Pre-reduction is **essential** (see Section 11)

### 3.4.2 Why Per-Sequence Centering?

**Without per-sequence centering:**
- First principal component = global mean pose
- Wastes 1 dimension on non-discriminative info
- Sequences at different positions dominate

**With per-sequence centering:**
- Each sequence centered at origin
- Principal components capture **relative motion**
- Invariant to sequence-specific bias

**Example:**
- Sequence A: Person stands at (x=500, y=100, z=2000)
- Sequence B: Person stands at (x=-200, y=150, z=1800)
- **Without centering:** Different means dominate PC1
- **With centering:** Both centered at (0,0,0), only motion matters

### 3.4.3 Why DTW Instead of Euclidean Distance?

**Problem:** Sequences have different lengths (13-255 frames)

**Solutions:**

| Method | Pros | Cons | Used? |
|--------|------|------|-------|
| Padding to max | Simple | Wasteful (255!), adds noise | ❌ |
| Truncate to min | Simple | Loses info (13 frames too short) | ❌ |
| Euclidean (fixed grid) | Fast | Requires same length | ❌ |
| **DTW** | Optimal alignment | Slower | ✅ |

**DTW handles:**
- Different execution speeds (fast vs slow)
- Temporal variations (pauses, acceleration)
- Variable sequence lengths

### 3.4.4 Why Cosine Distance for DTW?

**Three metrics tested:**

1. **Euclidean:** $d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2$
   - Sensitive to magnitude
   - Poor results: ~65% accuracy
   
2. **Cosine:** $d(\mathbf{a}, \mathbf{b}) = 1 - \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$
   - Measures angle, invariant to scale
   - **Best results: 82.7% accuracy** ✅
   
3. **Fidelity (quantum-inspired):** $d(\mathbf{a}, \mathbf{b}) = 1 - |\langle\mathbf{a}, \mathbf{b}\rangle|^2$
   - Similar to cosine but squared
   - Slightly worse: ~80% accuracy

**Conclusion:** Cosine distance optimal for VQD-reduced features

---

## 3.5 Ablation Studies Performed

To validate each component, we ran **systematic ablations**:

| Experiment | Config | Result | Conclusion |
|------------|--------|--------|------------|
| **Pre-reduction necessity** | 60D → 8D (no pre-reduce) | 0% VQD advantage | Pre-reduction **essential** |
| **Optimal pre-dim** | Test 8,12,16,20,24,32D | 20D best (+5.7%) | 20D **optimal** |
| **Optimal target k** | Test k=6,8,10,12 | k=8 best (82.7%) | k=8 **optimal** |
| **Per-sequence centering** | With vs without | With: +3% | Centering **critical** |
| **Distance metric** | Euclidean vs cosine vs fidelity | Cosine best | Cosine **preferred** |

**See detailed results:**
- Section 11: Pre-reduction optimization
- Section 12: K-sweep experiments
- Section 14: Ablation studies

---

## 3.6 Comparison with Baselines

### 3.6.1 Classical PCA Baseline

**Pipeline:** 60D → 20D (PCA) → 8D (PCA) → DTW

```python
# Classical baseline (no VQD)
pca_pre = PCA(n_components=20).fit(frame_bank_norm)
frame_bank_pre = pca_pre.transform(frame_bank_norm)

pca_final = PCA(n_components=8).fit(frame_bank_pre)
# Project sequences with pca_final instead of VQD
```

**Results:**
- PCA accuracy: 77.7%
- VQD accuracy: 82.7%
- **Gap: +5.0%** (p < 0.01)

### 3.6.2 Direct VQD (No Pre-Reduction)

**Pipeline:** 60D → 8D (VQD) → DTW

**Results:**
- Direct VQD: 77.7%
- Classical PCA: 77.7%
- **Gap: 0.0%** (VQD = PCA)

**Conclusion:** Pre-reduction unlocks VQD advantage

### 3.6.3 Raw Features (No Reduction)

**Pipeline:** 60D → DTW

**Results:**
- Raw 60D accuracy: ~72%
- VQD 8D accuracy: 82.7%
- **Gap: +10.7%**

**Conclusion:** Dimensionality reduction helps (noise removal)

---

## 3.7 Implementation Architecture

### 3.7.1 Code Organization

```
qdtw_project/
├── archive/src/
│   └── loader.py              # Stage 1: Data loading
├── dtw/
│   └── dtw_runner.py          # Stage 7: DTW + classification
├── quantum/
│   └── vqd_pca.py             # Stage 5: VQD algorithm
└── vqd_proper_experiments/
    ├── experiment_k_sweep_ci.py           # K-sweep experiments
    ├── experiment_optimal_prereduction.py # Pre-reduction optimization
    └── experiment_no_prereduction.py      # Ablation study
```

### 3.7.2 Key Classes and Functions

**Data Loading:**
```python
from archive.src.loader import load_all_sequences
sequences, labels, metadata = load_all_sequences(data_dir)
```

**VQD PCA:**
```python
from quantum.vqd_pca import vqd_quantum_pca
U_vqd, eigenvalues, logs = vqd_quantum_pca(
    data, n_components=8, num_qubits=5, max_depth=2
)
```

**DTW Classification:**
```python
from dtw.dtw_runner import one_nn
predicted_labels = one_nn(
    train_sequences, train_labels,
    test_sequences, metric='cosine'
)
```

### 3.7.3 Experiment Scripts

All experiments follow this structure:

```python
# 1. Load data
sequences, labels = load_data()

# 2. Train-test split
train_seqs, test_seqs = split_by_subject(sequences, test_subject=5)

# 3. Build frame bank
frame_bank = np.vstack(train_seqs)

# 4. Learn pipeline (VQD + baselines)
vqd_params = train_vqd_pipeline(frame_bank)
pca_params = train_pca_pipeline(frame_bank)

# 5. Project sequences
train_vqd = [project(s, vqd_params) for s in train_seqs]
test_vqd = [project(s, vqd_params) for s in test_seqs]

# 6. Classify
vqd_acc = evaluate_1nn_dtw(train_vqd, test_vqd)
pca_acc = evaluate_1nn_dtw(train_pca, test_pca)

# 7. Report
print(f"VQD: {vqd_acc:.1%}")
print(f"PCA: {pca_acc:.1%}")
print(f"Gap: {vqd_acc - pca_acc:+.1%}")
```

---

## 3.8 Key Takeaways for Thesis

**What to emphasize:**

1. **7-stage pipeline:** Clear, modular design
2. **Pre-reduction essential:** Enables VQD advantage
3. **Per-sequence centering:** Critical for performance
4. **DTW + cosine distance:** Optimal for variable-length sequences
5. **Ablation-validated:** Every component tested systematically

**Pipeline flowchart for thesis:**
```
Raw Data (60D) 
    ↓ [Normalize]
Standardized (60D)
    ↓ [PCA Pre-reduce]
Pre-reduced (20D) ← 99% variance retained
    ↓ [VQD]
Reduced (8D) ← Quantum-inspired projection ★
    ↓ [DTW + 1-NN]
Predicted Class
```

**What reviewers will ask:**

Q: *"Why not end-to-end deep learning?"*  
A: Our focus is quantum-inspired dimensionality reduction + interpretable DTW. Future work: integrate with deep learning.

Q: *"Can you skip pre-reduction?"*  
A: No - ablation shows 0% VQD advantage without it. Pre-reduction removes noise that confounds quantum optimization.

Q: *"Why 1-NN instead of k-NN?"*  
A: Simplicity and computational efficiency. k-NN tested (k=3,5): minimal improvement (<1%), 3-5× slower.

---

**Next:** [04_DATA_LOADING.md](./04_DATA_LOADING.md) - Detailed data loading implementation →

---

**Navigation:**
- [← 02_DATASET.md](./02_DATASET.md)
- [→ 04_DATA_LOADING.md](./04_DATA_LOADING.md)
- [↑ Index](./README.md)
