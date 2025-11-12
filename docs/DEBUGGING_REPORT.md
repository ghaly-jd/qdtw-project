# QDTW Ablation Accuracy Debugging Report
**Date**: November 7, 2025  
**Issue**: Ablation studies showing 3-5% accuracy (random performance) on 20-class problem

---

## Executive Summary

✅ **Root Cause Identified**: The amplitude encoding + PCA pipeline **destroys class discriminability**.

✅ **Evidence**: 
- Raw 60-D data: Inter-class distance (3736) >> Intra-class distance (~1900) ✓ Good separation
- Encoded+PCA 8-D: Inter-class distance (7.52) ≈ Intra-class distance (~7) ✗ No separation

✅ **Solution**: Use raw skeleton data OR fix the encoding approach

---

## Problem Analysis

### 1. Class Separability Test Results

#### Raw 60-D Skeleton Data (BEFORE encoding/PCA)
```
Test samples: 3 sequences from Action 1, 3 sequences from Action 2

Intra-class distances:
  Class 1 (High wave):        1991.33 ± 425.56
  Class 2 (Horizontal wave):  1796.70 ± 544.97

Inter-class distances:
  Class 1 vs Class 2:         3736.01 ± 1241.74

Separation Ratio: 3736 / 1900 ≈ 1.97x
✅ EXCELLENT SEPARATION - Classification should work!
```

#### Encoded+PCA 8-D Data (AFTER amplitude encoding + qPCA)
```
Test samples: Same sequences projected to 8-D subspace

Intra-class distances:
  Class 1 (High wave):        8.15 ± 0.56
  Class 2 (Horizontal wave):  6.34 ± 0.17

Inter-class distances:
  Class 1 vs Class 2:         7.52 ± 1.52

Separation Ratio: 7.52 / 7.2 ≈ 1.04x
⚠️  NO SEPARATION - Random predictions expected!
```

---

## Pipeline Breakdown

### Step 1: Load Raw Skeleton Data
```python
# File: a01_s01_e01_skeleton.txt (Action 1, Subject 1, Instance 1)
# Shape: (54, 60) - 54 frames, 60-D (20 joints × 3 coords)
# Values: Real-world coordinates in mm (range: 0-3000)
# L2 norm per frame: ~3100 (magnitude carries information!)
```

### Step 2: Amplitude Encoding ❌ **PROBLEM HERE**
```python
# File: features/amplitude_encoding.py
# Function: batch_encode_unit_vectors(X)

# BEFORE: X[i] has L2 norm ~3100 (magnitude varies by pose/action)
# AFTER:  X[i] has L2 norm = 1.0 (magnitude information LOST!)

# Example:
#   Action 1, frame 0: [300, 400, ...] → norm=3117 → [0.096, 0.128, ...]
#   Action 2, frame 0: [290, 405, ...] → norm=3122 → [0.093, 0.130, ...]
#
# The normalized vectors are nearly identical!
# Magnitude differences that discriminate actions are removed.
```

**Why this is problematic**:
- Different actions have different joint position magnitudes
- High wave: Arms extended far from body (large coordinates)
- Crouch: Joints closer together (smaller coordinates)
- Normalization to unit vectors **removes this discriminative signal**

### Step 3: PCA on Encoded Data ❌ **COMPOUNDED PROBLEM**
```python
# File: quantum/qpca.py or quantum/classical_pca.py
# Input: Frame bank of shape (N, 60), all rows normalized to ||x|| = 1

# PCA finds directions of maximum variance
# But variance is now ONLY in directional information (angles)
# Magnitude information (which separated classes) is gone

# Result: PCA basis captures noise/variation, not class structure
```

### Step 4: Project Sequences
```python
# File: scripts/project_sequences.py
# seq_projected = seq_encoded @ U

# Now sequences are in 8-D space, but they've lost:
# 1. Magnitude information (removed by encoding)
# 2. Class-discriminative structure (PCA on encoded data didn't capture it)
```

### Step 5: DTW Classification
```python
# File: dtw/dtw_runner.py
# DTW distance in 8-D space

# Problem: All sequences look similar because:
# - Normalization made them all unit-length
# - PCA found non-discriminative directions
# - Result: Random predictions
```

---

## Why Previous "Results" Were Fake

### 1. `eval/aggregate.py` - Synthetic Data Generator
```python
# Line 80-120: create_sample_metrics() function
# Generated FAKE metrics:
#   - Accuracy: 82.99% (completely made up)
#   - Time: Random values
#   - No real classification was performed
```

### 2. `scripts/run_dtw_subspace.py` - Fake Labels
```python
# Line 84 (OLD CODE - BUGGY):
label = i % 20  # Fake label based on sequence index!

# This should have been:
label = metadata['labels'][seq_idx]  # Real label from filename
```

### 3. Ablation Script - Fixed Labels ✅
```python
# File: scripts/run_ablations.py
# Lines 67-86: Proper label loading from metadata
# This revealed the TRUE performance: ~5% accuracy (random)
```

---

## Root Causes Summary

### Primary Cause: Amplitude Encoding Loss
**File**: `features/amplitude_encoding.py`  
**Function**: `batch_encode_unit_vectors()`

**Problem**: Normalizing skeleton frames to unit vectors removes magnitude information that discriminates between action classes.

```python
# LOSSY TRANSFORMATION:
#   Before: frame = [x₁, x₂, ..., x₆₀]  where ||frame|| ≈ 3000
#   After:  frame = [x₁, x₂, ..., x₆₀] / 3000  where ||frame|| = 1

# Information lost: The scale/magnitude of joint positions
# This scale encodes important action properties:
#   - How far joints move from body center
#   - Size of gesture (small vs large motions)
#   - Pose compactness (crouched vs extended)
```

### Secondary Cause: PCA on Encoded Data
**File**: `quantum/qpca.py`, `quantum/classical_pca.py`

**Problem**: PCA computed on unit-normalized frames finds variance in angular directions only, missing the magnitude-based class structure.

```python
# PCA on encoded data:
#   Input: All frames with ||x|| = 1
#   PCA finds: Directions of maximum ANGULAR variance
#   Misses: Magnitude-based discrimination

# What PCA should find: Directions that separate action classes
# What PCA actually finds: Generic pose variations (noise)
```

---

## Solutions

### Solution 1: Use Raw Data (Recommended for Baseline) ✅

**Skip encoding and PCA entirely - use raw 60-D skeleton data**

```python
# File: NEW - scripts/run_dtw_raw.py

from src.loader import load_all_sequences, flatten_sequence
from dtw.dtw_runner import one_nn

# Load raw sequences (no encoding!)
sequences, labels = load_all_sequences("msr_action_data")

# Split train/test
train_seqs, test_seqs, train_labels, test_labels = train_test_split(...)

# Run DTW classification on raw 60-D data
for test_seq, true_label in zip(test_seqs, test_labels):
    pred_label, _ = one_nn(
        train_seqs, train_labels, test_seq, 
        metric='euclidean'
    )
    # Compare pred_label vs true_label

# Expected accuracy: 60-80% (typical for skeleton DTW)
```

**Why this works**:
- Preserves all original information
- Class separability maintained (3736 vs 1900 distances)
- Standard baseline for skeleton action recognition

**Tradeoffs**:
- Slower (60-D DTW vs 8-D DTW)
- No "quantum enhancement"
- But it will actually work!

---

### Solution 2: Alternative Encoding (Preserve Magnitude) ⚠️

**Modify encoding to preserve discriminative information**

#### Option A: No Normalization
```python
# File: features/amplitude_encoding.py (MODIFIED)

def batch_encode_unit_vectors(X: np.ndarray) -> np.ndarray:
    """DON'T normalize - just return original data!"""
    return X.astype(np.float32)
```

#### Option B: Relative Normalization
```python
def batch_encode_with_scale(X: np.ndarray) -> np.ndarray:
    """
    Normalize direction but preserve relative magnitude.
    
    Returns:
        X_encoded: Shape [T, 61] where:
            - First 60 dims: Normalized direction
            - Last dim: Log-scale magnitude
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-12, 1.0, norms)
    
    X_normalized = X / norms_safe
    log_scale = np.log1p(norms)  # log(1 + norm) for stability
    
    return np.hstack([X_normalized, log_scale])
```

#### Option C: Standardization (Not Normalization)
```python
def standardize_frames(X: np.ndarray) -> np.ndarray:
    """
    Standardize to zero mean, unit variance (preserves relative magnitudes).
    """
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-12
    return (X - mean) / std
```

---

### Solution 3: Higher-Dimensional Projection 📊

**If encoding is required, use more dimensions**

Current: k=8 (too few to capture class structure after encoding loss)

Try: k=20, 30, 40, 50

```bash
# Rebuild with higher k
for k in 20 30 40 50; do
    python quantum/classical_pca.py --frames data/frame_bank.npy --k $k --output results/Uc_k${k}.npz
    python scripts/project_sequences.py --pca-file results/Uc_k${k}.npz --output-dir results/subspace/Uc/k${k}
done

# Test higher k values
python scripts/run_ablations.py --k-sweep --n-train 454 --n-test 113
```

**Hypothesis**: With k=40 or 50, enough variance might be preserved despite encoding loss.

---

### Solution 4: Alternative Distance Metrics 🔍

**Test if different metrics help with normalized data**

The ablation already tested:
- Euclidean: 5.31% ❌
- Cosine: 3.54% ❌  
- Fidelity: ~5% ❌

But could try:
- **Mahalanobis distance**: Account for feature correlations
- **Dynamic Time Warping with learned weights**: Weight dimensions by class discrimination
- **Fisher's Linear Discriminant**: Find projection that maximizes class separation

---

## Recommended Action Plan

### Phase 1: Validate Baseline (1-2 hours)

1. **Create raw data classifier** (no encoding/PCA):
   ```bash
   # Create script: scripts/run_dtw_raw.py
   python scripts/run_dtw_raw.py --n-train 454 --n-test 113
   ```

2. **Expected result**: 60-80% accuracy
   - If achieved: Confirms encoding is the problem ✅
   - If not achieved: Indicates deeper issues ⚠️

### Phase 2: Fix Encoding (2-3 hours)

3. **Test no-normalization encoding**:
   ```bash
   # Modify features/amplitude_encoding.py
   # Rebuild frame bank and PCA
   python scripts/build_frame_bank.py
   python quantum/classical_pca.py --k 8 --output results/Uc_k8_raw.npz
   python scripts/project_sequences.py --pca-file results/Uc_k8_raw.npz
   python scripts/run_ablations.py --distance --n-train 100 --n-test 30
   ```

4. **Test alternative encoding schemes** (Options B, C above)

### Phase 3: Optimize (if needed)

5. **Higher dimensionality**: Test k ∈ {20, 30, 40, 50}
6. **Hybrid approach**: Use raw data for classification, PCA only for visualization
7. **Feature engineering**: Add velocity, acceleration, joint angles

---

## Files to Modify

### Critical Files (Must Fix):

1. **`features/amplitude_encoding.py`**
   - Current: Normalizes to unit vectors (LOSSY)
   - Fix: Remove normalization OR add scale preservation

2. **`scripts/build_frame_bank.py`**
   - Line 79: `batch_encode_unit_vectors(frame_bank)`
   - Change to: `frame_bank` (no encoding) OR new encoding function

3. **`scripts/project_sequences.py`** ✅ (Already correct)
   - Just remove encoding step if using raw data

### Files Already Fixed ✅:

1. **`scripts/run_ablations.py`**
   - Lines 67-86: Proper label loading from metadata
   - Lines 220-250: Sanity checks for class distribution

2. **`scripts/create_label_metadata.py`** ✅
   - Creates correct label mappings

### Files to Create:

1. **`scripts/run_dtw_raw.py`** (NEW)
   - DTW classification on raw 60-D data
   - Baseline comparison

---

## Testing Checklist

- [ ] Run raw 60-D DTW classification
- [ ] Verify accuracy > 60% (confirms encoding is the issue)
- [ ] Test no-normalization encoding
- [ ] Test alternative encoding (with scale)
- [ ] Test higher k values (20, 30, 40, 50)
- [ ] Compare all approaches in final ablation study
- [ ] Update README with correct performance metrics
- [ ] Document which approach is best

---

## Expected Outcomes

| Approach | Expected Accuracy | Speed | Quantum? |
|----------|------------------|-------|----------|
| **Raw 60-D DTW** | 60-80% | Slow | No |
| **No-norm + PCA k=8** | 40-60%? | Fast | Partial |
| **No-norm + PCA k=40** | 50-70%? | Medium | Partial |
| **Scale-preserving + PCA k=8** | 50-65%? | Fast | Yes |
| **Current (broken)** | 3-5% | Fast | Yes |

---

## Conclusion

The QDTW pipeline has a fundamental design flaw: **amplitude encoding removes discriminative magnitude information**. This was masked by:

1. Synthetic metrics in `eval/aggregate.py`
2. Fake labels in `scripts/run_dtw_subspace.py` 
3. No proper validation until ablation studies (Nov 7, 2025)

**The fix is straightforward**: Either skip encoding entirely (use raw data) or redesign the encoding to preserve class-relevant information.

The quantum PCA and DTW algorithms themselves are likely correct - the problem is in the preprocessing pipeline that destroys the signal before the algorithms can use it.

---

**Next Steps**: Run Phase 1 (raw data baseline) to confirm this analysis, then proceed with fixes in Phase 2.
