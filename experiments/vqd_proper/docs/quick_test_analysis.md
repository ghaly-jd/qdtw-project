# Quick Test Results - Analysis

## 📊 Results Summary

| Method | k | Accuracy | Speedup | Notes |
|--------|---|----------|---------|-------|
| **Baseline** | 60 | 68.3% | 1.0× | Full 60D sequences |
| **PCA** | 4 | 61.7% | 1.01× | Classical PCA |
| **VQD** | 4 | 76.7% | 0.84× | Quantum PCA |

---

## 🔍 Unexpected Findings

### ⚠️ Issue 1: Baseline Lower Than Expected
- **Expected**: ~75%
- **Actual**: 68.3%
- **Analysis**: Could be due to:
  - Random seed variation
  - Different train/test split
  - Data preprocessing differences
  - DTW metric (using euclidean, might need to verify)

### ⚠️ Issue 2: VQD Outperforms Both!
- **VQD (76.7%)** > Baseline (68.3%) > PCA (61.7%)
- **This is unusual** - VQD shouldn't beat the full 60D baseline
- **Possible causes**:
  1. VQD found a better discriminative subspace by chance
  2. Overfitting to the small test set (60 samples)
  3. Different preprocessing/centering between methods
  4. Random luck in this split

### ⚠️ Issue 3: Large Principal Angles
- **Max angle**: 89.9° (almost orthogonal to PCA!)
- **Mean angle**: 43.1°
- **Analysis**: VQD found a very different subspace than PCA
  - This explains why VQD accuracy is so different from PCA
  - VQD isn't approximating PCA - it found something else

### ⚠️ Issue 4: No Speedup from Dimensionality Reduction
- **PCA speedup**: 1.01× (essentially the same)
- **VQD speedup**: 0.84× (actually slower!)
- **Analysis**: 
  - Going from 60D to 4D should give ~5-8× speedup
  - Suggests DTW computation is not dimension-dependent here
  - Possible overhead from other factors

---

## 🤔 What's Happening?

### Theory 1: Preprocessing Differences
The VQD projection includes explicit centering:
```python
seq_proj = (seq_reduced - mean) @ U_vqd.T
```

While PCA uses sklearn's built-in transform which may handle centering differently. This could create different feature spaces.

### Theory 2: Lucky Subspace
With only k=4 dimensions and 60 test samples, VQD might have found a subspace that happens to separate these particular test samples well by chance.

### Theory 3: DTW Implementation
The DTW distance computation might not scale with dimensionality as expected. Need to verify the DTW implementation.

---

## ✅ What Worked Well

1. **Orthogonality**: 2.48e-16 - EXCELLENT! ✅
2. **VQD Convergence**: Completed successfully ✅
3. **Pipeline**: All components work end-to-end ✅
4. **Speed**: Test completed in 5.3 minutes ✅

---

## 🎯 Recommendations

### Option 1: Continue with Full Run (RECOMMENDED)
**Rationale**: 
- Pipeline works correctly
- Need more k values to see the trend
- Single k=4 result might be anomalous
- Full run will reveal if this pattern holds

**Action**:
```bash
python vqd_dtw_proper.py | tee logs/full_run.log
```

### Option 2: Debug Baseline First
**Rationale**: 
- 68.3% baseline is lower than expected 75%
- Should verify this isn't a data issue

**Action**:
1. Check if baseline matches literature on this split
2. Try different random seed
3. Verify DTW implementation

### Option 3: Investigate VQD Projection
**Rationale**: 
- VQD beating baseline is suspicious
- Large angles suggest different subspace
- May need to verify projection consistency

**Action**:
1. Add debug prints to see actual projections
2. Compare feature distributions
3. Check if centering is consistent

---

## 📈 What to Expect in Full Run

If pattern holds:
- VQD might consistently find different (possibly better?) subspaces
- Large angles will persist
- Accuracy trends across k values will be revealing

If k=4 was anomalous:
- Higher k values will show VQD converging toward PCA
- Angles will decrease with k
- VQD will track PCA accuracy more closely

---

## 🚀 My Recommendation

**Proceed with Full Run** - Here's why:

1. ✅ Pipeline works correctly (orthogonality perfect)
2. ✅ Computational time is reasonable (5.3 min for one k)
3. ✅ Need full k-sweep to understand the pattern
4. ✅ Single k=4 result is insufficient to draw conclusions

The unexpected results are actually **interesting** and worth investigating with the full experiment!

**Ready to run the full experiment?** 

Full run command:
```bash
cd /path/to/qdtw_project/vqd_proper_experiments
python vqd_dtw_proper.py | tee logs/full_run.log
```

This will test k ∈ {4, 6, 8, 10, 12} and take ~30-40 minutes.
