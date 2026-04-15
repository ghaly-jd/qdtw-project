# Validation Plan

## 🎯 Goal
Verify if VQD's superior performance (85% vs 68% baseline, +16.7%) is:
1. Consistent across different random seeds
2. Due to fair/unfair comparison of projection methods
3. Reproducible and statistically significant

---

## 📋 Validation Steps

### Step 1: Verify Projection Methodology ✓ Created
**Script**: `verify_projection.py`

**What it checks**:
- Are PCA and VQD using same centering approach?
- Per-sequence centering (VQD) vs global centering (PCA)
- Impact on feature distributions

**Expected outcome**:
- Identify if projection inconsistency explains the gap
- Recommend fix if needed

**Runtime**: ~2-3 minutes

---

### Step 2: Cross-Validation ✓ Created
**Script**: `cross_validate.py`

**What it tests**:
- 5 different random seeds (42, 43, 44, 45, 46)
- 3 key k values (4, 8, 12)
- Total: 15 runs (5 seeds × 3 k values)

**Metrics tracked**:
- Mean accuracy ± std for Baseline, PCA, VQD
- VQD-PCA gap consistency
- VQD-Baseline gap consistency
- Principal angles variation

**Success criteria**:
- ✅ VQD beats PCA in ≥80% of runs
- ✅ VQD beats baseline in ≥80% of runs
- ✅ Standard deviation < 10% across seeds
- ✅ Consistent pattern across k values

**Runtime**: ~30-40 minutes (15 runs × 2-3 min each)

---

### Step 3: Fair Comparison Re-run (If Needed)
**Depends on**: Step 1 findings

If projection inconsistency found:
1. Fix centering to be consistent
2. Re-run experiment with aligned methods
3. Compare new results

---

## 🚀 Execution Plan

### Phase 1: Quick Verification (Now)
```bash
# Check projection methodology
python verify_projection.py

# Expected: 2-3 minutes
# Output: Identifies centering difference
```

**Decision point**: If major inconsistency found → fix before cross-validation

---

### Phase 2: Cross-Validation (After Phase 1)
```bash
# Run cross-validation
python cross_validate.py | tee logs/cross_validation.log

# Expected: 30-40 minutes
# Output: Statistical validation of findings
```

**Decision point**: If results hold → findings validated!

---

### Phase 3: Fair Re-run (If Needed)
Only if Phase 1 reveals unfair comparison:
1. Create `vqd_dtw_proper_fixed.py` with aligned projections
2. Re-run full experiment
3. Compare to original results

---

## 📊 Expected Outcomes

### Scenario A: Findings Hold (Best Case)
- Cross-validation shows VQD consistently outperforms
- Small standard deviation across seeds
- VQD advantages are real and reproducible
- **Action**: Proceed to write paper!

### Scenario B: Projection Issue Found
- Step 1 reveals centering inconsistency
- VQD uses per-sequence, PCA uses global
- Need to align and re-test
- **Action**: Fix and re-run

### Scenario C: High Variance
- Cross-validation shows inconsistent results
- Large std deviation across seeds
- VQD advantage is seed-dependent
- **Action**: More investigation needed

### Scenario D: Results Don't Hold
- VQD doesn't consistently beat baseline
- Original result was lucky seed
- **Action**: Re-evaluate approach

---

## 🎯 Let's Start!

**Recommended order**:
1. ✅ Run `verify_projection.py` first (2-3 min)
2. ✅ Review findings
3. ✅ Decide if fix needed
4. ✅ Run `cross_validate.py` (30-40 min)
5. ✅ Analyze results

**Which step would you like to start with?**

Option A: Projection verification (quick, 2-3 min)
Option B: Cross-validation (long, 30-40 min)
Option C: Both in sequence (best approach)
