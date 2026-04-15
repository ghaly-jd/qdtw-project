# No Pre-Reduction Experiment

**Status:** 🔄 Running  
**Started:** November 25, 2025  
**Purpose:** Test if 60D→16D pre-reduction helps or hurts VQD-DTW performance

---

## 🎯 Motivation

Current pipeline uses **two-stage dimensionality reduction:**
```
60D → 16D (PCA pre-reduction) → kD (PCA/VQD) → DTW
```

This experiment tests **single-stage reduction:**
```
60D → kD (PCA/VQD directly) → DTW
```

**Key Questions:**
1. Does pre-reduction help or hurt final accuracy?
2. Can VQD handle 60D directly? (requires 6 qubits instead of 4)
3. What's the true VQD advantage without intermediate PCA?
4. Is the pre-reduction an unfair advantage for classical PCA?

---

## 🔬 Experimental Setup

### Configuration
- **Input dimension:** 60D (no pre-reduction)
- **Output dimensions (k):** [6, 8, 10, 12]
- **VQD qubits:** 6 (2^6 = 64 ≥ 60)
- **Seeds:** [42] for quick test, expand to [42, 123, 456, 789, 2024] for full validation
- **Train/test:** 300/60 sequences

### Pipeline Comparison

| Stage | With Pre-Reduction | Without Pre-Reduction |
|-------|-------------------|----------------------|
| 1. Normalize | StandardScaler | StandardScaler |
| 2. Pre-reduce | 60D → 16D (PCA) | **SKIP** |
| 3. Learn subspace | 16D → kD (PCA/VQD) | **60D → kD (PCA/VQD)** |
| 4. Per-seq center | ✓ | ✓ |
| 5. DTW classify | 1-NN | 1-NN |

---

## ⏱️ Expected Runtime

**With 6 qubits (60D input):**
- VQD is slower than 4 qubits (16D input)
- Circuit simulation: 2^6 = 64 dimensional state space
- Optimization: ~200 iterations

**Estimates:**
- Single k value: ~10-15 minutes
- Full k-sweep (4 values): ~40-60 minutes  
- Full validation (5 seeds × 4 k): ~3-4 hours

---

## 📊 Expected Outcomes

### Scenario 1: Pre-Reduction Helps
**If with-prereduction > without-prereduction:**
- Pre-reduction removes noise
- 16D is better feature space than 60D
- Current pipeline is optimal

### Scenario 2: Pre-Reduction Hurts
**If without-prereduction > with-prereduction:**
- Pre-reduction loses important information
- VQD can handle 60D directly
- Should use 60D → kD pipeline

### Scenario 3: No Difference
**If accuracies similar:**
- Pre-reduction is computational optimization
- Main benefit is speed (fewer qubits)
- Can choose based on runtime

---

## 📈 Results (In Progress)

### Quick Test (k=8, seed=42)

**Status:** 🔄 Running...

Results will be compared against:
- **With pre-reduction:** PCA 77.7%, VQD 82.7% (from k_sweep_ci_results.json)

**Expected update:** ~15 minutes

### Full Results Table

| Pipeline | K=6 | K=8 | K=10 | K=12 | Average |
|----------|-----|-----|------|------|---------|
| **With Pre-reduction** |  |  |  |  |  |
| PCA | 72.7±4.2% | 77.7±3.8% | 78.0±3.0% | 79.3±3.5% | 76.9% |
| VQD | 77.0±4.6% | 82.7±2.8% | 83.0±4.8% | 83.7±2.7% | 81.6% |
| Gap | +4.3% | +5.0% | +5.0% | +4.3% | +4.7% |
| **Without Pre-reduction** |  |  |  |  |  |
| PCA | TBD | TBD | TBD | TBD | TBD |
| VQD | TBD | TBD | TBD | TBD | TBD |
| Gap | TBD | TBD | TBD | TBD | TBD |

---

## 🔍 Analysis Plan

Once results are available:

1. **Accuracy Comparison**
   - Plot: with-prereduction vs without-prereduction
   - Statistical test: paired t-test across seeds
   - Effect size: Cohen's d

2. **VQD Quality Metrics**
   - Orthogonality error: compare 4-qubit vs 6-qubit
   - Principal angles: vs classical PCA
   - Convergence: training loss curves

3. **Computational Cost**
   - VQD training time: 4-qubit vs 6-qubit
   - Total pipeline time
   - Memory usage

4. **Visualization**
   - Accuracy bar plot (4 configurations)
   - VQD advantage heatmap
   - Runtime comparison

---

## 📝 Monitoring

**Check progress:**
```bash
./monitor_no_prereduction.sh
```

**Watch live output:**
```bash
tail -f logs/no_prereduction.log
```

**Check if running:**
```bash
ps aux | grep experiment_no_prereduction
```

**View results so far:**
```bash
cat results/no_prereduction_results.json
```

---

## 🚀 Next Steps

### After Quick Test (k=8)
1. Analyze single result
2. Compare to baseline (with pre-reduction)
3. Decide: continue with full sweep or adjust parameters

### After Full K-Sweep
1. Complete analysis
2. Generate comparison plots
3. Update EXPERIMENT_GUIDE.md with findings
4. Create presentation figure

### If Results Promising
1. Run full statistical validation (5 seeds)
2. Test on different datasets
3. Write up findings for paper

---

## 💡 Key Insights (To Be Updated)

### Advantages of No Pre-Reduction
- TBD after results

### Disadvantages of No Pre-Reduction  
- Slower VQD training (6 qubits vs 4)
- Higher memory usage
- More complex quantum circuits

### Recommended Pipeline
- TBD based on results

---

**Monitor script:** `monitor_no_prereduction.sh`  
**Results file:** `results/no_prereduction_results.json`  
**Log file:** `logs/no_prereduction.log`
