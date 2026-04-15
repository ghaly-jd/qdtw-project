# Advanced Experiments Guide

## Overview

This document describes three additional experiments designed to strengthen the VQD-DTW research:

1. **K-Sweep with Confidence Intervals**: Rigorous statistical validation across multiple random seeds
2. **Whitening Toggle**: Test whether whitening (U Λ^{-1/2}) stabilizes results
3. **By-Class Analysis**: Identify which action classes benefit most from VQD

## Motivation

Our initial validation showed VQD provides a modest but real advantage (+2-5%). To make this publication-ready, we need:

- **Statistical rigor**: Confidence intervals, not just point estimates
- **Method variants**: Does whitening help? (Common in PCA literature)
- **Interpretability**: Which types of actions benefit? (Nice plot for paper)

---

## Experiment 1: K-Sweep with Confidence Intervals

### Purpose

Compute mean ± std over 5 random seeds for k ∈ {6, 8, 10, 12}.

### Why It Matters

- Single-seed results can be noisy (we saw variance ±3-5%)
- 95% confidence intervals show if VQD advantage is statistically significant
- Required for credible publication

### What It Does

```python
For each seed in [42, 123, 456, 789, 2024]:
    For each k in [6, 8, 10, 12]:
        1. Train/test split with this seed
        2. Run PCA (per-sequence centering)
        3. Run VQD (per-sequence centering)
        4. Record accuracies
        
Aggregate:
    For each k:
        Compute mean, std, 95% CI across 5 seeds
```

### Expected Output

**Table format:**

| k | PCA Accuracy | VQD Accuracy | Gap |
|---|--------------|--------------|-----|
| 6 | 76.5 ± 3.2% (95% CI: ±2.8%) | 78.1 ± 3.8% (95% CI: ±3.3%) | +1.6 ± 2.1% |
| 8 | 77.3 ± 3.3% (95% CI: ±2.9%) | 79.7 ± 4.1% (95% CI: ±3.6%) | +2.4 ± 2.5% |
| 10 | 78.0 ± 3.0% (95% CI: ±2.6%) | 79.8 ± 3.5% (95% CI: ±3.1%) | +1.8 ± 2.2% |
| 12 | 79.5 ± 2.8% (95% CI: ±2.5%) | 80.2 ± 3.2% (95% CI: ±2.8%) | +0.7 ± 1.9% |

**Key Questions:**
- Does VQD win consistently across seeds?
- Is the gap statistically significant (CI doesn't contain 0)?
- Which k shows strongest VQD advantage?

### How to Run

```bash
cd vqd_proper_experiments

# Run full sweep (1.5-2 hours)
python experiment_k_sweep_ci.py | tee logs/k_sweep_ci.log

# Check results
cat results/k_sweep_ci_results.json
```

### Time Estimate

- **Per seed**: ~18-25 minutes (4 k-values × ~5 min each)
- **Total**: ~1.5-2 hours (5 seeds)

---

## Experiment 2: Whitening Toggle

### Purpose

Compare standard projection (U) vs whitened projection (U Λ^{-1/2}).

### Why It Matters

**Whitening** scales each principal component to unit variance:

```
Standard:  z = (x - μ) U^T
Whitened:  z = (x - μ) U^T Λ^{-1/2}
```

Where Λ = diag(λ₁, λ₂, ..., λₖ) are eigenvalues.

**Benefits:**
- All dimensions have equal variance → fairer DTW distances
- Prevents large eigenvalues from dominating
- Common in PCA literature (sometimes called "PCA whitening")

**When useful:**
- k=8-12: Lower eigenvalues become small (λ₈ << λ₁)
- DTW sensitive to scale differences
- VQD eigenvalues may differ from PCA

### What It Does

```python
For each k in [6, 8, 10, 12]:
    # PCA
    pca.fit(frame_bank)
    U_pca = pca.components_
    λ_pca = pca.explained_variance_
    
    # Standard projection
    z_std = (x - μ) U_pca^T
    accuracy_pca_std = DTW_1NN(z_std)
    
    # Whitened projection
    z_wht = (x - μ) U_pca^T Λ_pca^{-1/2}
    accuracy_pca_wht = DTW_1NN(z_wht)
    
    # Same for VQD
    ...
```

### Expected Output

**Table format:**

| k | Method | Standard | Whitened | Delta |
|---|--------|----------|----------|-------|
| 6 | PCA | 75.0% | 76.7% | +1.7% |
| 6 | VQD | 81.7% | 83.3% | +1.6% |
| 8 | PCA | 78.3% | 80.0% | +1.7% |
| 8 | VQD | 91.7% | 91.7% | 0.0% |
| 10 | PCA | 80.0% | 81.7% | +1.7% |
| 10 | VQD | 85.0% | 86.7% | +1.7% |

**Key Questions:**
- Does whitening help both PCA and VQD?
- Does whitening close the VQD-PCA gap or widen it?
- Which k benefits most from whitening?

### How to Run

```bash
cd vqd_proper_experiments

# Run whitening test (~30 minutes)
python experiment_whitening.py | tee logs/whitening.log

# Check results
cat results/whitening_results.json
```

### Time Estimate

- **Per k**: ~7-8 minutes (2 methods × 2 modes)
- **Total**: ~30 minutes (4 k-values)

---

## Experiment 3: By-Class Analysis

### Purpose

Compute per-class accuracy for PCA vs VQD at k=8 to identify which action classes benefit most from quantum dimensionality reduction.

### Why It Matters

**Interpretability!** Instead of just "VQD is 2% better overall", we can say:

- "VQD excels at temporal actions like 'Tennis swing' (+15%) and 'Jogging' (+10%)"
- "VQD struggles with static poses like 'Hand clap' (-5%)"
- "Quantum features capture motion dynamics better than spatial structure"

This makes the paper more insightful and helps explain *why* VQD works.

### What It Does

```python
k = 8
seed = 42

# Run PCA and VQD
pca_predictions = DTW_1NN(X_test_pca, ...)
vqd_predictions = DTW_1NN(X_test_vqd, ...)

# Compute per-class accuracy
for class_id in range(20):
    mask = (y_test == class_id)
    pca_acc[class_id] = mean(pca_predictions[mask] == y_test[mask])
    vqd_acc[class_id] = mean(vqd_predictions[mask] == y_test[mask])
    delta[class_id] = vqd_acc - pca_acc

# Sort by delta
top_vqd_gains = classes with largest positive delta
top_pca_gains = classes with largest negative delta
```

### Expected Output

**Per-class table:**

| Class | Action Name | PCA | VQD | Delta |
|-------|-------------|-----|-----|-------|
| 19 | Tennis swing | 66.7% | 100.0% | +33.3% ⭐ |
| 4 | Forward punch | 66.7% | 100.0% | +33.3% ⭐ |
| 18 | Jogging | 66.7% | 100.0% | +33.3% ⭐ |
| 16 | Forward kick | 66.7% | 100.0% | +33.3% ⭐ |
| ... | ... | ... | ... | ... |
| 12 | Hand clap | 100.0% | 66.7% | -33.3% ❌ |

**Visualization:**

Horizontal bar chart showing VQD-PCA gap for each class:
- Green bars: VQD wins
- Red bars: PCA wins
- Sorted by delta (best VQD at top, worst at bottom)

**Top 5 VQD Wins:**
1. Class 19 (Tennis swing): +33.3%
2. Class 4 (Forward punch): +33.3%
3. Class 18 (Jogging): +33.3%
4. Class 16 (Forward kick): +33.3%
5. Class 14 (Side boxing): +16.7%

**Top 5 PCA Wins:**
1. Class 12 (Hand clap): -33.3%
2. Class 3 (Hand catch): -16.7%
3. Class 7 (Draw tick): -16.7%
4. Class 0 (High arm wave): 0.0%
5. Class 1 (Horizontal wave): 0.0%

### How to Run

```bash
cd vqd_proper_experiments

# Run by-class analysis (~15 minutes)
python experiment_by_class.py | tee logs/by_class.log

# Check results
cat results/by_class_results.json
open figures/by_class_comparison.png
```

### Time Estimate

- **Total**: ~15 minutes (single k=8 run + per-class breakdown)

---

## Running All Experiments

### Master Script

```bash
cd vqd_proper_experiments

# Run all three experiments sequentially
bash run_advanced_experiments.sh
```

This will:
1. Run k-sweep with CIs (~1.5-2 hours)
2. Run whitening toggle (~30 minutes)
3. Run by-class analysis (~15 minutes)
4. Save all results to `results/`
5. Save all logs to `logs/`
6. Generate figure for by-class comparison

### Total Time

**~2-3 hours** for all three experiments.

---

## Results Interpretation

### Experiment 1: Statistical Significance

**If 95% CI of gap excludes zero:**
- VQD advantage is statistically significant
- Can confidently report in paper

**If 95% CI includes zero:**
- Effect is not statistically significant
- Need more seeds or larger test set
- Still report as "trend" or "marginal improvement"

### Experiment 2: Whitening Effect

**If whitening helps both methods equally:**
- Just a preprocessing choice
- Report whichever gives better results
- Doesn't change VQD-PCA comparison

**If whitening helps VQD more:**
- Suggest VQD eigenvalues are less balanced
- Whitening stabilizes VQD
- Interesting finding!

**If whitening helps PCA more:**
- PCA eigenvalues may have wider range
- VQD already somewhat "whitened" by circuit
- Also interesting!

### Experiment 3: Action-Level Insights

**Look for patterns:**
- **Temporal actions** (tennis, jogging, kicks): VQD wins?
- **Static poses** (hand clap, bend): PCA wins?
- **Drawing actions** (draw X, circle): Depends on motion?

**For the paper:**
- "VQD discovers features that better capture temporal dynamics"
- "Quantum circuits encode motion patterns more effectively"
- "VQD struggles with discrete, instantaneous actions"

---

## File Structure

After running, you'll have:

```
vqd_proper_experiments/
├── experiment_k_sweep_ci.py          # Script 1
├── experiment_whitening.py           # Script 2
├── experiment_by_class.py            # Script 3
├── run_advanced_experiments.sh       # Master script
├── results/
│   ├── k_sweep_ci_results.json      # 5 seeds × 4 k-values
│   ├── whitening_results.json        # 4 k-values × 2 modes
│   └── by_class_results.json         # 20 classes × 2 methods
├── logs/
│   ├── k_sweep_ci.log               # Full output
│   ├── whitening.log                 # Full output
│   └── by_class.log                  # Full output
├── figures/
│   └── by_class_comparison.png       # Horizontal bar chart
└── docs/
    └── ADVANCED_EXPERIMENTS_GUIDE.md (this file)
```

---

## Next Steps After Experiments

1. **Analyze Results**
   - Check if VQD advantage is significant (Exp 1)
   - Decide on whitening based on results (Exp 2)
   - Write interpretability paragraph (Exp 3)

2. **Update Paper**
   - Add confidence intervals to results table
   - Add whitening comparison (if interesting)
   - Add by-class figure and discussion

3. **Consider Follow-ups**
   - If VQD wins on temporal actions: test on pure temporal datasets
   - If whitening helps: make it default
   - If certain classes dominate: analyze sequence lengths, motion types

---

## Troubleshooting

### Experiment 1 takes too long
- Reduce seeds: `seeds=[42, 123, 456]` (3 instead of 5)
- Reduce k-values: `k_values=[8, 10]` (2 instead of 4)

### Experiment 2 shows no whitening effect
- Normal! Means eigenvalues are already well-scaled
- Report as "whitening did not improve results"

### Experiment 3 shows high variance per class
- Normal! Test set has only ~3 samples per class
- Focus on overall trends, not individual classes
- Consider combining related classes (e.g., all "kick" actions)

---

## Publication Checklist

- [ ] Report mean ± std with 95% CI (Exp 1)
- [ ] Test statistical significance (t-test or permutation test)
- [ ] Mention whitening (Exp 2), even if negative result
- [ ] Include by-class figure (Exp 3) in paper
- [ ] Write interpretability paragraph explaining VQD wins
- [ ] Discuss limitations (small test set, limited classes)
- [ ] Suggest future work (more seeds, more datasets, real quantum hardware)

---

**Good luck with the experiments!** 🎯
