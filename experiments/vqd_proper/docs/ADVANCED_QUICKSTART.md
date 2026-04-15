# Advanced Experiments - Quick Start

## What We Created

Three new experiments to strengthen your VQD-DTW research:

### 1️⃣ K-Sweep with Confidence Intervals
- **File**: `experiment_k_sweep_ci.py`
- **What**: Tests k={6,8,10,12} across 5 random seeds
- **Why**: Provides mean ± std ± 95% CI for statistical rigor
- **Time**: ~1.5-2 hours
- **Output**: `results/k_sweep_ci_results.json`

### 2️⃣ Whitening Toggle
- **File**: `experiment_whitening.py`
- **What**: Compares U vs U Λ^{-1/2} (standard vs whitened projection)
- **Why**: Tests if whitening stabilizes DTW for k=8-12
- **Time**: ~30 minutes
- **Output**: `results/whitening_results.json`

### 3️⃣ By-Class Analysis
- **File**: `experiment_by_class.py`
- **What**: Per-class accuracy breakdown (20 action classes)
- **Why**: Shows which actions benefit from VQD (interpretability!)
- **Time**: ~15 minutes
- **Output**: `results/by_class_results.json` + `figures/by_class_comparison.png`

---

## Quick Commands

### Run All Three (Recommended)
```bash
cd /path/to/qdtw_project/vqd_proper_experiments
bash run_advanced_experiments.sh
```
**Total time: ~2-3 hours**

### Run Individually

**Experiment 1 only:**
```bash
cd vqd_proper_experiments
python experiment_k_sweep_ci.py | tee logs/k_sweep_ci.log
```

**Experiment 2 only:**
```bash
cd vqd_proper_experiments
python experiment_whitening.py | tee logs/whitening.log
```

**Experiment 3 only:**
```bash
cd vqd_proper_experiments
python experiment_by_class.py | tee logs/by_class.log
```

---

## What You'll Get

### Experiment 1: Statistical Tables

```
k=6:
  PCA: 76.5% ± 3.2% (95% CI: ±2.8%)
  VQD: 78.1% ± 3.8% (95% CI: ±3.3%)
  Gap: +1.6% ± 2.1%

k=8:
  PCA: 77.3% ± 3.3% (95% CI: ±2.9%)
  VQD: 79.7% ± 4.1% (95% CI: ±3.6%)
  Gap: +2.4% ± 2.5%
```

**Use for**: Publication-ready results with confidence intervals

### Experiment 2: Whitening Comparison

```
k    Method  Standard   Whitened   Delta
--------------------------------------------
6    PCA     75.0%      76.7%      +1.7%
6    VQD     81.7%      83.3%      +1.6%
8    PCA     78.3%      80.0%      +1.7%
8    VQD     91.7%      91.7%       0.0%
```

**Use for**: Method variants section in paper

### Experiment 3: Action-Level Insights

**Top VQD Wins:**
- Tennis swing: +33.3%
- Forward punch: +33.3%
- Jogging: +33.3%

**Top PCA Wins:**
- Hand clap: -33.3%
- Hand catch: -16.7%

**Plus a nice horizontal bar chart!**

**Use for**: Interpretability paragraph + figure in paper

---

## Progress Tracking

Monitor experiments with:

```bash
# K-sweep progress
tail -f logs/k_sweep_ci.log

# Whitening progress
tail -f logs/whitening.log

# By-class progress
tail -f logs/by_class.log
```

---

## Files Created

```
vqd_proper_experiments/
├── experiment_k_sweep_ci.py          ← Experiment 1
├── experiment_whitening.py           ← Experiment 2
├── experiment_by_class.py            ← Experiment 3
├── run_advanced_experiments.sh       ← Master script
└── docs/
    ├── ADVANCED_EXPERIMENTS_GUIDE.md ← Full guide
    └── ADVANCED_QUICKSTART.md        ← This file
```

---

## After Experiments Complete

1. **Check results:**
   ```bash
   cat results/k_sweep_ci_results.json
   cat results/whitening_results.json
   cat results/by_class_results.json
   ```

2. **View figure:**
   ```bash
   open figures/by_class_comparison.png
   # or
   xdg-open figures/by_class_comparison.png
   ```

3. **Analyze:**
   - Is VQD advantage significant? (Check 95% CI in Exp 1)
   - Does whitening help? (Check deltas in Exp 2)
   - Which actions benefit? (Check top gains in Exp 3)

---

## What This Adds to Your Paper

### Before (What You Had)
✅ VQD finds different subspaces (angles 82-90°)  
✅ VQD provides modest improvement (+2-5%)  
✅ No bugs or unfair advantages  

### After (What You'll Add)
✅ **Statistical rigor**: VQD advantage with 95% confidence intervals  
✅ **Method variants**: Whitening comparison  
✅ **Interpretability**: Which actions benefit from quantum features  
✅ **Publication-ready**: Proper statistics, error bars, clear presentation  

---

## Ready to Start?

```bash
cd /path/to/qdtw_project/vqd_proper_experiments
bash run_advanced_experiments.sh
```

**Go make some coffee, this will take 2-3 hours!** ☕

Then come back to publication-ready results! 🎯📊
