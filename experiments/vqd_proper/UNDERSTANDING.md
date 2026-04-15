# VQD-DTW Proper Experiment - Understanding & Execution Plan

## 📚 What I've Understood from the Code

### 1. **Experiment Goal**
Compare **VQD Quantum PCA** vs **Classical PCA** for dimensionality reduction in a proper DTW classification pipeline using full temporal sequences (not single frames).

**Key Question**: Can VQD match classical PCA accuracy while providing quantum benefits?

---

### 2. **Pipeline Architecture** (from `vqd_dtw_proper.py`)

```
DATA (567 sequences)
    ↓ stratified split
TRAIN (300) + TEST (60)
    ↓ collect all train frames
FRAME BANK (~11,900 frames, 60D)
    ↓ normalize + pre-reduce
FRAME BANK (60D → 16D via classical PCA)
    ↓ 
┌───────────────────┬──────────────────┐
│   CLASSICAL PCA   │   VQD QUANTUM    │
│   (16D → kD)      │   (16D → kD)     │
└───────────────────┴──────────────────┘
    ↓                     ↓
PROJECT SEQUENCES    PROJECT SEQUENCES
    ↓                     ↓
DTW 1-NN CLASSIFY    DTW 1-NN CLASSIFY
    ↓                     ↓
ACCURACY, SPEEDUP    ACCURACY, SPEEDUP
```

---

### 3. **Key Design Decisions**

#### ✅ **Why Full Sequences?**
- Previous experiments used single frames (1 frame per sequence)
- This is NOT proper DTW - DTW needs temporal sequences
- Now using full sequences: 13-255 frames each
- This is the **correct** way to do DTW classification

#### ✅ **Why Frame Bank?**
- Need many samples to learn good subspace
- One sequence = not enough data for PCA/VQD
- Frame bank = collect ALL frames from ALL training sequences
- ~11,900 frames from 300 sequences = good statistics

#### ✅ **Why Pre-reduction (60D → 16D)?**
- 60D requires 6 qubits for quantum simulation
- Too expensive, may not simulate well
- Pre-reduce with classical PCA first
- 16D requires 4 qubits = manageable
- Still keeps 95%+ variance

#### ✅ **Why Stratified Split?**
- Ensures balanced classes in train/test
- 300 train / 60 test ≈ 15 train + 3 test per class
- Fair evaluation across all 20 classes

#### ✅ **Why k-sweep {4, 6, 8, 10, 12}?**
- Test multiple target dimensions
- Trade-off: accuracy vs speed
- Lower k = faster but less accurate
- Higher k = more accurate but slower

---

### 4. **What the Code Does - Step by Step**

#### **Step 1: Load Data** (`load_data()`)
```python
• Load 567 MSR Action3D sequences
• Each sequence: variable length (13-255 frames), 60D per frame
• Split: 300 train, 60 test (stratified by class)
• Result: X_train_raw, X_test_raw, y_train, y_test
```

#### **Step 2: Build Frame Bank** (`build_frame_bank()`)
```python
• Collect all frames from 300 train sequences
• Frame bank shape: (11900, 60)
• Normalize: StandardScaler (fit on train only)
• Pre-reduce: 60D → 16D with classical PCA
• Result: frame_bank_reduced (11900, 16)
```

#### **Step 3: Baseline** (`evaluate_baseline()`)
```python
• DTW 1-NN on raw 60D normalized sequences
• For each test sequence:
    - Find nearest neighbor in train set (DTW distance)
    - Predict its class
• Measure: accuracy, time per query
• Expected: ~75% accuracy (from literature)
```

#### **Step 4: Classical PCA** (`evaluate_pca(k)`)
```python
For k ∈ {4, 6, 8, 10, 12}:
    • Learn PCA on frame_bank_reduced (16D → kD)
    • Project all sequences:
        - Normalize → pre-reduce → PCA project
    • DTW 1-NN classification on kD sequences
    • Measure: accuracy, speedup vs baseline
```

#### **Step 5: VQD Quantum PCA** (`evaluate_vqd(k)`)
```python
For k ∈ {4, 6, 8, 10, 12}:
    • Learn VQD on frame_bank_reduced (16D → kD)
    • VQD parameters:
        - num_qubits = ceil(log2(16)) = 4 qubits
        - max_depth = 2 (ansatz depth)
        - ramped_penalties = True (orthogonality)
        - entanglement = 'alternating'
        - maxiter = 200
        - validate = True (check quality)
    • Use Procrustes-aligned basis (if available)
    • Project all sequences:
        - Normalize → pre-reduce → VQD project
    • DTW 1-NN classification on kD sequences
    • Measure: accuracy, speedup, VQD quality
```

#### **Step 6: Compare** (`print_summary()`)
```python
• Print table comparing all methods
• Metrics: accuracy, speedup, VQD quality
• Save results to JSON
```

---

### 5. **Expected Results**

Based on the code and literature:

| Method | k | Expected Accuracy | Expected Speedup | Notes |
|--------|---|-------------------|------------------|-------|
| **Baseline** | 60 | ~75% | 1.0× | Full 60D sequences |
| **PCA** | 4 | 65-70% | 5-8× | Heavy dimensionality reduction |
| **PCA** | 8 | 70-75% | 3-5× | Good balance |
| **PCA** | 12 | 73-76% | 2-3× | Close to baseline |
| **VQD** | 4 | 63-68% | 5-8× | Should be close to PCA |
| **VQD** | 8 | 68-73% | 3-5× | Best VQD performance |
| **VQD** | 12 | 71-75% | 2-3× | Near-baseline accuracy |

**Success Criteria**:
- ✅ Baseline ≥ 70% (validates proper DTW setup)
- ✅ VQD within 2-3% of PCA at same k
- ✅ VQD orthogonality error < 1e-6
- ✅ VQD angles to PCA < 45° (ideally < 30°)

---

### 6. **VQD Quality Metrics**

The code tracks VQD quality via:

1. **Orthogonality Error**: How orthogonal is VQD basis?
   - Should be < 1e-6 (very orthogonal)
   
2. **Principal Angles**: How aligned is VQD to PCA?
   - Mean angle: should be < 30°
   - Max angle: should be < 45°
   - Lower = VQD found similar subspace to PCA

3. **Procrustes Alignment**: Rotates VQD basis to best match PCA
   - Removes rotation ambiguity
   - Fair comparison

---

## 🚀 Execution Plan - Step by Step

### ✅ **COMPLETED: Pre-flight Checks**
1. ✅ Explored code structure
2. ✅ Created experiment guide
3. ✅ Verified dependencies (loader, vqd_pca, dtw_runner)
4. ✅ Verified data (567 sequences in msr_action_data/)
5. ✅ Tested data loading
6. ✅ Fixed path issues

---

### 🎯 **NEXT STEPS**

#### **Step 1: Quick Validation Run (k=4 only)**
**Purpose**: Verify entire pipeline works before full run

**Action**:
```bash
# Create test version with k=[4] only
# Run to verify:
#   - Data loads correctly
#   - Baseline computes (~75% expected)
#   - PCA k=4 completes
#   - VQD k=4 completes with good quality
```

**Expected time**: ~5-10 minutes  
**Go/No-go**: If baseline < 70%, stop and debug

---

#### **Step 2: Full K-Sweep Run (k=4,6,8,10,12)**
**Purpose**: Complete experiment with all k values

**Action**:
```bash
cd /path/to/qdtw_project/vqd_proper_experiments
python vqd_dtw_proper.py | tee logs/full_run.log
```

**Expected time**: 30-60 minutes
- Data loading: 1 min
- Baseline: 5 min
- PCA (5 values): ~15 min
- VQD (5 values): ~20-40 min

**Monitor**:
- Baseline accuracy (~75%)
- PCA accuracies (increasing with k)
- VQD convergence (orthogonality, angles)
- VQD accuracies (close to PCA)

---

#### **Step 3: Analyze Results**
**Purpose**: Extract insights from completed experiment

**Action**:
```bash
# View results JSON
cat results/vqd_dtw_proper_results.json | python -m json.tool

# Review logs
less logs/full_run.log
```

**Look for**:
- Best k value (accuracy vs speedup trade-off)
- VQD vs PCA gap (should be < 3%)
- VQD quality metrics
- Speedup gains

---

#### **Step 4: Create Summary Report**
**Purpose**: Document findings

**Create**:
- Summary table
- Key insights
- Comparison to baseline
- VQD quality assessment
- Recommendations

---

## 🎬 Ready to Execute?

**Status**: ✅ All dependencies verified, code understood

**Next action**: 
1. **Quick test run** (k=4 only) - verify pipeline works
2. **Full run** (all k values) - complete experiment
3. **Analysis** - extract insights

**Shall we proceed with Step 1 (quick test run)?** 🚀
