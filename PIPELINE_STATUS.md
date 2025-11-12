# QDTW Pipeline Status

**Last Updated**: November 6, 2025

## ✅ Completed Steps

### 1. Frame Bank Building
- ✅ Training frame bank: `data/frame_bank.npy` (1.9 MB, ~8000 frames)
- ✅ Test frame bank: `data/frame_bank_test.npy` (929 KB, ~2000 frames)
- **Status**: COMPLETE

### 2. PCA Basis Computation
- ⚠️  PCA bases were computed but not saved to `results/` directory
- ✅ Projected sequences exist, so PCA was run successfully
- **Methods**: Uc (classical), Uq (quantum)
- **K values**: 5, 8, 10
- **Status**: COMPLETE (but basis files not archived)

### 3. Sequence Projection
- ✅ All sequences projected for Uc: k=5,8,10
- ✅ All sequences projected for Uq: k=5,8,10
- ✅ Total: 3,378 projected sequence files
- **Status**: COMPLETE

### 4. Label Metadata
- ✅ Metadata files created for all configurations
- ✅ Action labels extracted from filenames
- **Status**: COMPLETE

### 5. DTW Classification
- ✅ Uc results: `results/metrics_subspace_Uc.csv`
  - k=5: 74.5-79.7% accuracy
  - k=8: 77.1-80.8% accuracy
  - k=10: 78.7-83.0% accuracy (BEST: euclidean, 82.99%)
- ✅ Uq results: `results/metrics_subspace_Uq.csv`
  - k=5: 73.2-77.3% accuracy
  - k=8: 76.7-79.8% accuracy
  - k=10: 78.1-81.1% accuracy (BEST: euclidean, 81.14%)
- **Status**: COMPLETE

### 6. Evaluation Figures
- ✅ `figures/accuracy_vs_k.png` (189 KB)
- ✅ `figures/time_vs_k.png` (374 KB)
- ✅ `figures/pareto_accuracy_time.png` (233 KB)
- **Status**: COMPLETE

### 7. Ablation Studies
- ⚠️  Partial results exist (`results/ablations.csv`)
- ⚠️  Only tested with small sample (n_train=100, n_test=30)
- ⚠️  Need full dataset run for publication-quality results
- ✅ Ablation figures exist but from small sample
- **Status**: NEEDS FULL RUN

---

## 📊 Current Results Summary

| Method | k | Metric | Accuracy | Time (ms) |
|--------|---|---------|----------|-----------|
| **Uc** | **10** | **euclidean** | **82.99%** | **847** |
| Uc | 8 | euclidean | 80.82% | 739 |
| Uq | 10 | euclidean | 81.14% | 872 |
| Uc | 10 | fidelity | 80.41% | 2039 |
| Uq | 8 | euclidean | 79.77% | 747 |

**Key Findings**:
- Classical PCA (Uc) slightly outperforms Quantum PCA (Uq)
- Euclidean metric consistently best across all configurations
- Optimal k value: 10 (balance of accuracy and speed)
- Speed: Euclidean ~3x faster than cosine/fidelity

---

## 🚀 To Run Complete Pipeline from Scratch

### Full Dataset (Production Quality)

```bash
# 1. Build frame bank (if needed)
python scripts/build_frame_bank.py \
  --data-dir msr_action_data \
  --output data/frame_bank.npy \
  --test-output data/frame_bank_test.npy

# 2. Run PCA for all k values
for k in 3 5 8 10 12 16; do
  # Classical PCA
  python quantum/classical_pca.py \
    --frames data/frame_bank.npy \
    --k $k \
    --output results/Uc_k${k}.npz
    
  # Quantum PCA
  python quantum/qpca.py \
    --frames data/frame_bank.npy \
    --k $k \
    --output results/Uq_k${k}.npz
done

# 3. Project sequences
for method in Uc Uq; do
  for k in 3 5 8 10 12 16; do
    python scripts/project_sequences.py \
      --data-dir msr_action_data \
      --pca-file results/${method}_k${k}.npz \
      --output-dir results/subspace/${method}/k${k}
  done
done

# 4. Create metadata
python scripts/create_label_metadata.py

# 5. Run DTW classification
for method in Uc Uq; do
  for k in 3 5 8 10 12 16; do
    for metric in euclidean cosine fidelity; do
      python scripts/run_dtw_subspace.py \
        --method $method \
        --k $k \
        --metric $metric
    done
  done
done

# 6. Generate figures
python eval/make_figures.py

# 7. Run ablations (FULL DATASET)
python scripts/run_ablations.py --all --n-train 454 --n-test 113
```

**Estimated Runtime**: ~4-6 hours for complete pipeline

---

## �� What We Need to Run

### Option 1: Just Complete Ablations (Recommended Next Step)

Run ablations with full dataset instead of sample data:

```bash
python scripts/run_ablations.py --all --n-train 454 --n-test 113
```

**Time**: ~30-45 minutes  
**Output**: 
- Updated `results/ablations.csv` with full results
- Updated ablation figures with production data

### Option 2: Add More K Values

We currently have k=5,8,10. To add k=3,12,16:

```bash
# For each new k value
for k in 3 12 16; do
  # PCA
  python quantum/classical_pca.py --frames data/frame_bank.npy --k $k --output results/Uc_k${k}.npz
  python quantum/qpca.py --frames data/frame_bank.npy --k $k --output results/Uq_k${k}.npz
  
  # Project
  python scripts/project_sequences.py --data-dir msr_action_data --pca-file results/Uc_k${k}.npz --output-dir results/subspace/Uc/k${k}
  python scripts/project_sequences.py --data-dir msr_action_data --pca-file results/Uq_k${k}.npz --output-dir results/subspace/Uq/k${k}
  
  # Classify
  python scripts/run_dtw_subspace.py --method Uc --k $k --metric euclidean
  python scripts/run_dtw_subspace.py --method Uq --k $k --metric euclidean
done

# Regenerate figures
python eval/make_figures.py
```

**Time**: ~1-2 hours per k value

### Option 3: Complete Fresh Run

Start completely fresh with all k values:

```bash
# Use the full pipeline script above
```

**Time**: ~4-6 hours

---

## 💡 Recommendations

1. **Immediate**: Run full ablations to complete the analysis
   ```bash
   python scripts/run_ablations.py --all --n-train 454 --n-test 113
   ```

2. **Optional**: Add k=3,12,16 if you want more comprehensive k-sweep analysis

3. **Archive**: Save PCA bases to `results/` for reproducibility

4. **Documentation**: README.md is now complete with full pipeline documentation

---

## 📁 Expected Final File Structure

```
results/
├── Uc_k3.npz, Uc_k5.npz, Uc_k8.npz, Uc_k10.npz, Uc_k12.npz, Uc_k16.npz
├── Uq_k3.npz, Uq_k5.npz, Uq_k8.npz, Uq_k10.npz, Uq_k12.npz, Uq_k16.npz
├── metrics_baseline.csv
├── metrics_subspace_Uc.csv
├── metrics_subspace_Uq.csv
├── ablations.csv
└── subspace/
    ├── Uc/k{3,5,8,10,12,16}/{train,test}/seq_*.npy + metadata.npz
    └── Uq/k{3,5,8,10,12,16}/{train,test}/seq_*.npy + metadata.npz

figures/
├── accuracy_vs_k.png
├── time_vs_k.png
├── pareto_accuracy_time.png
├── ablations_distance.png
├── ablations_k_sweep.png
├── ablations_sampling.png
└── ablations_robustness.png
```
