# 13 - Ablation Studies

**File:** `13_ABLATION_STUDIES.md`  
**Purpose:** Validate necessity of each pipeline component  
**For Thesis:** Results/Discussion - design validation

---

## 13.1 Purpose

**Goal:** Systematically remove components to validate their necessity.

**Method:** Test all configurations, measure impact.

---

## 13.2 Component Ablations

### 13.2.1 Pre-Reduction Necessity

| Configuration | Pipeline | Accuracy | Conclusion |
|---------------|----------|----------|------------|
| **Full pipeline** | 60D → 20D (PCA) → 8D (VQD) | **83.4%** | Baseline |
| **No pre-reduction** | 60D → 8D (VQD direct) | 77.7% | **-5.7%** ❌ |

**Finding:** Pre-reduction **essential** for VQD advantage (Section 18.2).

### 13.2.2 Per-Sequence Centering

| Configuration | Centering | Accuracy | Conclusion |
|---------------|-----------|----------|------------|
| **With per-seq centering** | Yes | **83.4%** | Baseline |
| **Global only** | No | 80.1% | **-3.3%** ❌ |

**Finding:** Per-sequence centering **critical** (Section 18.5).

### 13.2.3 Distance Metric

| Metric | Accuracy | Notes |
|--------|----------|-------|
| **Cosine** | **82.7%** | ✓ Best (angle-based) |
| Euclidean | 65.3% | Poor (magnitude-sensitive) |
| Fidelity | 80.1% | Good but worse than cosine |

**Finding:** Cosine distance optimal (Section 18.7).

---

## 13.3 Hyperparameter Sensitivity

### 13.3.1 Circuit Depth

| Depth | Accuracy | Training Time |
|-------|----------|---------------|
| 1 | 81.2% | 6 min |
| **2** | **83.4%** | 9 min ✓ |
| 3 | 83.1% | 14 min |
| 5 | 82.8% | 24 min |

**Finding:** Depth=2 optimal (balance accuracy/time).

### 13.3.2 Penalty Scale

| Penalty | Accuracy | Notes |
|---------|----------|-------|
| 5.0 | 82.1% | Weak orthogonality |
| **10.0** | **83.4%** | ✓ Optimal |
| 20.0 | 82.8% | Slightly worse |
| 50.0 | 80.9% | Too strong (mode mixing) |

**Finding:** Penalty=10.0 optimal.

---

## 13.4 Summary Table

| Component | Necessity | Impact | Validated |
|-----------|-----------|--------|-----------|
| Pre-reduction (20D) | **Essential** | +5.7% | ✓✓✓ |
| Per-seq centering | **Critical** | +3.3% | ✓✓✓ |
| VQD (vs PCA) | **Significant** | +5.0% | ✓✓✓ |
| Cosine distance | **Optimal** | +17.4% vs Euclidean | ✓✓✓ |
| Depth=2 | **Good** | Baseline | ✓✓ |
| Penalty=10 | **Good** | Baseline | ✓✓ |

---

## 13.5 Key Takeaways

**All components validated:**
1. ✅ Pre-reduction: Essential (+5.7%)
2. ✅ Per-seq centering: Critical (+3.3%)
3. ✅ VQD: Significant (+5.0%)
4. ✅ Cosine distance: Optimal (+17.4%)
5. ✅ Hyperparameters: Well-tuned

**Pipeline is fully justified by ablations.**

---

**Next:** [14_VISUALIZATION.md](./14_VISUALIZATION.md) →

---

**Navigation:**
- [← 12_K_SWEEP.md](./12_K_SWEEP.md)
- [→ 14_VISUALIZATION.md](./14_VISUALIZATION.md)
- [↑ Index](./README.md)
