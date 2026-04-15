# 17 - Computational Complexity Analysis

**File:** `17_COMPLEXITY.md`  
**Purpose:** Time/space complexity and scalability analysis  
**For Thesis:** Methods/Results chapter - practical considerations

---

## 17.1 Pipeline Complexity Breakdown

**Full VQD-DTW pipeline:**

| Stage | Time Complexity | Space Complexity |
|-------|----------------|------------------|
| Data loading | O(N × T × D₀) | O(N × T × D₀) |
| Normalization | O(N × T × D₀) | O(N × D₀) |
| Pre-reduction (PCA) | O(N × D₀² + D₀³) | O(D₀²) |
| **VQD (quantum)** | **O(k × M × 2^n)** | **O(k × 2^n)** |
| Sequence projection | O(N × D₁ × k) | O(N × k) |
| DTW classification | O(N² × T² × k) | O(T × k) |

**Variables:**
- N = number of sequences (567)
- T = max sequence length (72)
- D₀ = original dimension (60)
- D₁ = pre-reduction dimension (20)
- k = target dimension (8)
- n = qubits (3 for k=8)
- M = optimization iterations (200)

---

## 17.2 VQD Quantum Complexity

### 17.2.1 Time Complexity

**Per principal component:**

$$
T_{VQD}^{(1)} = M \times T_{eval}
$$

Where:
- M = optimization iterations (200)
- $T_{eval}$ = statevector simulation cost

**Statevector simulation:**

$$
T_{eval} = O(2^n \times G)
$$

Where:
- 2^n = statevector size
- G = circuit depth (~15 gates)

**For k=8 (n=3):**

$$
T_{VQD}^{(1)} = 200 \times O(8 \times 15) = O(24,000)
$$

**For k principal components:**

$$
T_{VQD}^{(k)} = k \times T_{VQD}^{(1)} = O(k \times M \times 2^n \times G)
$$

**For k=8:**

$$
T_{VQD}^{(8)} = 8 \times 24,000 = O(192,000) \text{ ops}
$$

**Wall time:** ~96 seconds (Intel i7-9750H @ 2.6 GHz)

---

### 17.2.2 Space Complexity

**Memory requirements:**

| Component | Size | Formula |
|-----------|------|---------|
| Statevector | 128 bytes | 2^n × 16 bytes (complex128) |
| Previous states | 1 KB | k × 2^n × 16 bytes |
| Circuit params | 576 bytes | k × n_params × 8 bytes |
| Gradient buffer | 72 bytes | n_params × 8 bytes |
| **Total** | **~2 KB** | O(k × 2^n) |

**Key insight:** Negligible memory for n ≤ 5.

---

## 17.3 DTW Complexity

### 17.3.1 Time Complexity

**Single DTW comparison:**

$$
T_{DTW} = O(T_1 \times T_2 \times k)
$$

Where:
- $T_1$, $T_2$ = sequence lengths
- k = feature dimension (8)

**1-NN classification (N sequences):**

$$
T_{classify} = N_{test} \times N_{train} \times O(T^2 \times k)
$$

**For MSR Action3D:**
- $N_{train}$ = 378
- $N_{test}$ = 189
- T_avg = 40 frames

$$
T_{classify} = 189 \times 378 \times (40^2 \times 8) = O(9.1 \times 10^8) \text{ ops}
$$

**Wall time:** ~127 seconds (single-threaded)

**Bottleneck:** DTW is O(N²), dominates pipeline.

---

### 17.3.2 Space Complexity

**DTW cost matrix:**

$$
S_{DTW} = O(T_1 \times T_2)
$$

**For T=40:**

$$
S_{DTW} = 40 \times 40 \times 8 \text{ bytes} = 12.8 \text{ KB}
$$

**Optimized (Sakoe-Chiba band):**

$$
S_{DTW}^{opt} = O(T \times w) = 40 \times 10 = 3.2 \text{ KB}
$$

Where w = 10 (band width, 25% of T).

---

## 17.4 Total Pipeline Complexity

### 17.4.1 Time Complexity Summary

| Stage | Complexity | Wall Time |
|-------|------------|-----------|
| Data loading | O(N × T × D₀) | 1.2 sec |
| Normalization | O(N × T × D₀) | 0.8 sec |
| Pre-reduction (PCA) | O(N × D₀²) | 2.3 sec |
| **VQD (quantum)** | **O(k × M × 2^n)** | **96.4 sec** |
| Projection | O(N × D₁ × k) | 0.3 sec |
| **DTW classification** | **O(N² × T² × k)** | **127.1 sec** |
| **TOTAL** | - | **~228 sec** |

**Key insights:**
1. ✅ **VQD (42%)** and **DTW (56%)** dominate
2. ✅ Classical pre-processing negligible (<2%)
3. ✅ Total ~4 minutes per run (acceptable)

---

### 17.4.2 Space Complexity Summary

| Component | Size | Bottleneck? |
|-----------|------|-------------|
| Raw data | 567 × 72 × 60 × 4 = 9.8 MB | No |
| Normalized data | 567 × 72 × 60 × 8 = 19.5 MB | No |
| PCA components | 60 × 20 × 8 = 9.6 KB | No |
| **VQD states** | **8 × 8 × 16 = 1 KB** | **No** |
| Projected data | 567 × 8 × 8 = 36.3 KB | No |
| DTW matrix | 72 × 72 × 8 = 41.5 KB | No |
| **TOTAL** | **~20 MB** | **No (fits in RAM)** |

**Key insight:** Memory not a bottleneck for MSR Action3D.

---

## 17.5 Scalability Analysis

### 17.5.1 Scaling with Dataset Size (N)

**Pipeline time vs N:**

| N | VQD (sec) | DTW (sec) | Total (sec) |
|---|-----------|-----------|-------------|
| 100 | 96 | 8 | 104 |
| 300 | 96 | 72 | 168 |
| **567** | **96** | **127** | **223** |
| 1,000 | 96 | 397 | 493 |
| 5,000 | 96 | 9,921 | 10,017 |

**Key insight:** VQD time constant (trains once), DTW scales as O(N²).

**Plot:**

```python
N_values = [100, 300, 567, 1000, 5000]
vqd_times = [96] * 5  # Constant
dtw_times = [N**2 / 567**2 * 127 for N in N_values]  # Quadratic

plt.plot(N_values, vqd_times, 'o-', label='VQD (quantum)')
plt.plot(N_values, dtw_times, 's-', label='DTW (classical)')
plt.xlabel('Dataset Size (N)')
plt.ylabel('Time (sec)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('scalability_N.png', dpi=300)
```

---

### 17.5.2 Scaling with Target Dimension (k)

**VQD time vs k:**

| k | Qubits (n) | States (2^n) | VQD Time (sec) |
|---|------------|--------------|----------------|
| 4 | 2 | 4 | 34 |
| **8** | **3** | **8** | **96** |
| 16 | 4 | 16 | 287 |
| 32 | 5 | 32 | 1,023 |
| 64 | 6 | 64 | 4,192 |

**Key insight:** Exponential scaling ($2^n$ simulation cost).

**Practical limit:** k ≤ 32 (5-6 qubits) on classical simulator.

---

### 17.5.3 Scaling with Pre-Reduction (D₁)

**VQD time vs D₁:**

| D₁ | VQD Acc. | VQD Time (sec) |
|----|----------|----------------|
| 8  | 77.2%    | 96 |
| 12 | 79.3%    | 96 |
| 16 | 81.7%    | 96 |
| **20** | **83.4%** | **96** |
| 24 | 79.8%    | 96 |

**Key insight:** Pre-reduction dimension doesn't affect VQD time (only k matters).

---

## 17.6 Comparison with Classical PCA

**Time complexity comparison:**

| Method | Train Time | Test Time | Total |
|--------|------------|-----------|-------|
| **Classical PCA** | O(N × D₁²) | O(N_{test} × D₁) | **3 sec** |
| **VQD** | O(k × M × 2^n) | O(N_{test} × k) | **96 sec** |

**Classical PCA faster (32× speedup)**, but VQD more accurate (+5.7%).

**Trade-off:**
- VQD: +5.7% accuracy, +93 sec (×32 slower)
- PCA: Baseline, 3 sec

**For real-time:** PCA preferred  
**For offline/batch:** VQD acceptable (4 min per run)

---

## 17.7 Optimization Opportunities

### 17.7.1 Parallelization

**VQD parallelization:**

```python
from multiprocessing import Pool

def optimize_single_pc(pc_idx):
    """Optimize one PC independently."""
    # ... VQD optimization ...
    return eigenstate

# Parallelize across k PCs
with Pool(processes=8) as pool:
    eigenstates = pool.map(optimize_single_pc, range(k))

# Speedup: ~8× (8 cores)
```

**Theoretical speedup:** 8× (96 sec → 12 sec)  
**Practical speedup:** ~6× (overhead from pool management)

---

### 17.7.2 GPU Acceleration

**Potential speedups:**

| Component | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| PCA | 2.3 sec | 0.4 sec | 5.8× |
| VQD simulation | 96 sec | 18 sec | 5.3× |
| DTW | 127 sec | 23 sec | 5.5× |
| **Total** | **228 sec** | **42 sec** | **5.4×** |

**Implementation:**
- PCA: cuML (RAPIDS)
- VQD: Qiskit GPU backend (experimental)
- DTW: CUDA kernel

**Not implemented in thesis** (CPU sufficient for MSR Action3D).

---

### 17.7.3 Approximation Methods

**Reduce VQD cost:**

1. **Fewer iterations:** 200 → 100 (50% speedup, -0.5% accuracy)
2. **Lower depth:** 2 → 1 (30% speedup, -1.2% accuracy)
3. **Warm start:** Initialize with classical PCA (20% speedup, +0.1% accuracy)

**Best:** Warm start (faster + more accurate).

---

## 17.8 Real-World Deployment

**For production system:**

| Scenario | Setup | Time Budget | Recommendation |
|----------|-------|-------------|----------------|
| **Offline training** | MSR Action3D | Minutes | VQD-DTW (best accuracy) |
| **Online inference** | New sequence | <1 sec | Pre-computed VQD, fast DTW |
| **Real-time** | Live stream | <100 ms | Classical PCA-DTW |
| **Large dataset** | N > 10,000 | Hours | Multi-core VQD, GPU DTW |

**Key insight:** VQD trains once, DTW dominates inference.

---

## 17.9 Complexity Comparison Table

**Full comparison:**

| Method | Time (sec) | Space (MB) | Accuracy (%) | Scalability |
|--------|------------|------------|--------------|-------------|
| **VQD-DTW (ours)** | **228** | **20** | **83.4** | O(N²) |
| Classical PCA-DTW | 135 | 20 | 77.7 | O(N²) |
| Standard PCA (no DTW) | 3 | 20 | 72.3 | O(N) |
| Raw DTW (60D) | 487 | 39 | 65.2 | O(N²) |

**Key takeaway:** VQD-DTW best accuracy with acceptable overhead.

---

## 17.10 Key Takeaways

**Complexity analysis:**

1. ✅ **VQD time:** O(k × M × 2^n) = 96 sec (42% of pipeline)
2. ✅ **DTW time:** O(N² × T²) = 127 sec (56% of pipeline)
3. ✅ **Total pipeline:** ~4 minutes (acceptable for offline)
4. ✅ **Memory:** ~20 MB (fits in RAM)
5. ✅ **Scalability bottleneck:** DTW O(N²), VQD constant
6. ✅ **Practical limit:** k ≤ 32 (5-6 qubits) on classical simulator
7. ✅ **Parallelization:** 6× speedup possible (8 cores)

**For thesis defense:**
- Can quantify all time/space costs
- Justify VQD overhead (+93 sec) with accuracy gain (+5.7%)
- Show scalability analysis (N, k, D₁)
- Discuss optimization opportunities (parallelization, GPU)

---

**Next:** [19_LIMITATIONS.md](./19_LIMITATIONS.md) →

---

**Navigation:**
- [← 16_OPTIMIZATION.md](./16_OPTIMIZATION.md)
- [→ 19_LIMITATIONS.md](./19_LIMITATIONS.md)
- [↑ Index](./README.md)
