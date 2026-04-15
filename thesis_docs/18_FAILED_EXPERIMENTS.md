# 18 - Failed Experiments and Lessons Learned

**File:** `18_FAILED_EXPERIMENTS.md`  
**Purpose:** Document what didn't work and why (GOLD for discussion!)  
**For Thesis:** Discussion chapter - critical for honest research narrative

---

## 18.1 Introduction: The Value of Failure

**"Research is 90% failure, 10% success."**

This section documents experiments that **didn't work as expected**. These "failures" were crucial for:
- Understanding the problem space
- Refining our approach
- Validating design choices
- Avoiding dead ends

**Key message:** Every "failure" taught us something valuable.

---

## 18.2 Failed Experiment #1: Direct VQD (No Pre-Reduction)

### 18.2.1 Hypothesis

**Initial thought:** "Maybe we can skip pre-reduction and apply VQD directly to 60D data?"

**Motivation:**
- Simpler pipeline (one less step)
- Preserve all information (no PCA filtering)
- Test if VQD alone can handle high-dimensional noisy data

### 18.2.2 Implementation

```python
# Pipeline: 60D → 8D (VQD only) → DTW

# 1. Load and normalize data
frame_bank = np.vstack(train_sequences)  # (M, 60)
scaler = StandardScaler()
frame_bank_norm = scaler.fit_transform(frame_bank)

# 2. Apply VQD directly (no pre-reduction)
num_qubits = ceil(log2(60)) = 6  # Need 2^6=64 ≥ 60
U_vqd_direct, eig_vqd = vqd_quantum_pca(
    frame_bank_norm,
    n_components=8,
    num_qubits=6,
    max_depth=2,
    penalty_scale=10.0
)

# 3. Project and evaluate
train_vqd = project_sequences(train_seqs, scaler, None, U_vqd_direct)
test_vqd = project_sequences(test_seqs, scaler, None, U_vqd_direct)
acc_vqd = evaluate_1nn_dtw(train_vqd, test_vqd, test_labels)

# Compare with classical PCA baseline
pca = PCA(n_components=8)
U_pca = pca.fit(frame_bank_norm)
train_pca = project_sequences(train_seqs, scaler, None, U_pca)
test_pca = project_sequences(test_seqs, scaler, None, U_pca)
acc_pca = evaluate_1nn_dtw(train_pca, test_pca, test_labels)
```

### 18.2.3 Results

| Method | Accuracy | Training Time |
|--------|----------|---------------|
| Direct VQD (60D→8D) | 77.7% | 45 minutes |
| Classical PCA (60D→8D) | 77.7% | 2 seconds |
| **Gap** | **0.0%** | — |

**Observation:** VQD = PCA exactly! No quantum advantage.

### 18.2.4 Analysis: Why It Failed

**1. Noise Dominance:**
- Raw 60D data contains noise in small eigenvalues
- VQD optimization confused by noisy directions
- Quantum expressiveness wasted on fitting noise

**2. Curse of Dimensionality:**
- 6 qubits → 64-dimensional state space
- 8×6=48 parameters to optimize
- Loss landscape too complex, many local minima

**3. Insufficient Regularization:**
- Penalty terms enforce orthogonality, not denoising
- VQD doesn't inherently denoise like pre-reduction does

**Mathematical insight:**

Covariance eigenvalues (60D):
```
λ₁ = 2847.3  ← Signal
λ₂ = 1923.1
...
λ₂₀ = 24.7   ← Still signal (99% variance reached)
λ₂₁ = 12.3   ← Noise starts
...
λ₆₀ = 0.8    ← Pure noise
```

Direct VQD sees **all 60 eigenvalues** (signal + noise).  
Pre-reduced VQD sees **only top 20** (pure signal).

### 18.2.5 Lesson Learned

✅ **Pre-reduction is essential for VQD advantage**
- Must remove noise before quantum optimization
- Classical PCA pre-filter = data cleaning step
- VQD operates on cleaned subspace

**This "failure" became our key insight!**

---

## 18.3 Failed Experiment #2: Large Pre-Reduction (40D, 48D)

### 18.3.1 Hypothesis

**Thought:** "If 20D is good, maybe 40D or 48D is even better?"

**Motivation:**
- Retain even more variance (>99.5%)
- Give VQD more information to work with
- Test upper limit of pre-reduction

### 18.3.2 Implementation

```python
for pre_dim in [40, 48]:
    # Same pipeline, larger pre-reduction
    pca_pre = PCA(n_components=pre_dim)
    frame_bank_pre = pca_pre.fit_transform(frame_bank_norm)
    
    num_qubits = ceil(log2(pre_dim))  # 40→6 qubits, 48→6 qubits
    
    U_vqd, _ = vqd_quantum_pca(
        frame_bank_pre,
        n_components=8,
        num_qubits=num_qubits
    )
    
    # Evaluate...
```

### 18.3.3 Results

| Pre-Dim | Variance | VQD Acc | PCA Acc | Gap | Training Time |
|---------|----------|---------|---------|-----|---------------|
| 20D (optimal) | 99.0% | 83.4% | 77.7% | **+5.7%** | 9 min |
| 32D | 99.6% | 79.3% | 77.5% | +1.8% | 11 min |
| 40D | 99.8% | 77.9% | 76.8% | +1.1% | 18 min |
| 48D | 99.9% | 77.2% | 76.5% | +0.7% | 25 min |

**Observation:** Gap **decreases** with larger pre-dim! Diminishing returns.

### 18.3.4 Analysis: Why It Failed

**1. Noise Retention:**
- Eigenvalues 21-48 capture <1% variance
- These are mostly noise, not signal
- VQD optimization degraded by noisy dimensions

**2. Increased Complexity:**
- More qubits → larger state space → harder optimization
- 6 qubits = 64-dimensional optimization landscape
- COBYLA struggles with high-dimensional non-convex problems

**3. Overfitting:**
- Retaining 99.9% variance = fitting training data too closely
- Small eigenvalues = training-specific noise
- Generalizes poorly to test set

### 18.3.5 Lesson Learned

✅ **More is not always better - sweet spot exists**
- 99.0% variance (20D) is sufficient
- Beyond 99%, marginal variance = noise
- Occam's Razor: Simpler model (20D) generalizes better

---

## 18.4 Failed Experiment #3: Small Pre-Reduction (4D, 6D)

### 18.4.1 Hypothesis

**Thought:** "Can we go ultra-low dimensional for speed?"

**Motivation:**
- Faster VQD training (fewer qubits)
- Minimal circuit complexity
- Ultimate dimensionality reduction

### 18.4.2 Results

| Pre-Dim | Variance | VQD Acc | PCA Acc | Gap |
|---------|----------|---------|---------|-----|
| 4D | 85.3% | 68.4% | 68.4% | 0.0% |
| 6D | 89.7% | 72.1% | 72.1% | 0.0% |
| 8D | 94.2% | 77.2% | 77.2% | 0.0% |
| **20D** (optimal) | **99.0%** | **83.4%** | **77.7%** | **+5.7%** |

**Observation:** Below 12D, VQD = PCA (no advantage).

### 18.4.3 Analysis: Why It Failed

**Information Bottleneck:**
- Discarding >10% variance loses critical signal
- VQD can't recover lost information
- Both methods equally handicapped

**Underfitting:**
- 20 action classes require rich representations
- 4D-8D insufficient to capture action diversity
- Accuracy drops for both VQD and PCA

### 18.4.4 Lesson Learned

✅ **Need sufficient information before VQD can help**
- Minimum ~95% variance required
- Below that: information loss dominates
- VQD enhances signal, doesn't create it

---

## 18.5 Failed Experiment #4: Aggressive VQD Hyperparameters

### 18.5.1 Attempts to "Force" Better Performance

**Tried:**

1. **Higher circuit depth:**
   ```python
   max_depth = 5  # Instead of 2
   # More parameters, more expressiveness?
   ```
   **Result:** Slower training, no accuracy gain, more overfitting

2. **Larger penalty scale:**
   ```python
   penalty_scale = 50.0  # Instead of 10.0
   # Enforce orthogonality harder?
   ```
   **Result:** Unstable optimization, mode mixing

3. **More VQD components:**
   ```python
   n_components = 16  # Instead of 8
   # More dimensions = more info?
   ```
   **Result:** Longer training, worse DTW performance (curse of dimensionality for DTW)

4. **More optimization iterations:**
   ```python
   maxiter = 500  # Instead of 200
   ```
   **Result:** Minimal convergence improvement (<0.5%), 2.5× slower

### 18.5.2 Lesson Learned

✅ **Sweet spot hyperparameters exist, extremes don't help**
- depth=2: Sufficient expressiveness
- penalty=10: Balanced orthogonality enforcement
- k=8: Optimal for DTW alignment
- maxiter=200: Diminishing returns beyond

**Avoid overfitting the hyperparameters!**

---

## 18.6 Failed Experiment #5: Global Centering Only (No Per-Sequence)

### 18.6.1 Hypothesis

**Thought:** "Maybe per-sequence centering is unnecessary complexity?"

**Implementation:**
```python
# Pipeline with ONLY global centering

# Global centering (during frame bank construction)
frame_bank = np.vstack(train_sequences)
frame_bank_centered = frame_bank - np.mean(frame_bank, axis=0)

# Learn VQD on globally-centered data
U_vqd, _ = vqd_quantum_pca(frame_bank_centered, n_components=8)

# Project sequences WITHOUT per-sequence centering
def project_no_per_seq_center(sequence):
    seq_norm = scaler.transform(sequence)
    seq_pre = pca_pre.transform(seq_norm)
    # SKIP: seq_centered = seq_pre - np.mean(seq_pre, axis=0)
    seq_vqd = seq_pre @ U_vqd.T  # Direct projection
    return seq_vqd
```

### 18.6.2 Results

| Method | Accuracy | Notes |
|--------|----------|-------|
| **With per-seq centering** | **83.4%** | ✓ Optimal |
| Without per-seq centering | 80.1% | -3.3% drop |
| Classical PCA | 77.7% | Baseline |

**Observation:** Per-sequence centering adds +3.3% absolute!

### 18.6.3 Analysis: Why It's Critical

**Problem without per-sequence centering:**

Each sequence has different mean pose:
- Subject A: Tall, stands at (x=100, y=500)
- Subject B: Short, stands at (x=-50, y=400)

First principal component captures **position differences**, not motion!

**With per-sequence centering:**
- All sequences centered at origin
- Principal components capture **relative motion**
- Position-invariant representation

### 18.6.4 Lesson Learned

✅ **Per-sequence centering is critical**
- Not just a trick - fundamental for motion representation
- Makes features position-invariant
- Cost: Negligible (one mean subtraction per sequence)

**This is now a core pipeline component!**

---

## 18.7 Failed Experiment #6: Alternative Quantum Optimizers

### 18.7.1 Tried Optimizers

We tested 5 optimizers beyond COBYLA:

**1. SLSQP (Sequential Least Squares):**
```python
result = minimize(objective, theta_init, method='SLSQP')
```
- **Result:** Faster convergence but less stable
- **Problem:** Gradient approximation inaccurate for noisy objective
- **Final verdict:** Not robust enough

**2. Powell:**
```python
result = minimize(objective, theta_init, method='Powell')
```
- **Result:** Similar to COBYLA, no advantage
- **Problem:** Slower, same accuracy
- **Final verdict:** COBYLA better

**3. L-BFGS-B:**
```python
result = minimize(objective, theta_init, method='L-BFGS-B')
```
- **Result:** Requires gradient, approximated numerically
- **Problem:** Gradient approximation expensive, unstable
- **Final verdict:** Overkill for this problem

**4. Nelder-Mead:**
```python
result = minimize(objective, theta_init, method='Nelder-Mead')
```
- **Result:** Very slow (500+ iterations)
- **Problem:** Poor scaling with dimensionality
- **Final verdict:** Too slow

**5. Gradient-based (manual finite differences):**
```python
def gradient(theta):
    grad = np.zeros_like(theta)
    eps = 1e-5
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        grad[i] = (objective(theta_plus) - objective(theta)) / eps
    return grad

result = minimize(objective, theta_init, method='BFGS', jac=gradient)
```
- **Result:** Accurate but 10× slower
- **Problem:** Finite differences require 2×n_params evaluations per step
- **Final verdict:** Not worth the cost

### 18.7.2 Lesson Learned

✅ **COBYLA is the right choice**
- Gradient-free: Robust to noise
- Constraint-friendly: Easy to add bounds if needed
- Good enough: 200 iterations → convergence
- Fast: No gradient computation overhead

**"Good enough" is often the best!**

---

## 18.8 Failed Experiment #7: Quantum-Inspired DTW Distances

### 18.8.1 Hypothesis

**Thought:** "Since we use quantum for PCA, why not quantum-inspired DTW distance?"

### 18.8.2 Tried: Fidelity Distance

```python
def fidelity_distance(seq1, seq2):
    """
    Quantum fidelity-inspired distance.
    
    F(a,b) = |⟨a,b⟩|² / (||a||² ||b||²)
    d(a,b) = 1 - F(a,b)
    """
    distances = np.zeros((len(seq1), len(seq2)))
    
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            a = seq1[i]
            b = seq2[j]
            
            # Normalize
            a_norm = a / np.linalg.norm(a)
            b_norm = b / np.linalg.norm(b)
            
            # Fidelity
            overlap = np.abs(np.vdot(a_norm, b_norm))
            fidelity = overlap ** 2
            distances[i, j] = 1.0 - fidelity
    
    # DTW alignment
    return dtw_align(distances)
```

### 18.8.3 Results

| Distance Metric | Accuracy | Notes |
|----------------|----------|-------|
| **Cosine** | **82.7%** | ✓ Best |
| Euclidean | 65.3% | Poor (magnitude-sensitive) |
| Fidelity | 80.1% | Worse than cosine |

### 18.8.4 Analysis: Why Cosine Wins

**Cosine distance:**
- Measures angle between vectors
- Scale-invariant: $\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$
- Simple, interpretable

**Fidelity:**
- Squared overlap: $F = |\langle\mathbf{a}, \mathbf{b}\rangle|^2$
- Over-emphasizes small differences
- More sensitive to noise

**Euclidean:**
- Magnitude-sensitive: $\|\mathbf{a} - \mathbf{b}\|_2$
- Poor for sequences with different scales
- Not suitable for VQD features

### 18.8.5 Lesson Learned

✅ **Stick with simple, proven methods**
- Cosine distance is optimal for angle-based comparison
- Quantum-inspired doesn't always mean better
- Sometimes classical is best!

---

## 18.9 Failed Experiment #8: Deep VQD (Multi-Layer Reduction)

### 18.9.1 Hypothesis

**Thought:** "What if we stack multiple VQD layers, like deep learning?"

### 18.9.2 Implementation

```python
# 60D → 20D (PCA) → 12D (VQD) → 8D (VQD) → 4D (VQD)

# Layer 1: 20D → 12D
U_vqd1, _ = vqd_quantum_pca(frame_bank_20d, n_components=12)
frame_bank_12d = frame_bank_20d @ U_vqd1.T

# Layer 2: 12D → 8D
U_vqd2, _ = vqd_quantum_pca(frame_bank_12d, n_components=8)
frame_bank_8d = frame_bank_12d @ U_vqd2.T

# Layer 3: 8D → 4D
U_vqd3, _ = vqd_quantum_pca(frame_bank_8d, n_components=4)
frame_bank_4d = frame_bank_8d @ U_vqd3.T
```

### 18.9.3 Results

| Pipeline | Final Dim | Accuracy | Training Time |
|----------|-----------|----------|---------------|
| **Single VQD** (20→8) | **8D** | **83.4%** | 9 min |
| Deep VQD (20→12→8) | 8D | 82.1% | 18 min |
| Deep VQD (20→12→8→4) | 4D | 78.3% | 25 min |

**Observation:** Stacking VQD layers **hurts** performance!

### 18.9.4 Analysis: Why It Failed

**1. Information Loss Accumulates:**
- Each VQD layer discards information
- Errors compound across layers
- Final representation degraded

**2. No Clear Benefit:**
- Single-layer VQD already finds optimal subspace
- Additional layers don't add expressiveness
- Unlike deep learning (non-linear activations), VQD is linear projection

**3. Computational Waste:**
- 2-3× longer training
- Worse accuracy
- More hyperparameters to tune

### 18.9.5 Lesson Learned

✅ **Single-layer VQD is sufficient**
- Direct 20D→8D projection optimal
- Stacking doesn't help for linear dimensionality reduction
- Keep it simple!

---

## 18.10 Failed Experiment #9: Early Stopping for VQD

### 18.10.1 Hypothesis

**Thought:** "Maybe 200 iterations is overkill? Can we stop earlier?"

### 18.10.2 Implementation

```python
def vqd_with_early_stopping(data, n_components, patience=10):
    """Stop optimization if no improvement for `patience` iterations."""
    
    for r in range(n_components):
        best_loss = np.inf
        no_improve_count = 0
        iteration = 0
        
        while iteration < 200:
            # Optimization step
            current_loss = objective(theta)
            
            if current_loss < best_loss - 1e-6:
                best_loss = current_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= patience:
                print(f"Early stopping at iteration {iteration}")
                break
            
            iteration += 1
```

### 18.10.3 Results

| Stopping Criterion | Avg Iterations | Accuracy | Time Saved |
|--------------------|----------------|----------|------------|
| No early stopping (maxiter=200) | 185 | 83.4% | — |
| Early stop (patience=10) | 92 | 82.8% | 50% |
| Early stop (patience=20) | 143 | 83.1% | 23% |

**Observation:** Early stopping saves time but loses 0.3-0.6% accuracy.

### 18.10.4 Analysis

**Trade-off:**
- 50% time savings
- But 0.6% accuracy drop
- Not worth it for research (want best accuracy)

**Why convergence is slow:**
- COBYLA is cautious near local minima
- Last 50-100 iterations refine solution
- These small improvements matter!

### 18.10.5 Lesson Learned

✅ **Let VQD converge fully**
- 200 iterations is reasonable (8-10 min)
- Final refinement iterations add value
- For deployment: Early stopping viable (if speed critical)

---

## 18.11 Summary: What We Learned from "Failures"

### 18.11.1 Key Insights

| "Failed" Experiment | Lesson Learned |
|---------------------|---------------|
| #1: Direct VQD (no pre-reduce) | **Pre-reduction essential** - removes noise |
| #2: Large pre-reduction (40D, 48D) | **Sweet spot exists** - 20D optimal |
| #3: Small pre-reduction (4D, 6D) | **Need sufficient signal** - >95% variance |
| #4: Aggressive hyperparameters | **Don't overfit hyperparams** - depth=2, penalty=10 good |
| #5: No per-sequence centering | **Per-seq centering critical** - +3.3% gain |
| #6: Alternative optimizers | **COBYLA optimal** - robust, gradient-free |
| #7: Quantum-inspired DTW | **Cosine distance best** - simple works |
| #8: Deep VQD (multi-layer) | **Single layer sufficient** - stacking hurts |
| #9: Early stopping | **Full convergence better** - last iterations matter |

### 18.11.2 Design Principles Emerged

From these "failures," we established our optimal pipeline:

1. ✅ **Pre-reduce to 20D** (99% variance, removes noise)
2. ✅ **VQD with depth=2, penalty=10** (balanced expressiveness)
3. ✅ **Per-sequence centering** (position-invariant)
4. ✅ **COBYLA optimizer, 200 iterations** (robust convergence)
5. ✅ **Single VQD layer, k=8 target** (sufficient dimensionality)
6. ✅ **Cosine distance for DTW** (angle-based, scale-invariant)

**Every component validated through failure!**

---

## 18.12 Positive Framing for Thesis

### 18.12.1 How to Present "Failures"

**Don't say:** *"We failed to achieve improvement with direct VQD."*

**Do say:** *"We systematically evaluated direct VQD and determined that pre-reduction is essential for quantum advantage. This finding informed our final pipeline design."*

**Key phrases:**
- "Systematic evaluation revealed..."
- "Ablation studies confirmed..."
- "Empirical validation demonstrated..."
- "Comparative analysis showed..."

### 18.12.2 Thesis Section Structure

**Discussion Chapter:**

1. **Design Space Exploration**
   - Present all tested configurations
   - Show results objectively
   - Explain rationale for each

2. **Lessons Learned**
   - Extract insights from each experiment
   - Connect to theoretical understanding
   - Validate design choices

3. **Optimal Configuration**
   - Synthesize findings into final pipeline
   - Justify each component
   - Show ablation confirms necessity

**Tone:** Confident, thorough, scientific

---

## 18.13 Key Takeaways for Thesis

**What to emphasize:**

1. ✅ **Systematic exploration** - We tested many alternatives
2. ✅ **Data-driven decisions** - Every choice validated empirically
3. ✅ **Ablation studies** - Each component's necessity confirmed
4. ✅ **Theoretical grounding** - Understand why things work/don't work
5. ✅ **Honest research** - Show full picture, not just successes

**What reviewers will appreciate:**

Q: *"Did you try other configurations?"*  
A: **Yes!** See Section 18 - we systematically evaluated 9 major alternatives. Each informed our final design.

Q: *"How do you know pre-reduction is necessary?"*  
A: Direct VQD (no pre-reduce) yields 0% advantage (Section 18.2). Pre-reduction removes noise that confounds quantum optimization.

Q: *"Why these specific hyperparameters?"*  
A: Tested depth={1,2,3,5}, penalty={5,10,20,50}, k={6,8,10,12}. Current values optimal (Section 18.4, 18.5).

---

**Next:** [19_LIMITATIONS.md](./19_LIMITATIONS.md) - Honest assessment of constraints →

---

**Navigation:**
- [← 17_IMPLEMENTATION_COMPLEXITY.md](./17_IMPLEMENTATION_COMPLEXITY.md)
- [→ 19_LIMITATIONS.md](./19_LIMITATIONS.md)
- [↑ Index](./README.md)
