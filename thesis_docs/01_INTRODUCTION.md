# 01 - Introduction and Research Context

**File:** `01_INTRODUCTION.md`  
**Purpose:** Research motivation, problem statement, and contributions  
**For Thesis:** Introduction chapter

---

## 1.1 Research Motivation

### The Challenge: Temporal Action Recognition

Human action recognition from skeletal data is a fundamental problem in computer vision with applications in:
- **Healthcare:** Patient monitoring, fall detection, rehabilitation
- **Human-Computer Interaction:** Gesture recognition, gaming
- **Security:** Surveillance, anomaly detection
- **Robotics:** Human-robot interaction, behavior understanding

**Key Challenges:**
1. **High Dimensionality:** Skeletal data typically has 60+ dimensions (20 joints × 3 coordinates)
2. **Temporal Variability:** Actions occur at different speeds and durations
3. **Intra-class Variation:** Same action performed differently by different people
4. **Computational Cost:** Real-time processing requirements

### Why Dimensionality Reduction Matters

Raw 60D skeletal features contain:
- ✅ **Signal:** Discriminative motion patterns
- ❌ **Noise:** Sensor errors, tracking jitter
- ❌ **Redundancy:** Correlated joint movements

**Without reduction:**
- DTW computational cost: O(T²D) per pair
- Curse of dimensionality affects distance metrics
- Overfitting on high-dimensional sparse data

**With reduction (60D → 8D):**
- 87% reduction in computation
- Better generalization
- Noise removal improves accuracy

---

## 1.2 Research Questions

This thesis addresses three fundamental questions:

### RQ1: Can quantum-inspired methods improve dimensionality reduction for temporal data?

**Hypothesis:** Variational Quantum Deflation (VQD) can discover better subspaces than classical PCA by exploring the quantum-inspired feature space.

**Why it might work:**
- PCA is limited to linear, greedy eigenvector search
- VQD uses parameterized quantum circuits for global optimization
- Orthogonality penalties allow finding diverse, discriminative directions

**Tested:** Yes ✓ (Result: +5.0% improvement)

### RQ2: What is the optimal pre-processing pipeline for quantum-inspired methods?

**Hypothesis:** VQD benefits from classical pre-reduction to remove noise before quantum optimization.

**Sub-questions:**
- What is the optimal pre-reduction dimensionality? (Tested: 8D, 12D, 16D, 20D, 24D, 32D)
- Is pre-reduction necessary? (Tested: with vs without)
- How does it affect VQD quality metrics?

**Tested:** Yes ✓ (Result: 20D optimal, +5.7% advantage)

### RQ3: How does the approach scale with target dimensionality?

**Hypothesis:** There exists an optimal target dimension k that balances information retention and computational efficiency.

**Tested:** Yes ✓ (Result: k=8 optimal, 82.7% accuracy)

---

## 1.3 Problem Statement

**Formal Problem Definition:**

Given:
- Skeletal sequence dataset: $\mathcal{D} = \{(X_i, y_i)\}_{i=1}^N$
- Each sequence: $X_i \in \mathbb{R}^{T_i \times D}$ where $T_i$ = variable length, $D=60$
- Class labels: $y_i \in \{1, \ldots, C\}$ where $C=20$ action classes

Find:
- Projection operator $\Pi: \mathbb{R}^{D} \rightarrow \mathbb{R}^{k}$ where $k \ll D$
- Such that: DTW-based classification accuracy is maximized on projected sequences

Constraints:
- $\Pi$ must preserve temporal action patterns
- Computational cost: feasible for real-time inference
- Generalization: works across different subjects and viewpoints

**Why it's hard:**
1. **Temporal structure:** Can't just vectorize sequences (loses time information)
2. **Variable length:** $T_i$ varies from 13 to 255 frames
3. **Non-linear patterns:** Actions are not linearly separable in raw space
4. **Class imbalance:** Some actions are more common than others

---

## 1.4 Proposed Solution: VQD-DTW Pipeline

Our approach combines:
1. **Classical pre-reduction:** Remove noise (60D → 20D via PCA)
2. **Quantum-inspired subspace learning:** Find discriminative directions (20D → kD via VQD)
3. **Temporal sequence projection:** Project each sequence frame-by-frame
4. **DTW classification:** Align and compare projected sequences

**Key Innovation:** Using VQD's quantum-inspired optimization to find **better** subspaces than classical PCA for temporal action data.

**Pipeline:**
```
Input: 60D Skeleton Sequence
       ↓
[StandardScaler] → Normalize features
       ↓
[Classical PCA] → Pre-reduce to 20D (remove noise)
       ↓
[VQD Quantum PCA] → Further reduce to 8D (discriminative subspace)
       ↓
[Per-Sequence Centering] → Remove translation bias
       ↓
[DTW 1-NN] → Classify via nearest neighbor
       ↓
Output: Action Class Prediction
```

---

## 1.5 Contributions

This thesis makes the following contributions:

### 1. **Novel VQD-DTW Framework** (Technical)
- First application of VQD to temporal action recognition
- Custom pipeline with pre-reduction and per-sequence centering
- Qiskit-based implementation with hardware-efficient ansatz

**Impact:** Demonstrates quantum-inspired methods can improve classical tasks

### 2. **Optimal Pre-Reduction Analysis** (Empirical)
- Systematic study of pre-reduction size (8D to 32D)
- Discovery: 20D is optimal (99.0% variance, +5.7% VQD advantage)
- Validation: Pre-reduction is **essential** for VQD (without it: 0% advantage)

**Impact:** Provides design guidelines for hybrid classical-quantum pipelines

### 3. **Comprehensive Experimental Validation** (Methodological)
- Statistical validation: 5 seeds × 4 k-values × 6 pre-dims = 120 experiments
- Ablation studies: necessity of each component
- Per-class analysis: VQD excels on dynamic actions (+66.7% on kicks/waves)

**Impact:** Rigorous methodology ensures reproducible, reliable results

### 4. **Open-Source Implementation** (Practical)
- Complete codebase with documentation
- Reproducible experiments with fixed seeds
- Visualization scripts for all figures

**Impact:** Enables future research and extensions

---

## 1.6 Thesis Organization

**Chapter 2: Background**
- Skeletal action recognition
- Dimensionality reduction methods
- Quantum-inspired algorithms
- Dynamic Time Warping

**Chapter 3: Methodology** ← **Files 04-09**
- Data preprocessing (04, 05)
- Classical pre-reduction (06)
- VQD quantum PCA (07) ⭐
- Sequence projection (08)
- DTW classification (09)

**Chapter 4: Experimental Setup** ← **Files 10**
- Dataset description (02)
- Train/test split strategy
- Hyperparameter selection
- Evaluation metrics

**Chapter 5: Results** ← **Files 11-14**
- Pre-reduction optimization (11)
- K-sweep experiments (12)
- VQD vs PCA comparison
- Visualization (14)

**Chapter 6: Discussion** ← **Files 13, 18, 19**
- Ablation studies (13)
- Failed experiments (18) 🔥
- Limitations (19)
- Future work

**Chapter 7: Conclusions** ← **File 20**
- Summary of findings
- Contributions
- Impact and applications

---

## 1.7 Key Results Preview

To motivate the reader, here are the headline results:

| Metric | Value | Comparison |
|--------|-------|------------|
| **VQD Advantage** | +5.0% | vs Classical PCA |
| **Optimal Pre-dim** | 20D | 99.0% variance retained |
| **Optimal Target k** | 8D | 82.7% accuracy |
| **Best Config** | 60D→20D→8D | Statistically significant (5 seeds) |
| **Per-Class Boost** | +66.7% | On dynamic actions (kicks/waves) |

**Statistical Significance:**
- 5 random seeds: [42, 123, 456, 789, 2024]
- Mean gap: +5.0% ± 3.3% (non-overlapping confidence intervals)
- p < 0.01 (t-test for k=8)

**Computational Cost:**
- VQD training: ~5 seconds (4 qubits, 200 iterations)
- Total pipeline: ~2 minutes per seed (300 train + 60 test)
- Inference: Real-time capable after training

---

## 1.8 Target Audience

This documentation is written for:

1. **Thesis Committee Members**
   - Focus: Scientific rigor, novel contributions
   - Read: 01, 03, 07, 11-12, 20

2. **Machine Learning Researchers**
   - Focus: Technical details, reproducibility
   - Read: 07 (VQD), 11-13 (experiments), 18 (failures)

3. **Quantum Computing Practitioners**
   - Focus: Quantum algorithm design, optimization
   - Read: 07, 15, 16

4. **Future Students/Developers**
   - Focus: Implementation, code understanding
   - Read: 03-09 (pipeline), Appendix A (code)

---

## 1.9 Notation and Conventions

Throughout this documentation:

**Scalars:** lowercase $x$, $y$  
**Vectors:** lowercase bold $\mathbf{x}$, $\mathbf{y}$  
**Matrices:** uppercase bold $\mathbf{X}$, $\mathbf{U}$  
**Sets:** calligraphic $\mathcal{D}$, $\mathcal{X}$  
**Operators:** $\Pi$ (projection), $\mathcal{H}$ (Hamiltonian)

**Dimensions:**
- $N$ = number of sequences (567 total, 300 train, 60 test)
- $D$ = feature dimension (60 for raw data)
- $T$ = sequence length (variable, 13-255 frames)
- $k$ = target reduced dimension (typically 8)
- $C$ = number of classes (20 actions)

**Quantum:**
- $|\psi\rangle$ = quantum state (ket notation)
- $\langle\psi|$ = conjugate transpose (bra notation)
- $\langle\psi|\phi\rangle$ = inner product (overlap)
- $\theta$ = circuit parameters

---

## 1.10 Prerequisites

**Required Background:**
- Linear algebra (eigenvalues, projections, orthogonality)
- Machine learning (train/test split, cross-validation, overfitting)
- Basic quantum computing (qubits, gates, statevector)
- Time series analysis (temporal sequences, alignment)

**Optional (Helpful):**
- Variational quantum algorithms (VQE, QAOA)
- Dynamic programming (DTW algorithm details)
- Skeletal action recognition (domain knowledge)

**Software Requirements:**
- Python 3.9+
- NumPy, SciPy, scikit-learn
- Qiskit, Qiskit Aer
- Matplotlib (for visualization)

---

**Next:** [02_DATASET.md](./02_DATASET.md) - MSR Action3D Dataset Description →

---

**Navigation:**
- [← Index](./README.md)
- [02_DATASET.md →](./02_DATASET.md)
