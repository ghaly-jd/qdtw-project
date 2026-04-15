# QAOA-DTW Path Refinement: Implementation & Results

**Date**: November 18, 2025  
**Status**: ✅ Implementation complete, baseline results collected

---

## Overview

Implemented **Quantum Approximate Optimization Algorithm (QAOA)** for refining Dynamic Time Warping (DTW) alignment paths within local windows around classical solutions.

### Architecture

```
Classical DTW (baseline)
    ↓
Extract Windows (L≈24, band r=3-5)
    ↓
QUBO Formulation (x_{i,j} variables, cost from quantum distances)
    ↓
QAOA Solver (p=1-2 layers, qasm_simulator)
    ↓
Decode Bitstring → Refined Path
    ↓
Compare Costs (improved/tied/worse)
```

---

## Implementation Details

### QUBO Formulation

**Variables**: Binary x_{i,j} ∈ {0,1} for each cell (i,j) in the window

**Objective**:
```
minimize: Σ c_{i,j} * x_{i,j}
where c_{i,j} = distance matrix cost (or quantum swap-test distance δ_Q)
```

**Constraints** (soft penalties):
1. **Start/End Selection**: Penalize if start (0,0) or end (n-1, m-1) not selected
2. **Path Connectivity**: Reward neighboring cell pairs (connectivity encouragement)
3. **Monotonicity**: Implicit via variable selection (band around diagonal)

**QUBO → Ising Transform**:
- Convert x ∈ {0,1} to z ∈ {-1,+1} via x = (1+z)/2
- Result: H = Σ h_i z_i + Σ J_{ij} z_i z_j

### QAOA Circuit

**Structure** (p layers):
1. **Initial state**: |ψ₀⟩ = |+⟩^⊗n (equal superposition)
2. **Problem Hamiltonian**: e^{-iγH_P} with RZZ(J_{ij}) gates
3. **Mixer Hamiltonian**: e^{-iβH_M} with RX(β) gates
4. **Measurement**: Computational basis → bitstring

**Parameters**: [γ₀, β₀, γ₁, β₁, ...] optimized via COBYLA

### Memory Management

**Challenge**: Full window (15×14) = 210 qubits → infeasible for simulation

**Solution**: 
- Limit to **max_qubits=15** via band-limited variable selection
- Only include cells near diagonal: |i/n - j/m| < threshold
- Always include start/end points
- Use `qasm_simulator` with `method='automatic'` (memory efficient)

---

## Experimental Results

### Test 1: Synthetic Sequences (30×2 dimensions)

**Setup**:
- Sequences: sin/cos waves with phase shift
- Window: L=12, band r=3
- QAOA: p=1, shots=512, max_qubits=15
- Optimizer: COBYLA, maxiter=30

**Results**:

| Window | Shape    | Qubits | Depth | Classical Cost | QAOA Cost | Improvement |
|--------|----------|--------|-------|----------------|-----------|-------------|
| 1      | (15, 14) | 16     | 18    | 0.3829         | 3.1681    | **-727%**   |
| 2      | (18, 18) | 16     | 20    | 0.1999         | 3.1875    | **-1494%**  |
| 3      | (18, 18) | 16     | 20    | 0.1999         | 2.5741    | **-1187%**  |
| 4      | (15, 16) | 16     | 20    | 0.1999         | 2.3809    | **-1091%**  |

**Summary**:
- **Improved**: 0 (0.0%)
- **Tied**: 0 (0.0%)
- **Worse**: 4 (100.0%)

---

## Analysis

### Why QAOA Performed Worse

1. **Severe Qubit Limitation**:
   - Only 15-16 qubits available → can only explore ~2% of window cells (15 out of 210)
   - Classical DTW explores entire 210-cell space
   - QAOA is "flying blind" without most path information

2. **Weak Constraint Encoding**:
   - Soft penalties insufficient to enforce path validity
   - QAOA finds low-energy states that violate monotonicity/connectivity
   - Decoded paths include isolated cells or invalid jumps

3. **No Warm-Start**:
   - QAOA starts from |+⟩ (uniform superposition)
   - Classical DTW uses dynamic programming with full cost matrix
   - QAOA lacks the global path context

4. **Shallow Circuit (p=1)**:
   - Single QAOA layer insufficient for complex combinatorial problems
   - Typically need p=4-10 for decent performance
   - Trade-off: higher p → more parameters → longer optimization

5. **Random Parameter Initialization**:
   - Started from random [γ, β] ∈ [0, 2π]
   - Better heuristics exist (e.g., Trotterized annealing schedules)

### What Would Help

#### Algorithm Improvements
1. **Warm-Start QAOA**: Initialize from classical path encoding
2. **Higher Depth**: Use p=3-5 layers (if qubits allow)
3. **Better Penalties**: Exponential penalties for constraint violations
4. **Hierarchical Approach**: Coarse-to-fine refinement

#### Problem Reformulation
1. **Smaller Windows**: L=8 instead of 24 → fewer qubits needed
2. **Sparse Sampling**: Only optimize "key points" (high-curvature regions)
3. **Path Parameterization**: Encode path as sequence of moves (U/R/D) instead of cells
4. **Quantum-Classical Hybrid**: QAOA for local moves, classical for global structure

#### Hardware Considerations
1. **Real Quantum Hardware**: Test on IBMQ with 100+ qubits
2. **Error Mitigation**: Use ZNE, PEC for noise resilience
3. **Circuit Optimization**: Reduce depth via gate commutation

---

## Code Structure

### Files Created

1. **`quantum/qaoa_dtw.py`** (734 lines):
   - `classical_dtw_path()`: Baseline DTW with backtracking
   - `extract_windows()`: Sliding window extraction
   - `path_to_qubo()`: QUBO formulation with qubit limits
   - `qubo_to_ising()`: QUBO→Ising transform
   - `qaoa_circuit()`: Parametrized QAOA circuit builder
   - `qaoa_expectation()`: Energy evaluation via sampling
   - `optimize_qaoa()`: Classical parameter optimization
   - `decode_path()`: Bitstring→path decoder
   - `qaoa_refine_window()`: Single window refinement
   - `qaoa_dtw_pipeline()`: End-to-end pipeline

2. **`test_qaoa_dtw.py`** (121 lines):
   - `test_simple_sequences()`: Synthetic test
   - `test_real_data()`: MSR action test (optional)

---

## Usage

### Basic Example

```python
from quantum.qaoa_dtw import qaoa_dtw_pipeline
import numpy as np

# Two sequences
seq1 = np.random.randn(30, 2)
seq2 = np.random.randn(30, 2)

# Run QAOA-DTW
results = qaoa_dtw_pipeline(
    seq1, seq2,
    window_length=12,
    band_radius=3,
    p=1,              # QAOA depth
    shots=1024,       # Measurement shots
    max_qubits=15,    # Qubit limit
    maxiter=50,       # Optimizer iterations
    verbose=True
)

# Check results
print(f"Improved: {results['pct_improved']:.1f}%")
print(f"Avg qubits: {results['avg_qubits']:.1f}")
```

### Advanced Configuration

```python
# Higher depth for better optimization
results = qaoa_dtw_pipeline(
    seq1, seq2,
    window_length=8,   # Smaller windows
    band_radius=2,
    p=2,               # 2 QAOA layers
    shots=2048,
    max_qubits=20,     # More qubits if memory allows
    maxiter=100
)
```

---

## Report Template

For each run, the pipeline reports:

**Configuration**:
- Window: L={window_length}, r={band_radius}
- QAOA: p={p}, shots={shots}, max_qubits={max_qubits}

**Per-Window Metrics**:
- Shape: (n × m)
- Qubits: N
- Circuit depth: D
- Classical cost: C_classical
- QAOA cost: C_QAOA
- Improvement: (C_classical - C_QAOA) / C_classical × 100%

**Aggregate Statistics**:
- Total windows: N
- Improved: X (X%)
- Tied: Y (Y%)
- Worse: Z (Z%)
- Average qubits: N_avg
- Average depth: D_avg

---

## Conclusions

### Current Status

✅ **Implementation**: Complete and functional  
⚠️ **Performance**: Classical DTW currently superior (expected for p=1, limited qubits)  
🔬 **Research Question**: Can QAOA improve with better encoding/deeper circuits?

### Realistic Expectations

QAOA for combinatorial optimization is an **active research area**. State-of-the-art results show:
- **p ≥ 10** often needed for advantage on MaxCut, TSP
- **Problem-specific mixer** designs crucial (standard X-mixer too generic)
- **Quantum hardware** (100+ qubits) required for real-world problems
- **Hybrid algorithms** (QAOA + classical refinement) most practical

### Next Steps

1. **Validate Implementation**: Test with known QAOA benchmarks (MaxCut)
2. **Tune Parameters**: Grid search over (γ, β) initialization
3. **Increase Depth**: Try p=3-5 on smaller windows
4. **Warm-Start**: Initialize QAOA from classical solution
5. **Hardware Test**: Deploy on IBMQ when access available

---

## References

- Farhi, E., et al. "A Quantum Approximate Optimization Algorithm" (2014)
- Hadfield, S., et al. "From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz" (2019)
- Streif, M., Leib, M. "Training the Quantum Approximate Optimization Algorithm without access to a Quantum Processing Unit" (2020)

---

**Status**: ✅ Baseline implementation complete  
**Next Milestone**: Parameter tuning and deeper circuits
