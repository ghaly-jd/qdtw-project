# Real Quantum Implementation Plan

## Current Status (Nov 14, 2025)

**Problem**: Current "quantum" implementations are just classical simulations
- `quantum/qpca.py` → Classical density matrix eigendecomposition
- `dtw/fidelity_distance` → Classical overlap calculation
- No actual quantum circuits executed
- Misleading naming

## Goal

Add REAL quantum implementations that:
1. ✅ Run actual quantum circuits
2. ✅ Use quantum gates and measurements
3. ✅ Can execute on IBM Quantum hardware
4. ✅ Don't break existing pipeline
5. ✅ Allow comparison: Classical vs Simulated-Quantum vs Real-Quantum

---

## Implementation Plan

### Phase 1: Real Quantum PCA (HHL-based)

**File**: `quantum/real_qpca.py`

**Approach**: Implement quantum PCA using HHL algorithm for eigenvalue/eigenvector extraction

**Components**:
1. **State Preparation**: Encode data into quantum amplitudes
2. **Quantum Phase Estimation (QPE)**: Extract eigenvalues
3. **Controlled rotations**: Extract eigenvectors
4. **Measurement**: Collapse to classical results

**Dependencies**:
```bash
pip install qiskit qiskit-aer qiskit-ibm-runtime
```

**Code Structure**:
```python
def quantum_pca(X, k, backend='aer_simulator'):
    """
    Real quantum PCA implementation.
    
    Steps:
    1. Build density matrix classically (required preprocessing)
    2. Encode density matrix into quantum state
    3. Run QPE to extract eigenvalues
    4. Perform tomography to extract eigenvectors
    5. Return top-k components
    
    Args:
        X: Standardized data (N, D)
        k: Number of components
        backend: 'aer_simulator', 'ibmq_qasm_simulator', or real device
        
    Returns:
        U_quantum: (D, k) projection matrix from REAL quantum circuits
    """
```

---

### Phase 2: Real Quantum Fidelity (SWAP Test)

**File**: `quantum/real_fidelity.py`

**Approach**: Implement SWAP test for true quantum state fidelity

**Circuit**:
```
        ┌───┐     ┌───┐┌─┐
q_anc:  ┤ H ├──■──┤ H ├┤M├
        └───┘┌─┴─┐└───┘└╥┘
q_a[0]: ─────┤   ├──────╫─
             │ X │      ║
q_b[0]: ─────┤   ├──────╫─
             └───┘      ║
c:      ════════════════╩═
```

**Code Structure**:
```python
def quantum_fidelity_swap_test(state_a, state_b, shots=1024):
    """
    Compute fidelity using quantum SWAP test.
    
    Fidelity = 2 * P(measure |0⟩) - 1
    
    Args:
        state_a: First quantum state (normalized)
        state_b: Second quantum state (normalized)
        shots: Number of measurements
        
    Returns:
        fidelity: Real quantum fidelity from circuit execution
    """
```

---

### Phase 3: Variational Quantum Eigensolver (VQE) for PCA

**File**: `quantum/vqe_pca.py`

**Approach**: Use VQE to find eigenvectors of covariance/density matrix

**Why VQE**:
- Works on NISQ (Noisy Intermediate-Scale Quantum) devices
- More practical than HHL for near-term quantum computers
- Can handle noise and limited qubit count

---

### Phase 4: Quantum Amplitude Estimation for DTW

**File**: `quantum/qae_dtw.py`

**Approach**: Use QAE to estimate DTW distances with quadratic speedup

**Theory**:
- Classical DTW: O(N*M) for sequence comparison
- Quantum AE: O(√(N*M)) with amplitude encoding

---

## Directory Structure (New)

```
quantum/
├── __init__.py
├── classical_pca.py          # ✅ Already exists (classical SVD)
├── qpca.py                    # ⚠️  RENAME to simulated_qpca.py (honest naming)
├── real_qpca.py               # 🆕 Real quantum PCA with circuits
├── real_fidelity.py           # 🆕 Real SWAP test
├── vqe_pca.py                 # 🆕 VQE-based PCA
├── qae_dtw.py                 # 🆕 Quantum amplitude estimation DTW
└── utils/
    ├── state_preparation.py   # Amplitude encoding circuits
    ├── qpe.py                 # Quantum phase estimation
    └── backend_config.py      # IBM Quantum configuration
```

---

## Comparison Matrix

| Method | Type | Hardware | Speed | Accuracy |
|--------|------|----------|-------|----------|
| Classical PCA | Classical | CPU | Fast | 72% |
| Simulated Quantum PCA | Classical | CPU | Fast | 74% |
| **Real Quantum PCA** | Quantum | Simulator/IBM Q | Slow | TBD |
| **VQE PCA** | Quantum | NISQ | Medium | TBD |

---

## Implementation Priorities

### Must Have (Phase 1)
1. ✅ Rename `qpca.py` → `simulated_qpca.py` (honest naming)
2. ✅ Implement `real_qpca.py` with actual circuits
3. ✅ Implement `real_fidelity.py` with SWAP test
4. ✅ Add quantum backend configuration
5. ✅ Test on Qiskit Aer simulator

### Nice to Have (Phase 2)
1. VQE-based PCA for NISQ devices
2. Quantum Amplitude Estimation for DTW
3. Integration with IBM Quantum cloud
4. Noise models and error mitigation

### Future Work
1. Run on real IBM Quantum hardware
2. Benchmark quantum vs classical performance
3. Study quantum advantage with larger datasets
4. Implement quantum error correction

---

## Testing Strategy

```python
# Test 1: Verify quantum circuits execute
def test_quantum_circuit_execution():
    """Ensure circuits run without errors."""
    
# Test 2: Compare classical vs quantum results
def test_quantum_classical_consistency():
    """Quantum results should match classical (within noise)."""
    
# Test 3: Validate SWAP test
def test_swap_test_fidelity():
    """Known states should give known fidelities."""
    
# Test 4: End-to-end pipeline
def test_quantum_pipeline():
    """Full pipeline with real quantum components."""
```

---

## Ethics & Transparency

**Current Issue**: Calling classical simulations "quantum" is misleading

**Solution**:
1. ✅ Clearly label simulations as "simulated" or "quantum-inspired"
2. ✅ Distinguish real quantum circuits from classical code
3. ✅ Document what runs on quantum hardware vs classical
4. ✅ Be honest about quantum advantage (or lack thereof)

**Documentation Updates Needed**:
- README: Clarify what's actually quantum
- Papers/Reports: Distinguish simulation from real quantum
- Code comments: Label quantum circuits explicitly

---

## Next Steps

1. Install Qiskit: `pip install qiskit qiskit-aer`
2. Implement `real_qpca.py` with basic QPE circuit
3. Implement `real_fidelity.py` with SWAP test
4. Test on local simulator
5. Compare results with classical/simulated versions
6. (Optional) Run on IBM Quantum cloud

Let's build it! 🚀
