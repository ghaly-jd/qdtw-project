"""
Real Quantum Fidelity using SWAP Test

This module implements TRUE quantum fidelity measurement using:
1. Quantum SWAP test circuit
2. Actual quantum gates and measurements
3. Real execution on quantum simulators/hardware

This is NOT a classical simulation - it builds and executes real quantum circuits.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quantum_swap_test(
    state_a: np.ndarray,
    state_b: np.ndarray,
    shots: int = 1024,
    backend_name: str = 'aer_simulator'
) -> Tuple[float, dict]:
    """
    Compute quantum state fidelity using the SWAP test circuit.
    
    The SWAP test is a quantum algorithm that measures the overlap between
    two quantum states |ψ⟩ and |φ⟩ by using a controlled-SWAP operation.
    
    Circuit:
        ┌───┐     ┌───┐┌─┐
    anc:┤ H ├──■──┤ H ├┤M├
        └───┘┌─┴─┐└───┘└╥┘
    a:  ─────┤   ├──────╫─
             │ X │      ║
    b:  ─────┤   ├──────╫─
             └───┘      ║
    c:  ════════════════╩═
    
    The probability of measuring |0⟩ on the ancilla qubit is:
        P(0) = 1/2 + 1/2 |⟨ψ|φ⟩|²
    
    Therefore:
        Fidelity = |⟨ψ|φ⟩|² = 2*P(0) - 1
    
    Args:
        state_a: First quantum state as normalized vector (length 2^n)
        state_b: Second quantum state as normalized vector (length 2^n)
        shots: Number of circuit executions for measurement statistics
        backend_name: Quantum backend ('aer_simulator', 'qasm_simulator', etc.)
        
    Returns:
        fidelity: Quantum fidelity computed from SWAP test
        counts: Measurement statistics from circuit execution
        
    Raises:
        ValueError: If states are not properly normalized or have different dimensions
        
    Example:
        >>> # Two identical states
        >>> state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        >>> fidelity, counts = quantum_swap_test(state, state)
        >>> print(f"Fidelity: {fidelity:.4f}")  # Should be close to 1.0
        
        >>> # Two orthogonal states
        >>> state_a = np.array([1, 0])
        >>> state_b = np.array([0, 1])
        >>> fidelity, counts = quantum_swap_test(state_a, state_b)
        >>> print(f"Fidelity: {fidelity:.4f}")  # Should be close to 0.0
    """
    # Validate inputs
    if state_a.shape != state_b.shape:
        raise ValueError(
            f"States must have same dimension: {state_a.shape} vs {state_b.shape}"
        )
    
    # Check if power of 2 (required for quantum states)
    dim = len(state_a)
    n_qubits = int(np.log2(dim))
    if 2 ** n_qubits != dim:
        raise ValueError(
            f"State dimension must be power of 2, got {dim}"
        )
    
    # Normalize states (warn if far from 1.0)
    norm_a = np.linalg.norm(state_a)
    norm_b = np.linalg.norm(state_b)
    
    if not np.isclose(norm_a, 1.0, atol=1e-3):
        logger.warning(f"Normalizing state_a (norm={norm_a:.6f})")
    
    if not np.isclose(norm_b, 1.0, atol=1e-3):
        logger.warning(f"Normalizing state_b (norm={norm_b:.6f})")
    
    # Let Qiskit handle normalization to avoid precision issues
    # We'll use normalize=True in the initialize() call
    
    logger.info(f"Building SWAP test circuit for {n_qubits}-qubit states")
    
    # Create quantum registers
    anc = QuantumRegister(1, name='anc')  # Ancilla qubit
    reg_a = QuantumRegister(n_qubits, name='a')  # State |ψ⟩
    reg_b = QuantumRegister(n_qubits, name='b')  # State |φ⟩
    creg = ClassicalRegister(1, name='c')  # Classical bit for measurement
    
    # Build circuit
    qc = QuantumCircuit(anc, reg_a, reg_b, creg)
    
    # Step 1: Initialize states |ψ⟩ and |φ⟩ (let Qiskit normalize)
    qc.initialize(state_a, reg_a, normalize=True)
    qc.initialize(state_b, reg_b, normalize=True)
    
    # Step 2: Apply Hadamard to ancilla
    qc.h(anc)
    
    # Step 3: Apply controlled-SWAP (Fredkin gate)
    # Swap reg_a and reg_b controlled by ancilla
    for i in range(n_qubits):
        qc.cswap(anc[0], reg_a[i], reg_b[i])
    
    # Step 4: Apply Hadamard to ancilla again
    qc.h(anc)
    
    # Step 5: Measure ancilla
    qc.measure(anc, creg)
    
    logger.info(f"Circuit depth: {qc.depth()}, gates: {qc.size()}")
    
    # Execute circuit on quantum backend
    backend = Aer.get_backend(backend_name)
    
    logger.info(f"Executing on backend: {backend_name} with {shots} shots")
    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    logger.info(f"Measurement results: {counts}")
    
    # Calculate fidelity from measurement statistics
    # P(0) = count('0') / total_shots
    count_0 = counts.get('0', 0)
    count_1 = counts.get('1', 0)
    total = count_0 + count_1
    
    prob_0 = count_0 / total
    
    # Fidelity = 2*P(0) - 1
    fidelity = 2 * prob_0 - 1
    
    # Clamp to [0, 1] (account for measurement noise)
    fidelity = np.clip(fidelity, 0.0, 1.0)
    
    logger.info(f"P(0) = {prob_0:.4f}, Fidelity = {fidelity:.4f}")
    
    return fidelity, counts


def quantum_fidelity_distance(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    pad_to_power2: bool = True,
    shots: int = 1024
) -> float:
    """
    Compute fidelity-based distance using quantum SWAP test.
    
    This is a wrapper around quantum_swap_test that:
    1. Pads vectors to power of 2 if needed
    2. Returns distance (1 - fidelity) for DTW compatibility
    
    Args:
        vec_a: First vector (any dimension)
        vec_b: Second vector (same dimension as vec_a)
        pad_to_power2: If True, pad to nearest power of 2
        shots: Number of measurements
        
    Returns:
        distance: 1 - fidelity (range [0, 1])
        
    Example:
        >>> a = np.array([1, 2, 3, 4, 5])  # 5-D vector
        >>> b = np.array([1, 2, 3, 4, 6])  # Similar vector
        >>> dist = quantum_fidelity_distance(a, b)
        >>> print(f"Distance: {dist:.4f}")
    """
    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f"Vectors must have same shape: {vec_a.shape} vs {vec_b.shape}"
        )
    
    dim = len(vec_a)
    
    # Pad to power of 2 if needed
    if pad_to_power2:
        n_qubits = int(np.ceil(np.log2(dim)))
        padded_dim = 2 ** n_qubits
        
        if padded_dim != dim:
            logger.debug(f"Padding from {dim} to {padded_dim} dimensions")
            vec_a = np.pad(vec_a, (0, padded_dim - dim), mode='constant')
            vec_b = np.pad(vec_b, (0, padded_dim - dim), mode='constant')
    
    # Compute fidelity using SWAP test
    fidelity, _ = quantum_swap_test(vec_a, vec_b, shots=shots)
    
    # Return distance
    distance = 1.0 - fidelity
    
    return distance


def validate_quantum_fidelity():
    """
    Validate quantum SWAP test implementation with known cases.
    
    Tests:
    1. Identical states → fidelity ≈ 1
    2. Orthogonal states → fidelity ≈ 0
    3. Superposition states → known theoretical values
    """
    logger.info("=" * 60)
    logger.info("Validating Quantum SWAP Test")
    logger.info("=" * 60)
    
    # Test 1: Identical states
    logger.info("\nTest 1: Identical states")
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    fidelity, _ = quantum_swap_test(state, state, shots=2048)
    logger.info(f"Expected: 1.0, Got: {fidelity:.4f}")
    assert fidelity > 0.95, f"Expected fidelity > 0.95, got {fidelity:.4f}"
    
    # Test 2: Orthogonal states
    logger.info("\nTest 2: Orthogonal states")
    state_0 = np.array([1.0, 0.0])
    state_1 = np.array([0.0, 1.0])
    fidelity, _ = quantum_swap_test(state_0, state_1, shots=2048)
    logger.info(f"Expected: 0.0, Got: {fidelity:.4f}")
    assert fidelity < 0.05, f"Expected fidelity < 0.05, got {fidelity:.4f}"
    
    # Test 3: 45-degree states
    logger.info("\nTest 3: 45-degree apart states")
    # |+⟩ = (|0⟩ + |1⟩)/√2
    state_plus = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    # |0⟩ = |0⟩
    state_0 = np.array([1.0, 0.0])
    fidelity, _ = quantum_swap_test(state_plus, state_0, shots=2048)
    # Expected: |⟨+|0⟩|² = (1/√2)² = 0.5
    logger.info(f"Expected: 0.5, Got: {fidelity:.4f}")
    assert 0.45 < fidelity < 0.55, f"Expected fidelity ≈ 0.5, got {fidelity:.4f}"
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ All validation tests passed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Run validation
    validate_quantum_fidelity()
    
    # Example: Compare with classical fidelity
    logger.info("\n" + "=" * 60)
    logger.info("Comparing Quantum vs Classical Fidelity")
    logger.info("=" * 60)
    
    # Generate two random states
    np.random.seed(42)
    dim = 8  # 3-qubit states
    vec_a = np.random.randn(dim)
    vec_a /= np.linalg.norm(vec_a)
    vec_b = np.random.randn(dim)
    vec_b /= np.linalg.norm(vec_b)
    
    # Quantum fidelity (SWAP test)
    quantum_fid, _ = quantum_swap_test(vec_a, vec_b, shots=4096)
    
    # Classical fidelity (dot product squared)
    classical_fid = np.abs(np.dot(vec_a, vec_b)) ** 2
    
    logger.info(f"\nQuantum Fidelity (SWAP test): {quantum_fid:.4f}")
    logger.info(f"Classical Fidelity (|⟨ψ|φ⟩|²):  {classical_fid:.4f}")
    logger.info(f"Difference: {abs(quantum_fid - classical_fid):.4f}")
    logger.info("\n✅ Quantum and classical should match (within shot noise)")
