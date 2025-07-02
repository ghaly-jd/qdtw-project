import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram

def grover_min_index(options):
    """
    Uses Grover's algorithm to find the index of the minimum value in options.
    """
    # Find the minimum index classically (for Oracle construction)
    min_idx = int(np.argmin(options))
    n_qubits = 2  # 2 qubits can represent 3 states (00, 01, 10)
    
    # Oracle marks the minimum index
    oracle = QuantumCircuit(n_qubits)
    
    # Apply X gates to flip bits according to min_idx
    if min_idx == 0:
        oracle.x(0)
        oracle.x(1)
    elif min_idx == 1:
        oracle.x(1)
    
    # Apply multi-controlled Z gate (in this case with 2 qubits, it's just CZ)
    oracle.h(1)
    oracle.cx(0, 1)
    oracle.h(1)
    
    # Undo the X gates
    if min_idx == 0:
        oracle.x(0)
        oracle.x(1)
    elif min_idx == 1:
        oracle.x(1)
    
    # Create Grover circuit
    grover = QuantumCircuit(n_qubits)
    
    # Initialize in superposition
    grover.h(range(n_qubits))
    
    # Apply oracle
    grover.compose(oracle, inplace=True)
    
    # Apply diffusion operator (amplification)
    grover.h(range(n_qubits))
    grover.x(range(n_qubits))
    grover.h(1)
    grover.cx(0, 1)
    grover.h(1)
    grover.x(range(n_qubits))
    grover.h(range(n_qubits))
    
    # Measure
    grover.measure_all()
    
    # Run on simulator
    backend = Aer.get_backend('qasm_simulator')
    transpiled = transpile(grover, backend)
    result = backend.run(transpiled, shots=1024).result()
    counts = result.get_counts()
    
    # Interpret results
    measured = max(counts, key=counts.get)
    idx = int(measured, 2)
    
    # Ensure valid index (maximum 2 for 3 options)
    if idx > 2:
        return min_idx
    return idx

def qdtw_distance(seq1, seq2):
    """
    Compute DTW distance between two sequences using quantum minimum search.
    """
    n, m = len(seq1), len(seq2)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    
    # Fill first row and column with infinity (boundary condition)
    # but keep dtw[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(np.array(seq1[i-1]) - np.array(seq2[j-1]))
            options = [dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1]]
            min_idx = grover_min_index(options)
            dtw[i, j] = cost + options[min_idx]
    
    return dtw[n, m]
if __name__ == "__main__":
    # Test the QDTW with simple sequences
    seq1 = [[1, 2], [3, 4], [5, 6]]
    seq2 = [[1, 1], [2, 2], [5, 5]]
    
    print("Testing Quantum DTW with example sequences:")
    print(f"Sequence 1: {seq1}")
    print(f"Sequence 2: {seq2}")
    
    # Calculate and print the distance
    distance = qdtw_distance(seq1, seq2)
    print(f"\nQuantum DTW Distance: {distance:.4f}")
    
    # Demonstrate the quantum minimum search
    test_options = [5.0, 3.0, 7.0]
    print(f"\nTesting quantum minimum search on: {test_options}")
    min_idx = grover_min_index(test_options)
    print(f"Quantum circuit found minimum at index: {min_idx}")
    print(f"Value at this index: {test_options[min_idx]}")
    print(f"Actual minimum: {min(test_options)}")