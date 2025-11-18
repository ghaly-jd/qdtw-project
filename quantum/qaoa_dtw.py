"""
QAOA-based DTW Path Refinement

Uses Quantum Approximate Optimization Algorithm (QAOA) to refine DTW alignment paths
within local windows around the classical solution.

Strategy:
1. Compute classical DTW path as baseline
2. Extract L≈24 windows around the path with band width r=3-5
3. Formulate QUBO: variables x_{i,j} for cells, cost from quantum distances
4. Solve with QAOA (p=1-2 layers) on simulator
5. Decode bitstring to refined path, compare window costs

References:
- Farhi et al. "A Quantum Approximate Optimization Algorithm" (2014)
- QUBO formulation for constrained path problems
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
import warnings


def classical_dtw_path(dist_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Compute classical DTW alignment path.
    
    Parameters
    ----------
    dist_matrix : np.ndarray
        Distance matrix (n x m)
    
    Returns
    -------
    path : List[Tuple[int, int]]
        Alignment path as list of (i, j) coordinates
    """
    n, m = dist_matrix.shape
    
    # DTW cost matrix
    cost = np.full((n, m), np.inf)
    cost[0, 0] = dist_matrix[0, 0]
    
    # Forward pass
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            
            candidates = []
            if i > 0:
                candidates.append(cost[i-1, j])
            if j > 0:
                candidates.append(cost[i, j-1])
            if i > 0 and j > 0:
                candidates.append(cost[i-1, j-1])
            
            if candidates:
                cost[i, j] = dist_matrix[i, j] + min(candidates)
    
    # Backtrack to get path
    path = []
    i, j = n - 1, m - 1
    path.append((i, j))
    
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # Choose minimum predecessor
            candidates = [
                (cost[i-1, j], (i-1, j)),
                (cost[i, j-1], (i, j-1)),
                (cost[i-1, j-1], (i-1, j-1))
            ]
            _, (i, j) = min(candidates, key=lambda x: x[0])
        path.append((i, j))
    
    return list(reversed(path))


def extract_windows(path: List[Tuple[int, int]], 
                    dist_matrix: np.ndarray,
                    window_length: int = 24,
                    band_radius: int = 4) -> List[Dict]:
    """
    Extract local windows around the classical path.
    
    Parameters
    ----------
    path : List[Tuple[int, int]]
        Classical DTW path
    dist_matrix : np.ndarray
        Full distance matrix
    window_length : int
        Window length L (number of path steps)
    band_radius : int
        Band radius r for allowed deviations
    
    Returns
    -------
    windows : List[Dict]
        List of window dictionaries with metadata
    """
    windows = []
    path_len = len(path)
    
    # Sliding window over path
    step = window_length // 2  # 50% overlap
    for start_idx in range(0, path_len - window_length + 1, step):
        end_idx = start_idx + window_length
        window_path = path[start_idx:end_idx]
        
        # Get bounding box with band
        i_coords = [p[0] for p in window_path]
        j_coords = [p[1] for p in window_path]
        
        i_min = max(0, min(i_coords) - band_radius)
        i_max = min(dist_matrix.shape[0], max(i_coords) + band_radius + 1)
        j_min = max(0, min(j_coords) - band_radius)
        j_max = min(dist_matrix.shape[1], max(j_coords) + band_radius + 1)
        
        # Extract window distance matrix
        window_dist = dist_matrix[i_min:i_max, j_min:j_max]
        
        # Convert path to local coordinates
        local_path = [(i - i_min, j - j_min) for i, j in window_path]
        
        # Compute classical cost in this window
        classical_cost = sum(window_dist[i, j] for i, j in local_path)
        
        windows.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'global_bounds': (i_min, i_max, j_min, j_max),
            'dist_matrix': window_dist,
            'classical_path': local_path,
            'classical_cost': classical_cost,
            'shape': window_dist.shape
        })
    
    return windows


def path_to_qubo(dist_matrix: np.ndarray,
                 start: Tuple[int, int],
                 end: Tuple[int, int],
                 penalty_weight: float = 10.0,
                 max_qubits: int = 15) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Formulate DTW path as QUBO problem.
    
    Variables: x_{i,j} = 1 if cell (i,j) is in path, 0 otherwise
    Objective: minimize Σ c_{i,j} * x_{i,j}
    Constraints (soft): 
      - Path starts at start, ends at end
      - Monotonicity (only move right, up, or diagonal)
      - Connectivity (path must be connected)
    
    Parameters
    ----------
    dist_matrix : np.ndarray
        Distance matrix for window (n x m)
    start : Tuple[int, int]
        Start coordinates (typically (0, 0))
    end : Tuple[int, int]
        End coordinates (typically (n-1, m-1))
    penalty_weight : float
        Weight for constraint violations
    max_qubits : int
        Maximum number of qubits (subsamples if needed)
    
    Returns
    -------
    Q : np.ndarray
        QUBO matrix (quadratic terms)
    var_map : List[Tuple[int, int]]
        Mapping from variable index to (i, j) coordinates
    """
    n, m = dist_matrix.shape
    
    # Create variable mapping (only cells near diagonal to limit qubits)
    var_map = []
    var_index = {}
    
    # Limit to cells within a band around the diagonal
    max_band = max(2, int(np.sqrt(max_qubits)))  # Adaptive band width
    
    for i in range(n):
        for j in range(m):
            # Only include cells near diagonal: |i/n - j/m| < threshold
            if abs(i * m - j * n) <= max_band * max(n, m):
                idx = len(var_map)
                var_map.append((i, j))
                var_index[(i, j)] = idx
                
                if len(var_map) >= max_qubits:
                    break
        if len(var_map) >= max_qubits:
            break
    
    # Always include start and end
    if start not in var_index:
        var_index[start] = len(var_map)
        var_map.append(start)
    if end not in var_index:
        var_index[end] = len(var_map)
        var_map.append(end)
    
    n_vars = len(var_map)
    Q = np.zeros((n_vars, n_vars))
    
    # Linear terms: minimize path cost
    for idx, (i, j) in enumerate(var_map):
        Q[idx, idx] = dist_matrix[i, j]
    
    # Constraint 1: Start point must be selected
    start_idx = var_index.get(start)
    if start_idx is not None:
        Q[start_idx, start_idx] -= penalty_weight  # Reward for selecting start
    
    # Constraint 2: End point must be selected
    end_idx = var_index.get(end)
    if end_idx is not None:
        Q[end_idx, end_idx] -= penalty_weight  # Reward for selecting end
    
    # Constraint 3: Soft connectivity (penalize if neighbors not selected)
    # For each cell, if selected, encourage at least one neighbor
    for idx, (i, j) in enumerate(var_map):
        neighbors = []
        for di, dj in [(1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1)]:
            ni, nj = i + di, j + dj
            if (ni, nj) in var_index:
                neighbors.append(var_index[(ni, nj)])
        
        # Quadratic penalty: penalize isolated cells
        for neighbor_idx in neighbors:
            Q[idx, neighbor_idx] -= 0.5 * penalty_weight / max(1, len(neighbors))
    
    return Q, var_map


def qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert QUBO to Ising Hamiltonian.
    
    QUBO: x^T Q x, x ∈ {0, 1}^n
    Ising: Σ h_i z_i + Σ J_{ij} z_i z_j, z ∈ {-1, +1}^n
    
    Transform: x = (1 + z) / 2
    
    Parameters
    ----------
    Q : np.ndarray
        QUBO matrix
    
    Returns
    -------
    h : np.ndarray
        Linear terms (magnetic fields)
    J : np.ndarray
        Quadratic terms (couplings)
    offset : float
        Constant offset
    """
    n = Q.shape[0]
    
    # Symmetrize Q
    Q_sym = (Q + Q.T) / 2
    
    # Convert to Ising
    h = np.zeros(n)
    J = np.zeros((n, n))
    
    for i in range(n):
        h[i] = Q_sym[i, i] / 2 + np.sum(Q_sym[i, :]) / 4
        for j in range(i + 1, n):
            J[i, j] = Q_sym[i, j] / 4
    
    offset = np.sum(Q_sym) / 4
    
    return h, J, offset


def qaoa_circuit(h: np.ndarray, J: np.ndarray, p: int = 1) -> QuantumCircuit:
    """
    Build QAOA circuit for Ising Hamiltonian.
    
    Parameters
    ----------
    h : np.ndarray
        Linear terms (n,)
    J : np.ndarray
        Quadratic terms (n x n)
    p : int
        Number of QAOA layers
    
    Returns
    -------
    qc : QuantumCircuit
        Parametrized QAOA circuit
    """
    n = len(h)
    qc = QuantumCircuit(n)
    
    # Initial state: equal superposition
    qc.h(range(n))
    
    # QAOA layers
    params = []
    for layer in range(p):
        # Problem Hamiltonian: e^(-i γ H_P)
        gamma = Parameter(f'γ_{layer}')
        params.append(gamma)
        
        # Apply ZZ couplings (J terms)
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0:
                    qc.rzz(2 * gamma * J[i, j], i, j)
        
        # Apply Z rotations (h terms)
        for i in range(n):
            if h[i] != 0:
                qc.rz(2 * gamma * h[i], i)
        
        # Mixer Hamiltonian: e^(-i β H_M) = Π X_i
        beta = Parameter(f'β_{layer}')
        params.append(beta)
        
        for i in range(n):
            qc.rx(2 * beta, i)
    
    # Measure all qubits
    qc.measure_all()
    
    return qc


def qaoa_expectation(params: np.ndarray, 
                     qc: QuantumCircuit,
                     h: np.ndarray,
                     J: np.ndarray,
                     offset: float,
                     shots: int = 1024) -> float:
    """
    Compute QAOA expectation value via simulation.
    
    Parameters
    ----------
    params : np.ndarray
        QAOA parameters [γ_0, β_0, γ_1, β_1, ...]
    qc : QuantumCircuit
        Parametrized QAOA circuit
    h : np.ndarray
        Ising linear terms
    J : np.ndarray
        Ising quadratic terms
    offset : float
        Constant offset
    shots : int
        Number of measurement shots
    
    Returns
    -------
    expectation : float
        Expected energy
    """
    # Bind parameters
    param_dict = {qc.parameters[i]: params[i] for i in range(len(params))}
    bound_qc = qc.assign_parameters(param_dict)
    
    # Simulate with qasm_simulator (more memory efficient)
    simulator = AerSimulator(method='automatic')
    result = simulator.run(bound_qc, shots=shots, seed_simulator=42).result()
    counts = result.get_counts()
    
    # Compute expectation
    n = len(h)
    expectation = 0.0
    
    for bitstring, count in counts.items():
        # Convert bitstring to Ising configuration (reverse order for qiskit)
        z = np.array([1 if b == '0' else -1 for b in reversed(bitstring)])
        
        # Compute energy
        energy = offset
        energy += np.dot(h, z)
        for i in range(n):
            for j in range(i + 1, n):
                energy += J[i, j] * z[i] * z[j]
        
        expectation += energy * count / shots
    
    return expectation


def optimize_qaoa(qc: QuantumCircuit,
                  h: np.ndarray,
                  J: np.ndarray,
                  offset: float,
                  p: int = 1,
                  shots: int = 1024,
                  maxiter: int = 100) -> Tuple[np.ndarray, float]:
    """
    Optimize QAOA parameters using classical optimizer.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Parametrized QAOA circuit
    h : np.ndarray
        Ising linear terms
    J : np.ndarray
        Ising quadratic terms
    offset : float
        Constant offset
    p : int
        Number of QAOA layers
    shots : int
        Number of measurement shots
    maxiter : int
        Maximum optimizer iterations
    
    Returns
    -------
    optimal_params : np.ndarray
        Optimized parameters
    optimal_energy : float
        Minimum energy found
    """
    # Initial parameters (random)
    np.random.seed(42)
    initial_params = np.random.uniform(0, 2*np.pi, 2*p)
    
    # Optimize
    result = minimize(
        lambda params: qaoa_expectation(params, qc, h, J, offset, shots),
        initial_params,
        method='COBYLA',
        options={'maxiter': maxiter}
    )
    
    return result.x, result.fun


def decode_path(bitstring: str, var_map: List[Tuple[int, int]], 
                start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Decode QAOA bitstring to valid DTW path.
    
    Parameters
    ----------
    bitstring : str
        Measurement outcome (Qiskit format: reversed)
    var_map : List[Tuple[int, int]]
        Mapping from variable index to coordinates
    start : Tuple[int, int]
        Start coordinates
    end : Tuple[int, int]
        End coordinates
    
    Returns
    -------
    path : List[Tuple[int, int]]
        Decoded path (may be invalid)
    """
    # Convert bitstring to selected cells
    # Qiskit bitstrings are reversed
    selected = []
    for idx, bit in enumerate(reversed(bitstring)):
        if bit == '1':
            selected.append(var_map[idx])
    
    if not selected:
        # Fallback: diagonal path
        return [start, end]
    
    # Sort by row, then column (approximate path)
    selected.sort()
    
    # Ensure start and end are included
    if start not in selected:
        selected.insert(0, start)
    if end not in selected:
        selected.append(end)
    
    return selected


def qaoa_refine_window(window: Dict, 
                       p: int = 1,
                       shots: int = 1024,
                       maxiter: int = 50,
                       penalty_weight: float = 10.0,
                       max_qubits: int = 15) -> Dict:
    """
    Refine DTW path in a single window using QAOA.
    
    Parameters
    ----------
    window : Dict
        Window metadata from extract_windows
    p : int
        QAOA depth
    shots : int
        Measurement shots
    maxiter : int
        Optimizer iterations
    penalty_weight : float
        QUBO constraint weight
    max_qubits : int
        Maximum number of qubits
    
    Returns
    -------
    result : Dict
        Refinement results
    """
    dist_matrix = window['dist_matrix']
    n, m = dist_matrix.shape
    start = (0, 0)
    end = (n - 1, m - 1)
    
    # Formulate QUBO with qubit limit
    Q, var_map = path_to_qubo(dist_matrix, start, end, penalty_weight, max_qubits)
    
    # Check problem size
    n_vars = len(var_map)
    if n_vars > 20:
        warnings.warn(f"Window has {n_vars} qubits, may be slow. Consider smaller windows.")
    
    # Convert to Ising
    h, J, offset = qubo_to_ising(Q)
    
    # Build QAOA circuit
    qc = qaoa_circuit(h, J, p)
    
    # Optimize
    optimal_params, optimal_energy = optimize_qaoa(qc, h, J, offset, p, shots, maxiter)
    
    # Get best bitstring
    param_dict = {qc.parameters[i]: optimal_params[i] for i in range(len(optimal_params))}
    bound_qc = qc.assign_parameters(param_dict)
    
    simulator = AerSimulator(method='automatic')
    result = simulator.run(bound_qc, shots=shots, seed_simulator=42).result()
    counts = result.get_counts()
    
    best_bitstring = max(counts, key=counts.get)
    
    # Decode path
    qaoa_path = decode_path(best_bitstring, var_map, start, end)
    
    # Compute QAOA cost
    qaoa_cost = sum(dist_matrix[i, j] for i, j in qaoa_path if 0 <= i < n and 0 <= j < m)
    
    # Compare to classical
    classical_cost = window['classical_cost']
    improvement = classical_cost - qaoa_cost
    improvement_pct = 100 * improvement / classical_cost if classical_cost > 0 else 0
    
    return {
        'n_qubits': n_vars,
        'circuit_depth': qc.depth(),
        'shots': shots,
        'optimal_energy': optimal_energy,
        'classical_cost': classical_cost,
        'qaoa_cost': qaoa_cost,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'qaoa_path': qaoa_path,
        'best_bitstring': best_bitstring,
        'counts': counts
    }


def qaoa_dtw_pipeline(seq1: np.ndarray,
                      seq2: np.ndarray,
                      window_length: int = 24,
                      band_radius: int = 4,
                      p: int = 1,
                      shots: int = 1024,
                      maxiter: int = 50,
                      max_qubits: int = 15,
                      verbose: bool = True) -> Dict:
    """
    Full QAOA-DTW pipeline: classical baseline + QAOA refinement.
    
    Parameters
    ----------
    seq1 : np.ndarray
        First sequence (n, d)
    seq2 : np.ndarray
        Second sequence (m, d)
    window_length : int
        Window length L
    band_radius : int
        Band radius r
    p : int
        QAOA depth
    shots : int
        Measurement shots
    maxiter : int
        Optimizer iterations per window
    max_qubits : int
        Maximum qubits per window
    verbose : bool
        Print progress
    
    Returns
    -------
    results : Dict
        Aggregated results across all windows
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"QAOA-DTW Pipeline")
        print(f"{'='*80}")
        print(f"Sequences: {seq1.shape} × {seq2.shape}")
        print(f"Window: L={window_length}, r={band_radius}")
        print(f"QAOA: p={p}, shots={shots}, max_qubits={max_qubits}")
    
    # Compute distance matrix
    dist_matrix = np.linalg.norm(seq1[:, None, :] - seq2[None, :, :], axis=2)
    
    # Classical DTW path
    if verbose:
        print(f"\n{'─'*80}")
        print(f"Computing classical DTW path...")
    classical_path = classical_dtw_path(dist_matrix)
    classical_total_cost = sum(dist_matrix[i, j] for i, j in classical_path)
    
    if verbose:
        print(f"Classical path length: {len(classical_path)}")
        print(f"Classical total cost: {classical_total_cost:.4f}")
    
    # Extract windows
    if verbose:
        print(f"\n{'─'*80}")
        print(f"Extracting windows...")
    windows = extract_windows(classical_path, dist_matrix, window_length, band_radius)
    
    if verbose:
        print(f"Number of windows: {len(windows)}")
    
    # Refine each window with QAOA
    if verbose:
        print(f"\n{'─'*80}")
        print(f"Running QAOA refinement...")
    
    window_results = []
    n_improved = 0
    n_tied = 0
    n_worse = 0
    
    for i, window in enumerate(windows):
        if verbose:
            print(f"\nWindow {i+1}/{len(windows)}: shape {window['shape']}")
        
        result = qaoa_refine_window(window, p, shots, maxiter, max_qubits=max_qubits)
        window_results.append(result)
        
        if result['improvement'] > 1e-6:
            n_improved += 1
            status = "IMPROVED"
        elif result['improvement'] < -1e-6:
            n_worse += 1
            status = "WORSE"
        else:
            n_tied += 1
            status = "TIED"
        
        if verbose:
            print(f"  Qubits: {result['n_qubits']}, Depth: {result['circuit_depth']}")
            print(f"  Classical: {result['classical_cost']:.4f}")
            print(f"  QAOA: {result['qaoa_cost']:.4f}")
            print(f"  Improvement: {result['improvement_pct']:.2f}% [{status}]")
    
    # Summary statistics
    pct_improved = 100 * n_improved / len(windows) if windows else 0
    pct_tied = 100 * n_tied / len(windows) if windows else 0
    pct_worse = 100 * n_worse / len(windows) if windows else 0
    
    avg_qubits = np.mean([r['n_qubits'] for r in window_results]) if window_results else 0
    avg_depth = np.mean([r['circuit_depth'] for r in window_results]) if window_results else 0
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total windows: {len(windows)}")
        print(f"  Improved: {n_improved} ({pct_improved:.1f}%)")
        print(f"  Tied: {n_tied} ({pct_tied:.1f}%)")
        print(f"  Worse: {n_worse} ({pct_worse:.1f}%)")
        print(f"\nAverage qubits: {avg_qubits:.1f}")
        print(f"Average depth: {avg_depth:.1f}")
        print(f"Shots per window: {shots}")
        print(f"{'='*80}\n")
    
    return {
        'classical_path': classical_path,
        'classical_cost': classical_total_cost,
        'n_windows': len(windows),
        'n_improved': n_improved,
        'n_tied': n_tied,
        'n_worse': n_worse,
        'pct_improved': pct_improved,
        'pct_tied': pct_tied,
        'pct_worse': pct_worse,
        'avg_qubits': avg_qubits,
        'avg_depth': avg_depth,
        'shots': shots,
        'window_results': window_results,
        'config': {
            'window_length': window_length,
            'band_radius': band_radius,
            'p': p,
            'shots': shots,
            'maxiter': maxiter
        }
    }
