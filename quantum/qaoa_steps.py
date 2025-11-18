"""
QAOA-DTW with Step-Based Encoding

Encodes DTW path as sequence of moves (Right, Down, Diagonal) instead of cells.
This dramatically reduces qubit count from O(n*m) to O(n+m).

Key Ideas:
- Path of length L' encoded as L' steps
- Each step is one-hot encoded: {R, D, Diag} → 2 qubits/step
- Total qubits: 2*L' (e.g., L'=12 → 24 qubits, not 210!)
- Costs precomputed per move at each position
- Warm-start from classical path
- Constraint-preserving mixer (optional) or penalty terms

References:
- Hadfield et al. "Quantum Alternating Operator Ansatz" (2019)
- Warm-start QAOA techniques
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
import warnings


# Move encoding
MOVE_RIGHT = 0   # (i, j+1)
MOVE_DOWN = 1    # (i+1, j)
MOVE_DIAG = 2    # (i+1, j+1)
MOVE_NAMES = {0: 'R', 1: 'D', 2: 'Diag'}


def classical_dtw_path_in_window(dist_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Compute classical DTW path within a window.
    
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
    
    # Backtrack
    path = []
    i, j = n - 1, m - 1
    path.append((i, j))
    
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [
                (cost[i-1, j], (i-1, j)),
                (cost[i, j-1], (i, j-1)),
                (cost[i-1, j-1], (i-1, j-1))
            ]
            _, (i, j) = min(candidates, key=lambda x: x[0])
        path.append((i, j))
    
    return list(reversed(path))


def path_to_moves(path: List[Tuple[int, int]]) -> List[int]:
    """
    Convert path coordinates to sequence of moves.
    
    Parameters
    ----------
    path : List[Tuple[int, int]]
        Path as (i, j) coordinates
    
    Returns
    -------
    moves : List[int]
        Sequence of moves {0: R, 1: D, 2: Diag}
    """
    moves = []
    for k in range(len(path) - 1):
        i1, j1 = path[k]
        i2, j2 = path[k + 1]
        
        di = i2 - i1
        dj = j2 - j1
        
        if di == 0 and dj == 1:
            moves.append(MOVE_RIGHT)
        elif di == 1 and dj == 0:
            moves.append(MOVE_DOWN)
        elif di == 1 and dj == 1:
            moves.append(MOVE_DIAG)
        else:
            raise ValueError(f"Invalid move from ({i1},{j1}) to ({i2},{j2})")
    
    return moves


def moves_to_path(moves: List[int], start: Tuple[int, int] = (0, 0)) -> List[Tuple[int, int]]:
    """
    Convert sequence of moves to path coordinates.
    
    Parameters
    ----------
    moves : List[int]
        Sequence of moves
    start : Tuple[int, int]
        Starting coordinates
    
    Returns
    -------
    path : List[Tuple[int, int]]
        Path coordinates
    """
    path = [start]
    i, j = start
    
    for move in moves:
        if move == MOVE_RIGHT:
            j += 1
        elif move == MOVE_DOWN:
            i += 1
        elif move == MOVE_DIAG:
            i += 1
            j += 1
        else:
            raise ValueError(f"Invalid move: {move}")
        path.append((i, j))
    
    return path


def precompute_move_costs(dist_matrix: np.ndarray) -> np.ndarray:
    """
    Precompute costs for each move type at each position.
    
    Parameters
    ----------
    dist_matrix : np.ndarray
        Distance matrix (n x m)
    
    Returns
    -------
    move_costs : np.ndarray
        Shape (n, m, 3) where [i, j, move_type] = cost
    """
    n, m = dist_matrix.shape
    move_costs = np.full((n, m, 3), np.inf)
    
    for i in range(n):
        for j in range(m):
            # Right: (i, j) → (i, j+1)
            if j + 1 < m:
                move_costs[i, j, MOVE_RIGHT] = dist_matrix[i, j + 1]
            
            # Down: (i, j) → (i+1, j)
            if i + 1 < n:
                move_costs[i, j, MOVE_DOWN] = dist_matrix[i + 1, j]
            
            # Diagonal: (i, j) → (i+1, j+1)
            if i + 1 < n and j + 1 < m:
                move_costs[i, j, MOVE_DIAG] = dist_matrix[i + 1, j + 1]
    
    return move_costs


def moves_to_ising(moves_classical: List[int],
                   move_costs: np.ndarray,
                   path_length: int,
                   target_end: Tuple[int, int],
                   penalty_weight: float = 100.0) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """
    Convert move sequence problem to Ising Hamiltonian.
    
    Encoding: Each step k has 2 qubits (one-hot for 3 moves):
    - 00: Right
    - 01: Down  
    - 10: Diagonal
    - 11: Invalid (penalized)
    
    Parameters
    ----------
    moves_classical : List[int]
        Classical move sequence (for warm-start)
    move_costs : np.ndarray
        Precomputed costs [i, j, move_type]
    path_length : int
        Number of steps L'
    target_end : Tuple[int, int]
        Target endpoint (Δi, Δj)
    penalty_weight : float
        Weight for constraint violations
    
    Returns
    -------
    h : np.ndarray
        Linear Ising terms
    J : np.ndarray
        Quadratic Ising terms
    offset : float
        Constant offset
    metadata : Dict
        Encoding metadata
    """
    n_steps = path_length
    n_qubits = 2 * n_steps  # 2 qubits per step
    
    # Ising terms
    h = np.zeros(n_qubits)
    J = np.zeros((n_qubits, n_qubits))
    
    # Encoding: step k uses qubits [2*k, 2*k+1]
    # Move type from (z_{2k}, z_{2k+1}):
    #   z = (+1, +1) → bits = (0, 0) → RIGHT
    #   z = (+1, -1) → bits = (0, 1) → DOWN
    #   z = (-1, +1) → bits = (1, 0) → DIAG
    #   z = (-1, -1) → bits = (1, 1) → INVALID
    
    # We'll compute costs for each configuration at each step position
    # Cost for step k depends on current position (i, j) which depends on previous moves
    # Approximate: use average cost weighted by classical path
    
    # Simpler approach: penalize deviation from classical path
    # For each step k, if classical move is m_k, bias toward that encoding
    
    for k in range(n_steps):
        q0, q1 = 2 * k, 2 * k + 1
        
        if k < len(moves_classical):
            classical_move = moves_classical[k]
            
            # Encourage classical move by biasing h terms
            # Classical encoding: R=(+1,+1), D=(+1,-1), Diag=(-1,+1)
            if classical_move == MOVE_RIGHT:  # (+1, +1)
                h[q0] -= penalty_weight * 0.1  # Bias toward +1
                h[q1] -= penalty_weight * 0.1
            elif classical_move == MOVE_DOWN:  # (+1, -1)
                h[q0] -= penalty_weight * 0.1
                h[q1] += penalty_weight * 0.1  # Bias toward -1
            elif classical_move == MOVE_DIAG:  # (-1, +1)
                h[q0] += penalty_weight * 0.1
                h[q1] -= penalty_weight * 0.1
        
        # Penalize invalid encoding (both qubits -1)
        # z_0 = -1 AND z_1 = -1 → (1 - z_0)(1 - z_1)/4
        # Expand: 1/4 - z_0/4 - z_1/4 + z_0*z_1/4
        h[q0] += penalty_weight * 0.25
        h[q1] += penalty_weight * 0.25
        J[q0, q1] -= penalty_weight * 0.25  # Negative coupling discourages (-1,-1)
    
    # Endpoint constraint: penalize if final position != target
    # This is complex as final position depends on move sequence
    # Approximate: penalize if total counts don't match expected
    target_i, target_j = target_end
    
    # Expected moves (rough estimate):
    # - Number of R moves ≈ target_j
    # - Number of D moves ≈ target_i
    # - Can use Diag to reduce both
    
    # For simplicity, add global penalty terms (handled in expectation evaluation)
    
    offset = 0.0
    
    metadata = {
        'n_steps': n_steps,
        'n_qubits': n_qubits,
        'classical_moves': moves_classical,
        'target_end': target_end
    }
    
    return h, J, offset, metadata


def decode_moves_from_bitstring(bitstring: str, n_steps: int) -> List[int]:
    """
    Decode move sequence from measurement bitstring.
    
    Encoding (for each step k with qubits [2k, 2k+1]):
    - 00 → Right
    - 01 → Down
    - 10 → Diagonal
    - 11 → Invalid (map to Right as fallback)
    
    Parameters
    ----------
    bitstring : str
        Measurement outcome (Qiskit format: reversed)
    n_steps : int
        Number of steps
    
    Returns
    -------
    moves : List[int]
        Decoded move sequence
    """
    # Qiskit bitstrings are reversed
    bits = bitstring[::-1]
    
    moves = []
    for k in range(n_steps):
        b0 = bits[2 * k] if 2 * k < len(bits) else '0'
        b1 = bits[2 * k + 1] if 2 * k + 1 < len(bits) else '0'
        
        encoding = b0 + b1
        if encoding == '00':
            moves.append(MOVE_RIGHT)
        elif encoding == '01':
            moves.append(MOVE_DOWN)
        elif encoding == '10':
            moves.append(MOVE_DIAG)
        else:  # '11' - invalid
            moves.append(MOVE_RIGHT)  # Fallback
    
    return moves


def qaoa_step_circuit(h: np.ndarray, J: np.ndarray, p: int = 2,
                     warm_start_moves: Optional[List[int]] = None) -> QuantumCircuit:
    """
    Build QAOA circuit for step-based encoding with warm-start.
    
    Parameters
    ----------
    h : np.ndarray
        Linear Ising terms
    J : np.ndarray
        Quadratic Ising terms
    p : int
        QAOA depth
    warm_start_moves : Optional[List[int]]
        Classical move sequence for warm-start
    
    Returns
    -------
    qc : QuantumCircuit
        Parametrized QAOA circuit
    """
    n_qubits = len(h)
    qc = QuantumCircuit(n_qubits)
    
    # Warm-start: Initialize near classical solution
    if warm_start_moves is not None:
        n_steps = len(warm_start_moves)
        for k, move in enumerate(warm_start_moves):
            q0, q1 = 2 * k, 2 * k + 1
            
            # Initialize qubits to encode classical move
            if move == MOVE_RIGHT:  # 00
                pass  # Already |0⟩
            elif move == MOVE_DOWN:  # 01
                qc.x(q1)
            elif move == MOVE_DIAG:  # 10
                qc.x(q0)
        
        # Apply small angle rotations for exploration
        warm_angle = Parameter('θ_warm')
        for i in range(n_qubits):
            qc.ry(warm_angle, i)
    else:
        # Standard initialization: equal superposition
        qc.h(range(n_qubits))
    
    # QAOA layers
    for layer in range(p):
        # Problem Hamiltonian: e^(-i γ H_P)
        gamma = Parameter(f'γ_{layer}')
        
        # Apply ZZ couplings
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if J[i, j] != 0:
                    qc.rzz(2 * gamma * J[i, j], i, j)
        
        # Apply Z rotations
        for i in range(n_qubits):
            if h[i] != 0:
                qc.rz(2 * gamma * h[i], i)
        
        # Mixer Hamiltonian: e^(-i β H_M)
        # Custom mixer: swap moves while preserving some structure
        beta = Parameter(f'β_{layer}')
        
        # Apply RX on all qubits (standard X-mixer)
        for i in range(n_qubits):
            qc.rx(2 * beta, i)
        
        # Optional: Add SWAP gates between step-pairs for exploration
        # This allows exploring different move orderings
        if layer % 2 == 0:  # Every other layer
            for k in range(0, n_qubits - 2, 4):  # Swap pairs of steps
                if k + 3 < n_qubits:
                    # Swap step k with step k+1
                    qc.swap(k, k + 2)
                    qc.swap(k + 1, k + 3)
    
    qc.measure_all()
    return qc


def evaluate_move_cost(moves: List[int], move_costs: np.ndarray,
                      target_end: Tuple[int, int], 
                      penalty_weight: float = 100.0) -> float:
    """
    Evaluate cost of a move sequence.
    
    Parameters
    ----------
    moves : List[int]
        Move sequence
    move_costs : np.ndarray
        Precomputed costs [i, j, move_type]
    target_end : Tuple[int, int]
        Target endpoint
    penalty_weight : float
        Penalty for endpoint mismatch
    
    Returns
    -------
    cost : float
        Total cost (path cost + penalties)
    """
    # Simulate path
    path = moves_to_path(moves, start=(0, 0))
    
    # Compute path cost
    path_cost = 0.0
    for k, (i, j) in enumerate(path[:-1]):
        move = moves[k]
        if i < move_costs.shape[0] and j < move_costs.shape[1]:
            cost = move_costs[i, j, move]
            if not np.isinf(cost):
                path_cost += cost
            else:
                path_cost += penalty_weight  # Out of bounds
        else:
            path_cost += penalty_weight
    
    # Endpoint penalty
    final_i, final_j = path[-1]
    target_i, target_j = target_end
    endpoint_error = abs(final_i - target_i) + abs(final_j - target_j)
    endpoint_penalty = penalty_weight * endpoint_error
    
    return path_cost + endpoint_penalty


def qaoa_step_expectation(params: np.ndarray,
                          qc: QuantumCircuit,
                          move_costs: np.ndarray,
                          n_steps: int,
                          target_end: Tuple[int, int],
                          penalty_weight: float,
                          shots: int = 1024) -> float:
    """
    Compute QAOA expectation for step encoding.
    
    Parameters
    ----------
    params : np.ndarray
        QAOA parameters
    qc : QuantumCircuit
        Parametrized circuit
    move_costs : np.ndarray
        Precomputed move costs
    n_steps : int
        Number of steps
    target_end : Tuple[int, int]
        Target endpoint
    penalty_weight : float
        Constraint penalty
    shots : int
        Measurement shots
    
    Returns
    -------
    expectation : float
        Expected cost
    """
    # Bind parameters
    param_dict = {qc.parameters[i]: params[i] for i in range(len(params))}
    bound_qc = qc.assign_parameters(param_dict)
    
    # Simulate
    simulator = AerSimulator(method='automatic')
    result = simulator.run(bound_qc, shots=shots, seed_simulator=42).result()
    counts = result.get_counts()
    
    # Compute expectation over samples
    total_cost = 0.0
    for bitstring, count in counts.items():
        moves = decode_moves_from_bitstring(bitstring, n_steps)
        cost = evaluate_move_cost(moves, move_costs, target_end, penalty_weight)
        total_cost += cost * count / shots
    
    return total_cost


def optimize_qaoa_steps(qc: QuantumCircuit,
                       move_costs: np.ndarray,
                       n_steps: int,
                       target_end: Tuple[int, int],
                       penalty_weight: float,
                       p: int = 2,
                       shots: int = 2048,
                       maxiter: int = 100) -> Tuple[np.ndarray, float]:
    """
    Optimize QAOA parameters for step encoding.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Parametrized circuit
    move_costs : np.ndarray
        Precomputed move costs
    n_steps : int
        Number of steps
    target_end : Tuple[int, int]
        Target endpoint
    penalty_weight : float
        Constraint penalty
    p : int
        QAOA depth
    shots : int
        Measurement shots
    maxiter : int
        Optimizer iterations
    
    Returns
    -------
    optimal_params : np.ndarray
        Optimized parameters
    optimal_cost : float
        Minimum cost
    """
    # Initial parameters
    n_params = len(qc.parameters)
    np.random.seed(42)
    
    # Warm-start aware initialization
    if 'θ_warm' in [p.name for p in qc.parameters]:
        # Small warm-start angle
        initial_params = np.random.uniform(0, 0.5, n_params)
        initial_params[0] = 0.1  # θ_warm
    else:
        initial_params = np.random.uniform(0, np.pi, n_params)
    
    # Optimize
    result = minimize(
        lambda params: qaoa_step_expectation(
            params, qc, move_costs, n_steps, target_end, penalty_weight, shots
        ),
        initial_params,
        method='COBYLA',
        options={'maxiter': maxiter}
    )
    
    return result.x, result.fun


def qaoa_refine_window_steps(dist_matrix: np.ndarray,
                             p: int = 2,
                             shots: int = 2048,
                             maxiter: int = 100,
                             penalty_weight: float = 100.0,
                             verbose: bool = False) -> Dict:
    """
    Refine DTW path in window using step-based QAOA.
    
    Parameters
    ----------
    dist_matrix : np.ndarray
        Distance matrix for window
    p : int
        QAOA depth
    shots : int
        Measurement shots
    maxiter : int
        Optimizer iterations
    penalty_weight : float
        Constraint penalty
    verbose : bool
        Print progress
    
    Returns
    -------
    result : Dict
        Refinement results
    """
    n, m = dist_matrix.shape
    
    # Classical baseline
    classical_path = classical_dtw_path_in_window(dist_matrix)
    classical_moves = path_to_moves(classical_path)
    classical_cost = sum(dist_matrix[i, j] for i, j in classical_path)
    
    # Step encoding
    n_steps = len(classical_moves)
    target_end = (n - 1, m - 1)
    
    if verbose:
        print(f"  Window: {n}×{m}, Path length: {n_steps}, Qubits: {2*n_steps}")
    
    # Precompute move costs
    move_costs = precompute_move_costs(dist_matrix)
    
    # Build Ising Hamiltonian
    h, J, offset, metadata = moves_to_ising(
        classical_moves, move_costs, n_steps, target_end, penalty_weight
    )
    
    # Build QAOA circuit with warm-start
    qc = qaoa_step_circuit(h, J, p, warm_start_moves=classical_moves)
    
    # Optimize
    optimal_params, optimal_cost = optimize_qaoa_steps(
        qc, move_costs, n_steps, target_end, penalty_weight, p, shots, maxiter
    )
    
    # Get best solution
    param_dict = {qc.parameters[i]: optimal_params[i] for i in range(len(optimal_params))}
    bound_qc = qc.assign_parameters(param_dict)
    
    simulator = AerSimulator(method='automatic')
    result = simulator.run(bound_qc, shots=shots, seed_simulator=42).result()
    counts = result.get_counts()
    
    # Decode best bitstring
    best_bitstring = max(counts, key=counts.get)
    qaoa_moves = decode_moves_from_bitstring(best_bitstring, n_steps)
    qaoa_path = moves_to_path(qaoa_moves, start=(0, 0))
    
    # Compute QAOA cost (without penalties for fair comparison)
    qaoa_cost_raw = 0.0
    for k, (i, j) in enumerate(qaoa_path[:-1]):
        if k < len(qaoa_moves):
            move = qaoa_moves[k]
            if i < n and j < m:
                cost = move_costs[i, j, move]
                if not np.isinf(cost):
                    qaoa_cost_raw += cost
    
    # Compare
    improvement = classical_cost - qaoa_cost_raw
    improvement_pct = 100 * improvement / classical_cost if classical_cost > 0 else 0
    
    # Check endpoint match
    qaoa_end = qaoa_path[-1]
    endpoint_match = (qaoa_end == target_end)
    
    if verbose:
        print(f"  Classical: {classical_cost:.4f}")
        print(f"  QAOA: {qaoa_cost_raw:.4f}")
        print(f"  Improvement: {improvement_pct:.2f}%")
        print(f"  Endpoint: {qaoa_end} (target: {target_end}, match: {endpoint_match})")
    
    return {
        'n_qubits': 2 * n_steps,
        'circuit_depth': qc.depth(),
        'shots': shots,
        'classical_cost': classical_cost,
        'qaoa_cost': qaoa_cost_raw,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'classical_path': classical_path,
        'qaoa_path': qaoa_path,
        'endpoint_match': endpoint_match,
        'best_bitstring': best_bitstring,
        'counts': counts
    }
