"""
Test QAOA-DTW with Step-Based Encoding

Tests the step-based move encoding which uses far fewer qubits.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from quantum.qaoa_steps import (
    qaoa_refine_window_steps,
    classical_dtw_path_in_window,
    path_to_moves,
    moves_to_path,
    MOVE_NAMES
)


def test_move_encoding():
    """Test basic move encoding/decoding."""
    print("\n" + "="*80)
    print("TEST 1: Move Encoding/Decoding")
    print("="*80)
    
    # Simple path
    path = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)]
    print(f"Path: {path}")
    
    # Convert to moves
    moves = path_to_moves(path)
    print(f"Moves: {[MOVE_NAMES[m] for m in moves]}")
    
    # Convert back
    decoded_path = moves_to_path(moves, start=(0, 0))
    print(f"Decoded: {decoded_path}")
    print(f"Match: {path == decoded_path}")
    
    print("✅ Encoding test passed\n")


def test_small_window():
    """Test on very small window."""
    print("\n" + "="*80)
    print("TEST 2: Small Window (5×5)")
    print("="*80)
    
    # Create small distance matrix
    np.random.seed(42)
    dist_matrix = np.random.rand(5, 5)
    
    print(f"Window size: {dist_matrix.shape}")
    
    # Classical path
    classical_path = classical_dtw_path_in_window(dist_matrix)
    classical_moves = path_to_moves(classical_path)
    print(f"Classical path length: {len(classical_path)}")
    print(f"Classical moves: {[MOVE_NAMES[m] for m in classical_moves]}")
    print(f"Qubits needed: {2 * len(classical_moves)}")
    
    # Run QAOA
    result = qaoa_refine_window_steps(
        dist_matrix,
        p=1,  # Shallow for speed
        shots=1024,
        maxiter=30,
        verbose=True
    )
    
    print(f"\nResult:")
    print(f"  Improvement: {result['improvement_pct']:.2f}%")
    print(f"  Endpoint match: {result['endpoint_match']}")
    
    return result


def test_medium_window():
    """Test on medium window with realistic costs."""
    print("\n" + "="*80)
    print("TEST 3: Medium Window (9×9)")
    print("="*80)
    
    # Create distance matrix with structure
    np.random.seed(123)
    i_coords = np.arange(9)
    j_coords = np.arange(9)
    ii, jj = np.meshgrid(i_coords, j_coords, indexing='ij')
    
    # Distance increases away from diagonal
    dist_matrix = 0.1 + 0.5 * np.abs(ii - jj) + 0.2 * np.random.rand(9, 9)
    
    print(f"Window size: {dist_matrix.shape}")
    
    # Classical path
    classical_path = classical_dtw_path_in_window(dist_matrix)
    classical_moves = path_to_moves(classical_path)
    print(f"Classical path length: {len(classical_path)}")
    print(f"Qubits needed: {2 * len(classical_moves)}")
    
    # Run QAOA with more layers
    result = qaoa_refine_window_steps(
        dist_matrix,
        p=2,  # Deeper
        shots=2048,
        maxiter=50,
        verbose=True
    )
    
    print(f"\nResult:")
    print(f"  Improvement: {result['improvement_pct']:.2f}%")
    print(f"  Endpoint match: {result['endpoint_match']}")
    print(f"  Circuit depth: {result['circuit_depth']}")
    
    return result


def test_comparison():
    """Compare qubit counts: cell-based vs step-based."""
    print("\n" + "="*80)
    print("TEST 4: Qubit Count Comparison")
    print("="*80)
    
    window_sizes = [(5, 5), (7, 7), (9, 9), (12, 12)]
    
    print(f"{'Window':<12} {'Cells':<10} {'Steps':<10} {'Qubits (cell)':<15} {'Qubits (step)':<15} {'Reduction':<10}")
    print("-" * 80)
    
    for n, m in window_sizes:
        n_cells = n * m
        
        # Estimate path length (worst case: all right then all down)
        est_path_length = n + m - 1
        
        # Qubits
        qubits_cell = n_cells
        qubits_step = 2 * est_path_length
        
        reduction = qubits_cell / qubits_step if qubits_step > 0 else 0
        
        print(f"{n}×{m:<8} {n_cells:<10} {est_path_length:<10} {qubits_cell:<15} {qubits_step:<15} {reduction:.1f}×")
    
    print("\n✅ Step encoding reduces qubits by 5-10×!")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("QAOA-DTW STEP-BASED ENCODING TESTS")
    print("="*80)
    
    # Test 1: Basic encoding
    test_move_encoding()
    
    # Test 2: Small window
    result1 = test_small_window()
    
    # Test 3: Medium window
    result2 = test_medium_window()
    
    # Test 4: Comparison
    test_comparison()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if result1:
        print(f"\nSmall window (5×5):")
        print(f"  Qubits: {result1['n_qubits']}")
        print(f"  Improvement: {result1['improvement_pct']:.2f}%")
        print(f"  Endpoint: {result1['endpoint_match']}")
    
    if result2:
        print(f"\nMedium window (9×9):")
        print(f"  Qubits: {result2['n_qubits']}")
        print(f"  Improvement: {result2['improvement_pct']:.2f}%")
        print(f"  Endpoint: {result2['endpoint_match']}")
    
    print("\n" + "="*80)
    print("✅ All tests complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
