"""
Test QAOA-DTW Path Refinement

Validates the QAOA-based DTW path refinement on synthetic sequences.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from quantum.qaoa_dtw import qaoa_dtw_pipeline


def test_simple_sequences():
    """Test on simple synthetic sequences."""
    print("\n" + "="*80)
    print("TEST 1: Simple Synthetic Sequences")
    print("="*80)
    
    # Create two similar sequences
    np.random.seed(42)
    t = np.linspace(0, 2*np.pi, 30)
    seq1 = np.column_stack([np.sin(t), np.cos(t)])
    seq2 = np.column_stack([np.sin(t + 0.2), np.cos(t + 0.2)])  # Slight phase shift
    
    print(f"Sequence 1: {seq1.shape}")
    print(f"Sequence 2: {seq2.shape}")
    
    # Run QAOA-DTW
    results = qaoa_dtw_pipeline(
        seq1, seq2,
        window_length=12,  # Smaller for testing
        band_radius=3,
        p=1,
        shots=512,  # Fewer shots for speed
        maxiter=30,
        verbose=True
    )
    
    return results


def test_real_data():
    """Test on real MSR action data."""
    print("\n" + "="*80)
    print("TEST 2: Real MSR Action Data")
    print("="*80)
    
    # Load frame bank
    try:
        data = np.load('data/frame_bank.npy')
        print(f"Loaded data: {data.shape}")
        
        # Select two sequences from same action (should be similar)
        seq1 = data[0]  # First sample
        seq2 = data[3]  # Fourth sample (same action, different execution)
        
        print(f"Sequence 1: {seq1.shape}")
        print(f"Sequence 2: {seq2.shape}")
        
        # Run QAOA-DTW
        results = qaoa_dtw_pipeline(
            seq1, seq2,
            window_length=16,
            band_radius=4,
            p=1,
            shots=1024,
            maxiter=50,
            verbose=True
        )
        
        return results
        
    except FileNotFoundError:
        print("⚠️  frame_bank.npy not found, skipping real data test")
        return None


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("QAOA-DTW PATH REFINEMENT TESTS")
    print("="*80)
    
    # Test 1: Simple sequences
    results1 = test_simple_sequences()
    
    # Test 2: Real data (if available)
    results2 = test_real_data()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    if results1:
        print(f"\nTest 1 (Synthetic):")
        print(f"  Windows: {results1['n_windows']}")
        print(f"  Improved: {results1['pct_improved']:.1f}%")
        print(f"  Tied: {results1['pct_tied']:.1f}%")
        print(f"  Worse: {results1['pct_worse']:.1f}%")
        print(f"  Avg qubits: {results1['avg_qubits']:.1f}")
        print(f"  Avg depth: {results1['avg_depth']:.1f}")
    
    if results2:
        print(f"\nTest 2 (Real MSR):")
        print(f"  Windows: {results2['n_windows']}")
        print(f"  Improved: {results2['pct_improved']:.1f}%")
        print(f"  Tied: {results2['pct_tied']:.1f}%")
        print(f"  Worse: {results2['pct_worse']:.1f}%")
        print(f"  Avg qubits: {results2['avg_qubits']:.1f}")
        print(f"  Avg depth: {results2['avg_depth']:.1f}")
    
    print("\n" + "="*80)
    print("✅ All tests complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
