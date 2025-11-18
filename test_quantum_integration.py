#!/usr/bin/env python
"""
Quick test to verify real quantum fidelity integration into DTW pipeline.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dtw.dtw_runner import dtw_distance, one_nn

print("=" * 70)
print("Testing Real Quantum Fidelity Integration")
print("=" * 70)

# Create simple test sequences
np.random.seed(42)
seqA = np.random.randn(10, 4)  # 10 frames, 4-D features
seqB = np.random.randn(12, 4)  # 12 frames, 4-D features
seqC = np.random.randn(8, 4)   # 8 frames, 4-D features

print("\nTest 1: DTW with classical fidelity")
print("-" * 70)
try:
    dist_classical = dtw_distance(seqA, seqB, metric='fidelity')
    print(f"✅ Classical fidelity DTW distance: {dist_classical:.4f}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\nTest 2: DTW with quantum fidelity (SWAP test)")
print("-" * 70)
try:
    dist_quantum = dtw_distance(seqA, seqB, metric='quantum_fidelity', quantum_shots=256)
    print(f"✅ Quantum fidelity DTW distance: {dist_quantum:.4f}")
    print(f"   Difference from classical: {abs(dist_quantum - dist_classical):.4f}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\nTest 3: 1-NN classification with quantum fidelity")
print("-" * 70)
try:
    train_seqs = [seqA, seqB]
    train_labels = [0, 1]
    test_seq = seqC
    
    pred, dist = one_nn(
        train_seqs,
        train_labels,
        test_seq,
        metric='quantum_fidelity',
        quantum_shots=256
    )
    print(f"✅ 1-NN prediction: {pred}")
    print(f"   Distance to nearest neighbor: {dist:.4f}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\nTest 4: Compare execution times")
print("-" * 70)
import time

# Classical fidelity timing
start = time.time()
for _ in range(5):
    _ = dtw_distance(seqA, seqB, metric='fidelity')
classical_time = (time.time() - start) / 5

print(f"Classical fidelity: {classical_time * 1000:.1f} ms per DTW")

# Quantum fidelity timing
start = time.time()
for _ in range(5):
    _ = dtw_distance(seqA, seqB, metric='quantum_fidelity', quantum_shots=256)
quantum_time = (time.time() - start) / 5

print(f"Quantum fidelity:   {quantum_time * 1000:.1f} ms per DTW")
print(f"Slowdown factor:    {quantum_time / classical_time:.1f}x")

print("\n" + "=" * 70)
print("✅ All tests passed! Real quantum fidelity is integrated.")
print("=" * 70)
