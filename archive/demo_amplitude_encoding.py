"""
Demo script for amplitude encoding utilities.

Loads a real skeleton sequence and demonstrates normalization.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from features.amplitude_encoding import (
    encode_unit_vector,
    batch_encode_unit_vectors,
    verify_normalization
)
from src.loader import load_all_sequences


def demo_single_frame():
    """Demo encoding a single frame."""
    print("="*60)
    print("DEMO 1: Single Frame Encoding")
    print("="*60)
    
    # Create a random 60-D vector (simulating a skeleton frame)
    frame = np.random.uniform(0, 300, 60)
    print(f"Original frame shape: {frame.shape}")
    print(f"Original frame norm: {np.linalg.norm(frame):.4f}")
    
    # Encode as unit vector
    encoded = encode_unit_vector(frame)
    print(f"Encoded frame shape: {encoded.shape}")
    print(f"Encoded frame norm: {np.linalg.norm(encoded):.4f}")
    print(f"✅ Normalization successful: {np.abs(np.linalg.norm(encoded) - 1.0) < 1e-6}")


def demo_sequence():
    """Demo encoding a full sequence."""
    print("\n" + "="*60)
    print("DEMO 2: Full Sequence Encoding")
    print("="*60)
    
    # Load one real skeleton sequence
    sequences, labels = load_all_sequences("msr_action_data")
    
    if len(sequences) > 0:
        seq = sequences[0]
        print(f"Loaded sequence shape: {seq.shape}")
        print(f"Number of frames: {seq.shape[0]}")
        
        # Encode entire sequence
        encoded_seq = batch_encode_unit_vectors(seq)
        
        # Compute norms
        norms = np.linalg.norm(encoded_seq, axis=1)
        
        print(f"\nEncoded sequence shape: {encoded_seq.shape}")
        print(f"Min norm: {np.min(norms):.6f}")
        print(f"Max norm: {np.max(norms):.6f}")
        print(f"Mean norm: {np.mean(norms):.6f}")
        print(f"Std of norms: {np.std(norms):.6e}")
        
        # Verify normalization
        is_normalized = verify_normalization(encoded_seq)
        print(f"✅ All frames normalized: {is_normalized}")
    else:
        print("No sequences found in msr_action_data/")


def demo_zero_vector():
    """Demo zero vector handling."""
    print("\n" + "="*60)
    print("DEMO 3: Zero Vector Handling")
    print("="*60)
    
    # Create a zero vector
    zero_vec = np.zeros(60)
    print(f"Zero vector norm: {np.linalg.norm(zero_vec):.4f}")
    
    # Encode it
    encoded = encode_unit_vector(zero_vec)
    print(f"Encoded zero vector norm: {np.linalg.norm(encoded):.4f}")
    print(f"All elements equal to 1/√60: {np.allclose(encoded, 1/np.sqrt(60))}")
    print(f"✅ Zero vector handled correctly")


def demo_batch_with_zeros():
    """Demo batch encoding with some zero rows."""
    print("\n" + "="*60)
    print("DEMO 4: Batch with Zero Rows")
    print("="*60)
    
    # Create batch with some zero rows
    X = np.random.uniform(0, 300, (10, 60))
    X[3] = 0.0  # Zero row
    X[7] = 0.0  # Another zero row
    
    print(f"Batch shape: {X.shape}")
    print(f"Zero rows at indices: [3, 7]")
    
    # Encode
    encoded = batch_encode_unit_vectors(X)
    
    # Check norms
    norms = np.linalg.norm(encoded, axis=1)
    print(f"\nAll norms ≈ 1.0: {np.allclose(norms, 1.0, rtol=1e-6)}")
    print(f"Min norm: {np.min(norms):.6f}")
    print(f"Max norm: {np.max(norms):.6f}")
    print(f"✅ Batch encoding with zeros successful")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AMPLITUDE ENCODING DEMO")
    print("="*60 + "\n")
    
    demo_single_frame()
    demo_sequence()
    demo_zero_vector()
    demo_batch_with_zeros()
    
    print("\n" + "="*60)
    print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
    print("="*60)
