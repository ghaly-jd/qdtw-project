# src/extract_features.py
import numpy as np
import os
from loader import load_all_sequences

def extract_save(folder="msr_action_data",
                 output_path="data/features.npy"):
    """
    1) load_all_sequences returns:
         sequences: List of np.ndarray, each of shape (T,60)
         labels:    List of ints
    2) we flatten all frames into a single (N,60) array.
    """
    sequences, labels = load_all_sequences(folder)
    # flatten across all sequences
    frames = [frame for seq in sequences for frame in seq]  
    X = np.vstack(frames)   # shape (total_frames, 60)
    np.save(output_path, X)
    print(f"Saved {X.shape[0]} frames → {output_path}")

if __name__ == "__main__":
    extract_save()
