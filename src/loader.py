# src/loader.py
import numpy as np
import os

def load_skeleton_file(path):
    raw = np.loadtxt(path)
    num_frames = raw.shape[0] // 20
    data = raw.reshape((num_frames, 20, 4))  # includes confidence
    return data[:, :, :3]  # only x, y, z

def flatten_sequence(sequence):
    return sequence.reshape(sequence.shape[0], -1)  # (T, 60)

def load_all_sequences(folder):
    sequences = []
    labels = []
    for fname in os.listdir(folder):
        if fname.endswith("_skeleton.txt"):
            try:
                seq = flatten_sequence(load_skeleton_file(os.path.join(folder, fname)))
                label = int(fname[1:3]) - 1  # 'a01' → 0-indexed
                sequences.append(seq)
                labels.append(label)
            except Exception as e:
                print(f"Failed to load {fname}: {e}")
    return sequences, labels
