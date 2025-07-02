# src/classifier.py
from dtw import dtw_distance

def classify_knn(test_seq, train_seqs, train_labels):
    min_dist = float('inf')
    best_label = None
    for seq, label in zip(train_seqs, train_labels):
        dist = dtw_distance(test_seq, seq)
        if dist < min_dist:
            min_dist = dist
            best_label = label
    return best_label
