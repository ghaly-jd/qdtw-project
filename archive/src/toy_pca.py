# src/toy_pca.py
import sys, os
import numpy as np
from sklearn.decomposition import PCA

def run_pca(input_path="data/features.npy",
            output_path="data/features_pca2.npy",
            n_components=2):
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"{input_path!r} not found. First run extract_features.py")
    X = np.load(input_path)            # (N,60)
    pca = PCA(n_components=n_components)
    X2 = pca.fit_transform(X)          # (N,2)
    np.save(output_path, X2)
    print(f"PCA: {X.shape} → {X2.shape}, saved → {output_path}")

if __name__ == "__main__":
    run_pca()
