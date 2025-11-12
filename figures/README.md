# QDTW Evaluation Results

## Summary

This directory contains visualizations comparing the performance of Quantum PCA (Uq) and Classical PCA (Uc) for action recognition using Dynamic Time Warping (DTW). The analysis evaluates both accuracy and computational efficiency across different dimensionality reductions (k=5, 8, 10) and distance metrics (cosine, euclidean, fidelity). Results demonstrate the trade-offs between dimensionality reduction, classification accuracy, and query time. Lower k values significantly reduce computational cost while maintaining competitive accuracy, particularly for the euclidean metric. The Pareto frontier plot reveals optimal configurations balancing accuracy and speed.

## Figures

- `accuracy_vs_k.png`: Classification accuracy vs number of principal components

- `time_vs_k.png`: Average query time vs number of principal components

- `pareto_accuracy_time.png`: Pareto frontier showing accuracy-time trade-offs


## Best Results by Method and Metric


### Euclidean Distance

| Method | Best k | Accuracy | Avg Time (ms) |
|--------|--------|----------|---------------|
| Uc     |     10 |   0.8299 |         847.1 |
| Uq     |     10 |   0.8114 |         871.9 |

### Cosine Distance

| Method | Best k | Accuracy | Avg Time (ms) |
|--------|--------|----------|---------------|
| Uc     |     10 |   0.7870 |        2492.6 |
| Uq     |     10 |   0.7812 |        2304.3 |

### Fidelity Distance

| Method | Best k | Accuracy | Avg Time (ms) |
|--------|--------|----------|---------------|
| Uc     |     10 |   0.8041 |        2039.0 |
| Uq     |     10 |   0.7849 |        2115.7 |

### Baseline (60D Full Space)

| Metric | Accuracy | Avg Time (ms) |
|--------|----------|---------------|
| cosine     |   0.7876 |        4250.5 |
| euclidean  |   0.8319 |        1890.3 |
| fidelity   |   0.7965 |        3980.2 |

## Key Findings

- **Highest Accuracy**: Uc with k=10, euclidean metric achieved 0.8299 accuracy

- **Fastest Query**: Uc with k=5, euclidean metric at 438.7ms per query

- **Speedup**: Up to 4.3x faster than 60D baseline with minimal accuracy loss

- **Method Comparison**: Classical PCA (Uc) achieves 1.44% higher average accuracy than Quantum PCA (Uq)


---
*Generated on 2025-11-06 13:36:58*
