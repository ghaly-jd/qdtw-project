# 09 - DTW Classification

**File:** `09_DTW_CLASSIFICATION.md`  
**Purpose:** Dynamic Time Warping for 1-NN classification  
**For Thesis:** Methodology - evaluation method

---

## 9.1 Why DTW?

**Challenge:** Action sequences have **variable lengths** (13-255 frames).

**Traditional methods fail:**
- Euclidean distance: Requires same length
- Fixed-grid comparison: Loses temporal structure
- Padding/truncation: Adds noise or loses information

**DTW solution:** Optimal alignment of sequences with different lengths.

---

## 9.2 DTW Algorithm

### 9.2.1 Dynamic Programming Formulation

Given sequences $\mathbf{A} \in \mathbb{R}^{m \times d}$ and $\mathbf{B} \in \mathbb{R}^{n \times d}$:

**1. Cost matrix:**
$$C[i,j] = d(\mathbf{a}_i, \mathbf{b}_j)$$

where $d(\cdot, \cdot)$ is frame-to-frame distance (e.g., cosine).

**2. Accumulated cost (DP):**
$$D[i,j] = C[i,j] + \min\begin{cases}
D[i-1,j] & \text{(vertical)} \\
D[i,j-1] & \text{(horizontal)} \\
D[i-1,j-1] & \text{(diagonal)}
\end{cases}$$

**3. DTW distance:**
$$\text{DTW}(\mathbf{A}, \mathbf{B}) = D[m,n]$$

---

## 9.3 Implementation

```python
import numpy as np

def dtw_distance(seq1, seq2, metric='cosine'):
    """
    Compute DTW distance between two sequences.
    
    Parameters
    ----------
    seq1 : ndarray, shape (m, d)
    seq2 : ndarray, shape (n, d)
    metric : str
        'cosine', 'euclidean', or 'fidelity'
    
    Returns
    -------
    distance : float
        DTW distance
    """
    m, n = len(seq1), len(seq2)
    
    # Cost matrix
    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            C[i, j] = frame_distance(seq1[i], seq2[j], metric)
    
    # Accumulated cost matrix
    D = np.full((m + 1, n + 1), np.inf)
    D[0, 0] = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = C[i-1, j-1]
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    
    return D[m, n]


def frame_distance(a, b, metric='cosine'):
    """Frame-to-frame distance."""
    if metric == 'cosine':
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    elif metric == 'euclidean':
        return np.linalg.norm(a - b)
    elif metric == 'fidelity':
        overlap = np.abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        return 1 - overlap**2
```

---

## 9.4 Distance Metrics

| Metric | Formula | Best For |
|--------|---------|----------|
| **Cosine** | $1 - \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$ | ✓ **VQD features** (82.7%) |
| Euclidean | $\|\mathbf{a} - \mathbf{b}\|_2$ | Magnitude-sensitive (65.3%) |
| Fidelity | $1 - \left|\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}\right|^2$ | Quantum-inspired (80.1%) |

**Conclusion:** Cosine distance optimal for our features.

---

## 9.5 1-NN Classification

```python
def one_nn_classify(train_seqs, train_labels, test_seq):
    """
    1-Nearest Neighbor classification with DTW.
    
    Parameters
    ----------
    train_seqs : list of ndarray
        Training sequences
    train_labels : list of int
        Training labels
    test_seq : ndarray
        Test sequence
    
    Returns
    -------
    predicted_label : int
        Predicted class
    min_distance : float
        Distance to nearest neighbor
    """
    min_dist = np.inf
    predicted_label = None
    
    for train_seq, label in zip(train_seqs, train_labels):
        dist = dtw_distance(test_seq, train_seq, metric='cosine')
        
        if dist < min_dist:
            min_dist = dist
            predicted_label = label
    
    return predicted_label, min_dist


def evaluate_1nn(train_seqs, train_labels, test_seqs, test_labels):
    """Evaluate 1-NN accuracy."""
    correct = 0
    predictions = []
    
    for test_seq, true_label in zip(test_seqs, test_labels):
        pred_label, _ = one_nn_classify(train_seqs, train_labels, test_seq)
        predictions.append(pred_label)
        
        if pred_label == true_label:
            correct += 1
    
    accuracy = correct / len(test_seqs)
    return accuracy, predictions
```

---

## 9.6 Computational Complexity

**Time:** $O(mn)$ per DTW  
**Space:** $O(mn)$ for DP matrix

**For dataset:**
- Train: 510 sequences × 42 frames avg
- Test: 57 sequences × 42 frames avg
- Per test: 510 DTW calls × 42² ops ≈ **900K ops** → ~1 sec

**Total test time:** 57 test seqs × 1 sec ≈ **1 minute**

---

## 9.7 Key Results

| Method | Accuracy | Notes |
|--------|----------|-------|
| VQD + DTW (cosine) | **82.7%** | ✓ Best |
| PCA + DTW (cosine) | 77.7% | Baseline |
| VQD + DTW (euclidean) | 65.3% | Poor |
| Raw 60D + DTW | 72.0% | No reduction |

**Conclusion:** VQD+DTW+cosine is optimal combination.

---

**Next:** [10_EXPERIMENTAL_SETUP.md](./10_EXPERIMENTAL_SETUP.md) →

---

**Navigation:**
- [← 08_SEQUENCE_PROJECTION.md](./08_SEQUENCE_PROJECTION.md)
- [→ 10_EXPERIMENTAL_SETUP.md](./10_EXPERIMENTAL_SETUP.md)
- [↑ Index](./README.md)
