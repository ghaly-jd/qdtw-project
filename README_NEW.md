# QDTW — Quantum-Enhanced Dynamic Time Warping for Skeleton-Based Action Recognition

> **Master's Research Project**  
> Quantum dimensionality reduction (VQD) + DTW 1-NN classification on MSR Action3D

---

## Overview

This project explores whether **quantum computing techniques** can improve skeleton-based human action recognition compared to classical methods. We build a full pipeline: raw skeleton data → standardization → PCA dimensionality reduction → DTW 1-Nearest Neighbor classification — and replace key classical components with quantum algorithms.

**Three quantum components are validated:**

| Component | What it replaces | Algorithm | Result |
|-----------|-----------------|-----------|--------|
| **Quantum PCA** | Classical SVD PCA | VQD (Variational Quantum Deflation) | **83.4%** vs 77.7% classical (+5.7%) |
| **Quantum Fidelity** | Classical cosine distance | SWAP-test circuit | Equivalent accuracy, true quantum measurement |
| **Quantum Path Refinement** | Greedy DTW path | QAOA optimization | Improved alignment on 87% of test pairs |

**Dataset**: [MSR Action3D](https://sites.google.com/view/wanqingli/data-sets/msr-action3d) — 567 skeleton sequences, 20 action classes, 60D features (20 joints × 3 coordinates)

---

## Key Results

### VQD Quantum PCA vs Classical PCA

| Method | Pipeline | Accuracy |
|--------|----------|----------|
| **VQD Quantum PCA** | Standardize → PCA 60→16D → VQD 16→8D → DTW 1-NN | **83.4%** |
| Classical PCA | Standardize → PCA 60→8D → DTW 1-NN | 77.7% |
| Raw baseline | Standardize → DTW 1-NN (60D) | 75.0% |

> VQD achieves a **+5.7% accuracy advantage** over classical PCA, demonstrating that quantum eigenvalue extraction captures more discriminative structure for action recognition.

### QAOA DTW Path Refinement

- Optimized DTW alignment paths using QAOA
- **87% of test pairs** showed improved alignment vs greedy DTW
- Average path cost reduction observed across all action classes

### SWAP-Test Quantum Fidelity

- Real quantum circuit (Hadamard → CSWAP → measurement) for state similarity
- Produces true quantum probabilistic fidelity measurements
- Validated against classical fidelity — equivalent accuracy with genuine quantum properties

---

## Pipeline Architecture

```
Raw Skeleton Data (567 sequences, 60D per frame)
        │
        ▼
┌──────────────────────────┐
│  Z-score Standardization │  Mean=0, Std=1 per feature
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  PCA Pre-reduction       │  60D → 16D (classical SVD)
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  VQD Quantum PCA         │  16D → 8D (variational quantum deflation)
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  DTW 1-NN Classification │  Cosine / Euclidean / Fidelity distance
│  + QAOA Path Refinement  │  Optional SWAP-test quantum fidelity
└──────────┬───────────────┘
           ▼
       Predicted Action Label (20 classes)
```

---

## Directory Structure

```
qdtw_project/
├── quantum/                        # Quantum & classical PCA implementations
│   ├── classical_pca.py            #   Classical SVD-based PCA
│   ├── qpca.py                     #   Quantum PCA (density matrix method)
│   ├── real_fidelity.py            #   SWAP-test quantum fidelity circuit
│   └── project.py                  #   Sequence projection into PCA subspace
├── dtw/                            # DTW distance computation
│   └── dtw_runner.py               #   DTW with cosine/euclidean/fidelity
├── features/                       # Feature encoding
│   └── amplitude_encoding.py       #   Z-score standardization + encoding
├── eval/                           # Evaluation framework
│   ├── ablations.py                #   Ablation study runner
│   ├── aggregate.py                #   Results aggregation
│   ├── plotting.py                 #   Visualization utilities
│   └── make_figures.py             #   Generate publication figures
├── scripts/                        # Pipeline execution scripts
│   ├── build_frame_bank.py         #   Build standardized frame bank
│   ├── project_sequences.py        #   Project sequences to subspace
│   ├── run_ablations.py            #   Run DTW 1-NN evaluation
│   ├── run_dtw_raw.py              #   Raw 60D DTW baseline
│   ├── run_dtw_subspace.py         #   DTW on projected data
│   ├── create_label_metadata.py    #   Generate label mappings
│   └── sanity_checks.py            #   Validation tests
├── experiments/                    # VQD & QAOA experiments
│   ├── vqd_pipeline.py             #   Main VQD quantum PCA experiment
│   └── vqd_proper_experiments/     #   Full experiment suite
│       ├── vqd_quantum_pca.py
│       ├── quantum_dtw_pipeline.py
│       ├── run_vqd_experiment.py
│       ├── qaoa_dtw_optimizer.py
│       └── results/
├── docs/                           # Result summaries & documentation
├── thesis_docs/                    # Thesis drafts & sections
├── thesis_figures/                 # Publication-quality figures
├── src/                            # Classical baseline implementation
├── quantum_src/                    # Grover's search & amplitude estimation
├── tests/                          # Unit tests
├── _archive/                       # Old experimental files (reference)
└── .gitignore
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/<your-username>/qdtw_project.git
cd qdtw_project

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install numpy scipy matplotlib scikit-learn qiskit qiskit-aer
```

### 2. Dataset

Download [MSR Action3D](https://sites.google.com/view/wanqingli/data-sets/msr-action3d) and place skeleton files in `msr_action_data/`:

```
msr_action_data/
├── a01_s01_e01_skeleton.txt
├── a01_s01_e02_skeleton.txt
└── ...
```

File naming: `aXX_sYY_eZZ_skeleton.txt` — Action XX, Subject YY, Instance ZZ.

### 3. Run the Full Pipeline

```bash
# Build standardized frame bank
python scripts/build_frame_bank.py --output data/frame_bank_std.npy --per-seq 20 --seed 42

# Compute PCA bases & project sequences
python quantum/classical_pca.py --frames data/frame_bank_std.npy --k 8 --output results/Uc_k8_std.npz
python scripts/project_sequences.py --k 8 --method Uc --output-dir results/subspace_std

# Evaluate
python scripts/run_ablations.py --distance --n-train 454 --n-test 113
```

### 4. Run VQD Quantum PCA Experiment

```bash
cd experiments/vqd_proper_experiments
python run_vqd_experiment.py
```

### 5. Run QAOA DTW Path Refinement

```bash
cd experiments/vqd_proper_experiments
python qaoa_dtw_optimizer.py
```

---

## Configuration

All scripts use **relative paths** by default. Pass custom paths via CLI arguments:

```bash
python scripts/build_frame_bank.py \
    --data-dir /your/path/to/msr_action_data \
    --output /your/path/to/output/frame_bank_std.npy
```

Each script supports `--help` for full argument documentation.

---

## Methodology

### Standardization

Z-score standardization (column-wise) is critical:

```
X_std = (X - mean) / std
```

Preserves magnitude differences between actions (jump height, reach distance). L2 normalization destroys this → ~5% accuracy. Standardization → 74%+.

### VQD (Variational Quantum Deflation)

1. Classical PCA pre-reduces 60D → 16D (tractable for quantum circuits)
2. VQD extracts top-8 quantum eigenvectors from the 16D covariance matrix
3. Sequences projected into 8D quantum subspace
4. DTW 1-NN classifies using cosine distance

Quantum eigenvectors capture more discriminative structure than classical SVD → +5.7% accuracy.

### SWAP-Test Fidelity

Replaces classical cosine similarity with quantum measurement:

```
|0> ---H---o---H--- Measure
           |
|psi> -----X-----------
           |
|phi> -----X-----------

Fidelity = 2*P(|0>) - 1
```

### QAOA Path Refinement

QAOA optimizes DTW alignment paths beyond greedy dynamic programming, finding lower-cost alignments for 87% of test pairs.

---

## Results Summary

| Experiment | Key Finding |
|------------|-------------|
| VQD vs Classical PCA | **83.4% vs 77.7%** — quantum extracts more discriminative features |
| SWAP-test fidelity | Equivalent accuracy, true quantum measurement properties |
| QAOA path refinement | Improved alignment on 87% of test pairs |
| Standardization vs L2-norm | **74% vs 5%** — standardization is critical |
| Best distance metric | Cosine > Fidelity > Euclidean |
| Optimal dimensionality | k=8 balances accuracy with 7.5x compression |

For detailed results, see `docs/`.

---

## Citation

```bibtex
@mastersthesis{qdtw2025,
  title={Quantum-Enhanced Dynamic Time Warping for Skeleton-Based Action Recognition},
  author={Your Name},
  year={2025}
}
```

## License

MIT License
