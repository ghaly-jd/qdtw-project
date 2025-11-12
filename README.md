# Quantum DTW (QDTW) Project

A comprehensive implementation of Quantum Dynamic Time Warping for skeleton-based action recognition using Grover's algorithm and Quantum Amplitude Estimation.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [File Descriptions](#file-descriptions)
- [Data Format](#data-format)
- [Installation](#installation)
- [Usage](#usage)
- [Modern Subspace Projection Pipeline](#modern-subspace-projection-pipeline)
  - [Step 1: Build Frame Bank](#step-1-build-frame-bank)
  - [Step 2: Compute PCA Basis](#step-2-compute-pca-basis)
  - [Step 3: Project Sequences](#step-3-project-sequences)
  - [Step 4: Run DTW Classification](#step-4-run-dtw-classification)
  - [Step 5: Generate Evaluation Figures](#step-5-generate-evaluation-figures)
  - [Step 6: Run Ablation Studies](#step-6-run-ablation-studies)
  - [Step 7: Create Label Metadata](#step-7-create-label-metadata-if-needed)
  - [Complete Pipeline Example](#complete-pipeline-example)
- [Legacy Pipeline Workflow](#legacy-pipeline-workflow-original-demo)
- [Key Algorithms](#key-algorithms)

---

## 🎯 Project Overview

This project implements a quantum-enhanced action recognition system that compares the performance of classical and quantum approaches to Dynamic Time Warping (DTW). The system processes 3D skeleton sequences from the MSR Action3D dataset and classifies human actions using k-Nearest Neighbors (k-NN) with DTW distance metrics.

**Key Features:**
- Classical DTW implementation (CPU and GPU)
- Quantum DTW using Grover's algorithm for O(√N) speedup
- Quantum Amplitude Estimation for advanced optimization
- Hybrid quantum-classical approaches
- Comprehensive benchmarking suite
- Amplitude encoding for quantum state preparation
- Visualization tools for skeleton data and quantum circuits

---

## ⚠️ CRITICAL FINDINGS - Encoding Failure (Nov 7, 2025)

### Problem Discovered

During ablation studies, we discovered a **fundamental failure in the quantum encoding approach**:

**Observed Accuracy**: 3-5% (near-random for 20-class problem)  
**Expected Baseline**: 60-80% (typical for classical DTW on skeleton data)

### Root Cause Analysis

The quantum amplitude encoding + PCA projection **destroys class discriminability**:

```
Distance Analysis (Euclidean DTW, k=8):
┌────────────────────────────────────────┐
│ Intra-class (Class 1):  8.15 ± 0.68   │  ← Same action
│ Intra-class (Class 2):  6.34 ± 0.21   │  ← Same action
│ Inter-class (1 vs 2):   7.18 ± 1.70   │  ← Different actions
└────────────────────────────────────────┘
```

**Problem**: Inter-class distances overlap heavily with intra-class distances, making classification impossible.

### What Went Wrong

1. **Amplitude Encoding**: Normalizing skeleton frames to unit vectors may lose discriminative magnitude information
2. **PCA on Encoded Data**: The quantum/classical PCA bases computed on normalized frames don't capture action-specific patterns
3. **Low-Dimensional Projection**: k=8 dimensions insufficient to preserve class structure after encoding

### Previous "Results" Were Fake

The documented **82.99% accuracy** came from:
- `eval/aggregate.py`: `create_sample_metrics()` function that generates synthetic data
- `scripts/run_dtw_subspace.py`: Used `label = i % 20` (fake labels based on sequence index)

**No real benchmark was ever run with correct labels until the ablation studies.**

### Evidence

1. **Label Loading Bug Fixed** (Nov 6-7, 2025):
   - Original code: `label_dict[seq_idx]` failed (KeyError)
   - Fixed to: `metadata['labels'][seq_idx]` (correct indexing)
   
2. **Ablation Results** (Nov 7, 2025):
   - Distance choice: 3.54% (cosine), 5.31% (euclidean), ~5% (fidelity)
   - All configurations fail regardless of metric or k value
   
3. **Separability Test** (Nov 7, 2025):
   - Loaded first 3 samples from Class 1 and Class 2
   - Computed DTW distances within and between classes
   - Result: No clear separation between classes

### Implications

- **Quantum encoding as implemented does NOT preserve action information**
- **The entire quantum DTW pipeline produces random predictions**
- **Classical DTW on raw skeleton data likely performs better**

### Next Steps to Fix

1. **Test Classical Baseline**: Run DTW on raw 60-D skeleton features (no encoding)
2. **Investigate Encoding**: 
   - Try alternative normalizations (preserve magnitude information)
   - Use angle encoding or other quantum feature maps
   - Increase dimensionality (k=20, 30, 40)
3. **Validate PCA**: Check if PCA bases actually capture action-discriminative patterns
4. **Benchmark Properly**: Re-run full pipeline with correct labels and compare to classical baseline

### Files Documenting This Issue

- This README (updated Nov 7, 2025)
- `scripts/run_ablations.py`: Lines 220-250 (sanity checks added)
- `scripts/run_ablations.py`: Lines 67-86 (label loading fix)
- Ablation output: `results/ablations.csv` (real performance data)

---

## 🏗️ Architecture

The project consists of four main components:

1. **Data Processing**: Load and preprocess MSR skeleton data
2. **Classical Baselines**: Standard DTW implementations (CPU/GPU)
3. **Quantum Algorithms**: Grover's search and QAE-based DTW
4. **Benchmarking & Visualization**: Performance comparison and visual outputs

---

## 📁 Directory Structure

```
qdtw_project/
├── README.md                          # This file
├── msr_action_data/                   # Skeleton data files
│   ├── a01_s01_e01_skeleton.txt      # Action 1, Subject 1, Instance 1
│   ├── a02_s01_e01_skeleton.txt      # Action 2, Subject 1, Instance 1
│   └── ...                            # 300+ skeleton files
├── data/                              # Processed features
│   ├── features.npy                   # Raw 60-D frame features
│   ├── features_pca2.npy              # PCA-reduced features
│   ├── frame_bank.npy                 # Training frame bank
│   └── frame_bank_test.npy            # Test frame bank
├── results/                           # Experimental results
│   ├── Uc_k*.npz                      # Classical PCA bases
│   ├── Uq_k*.npz                      # Quantum PCA bases
│   ├── metrics_baseline.csv           # Baseline (60-D) results
│   ├── metrics_subspace_Uc.csv        # Classical PCA results
│   ├── metrics_subspace_Uq.csv        # Quantum PCA results
│   ├── ablations.csv                  # Ablation study results
│   └── subspace/                      # Projected sequences
│       ├── Uc/k{3,5,8,10,12,16}/     # Classical projections
│       │   ├── train/seq_*.npy
│       │   └── test/seq_*.npy
│       └── Uq/k{3,5,8,10,12,16}/     # Quantum projections
│           ├── train/seq_*.npy
│           └── test/seq_*.npy
├── figures/                           # Generated visualizations
│   ├── accuracy_vs_k.png              # Accuracy comparison
│   ├── time_vs_k.png                  # Speed comparison
│   ├── pareto_accuracy_time.png       # Pareto frontier
│   ├── ablations_distance.png         # Distance metric analysis
│   ├── ablations_k_sweep.png          # Dimensionality analysis
│   ├── ablations_sampling.png         # Sampling strategy analysis
│   └── ablations_robustness.png       # Robustness analysis
├── src/                               # Core classical implementations
│   ├── loader.py                      # Data loading utilities
│   ├── dtw.py                         # Classical DTW algorithm
│   ├── classifier.py                  # k-NN classifier
│   ├── extract_features.py            # Feature extraction
│   ├── msr_visualizer.py              # 3D skeleton visualization
│   ├── visualizer.py                  # General visualization tools
│   ├── toy_pca.py                     # PCA dimensionality reduction
│   └── main.py                        # Classical pipeline entry point
├── quantum_src/                       # Quantum implementations
│   ├── classifier.py                  # Grover's search k-NN
│   └── quantum_amp_est.py             # Quantum Amplitude Estimation
├── quantum/                           # Quantum PCA implementations
│   ├── __init__.py
│   ├── classical_pca.py               # Classical SVD-based PCA
│   ├── qpca.py                        # Quantum density matrix PCA
│   └── project.py                     # Sequence projection utilities
├── dtw/                               # DTW implementations
│   ├── __init__.py
│   └── dtw_runner.py                  # DTW distance computation
├── eval/                              # Evaluation and analysis
│   ├── __init__.py
│   ├── aggregate.py                   # Aggregate results from CSVs
│   ├── plotting.py                    # Plotting utilities
│   ├── make_figures.py                # Generate all figures
│   └── ablations.py                   # Ablation study framework
├── features/                          # Quantum encoding utilities
│   └── amplitude_encoding.py          # Unit vector normalization
├── tests/                             # Unit tests
│   ├── test_amplitude_encoding.py     # Encoding tests
│   ├── test_frame_bank.py             # Frame bank tests
│   ├── test_classical_pca.py          # Classical PCA tests
│   ├── test_qpca.py                   # Quantum PCA tests
│   ├── test_project.py                # Projection tests
│   ├── test_dtw_runner.py             # DTW tests
│   ├── test_aggregate.py              # Aggregation tests
│   └── test_ablations.py              # Ablation tests
├── scripts/                           # Utility scripts
│   ├── build_frame_bank.py            # Frame preprocessing
│   ├── project_sequences.py           # Project sequences to subspace
│   ├── run_dtw_subspace.py            # Run DTW classification
│   ├── run_ablations.py               # Run ablation experiments
│   └── create_label_metadata.py       # Generate label mappings
├── visualizations/                    # Generated visual outputs
├── qdtw.py                            # Simple quantum DTW (demo)
├── q_classifier.py                    # Quantum k-NN wrapper
├── benchmark.py                       # Classical vs Quantum benchmark
├── grover_benchmark.py                # Comprehensive quantum benchmark
├── verify_quantum.py                  # Quantum circuit verification
├── gpu_classical_dtw.py               # GPU-accelerated DTW
├── gpu_classical_classifier.py        # GPU k-NN classifier
├── demo_amplitude_encoding.py         # Encoding demonstration
└── generate_all_visuals.py            # Generate all visualizations
```

---

## 📄 File Descriptions

### Root Directory Files

#### `qdtw.py`
**Purpose**: Simple demonstration of quantum DTW using Grover's algorithm.

**Key Functions**:
- `grover_min_index(options)`: Uses Grover's algorithm to find minimum value index in a small list (3 options)
- `qdtw_distance(seq1, seq2)`: Computes DTW distance using quantum minimum search at each step

**Usage**:
```bash
python qdtw.py
```
**What it does**: Demonstrates quantum minimum search on simple test sequences.

---

#### `q_classifier.py`
**Purpose**: Quantum k-NN classifier wrapper.

**Key Functions**:
- `classify_knn_quantum(test_seq, train_seqs, train_labels, k=1)`: Classifies test sequence using quantum DTW distance

**Dependencies**: Uses `qdtw_distance` from `qdtw.py`

---

#### `benchmark.py`
**Purpose**: Compares classical vs quantum DTW performance.

**Key Functions**:
- `run_benchmark()`: Runs both classical and quantum k-NN on test set

**Workflow**:
1. Loads MSR skeleton data from `msr_action_data/`
2. Splits into 70% train, 30% test
3. Runs classical DTW k-NN classification
4. Runs quantum DTW k-NN classification
5. Reports accuracy and execution time

**Usage**:
```bash
python benchmark.py
```

**Expected Output**:
```
===== BENCHMARK RESULTS =====
Classical DTW:
  - Accuracy: XX.XX%
  - Time: XX.XX seconds

Quantum DTW:
  - Accuracy: XX.XX%
  - Time: XX.XX seconds
```

---

#### `grover_benchmark.py`
**Purpose**: Comprehensive benchmark of multiple quantum algorithms.

**Algorithms Tested**:
1. Classical GPU DTW (baseline)
2. Quantum DTW (scipy optimized)
3. Grover's Search (O(√N) advantage)
4. Quantum Amplitude Estimation (advanced)
5. Hybrid Quantum-Classical
6. Error-Mitigated Quantum
7. Adaptive Quantum Classifier

**Key Functions**:
- `warmup_gpu()`: Prepares GPU for computation
- `run_comprehensive_quantum_benchmark()`: Main benchmarking routine

**Usage**:
```bash
python grover_benchmark.py
```

**Requirements**: CuPy for GPU acceleration

---

#### `verify_quantum.py`
**Purpose**: Verifies quantum circuit correctness.

**Key Tests**:
- Creates sample distance list (50 items)
- Runs safe quantum function (with classical fallback)
- Runs raw quantum function (without fallback)
- Compares results with classical argmin

**Usage**:
```bash
python verify_quantum.py
```

---

#### `gpu_classical_dtw.py`
**Purpose**: Pure GPU implementation of classical DTW.

**Key Functions**:
- `_dtw_distance_gpu_classical(seq1_gpu, seq2_gpu)`: GPU DTW computation
- `dtw_distance_gpu(seq1, seq2)`: Wrapper with data transfer

**Implementation**: Uses CuPy for GPU arrays and operations

---

#### `gpu_classical_classifier.py`
**Purpose**: GPU-accelerated classical k-NN classifier.

**Key Functions**:
- `_dtw_distance_gpu_fair(seq1_gpu, seq2_gpu)`: Fair comparison DTW (uses scipy like quantum)
- `classify_knn_classical_gpu(test_seq, train_seqs_gpu, train_labels, k=1)`: GPU k-NN classifier

---

#### `demo_amplitude_encoding.py`
**Purpose**: Demonstrates amplitude encoding for quantum states.

**Demos**:
1. Single frame encoding
2. Full sequence encoding
3. Zero vector handling
4. Batch encoding with zeros

**Usage**:
```bash
python demo_amplitude_encoding.py
```

---

#### `generate_all_visuals.py`
**Purpose**: Generates all project visualizations.

**Visualizations Created**:
1. **skeleton_pose.png**: Static 3D skeleton pose
2. **skeleton_animation.gif**: Animated action sequence
3. **dtw_alignment.png**: DTW cost matrix and alignment path
4. **grover_circuit.png**: Grover's algorithm circuit diagram
5. **quantum_performance.png**: Performance comparison charts
6. **accuracy_comparison.png**: Accuracy bar chart
7. **time_comparison.png**: Execution time comparison

**Usage**:
```bash
python generate_all_visuals.py
```

**Output Directory**: `visualizations/`

---

### `src/` Directory (Classical Implementation)

#### `src/loader.py`
**Purpose**: Load MSR skeleton data files.

**Key Functions**:
- `load_skeleton_file(path)`: Loads single skeleton file
  - Reads text file with 20 joints × 4 values (x, y, z, confidence)
  - Returns shape: `(num_frames, 20, 3)` - only x, y, z coordinates
- `flatten_sequence(sequence)`: Flattens to `(T, 60)` - 20 joints × 3 coords
- `load_all_sequences(folder)`: Loads all skeleton files from folder
  - Returns: `(sequences, labels)` where labels are extracted from filenames

**Data Format**:
- Filename: `aXX_sYY_eZZ_skeleton.txt`
  - `XX`: Action number (01-20)
  - `YY`: Subject number (01-10)
  - `ZZ`: Instance number (01-03)

---

#### `src/dtw.py`
**Purpose**: Classical DTW distance computation.

**Key Functions**:
- `dtw_distance(seq1, seq2)`: Computes DTW distance between two sequences
  - Uses Euclidean distance for frame-to-frame cost
  - Dynamic programming approach: O(nm) time complexity
  - Returns final DTW distance

**Algorithm**:
```python
dtw[i, j] = cost(i, j) + min(
    dtw[i-1, j],    # Insertion
    dtw[i, j-1],    # Deletion
    dtw[i-1, j-1]   # Match
)
```

---

#### `src/classifier.py`
**Purpose**: Classical k-NN classifier.

**Key Functions**:
- `classify_knn(test_seq, train_seqs, train_labels)`: 1-NN classifier using DTW distance
  - Computes DTW distance to all training sequences
  - Returns label of nearest neighbor

---

#### `src/extract_features.py`
**Purpose**: Extract and save features from skeleton data.

**Workflow**:
1. Loads all sequences using `loader.py`
2. Concatenates all frames
3. Saves to `data/features.npy`

**Usage**:
```bash
cd src
python extract_features.py
```

---

#### `src/toy_pca.py`
**Purpose**: PCA dimensionality reduction demonstration.

**Workflow**:
1. Loads features from `data/features.npy`
2. Applies PCA to reduce 60D → 2D
3. Saves to `data/features_pca2.npy`

---

#### `src/msr_visualizer.py`
**Purpose**: 3D visualization of MSR skeleton data.

**Key Functions**:
- `loadData(data_dir, action, subject, instance)`: Loads specific skeleton file
- Visualizes skeleton joints in 3D space

**Usage**:
```bash
cd src
python msr_visualizer.py
```

---

#### `src/main.py`
**Purpose**: Main entry point for classical pipeline.

**Workflow**:
1. Loads all sequences
2. Splits into train/test (70/30)
3. Runs k-NN classification
4. Reports accuracy

**Usage**:
```bash
cd src
python main.py
```

---

### `quantum_src/` Directory (Quantum Implementation)

#### `quantum_src/classifier.py`
**Purpose**: Grover's algorithm-based quantum classifier.

**Key Functions**:
- `grover_search_minimum(distances)`: Finds minimum distance index using Grover's algorithm
  - **Quantum advantage**: O(√N) vs O(N) classical search
  - Uses Qiskit for quantum circuit construction
  - Implements oracle to mark minimum value
  - Implements diffusion operator for amplitude amplification
  - Includes classical fallback for small N or errors

- `grover_search_minimum_raw_unsafe(distances)`: Raw quantum search without fallback (for testing)

- `classify_knn_quantum_grover(test_seq, train_seqs_gpu, train_labels, k=1)`: Quantum k-NN classifier
  - Computes all DTW distances
  - Uses Grover's algorithm to find minimum
  - Returns predicted label

**Circuit Components**:
1. **Oracle**: Marks the state corresponding to minimum index
2. **Diffusion Operator**: Amplifies marked state amplitude
3. **Iterations**: π/4 × √N iterations for optimal amplification

---

#### `quantum_src/quantum_amp_est.py`
**Purpose**: Advanced quantum algorithms using Amplitude Estimation.

**Key Functions**:

1. **`hybrid_quantum_classical_search(distances, quantum_threshold=0.15)`**
   - Intelligent selection between quantum and classical
   - Uses quantum only when classical is ambiguous
   - Analyzes distance distribution

2. **`classify_knn_quantum_hybrid(test_seq, train_seqs_gpu, train_labels, k=1)`**
   - Hybrid quantum-classical k-NN

3. **`quantum_amplitude_search_fixed(distances)`**
   - Quantum Amplitude Estimation for minimum search
   - Creates amplitude-based quantum states
   - Uses phase rotations and entanglement

4. **`quantum_amplitude_search_enhanced(distances)`**
   - Enhanced QAE with better circuit design

5. **`classify_knn_quantum_advanced(test_seq, train_seqs_gpu, train_labels, k=1)`**
   - Advanced quantum k-NN using QAE

6. **`classify_knn_quantum_error_mitigated(test_seq, train_seqs_gpu, train_labels, k=1)`**
   - Error mitigation techniques

7. **`adaptive_quantum_classifier(test_seq, train_seqs_gpu, train_labels, k=1)`**
   - Adapts strategy based on problem characteristics

---

### `quantum/` Directory (PCA Implementations)

#### `quantum/classical_pca.py`
**Purpose**: Classical PCA using SVD for dimensionality reduction.

**Key Functions**:
- `classical_pca(X, k)`: Performs SVD-based PCA
  - Input: Frame bank X of shape `(N, 60)`
  - Output: Top-k principal components U of shape `(60, k)`
  - Uses `numpy.linalg.svd` for efficient computation
  - Returns explained variance ratio for analysis

- `load_pca_components(filepath)`: Load saved PCA basis
- `save_pca_components(U, explained_variance, filepath)`: Save PCA results

**Usage**:
```bash
python quantum/classical_pca.py \
  --frames data/frame_bank.npy \
  --k 8 \
  --output results/Uc_k8.npz
```

**When to use**: Production systems, when speed matters

---

#### `quantum/qpca.py`
**Purpose**: Quantum PCA using density matrix diagonalization.

**Algorithm**:
1. Constructs density matrix ρ = (1/N) Σᵢ |xᵢ⟩⟨xᵢ|
2. Diagonalizes ρ to find eigenvectors
3. Returns top-k eigenvectors as quantum PCA basis

**Key Functions**:
- `quantum_pca(X, k, backend='aer_simulator', shots=8192)`: Quantum PCA simulation
  - Uses Qiskit quantum circuits
  - Simulates measurement outcomes
  - Approximates density matrix eigenvectors
  
- `density_matrix_qpca(X, k)`: Direct density matrix method
  - Faster alternative using classical simulation
  - Mathematically equivalent to quantum approach
  - No quantum circuits needed

**Usage**:
```bash
python quantum/qpca.py \
  --frames data/frame_bank.npy \
  --k 8 \
  --output results/Uq_k8.npz \
  --shots 8192
```

**When to use**: Research, quantum algorithm exploration

---

#### `quantum/project.py`
**Purpose**: Project sequences into PCA subspace.

**Key Functions**:
- `project_sequence(sequence, U)`: Projects single sequence
  - Input: Sequence of shape `(T, 60)`
  - Output: Projected sequence of shape `(T, k)`
  - Formula: X_proj = X @ U

- `project_batch(sequences, U)`: Batch projection for efficiency
- `inverse_project(sequence_proj, U)`: Reconstruct from subspace

**Why it works**: 
- PCA finds directions of maximum variance
- Projecting preserves most important motion patterns
- Lower dimensionality = faster DTW
- Often improves classification (noise reduction)

---

### `dtw/` Directory

#### `dtw/dtw_runner.py`
**Purpose**: DTW distance computation with multiple metrics.

**Key Functions**:
- `dtw_distance(seq1, seq2, metric='euclidean')`: Compute DTW distance
  - Supports: `euclidean`, `cosine`, `fidelity`
  - Returns: Scalar distance value
  - Uses dynamic programming for optimal alignment

- `classify_1nn(test_seq, train_seqs, train_labels, metric='euclidean')`: 1-NN classification
  - Computes DTW to all training sequences
  - Returns label of nearest neighbor

**Distance Metrics**:

1. **Euclidean**: Standard L2 distance
   ```python
   dist = sqrt(sum((a - b)^2))
   ```

2. **Cosine**: Angular distance
   ```python
   dist = 1 - (a·b) / (||a|| ||b||)
   ```

3. **Fidelity**: Quantum state fidelity
   ```python
   dist = 1 - sqrt(sum(sqrt(a * b)))
   ```

**Usage**:
```python
from dtw.dtw_runner import dtw_distance, classify_1nn

# Compute distance
dist = dtw_distance(seq1, seq2, metric='euclidean')

# Classify
pred = classify_1nn(test_seq, train_seqs, train_labels)
```

---

### `eval/` Directory (Evaluation Framework)

#### `eval/aggregate.py`
**Purpose**: Aggregate results from multiple CSV files.

**Key Functions**:
- `load_results(results_dir)`: Load all CSV result files
- `aggregate_metrics(results_dir)`: Compute summary statistics
- `compare_methods(df, metric='accuracy')`: Statistical comparison

**What it does**:
- Reads `metrics_*.csv` files from results directory
- Combines into single DataFrame
- Computes means, stds, min, max
- Enables cross-method comparison

---

#### `eval/plotting.py`
**Purpose**: Plotting utilities for evaluation.

**Key Functions**:
- `plot_accuracy_vs_k(df, output_path)`: Accuracy comparison plot
- `plot_time_vs_k(df, output_path)`: Speed comparison plot  
- `plot_pareto_frontier(df, output_path)`: Pareto frontier analysis
- `set_plot_style()`: Consistent plot styling

**Plot Types**:
1. **Line plots**: Show trends across k values
2. **Bar charts**: Compare discrete configurations
3. **Scatter plots**: Pareto accuracy-time tradeoffs

---

#### `eval/make_figures.py`
**Purpose**: Generate all evaluation figures from results.

**Workflow**:
1. Loads results from CSV files
2. Calls plotting functions for each figure type
3. Saves publication-quality PNGs

**Usage**:
```bash
python eval/make_figures.py \
  --results-dir results \
  --output-dir figures \
  --dpi 300
```

**Output**: 3 figures in `figures/` directory

---

#### `eval/ablations.py`
**Purpose**: Ablation study framework.

**Utility Functions**:
- `add_temporal_jitter(seq, drop_rate, seed)`: Random frame dropping
- `add_joint_noise(seq, sigma, seed)`: Gaussian noise injection
- `sample_frames_uniform(seq, n_samples, seed)`: Uniform temporal sampling
- `sample_frames_energy(seq, n_samples, seed)`: Energy-based sampling

**Experiment Runners**:
- `run_distance_choice_ablation()`: Test distance metrics
- `run_k_sweep_ablation()`: Test dimensionality values
- `run_sampling_strategy_ablation()`: Test frame sampling
- `run_robustness_ablation()`: Test noise/jitter robustness

**Plotting Functions**:
- `plot_distance_choice_ablation()`: Bar chart comparison
- `plot_k_sweep_ablation()`: Line plot of k sweep
- `plot_sampling_strategy_ablation()`: Bar chart comparison
- `plot_robustness_ablation()`: Line plot of robustness

**Why ablations matter**:
- Understand design choice impacts
- Identify optimal configurations
- Test system robustness
- Validate algorithmic decisions

---

### `scripts/` Directory (Pipeline Scripts)

#### `scripts/build_frame_bank.py`
**Purpose**: Extract and organize frames from sequences.

**See**: [Step 1 in Modern Pipeline](#step-1-build-frame-bank)

---

#### `scripts/project_sequences.py`
**Purpose**: Project sequences into PCA subspace.

**See**: [Step 3 in Modern Pipeline](#step-3-project-sequences)

---

#### `scripts/run_dtw_subspace.py`
**Purpose**: Run DTW classification on projected sequences.

**See**: [Step 4 in Modern Pipeline](#step-4-run-dtw-classification)

---

#### `scripts/run_ablations.py`
**Purpose**: Run all ablation experiments.

**See**: [Step 6 in Modern Pipeline](#step-6-run-ablation-studies)

---

#### `scripts/create_label_metadata.py`
**Purpose**: Generate label mappings for projected sequences.

**See**: [Step 7 in Modern Pipeline](#step-7-create-label-metadata-if-needed)

---

### `features/` Directory

#### `features/amplitude_encoding.py`
**Purpose**: Quantum amplitude encoding utilities.

**Key Functions**:
- `encode_unit_vector(x)`: Normalizes single 60-D vector to unit length (L2 norm = 1)
  - Required for quantum amplitude encoding
  - Handles zero vectors gracefully

- `batch_encode_unit_vectors(X)`: Row-wise normalization of multiple vectors
  - Shape: `(T, 60)` → `(T, 60)` with each row normalized

- `verify_normalization(X, tolerance=1e-6)`: Verifies all rows are unit-normalized

**Why Needed**: Quantum states must be normalized (probability amplitudes sum to 1)

---

### `tests/` Directory

#### `tests/test_amplitude_encoding.py`
**Purpose**: Unit tests for amplitude encoding.

**Test Cases**:
- Single vector encoding
- Batch encoding
- Zero vector handling
- Normalization verification
- Edge cases

---

### `scripts/` Directory

#### `scripts/build_frame_bank.py`
**Purpose**: Preprocess and build frame database.

**Workflow**:
1. Loads all skeleton sequences
2. Extracts individual frames
3. Organizes by action class
4. Saves processed frame bank

---

## 📊 Data Format

### MSR Skeleton Files

**File Naming**: `aXX_sYY_eZZ_skeleton.txt`
- `XX`: Action ID (01-20)
  - 01: High wave
  - 02: Horizontal wave
  - 03: Hammer
  - 08: High throw
  - ... (20 total actions)
- `YY`: Subject ID (01-10)
- `ZZ`: Instance/Execution number (01-03)

**File Content**:
- Each line: 4 values (x, y, z, confidence)
- 20 joints per frame
- Variable number of frames per action

**Joint Structure** (20 joints):
```
Joint 0: Hip Center
Joint 1: Spine
Joint 2: Shoulder Center
Joint 3: Head
Joint 4: Shoulder Left
Joint 5: Elbow Left
Joint 6: Wrist Left
Joint 7: Hand Left
Joint 8: Shoulder Right
Joint 9: Elbow Right
Joint 10: Wrist Right
Joint 11: Hand Right
Joint 12: Hip Left
Joint 13: Knee Left
Joint 14: Ankle Left
Joint 15: Foot Left
Joint 16: Hip Right
Joint 17: Knee Right
Joint 18: Ankle Right
Joint 19: Foot Right
```

**After Processing**:
- Shape: `(num_frames, 20, 3)` → `(num_frames, 60)` (flattened)
- Each frame: 60-dimensional vector (20 joints × 3 coordinates)

---

## 🔧 Installation

### Requirements

```bash
# Core dependencies
pip install numpy scipy matplotlib scikit-learn

# Quantum computing
pip install qiskit qiskit-aer

# GPU acceleration (optional but recommended)
pip install cupy-cuda11x  # Replace 11x with your CUDA version

# Visualization
pip install imageio
```

### Full Installation

```bash
# Clone the repository
git clone <repository-url>
cd qdtw_project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy scipy matplotlib scikit-learn qiskit qiskit-aer imageio

# Optional: GPU support
pip install cupy-cuda11x
```

---

## 🚀 Usage

### 1. Basic Classical Pipeline

Run the classical DTW k-NN classifier:

```bash
cd src
python main.py
```

**Output**: Classification accuracy on test set.

---

### 2. Simple Quantum Demo

Test the basic quantum DTW implementation:

```bash
python qdtw.py
```

**Output**: 
- DTW distance for test sequences
- Quantum minimum search demonstration

---

### 3. Classical vs Quantum Benchmark

Compare classical and quantum approaches:

```bash
python benchmark.py
```

**Output**:
- Classical DTW accuracy and time
- Quantum DTW accuracy and time
- Performance comparison

---

### 4. Comprehensive Quantum Benchmark

Run full quantum algorithm suite:

```bash
python grover_benchmark.py
```

**Output**:
- Multiple algorithm comparisons
- Detailed performance metrics
- Speed and accuracy analysis

**Note**: Requires GPU (CuPy) for optimal performance.

---

### 5. Verify Quantum Circuits

Test quantum circuit correctness:

```bash
python verify_quantum.py
```

**Output**:
- Classical result
- Safe quantum result (with fallback)
- Raw quantum result (no fallback)
- Verification status

---

### 6. Generate Visualizations

Create all project visualizations:

```bash
python generate_all_visuals.py
```

**Output Files** (in `visualizations/`):
- `skeleton_pose.png`
- `skeleton_animation.gif`
- `dtw_alignment.png`
- `grover_circuit.png`
- `quantum_performance.png`
- `accuracy_comparison.png`
- `time_comparison.png`

---

### 7. Amplitude Encoding Demo

Demonstrate quantum encoding:

```bash
python demo_amplitude_encoding.py
```

**Output**:
- Single frame encoding
- Sequence encoding
- Zero vector handling
- Normalization verification

---

## � Modern Subspace Projection Pipeline

### Overview

**⚠️ CRITICAL WARNING**: This pipeline currently produces **3-5% accuracy** (near-random) due to encoding failure. See "Critical Findings" section at top of README for details. The amplitude encoding + PCA approach destroys class separability.

This is the originally intended pipeline for production use. It uses PCA-based dimensionality reduction to project sequences into a lower-dimensional subspace before applying DTW, with the goal of improving both speed and accuracy.

**Pipeline Steps**:
1. Build frame bank from training data
2. Compute PCA basis (classical or quantum)
3. Project all sequences into k-dimensional subspace
4. Run DTW classification in reduced space
5. Generate evaluation metrics and figures
6. Run ablation studies to analyze design choices

### Step 1: Build Frame Bank

Extract and organize frames from all skeleton sequences:

```bash
python scripts/build_frame_bank.py \
  --data-dir msr_action_data \
  --output data/frame_bank.npy \
  --test-output data/frame_bank_test.npy \
  --test-fraction 0.2 \
  --seed 42
```

**What it does**:
- Loads all MSR Action3D skeleton sequences
- Performs train/test split (80/20 by default)
- Extracts individual frames from each sequence
- Saves frame banks: `frame_bank.npy` (train), `frame_bank_test.npy` (test)

**Output**:
- `data/frame_bank.npy`: Training frames, shape `(N_train_frames, 60)`
- `data/frame_bank_test.npy`: Test frames, shape `(N_test_frames, 60)`

---

### Step 2: Compute PCA Basis

Generate the subspace projection matrix using either classical SVD or quantum PCA:

#### Classical PCA (Recommended - Fast & Accurate)

```bash
python quantum/classical_pca.py \
  --frames data/frame_bank.npy \
  --k 8 \
  --output results/Uc_k8.npz
```

#### Quantum PCA (Research/Educational)

```bash
python quantum/qpca.py \
  --frames data/frame_bank.npy \
  --k 8 \
  --output results/Uq_k8.npz \
  --backend aer_simulator \
  --shots 8192
```

**Arguments**:
- `--frames`: Path to frame bank (from Step 1)
- `--k`: Number of principal components (dimensionality of subspace)
- `--output`: Where to save the PCA basis matrix U
- `--backend`: (qPCA only) Quantum simulator backend
- `--shots`: (qPCA only) Number of quantum circuit shots

**Output**:
- `results/U{c|q}_k{k}.npz`: Contains projection matrix U of shape `(60, k)`
  - `Uc`: Classical PCA basis
  - `Uq`: Quantum PCA basis

**Typical k values**: 3, 5, 8, 10, 12, 16 (8 is a good default)

---

### Step 3: Project Sequences

Project all sequences from 60-D to k-D subspace:

```bash
python scripts/project_sequences.py \
  --data-dir msr_action_data \
  --pca-file results/Uc_k8.npz \
  --output-dir results/subspace/Uc/k8 \
  --test-fraction 0.2 \
  --seed 42
```

**Arguments**:
- `--data-dir`: Directory containing original skeleton files
- `--pca-file`: PCA basis from Step 2 (Uc or Uq)
- `--output-dir`: Where to save projected sequences
- `--test-fraction`: Train/test split ratio
- `--seed`: Random seed for reproducibility

**Output Structure**:
```
results/subspace/Uc/k8/
├── train/
│   ├── seq_0000.npy  # Projected sequence 0, shape (T_0, 8)
│   ├── seq_0001.npy  # Projected sequence 1, shape (T_1, 8)
│   ├── ...
│   └── metadata.npz  # Labels and filenames
└── test/
    ├── seq_0000.npy
    ├── ...
    └── metadata.npz
```

**Run for multiple configurations**:
```bash
# Classical PCA, multiple k values
for k in 5 8 10; do
  python scripts/project_sequences.py \
    --data-dir msr_action_data \
    --pca-file results/Uc_k${k}.npz \
    --output-dir results/subspace/Uc/k${k}
done

# Quantum PCA, multiple k values
for k in 5 8 10; do
  python scripts/project_sequences.py \
    --data-dir msr_action_data \
    --pca-file results/Uq_k${k}.npz \
    --output-dir results/subspace/Uq/k${k}
done
```

---

### Step 4: Run DTW Classification

**⚠️ WARNING**: This step produces random predictions (~5% accuracy) due to encoding failure. The `run_dtw_subspace.py` script also had a bug where it used fake labels (`label = i % 20`). Use `run_ablations.py` instead for corrected label loading.

Perform 1-NN classification using DTW distance on projected sequences:

```bash
python scripts/run_dtw_subspace.py \
  --method Uc \
  --k 8 \
  --metric euclidean \
  --subspace-dir results/subspace
```

**Arguments**:
- `--method`: PCA method (`Uc` for classical, `Uq` for quantum)
- `--k`: Subspace dimensionality
- `--metric`: Distance metric (`euclidean`, `cosine`, or `fidelity`)
- `--subspace-dir`: Base directory containing projected sequences

**Known Issues**:
- Line 84 in `run_dtw_subspace.py` uses `label = i % 20` (fake labels)
- Should load labels from `metadata.npz` files created by `create_label_metadata.py`
- Even with correct labels, accuracy is ~5% due to encoding failure

**Output**:
- `results/metrics_subspace_{method}.csv`: Accuracy and timing for each k value
- Console output: Per-configuration results

**Distance Metrics**:
- `euclidean`: Standard L2 distance
- `cosine`: 1 - cosine_similarity (angular distance)
- `fidelity`: Quantum state fidelity distance (1 - sqrt(fidelity))

**Run all configurations**:
```bash
# Test all methods, k values, and metrics
for method in Uc Uq; do
  for k in 5 8 10; do
    for metric in euclidean cosine fidelity; do
      python scripts/run_dtw_subspace.py \
        --method $method \
        --k $k \
        --metric $metric
    done
  done
done
```

---

### Step 5: Generate Evaluation Figures

Create publication-quality visualizations of results:

```bash
python eval/make_figures.py \
  --results-dir results \
  --output-dir figures \
  --dpi 300
```

**Arguments**:
- `--results-dir`: Directory containing CSV results files
- `--output-dir`: Where to save generated figures
- `--dpi`: Resolution for saved images (default: 300)

**Generated Figures**:

1. **`accuracy_vs_k.png`**: Classification accuracy vs subspace dimensionality
   - Compares Uq and Uc across different k values
   - Shows baseline (60-D) accuracy for reference
   - Helps identify optimal k value

2. **`time_vs_k.png`**: Average computation time vs k
   - Shows time per test sample
   - Illustrates speed/accuracy tradeoff
   - Compares different distance metrics

3. **`pareto_accuracy_time.png`**: Pareto frontier analysis
   - 2D plot: accuracy vs time
   - Identifies optimal configurations
   - Shows dominated vs non-dominated solutions

**Sample Output**:
```
Generated figures:
  - figures/accuracy_vs_k.png (189 KB)
  - figures/time_vs_k.png (374 KB)
  - figures/pareto_accuracy_time.png (233 KB)
```

---

### Step 6: Run Ablation Studies

**✅ USES CORRECT LABELS**: This script (fixed Nov 6-7, 2025) properly loads labels from `metadata.npz` and reveals the true performance (~3-5% accuracy).

Analyze the impact of design choices and robustness properties:

```bash
python scripts/run_ablations.py \
  --all \
  --n-train 454 \
  --n-test 113 \
  --output-dir results \
  --figures-dir figures
```

**Arguments**:
- `--all`: Run all ablation experiments (or use specific flags below)
- `--distance`: Distance metric comparison (cosine vs euclidean vs fidelity)
- `--k-sweep`: Subspace dimensionality sweep (k ∈ {3, 5, 8, 10, 12, 16})
- `--sampling`: Frame sampling strategy (uniform vs energy-based)
- `--robustness`: Noise and temporal jitter robustness
- `--n-train`: Number of training samples to use
- `--n-test`: Number of test samples to use
- `--use-sample-data`: Use synthetic data for testing (optional)

**Current Results** (Nov 7, 2025 - Real Data):
```
Distance Choice (Uq, k=8):
  - cosine:    3.54% accuracy
  - euclidean: 5.31% accuracy  
  - fidelity:  ~5% accuracy

⚠️ Near-random performance for 20-class problem (expected 5%)
⚠️ Encoding approach has failed
```

**Ablation Experiments**:

1. **Distance Choice** (`--distance`):
   - Compares cosine, euclidean, and fidelity metrics
   - Tests both Uq and Uc methods
   - Identifies best metric for this task

2. **K Sweep** (`--k-sweep`):
   - Tests k ∈ {3, 5, 8, 10, 12, 16}
   - Analyzes dimensionality/accuracy tradeoff
   - Finds optimal subspace size

3. **Sampling Strategy** (`--sampling`):
   - **Uniform**: Sample frames at regular intervals
   - **Energy**: Sample high-motion frames (high L2 norm)
   - Determines best frame selection method

4. **Robustness** (`--robustness`):
   - **Noise**: Add Gaussian noise (σ ∈ {0.0, 0.01, 0.02})
   - **Temporal Jitter**: Random frame drops (drop_rate ∈ {0.0, 0.05})
   - Tests system robustness to data corruption

**Output Files**:

1. **`results/ablations.csv`**: Raw results for all experiments (REAL PERFORMANCE DATA)
   ```csv
   exp,method,k,metric,setting,accuracy,time_ms
   distance_choice,Uq,,cosine,cosine,0.0354,41213.4
   distance_choice,Uq,,euclidean,euclidean,0.0531,15523.5
   ...
   ```

2. **Generated Figures** (in `figures/`):
   - `ablations_distance.png`: Distance metric comparison (2-panel bar chart)
   - `ablations_k_sweep.png`: Accuracy and time vs k (2-panel line plot)
   - `ablations_sampling.png`: Sampling strategy comparison (2-panel bar chart)
   - `ablations_robustness.png`: Robustness to noise/jitter (2-panel line plot)

**Example Individual Experiments**:
```bash
# Just distance comparison
python scripts/run_ablations.py --distance --n-train 200 --n-test 50

# Just k sweep
python scripts/run_ablations.py --k-sweep --n-train 200 --n-test 50

# Multiple experiments
python scripts/run_ablations.py --distance --k-sweep --robustness
```

---

### Step 7: Create Label Metadata (if needed)

If you need to regenerate label mappings for projected sequences:

```bash
python scripts/create_label_metadata.py
```

**What it does**:
- Parses MSR Action3D filenames (e.g., `a01_s05_e02_skeleton.txt`)
- Extracts action IDs (1-20)
- Creates `metadata.npz` files in each subspace directory
- Maps sequence indices to ground-truth action labels

**When to use**:
- After running `project_sequences.py` for the first time
- If metadata files are missing or corrupted
- To ensure correct labels for evaluation

---

### Complete Pipeline Example

Run the entire pipeline from scratch:

```bash
# 1. Build frame bank
python scripts/build_frame_bank.py \
  --data-dir msr_action_data \
  --output data/frame_bank.npy \
  --test-output data/frame_bank_test.npy

# 2a. Classical PCA (k=8)
python quantum/classical_pca.py \
  --frames data/frame_bank.npy \
  --k 8 \
  --output results/Uc_k8.npz

# 2b. Quantum PCA (k=8)
python quantum/qpca.py \
  --frames data/frame_bank.npy \
  --k 8 \
  --output results/Uq_k8.npz

# 3. Project sequences (both methods)
python scripts/project_sequences.py \
  --data-dir msr_action_data \
  --pca-file results/Uc_k8.npz \
  --output-dir results/subspace/Uc/k8

python scripts/project_sequences.py \
  --data-dir msr_action_data \
  --pca-file results/Uq_k8.npz \
  --output-dir results/subspace/Uq/k8

# 4. Create label metadata
python scripts/create_label_metadata.py

# 5. Run DTW classification
python scripts/run_dtw_subspace.py \
  --method Uc \
  --k 8 \
  --metric euclidean

python scripts/run_dtw_subspace.py \
  --method Uq \
  --k 8 \
  --metric euclidean

# 6. Generate evaluation figures
python eval/make_figures.py

# 7. Run ablation studies
python scripts/run_ablations.py --all
```

**Expected Runtime** (on typical workstation):
- Frame bank: ~1 minute
- Classical PCA: ~5 seconds per k value
- Quantum PCA: ~2-5 minutes per k value
- Projection: ~30 seconds per configuration
- DTW classification: ~5-10 minutes per configuration
- Ablations: ~20-40 minutes for all experiments

---

### Pipeline Output Summary

After running the complete pipeline, you should have:

**Data Files**:
- `data/frame_bank.npy` - Training frames
- `data/frame_bank_test.npy` - Test frames
- `results/Uc_k*.npz` - Classical PCA bases
- `results/Uq_k*.npz` - Quantum PCA bases
- `results/subspace/{Uc|Uq}/k{k}/{train|test}/seq_*.npy` - Projected sequences
- `results/subspace/{Uc|Uq}/k{k}/{train|test}/metadata.npz` - Label mappings

**Results Files**:
- `results/metrics_baseline.csv` - 60-D baseline results
- `results/metrics_subspace_Uc.csv` - Classical PCA results
- `results/metrics_subspace_Uq.csv` - Quantum PCA results
- `results/ablations.csv` - Ablation study results

**Figures**:
- `figures/accuracy_vs_k.png` - Accuracy comparison
- `figures/time_vs_k.png` - Speed comparison
- `figures/pareto_accuracy_time.png` - Pareto frontier
- `figures/ablations_distance.png` - Distance metric analysis
- `figures/ablations_k_sweep.png` - Dimensionality analysis
- `figures/ablations_sampling.png` - Sampling strategy analysis
- `figures/ablations_robustness.png` - Robustness analysis

---

## �🔄 Legacy Pipeline Workflow (Original Demo)

### Complete Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    1. DATA LOADING                          │
│  msr_action_data/*.txt → loader.py → (sequences, labels)   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                2. PREPROCESSING                             │
│  Flatten: (T, 20, 3) → (T, 60)                             │
│  Optional: Amplitude encoding (normalize to unit vectors)   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               3. TRAIN/TEST SPLIT                           │
│  70% training, 30% testing (sklearn.train_test_split)       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              4. DISTANCE COMPUTATION                        │
│  For each test sample:                                      │
│    Classical: dtw_distance(test, train_i)                   │
│    Quantum: qdtw_distance(test, train_i)                    │
│    GPU: dtw_distance_gpu(test, train_i)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          5. MINIMUM SEARCH (k-NN, k=1)                      │
│  Classical: np.argmin(distances)          O(N)              │
│  Grover's: grover_search_minimum()        O(√N)             │
│  QAE: quantum_amplitude_search()          Advanced          │
│  Hybrid: hybrid_quantum_classical()       Adaptive          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                6. CLASSIFICATION                            │
│  Assign label of nearest neighbor                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              7. EVALUATION & RESULTS                        │
│  Accuracy, execution time, speedup metrics                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧮 Key Algorithms

### 1. Classical DTW

**Time Complexity**: O(nm) where n, m are sequence lengths

**Algorithm**:
```python
for i in range(1, n+1):
    for j in range(1, m+1):
        cost = distance(seq1[i], seq2[j])
        dtw[i,j] = cost + min(
            dtw[i-1, j],      # Insertion
            dtw[i, j-1],      # Deletion
            dtw[i-1, j-1]     # Match
        )
return dtw[n, m]
```

**Implementation**: `src/dtw.py`

---

### 2. Grover's Quantum Search

**Time Complexity**: O(√N) for searching N items

**Quantum Advantage**: Quadratic speedup over classical O(N) search

**Algorithm Steps**:
1. **Initialize**: Put all qubits in superposition |+⟩
2. **Oracle**: Mark the target state (minimum value index)
3. **Diffusion**: Amplify amplitude of marked state
4. **Repeat**: π/4 × √N iterations
5. **Measure**: Obtain result with high probability

**Key Components**:
```python
# Oracle (marks minimum index)
for each qubit:
    if bit == 0: apply X gate
apply multi-controlled Z gate
uncompute X gates

# Diffusion Operator
apply H to all qubits
apply X to all qubits
apply multi-controlled Z
uncompute X gates
apply H to all qubits
```

**Implementation**: `quantum_src/classifier.py`

---

### 3. Quantum Amplitude Estimation (QAE)

**Purpose**: Estimate amplitude of quantum state more efficiently

**Use Case**: Finding minimum by encoding distances as amplitudes

**Algorithm**:
1. Encode distances as quantum amplitudes
2. Apply phase rotations based on values
3. Use entanglement to correlate states
4. Measure and extract minimum

**Advantage**: Can provide better precision than Grover's for certain problems

**Implementation**: `quantum_src/quantum_amp_est.py`

---

### 4. Hybrid Quantum-Classical

**Strategy**: Use quantum only when beneficial

**Decision Logic**:
```python
if N < threshold:
    return classical_search()  # Faster for small N
elif close_values < 3:
    return classical_search()  # Clear winner
elif close_values > N/3:
    return classical_search()  # Too noisy for quantum
else:
    return quantum_search()    # Sweet spot for quantum
```

**Implementation**: `quantum_src/quantum_amp_est.py`

---

## 📈 Performance Characteristics

### Expected Results

| Algorithm | Complexity | Accuracy | Speed (relative) |
|-----------|-----------|----------|------------------|
| Classical CPU DTW | O(N) | Baseline | 1.0x |
| Classical GPU DTW | O(N) | Same | 2-5x faster |
| Quantum DTW (simple) | O(√N) | Same | Depends on N |
| Grover's Search | O(√N) | Same | Faster for N>100 |
| QAE | O(log N) | Same | Faster for large N |
| Hybrid | Adaptive | Same | Best overall |

**Note**: Quantum speedup depends on:
- Problem size (N)
- Quantum hardware quality
- Overhead of classical-quantum interface

---

## 🔍 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install qiskit qiskit-aer numpy scipy scikit-learn
   ```

2. **GPU Not Available**
   - GPU code (CuPy) is optional
   - Project works on CPU only (slower)

3. **Data Files Not Found**
   - Ensure `msr_action_data/` directory exists
   - Check skeleton files are present (`.txt` format)

4. **Quantum Circuit Errors**
   - Increase shots for better accuracy
   - Check Qiskit version compatibility

5. **Memory Issues**
   - Reduce test set size
   - Use batch processing
   - Close unused applications

---

## 📚 References

### Papers & Theory

- **Dynamic Time Warping**: Sakoe & Chiba (1978)
- **Grover's Algorithm**: Grover (1996)
- **Quantum Amplitude Estimation**: Brassard et al. (2002)
- **MSR Action3D Dataset**: Li et al. (2010)

### Libraries Used

- **Qiskit**: Quantum computing framework
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing
- **Scikit-learn**: Machine learning
- **CuPy**: GPU-accelerated NumPy
- **Matplotlib**: Visualization

---

## 🎓 Understanding the Code

### For Beginners

1. **Start with classical implementation**:
   - Read `src/loader.py` to understand data format
   - Study `src/dtw.py` for DTW algorithm
   - Check `src/main.py` for complete pipeline

2. **Learn quantum concepts**:
   - Run `qdtw.py` to see basic quantum search
   - Read `verify_quantum.py` to understand circuit verification
   - Study `quantum_src/classifier.py` for Grover's implementation

3. **Explore benchmarking**:
   - Run `benchmark.py` for simple comparison
   - Try `grover_benchmark.py` for comprehensive analysis

### For Advanced Users

- Modify quantum circuits in `quantum_src/`
- Implement custom distance metrics
- Experiment with different quantum algorithms
- Optimize GPU kernels in GPU files
- Add new visualization types

---

## 📝 Notes

- **Quantum Simulator**: Uses Qiskit Aer simulator (not real quantum hardware)
- **Scalability**: Currently optimized for datasets with 100-1000 samples
- **Accuracy**: Quantum and classical methods should have identical accuracy (same DTW logic)
- **Speed**: Quantum speedup is theoretical; simulator overhead may dominate for small problems

---

## 🤝 Contributing


To extend this project:
1. **FIX THE ENCODING FIRST** - Test classical DTW on raw 60-D features without encoding
2. Investigate alternative quantum encoding schemes that preserve discriminative information
3. Try higher dimensionality (k=20, 30, 40) to see if more dimensions help
4. Validate that PCA bases capture action-specific patterns (not just variance)
5. Add new quantum algorithms in `quantum_src/`
6. Implement additional classical baselines
7. Add more visualization types in `generate_all_visuals.py`
8. Improve encoding schemes in `features/`
9. Add unit tests in `tests/`

---

## 🚨 Known Issues & Limitations

### Critical Issues (Nov 7, 2025)

1. **Encoding Failure**: Amplitude encoding + PCA destroys class separability
   - Current accuracy: 3-5% (near-random)
   - Expected baseline: 60-80%
   - Root cause: Normalized frames lose discriminative structure
   
2. **Fake Metrics Files**: Previous "results" were synthetic data
   - `metrics_subspace_Uc.csv` and `metrics_subspace_Uq.csv` generated by `create_sample_metrics()`
   - Documented "82.99% accuracy" never actually achieved
   - Real performance revealed by ablation studies (Nov 7, 2025)

3. **Label Loading Bug** (FIXED Nov 6-7, 2025):
   - `scripts/run_dtw_subspace.py` line 84: Used `label = i % 20` (fake labels)
   - `scripts/run_ablations.py`: Now correctly loads from `metadata.npz`

### Recommendations

**Before using this project**:
1. ⚠️ Do NOT trust the documented 82.99% accuracy - it was fake
2. ⚠️ Current quantum encoding approach does NOT work
3. ✅ Use this as a framework/reference implementation only
4. ✅ Test classical baseline first (raw features, no encoding)

**To fix the pipeline**:
1. Run classical DTW on original 60-D skeleton features (no PCA, no encoding)
2. Compare to encoded+projected results to quantify information loss
3. Investigate why encoding destroys class structure
4. Try alternative quantum feature maps or skip encoding entirely

---

## 📧 Contact & Support

For questions or issues:
1. Check this README thoroughly
2. Review code comments in relevant files
3. Test with simple examples first
4. Verify dependencies are installed correctly
5. **READ THE CRITICAL FINDINGS SECTION** at the top of this file

---

**Last Updated**: November 7, 2025

**Project Status**: ⚠️ ENCODING FAILURE - Pipeline produces random predictions (3-5% accuracy)

**Previous Claim**: "Production Ready - 82.99% accuracy" (FAKE - never achieved)

**Actual Status**: Framework complete, but encoding approach fundamentally broken

**License**: [Add your license here]

