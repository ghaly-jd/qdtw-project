# Project Cleanup Plan

## Overview
This document outlines the cleanup and organization strategy for the QDTW project.

---

## Directory Structure (After Cleanup)

```
qdtw_project/
├── README.md                      # Main documentation (KEEP, UPDATE)
├── PIPELINE_GUIDE.md              # Comprehensive pipeline guide (KEEP)
├── DEBUGGING_REPORT.md            # Technical debugging analysis (KEEP)
├── .gitignore                     # Git ignore file (KEEP)
│
├── data/                          # Processed data artifacts (KEEP)
│   ├── frame_bank_std.npy        # Standardized frame bank (KEEP - used)
│   ├── features.npy               # DELETE - old raw features
│   ├── features_pca2.npy          # DELETE - old PCA features
│   ├── frame_bank.npy             # DELETE - old non-standardized
│   └── frame_bank_test.npy        # DELETE - old test bank
│
├── msr_action_data/               # Raw skeleton dataset (KEEP)
│   └── *.txt                      # All skeleton files (KEEP)
│
├── features/                      # Feature encoding module (KEEP)
│   ├── __init__.py
│   └── amplitude_encoding.py      # Z-score standardization (KEEP)
│
├── quantum/                       # PCA implementations (KEEP)
│   ├── __init__.py
│   ├── classical_pca.py           # Classical SVD PCA (KEEP)
│   └── qpca.py                    # Quantum PCA (KEEP)
│
├── dtw/                           # DTW module (KEEP)
│   ├── __init__.py
│   └── dtw_runner.py              # DTW distance functions (KEEP)
│
├── eval/                          # Evaluation module (KEEP)
│   ├── __init__.py
│   ├── ablations.py               # Ablation experiments (KEEP)
│   ├── aggregate.py               # Result aggregation (KEEP)
│   └── plotting.py                # Visualization (KEEP)
│
├── scripts/                       # Pipeline execution scripts (KEEP, ORGANIZE)
│   ├── __init__.py
│   ├── build_frame_bank.py        # Stage 1: Frame extraction (KEEP)
│   ├── project_sequences.py       # Stage 3: Projection (KEEP)
│   ├── run_ablations.py           # Stage 4: Evaluation (KEEP)
│   ├── run_dtw_raw.py             # Baseline test (KEEP)
│   ├── sanity_checks.py           # Validation tests (KEEP)
│   ├── create_label_metadata.py   # REVIEW - may be obsolete
│   ├── demo_dtw.py                # KEEP - demo script
│   ├── make_figures.py            # KEEP - visualization
│   └── run_dtw_subspace.py        # REVIEW - may be obsolete
│
├── tests/                         # Unit tests (KEEP)
│   ├── __init__.py
│   ├── test_*.py                  # All test files (KEEP)
│
├── results/                       # Model outputs (KEEP, CLEAN)
│   ├── Uc_k8_std.npz              # Classical PCA (KEEP - used)
│   ├── Uq_k8_std.npz              # Quantum PCA (KEEP - used)
│   ├── ablations.csv              # Results (KEEP)
│   └── subspace_std/              # Projected sequences (KEEP)
│       ├── Uc/k8/
│       │   ├── train/             # 454 sequences (KEEP)
│       │   └── test/              # 113 sequences (KEEP)
│       └── Uq/k8/
│           ├── train/             # 454 sequences (KEEP)
│           └── test/              # 113 sequences (KEEP)
│
├── figures/                       # Generated plots (KEEP)
│   ├── README.md
│   └── *.png                      # All figures (KEEP)
│
├── docs/                          # Documentation (NEW - ORGANIZE)
│   ├── DEBUGGING_REPORT.md        # MOVE from root
│   ├── PIPELINE_GUIDE.md          # MOVE from root
│   ├── SOLUTION_SUMMARY.md        # MOVE from root
│   └── archive/                   # Old status documents
│       ├── ABLATIONS_COMPLETE.md
│       ├── ABLATIONS_SUMMARY.md
│       ├── DTW_SUMMARY.md
│       ├── EVAL_SUMMARY.md
│       ├── PIPELINE_STATUS.md
│       └── PROJECT_COMPLETE.md
│
└── archive/                       # OLD/UNUSED CODE (NEW)
    ├── benchmark.py               # MOVE - old benchmark
    ├── demo_amplitude_encoding.py # MOVE - old demo
    ├── generate_all_visuals.py    # MOVE - may be useful
    ├── gpu_classical_classifier.py# MOVE - GPU version
    ├── gpu_classical_dtw.py       # MOVE - GPU version
    ├── grover_benchmark.py        # MOVE - Grover test
    ├── q_classifier.py            # MOVE - old classifier
    ├── qdtw.py                    # MOVE - old main
    ├── verify_quantum.py          # MOVE - verification
    ├── quantum_src/               # MOVE - old quantum code
    ├── src/                       # MOVE - old source
    └── vizualizations/            # MOVE - duplicate (typo)

---

## Files to DELETE

### Root Level:
- `=1.0` - Strange file, delete
- `.DS_Store` - macOS metadata
- `__pycache__/` - Python cache (git should ignore)

### data/:
- `features.npy` - Old raw features (pre-standardization)
- `features_pca2.npy` - Old PCA (pre-fix)
- `frame_bank.npy` - Old frame bank (L2-norm version)
- `frame_bank_test.npy` - Old test bank

### results/:
- Any old `Uc_k8.npz` or `Uq_k8.npz` (non-_std versions)
- `subspace/` directory (old projections, not subspace_std)

### Duplicates:
- `MSR-Action-Recognition/` - Submodule, likely unused
- `vizualizations/` - Typo duplicate of `visualizations/`

---

## Files to MOVE

### To `docs/`:
- `DEBUGGING_REPORT.md`
- `PIPELINE_GUIDE.md`
- `SOLUTION_SUMMARY.md`
- `ABLATIONS_COMPLETE.md` → `docs/archive/`
- `ABLATIONS_SUMMARY.md` → `docs/archive/`
- `DTW_SUMMARY.md` → `docs/archive/`
- `EVAL_SUMMARY.md` → `docs/archive/`
- `PIPELINE_STATUS.md` → `docs/archive/`
- `PROJECT_COMPLETE.md` → `docs/archive/`

### To `archive/`:
- `benchmark.py`
- `demo_amplitude_encoding.py`
- `generate_all_visuals.py`
- `gpu_classical_classifier.py`
- `gpu_classical_dtw.py`
- `grover_benchmark.py`
- `q_classifier.py`
- `qdtw.py`
- `verify_quantum.py`
- `quantum_src/` directory
- `src/` directory

---

## Files to KEEP (Current Location)

### Core Modules:
- `features/` - Encoding functions
- `quantum/` - PCA implementations
- `dtw/` - DTW algorithms
- `eval/` - Evaluation framework
- `scripts/` - Pipeline scripts
- `tests/` - Unit tests

### Data:
- `msr_action_data/` - Raw dataset
- `data/frame_bank_std.npy` - Active frame bank
- `results/Uc_k8_std.npz` - Active classical PCA
- `results/Uq_k8_std.npz` - Active quantum PCA
- `results/subspace_std/` - Active projections
- `results/ablations.csv` - Latest results

### Documentation:
- `README.md` - Main entry point
- `figures/` - Generated plots
- `visualizations/` - Visual assets

---

## Final Clean Structure

```
qdtw_project/
├── README.md                     # Updated main docs
├── .gitignore
│
├── data/
│   └── frame_bank_std.npy
│
├── msr_action_data/              # 567 skeleton files
│
├── features/                     # Core encoding
│   ├── __init__.py
│   └── amplitude_encoding.py
│
├── quantum/                      # PCA implementations
│   ├── __init__.py
│   ├── classical_pca.py
│   └── qpca.py
│
├── dtw/                          # DTW module
│   ├── __init__.py
│   └── dtw_runner.py
│
├── eval/                         # Evaluation
│   ├── __init__.py
│   ├── ablations.py
│   ├── aggregate.py
│   └── plotting.py
│
├── scripts/                      # Pipeline runners
│   ├── build_frame_bank.py       # Stage 1
│   ├── project_sequences.py      # Stage 3
│   ├── run_ablations.py          # Stage 4
│   ├── run_dtw_raw.py            # Baseline
│   ├── sanity_checks.py          # Validation
│   └── ...
│
├── tests/                        # Unit tests
│   └── test_*.py
│
├── results/                      # Outputs
│   ├── Uc_k8_std.npz
│   ├── Uq_k8_std.npz
│   ├── ablations.csv
│   └── subspace_std/
│
├── figures/                      # Generated plots
│   └── *.png
│
├── docs/                         # Documentation
│   ├── PIPELINE_GUIDE.md
│   ├── DEBUGGING_REPORT.md
│   ├── SOLUTION_SUMMARY.md
│   └── archive/                  # Old status docs
│
├── archive/                      # Old/unused code
│   ├── benchmark.py
│   ├── qdtw.py
│   ├── quantum_src/
│   └── ...
│
└── visualizations/               # Visual assets
    └── *.png
```

---

## Cleanup Commands

Will execute carefully after user confirmation!

