# VQD-DTW Technical Documentation Index

**Comprehensive Technical Report for Thesis**  
**Project:** Quantum-Inspired Dimensionality Reduction for Temporal Action Recognition  
**Date:** December 30, 2025  
**Author:** [Your Name]

---

## 📚 Documentation Structure

This documentation is organized into modular sections for easy navigation and thesis integration. Each file is self-contained and can be read independently or incorporated into your thesis chapters.

### Part 1: Foundation & Background
- **[01_INTRODUCTION.md](./01_INTRODUCTION.md)**  
  Research motivation, problem statement, and contributions

- **[02_DATASET.md](./02_DATASET.md)**  
  MSR Action3D dataset description, preprocessing, and characteristics

- **[03_PIPELINE_OVERVIEW.md](./03_PIPELINE_OVERVIEW.md)**  
  High-level architecture and workflow of the VQD-DTW system

### Part 2: Core Components (Deep Technical Details)
- **[04_DATA_LOADING.md](./04_DATA_LOADING.md)**  
  Skeleton data loading, parsing, and flattening (with code)

- **[05_NORMALIZATION_PREPROCESSING.md](./05_NORMALIZATION_PREPROCESSING.md)**  
  StandardScaler, train/test split, frame bank construction (with code)

- **[06_PRE_REDUCTION.md](./06_PRE_REDUCTION.md)**  
  Classical PCA pre-reduction (60D→20D), variance analysis (with code)

- **[07_VQD_QUANTUM_PCA.md](./07_VQD_QUANTUM_PCA.md)**  
  **DEEP DIVE:** VQD algorithm, quantum circuits, optimization (with code)

- **[08_SEQUENCE_PROJECTION.md](./08_SEQUENCE_PROJECTION.md)**  
  Per-sequence centering, projection pipeline (with code)

- **[09_DTW_CLASSIFICATION.md](./09_DTW_CLASSIFICATION.md)**  
  Dynamic Time Warping, distance metrics, 1-NN classification (with code)

### Part 3: Experiments & Results
- **[10_EXPERIMENTAL_SETUP.md](./10_EXPERIMENTAL_SETUP.md)**  
  Seeds, train/test splits, statistical validation methodology

- **[11_PREREDUCTION_OPTIMIZATION.md](./11_PREREDUCTION_OPTIMIZATION.md)**  
  Finding optimal pre-reduction size (8D→32D sweep), results

- **[12_K_SWEEP_EXPERIMENTS.md](./12_K_SWEEP_EXPERIMENTS.md)**  
  Target dimension experiments (k=6,8,10,12), VQD vs PCA comparison

- **[13_ABLATION_STUDIES.md](./13_ABLATION_STUDIES.md)**  
  Necessity of pre-reduction, per-class analysis, failure cases

- **[14_RESULTS_VISUALIZATION.md](./14_RESULTS_VISUALIZATION.md)**  
  All figures, plots, tables with explanations

### Part 4: Implementation & Technical Details
- **[15_QUANTUM_FRAMEWORK.md](./15_QUANTUM_FRAMEWORK.md)**  
  Qiskit usage, statevector simulation, circuit design

- **[16_OPTIMIZATION_DETAILS.md](./16_OPTIMIZATION_DETAILS.md)**  
  COBYLA optimizer, convergence, hyperparameters

- **[17_COMPUTATIONAL_ANALYSIS.md](./17_COMPUTATIONAL_ANALYSIS.md)**  
  Runtime, complexity, scalability analysis

### Part 5: Discussion & Conclusions
- **[18_FAILED_EXPERIMENTS.md](./18_FAILED_EXPERIMENTS.md)**  
  What didn't work and why (GOLD for thesis discussion!)

- **[19_LIMITATIONS.md](./19_LIMITATIONS.md)**  
  Current limitations and future work

- **[20_CONCLUSIONS.md](./20_CONCLUSIONS.md)**  
  Summary of findings, contributions, impact

### Appendices
- **[APPENDIX_A_CODE_REFERENCE.md](./APPENDIX_A_CODE_REFERENCE.md)**  
  Complete code listings with line-by-line explanations

- **[APPENDIX_B_HYPERPARAMETERS.md](./APPENDIX_B_HYPERPARAMETERS.md)**  
  All hyperparameter choices and justifications

- **[APPENDIX_C_MATHEMATICAL_DERIVATIONS.md](./APPENDIX_C_MATHEMATICAL_DERIVATIONS.md)**  
  Mathematical proofs and derivations

---

## 🎯 How to Use This Documentation

### For Thesis Writing
1. **Introduction Chapter:** Use files 01-03
2. **Methods Chapter:** Use files 04-09 (core technical details)
3. **Experiments Chapter:** Use files 10-14
4. **Results Chapter:** Use files 11-12, 14
5. **Discussion Chapter:** Use files 13, 18-19
6. **Conclusion Chapter:** Use file 20

### For Code Understanding
- Start with **03_PIPELINE_OVERVIEW.md** for the big picture
- Then read **07_VQD_QUANTUM_PCA.md** for the quantum algorithm
- Follow with **04-09** for component-by-component details

### For Reviewers/Defense
- **Quick overview:** Read 01, 03, 20
- **Technical details:** Read 07 (VQD), 09 (DTW), 11-12 (experiments)
- **Addressing concerns:** Read 18 (failures), 19 (limitations)

---

## 📊 Key Statistics Summary

**Dataset:** MSR Action3D
- 567 sequences, 20 action classes
- 60D skeletal features per frame
- 13-255 frames per sequence

**Pipeline:** 60D → 20D (PCA) → 8D (VQD) → DTW → Classification

**Best Results:**
- **Pre-reduction:** 20D optimal (99.0% variance, +5.7% VQD advantage)
- **Target dimension:** k=8 (82.7% VQD vs 77.7% PCA)
- **Overall improvement:** +5.0% over classical PCA

**Statistical Validation:**
- 5 random seeds: [42, 123, 456, 789, 2024]
- 300 training / 60 test sequences per split
- Mean ± standard deviation reported for all results

**Quantum Framework:**
- **Library:** Qiskit + Qiskit Aer
- **Simulator:** Statevector (exact, no shot noise)
- **Circuit:** Hardware-efficient ansatz, 4 qubits, depth=2
- **Optimizer:** COBYLA, 200 iterations
- **Entanglement:** Alternating pattern

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@thesis{yourname2025vqdtw,
  title={Quantum-Inspired Dimensionality Reduction for Temporal Action Recognition},
  author={Your Name},
  year={2025},
  school={Your University},
  type={Master's Thesis}
}
```

---

## 🔗 Navigation

**Start Reading:** [01_INTRODUCTION.md](./01_INTRODUCTION.md) →

---

**Last Updated:** December 30, 2025
