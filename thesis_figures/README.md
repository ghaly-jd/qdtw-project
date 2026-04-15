# Thesis Figures Directory

**Generated:** January 2, 2026  
**Total Figures:** 15 (30 files: PNG + PDF)  
**Resolution:** 300 DPI (publication quality)

---

## 📊 Figure Inventory

### 1. Pipeline Architecture (`01_pipeline_architecture`)
- **Description:** Complete VQD-DTW pipeline with 7 stages
- **Use in Thesis:** Introduction / Methodology overview
- **Format:** Flowchart with annotations
- **Key Elements:** Raw data → Normalization → Pre-reduction → VQD → Projection → DTW → Classification

### 2. Pre-Reduction Optimization (`02_prereduction_optimization_4panel`)
- **Description:** 4-panel comprehensive analysis (accuracy, gap, variance, significance)
- **Use in Thesis:** Results Chapter - KEY FINDING
- **Key Result:** 20D optimal with +5.7% advantage (p < 0.001)
- **Subplots:**
  - (A) Accuracy vs pre-dimension (VQD vs PCA)
  - (B) VQD advantage (U-shaped curve)
  - (C) Variance retained
  - (D) Statistical significance (p-values)

### 3. K-Sweep Results (`03_k_sweep_results`)
- **Description:** Optimal target dimensionality analysis
- **Use in Thesis:** Results Chapter - RQ3 answer
- **Key Result:** k=8 optimal with +5.0% advantage
- **Subplots:**
  - (A) Accuracy vs k (VQD vs PCA)
  - (B) VQD advantage per k value

### 4. Per-Class Performance (`04_per_class_performance`)
- **Description:** Accuracy breakdown for 20 action classes
- **Use in Thesis:** Results Chapter - detailed analysis
- **Key Insight:** VQD excels on dynamic actions (+13.3% on arm waves, kicks)
- **Format:** Grouped bar chart

### 5. VQD Circuit Diagram (`05_vqd_circuit_diagram`)
- **Description:** Hardware-efficient quantum circuit visualization
- **Use in Thesis:** Methodology Chapter - VQD explanation
- **Elements:** 4 qubits, 2 layers, RY gates, CNOT entanglement (alternating pattern)
- **Format:** Circuit schematic

### 6. Ablation Study (`06_ablation_study`)
- **Description:** Component necessity validation
- **Use in Thesis:** Results Chapter - design validation
- **Configurations Tested:**
  - Full pipeline: 83.4%
  - No pre-reduction: 77.7% (-5.7%)
  - No per-seq centering: 80.1% (-3.3%)
  - Classical PCA: 77.7% (baseline)
  - Raw features: 72.0% (-11.4%)

### 7. VQD Convergence (`07_vqd_convergence`)
- **Description:** Training convergence for 4 eigenvectors
- **Use in Thesis:** Implementation Chapter - optimization details
- **Shows:** Loss curves, convergence iterations (~50-150 iter), final loss values

### 8. Methods Comparison (`08_methods_comparison`)
- **Description:** VQD vs other dimensionality reduction methods
- **Use in Thesis:** Results Chapter - positioning against baselines
- **Methods:** Raw 60D, PCA, Kernel PCA, Direct VQD, VQD+PreRed
- **Metrics:** Accuracy and training time

### 9. Eigenvalue Spectrum (`09_eigenvalue_spectrum`)
- **Description:** PCA eigenvalue decay and cumulative variance
- **Use in Thesis:** Background / Methodology - motivation for pre-reduction
- **Shows:** Why 20D captures 99% variance, rapid decay after ~25 components

### 10. DTW Alignment Example (`10_dtw_alignment_example`)
- **Description:** DTW algorithm visualization with cost matrix
- **Use in Thesis:** Methodology Chapter - DTW explanation
- **Shows:** Two sequences (different lengths), cost matrix, optimal warping path

### 11. Confusion Matrix (`11_confusion_matrix`)
- **Description:** 20×20 confusion matrix for VQD-DTW
- **Use in Thesis:** Results Chapter - detailed performance
- **Shows:** Per-class accuracy, common confusions, overall pattern

### 12. Computational Complexity (`12_computational_complexity`)
- **Description:** Time and memory scaling analysis
- **Use in Thesis:** Implementation Chapter - scalability
- **Metrics:** Training time vs dimensionality, memory vs qubits

### 13. Failed Experiments Summary (`13_failed_experiments_summary`)
- **Description:** Gap from optimal for 9 failed approaches
- **Use in Thesis:** Discussion Chapter - design space exploration
- **Shows:** What didn't work and magnitude of performance loss

### 14. Future Work Roadmap (`14_future_work_roadmap`)
- **Description:** Timeline for future research directions
- **Use in Thesis:** Conclusion Chapter - future work section
- **Milestones:** 6mo to 3+ years with priorities

### 15. Summary Statistics Table (`15_summary_statistics_table`)
- **Description:** All key metrics in table format
- **Use in Thesis:** Results Chapter introduction / Abstract
- **Metrics:** Accuracies, gaps, timing, configurations, validation details

---

## 🎨 Color Scheme

Consistent across all figures:

```python
VQD:      #2E86AB (Blue)
PCA:      #A23B72 (Purple)
Gap:      #F18F01 (Orange)
Variance: #06A77D (Green)
Error:    #C73E1D (Red)
```

---

## 📐 Figure Specifications

**Resolution:** 300 DPI (both PNG and PDF)  
**Font:** Serif (publication quality)  
**Font sizes:**
- Title: 13pt
- Axis labels: 11pt
- Tick labels: 9pt
- Legend: 9pt

**Dimensions:**
- Single plots: 12×5 inches
- Multi-panel (2×1): 12×10 inches
- Multi-panel (2×2): 12×10 inches
- Wide diagrams: 14×8 inches

---

## 📝 LaTeX Integration

### Basic Figure Template

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{thesis_figures/01_pipeline_architecture.pdf}
    \caption{VQD-DTW pipeline architecture showing 7 sequential stages from raw skeletal data to classification.}
    \label{fig:pipeline}
\end{figure}
```

### Multi-Panel Figure

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{thesis_figures/02_prereduction_optimization_4panel.pdf}
    \caption{Pre-reduction optimization analysis. (A) Accuracy vs pre-dimension for VQD and classical PCA. (B) VQD advantage showing U-shaped curve with optimum at 20D. (C) Variance retention increasing with dimensionality. (D) Statistical significance with p-values, showing 20D achieves p < 0.001.}
    \label{fig:prereduction}
\end{figure}
```

### Side-by-Side Figures

```latex
\begin{figure}[htbp]
    \centering
    \begin{subfigure}{0.48\textwidth}
        \includegraphics[width=\textwidth]{thesis_figures/03_k_sweep_results.pdf}
        \caption{K-sweep results}
        \label{fig:ksweep}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.48\textwidth}
        \includegraphics[width=\textwidth]{thesis_figures/06_ablation_study.pdf}
        \caption{Ablation study}
        \label{fig:ablation}
    \end{subfigure}
    \caption{Experimental validation. (a) Target dimensionality optimization showing k=8 optimal. (b) Component necessity via ablation studies.}
    \label{fig:validation}
\end{figure}
```

### Full-Width Figure

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{thesis_figures/04_per_class_performance.pdf}
    \caption{Per-class performance comparison between VQD and classical PCA across 20 action classes. VQD shows largest improvements on dynamic actions involving large arm and leg motions.}
    \label{fig:perclass}
\end{figure*}
```

---

## 🔧 Regeneration

To regenerate all figures:

```bash
cd /path/to/qdtw_project
python generate_thesis_figures.py
```

**Time:** ~30 seconds  
**Dependencies:** matplotlib, seaborn, numpy, scipy, pandas

---

## 📊 Figure Usage Guide

### Chapter 1: Introduction
- **01_pipeline_architecture** - Overview of approach
- **15_summary_statistics_table** - Key results preview

### Chapter 2: Background
- **09_eigenvalue_spectrum** - PCA background
- **10_dtw_alignment_example** - DTW explanation

### Chapter 3: Methodology
- **05_vqd_circuit_diagram** - VQD algorithm visualization
- **01_pipeline_architecture** - Complete pipeline

### Chapter 4: Implementation
- **07_vqd_convergence** - Training details
- **12_computational_complexity** - Scalability analysis

### Chapter 5: Results
- **02_prereduction_optimization_4panel** - Main finding (20D optimal)
- **03_k_sweep_results** - Target dimension optimization
- **04_per_class_performance** - Detailed breakdown
- **06_ablation_study** - Component validation
- **08_methods_comparison** - Baseline comparisons
- **11_confusion_matrix** - Classification details

### Chapter 6: Discussion
- **13_failed_experiments_summary** - What didn't work

### Chapter 7: Conclusion
- **14_future_work_roadmap** - Future directions
- **15_summary_statistics_table** - Final summary

---

## 📋 Checklist for Thesis

- [x] All 15 figures generated
- [x] Both PNG (for preview) and PDF (for LaTeX) available
- [x] 300 DPI publication quality
- [x] Consistent color scheme
- [x] Clear labels and legends
- [x] Accessible to colorblind readers (patterns + colors)
- [ ] Captions written for each figure
- [ ] Figures referenced in text
- [ ] Figure numbers consistent with thesis structure

---

## 🎓 Tips for Thesis Writing

1. **Always use PDF in LaTeX** (vector graphics, scalable)
2. **Reference figures in text** before showing them
3. **Explain all panels** in multi-panel figures
4. **Use \autoref{fig:label}** for automatic "Figure X" references
5. **Place figures at top of page** `[t]` for consistency
6. **Caption structure:** What is shown, what it means, key takeaway
7. **Accessibility:** Mention patterns in caption if colors important

---

## 📞 Support

Issues or regeneration needed? Check:
1. `generate_thesis_figures.py` - Main script
2. `vqd_proper_experiments/results/*.json` - Data sources
3. Dependencies: `pip install matplotlib seaborn scipy pandas`

---

**Status:** ✓ Complete - All 15 figures generated and ready for thesis integration
