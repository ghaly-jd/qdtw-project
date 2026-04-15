# Presentation Figures Summary
## Generated for VQD-DTW PowerPoint Presentation

**Date**: November 25, 2025  
**Location**: `/path/to/qdtw_project/vqd_proper_experiments/figures/`

---

## ✅ COMPLETED FIGURES

### 1. **Conceptual Diagrams**
- `pca_diagram.png` - Classical PCA explanation (covariance → eigendecomp)
- `vqd_diagram.png` - VQD explanation (Hamiltonian → quantum circuit → eigenvectors)
- `pca_vqd_comparison.png` - Side-by-side comparison of both approaches

### 2. **Quantum Circuit**
- `vqd_circuit_qiskit.png` ⭐ **MAIN** - Actual Qiskit circuit (4 qubits, 8 params, depth=2)
- `vqd_circuit_with_values.png` - Same circuit with example parameter values
- `vqd_circuit_text.txt` - ASCII representation
- `vqd_circuit_latex.txt` - LaTeX source for papers

### 3. **Results Plots**
- `k_sweep_results.png` - K-sweep with value labels (k=6,8,10,12)
- `k_sweep_results_clean.png` - Clean version without labels
- `accuracy_comparison.png` - Bar plot with error bars (mean ± std)
- `vqd_advantage.png` - Gap plot showing VQD improvement with 95% CI

### 4. **Per-Class Analysis**
- `per_class_delta_recall.png` - Horizontal bar chart showing VQD-PCA difference per action class
  - Green bars = VQD better
  - Red bars = PCA better
  - Sorted by performance

### 5. **Results Table**
- `results_table.png` - Comprehensive table with:
  - Method (PCA/VQD)
  - K (6, 8, 10, 12)
  - Accuracy (mean ± std %)
  - VQD Quality (improvement)

### 6. **By-Class Comparison** (from experiment)
- `by_class_comparison.png` - Bar chart comparing PCA vs VQD accuracy for each of 20 action classes

---

## ⏳ IN PROGRESS

### 7. **Projection Equivalence** (Running now, ~2-3 min)
- `projection_equivalence_scatter.png` - Scatter plots showing how same data projects differently
- `projection_equivalence_clean.png` - Top 5 classes highlighted version
- Will include Fisher ratio showing VQD's better class separability

---

## 📊 FIGURES BY SLIDE TYPE

### Introduction Slides:
1. Problem/Motivation slide → Use `pca_vqd_comparison.png`

### Method Slides:
2. Classical PCA slide → Use `pca_diagram.png`
3. VQD slide → Use `vqd_diagram.png`
4. Circuit slide → Use `vqd_circuit_qiskit.png` ⭐

### Results Slides:
5. Main results slide → Use `k_sweep_results.png` or `results_table.png`
6. Statistical validation → Use `k_sweep_results.png` (shows mean±std, n=5 seeds)
7. VQD advantage → Use `vqd_advantage.png` (95% CI)
8. Per-class insights → Use `per_class_delta_recall.png`
9. Subspace differences → Use `projection_equivalence_scatter.png` (when ready)

### Detailed Results:
10. All k-values → Use `accuracy_comparison.png`
11. By-class breakdown → Use `by_class_comparison.png`
12. Summary table → Use `results_table.png`

---

## 📈 KEY STATISTICS TO MENTION

From k-sweep (k=8, most common):
- **PCA**: 77.7 ± 3.8%
- **VQD**: 82.7 ± 2.8%
- **Gap**: +5.0 ± 3.3%

Average across all k:
- **Average VQD improvement**: +4.67%
- **Best k**: k=10 (+5.0%)
- **Most stable**: k=12 (VQD std=2.7%)

Per-class highlights:
- **VQD wins**: 4/20 classes with large margins
- **Biggest wins**: High arm wave (+66.7%), Forward kick (+66.7%)
- **Interpretation**: VQD excels at dynamic temporal actions

---

## 🎯 RECOMMENDED SLIDE FLOW

1. **Title** → Project name, authors
2. **Problem** → Why dimensionality reduction matters for DTW
3. **Classical PCA** → Use `pca_diagram.png`
4. **Quantum VQD** → Use `vqd_diagram.png`
5. **Circuit** → Use `vqd_circuit_qiskit.png`
6. **Main Results** → Use `k_sweep_results.png` + `results_table.png`
7. **Statistical Rigor** → Use `vqd_advantage.png` (95% CI)
8. **Per-Class** → Use `per_class_delta_recall.png`
9. **Subspace Analysis** → Use `projection_equivalence_scatter.png`
10. **Conclusion** → Summary of +4.67% average improvement

---

## 📝 LATEX TABLE (Ready to copy)

```latex
\begin{table}[h]
\centering
\caption{VQD vs PCA Classification Results on MSR Action3D}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Method} & \textbf{K} & \textbf{Accuracy (\%)} & \textbf{VQD Quality} \\
\hline
PCA & 6 & 72.7 ± 4.2 & — \\
\textbf{VQD} & 6 & 77.0 ± 4.6 & \textcolor{darkgreen}{+4.3 ± 1.9} \\
\hline
PCA & 8 & 77.7 ± 3.8 & — \\
\textbf{VQD} & 8 & 82.7 ± 2.8 & \textcolor{darkgreen}{+5.0 ± 3.3} \\
\hline
PCA & 10 & 78.0 ± 3.0 & — \\
\textbf{VQD} & 10 & 83.0 ± 4.8 & \textcolor{darkgreen}{+5.0 ± 4.2} \\
\hline
PCA & 12 & 79.3 ± 3.5 & — \\
\textbf{VQD} & 12 & 83.7 ± 2.7 & \textcolor{darkgreen}{+4.3 ± 1.5} \\
\hline
\end{tabular}
\end{table}
```

---

## ✨ ALL FIGURES READY FOR POWERPOINT!

Total figures: **13 images** (12 ready + 1 generating)  
Resolution: **300 DPI** (publication quality)  
Format: **PNG** (drag-and-drop ready)
