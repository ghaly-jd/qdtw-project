# 14 - Visualization and Figures

**File:** `14_VISUALIZATION.md`  
**Purpose:** All publication-ready figures with explanations  
**For Thesis:** Results chapter - visual evidence

---

## 14.1 Figure Inventory

### 14.1.1 Main Results (4-panel figure)

**File:** `figures/prereduction_analysis/prereduction_sweep.png` (300 DPI)

**Panels:**
- **(A) Accuracy vs Pre-Dimension:** Shows VQD vs PCA across 6 pre-dims
- **(B) VQD Advantage:** Gap plot highlighting 20D peak (+5.7%)
- **(C) Variance Retained:** Cumulative variance curve (99% at 20D)
- **(D) Per-Class Comparison:** Heatmap of 20 action classes

**Key insight:** U-shaped curve with 20D optimal.

---

### 14.1.2 K-Sweep Results

**File:** `figures/k_sweep_results.png`

Shows accuracy for k ∈ {6,8,10,12}:
- VQD (blue line): Peaks at k=8 (82.7%)
- PCA (red line): Flat around 77% (no clear trend)
- Error bars: ±1 std across 5 seeds

**Key insight:** k=8 optimal for VQD-DTW.

---

### 14.1.3 Eigenvalue Spectrum

**File:** `figures/eigenvalue_spectrum.png`

**Two subplots:**
- **Scree plot:** Log-scale eigenvalues (sharp drop after 20)
- **Cumulative variance:** Reaches 99% at 20D

**Key insight:** 20D captures signal, rest is noise.

---

### 14.1.4 t-SNE Visualization

**File:** `figures/prereduction_tsne.png`

2D t-SNE projection of 20D pre-reduced features, colored by action class.

**Key insight:** Some class separation visible, but not perfect (VQD improves).

---

### 14.1.5 Projection Stages

**File:** `figures/projection_stages.png`

**4-panel trajectory visualization:**
- Raw 60D (first 2 features)
- Normalized 60D
- Pre-reduced 20D
- VQD 8D ★

Shows how features evolve through pipeline.

---

### 14.1.6 Confusion Matrix

**File:** `figures/confusion_matrix_vqd.png`

20×20 heatmap showing predicted vs true labels.

**Key insight:** Strong diagonal (correct predictions), few off-diagonal errors.

---

## 14.2 LaTeX Tables

### 14.2.1 Pre-Reduction Results

**File:** `figures/prereduction_analysis/prereduction_table.tex`

```latex
\begin{table}[ht]
\centering
\caption{Pre-Reduction Optimization Results}
\begin{tabular}{ccccccc}
\toprule
Pre-Dim & Qubits & Var. & VQD Acc. & PCA Acc. & Gap & $p$ \\
\midrule
8  & 3 & 94.2\% & $77.2 \pm 0.8$ & $77.2 \pm 0.8$ & $+0.0$ & 1.000 \\
\textbf{20} & \textbf{5} & \textbf{99.0\%} & $\mathbf{83.4 \pm 0.7}$ & $77.7 \pm 1.0$ & $\mathbf{+5.7}$ & $<0.001^{***}$ \\
32 & 6 & 99.6\% & $79.3 \pm 1.3$ & $77.5 \pm 1.2$ & $+1.8$ & 0.178 \\
\bottomrule
\end{tabular}
\end{table}
```

### 14.2.2 K-Sweep Results

**File:** `figures/k_sweep_table.tex`

Similar format for k ∈ {6,8,10,12}.

---

## 14.3 Figure Generation Code

### 14.3.1 Pre-Reduction Sweep

```python
import matplotlib.pyplot as plt
import numpy as np

# Load results
results = json.load(open('results/optimal_prereduction_results.json'))

# Extract data
pre_dims = [8, 12, 16, 20, 24, 32]
vqd_accs = [...]  # Extract from results
pca_accs = [...]
gaps = [...]
variances = [...]

# Create 4-panel figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (A) Accuracy
axes[0,0].plot(pre_dims, vqd_accs, 'o-', label='VQD', linewidth=2)
axes[0,0].plot(pre_dims, pca_accs, 's--', label='PCA', linewidth=2)
axes[0,0].axvline(20, color='red', linestyle=':', alpha=0.5, label='Optimal (20D)')
axes[0,0].set_xlabel('Pre-Reduction Dimension')
axes[0,0].set_ylabel('Accuracy (%)')
axes[0,0].set_title('(A) Accuracy vs Pre-Dimension')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# (B) Gap
axes[0,1].plot(pre_dims, gaps, 'o-', color='green', linewidth=2)
axes[0,1].axvline(20, color='red', linestyle=':', alpha=0.5)
axes[0,1].axhline(0, color='black', linestyle='-', alpha=0.3)
axes[0,1].set_xlabel('Pre-Reduction Dimension')
axes[0,1].set_ylabel('VQD Advantage (%)')
axes[0,1].set_title('(B) VQD - PCA Gap')
axes[0,1].grid(True, alpha=0.3)

# (C) Variance
axes[1,0].plot(pre_dims, variances, 'o-', color='purple', linewidth=2)
axes[1,0].axvline(20, color='red', linestyle=':', alpha=0.5)
axes[1,0].axhline(99, color='orange', linestyle='--', alpha=0.5, label='99% threshold')
axes[1,0].set_xlabel('Pre-Reduction Dimension')
axes[1,0].set_ylabel('Variance Retained (%)')
axes[1,0].set_title('(C) Variance vs Pre-Dimension')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# (D) Per-class (heatmap)
per_class_data = np.array(...)  # 20 classes × 6 pre-dims
im = axes[1,1].imshow(per_class_data, aspect='auto', cmap='RdYlGn', vmin=60, vmax=100)
axes[1,1].set_xlabel('Pre-Dimension Index')
axes[1,1].set_ylabel('Action Class')
axes[1,1].set_title('(D) Per-Class Performance')
plt.colorbar(im, ax=axes[1,1], label='Accuracy (%)')

plt.tight_layout()
plt.savefig('figures/prereduction_analysis/prereduction_sweep.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/prereduction_analysis/prereduction_sweep.pdf', bbox_inches='tight')
```

---

## 14.4 All Figures Summary

| Figure | File | Purpose | For Thesis |
|--------|------|---------|------------|
| Pre-reduction sweep | `prereduction_sweep.png` | Main result (20D optimal) | Results (Figure 1) |
| K-sweep | `k_sweep_results.png` | Target k optimization | Results (Figure 2) |
| Eigenvalue spectrum | `eigenvalue_spectrum.png` | Signal/noise separation | Methods (Figure 3) |
| t-SNE | `prereduction_tsne.png` | Feature space visualization | Results (Figure 4) |
| Projection stages | `projection_stages.png` | Pipeline visualization | Methods (Figure 5) |
| Confusion matrix | `confusion_matrix_vqd.png` | Per-class performance | Results (Figure 6) |
| Normalization effect | `normalization_effect.png` | Preprocessing impact | Methods (Figure 7) |
| Sequence lengths | `sequence_lengths.png` | Dataset statistics | Background (Figure 8) |

**Total: 8 high-quality figures** (300 DPI, PNG + PDF)

---

## 14.5 Figure Quality Guidelines

**All figures follow:**
- ✅ 300 DPI resolution (publication-quality)
- ✅ Vector format available (PDF)
- ✅ Clear labels and legends
- ✅ Consistent color scheme
- ✅ Grid for readability
- ✅ Annotations for key points

**Fonts:**
- Title: 14pt bold
- Axis labels: 12pt
- Tick labels: 10pt
- Legends: 10pt

**Colors:**
- VQD: Blue (#1f77b4)
- PCA: Red (#d62728)
- Optimal point: Red dashed vertical line
- Grid: Gray (alpha=0.3)

---

## 14.6 Key Takeaways

**Figures tell the story:**

1. ✅ **Pre-reduction sweep:** U-shaped curve, 20D optimal
2. ✅ **K-sweep:** Inverted-U, k=8 optimal
3. ✅ **Eigenvalue spectrum:** Clear signal/noise separation at 20D
4. ✅ **Confusion matrix:** Strong diagonal, few errors
5. ✅ **All figures publication-ready** (300 DPI, PDF available)

**For thesis defense:**
- Can show visual evidence for all claims
- Figures support quantitative results
- Professional presentation quality

---

**Next:** [15_FRAMEWORK.md](./15_FRAMEWORK.md) →

---

**Navigation:**
- [← 13_ABLATION_STUDIES.md](./13_ABLATION_STUDIES.md)
- [→ 15_FRAMEWORK.md](./15_FRAMEWORK.md)
- [↑ Index](./README.md)
