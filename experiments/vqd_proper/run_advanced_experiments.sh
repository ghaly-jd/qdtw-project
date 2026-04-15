#!/bin/bash

# Master script to run all three advanced experiments
# Author: VQD-DTW Research Team
# Date: November 24, 2025

echo "========================================================================"
echo "ADVANCED EXPERIMENTS SUITE"
echo "========================================================================"
echo ""
echo "This will run three experiments:"
echo "  1. K-Sweep with Confidence Intervals (k=6,8,10,12, 5 seeds)"
echo "  2. Whitening Toggle (U vs U Λ^{-1/2})"
echo "  3. By-Class Analysis (which actions benefit from VQD)"
echo ""
echo "Estimated total time: ~2-3 hours"
echo "========================================================================"
echo ""

# Change to experiment directory
cd "$(dirname "$0")"

# Create output directories
mkdir -p results
mkdir -p figures
mkdir -p logs

# ========== EXPERIMENT 1: K-Sweep with CIs ==========
echo ""
echo "========================================================================"
echo "EXPERIMENT 1: K-SWEEP WITH CONFIDENCE INTERVALS"
echo "========================================================================"
echo "Running 5 seeds × 4 k-values = 20 runs"
echo "Estimated time: ~1.5-2 hours"
echo ""

python experiment_k_sweep_ci.py 2>&1 | tee logs/k_sweep_ci.log

if [ $? -eq 0 ]; then
    echo "✓ Experiment 1 completed successfully!"
else
    echo "✗ Experiment 1 failed!"
    exit 1
fi

# ========== EXPERIMENT 2: Whitening Toggle ==========
echo ""
echo "========================================================================"
echo "EXPERIMENT 2: WHITENING TOGGLE"
echo "========================================================================"
echo "Testing U vs U Λ^{-1/2} for k=6,8,10,12"
echo "Estimated time: ~30 minutes"
echo ""

python experiment_whitening.py 2>&1 | tee logs/whitening.log

if [ $? -eq 0 ]; then
    echo "✓ Experiment 2 completed successfully!"
else
    echo "✗ Experiment 2 failed!"
    exit 1
fi

# ========== EXPERIMENT 3: By-Class Analysis ==========
echo ""
echo "========================================================================"
echo "EXPERIMENT 3: BY-CLASS ANALYSIS"
echo "========================================================================"
echo "Computing per-class VQD vs PCA for k=8"
echo "Estimated time: ~15 minutes"
echo ""

python experiment_by_class.py 2>&1 | tee logs/by_class.log

if [ $? -eq 0 ]; then
    echo "✓ Experiment 3 completed successfully!"
else
    echo "✗ Experiment 3 failed!"
    exit 1
fi

# ========== Summary ==========
echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - results/k_sweep_ci_results.json"
echo "  - results/whitening_results.json"
echo "  - results/by_class_results.json"
echo ""
echo "Logs saved to:"
echo "  - logs/k_sweep_ci.log"
echo "  - logs/whitening.log"
echo "  - logs/by_class.log"
echo ""
echo "Figures saved to:"
echo "  - figures/by_class_comparison.png"
echo ""
echo "Next step: Analyze results and create summary report"
echo "========================================================================"
