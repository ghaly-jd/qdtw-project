#!/bin/bash

# Optimal Pre-Reduction Dimensionality Experiment
# ================================================
# Tests: 60D → {8, 12, 16, 20, 24, 32}D → 8D (PCA/VQD) → DTW
#
# Configuration:
# - Pre-reduction dims: [8, 12, 16, 20, 24, 32]
# - Target k: 8 (known optimal)
# - Seeds: [42, 123, 456, 789, 2024]
# - Train: 300, Test: 60
#
# Expected runtime: ~4-6 hours
# Produces: 6 pre-dims × 5 seeds × 2 methods = 60 experiments

cd /path/to/qdtw_project/vqd_proper_experiments

echo "========================================================"
echo "  OPTIMAL PRE-REDUCTION DIMENSIONALITY EXPERIMENT"
echo "========================================================"
echo "Pipeline: 60D → {8,12,16,20,24,32}D → 8D (VQD)"
echo "Seeds: 5 (statistical validation)"
echo "Expected runtime: 4-6 hours"
echo ""
echo "Research Question:"
echo "  What is the optimal pre-reduction size?"
echo ""
echo "Expected findings:"
echo "  • Too small (8D): Information loss"
echo "  • Sweet spot (16D?): Best balance"
echo "  • Too large (32D): Noise retained"
echo "========================================================"
echo ""

# Create log directory
mkdir -p logs

# Run experiment
LOG_FILE="logs/optimal_prereduction_$(date +%Y%m%d_%H%M%S).log"

echo "Starting experiment at $(date)"
echo "Log file: $LOG_FILE"
echo ""

python experiment_optimal_prereduction.py \
    --pre-dims 8 12 16 20 24 32 \
    --k 8 \
    --seeds 42 123 456 789 2024 \
    --n-train 300 \
    --n-test 60 \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Experiment completed successfully at $(date)"
    echo "Results: results/optimal_prereduction_results.json"
    echo "Log: $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "  1. Visualize results:"
    echo "     python plot_optimal_prereduction.py"
    echo "  2. Generate thesis figures"
    echo "  3. Update EXPERIMENT_GUIDE.md"
else
    echo "✗ Experiment failed with exit code $EXIT_CODE"
    echo "Check log: $LOG_FILE"
fi
echo "========================================================"

exit $EXIT_CODE
