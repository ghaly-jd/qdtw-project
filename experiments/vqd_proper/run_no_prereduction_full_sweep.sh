#!/bin/bash

# Full K-Sweep Experiment WITHOUT Pre-Reduction
# ==============================================
# Tests: 60D → kD (PCA/VQD) → DTW
# No intermediate 60D→16D PCA step
#
# Configuration:
# - K values: [6, 8, 10, 12]
# - Seeds: [42, 123, 456, 789, 2024]
# - Train: 300, Test: 60
# - VQD: 6 qubits (2^6=64 >= 60D), depth=2, 200 iterations
#
# Expected runtime: ~4-6 hours (6-qubit VQD is slower than 4-qubit)

cd /path/to/qdtw_project/vqd_proper_experiments

echo "=================================================="
echo "  NO PRE-REDUCTION FULL K-SWEEP EXPERIMENT"
echo "=================================================="
echo "Pipeline: 60D → kD (NO 60D→16D step)"
echo "K values: 6, 8, 10, 12"
echo "Seeds: 42, 123, 456, 789, 2024"
echo "VQD: 6 qubits, depth=2, 200 iterations"
echo ""
echo "⚠️  WARNING: 6-qubit VQD is computationally expensive!"
echo "⚠️  Expected runtime: 4-6 hours"
echo "=================================================="
echo ""

# Create log directory
mkdir -p logs

# Run experiment
LOG_FILE="logs/no_prereduction_full_sweep_$(date +%Y%m%d_%H%M%S).log"

echo "Starting experiment at $(date)"
echo "Log file: $LOG_FILE"
echo ""

python experiment_no_prereduction.py \
    --k-values 6 8 10 12 \
    --seeds 42 123 456 789 2024 \
    --n-train 300 \
    --n-test 60 \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Experiment completed successfully at $(date)"
    echo "Results: results/no_prereduction_results.json"
    echo "Log: $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "  1. Compare with pre-reduction results:"
    echo "     python compare_prereduction_vs_no_prereduction.py"
    echo "  2. Create comparison figures"
    echo "  3. Update EXPERIMENT_GUIDE.md"
else
    echo "✗ Experiment failed with exit code $EXIT_CODE"
    echo "Check log: $LOG_FILE"
fi
echo "=================================================="

exit $EXIT_CODE
