#!/bin/bash
# Run No Pre-Reduction Experiment
# Tests PCA and VQD directly on 60D data (no 60D→16D pre-reduction)

echo "========================================================================"
echo "NO PRE-REDUCTION EXPERIMENT"
echo "Pipeline: 60D → kD directly (NO 60D→16D step)"
echo "========================================================================"

cd /path/to/qdtw_project/vqd_proper_experiments

# Create logs directory
mkdir -p logs

# Option 1: Quick test with single seed, single k
echo ""
echo "Option 1: Quick test (seed=42, k=8 only)"
echo "Estimated time: ~10-15 minutes"
echo ""
echo "Run: python experiment_no_prereduction.py --k-values 8 --seeds 42"
echo ""

# Option 2: Full sweep with single seed
echo "Option 2: Full k-sweep (seed=42, k=[6,8,10,12])"
echo "Estimated time: ~40-60 minutes"
echo ""
echo "Run: python experiment_no_prereduction.py --k-values 6 8 10 12 --seeds 42"
echo ""

# Option 3: Full statistical validation
echo "Option 3: Full statistical validation (5 seeds × 4 k-values)"
echo "Estimated time: ~3-4 hours (VQD is slower with 6 qubits)"
echo ""
echo "Run: python experiment_no_prereduction.py --k-values 6 8 10 12 --seeds 42 123 456 789 2024"
echo ""

echo "========================================================================"
echo "Choose your option and run the command above"
echo "========================================================================"
echo ""
echo "Monitor progress: tail -f logs/no_prereduction.log"
echo ""
