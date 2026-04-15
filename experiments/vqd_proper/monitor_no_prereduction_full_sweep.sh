#!/bin/bash

# Monitor No Pre-Reduction Full K-Sweep Experiment
# Shows progress and key metrics

cd /path/to/qdtw_project/vqd_proper_experiments

echo "=================================================="
echo "  NO PRE-REDUCTION K-SWEEP MONITOR"
echo "=================================================="
echo ""

# Check if experiment is running
if pgrep -f "experiment_no_prereduction.py" > /dev/null; then
    echo "✓ Experiment is RUNNING"
    echo ""
else
    echo "✗ Experiment is NOT running"
    echo ""
fi

# Show latest log
LATEST_LOG=$(ls -t logs/no_prereduction_full_sweep_*.log 2>/dev/null | head -1)

if [ -n "$LATEST_LOG" ]; then
    echo "Latest log: $LATEST_LOG"
    echo ""
    
    # Show progress
    echo "--- PROGRESS ---"
    grep -E "Running: seed=|PCA Accuracy:|VQD Accuracy:|VQD training time:" "$LATEST_LOG" | tail -20
    echo ""
    
    # Show VQD quality metrics
    echo "--- VQD QUALITY METRICS ---"
    grep -E "VQD orthogonality error:|VQD max principal angle:" "$LATEST_LOG" | tail -10
    echo ""
fi

# Show results if available
if [ -f "results/no_prereduction_results.json" ]; then
    echo "--- CURRENT RESULTS (from JSON) ---"
    python -c "
import json
with open('results/no_prereduction_results.json') as f:
    data = json.load(f)
    
print('Completed runs:')
for seed_key, seed_data in data['by_seed'].items():
    seed = seed_key.split('_')[1]
    k_values = sorted([int(k.split('_')[1]) for k in seed_data.keys()])
    print(f'  Seed {seed}: k = {k_values}')

print('\nAggregated results (available k values):')
if data['aggregated']:
    print('  K     PCA          VQD          Gap')
    print('  ' + '-'*50)
    for k in sorted([int(k) for k in data['aggregated'].keys()]):
        agg = data['aggregated'][str(k)]
        if agg['pca']['mean'] is not None:
            pca_m = agg['pca']['mean'] * 100
            pca_s = agg['pca']['std'] * 100
            
            if agg['vqd']['mean'] is not None:
                vqd_m = agg['vqd']['mean'] * 100
                vqd_s = agg['vqd']['std'] * 100
                gap_m = agg['gap']['mean'] * 100
                print(f'  {k:<5} {pca_m:.1f}±{pca_s:.1f}%    {vqd_m:.1f}±{vqd_s:.1f}%    {gap_m:+.1f}%')
            else:
                print(f'  {k:<5} {pca_m:.1f}±{pca_s:.1f}%    (running...)  N/A')
else:
    print('  No aggregated results yet')
" 2>/dev/null
    echo ""
fi

echo "=================================================="
echo "Refresh: watch -n 30 ./monitor_no_prereduction_full_sweep.sh"
echo "=================================================="
