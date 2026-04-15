#!/bin/bash

# Monitor Optimal Pre-Reduction Experiment
# Shows progress and current results

cd /path/to/qdtw_project/vqd_proper_experiments

echo "========================================================"
echo "  OPTIMAL PRE-REDUCTION EXPERIMENT MONITOR"
echo "========================================================"
echo ""

# Check if experiment is running
if pgrep -f "experiment_optimal_prereduction.py" > /dev/null; then
    echo "✓ Experiment is RUNNING"
    echo ""
else
    echo "✗ Experiment is NOT running"
    echo ""
fi

# Show latest log
LATEST_LOG=$(ls -t logs/optimal_prereduction_*.log 2>/dev/null | head -1)

if [ -n "$LATEST_LOG" ]; then
    echo "Latest log: $LATEST_LOG"
    echo ""
    
    # Show progress
    echo "--- RECENT PROGRESS ---"
    grep -E "Running: seed=|Pre-reduced|PCA Accuracy:|VQD Accuracy:|VQD - PCA Gap:" "$LATEST_LOG" | tail -30
    echo ""
fi

# Show results if available
if [ -f "results/optimal_prereduction_results.json" ]; then
    echo "--- CURRENT RESULTS ---"
    python3 << 'PYEOF'
import json
import sys

try:
    with open('results/optimal_prereduction_results.json') as f:
        data = json.load(f)
    
    print('\nCompleted configurations:')
    for seed_key, seed_data in data['by_seed'].items():
        seed = seed_key.split('_')[1]
        pre_dims = sorted([int(k.split('_')[1]) for k in seed_data.keys()])
        print(f'  Seed {seed}: pre_dims = {pre_dims}')
    
    if data['aggregated']:
        print('\nAggregated Results:')
        print('  Pre-Dim  Variance   PCA          VQD          Gap')
        print('  ' + '-'*60)
        
        for pre_dim in sorted([int(k) for k in data['aggregated'].keys()]):
            agg = data['aggregated'][str(pre_dim)]
            
            var = agg['variance_explained'] * 100
            pca_m = agg['pca']['mean'] * 100
            pca_s = agg['pca']['std'] * 100
            
            if agg['vqd']['mean'] is not None:
                vqd_m = agg['vqd']['mean'] * 100
                vqd_s = agg['vqd']['std'] * 100
                gap_m = agg['gap']['mean'] * 100
                gap_s = agg['gap']['std'] * 100
                
                advantage = '✓' if gap_m > 1.0 else '✗'
                
                print(f'  {pre_dim:<8} {var:>5.1f}%    '
                      f'{pca_m:.1f}±{pca_s:.1f}%    '
                      f'{vqd_m:.1f}±{vqd_s:.1f}%    '
                      f'{gap_m:+.1f}±{gap_s:.1f}% {advantage}')
            else:
                print(f'  {pre_dim:<8} {var:>5.1f}%    '
                      f'{pca_m:.1f}±{pca_s:.1f}%    (running...)')
        
        # Find best so far
        print()
        best_pre_dim = max([int(k) for k in data['aggregated'].keys()],
                          key=lambda d: data['aggregated'][str(d)]['gap']['mean'] or -1)
        best_gap = data['aggregated'][str(best_pre_dim)]['gap']['mean'] * 100
        print(f'  🏆 Best so far: {best_pre_dim}D (gap = {best_gap:+.2f}%)')
    else:
        print('\nNo aggregated results yet...')
        
except Exception as e:
    print(f'Error reading results: {e}', file=sys.stderr)
PYEOF
    echo ""
fi

echo "========================================================"
echo "Refresh: watch -n 30 ./monitor_optimal_prereduction.sh"
echo "========================================================"
