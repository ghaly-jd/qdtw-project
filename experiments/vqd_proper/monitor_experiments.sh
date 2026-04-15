#!/bin/bash
# Monitor running experiments

echo "==================================================================="
echo "EXPERIMENT MONITORING DASHBOARD"
echo "==================================================================="
echo ""

# Check running processes
echo "📊 Running Experiments:"
echo "-------------------------------------------------------------------"
ps aux | grep -E "experiment_(by_class|whitening|k_sweep_ci)" | grep -v grep | awk '{print "  PID:", $2, " - ", $11, $12, $13}'
echo ""

# Check log files
echo "📝 Log File Status:"
echo "-------------------------------------------------------------------"
for log in logs/by_class.log logs/whitening.log logs/k_sweep_ci.log; do
    if [ -f "$log" ]; then
        lines=$(wc -l < "$log")
        size=$(ls -lh "$log" | awk '{print $5}')
        echo "  $log: $lines lines ($size)"
    else
        echo "  $log: Not started yet"
    fi
done
echo ""

# Show recent progress
echo "🔄 Recent Activity (last 10 lines of each log):"
echo "-------------------------------------------------------------------"
for log in logs/by_class.log logs/whitening.log logs/k_sweep_ci.log; do
    if [ -f "$log" ]; then
        echo ""
        echo "=== $log ==="
        tail -10 "$log"
    fi
done
