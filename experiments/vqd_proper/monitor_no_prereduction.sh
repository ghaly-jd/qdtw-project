#!/bin/bash
# Monitor No Pre-Reduction Experiment

LOG_FILE="/path/to/qdtw_project/vqd_proper_experiments/logs/no_prereduction.log"

echo "========================================================================"
echo "NO PRE-REDUCTION EXPERIMENT MONITOR"
echo "========================================================================"
echo ""

# Check if process is running
if pgrep -f "experiment_no_prereduction.py" > /dev/null; then
    echo "✓ Process is RUNNING"
    PID=$(pgrep -f "experiment_no_prereduction.py")
    echo "  PID: $PID"
else
    echo "✗ Process is NOT RUNNING"
fi

echo ""
echo "Log file size: $(du -h $LOG_FILE 2>/dev/null | cut -f1 || echo '0')"
echo ""

echo "========================================================================"
echo "RECENT OUTPUT (last 30 lines):"
echo "========================================================================"
tail -30 "$LOG_FILE" 2>/dev/null || echo "No log file yet"

echo ""
echo "========================================================================"
echo "To watch live: tail -f $LOG_FILE"
echo "========================================================================"
