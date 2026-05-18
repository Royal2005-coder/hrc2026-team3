#!/bin/bash
cd "$(dirname "$0")/.."
echo "=== HRC2026 Task 1 ==="
echo "CWD: $(pwd)"
/isaac-sim/python.sh src/baseline_source/main_fixed.py "$@"
