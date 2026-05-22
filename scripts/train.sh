#!/bin/bash
# Wrapper: fix LD_LIBRARY_PATH for libnvrtc-builtins before launching Python.
# Usage: bash scripts/train.sh [--epochs N] [--data /path/to.yaml] ...

NVRTC_SO=$(find /usr/local/cuda* /usr/lib/x86_64-linux-gnu -name "libnvrtc-builtins.so*" 2>/dev/null | sort -rV | head -1)
if [ -n "$NVRTC_SO" ]; then
    NVRTC_DIR=$(dirname "$NVRTC_SO")
    export LD_LIBRARY_PATH="$NVRTC_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[train.sh] libnvrtc-builtins found: $NVRTC_SO"
    echo "[train.sh] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
else
    echo "[train.sh] WARNING: libnvrtc-builtins.so not found — validation may crash"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/train_yolo.py" "$@"
