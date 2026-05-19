#!/bin/bash
echo "=== HRC2026 Task 1 - main_fixed ==="
export DISPLAY=:20
export XDG_RUNTIME_DIR=/tmp/runtime-ubuntu
export OMNI_KIT_ALLOW_ROOT=1

# Tạo thư mục source/ giả bằng symlink
BASELINE="/home/ubuntu/tai/src/baseline_source"
SRC="/home/ubuntu/tai/src"

# Tạo source/ symlink nếu chưa có
if [ ! -d "$SRC/source" ]; then
    ln -s "$BASELINE" "$SRC/source"
    echo "[OK] Created symlink: src/source -> baseline_source"
fi

# Tạo config/ symlink nếu chưa có
if [ ! -d "$BASELINE/config" ]; then
    ln -s "/home/ubuntu/tai/configs" "$BASELINE/config"
    echo "[OK] Created symlink: baseline_source/config -> configs"
fi

cd "$BASELINE"
echo "CWD: $(pwd)"

/isaac-sim/python.sh main_fixed.py "$@"
