#!/bin/bash
# CHEATSHEET — HRC2026 Task 1 Perception
# Machine: teleop-team3-hrc2026
# =========================================================

ISAAC="/isaac-sim/python.sh"
UBTECH="/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim"

# Buoc 1: Lan dau — luu camera intrinsics + T_base_camera
# Chay 3 frame roi thoat, luu extracted_camera_params.yaml
$ISAAC perception_runner.py \
    --ubtech-dir $UBTECH \
    --frames 3 \
    --save-params

# Buoc 2: Capture sample RGB + depth, kiem tra perception
$ISAAC perception_runner.py \
    --ubtech-dir $UBTECH \
    --frames 5

# Buoc 3: Do shape features (khong can Isaac Sim)
python scripts/measure_shape_features.py \
    --rgb lab_outputs/perception/sample_rgb.png \
    --depth lab_outputs/perception/sample_depth.npy \
    --no-gui

# Buoc 4: Chay full (Ctrl+C de dung)
$ISAAC perception_runner.py \
    --ubtech-dir $UBTECH

# Unit tests (khong can Isaac Sim)
python -m pytest tests/ -v
