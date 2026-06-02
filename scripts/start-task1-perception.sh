#!/bin/bash
export DISPLAY=:20
export XDG_RUNTIME_DIR=/tmp/runtime-ubuntu
export OMNI_KIT_ALLOW_ROOT=1
cd /workspace/GlobalHumanoidRobotChallenge_2026_Baseline
exec /isaac-sim/python.sh lerobot/scripts/control_robot.py \
  --robot.type=walker_s2_sim \
  --control.type=teleoperate \
  --control.task=Part_Sorting \
  --control.fps=30 \
  --control.teleop_time_s=3600
