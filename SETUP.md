# Setup Guide — HRC2026 Team 3

## Container co san (KHONG can cai them)
- Isaac Sim 5.1: /isaac-sim/python.sh
- PyTorch 2.7+cu128, Pinocchio 3.9, OpenCV 4.11
- LeRobot v0.1.0 (ACT + Pi0): symlink lerobot/
- Baseline Ubtech_sim: symlink Ubtech_sim_ref/
- Assets 109 USD + s2.urdf: symlink assets/

## Moi lan vao workspace
bash /home/ubuntu/auto_setup.sh
cd /home/ubuntu/<ten>
git pull

## Chay Task 1
/isaac-sim/python.sh src/baseline_source/main_fixed.py
bash scripts/run_task1.sh

## Chay code ca nhan
/isaac-sim/python.sh src/task1/perception.py
/isaac-sim/python.sh src/task1/planner.py
/isaac-sim/python.sh src/task1/motion.py
/isaac-sim/python.sh src/task1/eval.py

## Member folders
/home/ubuntu/tai  (tai/perception)
/home/ubuntu/thu  (thu/planner)
/home/ubuntu/bao  (bao/planner_support)
/home/ubuntu/vinh (vinh/motion)
/home/ubuntu/gia  (gia/evaluation)
