"""
Chạy Imitation Learning policy trong Isaac Sim.

Dùng model đã train ở robot_arm_training/ để điều khiển robot thật trong sim.

Command:
    python run_il_policy.py                              # part_sorting_long, lstm, có GUI
    python run_il_policy.py --task part_sorting_long     # long dataset (4 grasps)
    python run_il_policy.py --task task1                 # task1 dataset (1 grasp)
    python run_il_policy.py --model mlp                  # dùng MLP thay vì LSTM
    python run_il_policy.py --headless                   # không GUI
    python run_il_policy.py --episodes 5
"""

import argparse
import os
import sys

# ── Parse args trước khi khởi động Isaac Sim ─────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",     choices=["task1", "part_sorting_long"],
                   default="part_sorting_long",
                   help="Dataset/task đã dùng để train model")
    p.add_argument("--model",    choices=["mlp", "lstm"], default="lstm")
    p.add_argument("--headless", action="store_true",
                   help="Chạy không GUI (nhanh hơn, dùng khi không có màn hình)")
    p.add_argument("--max_steps", type=int, default=800,
                   help="Số physics steps tối đa mỗi episode")
    p.add_argument("--episodes", type=int, default=3,
                   help="Số episode chạy liên tiếp")
    return p.parse_args()

args = parse_args()

# ── Khởi động Isaac Sim ───────────────────────────────────────────────────────
from isaacsim import SimulationApp

kit = SimulationApp({
    "width": 1280, "height": 720,
    "headless": args.headless,
})

# Import sau khi SimulationApp đã khởi động
from isaacsim.core.api import World
import omni
import omni.replicator.core as rep
import numpy as np

# Import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "baseline_source"))
sys.path.insert(0, os.path.dirname(__file__))

from config_loader import load_config, apply_scatter_config
from SceneBuilder import SceneBuilder
from isaac_sim_robot_interface import IsaacSimRobotInterface
from DataLogger import DataLogger

# Import IL policy theo task được chọn
if args.task == "part_sorting_long":
    from robot_arm_training.part_sorting_long.inference_isaac import ILPolicyRunner
else:
    from robot_arm_training.task1.inference_isaac import ILPolicyRunner

# ─────────────────────────────────────────────────────────────────────────────
# Setup scene (giống test_isaac_init.py)
# ─────────────────────────────────────────────────────────────────────────────
config_path = os.path.join(os.path.dirname(__file__), "configs", "Part_Sorting.yaml")
cfg = load_config(config_path)
grasp_cfg = cfg.get("grasp", {})

omni.usd.get_context().open_stage(
    os.path.join(cfg["root_path"], cfg["scene_usd"])
)

world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 60.0,
    rendering_dt=1.0 / 20.0,
)
world.initialize_physics()

base_dir = os.path.dirname(__file__)
data_logger = DataLogger(
    enabled=False,
    csv_path=os.path.join(base_dir, "logs", "il_policy.csv"),
    camera_enabled=False,
    camera_hdf5_path=os.path.join(base_dir, "logs", "il_camera.hdf5"),
)

scene = SceneBuilder(cfg, data_logger=data_logger)
apply_scatter_config(cfg)
scene.build_all()
rep.orchestrator.step()

world.play()
settle_steps = int(grasp_cfg.get("settle_time", 2.0) / world.get_physics_dt())
for i in range(settle_steps):
    world.step(render=False)
print(f"[IL] Physics settled ({settle_steps} steps)")

# Build robot
world.pause()
scene.build_robot()

robot = IsaacSimRobotInterface(
    prim_path="/Root/Ref_Xform/Ref",
    name="walkerS2",
    world=world,
)
robot.initialize()
world.play()
for _ in range(10):
    world.step(render=False)
print("[IL] Robot ready")

# ── Load IL policy ────────────────────────────────────────────────────────────
runner = ILPolicyRunner(model_type=args.model)
runner.load()

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
PHYSICS_DT = world.get_physics_dt()

for episode in range(args.episodes):
    print(f"\n[IL] ===== Episode {episode + 1}/{args.episodes} =====")
    runner.reset()

    part_poses = scene.get_parts_world_poses()
    print(f"[IL] {len(part_poses)} parts detected")

    gripper_state = [-1.0, -1.0]   # cả hai gripper đang mở
    _prev_gripper_closed = False

    # Stall detection: if arm MAE stays < threshold for this many steps → stuck
    _STALL_MAE_THRESH  = 0.002   # rad
    _STALL_MIN_STEPS   = 120     # tăng lên để cho phép robot dừng ngắn giữa các grasp
    _stall_counter     = 0

    for step in range(args.max_steps):

        # ── Cập nhật pose vật thể mỗi bước (quan trọng: model cần biết vật đang ở đâu) ──
        part_poses = scene.get_parts_world_poses()

        # ── Lấy state từ sim ──────────────────────────────────────────────
        joint_states = robot.get_joint_states()
        if joint_states is None:
            print(f"[IL] Step {step}: cannot read joint states, skipping")
            world.step(render=True)
            continue

        # ── Chạy model → predict action ───────────────────────────────────
        action = runner.step(
            robot,
            part_poses,
            gripper_control=gripper_state,
            apply=True,   # gửi action xuống robot ngay
        )

        # Cập nhật trạng thái gripper cho bước tiếp theo
        # action[9] = right_gripper; left arm cố định nên luôn -1 (mở)
        new_gripper_closed = float(action[9]) > 0
        gripper_state = [-1.0, float(action[9])]

        # Log khi gripper chuyển trạng thái
        if new_gripper_closed != _prev_gripper_closed:
            state_str = "CLOSE" if new_gripper_closed else "OPEN"
            print(f"  step={step:4d} | gripper → {state_str}")
        _prev_gripper_closed = new_gripper_closed

        # ── Bước simulation ───────────────────────────────────────────────
        world.step(render=not args.headless)

        # ── Tính MAE và kiểm tra stall ────────────────────────────────────
        right_arm_mae = float(np.mean(np.abs(
            np.array(joint_states["arm_positions"])[7:14] - action[:7]
        )))

        if right_arm_mae < _STALL_MAE_THRESH:
            _stall_counter += 1
        else:
            _stall_counter = 0

        if step % 60 == 0:
            print(f"  step={step:4d} | right arm MAE={right_arm_mae:.4f} rad "
                  f"| right gripper={'C' if gripper_state[1] > 0 else 'O'}"
                  f"| stall={_stall_counter}/{_STALL_MIN_STEPS}")

        # Thoát sớm nếu arm bị kẹt và gripper đã đóng (không gắp được gì)
        if _stall_counter >= _STALL_MIN_STEPS and gripper_state[1] > 0:
            print(f"  [IL] Stall detected at step {step} (MAE<{_STALL_MAE_THRESH} for {_STALL_MIN_STEPS} steps, gripper=C) — ending episode early")
            break

    # Reset robot về vị trí ban đầu sau mỗi episode
    robot.reset()
    for _ in range(30):
        world.step(render=False)
    print(f"[IL] Episode {episode + 1} done, robot reset")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[IL] Finished. Cleaning up...")
data_logger.close()
kit.close()
