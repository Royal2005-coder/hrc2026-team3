"""
Chạy Imitation Learning policy trong Isaac Sim.

Dùng model đã train ở robot_arm_training/ để điều khiển robot thật trong sim.

Command:
    python run_il_policy.py                    # lstm, có GUI
    python run_il_policy.py --headless         # không GUI (nhanh hơn)
    python run_il_policy.py --episodes 5
"""

import argparse
import os
import sys

# ── Parse args trước khi khởi động Isaac Sim ─────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    choices=["mlp", "lstm"], default="lstm")
    p.add_argument("--headless", action="store_true",
                   help="Chạy không GUI (nhanh hơn, dùng khi không có màn hình)")
    p.add_argument("--max_steps", type=int, default=500,
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

# Import IL policy
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

    # Lấy pose các vật thể
    part_poses = scene.get_parts_world_poses()
    print(f"[IL] {len(part_poses)} parts detected")

    gripper_state = [-1.0, -1.0]   # cả hai gripper đang mở

    for step in range(args.max_steps):

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
        gripper_state = [-1.0, float(action[9])]

        # ── Bước simulation ───────────────────────────────────────────────
        world.step(render=not args.headless)

        if step % 60 == 0:
            # action[0:7] = R arm joints; robot arm_positions[7:14] = right arm
            right_arm_mae = float(np.mean(np.abs(
                np.array(joint_states["arm_positions"])[7:14] - action[:7]
            )))
            print(f"  step={step:4d} | right arm MAE={right_arm_mae:.4f} rad "
                  f"| right gripper={'C' if gripper_state[1] > 0 else 'O'}")

    # Reset robot về vị trí ban đầu sau mỗi episode
    robot.reset()
    for _ in range(30):
        world.step(render=False)
    print(f"[IL] Episode {episode + 1} done, robot reset")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[IL] Finished. Cleaning up...")
data_logger.close()
kit.close()
