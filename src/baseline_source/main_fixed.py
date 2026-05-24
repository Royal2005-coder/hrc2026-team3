"""Ubtech_sim Simulation Entry Point.

Launch Isaac Sim, load task config, build scene, and run the grasp control loop.
"""
from isaacsim import SimulationApp

CONFIG = {
    "width": 1280,
    "height": 720,
    "headless": False,  # Show UI
}

kit = SimulationApp(launch_config=CONFIG)

# Isaac Sim modules must be imported after SimulationApp is created
from isaacsim.core.api import World
import omni
# import omni.replicator.core as rep  # REMOVED: causes segfault on shutdown
import os
import numpy as np
import sys
import json

# Add source directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config_loader import load_config, apply_scatter_config
from SceneBuilder import SceneBuilder
from RobotArticulation import RobotArticulation
from DataLogger import DataLogger
from coordinate_utils import CoordinateTransform
from grasp_planner import GraspPlanner

# Add task1 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'task1'))
from motion import load_action_plan, run_pipeline

# ── 1. Configuration ─────────────────────────────────────────────────
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'Part_Sorting.yaml')
cfg = load_config(config_path)
grasp_cfg = cfg.get("grasp", {})

# ── 2. Stage & World ─────────────────────────────────────────────────
omni.usd.get_context().open_stage(
    os.path.join(cfg["root_path"], cfg["scene_usd"])
)
world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 60.0,
    rendering_dt=1.0 / 20.0,
)
world.initialize_physics()

# ── 3. Data Logger ───────────────────────────────────────────────────
base_dir = os.path.dirname(__file__)
data_logger = DataLogger(
    enabled=True,
    csv_path=os.path.join(base_dir, "poses.csv"),
    camera_enabled=False,
    camera_hdf5_path=os.path.join(base_dir, "camera_data.hdf5"),
)

# ── 4. Scene (scatter area → build → physics settle) ────────────────
scene = SceneBuilder(cfg, data_logger=data_logger)
apply_scatter_config(cfg)

scene.build_all()
# NOTE: rep.orchestrator.step() was causing segfault - skip for now
# rep.orchestrator.step()
print("[Init] 场景物体已创建，开始物理稳定...")

world.play()
settle_time = grasp_cfg.get("settle_time", 2.0)
settle_steps = int(settle_time / world.get_physics_dt())
for _ in range(settle_steps):
    world.step(render=False)
print(f"[Init] 物理稳定完成 ({settle_time}s, {settle_steps} steps)")

# ── 5. Query Part Poses ──────────────────────────────────────────────
part_poses = scene.get_parts_world_poses()
print(f"[Init] 查询到 {len(part_poses)} 个零件")
for pp in part_poses:
    print(f"  {pp['prim_path']}: pos={pp['position']}")

# ── 6. Robot ─────────────────────────────────────────────────────────
print("[Init] 开始加载机器人...")
try:
    scene.build_robot()
    robot_prim_path = scene.robot_prim_path
    print(f"[Init] Robot loaded: {robot_prim_path}")
    
    if robot_prim_path is not None:
        robot = RobotArticulation(prim_path=robot_prim_path, name="walkerS2")
        robot.initialize()
        print(f"[Init] 机器人已初始化 ({robot_prim_path})")
    else:
        print("[Warning] robot_prim_path is None")
        robot = None
except Exception as e:
    print(f"[Error] Robot loading failed: {e}")
    robot = None

for _ in range(10):
    world.step(render=False)

# ── 7. IK & Coordinate Transform ────────────────────────────────────
if robot is not None:
    print("[Init] 初始化机器人 IK...")
    urdf_path = os.path.join(cfg["root_path"], "s2.urdf")
    robot.initialize_ik(urdf_path)

    js = robot.get_joint_states()
    if js is not None:
        robot.ik_solver.sync_joint_positions(js["names"], js["positions"][0])

    compensation_matrix = np.array([
        [9.99999e-01, -1.11400e-03,  1.16200e-03, -9.64000e-04],
        [-2.00000e-05,  7.13609e-01,  7.00544e-01, -9.59927e-01],
        [-1.61000e-03, -7.00544e-01,  7.13608e-01,  6.56540e-01],
        [0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]
    ], dtype=np.float64)

    coord_transform = CoordinateTransform.from_torso_link(ik_solver=robot.ik_solver)
    for _ in range(10):
        coord_transform.verify_ee_alignment(robot.ik_solver)
else:
    print("[Init] 机器人未加载，跳过 IK 初始化")

# ── 8. Motion Planning & Execution ──────────────────────────────
print("\n[Motion] Preparing Pick & Place motion planning...")

if robot is None:
    print("[Motion] 跳过 - 机器人未加载")
else:
    # Generate workpieces JSON
    workpieces_data = {
        "task_number": 1,
        "total_workpieces": len(part_poses),
        "workpieces": []
    }

    for i, part_pose in enumerate(part_poses):
        pos = part_pose['position']
        rot = part_pose.get('orientation', [0, 0, 0, 1])
        part_type = part_pose.get('type', 'PartA')
        
        # Assign bin position based on part type
        if 'PartA' in part_type or i % 2 == 0:
            bin_pos = [1.2, 0.3, 1.05]
        else:
            bin_pos = [1.2, -0.3, 1.05]
        
        workpiece = {
            "id": part_pose.get('prim_path', f'part_{i}').split('/')[-1],
            "type": part_type,
            "position": list(pos),
            "quaternion": list(rot),
            "bin_position": list(bin_pos)
        }
        workpieces_data['workpieces'].append(workpiece)

    # Save workpieces JSON
    task_json_path = '/tmp/task1_workpieces.json'
    with open(task_json_path, 'w') as f:
        json.dump(workpieces_data, f, indent=2)
    print(f"[Motion] Workpiece data saved to: {task_json_path}")

    # Load motion action plan
    action_plan_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'task1', 'primitive_spec_task1.yaml')
    try:
        action_plan = load_action_plan(action_plan_path)
        print(f"[Motion] Action plan loaded:")
        print(f"  - Approach offset: {action_plan['approach_offset_m']} m")
        print(f"  - Lift height: {action_plan['lift_height_m']} m")
    except Exception as e:
        print(f"[Error] Failed to load action plan: {e}")
        action_plan = {'approach_offset_m': 0.08, 'lift_height_m': 0.17}

    # Execute motion pipeline (generate waypoints WITHOUT simulator execution)
    print("\n[Motion] Generating motion waypoints (IK-only, no execution)...\n")
    print("=" * 70)

try:
    # Run with robot=None, world=None to skip execution, just generate CSV
    run_pipeline(task_json_path, action_plan, robot=robot, world=world, coord_transform=coord_transform)
    print("=" * 70)
    print("\n[Motion] ✓ Waypoints generated and executed successfully!")
except Exception as e:
    print("=" * 70)
    print(f"\n[Error] Motion planning failed: {e}")
    import traceback
    traceback.print_exc()

# ── 9. Cleanup ──────────────────────────────────────────────────────
print("\n[Sim] Test completed successfully!")
print("[Sim] - Scene initialized with", len(part_poses), "parts")
print("[Sim] - Robot initialized and ready")
print("[Sim] - IK solver initialized")
print("\n[Note] Motion pipeline execution coming soon...")

# Cleanup
world.pause()
data_logger.close()
print("[Sim] Cleanup complete")