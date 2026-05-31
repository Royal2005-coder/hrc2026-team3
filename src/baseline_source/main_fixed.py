"""Ubtech_sim Simulation Entry Point.

Launch Isaac Sim, load task config, build scene, and run the grasp control loop.
"""
from isaacsim import SimulationApp

CONFIG = {
    "width": 1280,
    "height": 720,
    "headless": False,
}

kit = SimulationApp(launch_config=CONFIG)

# Isaac Sim modules must be imported after SimulationApp is created
from isaacsim.core.api import World
import omni
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

# Add task1 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'task1'))

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

# ── 4. Scene Build ───────────────────────────────────────────────────
scene = SceneBuilder(cfg, data_logger=data_logger)
apply_scatter_config(cfg)
scene.build_all()
print("[Init] Scene objects created. Starting physics settle...")

world.play()

# FIX: render=True so Replicator scatter_2d actually applies positions.
# With render=False the Replicator graph is skipped → parts stay at [0,0,0].
settle_time = grasp_cfg.get("settle_time", 2.0)
settle_steps = int(settle_time / world.get_physics_dt())
for _ in range(settle_steps):
    world.step(render=True)
print(f"[Init] Physics settle done ({settle_time}s / {settle_steps} steps)")

# ── 5. Query Part Poses ──────────────────────────────────────────────
part_poses = scene.get_parts_world_poses()
print(f"[Init] Found {len(part_poses)} parts:")

_BIN_WORLD_BY_TYPE = {
    "PartA": [1.2,  0.3, 1.05],   # from Part_Sorting.yaml box_position
    "PartB": [1.2, -0.3, 1.05],   # symmetric target position for PartB
}

workpieces_out = {
    "task_number": cfg.get("task_number", 1),
    "total_workpieces": len(part_poses),
    "workpieces": [],
}

_n_at_origin = 0
for pp in part_poses:
    part_type = pp.get("type", "PartA")
    prim_id = pp["prim_path"].split("/")[-1]
    bin_world = _BIN_WORLD_BY_TYPE.get(part_type, _BIN_WORLD_BY_TYPE["PartA"])
    workpieces_out["workpieces"].append({
        "id": prim_id,
        "prim_path": pp["prim_path"],
        "type": part_type,
        "position": pp["position"],
        "quaternion": pp["orientation"],
        "bin_position": [bin_world],
    })
    at_origin = all(abs(v) < 1e-3 for v in pp["position"])
    if at_origin:
        _n_at_origin += 1
    print(f"  {prim_id} [{part_type}] pos={[round(v,3) for v in pp['position']]} → bin={bin_world}"
          + ("  ⚠ AT ORIGIN — scatter may not have applied" if at_origin else ""))

if _n_at_origin > 0:
    print(f"\n[WARNING] {_n_at_origin}/{len(part_poses)} parts are at [0,0,0].")
    print("[WARNING] Increasing settle time or ensuring render=True during settle may fix this.")

_wp_json_path = os.path.join(os.path.dirname(__file__), '..', '..', 'task1_workpieces.json')
with open(_wp_json_path, 'w') as _f:
    json.dump(workpieces_out, _f, indent=2)
print(f"[Init] task1_workpieces.json written ({len(part_poses)} parts, with type + bin info)")

# ── 6. Robot ─────────────────────────────────────────────────────────
print("[Init] Building robot...")
try:
    scene.build_robot()
    robot_prim_path = scene.robot_prim_path
    print(f"[Init] Robot prim path: {robot_prim_path}")

    if robot_prim_path is not None:
        robot = RobotArticulation(prim_path=robot_prim_path, name="walkerS2")
        robot.initialize()
        print(f"[Init] Robot articulation initialized")
    else:
        print("[Warning] robot_prim_path is None — motion will be skipped")
        robot = None
except Exception as e:
    print(f"[Error] Robot loading failed: {e}")
    import traceback; traceback.print_exc()
    robot = None

# FIX: Settle robot with render=True for 2 seconds so physics properly places
# the robot at its initial pose before IK is synced to the joint state.
# 10 frames (original) is only ~0.17s — not enough for the robot to stabilize.
_robot_settle_steps = 120   # 2 seconds at 60 Hz
for _ in range(_robot_settle_steps):
    world.step(render=True)
print(f"[Init] Robot settle done ({_robot_settle_steps} steps)")

# ── 7. IK & Coordinate Transform ────────────────────────────────────
coord_transform = None
if robot is not None:
    print("[Init] Initializing IK solver...")
    urdf_path = os.path.join(cfg["root_path"], "s2.urdf")
    robot.initialize_ik(urdf_path)

    # Sync IK solver to current (settled) joint positions
    js = robot.get_joint_states()
    if js is not None:
        _pos = js["positions"]
        robot.ik_solver.sync_joint_positions(
            js["names"], _pos[0] if isinstance(_pos[0], list) else _pos)
        print("[Init] IK solver synced with settled joint state")

    # Build world↔base coordinate transform from torso_link (one call is enough)
    coord_transform = CoordinateTransform.from_torso_link(ik_solver=robot.ik_solver)
    coord_transform.verify_ee_alignment(robot.ik_solver)
else:
    print("[Init] Skipping IK — robot not loaded")

# ── 8. Motion Planning & Execution ───────────────────────────────────
print("\n[Motion] Starting pick-and-place pipeline...")

if robot is None:
    print("[Motion] Skipped — robot not loaded")
else:
    action_plan_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'src', 'task1', 'primitive_spec_task1.yaml')

    try:
        from motion import run_pipeline_from_person2
        print("=" * 70)
        run_pipeline_from_person2(
            action_plan_yaml=action_plan_path,
            robot=robot,
            world=world,
            coord_transform=coord_transform,
        )
        print("=" * 70)
        print("\n[Motion] Pipeline complete!")
    except Exception as e:
        print("=" * 70)
        print(f"\n[Error] Motion pipeline failed: {e}")
        import traceback; traceback.print_exc()

# ── 9. Cleanup ───────────────────────────────────────────────────────
print("\n[Sim] Shutting down...")

# Remove any lingering grasp_attach FixedJoints before shutdown
try:
    from isaacsim.core.utils.stage import get_current_stage
    _stage = get_current_stage()
    for _prim in list(_stage.TraverseAll()):
        if _prim.GetName() == "grasp_attach":
            _stage.RemovePrim(_prim.GetPath())
except Exception:
    pass

world.pause()
data_logger.close()

# Must call kit.close() before Python exits — otherwise OmniGraph atexit
# handlers run on a dirty state and produce a segfault.
kit.close()
