"""Test Isaac Sim initialization without motion pipeline"""
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
import omni.replicator.core as rep
import os
import numpy as np
import sys

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'baseline_source'))

from config_loader import load_config, apply_scatter_config
from SceneBuilder import SceneBuilder
from RobotArticulation import RobotArticulation
from DataLogger import DataLogger

print("[Init] Starting Isaac Sim initialization test...")

# ── 1. Configuration ─────────────────────────────────────────────────
config_path = os.path.join(os.path.dirname(__file__), 'configs', 'Part_Sorting.yaml')
cfg = load_config(config_path)
grasp_cfg = cfg.get("grasp", {})
print("[Init] Configuration loaded")

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
print("[Init] World initialized")

# ── 3. Data Logger ───────────────────────────────────────────────────
base_dir = os.path.dirname(__file__)
data_logger = DataLogger(
    enabled=True,
    csv_path=os.path.join(base_dir, 'logs', 'poses.csv'),
    camera_enabled=False,
    camera_hdf5_path=os.path.join(base_dir, 'logs', 'camera_data.hdf5'),
)
print("[Init] Data logger initialized")

# ── 4. Scene (scatter area → build → physics settle) ────────────────
scene = SceneBuilder(cfg, data_logger=data_logger)
apply_scatter_config(cfg)

scene.build_all()
rep.orchestrator.step()
print("[Init] Scene built and scattered")

world.play()
settle_time = grasp_cfg.get("settle_time", 2.0)
settle_steps = int(settle_time / world.get_physics_dt())
for i in range(settle_steps):
    world.step(render=False)
    if (i+1) % 30 == 0:
        print(f"  Physics settle: {i+1}/{settle_steps} steps")
print(f"[Init] Physics settled ({settle_time}s, {settle_steps} steps)")

# ── 5. Query Part Poses ──────────────────────────────────────────────
part_poses = scene.get_parts_world_poses()
print(f"[Init] Found {len(part_poses)} parts:")
for pp in part_poses:
    print(f"  {pp['prim_path']}: pos={pp['position']}")

# ── 6. Robot ─────────────────────────────────────────────────────────
world.pause()
scene.build_robot()
robot = RobotArticulation(prim_path="/Root/Ref_Xform/Ref", name="walkerS2")
robot.initialize()
print("[Init] Robot initialized")
world.play()

for _ in range(10):
    world.step(render=False)
print("[Init] Robot ready!")

# ── 7. Cleanup ──────────────────────────────────────────────────────
print("[Test] SUCCESS - Isaac Sim initialized without crash!")
world.pause()
data_logger.close()
