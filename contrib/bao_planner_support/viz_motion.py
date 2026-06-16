"""
Simple motion visualization - shows robot pick & place
"""
from isaacsim import SimulationApp

cfg = {"headless": False}  # SHOW UI!
app = SimulationApp(launch_config=cfg)

import os, sys
os.chdir('/home/ubuntu/vinh')
sys.path.insert(0, '/home/ubuntu/vinh/src/baseline_source')
sys.path.insert(0, '/home/ubuntu/vinh/src/task1')

from isaacsim.core.api import World
import omni

# Load config first
from config_loader import load_config, apply_scatter_config
config = load_config('/home/ubuntu/vinh/configs/Part_Sorting.yaml')

# Load scene from config
print("[Load] Opening scene...")
scene_path = os.path.join(config["root_path"], config["scene_usd"])
print(f"  Scene: {scene_path}")
omni.usd.get_context().open_stage(scene_path)

# Setup world
world = World(stage_units_in_meters=1.0, physics_dt=1/60, rendering_dt=1/20)
world.initialize_physics()

# Build scene first
from SceneBuilder import SceneBuilder
from DataLogger import DataLogger
import omni.replicator.core as rep

data_logger = DataLogger(enabled=False)
scene = SceneBuilder(config, data_logger=data_logger)
apply_scatter_config(config)
scene.build_all()
rep.orchestrator.step()

print("[Robot] Initializing robot...")
from RobotArticulation import RobotArticulation
robot = RobotArticulation(prim_path="/Root/Ref_Xform/Ref", name="walkerS2")
robot.initialize()

# Settle physics
print("[Init] Settling scene...")
world.play()
for _ in range(120):
    world.step(render=True)

# Get parts
part_poses = scene.get_parts_world_poses()
print(f"[Motion] Found {len(part_poses)} parts - showing first 1")

# Prepare motion data
import json
task_data = {
    "task_number": 1,
    "total_workpieces": 1,
    "workpieces": [{
        "id": part_poses[0].get('prim_path', 'part_0').split('/')[-1],
        "type": "PartA",
        "position": list(part_poses[0]['position']),
        "quaternion": list(part_poses[0].get('orientation', [0,0,0,1])),
        "bin_position": [1.2, 0.3, 1.05]
    }]
}

with open('/tmp/motion_demo.json', 'w') as f:
    json.dump(task_data, f)

# Load action plan
from motion import load_action_plan, run_pipeline
action_plan = load_action_plan('/home/ubuntu/vinh/src/task1/primitive_spec_task1.yaml')

# RUN MOTION WITH UI!
print("\n🤖 WATCH THE SIMULATION - Robot is picking up object!\n")
world.pause()
world.play()

try:
    run_pipeline('/tmp/motion_demo.json', action_plan, robot=robot, world=world)
    print("\n✅ Motion complete! Window stays open to view...")
except Exception as e:
    print(f"❌ Motion error: {e}")

# Keep window open
print("Close the window to exit...")
for _ in range(300):  # 5 more seconds
    world.step(render=True)

app.close()
