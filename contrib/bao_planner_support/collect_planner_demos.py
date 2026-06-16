"""
Teacher-Student Data Collection — Planner → IL policy
========================================================
Chạy planner IK+FSM nhiều episode với vị trí vật ngẫu nhiên,
log (state, action) mỗi physics step vào CSV cùng format với
data/task1/part_sorting_sample.csv.

Usage:
    python collect_planner_demos.py --episodes 200
    python collect_planner_demos.py --episodes 500 --headless
    python collect_planner_demos.py --episodes 100 --output data/task1/planner_demos.csv
"""

import argparse
import csv
import os
import sys

# ── Parse args trước khi khởi động Isaac Sim ──────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Collect planner demos for teacher-student distillation")
    p.add_argument("--episodes",  type=int,   default=200,
                   help="Số episode cần thu thập")
    p.add_argument("--output",    type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "data", "task1", "planner_demos.csv"),
                   help="Đường dẫn CSV output")
    p.add_argument("--headless",  action="store_true",
                   help="Chạy không GUI (nhanh hơn)")
    p.add_argument("--append",    action="store_true",
                   help="Append vào CSV hiện có thay vì ghi đè")
    return p.parse_args()

args = parse_args()

# ── Khởi động Isaac Sim ───────────────────────────────────────────────────────
from isaacsim import SimulationApp
kit = SimulationApp({"width": 1280, "height": 720, "headless": args.headless})

from isaacsim.core.api import World
import omni
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src", "baseline_source"))
sys.path.insert(0, os.path.join(_ROOT, "src", "task1"))

from config_loader import load_config, apply_scatter_config
from SceneBuilder import SceneBuilder
from RobotArticulation import RobotArticulation
from DataLogger import DataLogger
from coordinate_utils import CoordinateTransform

# ── Config ────────────────────────────────────────────────────────────────────
config_path = os.path.join(_ROOT, "configs", "Part_Sorting.yaml")
cfg         = load_config(config_path)
grasp_cfg   = cfg.get("grasp", {})

# ── Stage & World ─────────────────────────────────────────────────────────────
omni.usd.get_context().open_stage(os.path.join(cfg["root_path"], cfg["scene_usd"]))
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 20.0)
world.initialize_physics()

# ── Scene ─────────────────────────────────────────────────────────────────────
_dummy_logger = DataLogger(enabled=False, csv_path=os.path.join(_ROOT, "logs", "dummy.csv"))
scene = SceneBuilder(cfg, data_logger=_dummy_logger)
apply_scatter_config(cfg)
scene.build_all()

world.play()
_settle_steps = int(grasp_cfg.get("settle_time", 2.0) / world.get_physics_dt())
for _ in range(_settle_steps):
    world.step(render=True)
print(f"[Collect] Initial physics settle done ({_settle_steps} steps)")

# ── Robot ─────────────────────────────────────────────────────────────────────
scene.build_robot()
robot = RobotArticulation(prim_path=scene.robot_prim_path, name="walkerS2")
robot.initialize()

for _ in range(120):   # 2s settle
    world.step(render=True)

urdf_path = os.path.join(cfg["root_path"], "s2.urdf")
robot.initialize_ik(urdf_path)

_js = robot.get_joint_states()
if _js:
    _p = _js["positions"]
    robot.ik_solver.sync_joint_positions(_js["names"], _p[0] if isinstance(_p[0], list) else _p)
    robot.ik_solver.save_initial_q()

coord_transform = CoordinateTransform.from_torso_link(ik_solver=robot.ik_solver)
coord_transform.verify_ee_alignment(robot.ik_solver)
print("[Collect] Robot + IK ready")


# ─────────────────────────────────────────────────────────────────────────────
# Joint name → DOF index map (valid across episodes; DOF order never changes)
# ─────────────────────────────────────────────────────────────────────────────
_dof_names = robot._articulation.dof_names
_name2idx   = {n: i for i, n in enumerate(_dof_names)}

L_ARM_JOINTS    = ["L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
                   "L_elbow_roll_joint",     "L_elbow_yaw_joint",
                   "L_wrist_pitch_joint",    "L_wrist_roll_joint"]
R_ARM_JOINTS    = ["R_shoulder_pitch_joint", "R_shoulder_roll_joint", "R_shoulder_yaw_joint",
                   "R_elbow_roll_joint",     "R_elbow_yaw_joint",
                   "R_wrist_pitch_joint",    "R_wrist_roll_joint"]
FINGER_JOINTS   = ["L_finger1_joint", "L_finger2_joint", "R_finger1_joint", "R_finger2_joint"]


# ─────────────────────────────────────────────────────────────────────────────
# CSV header — exact same column order as part_sorting_sample.csv
# ─────────────────────────────────────────────────────────────────────────────
def _build_fieldnames():
    cols = ["episode_index", "frame_index", "timestamp", "task_index"]
    for j in L_ARM_JOINTS + R_ARM_JOINTS:
        cols.append(f"state.{j}.pos")
    for j in FINGER_JOINTS:
        cols.append(f"state.{j}.pos")
    cols += ["state.left_gripper_control", "state.right_gripper_control"]
    for i in range(4):
        for attr in ["x", "y", "z", "qx", "qy", "qz", "qw"]:
            cols.append(f"state.obj{i}_{attr}")
    for j in L_ARM_JOINTS + R_ARM_JOINTS:
        cols.append(f"action.{j}.pos")
    for j in FINGER_JOINTS:
        cols.append(f"action.{j}.pos")
    cols += ["action.left_gripper_control", "action.right_gripper_control"]
    return cols

FIELDNAMES = _build_fieldnames()


# ─────────────────────────────────────────────────────────────────────────────
# PlannerDataHook — intercepts planner output, captures (state, action) rows
# ─────────────────────────────────────────────────────────────────────────────

class PlannerDataHook:
    """
    Patches RobotArticulation.close_gripper / open_gripper to track gripper
    commands. Capturing is triggered externally via capture() before each
    world.step() while self.collecting == True.

    State  = current robot joint positions + current object poses (read each step)
    Action = last IK target (robot._last_arm_positions) + last gripper command
    """

    def __init__(self, robot: RobotArticulation, scene: SceneBuilder):
        self._robot   = robot
        self._scene   = scene
        self.collecting = False

        # Gripper command tracking (-1=open, +1=close)
        self._r_gripper = -1.0
        self._l_gripper = -1.0

        # Per-episode counters
        self._episode_idx = 0
        self._frame_idx   = 0
        self._step_tick   = 0   # physics step counter for 30Hz subsampling

        # Accumulated rows; flushed externally after each episode
        self.rows: list = []

        # Monkey-patch close/open gripper to track commanded gripper state
        _orig_close = robot.close_gripper
        _orig_open  = robot.open_gripper

        def _patched_close(side=None, **kw):
            if side in (None, "right"):
                self._r_gripper = 1.0
            if side in (None, "left"):
                self._l_gripper = 1.0
            return _orig_close(side=side, **kw)

        def _patched_open(side=None, **kw):
            if side in (None, "right"):
                self._r_gripper = -1.0
            if side in (None, "left"):
                self._l_gripper = -1.0
            return _orig_open(side=side, **kw)

        robot.close_gripper = _patched_close
        robot.open_gripper  = _patched_open

    def reset_episode(self, episode_idx: int):
        self._episode_idx = episode_idx
        self._frame_idx   = 0
        self._step_tick   = 0
        self._r_gripper   = -1.0
        self._l_gripper   = -1.0

    def capture(self):
        """
        Call this BEFORE each world.step() during planner execution.
        Captures one row at ~30Hz (every 2 physics steps).
        Skips automatically if planner hasn't issued any IK command yet.
        """
        if not self.collecting:
            return

        # Subsample to 30Hz to match training data distribution
        self._step_tick += 1
        if self._step_tick % 2 != 0:
            return

        # Wait for planner to issue first IK command
        if "right" not in self._robot._last_arm_positions:
            return

        js = self._robot.get_joint_states()
        if js is None:
            return

        all_pos = js["positions"]
        if all_pos and isinstance(all_pos[0], list):
            all_pos = all_pos[0]
        all_pos = np.array(all_pos, dtype=np.float32)

        def _get(name):
            idx = _name2idx.get(name)
            return float(all_pos[idx]) if idx is not None else 0.0

        # ── State ─────────────────────────────────────────────────────────────
        l_arm_state = [_get(n) for n in L_ARM_JOINTS]
        r_arm_state = [_get(n) for n in R_ARM_JOINTS]
        l_fin_state = [_get(n) for n in FINGER_JOINTS[:2]]
        r_fin_state = [_get(n) for n in FINGER_JOINTS[2:]]

        # Object poses — read current from scene (objects move when carried)
        # Isaac orientation is [qw, qx, qy, qz]; CSV stores [qx, qy, qz, qw]
        obj_vals = []
        for part in (self._scene.get_parts_world_poses() or [])[:4]:
            pos = part["position"]
            ori = part["orientation"]   # [qw, qx, qy, qz]
            qw, qx, qy, qz = float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])
            obj_vals.extend([float(pos[0]), float(pos[1]), float(pos[2]), qx, qy, qz, qw])
        # Pad to exactly 4 objects × 7 = 28 values
        while len(obj_vals) < 28:
            obj_vals.extend([0.0] * 7)

        # ── Action ────────────────────────────────────────────────────────────
        # Right arm: last IK target (smoothed by EMA in control_dual_arm_ik)
        r_arm_action = list(self._robot._last_arm_positions["right"])

        # Left arm: IK target if available, else maintain current state
        _l_tgt = self._robot._last_arm_positions.get("left")
        l_arm_action = list(_l_tgt) if _l_tgt is not None else l_arm_state

        # Fingers: commanded width based on gripper state
        _close_w = self._robot.gripper_close_width   #  0.01
        _open_w  = self._robot.gripper_open_width    # -0.0215
        r_fin_action = [(_close_w if self._r_gripper > 0 else _open_w)] * 2
        l_fin_action = l_fin_state   # left arm not controlled in task1

        # ── Assemble row ──────────────────────────────────────────────────────
        timestamp = self._frame_idx / 30.0   # 30Hz → matches demo timestamps

        row = (
            [self._episode_idx, self._frame_idx, timestamp, 1]
            + l_arm_state + r_arm_state
            + l_fin_state + r_fin_state
            + [self._l_gripper, self._r_gripper]
            + obj_vals
            + l_arm_action + r_arm_action
            + l_fin_action + r_fin_action
            + [self._l_gripper, self._r_gripper]
        )
        self.rows.append(row)
        self._frame_idx += 1


# ─────────────────────────────────────────────────────────────────────────────
# Wire up the hook into world.step
# ─────────────────────────────────────────────────────────────────────────────
hook = PlannerDataHook(robot, scene)

_orig_world_step = world.step

def _hooked_step(render=True):
    hook.capture()                  # capture BEFORE physics step
    return _orig_world_step(render=render)

world.step = _hooked_step


# ─────────────────────────────────────────────────────────────────────────────
# Open output CSV
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(args.output), exist_ok=True)

# Determine starting episode index when appending to existing file
_start_episode = 0
if args.append and os.path.exists(args.output) and os.path.getsize(args.output) > 0:
    with open(args.output, "r") as _f:
        _r = csv.DictReader(_f)
        for _row in _r:
            pass
        try:
            _start_episode = int(_row.get("episode_index", -1)) + 1
        except Exception:
            _start_episode = 0
    print(f"[Collect] Appending to existing CSV — starting at episode_index={_start_episode}")

_csv_mode = "a" if (args.append and os.path.exists(args.output)) else "w"
_csv_file = open(args.output, _csv_mode, newline="")
_writer   = csv.writer(_csv_file)
if _csv_mode == "w":
    _writer.writerow(FIELDNAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Episode loop
# ─────────────────────────────────────────────────────────────────────────────
from motion import run_pipeline_from_person2, reset_robot_state_full, patch_robot_workarounds

_action_plan_path = os.path.join(_ROOT, "src", "task1", "primitive_spec_task1.yaml")

# Apply EMA patch once (same as main_fixed.py does inside run_pipeline_from_person2)
patch_robot_workarounds(robot)

total_episodes_ok  = 0
total_frames       = 0

for ep in range(args.episodes):
    abs_ep = _start_episode + ep
    print(f"\n{'='*60}")
    print(f"[Collect] Episode {ep + 1}/{args.episodes}  (episode_index={abs_ep})")
    print(f"{'='*60}")

    # ── Reset robot + re-scatter objects ──────────────────────────────────────
    hook.collecting = False

    # Teleport robot to initial, clear physics state, settle
    reset_robot_state_full(robot, world, verbose=False)

    # Re-scatter parts at new random positions
    scene.scatter_after_reset()

    # Settle ~2s after scatter so objects rest on table
    for _ in range(120):
        _orig_world_step(render=True)

    # Verify scatter succeeded (at least one part not at origin)
    part_poses = scene.get_parts_world_poses()
    _n_valid = sum(1 for p in part_poses if not all(abs(v) < 1e-3 for v in p["position"]))
    if _n_valid == 0:
        print(f"[Collect] ⚠ All parts at origin — scatter failed, skipping episode")
        continue

    print(f"[Collect] {len(part_poses)} parts scattered:")
    for p in part_poses:
        print(f"  {p.get('type','?'):6s} @ {[round(v, 3) for v in p['position']]}")

    # Re-sync IK solver to settled joint positions
    _js = robot.get_joint_states()
    if _js:
        _p = _js["positions"]
        if _p and isinstance(_p[0], list):
            _p = _p[0]
        robot.ik_solver.sync_joint_positions(_js["names"], _p)

    # ── Start data collection for this episode ────────────────────────────────
    hook.reset_episode(abs_ep)
    hook.collecting = True
    _rows_before = len(hook.rows)

    # ── Run planner ───────────────────────────────────────────────────────────
    _success = False
    try:
        run_pipeline_from_person2(
            action_plan_yaml=_action_plan_path,
            robot=robot,
            world=world,
            coord_transform=coord_transform,
        )
        _success = True
    except Exception as _exc:
        import traceback
        print(f"[Collect] Planner error in episode {abs_ep}: {_exc}")
        traceback.print_exc()

    hook.collecting = False
    _frames_this_ep = len(hook.rows) - _rows_before
    print(f"[Collect] Episode {abs_ep}: {'✅' if _success else '⚠️'} "
          f"{_frames_this_ep} frames logged")

    # ── Flush rows to CSV ─────────────────────────────────────────────────────
    for row in hook.rows[_rows_before:]:
        _writer.writerow(row)
    _csv_file.flush()

    total_episodes_ok += 1
    total_frames      += _frames_this_ep

print(f"\n{'='*60}")
print(f"[Collect] Done!")
print(f"  Episodes collected : {total_episodes_ok}/{args.episodes}")
print(f"  Total frames logged: {total_frames}")
print(f"  Output CSV         : {args.output}")
print(f"{'='*60}")

# ── Cleanup ───────────────────────────────────────────────────────────────────
_csv_file.close()

try:
    from isaacsim.core.utils.stage import get_current_stage
    _stage = get_current_stage()
    for _prim in list(_stage.TraverseAll()):
        if _prim.GetName() == "grasp_attach":
            _stage.RemovePrim(_prim.GetPath())
except Exception:
    pass

world.pause()
_dummy_logger.close()
kit.close()
