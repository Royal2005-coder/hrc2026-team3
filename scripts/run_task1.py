"""
run_task1.py — Task 1 main runner: Perception (N1) + Pick-and-Place FSM (N2).

Chạy:
    /isaac-sim/python.sh scripts/run_task1.py

Luồng:
    Build scene → Settle → Detect 4 parts (N1) → FSM × 4: APPROACH → GRASP_DOWN
    → CLOSE_GRIPPER → LIFT → MOVE_TO_BIN → OPEN_GRIPPER → NEXT → DONE
"""

from isaacsim import SimulationApp
kit = SimulationApp(launch_config={"width": 1280, "height": 720, "headless": False})

from isaacsim.core.api import World
import omni
import omni.replicator.core as rep
import numpy as np
import torch
import os, sys, time

ROOT = "/home/ubuntu/tai"
sys.path.insert(0, os.path.join(ROOT, "src"))

from baseline_source.config_loader import load_config, apply_scatter_config
from baseline_source.SceneBuilder import SceneBuilder
from baseline_source.RobotArticulation import RobotArticulation
from baseline_source.DataLogger import DataLogger
from baseline_source.coordinate_utils import CoordinateTransform
from task1.camera_utils import CameraIntrinsics
from task1.perception import detect_parts as _detect_parts_rgbd

# ═══════════════════════════════════════════════════════════════════════
# Config (baseline — không sửa Part_Sorting.yaml)
# ═══════════════════════════════════════════════════════════════════════
cfg = load_config(os.path.join(ROOT, "configs/Part_Sorting.yaml"))
cfg["root_path"] = os.path.join(ROOT, "assets/resources/")
grasp_cfg = cfg.get("grasp", {})  # baseline params (settle_time, ik_rot_weight, ...)

# ── N1/N2 params — chỉnh tại đây, KHÔNG sửa Part_Sorting.yaml ────────
# /Root/Box tại world=[1.2, 0.3, 1.05], có vách ngăn chia 2 khu theo trục Y
# z=1.1 để gripper release hơi trên miệng box (không va vào thành)
BIN_A_WORLD = np.array([1.2,  0.15, 1.1])   # khu trước (y nhỏ)
BIN_B_WORLD = np.array([1.2,  0.45, 1.1])   # khu sau  (y lớn)

LIFT_HEIGHT       = 0.20   # m — nhấc lên 20cm so với mặt part
APPROACH_OFFSET_Z = 0.10   # m — tiếp cận từ trên 10cm trước khi hạ

# Gripper — tune theo URDF finger joint limits (xem s2.urdf để biết max)
FINGER_OPEN  = 0.0    # rad — mở hoàn toàn
FINGER_CLOSE = 0.7    # rad — tune nếu part bị rơi (tăng lên 1.0, 1.2, ...)
GRIPPER_CLOSE_STEPS = 60   # physics steps chờ gripper đóng (~1s @ 60Hz)
GRIPPER_OPEN_STEPS  = 30   # physics steps chờ gripper mở (~0.5s)

# ═══════════════════════════════════════════════════════════════════════
# Scene + World
# ═══════════════════════════════════════════════════════════════════════
omni.usd.get_context().open_stage(os.path.join(cfg["root_path"], cfg["scene_usd"]))
world = World(stage_units_in_meters=1.0, physics_dt=1/60, rendering_dt=1/20)
world.initialize_physics()

logger = DataLogger(enabled=False, csv_path="/tmp/task1.csv",
                    camera_enabled=False, camera_hdf5_path="/tmp/task1.hdf5")
scene = SceneBuilder(cfg, data_logger=logger)
apply_scatter_config(cfg)
scene.build_all()
rep.orchestrator.step()

print("[1/5] Physics settling...")
settle_steps = int(grasp_cfg.get("settle_time", 2.0) / world.get_physics_dt())
world.play()
for _ in range(settle_steps):
    world.step(render=False)
print(f"      Done ({settle_steps} steps)")

# ═══════════════════════════════════════════════════════════════════════
# Robot
# ═══════════════════════════════════════════════════════════════════════
world.pause()
scene.build_robot()
robot = RobotArticulation(prim_path="/Root/Ref_Xform/Ref", name="walkerS2")
robot.initialize()

urdf_path = os.path.join(cfg["root_path"], "s2.urdf")
robot.initialize_ik(urdf_path)
js = robot.get_joint_states()
if js:
    robot.ik_solver.sync_joint_positions(js["names"], js["positions"][0])

world.play()
for _ in range(30):
    world.step(render=True)
print("[2/5] Robot initialized")

# ── Gripper helpers ───────────────────────────────────────────────────
_GRIPPER_JOINTS = {
    "left":  ["L_finger1_joint", "L_finger2_joint"],
    "right": ["R_finger1_joint", "R_finger2_joint"],
}
_dof_names = robot._articulation.dof_names


def _gripper_indices(side: str) -> list[int]:
    return [robot._articulation.get_dof_index(j)
            for j in _GRIPPER_JOINTS[side] if j in _dof_names]


def set_gripper(side: str, pos: float):
    """Set both finger joints of one gripper to pos."""
    idxs = _gripper_indices(side)
    if not idxs:
        print(f"[Gripper] WARNING: no finger joints found for {side}")
        return
    robot._articulation.set_joint_positions(
        torch.tensor([pos] * len(idxs), dtype=torch.float32),
        joint_indices=torch.tensor(idxs, dtype=torch.int32),
    )

# ═══════════════════════════════════════════════════════════════════════
# Camera + Annotators
# ═══════════════════════════════════════════════════════════════════════
CAMERA_PRIM = "/Root/Ref_Xform/Ref/head_pitch_link/head_stereo_left/head_stereo_left_Camera_01"
_CAM_W, _CAM_H = 640, 480
_rp        = rep.create.render_product(CAMERA_PRIM, (_CAM_W, _CAM_H))
_rgb_ann   = rep.AnnotatorRegistry.get_annotator("rgb")
_depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
_sem_ann   = rep.AnnotatorRegistry.get_annotator("semantic_segmentation",
                                                  init_params={"colorize": False})
_bbox_ann  = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight_fast",
                                                  init_params={"semanticTypes": ["class"]})
_rgb_ann.attach(_rp)
_depth_ann.attach(_rp)
_sem_ann.attach(_rp)
_bbox_ann.attach(_rp)

for _ in range(5):
    world.step(render=True)

# ── Intrinsics ────────────────────────────────────────────────────────
stage = omni.usd.get_context().get_stage()
from pxr import UsdGeom
_cam_prim = stage.GetPrimAtPath(CAMERA_PRIM)
_fl = _cam_prim.GetAttribute("focalLength").Get()
_ha = _cam_prim.GetAttribute("horizontalAperture").Get()
_va = _cam_prim.GetAttribute("verticalAperture").Get()
intr = CameraIntrinsics(
    fx=(_CAM_W * _fl) / _ha, fy=(_CAM_H * _fl) / _va,
    cx=_CAM_W / 2.0, cy=_CAM_H / 2.0,
    width=_CAM_W, height=_CAM_H, depth_unit="meter",
)
print(f"[3/5] Intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f}")

# ── Transforms ────────────────────────────────────────────────────────
def _world_tf(path):
    p = stage.GetPrimAtPath(path)
    if not p.IsValid(): return None
    return np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)).T

T_wc = _world_tf(CAMERA_PRIM)
T_wb = _world_tf("/Root/Ref_Xform/Ref/base_link")
T_base_camera = np.linalg.inv(T_wb) @ T_wc
t_base_world  = np.linalg.inv(T_wb)

coord_transform = CoordinateTransform.from_torso_link(ik_solver=robot.ik_solver)
print("[4/5] Transforms ready")

# ── Top-down grasp RPY ────────────────────────────────────────────────
def _top_down_rpy(pos_base: np.ndarray) -> np.ndarray:
    """
    Compute [roll, pitch, yaw] for a top-down grasp at pos_base (Pinocchio frame).
    End-effector Z-axis points in the world-down direction.
    """
    import pinocchio as pin
    world_down = np.array([0.0, 0.0, -1.0])
    base_down  = coord_transform.robot_world_R_inv @ world_down
    base_down  = base_down / np.linalg.norm(base_down)

    reach = pos_base.copy()
    reach_horiz = reach - np.dot(reach, base_down) * base_down
    if np.linalg.norm(reach_horiz) < 1e-6:
        reach_horiz = np.array([1.0, 0.0, 0.0])
    x_grasp = reach_horiz / np.linalg.norm(reach_horiz)
    y_grasp = np.cross(base_down, x_grasp)
    y_grasp = y_grasp / np.linalg.norm(y_grasp)

    R_grasp = np.column_stack([x_grasp, y_grasp, base_down])
    return pin.rpy.matrixToRpy(R_grasp)


# ═══════════════════════════════════════════════════════════════════════
# N1 — Detect parts from camera (semantic_bbox → prim transform)
# ═══════════════════════════════════════════════════════════════════════
def detect_parts_camera() -> list[dict]:
    """
    Detect 4 workpieces using bounding_box_2d_tight_fast + USD prim pose.
    Returns list[dict]: class_id, pos_world, yaw_rad.
    Fallback to depth_fg if annotator has no primPaths.
    """
    for _ in range(5):
        world.step(render=True)
    rep.orchestrator.step()

    rgb_raw   = _rgb_ann.get_data()
    depth_raw = _depth_ann.get_data()
    bbox_raw  = _bbox_ann.get_data()

    if rgb_raw is None or depth_raw is None:
        print("[N1] Camera data not ready")
        return []

    rgb   = np.array(rgb_raw,   dtype=np.uint8)
    depth = np.array(depth_raw, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    has_bbox = (bbox_raw is not None
                and isinstance(bbox_raw, dict)
                and bbox_raw.get("info", {}).get("primPaths"))
    method = "semantic_bbox" if has_bbox else "depth_fg"
    print(f"[N1] method={method}")

    objects = _detect_parts_rgbd(
        rgb=rgb, depth=depth, intr=intr,
        T_base_camera=T_base_camera,
        detection_method=method,
        bbox_ann_data=bbox_raw,
        stage=stage,
        t_base_world=t_base_world,
    )

    parts = []
    for obj in objects:
        if obj.get("pose_base") is None:
            continue
        pos_base  = np.array(obj["pose_base"]["position_m"])
        pos_world = (T_wb @ np.append(pos_base, 1.0))[:3]
        parts.append({
            "class_id":  obj["class_id"],
            "pos_world": pos_world,
            "yaw_rad":   obj["grasp_hint"]["yaw_rad"],
        })

    print(f"[N1] Detected {len(parts)} parts: {[p['class_id'] for p in parts]}")
    for p in parts:
        print(f"  {p['class_id']}  world={p['pos_world'].round(3)}  "
              f"yaw={p['yaw_rad']:.2f}rad")
    return parts


# ── Fallback: read directly from USD stage (debug / eval only) ────────
def detect_parts_stage() -> list[dict]:
    num_per_class = cfg["part"].get("num_parts", 2)
    parts = []
    for i, prim_path in enumerate(scene.parts_prim_paths):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        T = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
        R = T[:3, :3]
        parts.append({
            "class_id":  "part_A" if i < num_per_class else "part_B",
            "pos_world": T[:3, 3],
            "yaw_rad":   float(np.arctan2(R[1, 0], R[0, 0])),
            "prim_path": prim_path,
        })
    print(f"[Stage] {len(parts)} parts: {[p['class_id'] for p in parts]}")
    for p in parts:
        print(f"  {p['class_id']}  world={p['pos_world'].round(3)}  yaw={p['yaw_rad']:.2f}rad")
    return parts


# ═══════════════════════════════════════════════════════════════════════
# N2 — Task 1 FSM
# ═══════════════════════════════════════════════════════════════════════
class Task1FSM:
    """
    Pick-and-place FSM for 4 workpieces.

    State sequence per part:
      DETECT → APPROACH → GRASP_DOWN → CLOSE_GRIPPER → LIFT
             → MOVE_TO_BIN → OPEN_GRIPPER → NEXT → [repeat] → DONE
    """

    def __init__(self):
        self.state       = "DETECT"
        self.parts: list = []
        self.part_idx    = 0
        self.grasp_arm   = "left"
        self.target_6d   = None   # current IK target [x,y,z,r,p,y] in Pinocchio base frame
        self.hold_steps  = 0
        self._arm_init   = {}
        self._reach_log_throttle = 0
        self._gripper_state = {"left": FINGER_OPEN, "right": FINGER_OPEN}

    # ── pose helpers ──────────────────────────────────────────────────
    def _ee_pos(self, side: str) -> np.ndarray:
        js = robot.get_joint_states()
        if js is None:
            return np.array(self._arm_init.get(side, np.zeros(6))[:3])
        robot.ik_solver.sync_joint_positions(js["names"], js["positions"][0])
        return np.array(robot.ik_solver.get_ee_pose(side).translation)

    def _reached(self, side: str, tol: float = 0.025) -> bool:
        if self.target_6d is None:
            return True
        err = float(np.linalg.norm(self._ee_pos(side) - self.target_6d[:3]))
        self._reach_log_throttle += 1
        if self._reach_log_throttle >= 30:
            print(f"[reach] {side} err={err:.4f}m tol={tol}")
            self._reach_log_throttle = 0
        return err < tol

    def _world_to_6d(self, pos_world: np.ndarray,
                     top_down: bool = False) -> np.ndarray:
        """World pos → 6D target [x,y,z,roll,pitch,yaw] in Pinocchio base frame."""
        pos_base = coord_transform.world_to_robot(pos_world)
        rpy = _top_down_rpy(pos_base) if top_down else \
              np.array(self._arm_init[self.grasp_arm][3:])
        return np.concatenate([pos_base, rpy])

    def _select_arm(self, pos_world: np.ndarray) -> str:
        pos_base = coord_transform.world_to_robot(pos_world)
        return "left" if pos_base[1] > 0 else "right"

    # ── gripper ───────────────────────────────────────────────────────
    def open_gripper(self, side: str = "left"):
        self._gripper_state[side] = FINGER_OPEN
        set_gripper(side, FINGER_OPEN)
        print(f"[Gripper] {side} OPEN ({FINGER_OPEN})")

    def close_gripper(self, side: str = "left"):
        self._gripper_state[side] = FINGER_CLOSE
        set_gripper(side, FINGER_CLOSE)
        print(f"[Gripper] {side} CLOSE ({FINGER_CLOSE})")

    # ── FSM step ──────────────────────────────────────────────────────
    def step(self, step_size: float):
        """Called every physics step. Returns (left_target_6d, right_target_6d)."""
        if not self._arm_init:
            ee = robot.get_ee_poses()
            self._arm_init = {"left": list(ee["left"]), "right": list(ee["right"])}
            # Open both grippers at start
            self.open_gripper("left")
            self.open_gripper("right")

        left_target  = self._arm_init["left"]
        right_target = self._arm_init["right"]

        # ── DETECT ────────────────────────────────────────────────────
        if self.state == "DETECT":
            for _ in range(10):
                world.step(render=True)
            self.parts = detect_parts_camera()
            if not self.parts:
                print("[FSM] No parts detected — fallback to stage")
                self.parts = detect_parts_stage()
            if not self.parts:
                print("[FSM] Still no parts — DONE")
                self.state = "DONE"
            else:
                self.part_idx = 0
                self.state = "APPROACH"
            return left_target, right_target

        if self.state == "DONE":
            return left_target, right_target

        # ── Current part ──────────────────────────────────────────────
        part = self.parts[self.part_idx]
        self.grasp_arm = self._select_arm(part["pos_world"])

        # ── APPROACH — di chuyển tới vị trí trên part ─────────────────
        if self.state == "APPROACH":
            approach_pos = part["pos_world"] + np.array([0, 0, APPROACH_OFFSET_Z])
            self.target_6d = self._world_to_6d(approach_pos, top_down=True)
            print(f"[FSM] APPROACH part[{self.part_idx}] ({part['class_id']}) "
                  f"arm={self.grasp_arm}  approach={approach_pos.round(3)}")
            self.state = "GRASP_DOWN"

        # ── GRASP_DOWN — chờ đến approach rồi hạ xuống part ──────────
        elif self.state == "GRASP_DOWN":
            if self._reached(self.grasp_arm, tol=0.03):
                grasp_pos = part["pos_world"].copy()
                self.target_6d = self._world_to_6d(grasp_pos, top_down=True)
                print(f"[FSM] GRASP_DOWN  grasp={grasp_pos.round(3)}")
                self.state = "CLOSE_GRIPPER"

        # ── CLOSE_GRIPPER — chờ tới grasp pos rồi đóng gripper ────────
        elif self.state == "CLOSE_GRIPPER":
            if self._reached(self.grasp_arm, tol=0.025):
                self.close_gripper(self.grasp_arm)
                self.hold_steps = 0
                self.state = "LIFT"
                print("[FSM] CLOSE_GRIPPER — waiting for gripper to close")

        # ── LIFT — chờ gripper đóng rồi nhấc lên ─────────────────────
        elif self.state == "LIFT":
            self.hold_steps += 1
            if self.hold_steps >= GRIPPER_CLOSE_STEPS:
                lift_pos = part["pos_world"] + np.array([0, 0, LIFT_HEIGHT])
                self.target_6d = self._world_to_6d(lift_pos, top_down=True)
                print(f"[FSM] LIFT  height={LIFT_HEIGHT}m")
                self.state = "MOVE_TO_BIN"

        # ── MOVE_TO_BIN — di chuyển đến bin tương ứng ─────────────────
        elif self.state == "MOVE_TO_BIN":
            if self._reached(self.grasp_arm, tol=0.04):
                bin_world = (BIN_A_WORLD if part["class_id"] == "part_A"
                             else BIN_B_WORLD)
                self.target_6d = self._world_to_6d(bin_world, top_down=True)
                print(f"[FSM] MOVE_TO_BIN {part['class_id']} → {bin_world}")
                self.state = "OPEN_GRIPPER"

        # ── OPEN_GRIPPER — chờ đến bin rồi thả ───────────────────────
        elif self.state == "OPEN_GRIPPER":
            if self._reached(self.grasp_arm, tol=0.04):
                self.open_gripper(self.grasp_arm)
                self.hold_steps = 0
                self.state = "RETREAT"
                print(f"[FSM] OPEN_GRIPPER — {part['class_id']} placed")

        # ── RETREAT — nhấc tay lên khỏi bin trước khi sang part tiếp ──
        elif self.state == "RETREAT":
            self.hold_steps += 1
            if self.hold_steps >= GRIPPER_OPEN_STEPS:
                bin_world = (BIN_A_WORLD if part["class_id"] == "part_A"
                             else BIN_B_WORLD)
                retreat_pos = bin_world + np.array([0, 0, APPROACH_OFFSET_Z])
                self.target_6d = self._world_to_6d(retreat_pos, top_down=False)
                self.hold_steps = 0
                self.state = "NEXT"

        # ── NEXT — sang part tiếp ──────────────────────────────────────
        elif self.state == "NEXT":
            self.hold_steps += 1
            if self.hold_steps >= 20:
                self.part_idx += 1
                if self.part_idx >= len(self.parts):
                    print("[FSM] All parts placed! DONE.")
                    self.state = "DONE"
                else:
                    print(f"[FSM] Moving to part[{self.part_idx}]")
                    self.state = "APPROACH"

        # ── Build IK targets ──────────────────────────────────────────
        if self.target_6d is not None:
            if self.grasp_arm == "left":
                left_target = self.target_6d
            else:
                right_target = self.target_6d

        return left_target, right_target


# ═══════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════
fsm = Task1FSM()
print("[5/5] Starting Task 1 FSM...")
start_time = time.time()


def robot_control_callback(step_size):
    left_t, right_t = fsm.step(step_size)
    robot.control_dual_arm_ik(
        step_size,
        left_target_xyzrpy=left_t,
        right_target_xyzrpy=right_t,
        rot_weight=grasp_cfg.get("ik_rot_weight", 0.1),
        null_weight=0.0,
        max_iter=300,
    )
    # Re-apply gripper every step to resist physics-driven creep
    for side, pos in fsm._gripper_state.items():
        set_gripper(side, pos)


world.add_physics_callback("task1_fsm", robot_control_callback)

try:
    while kit.is_running():
        world.step()
        if fsm.state == "DONE":
            elapsed = time.time() - start_time
            print(f"\n[DONE] Total time: {elapsed:.1f}s")
            break
finally:
    logger.close()
    kit.close()
