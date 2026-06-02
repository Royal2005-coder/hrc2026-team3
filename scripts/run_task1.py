"""
run_task1.py — Task 1 complete runner: N1 Perception + N2 Planner/FSM + N3 Motion.

Chạy:
    /isaac-sim/python.sh scripts/run_task1.py

FSM (N2 design):
    RESET → OBSERVE → SELECT_OBJECT → EXECUTE_PICK_PLACE
          → VERIFY → OBSERVE (lặp × 4) → DONE
          → RETRY nếu motion fail
          → FAIL nếu hết retry hoặc lỗi nghiêm trọng
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
from task1.planner import Task1Planner
from task1.motion import MotionPrimitiveRunner

# ═══════════════════════════════════════════════════════════════════════
# ── N1/N2/N3 Params — chỉnh tại đây ────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

# N2 — Bin config  (/Root/Box tại world=[1.2, 0.3, 1.05], 2 khu theo Y)
BINS_CONFIG = {
    "class_to_bin": {
        "part_A": "bin_A",
        "part_B": "bin_B",
    },
    "bins": {
        "bin_A": {
            "pos_world":  [1.2,  0.15, 1.1],
            "pose_base":  {"position_m": [1.2,  0.15, 1.1],
                           "quaternion_xyzw": [0, 0, 0, 1]},
        },
        "bin_B": {
            "pos_world":  [1.2,  0.45, 1.1],
            "pose_base":  {"position_m": [1.2,  0.45, 1.1],
                           "quaternion_xyzw": [0, 0, 0, 1]},
        },
    },
}

# N2 — Workspace limits (world frame, bao phủ parts + bins)
WORKSPACE = {
    "x": [0.3, 1.5],
    "y": [-0.2, 0.7],
    "z": [0.8,  1.35],
}

# N3 — Motion params
MOTION_PARAMS = {
    "pre_grasp_height_m":  0.10,   # m trên vật trước khi hạ
    "lift_height_m":       0.20,   # m nhấc lên sau khi gắp
    "pre_place_height_m":  0.12,   # m trên bin trước khi hạ
    "gripper_close_steps": 60,     # ~1s @ 60Hz
    "gripper_open_steps":  30,     # ~0.5s
    "timeout_s":           60.0,   # timeout per pick_place
}

# N3 — Gripper (tune theo URDF finger joint limits)
FINGER_OPEN  = 0.0    # rad
FINGER_CLOSE = 0.7    # rad — tăng nếu part bị rơi (thử 1.0, 1.2)

# Log path
OUT_DIR    = os.path.join(ROOT, "lab_outputs")
TRACE_PATH = os.path.join(OUT_DIR, "planner_trace.jsonl")
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# Scene + World (baseline)
# ═══════════════════════════════════════════════════════════════════════
cfg = load_config(os.path.join(ROOT, "configs/Part_Sorting.yaml"))
cfg["root_path"] = os.path.join(ROOT, "assets/resources/")
grasp_cfg = cfg.get("grasp", {})

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
settle = int(grasp_cfg.get("settle_time", 2.0) / world.get_physics_dt())
world.play()
for _ in range(settle):
    world.step(render=False)
print(f"      Done ({settle} steps)")

# ═══════════════════════════════════════════════════════════════════════
# Robot
# ═══════════════════════════════════════════════════════════════════════
world.pause()
scene.build_robot()
robot = RobotArticulation(prim_path="/Root/Ref_Xform/Ref", name="walkerS2")
robot.initialize()
robot.initialize_ik(os.path.join(cfg["root_path"], "s2.urdf"))
js = robot.get_joint_states()
if js:
    robot.ik_solver.sync_joint_positions(js["names"], js["positions"][0])

world.play()
for _ in range(30):
    world.step(render=True)
print("[2/5] Robot initialized")

# ── Gripper helper ────────────────────────────────────────────────────
_FINGER_JOINTS = {"left":  ["L_finger1_joint", "L_finger2_joint"],
                  "right": ["R_finger1_joint", "R_finger2_joint"]}
_dof_names = robot._articulation.dof_names

def set_gripper(side, pos):
    idxs = [robot._articulation.get_dof_index(j)
             for j in _FINGER_JOINTS[side] if j in _dof_names]
    if idxs:
        robot._articulation.set_joint_positions(
            torch.tensor([pos] * len(idxs), dtype=torch.float32),
            joint_indices=torch.tensor(idxs, dtype=torch.int32))

# ═══════════════════════════════════════════════════════════════════════
# Camera + Annotators
# ═══════════════════════════════════════════════════════════════════════
CAMERA_PRIM = ("/Root/Ref_Xform/Ref/head_pitch_link"
               "/head_stereo_left/head_stereo_left_Camera_01")
W, H = 640, 480
_rp        = rep.create.render_product(CAMERA_PRIM, (W, H))
_rgb_ann   = rep.AnnotatorRegistry.get_annotator("rgb")
_depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
_sem_ann   = rep.AnnotatorRegistry.get_annotator("semantic_segmentation",
                                                  init_params={"colorize": False})
_bbox_ann  = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight_fast",
                                                  init_params={"semanticTypes": ["class"]})
for ann in [_rgb_ann, _depth_ann, _sem_ann, _bbox_ann]:
    ann.attach(_rp)

for _ in range(5):
    world.step(render=True)

# ── Intrinsics ────────────────────────────────────────────────────────
stage = omni.usd.get_context().get_stage()
from pxr import UsdGeom
_cp = stage.GetPrimAtPath(CAMERA_PRIM)
_fl = _cp.GetAttribute("focalLength").Get()
_ha = _cp.GetAttribute("horizontalAperture").Get()
_va = _cp.GetAttribute("verticalAperture").Get()
intr = CameraIntrinsics(
    fx=(W * _fl) / _ha, fy=(H * _fl) / _va,
    cx=W / 2.0, cy=H / 2.0,
    width=W, height=H, depth_unit="meter")
print(f"[3/5] Intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f}")

# ── Transforms ────────────────────────────────────────────────────────
def _world_tf(path):
    p = stage.GetPrimAtPath(path)
    return np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)).T \
           if p.IsValid() else None

T_wc = _world_tf(CAMERA_PRIM)
T_wb = _world_tf("/Root/Ref_Xform/Ref/base_link")
T_base_camera = np.linalg.inv(T_wb) @ T_wc
t_base_world  = np.linalg.inv(T_wb)

coord_transform = CoordinateTransform.from_torso_link(ik_solver=robot.ik_solver)
print("[4/5] Transforms ready")

# ═══════════════════════════════════════════════════════════════════════
# N1 — Detect parts (returns full object list for N2 Planner)
# ═══════════════════════════════════════════════════════════════════════
def observe():
    """
    N1 perception: chạy semantic_bbox detection, trả list object đầy đủ.
    Mỗi object có pos_world (world frame) để N2/N3 dùng.
    """
    for _ in range(5):
        world.step(render=True)
    rep.orchestrator.step()

    rgb_raw   = _rgb_ann.get_data()
    depth_raw = _depth_ann.get_data()
    bbox_raw  = _bbox_ann.get_data()

    if rgb_raw is None or depth_raw is None:
        print("[N1] Camera not ready")
        return []

    rgb   = np.array(rgb_raw,   dtype=np.uint8)
    depth = np.array(depth_raw, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    has_bbox = (bbox_raw is not None and isinstance(bbox_raw, dict)
                and bbox_raw.get("info", {}).get("primPaths"))
    method = "semantic_bbox" if has_bbox else "depth_fg"

    objects = _detect_parts_rgbd(
        rgb=rgb, depth=depth, intr=intr,
        T_base_camera=T_base_camera,
        detection_method=method,
        bbox_ann_data=bbox_raw,
        stage=stage, t_base_world=t_base_world,
    )

    # Thêm pos_world vào mỗi object (N2 và N3 dùng để IK trong world frame)
    for obj in objects:
        pb = obj.get("pose_base", {}).get("position_m")
        if pb:
            pw = (T_wb @ np.append(np.array(pb, dtype=float), 1.0))[:3]
            obj["pos_world"] = pw.tolist()

    print(f"[N1] method={method}  objects={len(objects)}")
    for o in objects:
        pw = o.get("pos_world", [])
        print(f"  {o['object_id']}  {o['class_id']}  conf={o['confidence']:.2f}"
              f"  world={[round(v,3) for v in pw]}")
    return objects


# N1 Fallback: đọc trực tiếp từ USD stage (debug only)
def observe_from_stage():
    num_per = cfg["part"].get("num_parts", 2)
    objects = []
    for i, pp in enumerate(scene.parts_prim_paths):
        prim = stage.GetPrimAtPath(pp)
        if not prim.IsValid():
            continue
        T  = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
        R  = T[:3, :3]
        pw = T[:3, 3]
        pb = (t_base_world @ np.append(pw, 1.0))[:3]
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        cls = "part_A" if i < num_per else "part_B"
        objects.append({
            "object_id":   f"stage_{i:03d}",
            "class_id":    cls,
            "confidence":  1.0,
            "pose_base":   {"position_m": pb.tolist(),
                            "quaternion_xyzw": [0, 0, 0, 1]},
            "grasp_hint":  {"yaw_rad": yaw, "grasp_width_m": 0.045,
                            "approach_axis": "z_down"},
            "failure_reason": None,
            "pos_world":   pw.tolist(),
        })
    print(f"[N1-stage] {len(objects)} parts")
    return objects


# ═══════════════════════════════════════════════════════════════════════
# N2 — Planner + N3 — Motion runner
# ═══════════════════════════════════════════════════════════════════════
planner = Task1Planner(
    bins_config=BINS_CONFIG,
    workspace=WORKSPACE,
    min_confidence=0.70,
    trace_path=TRACE_PATH,
)

motion = MotionPrimitiveRunner(
    robot=robot,
    coord_transform=coord_transform,
    workspace=WORKSPACE,
    params=MOTION_PARAMS,
    finger_open=FINGER_OPEN,
    finger_close=FINGER_CLOSE,
    trace_path=TRACE_PATH,
)


# ═══════════════════════════════════════════════════════════════════════
# N2 — FSM (theo N2 doc: RESET→OBSERVE→SELECT→EXECUTE→VERIFY→DONE/RETRY/FAIL)
# ═══════════════════════════════════════════════════════════════════════
class Task1FSM:
    def __init__(self):
        self.state           = "RESET"
        self._objects        = []
        self._action_plan    = None
        self._handled_count  = 0
        self._current_attempt = 0   # attempt number for motion retry
        self._arm_init       = {}

    def _cache_arm_init(self):
        if not self._arm_init:
            ee = robot.get_ee_poses()
            if ee:
                self._arm_init = {"left": list(ee["left"]),
                                  "right": list(ee["right"])}
            # Mở cả 2 gripper lúc bắt đầu
            set_gripper("left",  FINGER_OPEN)
            set_gripper("right", FINGER_OPEN)

    def get_ik_targets(self, step_size):
        """Gọi từ physics callback. Trả về IK targets cho cả 2 tay."""
        self._cache_arm_init()

        if self.state == "EXECUTE_PICK_PLACE":
            left_t, right_t = motion.step(step_size)

            # Nếu motion xong → chuyển state ngay trong tick này
            if motion.is_done():
                result = motion.get_result()
                print(f"[FSM] Motion done  success={result['primitive_success']}"
                      f"  reason={result['failure_reason']}")
                if motion.is_success():
                    self.state = "VERIFY"
                else:
                    self.state = "RETRY"

            return (left_t  if left_t  is not None
                    else (self._arm_init.get("left")  or [0]*6),
                    right_t if right_t is not None
                    else (self._arm_init.get("right") or [0]*6))

        # Trả về init pose khi không execute motion
        l = self._arm_init.get("left",  [0]*6)
        r = self._arm_init.get("right", [0]*6)
        return l, r

    def tick(self):
        """
        Gọi từ main loop (ngoài physics callback).
        Xử lý các state OBSERVE / SELECT / VERIFY / RETRY / DONE / FAIL.
        """
        if self.state == "RESET":
            planner.reset()
            print("[FSM] RESET → OBSERVE")
            self.state = "OBSERVE"

        elif self.state == "OBSERVE":
            print("[FSM] OBSERVE — running N1 detection...")
            objects = observe()
            if not objects:
                print("[FSM] No objects from camera → try stage fallback")
                objects = observe_from_stage()
            self._objects = objects
            if objects:
                self.state = "SELECT_OBJECT"
            else:
                print("[FSM] No objects at all → FAIL")
                self.state = "FAIL"

        elif self.state == "SELECT_OBJECT":
            action_plan, info = planner.plan(self._objects)
            if action_plan:
                self._action_plan = action_plan
                print(f"[FSM] SELECT OK → PLAN_PICK_PLACE  "
                      f"obj={action_plan['object_id']}  bin={action_plan['target_bin']}")
                self.state = "PLAN_PICK_PLACE"
            else:
                print(f"[FSM] SELECT FAIL: {info['failure_reason']} → FAIL")
                self.state = "FAIL"

        elif self.state == "PLAN_PICK_PLACE":
            ok = motion.start_pick_place(self._action_plan,
                                         attempt=self._current_attempt)
            if ok:
                print(f"[FSM] PLAN → EXECUTE_PICK_PLACE  "
                      f"obj={self._action_plan['object_id']}  attempt={self._current_attempt}")
                self.state = "EXECUTE_PICK_PLACE"
            else:
                print(f"[FSM] PLAN → FAIL (precondition failed)")
                self.state = "FAIL"

        elif self.state == "VERIFY":
            oid = self._action_plan["object_id"]
            planner.mark_handled(oid)
            self._handled_count += 1
            self._current_attempt = 0
            print(f"[FSM] VERIFY OK — {oid} placed ({self._handled_count}/4)")
            if self._handled_count >= 4:
                self.state = "DONE"
            else:
                self.state = "OBSERVE"

        elif self.state == "RETRY":
            oid = self._action_plan["object_id"]
            if planner.can_retry(oid):
                planner.increment_retry(oid)
                self._current_attempt += 1
                print(f"[FSM] RETRY {oid}  attempt={self._current_attempt} → OBSERVE")
                self.state = "OBSERVE"
            else:
                print(f"[FSM] Max retries for {oid} → SELECT_OBJECT")
                self._current_attempt = 0
                self.state = "SELECT_OBJECT"

        # EXECUTE_PICK_PLACE và DONE / FAIL: không cần tick


# ═══════════════════════════════════════════════════════════════════════
# Physics callback + Main loop
# ═══════════════════════════════════════════════════════════════════════
fsm = Task1FSM()
print("[5/5] Starting Task 1  (N1+N2+N3)")
start_time = time.time()


def robot_control_callback(step_size):
    left_t, right_t = fsm.get_ik_targets(step_size)
    robot.control_dual_arm_ik(
        step_size,
        left_target_xyzrpy=left_t,
        right_target_xyzrpy=right_t,
        rot_weight=0.1,     # like baseline: position priority + orientation constraint
        null_weight=0.1,
        max_iter=500,
        pos_tol=5e-3,
        damping=1e-3,
    )
    # Re-apply gripper every step (chống physics creep)
    for side, pos in motion.get_gripper_state().items():
        set_gripper(side, pos)


world.add_physics_callback("task1_fsm", robot_control_callback)

try:
    while kit.is_running():
        # FSM tick: xử lý OBSERVE / SELECT / PLAN / VERIFY / RESET / RETRY
        if fsm.state not in ("EXECUTE_PICK_PLACE", "DONE", "FAIL"):
            fsm.tick()

        # Một physics step (sẽ trigger robot_control_callback)
        world.step()

        if fsm.state == "DONE":
            elapsed = time.time() - start_time
            print(f"\n[DONE] {fsm._handled_count}/4 parts placed  "
                  f"total={elapsed:.1f}s")
            break

        if fsm.state == "FAIL":
            elapsed = time.time() - start_time
            print(f"\n[FAIL] handled={fsm._handled_count}/4  "
                  f"total={elapsed:.1f}s")
            break

finally:
    logger.close()
    kit.close()
