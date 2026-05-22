"""
run_task1.py — Task 1 main runner: Perception (N1) + Pick-and-Place FSM (N2).

Chạy:
    /isaac-sim/python.sh scripts/run_task1.py

Luồng:
    Build scene → Settle → Detect 4 parts (N1) → FSM: grasp → lift → place × 4
"""

from isaacsim import SimulationApp
kit = SimulationApp(launch_config={"width": 1280, "height": 720, "headless": False})

from isaacsim.core.api import World
import omni
import omni.replicator.core as rep
import numpy as np
import os, sys, time

ROOT = "/home/ubuntu/tai"
sys.path.insert(0, os.path.join(ROOT, "src"))

from baseline_source.config_loader import load_config, apply_scatter_config
from baseline_source.SceneBuilder import SceneBuilder
from baseline_source.RobotArticulation import RobotArticulation
from baseline_source.DataLogger import DataLogger
from baseline_source.coordinate_utils import CoordinateTransform
from task1.camera_utils import CameraIntrinsics

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════
cfg = load_config(os.path.join(ROOT, "configs/Part_Sorting.yaml"))
cfg["root_path"] = os.path.join(ROOT, "assets/resources/")
grasp_cfg = cfg.get("grasp", {})

# ── Bin positions (world coords) — N2 đo và điền vào ──────────────────
# Box ở [1.2, 0.3, 1.05], có vách ngăn chia 2 khu vực
BIN_A_WORLD = np.array([1.15, 0.15, 1.15])   # TODO: N2 đo vị trí thực
BIN_B_WORLD = np.array([1.15, 0.45, 1.15])   # TODO: N2 đo vị trí thực
LIFT_HEIGHT  = grasp_cfg.get("lift_height", 0.17)   # 17cm — đủ >10cm để tính điểm
APPROACH_OFFSET_Z = 0.05                             # tiếp cận từ trên xuống 5cm

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
world.play()
settle_steps = int(grasp_cfg.get("settle_time", 2.0) / world.get_physics_dt())
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

# ═══════════════════════════════════════════════════════════════════════
# Camera + Intrinsics (N1)
# ═══════════════════════════════════════════════════════════════════════
CAMERA_PRIM = "/Root/Ref_Xform/Ref/head_pitch_link/head_stereo_left/head_stereo_left_Camera_01"
_CAM_W, _CAM_H = 640, 480
_rp        = rep.create.render_product(CAMERA_PRIM, (_CAM_W, _CAM_H))
_rgb_ann   = rep.AnnotatorRegistry.get_annotator("rgb")
_depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
_rgb_ann.attach(_rp)
_depth_ann.attach(_rp)

for _ in range(5):
    world.step(render=True)

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

# ═══════════════════════════════════════════════════════════════════════
# T_base_camera + CoordinateTransform
# ═══════════════════════════════════════════════════════════════════════
def _world_tf(path):
    p = stage.GetPrimAtPath(path)
    if not p.IsValid(): return None
    return np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)).T

T_wc = _world_tf(CAMERA_PRIM)
T_wb = _world_tf("/Root/Ref_Xform/Ref/base_link")
T_base_camera = np.linalg.inv(T_wb) @ T_wc

coord_transform = CoordinateTransform.from_torso_link(ik_solver=robot.ik_solver)
print(f"[Coord] R_base=\n{coord_transform.robot_world_R.round(3)}")
_ee_init = robot.get_ee_poses()
print(f"[Coord] EE_init left={np.array(_ee_init['left'][:3]).round(3)}  right={np.array(_ee_init['right'][:3]).round(3)}")
print("[4/5] Transforms ready")

# ═══════════════════════════════════════════════════════════════════════
# N1: Detect parts từ stage (class_id + exact pose)
# ═══════════════════════════════════════════════════════════════════════
def detect_parts_stage():
    """
    Đọc vị trí + class_id từ USD stage.
    Trả về list[dict]: class_id, pos_world, yaw_rad
    """
    num_per_class = cfg["part"].get("num_parts", 2)
    parts = []
    for i, prim_path in enumerate(scene.parts_prim_paths):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        T = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
        pos_world = T[:3, 3]
        R = T[:3, :3]
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        class_id = "part_A" if i < num_per_class else "part_B"
        parts.append({
            "class_id":  class_id,
            "pos_world": pos_world,
            "yaw_rad":   yaw,
            "prim_path": prim_path,
        })
    print(f"[Detect] {len(parts)} parts: {[p['class_id'] for p in parts]}")
    for p in parts:
        print(f"  {p['class_id']}  world={p['pos_world'].round(3)}  yaw={p['yaw_rad']:.2f}rad")
    return parts

# ═══════════════════════════════════════════════════════════════════════
# FSM — N2 điền phần gripper và tuning
# ═══════════════════════════════════════════════════════════════════════
class Task1FSM:
    STATES = ["DETECT", "APPROACH", "GRASP_DOWN", "CLOSE_GRIPPER",
              "LIFT", "MOVE_TO_BIN", "OPEN_GRIPPER", "NEXT", "DONE"]

    def __init__(self):
        self.state       = "DETECT"
        self.parts       = []          # danh sách từ detect_parts_stage()
        self.part_idx    = 0           # đang xử lý part thứ mấy
        self.grasp_arm   = "left"      # tay đang dùng
        self.target_xyzrpy = None      # target 6D hiện tại
        self.hold_steps  = 0           # đếm steps ở trạng thái chờ
        self._arm_init   = {}          # cache initial ee poses

    # ── helpers ───────────────────────────────────────────────────────
    def _ee_pos(self, side):
        """Vị trí end-effector hiện tại trong base frame."""
        js = robot.get_joint_states()
        if js is None:
            return np.array(self._arm_init.get(side, np.zeros(6))[:3])
        robot.ik_solver.sync_joint_positions(js["names"], js["positions"][0])
        se3 = robot.ik_solver.get_ee_pose(side)
        return np.array(se3.translation)

    def _reached(self, side, tol=0.015):
        """Kiểm tra tay đã đến gần target chưa (tolerance mặc định 1.5cm)."""
        if self.target_xyzrpy is None:
            return True
        err = np.linalg.norm(self._ee_pos(side) - self.target_xyzrpy[:3])
        print(f"[reach] {side} err={err:.4f}m tol={tol}")
        return err < tol

    def _world_to_6d(self, pos_world):
        """Chuyển world position → [x,y,z,roll,pitch,yaw] trong pinocchio base frame.

        Giữ nguyên rotation ban đầu của tay — IK chỉ cần giải position,
        tránh yêu cầu flip arm orientation làm Jacobian không hội tụ.
        """
        pos_base = coord_transform.world_to_robot(pos_world)
        init_rpy = np.array(self._arm_init[self.grasp_arm][3:])
        ee_pos   = np.array(self._arm_init[self.grasp_arm][:3])
        print(f"[6D] pos_base={pos_base.round(3)}  ee_init={ee_pos.round(3)}"
              f"  dist={np.linalg.norm(pos_base-ee_pos):.3f}m")
        return np.concatenate([pos_base, init_rpy])

    def _select_arm(self, pos_world):
        """Chọn tay dựa vào vị trí part (y > robot_center → left, ngược lại → right)."""
        pos_base = coord_transform.world_to_robot(pos_world)
        return "left" if pos_base[1] > 0 else "right"

    # ── gripper ───────────────────────────────────────────────────────
    def open_gripper(self, side="left"):
        """TODO N2: điều khiển mở gripper."""
        pass

    def close_gripper(self, side="left"):
        """TODO N2: điều khiển đóng gripper."""
        pass

    # ── FSM step ──────────────────────────────────────────────────────
    def step(self, step_size):
        """Gọi mỗi physics step. Trả về (left_target, right_target) cho IK."""
        if not self._arm_init:
            ee = robot.get_ee_poses()
            self._arm_init = {"left": ee["left"].copy(), "right": ee["right"].copy()}

        left_target  = self._arm_init["left"]
        right_target = self._arm_init["right"]

        # ── DETECT ────────────────────────────────────────────────────
        if self.state == "DETECT":
            for _ in range(10):
                world.step(render=True)     # warm up thêm vài frame
            self.parts = detect_parts_stage()
            if not self.parts:
                print("[FSM] No parts found!")
                self.state = "DONE"
            else:
                self.part_idx = 0
                self.state = "APPROACH"
            return left_target, right_target

        if self.state == "DONE":
            return left_target, right_target

        # ── Lấy part hiện tại ─────────────────────────────────────────
        part = self.parts[self.part_idx]
        self.grasp_arm = self._select_arm(part["pos_world"])

        # ── APPROACH — di chuyển tới vị trí trên part ─────────────────
        if self.state == "APPROACH":
            approach_world = part["pos_world"] + np.array([0, 0, APPROACH_OFFSET_Z])
            self.target_xyzrpy = self._world_to_6d(approach_world)
            print(f"[FSM] APPROACH part {self.part_idx} ({part['class_id']}) "
                  f"arm={self.grasp_arm}")
            self.state = "GRASP_DOWN"

        # ── GRASP_DOWN — hạ xuống vị trí part ────────────────────────
        elif self.state == "GRASP_DOWN":
            if self._reached(self.grasp_arm, tol=0.02):
                grasp_world = part["pos_world"].copy()
                self.target_xyzrpy = self._world_to_6d(grasp_world)
                print(f"[FSM] GRASP_DOWN")
                self.state = "CLOSE_GRIPPER"

        # ── CLOSE_GRIPPER ─────────────────────────────────────────────
        elif self.state == "CLOSE_GRIPPER":
            if self._reached(self.grasp_arm, tol=0.015):
                self.close_gripper(self.grasp_arm)
                self.hold_steps = 0
                self.state = "LIFT"
                print(f"[FSM] CLOSE_GRIPPER")

        # ── LIFT — nhấc lên >10cm ─────────────────────────────────────
        elif self.state == "LIFT":
            self.hold_steps += 1
            if self.hold_steps > 30:    # chờ 30 steps (~0.5s) cho gripper đóng
                lift_world = part["pos_world"] + np.array([0, 0, LIFT_HEIGHT])
                self.target_xyzrpy = self._world_to_6d(lift_world)
                print(f"[FSM] LIFT  height={LIFT_HEIGHT}m")
                self.state = "MOVE_TO_BIN"

        # ── MOVE_TO_BIN ───────────────────────────────────────────────
        elif self.state == "MOVE_TO_BIN":
            if self._reached(self.grasp_arm, tol=0.03):
                bin_world = BIN_A_WORLD if part["class_id"] == "part_A" else BIN_B_WORLD
                self.target_xyzrpy = self._world_to_6d(bin_world)
                print(f"[FSM] MOVE_TO_BIN {part['class_id']} → {bin_world}")
                self.state = "OPEN_GRIPPER"

        # ── OPEN_GRIPPER ──────────────────────────────────────────────
        elif self.state == "OPEN_GRIPPER":
            if self._reached(self.grasp_arm, tol=0.03):
                self.open_gripper(self.grasp_arm)
                self.hold_steps = 0
                self.state = "NEXT"
                print(f"[FSM] OPEN_GRIPPER → placed {part['class_id']}")

        # ── NEXT — chuyển sang part tiếp theo ─────────────────────────
        elif self.state == "NEXT":
            self.hold_steps += 1
            if self.hold_steps > 30:
                self.part_idx += 1
                if self.part_idx >= len(self.parts):
                    print("[FSM] All parts placed! DONE.")
                    self.state = "DONE"
                else:
                    self.state = "APPROACH"

        # ── Build IK targets ──────────────────────────────────────────
        if self.target_xyzrpy is not None:
            if self.grasp_arm == "left":
                left_target = self.target_xyzrpy
            else:
                right_target = self.target_xyzrpy

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
    )

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
