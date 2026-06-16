"""
perception_runner.py
====================
Script perception chính cho Task 1, tích hợp trực tiếp với baseline
của ban tổ chức (Ubtech_sim).

Chạy bằng python.sh:
    /isaac-sim/python.sh perception_runner.py
    /isaac-sim/python.sh perception_runner.py --save-params   # lần đầu: lưu intrinsics
    /isaac-sim/python.sh perception_runner.py --frames 5      # chạy 5 frame rồi thoát

Đặt file này trong:
    /workspace/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim/perception_runner.py

Hoặc chạy từ thư mục tai/:
    /isaac-sim/python.sh perception_runner.py --ubtech-dir /workspace/.../Ubtech_sim
"""

import argparse
import os
import sys
import json
import numpy as np
import cv2

# ── Parse args trước khi import Isaac Sim ────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ubtech-dir",
        default="/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim",
        help="Đường dẫn đến thư mục Ubtech_sim")
    parser.add_argument("--output-dir",
        default="lab_outputs/perception",
        help="Thư mục lưu output")
    parser.add_argument("--frames", type=int, default=0,
        help="Số frame chạy rồi thoát (0 = không thoát tự động)")
    parser.add_argument("--save-params", action="store_true",
        help="Lần đầu chạy: lưu camera intrinsics + T_base_camera ra YAML")
    parser.add_argument("--no-perception", action="store_true",
        help="Chỉ chạy scene, không chạy perception (để test)")
    args, _ = parser.parse_known_args()
    return args


# ── Launch Isaac Sim ─────────────────────────────────────────────────
from isaacsim import SimulationApp
CONFIG = {"width": 1280, "height": 720, "headless": True}
kit = SimulationApp(launch_config=CONFIG)

# Import sau khi SimulationApp khởi động
from isaacsim.core.api import World
import omni
import omni.replicator.core as rep

args = parse_args()

# Thêm Ubtech_sim vào path để import source modules
UBTECH_DIR = args.ubtech_dir
sys.path.insert(0, UBTECH_DIR)
from source.config_loader import load_config, apply_scatter_config
from source.SceneBuilder import SceneBuilder
from source.RobotArticulation import RobotArticulation

# Thêm src của perception pack vào path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))
from camera_utils import CameraIntrinsics, depth_sanity, write_depth_sanity_report
from transform_utils import run_transform_sanity
from perception import run_perception, save_perception_json, save_pose_report_csv, save_yaw_report_csv, save_failure_cases_jsonl
from perception_debug import draw_detection_overlay, draw_centroid_overlay, draw_mask_overlay, save_depth_preview

OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
p = lambda name: os.path.join(OUTPUT_DIR, name)

print("=" * 60)
print("  HRC2026 Task 1 — Perception Runner")
print(f"  Output: {OUTPUT_DIR}")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 1. Load config + build scene (giống main.py)
# ═══════════════════════════════════════════════════════════════
config_path = os.path.join(UBTECH_DIR, "config/Part_Sorting.yaml")
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

from source.DataLogger import DataLogger
data_logger = DataLogger(enabled=False, csv_path="/tmp/perception_poses.csv", camera_enabled=False, camera_hdf5_path="/tmp/perception_camera.hdf5")
scene = SceneBuilder(cfg, data_logger=data_logger)
apply_scatter_config(cfg)
scene.build_all()
rep.orchestrator.step()

print("[1/5] Physics settling...")
settle_steps = int(grasp_cfg.get("settle_time", 2.0) / world.get_physics_dt())
world.play()
for _ in range(settle_steps):
    world.step(render=False)
print(f"      Done ({settle_steps} steps)")

# ═══════════════════════════════════════════════════════════════
# 2. Build robot + initialize cameras
# ═══════════════════════════════════════════════════════════════
world.pause()
scene.build_robot()
robot = RobotArticulation(prim_path="/Root/Ref_Xform/Ref", name="walkerS2")
robot.initialize()   # ← _setup_cameras() được gọi bên trong đây
world.play()

for _ in range(10):
    world.step(render=False)

print("[2/5] Robot + cameras initialized")
print(f"      Available cameras: {list(robot.cameras.keys())}")


# ═══════════════════════════════════════════════════════════════
# 3. Lấy camera intrinsics từ Isaac Sim Camera API
# ═══════════════════════════════════════════════════════════════
def get_intrinsics(camera_obj, width=640, height=480) -> CameraIntrinsics:
    """Lấy intrinsics từ isaacsim.sensors.camera.Camera object."""
    try:
        # Isaac Sim Camera API trả về 3x3 matrix
        K = camera_obj.get_intrinsics_matrix()
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        print(f"      K matrix: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    except Exception as e:
        print(f"      [WARN] get_intrinsics_matrix() failed: {e}")
        print(f"      → Tính từ USD attributes...")
        # Fallback: tính từ USD attributes
        from pxr import UsdGeom
        import math
        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath(camera_obj.prim_path)
        fl = cam_prim.GetAttribute("focalLength").Get()
        ha = cam_prim.GetAttribute("horizontalAperture").Get()
        va = cam_prim.GetAttribute("verticalAperture").Get()
        if fl and ha and va:
            fx = (width * fl) / ha
            fy = (height * fl) / va
            cx, cy = width / 2.0, height / 2.0
            print(f"      Computed: fx={fx:.2f}, fy={fy:.2f}")
        else:
            print(f"      [WARN] USD attrs also None — dùng placeholder")
            fx, fy, cx, cy = 615.0, 615.0, width/2.0, height/2.0

    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy,
                            width=width, height=height, depth_unit="meter")


# ═══════════════════════════════════════════════════════════════
# 4. Lấy T_base_camera từ world transforms
# ═══════════════════════════════════════════════════════════════
def get_T_base_camera(camera_prim_path: str,
                      base_prim_path: str = "/Root/Ref_Xform/Ref/base_link") -> np.ndarray:
    """Tính T_base_camera = inv(T_world_base) @ T_world_camera."""
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()

    def world_transform(path):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"      [WARN] Prim not found: {path}")
            return None
        xf = UsdGeom.Xformable(prim)
        mat = xf.ComputeLocalToWorldTransform(0)
        return np.array(mat).T

    T_wc = world_transform(camera_prim_path)
    T_wb = world_transform(base_prim_path)

    if T_wc is None or T_wb is None:
        print("      [WARN] Cannot compute T_base_camera — using identity")
        return np.eye(4)

    T_bc = np.linalg.inv(T_wb) @ T_wc
    det = np.linalg.det(T_bc[:3, :3])
    dist = np.linalg.norm(T_bc[:3, 3])
    print(f"      det(R)={det:.4f}, camera-base dist={dist:.3f}m")
    return T_bc


CAMERA_NAME = "head_left"  # head_stereo_left
CAMERA_PRIM = "/Root/Ref_Xform/Ref/head_pitch_link/head_stereo_left/head_stereo_left_Camera_01"

print("[3/5] Getting camera intrinsics...")
intr = get_intrinsics(robot.cameras[CAMERA_NAME])

print("[4/5] Computing T_base_camera...")
T_base_camera = get_T_base_camera(CAMERA_PRIM)

# ═══════════════════════════════════════════════════════════════
# 5. Lưu params lần đầu nếu --save-params
# ═══════════════════════════════════════════════════════════════
if args.save_params:
    import yaml

    out_yaml = p("extracted_camera_params.yaml")
    params = {
        "camera": {
            "name": CAMERA_NAME,
            "prim_path": CAMERA_PRIM,
            "rgb_resolution": [intr.width, intr.height],
            "depth_unit": "meter",
        },
        "intrinsics": {
            "fx": round(intr.fx, 4),
            "fy": round(intr.fy, 4),
            "cx": round(intr.cx, 4),
            "cy": round(intr.cy, 4),
        },
        "extrinsics": {
            "T_base_camera_source": "computed_from_world_transforms",
            "robot_base_prim": "/Root/Ref_Xform/Ref/base_link",
            "T_base_camera": T_base_camera.tolist(),
        },
        "sanity": {
            "det_R": round(float(np.linalg.det(T_base_camera[:3, :3])), 6),
            "translation": T_base_camera[:3, 3].tolist(),
        }
    }

    with open(out_yaml, "w") as f:
        yaml.dump(params, f, default_flow_style=False, allow_unicode=True)

    print(f"\n[SAVED] Camera params → {out_yaml}")
    print("  → Copy vào configs/task1_perception.yaml")

    # Run transform sanity report
    run_transform_sanity(T_base_camera, report_path=p("transform_sanity_report.md"))

# HSV config cho perception
HSV_RANGES = {
    "red": {
        "lower": [0, 100, 100], "upper": [10, 255, 255],
        "lower2": [170, 100, 100], "upper2": [179, 255, 255],
        "implies_class": "part_A",
    },
    "blue": {
        "lower": [100, 100, 100], "upper": [130, 255, 255],
        "implies_class": "part_B",
    },
    "ori_color": {
        "lower": [0, 0, 50], "upper": [179, 60, 200],
        "implies_class": None,
    },
}

# ═══════════════════════════════════════════════════════════════
# 6. Perception callback — chạy mỗi frame
# ═══════════════════════════════════════════════════════════════
frame_count = [0]
last_perception_state = [None]

def perception_callback(step_size):
    frame_count[0] += 1
    fid = frame_count[0]

    # Lấy RGB + depth từ camera API
    rgbd = robot.get_head_left_camera_rgbd()
    rgb = rgbd.get("rgb")     # numpy (H, W, 3) hoặc (H, W, 4)
    depth = rgbd.get("depth") # numpy (H, W) float32, đơn vị meter

    if rgb is None or depth is None:
        print(f"[Frame {fid}] Camera data not ready yet")
        return

    # Chuyển về BGR cho OpenCV
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        bgr = rgb[:, :, :3][:, :, ::-1]
    else:
        bgr = rgb[:, :, ::-1]

    # ── Chạy perception ──
    state = run_perception(
        bgr, depth, intr, T_base_camera,
        hsv_ranges=HSV_RANGES,
        frame_id=fid,
        camera_name=CAMERA_NAME,
        detection_method="depth_fg",
    )
    last_perception_state[0] = state

    n = state["summary"]["num_objects"]
    n_valid = state["summary"]["num_valid_objects"]
    n_A = sum(1 for o in state["objects"] if o["class_id"] == "part_A")
    n_B = sum(1 for o in state["objects"] if o["class_id"] == "part_B")

    print(f"[Frame {fid:4d}] objects={n} valid={n_valid} A={n_A} B={n_B}", end="")

    # Warn nếu không thấy đủ 4 vật
    if n != 4:
        print(f"  ← expected 4!", end="")
    print()

    # Lưu artifacts mỗi 30 frame hoặc frame đầu
    if fid == 1 or fid % 30 == 0:
        _save_artifacts(bgr, depth, state, fid)

    # Dừng nếu đã đủ số frame
    if args.frames > 0 and fid >= args.frames:
        print(f"\n[Done] Reached {args.frames} frames — shutting down")
        kit.close()


def _save_artifacts(bgr, depth, state, fid):
    """Lưu tất cả debug artifacts cho 1 frame."""
    objects = state["objects"]

    # Ground truth poses từ scene (để compare)
    try:
        gt_poses = scene.get_parts_world_poses()
        print(f"      GT poses ({len(gt_poses)} parts):")
        for gp in gt_poses:
            print(f"        {gp['prim_path'].split('/')[-1]}: {[round(x,3) for x in gp['position']]}")
    except Exception as e:
        print(f"      GT poses unavailable: {e}")

    # Save images + JSONs
    cv2.imwrite(p(f"sample_rgb_f{fid:04d}.png"), bgr)
    np.save(p(f"sample_depth_f{fid:04d}.npy"), depth)
    save_depth_preview(depth, p(f"sample_depth_preview_f{fid:04d}.png"))

    # Depth sanity (lần đầu)
    if fid == 1:
        info = depth_sanity(depth)
        write_depth_sanity_report(info, p("depth_sanity_report.md"))
        print(f"      Depth: valid={info['valid_ratio']:.1%}, "
              f"median={info['median']}, unit={info['guessed_unit']}")

    # Overlays
    draw_detection_overlay(bgr, objects, p(f"overlay_detection_f{fid:04d}.png"))
    draw_centroid_overlay(bgr, objects, save_path=p(f"overlay_centroid_f{fid:04d}.png"))

    # JSON + CSVs
    save_perception_json(state, p("perception_interface.json"))
    save_pose_report_csv(objects, p("pose_estimator_report.csv"))
    save_yaw_report_csv(objects, p("yaw_report.csv"))
    save_failure_cases_jsonl(objects, p("failure_cases_perception.jsonl"))

    # Copy latest RGB/depth as canonical sample (no frame suffix)
    cv2.imwrite(p("sample_rgb.png"), bgr)
    np.save(p("sample_depth.npy"), depth)

    print(f"      Artifacts saved → {OUTPUT_DIR}/")


# ═══════════════════════════════════════════════════════════════
# 7. Register callback + run loop
# ═══════════════════════════════════════════════════════════════
print("[5/5] Starting main loop...")

if not args.no_perception:
    world.add_render_callback("perception", perception_callback)

try:
    while kit.is_running():
        world.step()
except KeyboardInterrupt:
    print("\n[Interrupted] Saving final state...")
    if last_perception_state[0]:
        save_perception_json(last_perception_state[0],
                             p("perception_interface_final.json"))
        print(f"  Saved final perception_interface.json")
finally:
    kit.close()
