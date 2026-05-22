"""
run_task1_perception.py — Isaac Sim runner for Task 1 perception pipeline.

Chạy:
    /isaac-sim/python.sh scripts/run_task1_perception.py
    /isaac-sim/python.sh scripts/run_task1_perception.py --save-params
    /isaac-sim/python.sh scripts/run_task1_perception.py --frames 5
    /isaac-sim/python.sh scripts/run_task1_perception.py --method color

Author: Thanh Tai (N1)
"""

import argparse
import os
import sys
import numpy as np
import cv2

# ── Parse args trước khi import Isaac Sim ──────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/ubuntu/tai",
                        help="Project root path")
    parser.add_argument("--output-dir", default="lab_outputs/perception",
                        help="Thư mục lưu output (relative to --root)")
    parser.add_argument("--frames", type=int, default=0,
                        help="Số frame rồi thoát (0 = chạy mãi)")
    parser.add_argument("--save-params", action="store_true",
                        help="Lưu camera intrinsics + T_base_camera ra YAML")
    parser.add_argument("--method", default="depth_fg",
                        choices=["color", "depth_fg"],
                        help="Detection method: color (legacy) | depth_fg (recommended)")
    parser.add_argument("--no-perception", action="store_true",
                        help="Chỉ build scene, không chạy perception")
    args, _ = parser.parse_known_args()
    return args


# ── Launch Isaac Sim ────────────────────────────────────────────────────────
from isaacsim import SimulationApp
CONFIG = {"width": 1280, "height": 720, "headless": True}
kit = SimulationApp(launch_config=CONFIG)

from isaacsim.core.api import World
import omni
import omni.replicator.core as rep

args = parse_args()

ROOT = args.root
OUT = os.path.join(ROOT, args.output_dir)
os.makedirs(OUT, exist_ok=True)
p = lambda name: os.path.join(OUT, name)

# Thêm src vào path để import baseline_source + task1
sys.path.insert(0, os.path.join(ROOT, "src"))

from baseline_source.config_loader import load_config, apply_scatter_config
from baseline_source.SceneBuilder import SceneBuilder
from baseline_source.RobotArticulation import RobotArticulation
from baseline_source.DataLogger import DataLogger

from task1.camera_utils import (
    CameraIntrinsics,
    depth_sanity,
    write_depth_sanity_report,
)
from task1.transform_utils import run_transform_sanity
from task1.perception import (
    run_perception,
    save_perception_json,
    save_pose_report_csv,
    save_yaw_report_csv,
    save_failure_cases_jsonl,
)

print("=" * 60)
print("  HRC2026 Task 1 — Perception Runner")
print(f"  Method: {args.method}")
print(f"  Output: {OUT}")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════
# 1. Load config + build scene
# ═══════════════════════════════════════════════════════════════════════
cfg = load_config(os.path.join(ROOT, "configs/Part_Sorting.yaml"))
cfg["root_path"] = os.path.join(ROOT, "assets/resources/")
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

logger = DataLogger(enabled=False, csv_path="/tmp/perception_poses.csv",
                    camera_enabled=False, camera_hdf5_path="/tmp/perception_cam.hdf5")
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
# 2. Build robot + initialize cameras
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
print("[2/5] Robot + cameras initialized")
print(f"      Available cameras: {list(robot.cameras.keys())}")

# Override camera resolution — baseline init không truyền resolution nên default 128×128
_CAM_W, _CAM_H = 640, 480
for _cam_name in ("head_left", "head_right"):
    if _cam_name in robot.cameras:
        robot.cameras[_cam_name].set_resolution((_CAM_W, _CAM_H))
        print(f"      [{_cam_name}] resolution set to {_CAM_W}×{_CAM_H}")

# Vài step để resolution kick in
for _ in range(5):
    world.step(render=True)


# ═══════════════════════════════════════════════════════════════════════
# 3. Camera intrinsics
# ═══════════════════════════════════════════════════════════════════════
CAMERA_NAME = "head_left"
CAMERA_PRIM = "/Root/Ref_Xform/Ref/head_pitch_link/head_stereo_left/head_stereo_left_Camera_01"


def get_intrinsics(camera_obj, width=640, height=480) -> CameraIntrinsics:
    """Lấy intrinsics từ Isaac Sim Camera API, fallback về USD attributes."""
    try:
        K = camera_obj.get_intrinsics_matrix()
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        print(f"      K: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    except Exception as e:
        print(f"      [WARN] get_intrinsics_matrix() failed: {e} → USD fallback")
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath(camera_obj.prim_path)
        fl = cam_prim.GetAttribute("focalLength").Get()
        ha = cam_prim.GetAttribute("horizontalAperture").Get()
        va = cam_prim.GetAttribute("verticalAperture").Get()
        if fl and ha and va:
            fx = (width * fl) / ha
            fy = (height * fl) / va
            cx, cy = width / 2.0, height / 2.0
            print(f"      Computed: fx={fx:.2f} fy={fy:.2f}")
        else:
            # Last resort: head_stereo_left known specs from extracted_camera_params.yaml
            fx, fy = 259.07, 194.30
            cx, cy = 320.0, 240.0
            width, height = 640, 480
            print(f"      [WARN] Using known camera specs fallback: fx={fx:.2f}")
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy,
                            width=width, height=height, depth_unit="meter")


print("[3/5] Getting camera intrinsics...")
intr = get_intrinsics(robot.cameras[CAMERA_NAME])


# ═══════════════════════════════════════════════════════════════════════
# 4. T_base_camera
# ═══════════════════════════════════════════════════════════════════════
def get_T_base_camera(camera_prim_path: str,
                      base_prim_path: str = "/Root/Ref_Xform/Ref/base_link") -> np.ndarray:
    """T_base_camera = inv(T_world_base) @ T_world_camera."""
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()

    def world_tf(path):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"      [WARN] Prim not found: {path}")
            return None
        mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        return np.array(mat).T

    T_wc = world_tf(camera_prim_path)
    T_wb = world_tf(base_prim_path)

    if T_wc is None or T_wb is None:
        print("      [WARN] Cannot compute T_base_camera — using identity")
        return np.eye(4)

    T_bc = np.linalg.inv(T_wb) @ T_wc
    print(f"      det(R)={np.linalg.det(T_bc[:3,:3]):.4f}, "
          f"cam-base dist={np.linalg.norm(T_bc[:3,3]):.3f}m")
    return T_bc


print("[4/5] Computing T_base_camera...")
T_base_camera = get_T_base_camera(CAMERA_PRIM)

# ═══════════════════════════════════════════════════════════════════════
# 5. Save params (--save-params)
# ═══════════════════════════════════════════════════════════════════════
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
            "fx": round(intr.fx, 4), "fy": round(intr.fy, 4),
            "cx": round(intr.cx, 4), "cy": round(intr.cy, 4),
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
    run_transform_sanity(T_base_camera, report_path=p("transform_sanity_report.md"))
    print(f"\n[SAVED] Camera params → {out_yaml}")
    print("  → Copy vào configs/task1_perception.yaml nếu cần")

# HSV config
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
        "implies_class": None,  # ambiguous — handled by shape classifier
    },
}


# ═══════════════════════════════════════════════════════════════════════
# 6. Perception callback
# ═══════════════════════════════════════════════════════════════════════
frame_count = [0]
last_state = [None]


def perception_callback(step_size):
    frame_count[0] += 1
    fid = frame_count[0]

    rgbd = robot.get_camera_rgbd(CAMERA_NAME)
    rgb = rgbd.get("rgb")
    depth = rgbd.get("depth")

    if rgb is None or depth is None:
        print(f"[Frame {fid}] Camera data not ready")
        return

    # Normalise to float32 single-channel depth
    depth = np.array(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    # RGB → BGR for OpenCV
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.shape[2] == 3 else \
          cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)

    state = run_perception(
        bgr, depth, intr, T_base_camera,
        hsv_ranges=HSV_RANGES,
        frame_id=fid,
        camera_name=CAMERA_NAME,
        detection_method=args.method,
    )
    last_state[0] = state

    s = state["summary"]
    n_A = sum(1 for o in state["objects"] if o["class_id"] == "part_A")
    n_B = sum(1 for o in state["objects"] if o["class_id"] == "part_B")
    warn = "  ← expected 4!" if s["num_objects"] != 4 else ""
    print(f"[Frame {fid:4d}] total={s['num_objects']} valid={s['num_valid_objects']} "
          f"A={n_A} B={n_B}{warn}")

    if fid == 1 or fid % 30 == 0:
        _save_artifacts(bgr, depth, state, fid)

    if args.frames > 0 and fid >= args.frames:
        print(f"\n[Done] {args.frames} frames complete — shutting down")
        kit.close()


def _save_artifacts(bgr, depth, state, fid):
    objects = state["objects"]

    cv2.imwrite(p(f"sample_rgb_f{fid:04d}.png"), bgr)
    np.save(p(f"sample_depth_f{fid:04d}.npy"), depth)

    # Depth preview (jet colormap)
    valid = np.isfinite(depth) & (depth > 0)
    d_vis = depth.copy(); d_vis[~valid] = 0
    d_norm = cv2.normalize(d_vis, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(p(f"sample_depth_preview_f{fid:04d}.png"),
                cv2.applyColorMap(d_norm.astype(np.uint8), cv2.COLORMAP_JET))

    if fid == 1:
        info = depth_sanity(depth)
        write_depth_sanity_report(info, p("depth_sanity_report.md"))
        print(f"      Depth: valid={info['valid_ratio']:.1%}, "
              f"median={info['median']}, unit={info['guessed_unit']}")

    # Detection overlay
    overlay = bgr.copy()
    for o in objects:
        if not o["bbox_xyxy"]:
            continue
        x1, y1, x2, y2 = o["bbox_xyxy"]
        u, v = int(o["centroid_px"][0]), int(o["centroid_px"][1])
        col = (0, 255, 0) if o["class_id"] == "part_A" else (255, 128, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 1)
        cv2.circle(overlay, (u, v), 3, (0, 0, 255), -1)
        label = f"{o['class_id'][-1]}{o['confidence']:.2f}"
        cv2.putText(overlay, label, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.imwrite(p(f"overlay_detection_f{fid:04d}.png"), overlay)

    # Canonical (no frame suffix) for easy inspection
    cv2.imwrite(p("sample_rgb.png"), bgr)
    np.save(p("sample_depth.npy"), depth)
    cv2.imwrite(p("overlay_detection.png"), overlay)

    # JSON + CSVs
    save_perception_json(state, p("perception_interface.json"))
    save_pose_report_csv(objects, p("pose_estimator_report.csv"))
    save_yaw_report_csv(objects, p("yaw_report.csv"))
    save_failure_cases_jsonl(objects, p("failure_cases_perception.jsonl"))
    print(f"      Artifacts saved → {OUT}/")


# ═══════════════════════════════════════════════════════════════════════
# 7. Main loop
# ═══════════════════════════════════════════════════════════════════════
print("[5/5] Starting main loop...")

if not args.no_perception:
    world.add_render_callback("perception", perception_callback)

try:
    while kit.is_running():
        world.step()
except KeyboardInterrupt:
    print("\n[Interrupted] Saving final state...")
    if last_state[0]:
        save_perception_json(last_state[0], p("perception_interface_final.json"))
        print(f"  Saved perception_interface_final.json")
finally:
    logger.close()
    kit.close()
