"""
collect_perception_data.py — Thu thập synthetic data từ head_left camera
để train YOLO phân biệt Part A ori vs Part B ori.

Chạy:
    /isaac-sim/python.sh scripts/collect_perception_data.py
    /isaac-sim/python.sh scripts/collect_perception_data.py --frames 300
    /isaac-sim/python.sh scripts/collect_perception_data.py --output dataset/

Output (YOLO format):
    dataset/
    ├── images/train/frame_0000.jpg ...
    ├── labels/train/frame_0000.txt ...
    └── task1.yaml

Label format mỗi dòng: class_id cx cy w h  (normalized 0-1)
    class 0 = part_A
    class 1 = part_B

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
    parser.add_argument("--root", default="/home/ubuntu/tai")
    parser.add_argument("--output", default="dataset",
                        help="Thư mục output dataset (relative to --root)")
    parser.add_argument("--frames", type=int, default=200,
                        help="Số frame thu thập")
    parser.add_argument("--settle-steps", type=int, default=60,
                        help="Physics steps sau mỗi scatter (60 = 1 giây)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Tỉ lệ validation set (0.1 = 10%)")
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
ROOT   = args.root
N_FRAMES   = args.frames
N_VAL      = max(1, int(N_FRAMES * args.val_split))
N_TRAIN    = N_FRAMES - N_VAL

sys.path.insert(0, os.path.join(ROOT, "src"))
from baseline_source.config_loader import load_config, apply_scatter_config
from baseline_source.SceneBuilder import SceneBuilder
from baseline_source.RobotArticulation import RobotArticulation
from baseline_source.DataLogger import DataLogger

# ── Tạo thư mục dataset ────────────────────────────────────────────────────
OUT = os.path.join(ROOT, args.output)
for split in ("train", "val"):
    os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)

print("=" * 60)
print(f"  HRC2026 — Collect Perception Data")
print(f"  Frames: {N_FRAMES} (train={N_TRAIN} val={N_VAL})")
print(f"  Output: {OUT}")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Build scene + robot (giống runner)
# ═══════════════════════════════════════════════════════════════════════════
cfg = load_config(os.path.join(ROOT, "configs/Part_Sorting.yaml"))
cfg["root_path"] = os.path.join(ROOT, "assets/resources/")
grasp_cfg = cfg.get("grasp", {})

omni.usd.get_context().open_stage(
    os.path.join(cfg["root_path"], cfg["scene_usd"])
)
world = World(stage_units_in_meters=1.0, physics_dt=1/60, rendering_dt=1/20)
world.initialize_physics()

logger = DataLogger(enabled=False, csv_path="/tmp/dc_poses.csv",
                    camera_enabled=False, camera_hdf5_path="/tmp/dc_cam.hdf5")
scene = SceneBuilder(cfg, data_logger=logger)
apply_scatter_config(cfg)
scene.build_all()
rep.orchestrator.step()

print("[1/4] Physics settling...")
settle_steps = int(grasp_cfg.get("settle_time", 2.0) / world.get_physics_dt())
world.play()
for _ in range(settle_steps):
    world.step(render=False)

world.pause()
scene.build_robot()
robot = RobotArticulation(prim_path="/Root/Ref_Xform/Ref", name="walkerS2")
robot.initialize()
world.play()
for _ in range(30):
    world.step(render=True)
print("[2/4] Robot + cameras ready")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Camera intrinsics + T_camera_world
# ═══════════════════════════════════════════════════════════════════════════
CAMERA_NAME = "head_left"
CAMERA_PRIM = "/Root/Ref_Xform/Ref/head_pitch_link/head_stereo_left/head_stereo_left_Camera_01"

# Render product 640×480 — bypass baseline 128×128
_CAM_W, _CAM_H = 640, 480
_rp = rep.create.render_product(CAMERA_PRIM, (_CAM_W, _CAM_H))
_rgb_ann   = rep.AnnotatorRegistry.get_annotator("rgb")
_depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
_rgb_ann.attach(_rp)
_depth_ann.attach(_rp)
for _ in range(5):
    world.step(render=True)
print(f"      Render product: {_CAM_W}x{_CAM_H}")

def get_intrinsics(camera_obj, width=640, height=480):
    try:
        K = camera_obj.get_intrinsics_matrix()
        return float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2]), width, height
    except:
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath(camera_obj.prim_path)
        fl = cam_prim.GetAttribute("focalLength").Get()
        ha = cam_prim.GetAttribute("horizontalAperture").Get()
        va = cam_prim.GetAttribute("verticalAperture").Get()
        if fl and ha and va:
            return (width*fl/ha), (height*fl/va), width/2.0, height/2.0, width, height
        return 259.07, 194.30, 320.0, 240.0, 640, 480  # fallback từ extracted_camera_params.yaml

def get_T_world_camera(camera_prim_path):
    """T_world_camera: camera frame → world frame."""
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(camera_prim_path)
    if not prim.IsValid():
        return np.eye(4)
    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    return np.array(mat).T

fx, fy, cx, cy, IMG_W, IMG_H = get_intrinsics(robot.cameras[CAMERA_NAME], width=640, height=480)
T_wc = get_T_world_camera(CAMERA_PRIM)
T_cw = np.linalg.inv(T_wc)  # T_camera_world: world → camera frame
print(f"[3/4] Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} res={IMG_W}x{IMG_H}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Helper functions
# ═══════════════════════════════════════════════════════════════════════════
CLASS_MAP = {
    "part_a": 0,
    "part_b": 1,
}

def get_class_from_prim(prim_path: str) -> str | None:
    """Xác định class từ prim path."""
    p = prim_path.lower()
    if "parta" in p or "part_a" in p:
        return "part_a"
    if "partb" in p or "part_b" in p:
        return "part_b"
    # Replicator prims (Ref_Xform_XX) — class được xác định sau bằng màu
    if "ref_xform" in p or "replicator" in p:
        return "unknown"
    return None


def classify_part_by_color(bgr: np.ndarray, u: float, v: float,
                            patch_r: int = 8) -> str:
    """
    Sample màu tại (u,v) ± patch_r px để xác định class.
    Part A (red/copper): H 0-20 hoặc 155-179, S>60
    Part B (blue/ori):   H 85-135, S>60
    Fallback → part_a (prefer not skipping)
    """
    h, w = bgr.shape[:2]
    y1 = max(0, int(v) - patch_r);  y2 = min(h, int(v) + patch_r)
    x1 = max(0, int(u) - patch_r);  x2 = min(w, int(u) + patch_r)
    patch = bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return "part_a"
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_ch, s_ch = hsv[:,:,0], hsv[:,:,1]
    sat_mask = s_ch > 60
    if sat_mask.sum() == 0:
        return "part_a"
    h_vals = h_ch[sat_mask].astype(float)
    red_px  = ((h_vals <= 20) | (h_vals >= 155)).sum()
    blue_px = ((h_vals >= 85) & (h_vals <= 135)).sum()
    return "part_b" if blue_px > red_px else "part_a"


def world_to_pixel(pos_world: list, T_cw: np.ndarray,
                   fx, fy, cx, cy) -> tuple[float, float, float] | None:
    """
    Chiếu điểm 3D world → pixel (u, v) và depth z_cam.
    Isaac Sim USD dùng OpenGL convention (-Z forward),
    cần convert sang OpenCV (+Z forward) trước khi project.
    """
    p_h = np.array([pos_world[0], pos_world[1], pos_world[2], 1.0])
    p_cam = T_cw @ p_h
    # OpenGL → OpenCV: flip Y và Z
    z = -p_cam[2]
    if z <= 0.05:
        return None
    u = fx * p_cam[0] / z + cx
    v = fy * (-p_cam[1]) / z + cy
    return float(u), float(v), float(z)


def estimate_bbox_pixels(u, v, z, obj_size_m=0.08, fx=fx, fy=fy):
    """
    Ước lượng bbox pixel từ kích thước thật của vật (mặc định 8cm).
    Returns (w_px, h_px).
    """
    half = obj_size_m / 2.0
    w_px = (fx * obj_size_m) / z
    h_px = (fy * obj_size_m) / z
    return float(w_px), float(h_px)


def to_yolo(u, v, w_px, h_px, img_w, img_h):
    """Chuyển bbox pixel → YOLO normalized format (cx, cy, w, h)."""
    cx_n = np.clip(u / img_w, 0.0, 1.0)
    cy_n = np.clip(v / img_h, 0.0, 1.0)
    w_n  = np.clip(w_px / img_w, 0.01, 1.0)
    h_n  = np.clip(h_px / img_h, 0.01, 1.0)
    return cx_n, cy_n, w_n, h_n


def save_frame(bgr, labels, frame_idx, split):
    """Lưu ảnh + label YOLO vào đúng split."""
    name = f"frame_{frame_idx:04d}"
    img_path = os.path.join(OUT, "images", split, f"{name}.jpg")
    lbl_path = os.path.join(OUT, "labels", split, f"{name}.txt")

    cv2.imwrite(img_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    with open(lbl_path, "w") as f:
        for line in labels:
            f.write(line + "\n")
    return len(labels) > 0  # True nếu có label hợp lệ


# ═══════════════════════════════════════════════════════════════════════════
# 4. Thu thập data
# ═══════════════════════════════════════════════════════════════════════════
print(f"[4/4] Collecting {N_FRAMES} frames...\n")

collected = 0
skipped   = 0

for frame_idx in range(N_FRAMES):
    # ── Re-randomize màu + vị trí ──────────────────────────────────────
    try:
        scene.reset()   # _randomize_task1_assets + scatter → đổi cả màu lẫn vị trí
    except Exception as e:
        print(f"  [WARN] scene.reset() failed: {e}")
        try:
            scene._scatter_parts_direct(plane_index=0)
        except Exception as e2:
            print(f"  [WARN] scatter fallback failed: {e2}")

    # ── Settle ──────────────────────────────────────────────────────────
    world.reset()
    scene.scatter_after_reset()
    for _ in range(args.settle_steps):
        world.step(render=False)
    world.step(render=True)

    # ── Lấy ảnh từ Replicator annotators ───────────────────────────────
    world.step(render=True)
    rgb   = _rgb_ann.get_data()
    depth = _depth_ann.get_data()

    if rgb is None:
        skipped += 1
        continue

    depth = np.array(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)

    # ── Lấy GT poses từ scene ───────────────────────────────────────────
    try:
        gt_poses = scene.get_parts_world_poses()
    except Exception as e:
        print(f"  [WARN] get_parts_world_poses failed: {e}")
        skipped += 1
        continue

    if frame_idx == 0:
        print(f"  [DEBUG] gt_poses count={len(gt_poses)}")
        for p in gt_poses[:4]:
            print(f"    prim={p.get('prim_path','?')}  pos={p.get('position','?')}")

    # ── Tạo YOLO labels ─────────────────────────────────────────────────
    yolo_lines = []
    for part in gt_poses:
        class_name = get_class_from_prim(part["prim_path"])
        if class_name is None:
            continue

        result = world_to_pixel(part["position"], T_cw, fx, fy, cx, cy)
        if result is None:
            if frame_idx == 0:
                print(f"  [DEBUG] behind camera: {part.get('prim_path','?')}")
            continue

        u, v, z = result

        # Bỏ qua nếu centroid ngoài ảnh
        if not (0 <= u < IMG_W and 0 <= v < IMG_H):
            if frame_idx == 0:
                print(f"  [DEBUG] out of frame: u={u:.1f} v={v:.1f} ({IMG_W}x{IMG_H})")
            continue

        # Xác định class bằng màu nếu là Replicator prim
        if class_name == "unknown":
            class_name = classify_part_by_color(bgr, u, v)

        # Lấy depth thật tại centroid (nếu có) để bbox chính xác hơn
        u_int, v_int = int(round(u)), int(round(v))
        patch = depth[max(0,v_int-3):v_int+4, max(0,u_int-3):u_int+4]
        valid = patch[(patch > 0) & np.isfinite(patch)]
        z_real = float(np.median(valid)) if len(valid) > 0 else z

        w_px, h_px = estimate_bbox_pixels(u, v, z_real)
        cx_n, cy_n, w_n, h_n = to_yolo(u, v, w_px, h_px, IMG_W, IMG_H)

        class_id = CLASS_MAP[class_name]
        yolo_lines.append(f"{class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

    # ── Bỏ qua frame không có label ─────────────────────────────────────
    if not yolo_lines:
        skipped += 1
        continue

    # ── Lưu vào train hoặc val ──────────────────────────────────────────
    split = "val" if collected < N_VAL else "train"
    save_frame(bgr, yolo_lines, collected, split)
    collected += 1

    if collected % 20 == 0 or collected == N_FRAMES:
        print(f"  [{collected:3d}/{N_FRAMES}] frame_{collected-1:04d} "
              f"labels={len(yolo_lines)} split={split}")

print(f"\n[Done] Collected={collected} Skipped={skipped}")
print(f"  Train: {max(0, collected-N_VAL)} frames")
print(f"  Val:   {min(collected, N_VAL)} frames")

# ═══════════════════════════════════════════════════════════════════════════
# 5. Sinh task1.yaml cho YOLO training
# ═══════════════════════════════════════════════════════════════════════════
yaml_path = os.path.join(OUT, "task1.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {OUT}\n")
    f.write(f"train: images/train\n")
    f.write(f"val:   images/val\n")
    f.write(f"\nnc: 2\n")
    f.write(f"names: ['part_A', 'part_B']\n")

print(f"\n[Saved] {yaml_path}")
print(f"\nTrain YOLO:")
print(f"  yolo train model=yolov8n.pt data={yaml_path} epochs=50 imgsz={IMG_W}")

logger.close()
kit.close()
