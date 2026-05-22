"""
calibrate_hsv.py — Capture 1 frame, sample HSV from each part, print ranges.

Chạy:
    /isaac-sim/python.sh scripts/calibrate_hsv.py
"""
import os, sys
import numpy as np
import cv2

from isaacsim import SimulationApp
kit = SimulationApp({"width": 1280, "height": 720, "headless": True})

from isaacsim.core.api import World
import omni
import omni.replicator.core as rep

ROOT = "/home/ubuntu/tai"
OUT  = os.path.join(ROOT, "lab_outputs/perception")
LOG  = os.path.join(ROOT, "lab_outputs/calib_out.txt")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "src"))
from baseline_source.config_loader import load_config, apply_scatter_config
from baseline_source.SceneBuilder import SceneBuilder
from baseline_source.RobotArticulation import RobotArticulation
from baseline_source.DataLogger import DataLogger
from task1.camera_utils import CameraIntrinsics

# ── Build scene ───────────────────────────────────────────────────────────────
cfg = load_config(os.path.join(ROOT, "configs/Part_Sorting.yaml"))
cfg["root_path"] = os.path.join(ROOT, "assets/resources/")

omni.usd.get_context().open_stage(
    os.path.join(cfg["root_path"], cfg["scene_usd"]))
world = World(stage_units_in_meters=1.0, physics_dt=1/60, rendering_dt=1/20)
world.initialize_physics()

logger = DataLogger(enabled=False, csv_path="/tmp/x.csv",
                    camera_enabled=False, camera_hdf5_path="/tmp/x.hdf5")
scene = SceneBuilder(cfg, data_logger=logger)
apply_scatter_config(cfg)
scene.build_all()
rep.orchestrator.step()

world.play()
for _ in range(int(2.0 / world.get_physics_dt())):
    world.step(render=False)

world.pause()
scene.build_robot()
robot = RobotArticulation(prim_path="/Root/Ref_Xform/Ref", name="walkerS2")
robot.initialize()

CAMERA_PRIM = "/Root/Ref_Xform/Ref/head_pitch_link/head_stereo_left/head_stereo_left_Camera_01"
_CAM_W, _CAM_H = 640, 480
_rp        = rep.create.render_product(CAMERA_PRIM, (_CAM_W, _CAM_H))
_rgb_ann   = rep.AnnotatorRegistry.get_annotator("rgb")
_depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
_rgb_ann.attach(_rp)
_depth_ann.attach(_rp)

world.play()
for _ in range(30):
    world.step(render=True)

# ── Capture 1 frame ───────────────────────────────────────────────────────────
world.step(render=True)
rgb   = _rgb_ann.get_data()
depth = _depth_ann.get_data()

bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

cv2.imwrite(os.path.join(OUT, "calib_rgb.png"), bgr)
print(f"[SAVED] calib_rgb.png")

# ── Get part positions from stage ─────────────────────────────────────────────
from pxr import UsdGeom
stage = omni.usd.get_context().get_stage()

# Camera intrinsics
intr = CameraIntrinsics(fx=259.07, fy=194.30, cx=320.0, cy=240.0,
                        width=_CAM_W, height=_CAM_H)

# T_base_camera
def world_tf(path):
    p = stage.GetPrimAtPath(path)
    if not p.IsValid(): return None
    m = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)
    return np.array(m).T

T_wc = world_tf(CAMERA_PRIM)
T_wb = world_tf("/Root/Ref_Xform/Ref/base_link")
T_bc = np.linalg.inv(T_wb) @ T_wc
T_cb = np.linalg.inv(T_bc)
T_bw = np.linalg.inv(T_wb)

# ── Dump ALL top-level prims to find actual part paths ────────────────────────
print("\n=== ALL stage prims (top 2 levels) ===")
all_paths = []
for prim in stage.TraverseAll():
    path_str = str(prim.GetPath())
    depth = path_str.count("/")
    if depth > 3:
        continue  # skip deep children, only show top structure
    all_paths.append(path_str)
    print(f"  {path_str}  type={prim.GetTypeName()}")

print(f"\n  Total top-level prims: {len(all_paths)}")

# Also dump ROOT children specifically
print("\n=== /Root children ===")
root_prim = stage.GetPrimAtPath("/Root")
if root_prim.IsValid():
    for child in root_prim.GetChildren():
        path_str = str(child.GetPath())
        xf = UsdGeom.Xformable(child)
        try:
            T = np.array(xf.ComputeLocalToWorldTransform(0)).T
            pos = T[:3, 3]
            print(f"  {path_str}  type={child.GetTypeName()}  pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
        except Exception:
            print(f"  {path_str}  type={child.GetTypeName()}")
else:
    print("  /Root not found — trying /World")
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim.IsValid():
        for child in world_prim.GetChildren():
            print(f"  /World/{child.GetName()}  type={child.GetTypeName()}")

print("\n=== Searching stage for part prims (broad) ===")
found_parts = []
for prim in stage.TraverseAll():
    path_str = str(prim.GetPath())
    name_lower = path_str.lower()
    # broad match — any object that might be a workpiece
    if any(kw in name_lower for kw in ["part", "workpiece", "sorting", "obj", "cylinder", "box", "item"]):
        xf = UsdGeom.Xformable(prim)
        try:
            T = np.array(xf.ComputeLocalToWorldTransform(0)).T
            pos = T[:3, 3]
            # Only include prims that are off the origin (likely scattered objects)
            if abs(pos[0]) + abs(pos[1]) + abs(pos[2]) > 0.01:
                print(f"  {path_str}  pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
                found_parts.append(path_str)
        except Exception:
            pass

print(f"  Total found: {len(found_parts)}")

# Auto-detect part_A and part_B paths from stage
import re
part_prims = []
for path in found_parts:
    low = path.lower()
    if re.search(r'part.?a', low) or re.search(r'part_a', low):
        part_prims.append((path, "part_A"))
    elif re.search(r'part.?b', low) or re.search(r'part_b', low):
        part_prims.append((path, "part_B"))

# Fallback: use first 2 and second 2 found if labels not in name
if not part_prims and len(found_parts) >= 4:
    for i, path in enumerate(found_parts[:4]):
        cid = "part_A" if i < 2 else "part_B"
        part_prims.append((path, cid))

print(f"\n  Using {len(part_prims)} parts for HSV sampling:")
for pp, cid in part_prims:
    print(f"    {pp} → {cid}")

# ── Sample HSV at each part centroid ─────────────────────────────────────────
RADIUS = 8   # sample patch radius (pixels)
results = {}

print("\n=== HSV samples per part ===")
vis = bgr.copy()

for prim_path, class_id in part_prims:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"  {prim_path}: NOT FOUND")
        continue

    T_wobj  = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
    pos_w   = T_wobj[:3, 3]
    p_base  = (T_bw @ np.r_[pos_w, 1.0])[:3]
    p_cam   = (T_cb @ np.r_[p_base, 1.0])[:3]

    z = -p_cam[2]
    if z <= 0.05:
        print(f"  {prim_path}: behind camera (z={z:.3f})")
        continue

    u = int(intr.fx * p_cam[0] / z + intr.cx)
    v = int(intr.fy * (-p_cam[1]) / z + intr.cy)

    if not (RADIUS <= u < _CAM_W - RADIUS and RADIUS <= v < _CAM_H - RADIUS):
        print(f"  {prim_path}: out of frame (u={u}, v={v})")
        continue

    patch = hsv[v-RADIUS:v+RADIUS+1, u-RADIUS:u+RADIUS+1]
    h_vals = patch[:, :, 0].flatten()
    s_vals = patch[:, :, 1].flatten()
    v_vals = patch[:, :, 2].flatten()

    h_med = int(np.median(h_vals))
    s_med = int(np.median(s_vals))
    v_med = int(np.median(v_vals))
    h_min, h_max = int(np.percentile(h_vals, 5)), int(np.percentile(h_vals, 95))
    s_min, s_max = int(np.percentile(s_vals, 5)), int(np.percentile(s_vals, 95))
    v_min, v_max = int(np.percentile(v_vals, 5)), int(np.percentile(v_vals, 95))

    col = (0, 200, 0) if class_id == "part_A" else (200, 50, 50)
    cv2.circle(vis, (u, v), RADIUS, col, 2)
    cv2.putText(vis, f"{class_id[-1]}({h_med},{s_med},{v_med})",
                (u+RADIUS+2, v), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    key = f"{prim_path}({class_id})"
    results[key] = dict(h_med=h_med, s_med=s_med, v_med=v_med,
                        h_range=[h_min, h_max],
                        s_range=[s_min, s_max],
                        v_range=[v_min, v_max])

    print(f"  {prim_path} [{class_id}]  u={u} v={v}  "
          f"H={h_med}({h_min}-{h_max})  S={s_med}({s_min}-{s_max})  V={v_med}({v_min}-{v_max})")

cv2.imwrite(os.path.join(OUT, "calib_overlay.png"), vis)
print(f"\n[SAVED] calib_overlay.png")

# ── Suggest HSV ranges ────────────────────────────────────────────────────────
print("\n=== Suggested HSV ranges ===")
a_vals = [v for k, v in results.items() if "part_A" in k]
b_vals = [v for k, v in results.items() if "part_B" in k]

def suggest(vals, name):
    if not vals: return
    h_lo = max(0,   min(v["h_range"][0] for v in vals) - 8)
    h_hi = min(179, max(v["h_range"][1] for v in vals) + 8)
    s_lo = max(0,   min(v["s_range"][0] for v in vals) - 20)
    s_hi = 255
    v_lo = max(0,   min(v["v_range"][0] for v in vals) - 20)
    v_hi = 255
    print(f"  {name}: lower=[{h_lo},{s_lo},{v_lo}]  upper=[{h_hi},{s_hi},{v_hi}]")
    if h_lo <= 10 or h_hi >= 165:
        h2_lo = max(0,   min(v["h_range"][0] for v in vals))
        h2_hi = min(179, max(v["h_range"][1] for v in vals))
        if h_lo < 10:
            print(f"  {name} red-wrap: add lower2=[{170},{s_lo},{v_lo}]  upper2=[179,255,255]")

suggest(a_vals, "part_A")
suggest(b_vals, "part_B")

logger.close()
kit.close()
