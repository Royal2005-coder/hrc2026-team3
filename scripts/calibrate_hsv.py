"""
calibrate_hsv.py — Chạy N scatter iterations, gộp HSV samples, suggest robust ranges.

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

ROOT     = "/home/ubuntu/tai"
OUT      = os.path.join(ROOT, "lab_outputs/perception")
N_ITER   = 8      # số lần re-scatter để lấy mẫu
RADIUS   = 6      # patch radius (pixels) để sample HSV
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "src"))
from baseline_source.config_loader import load_config, apply_scatter_config
from baseline_source.SceneBuilder import SceneBuilder
from baseline_source.RobotArticulation import RobotArticulation
from baseline_source.DataLogger import DataLogger
from task1.camera_utils import CameraIntrinsics

# ── Build scene (1 lần) ───────────────────────────────────────────────────────
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
_rp      = rep.create.render_product(CAMERA_PRIM, (_CAM_W, _CAM_H))
_rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
_rgb_ann.attach(_rp)

world.play()
for _ in range(30):
    world.step(render=True)

# ── Intrinsics + transforms (cố định trong session) ───────────────────────────
from pxr import UsdGeom
stage = omni.usd.get_context().get_stage()

_cam_prim = stage.GetPrimAtPath(CAMERA_PRIM)
_fl = _cam_prim.GetAttribute("focalLength").Get()
_ha = _cam_prim.GetAttribute("horizontalAperture").Get()
_va = _cam_prim.GetAttribute("verticalAperture").Get()
_hao = _cam_prim.GetAttribute("horizontalApertureOffset").Get() or 0.0
_vao = _cam_prim.GetAttribute("verticalApertureOffset").Get() or 0.0
_fx = (_CAM_W * _fl) / _ha
_fy = (_CAM_H * _fl) / _va
intr = CameraIntrinsics(fx=_fx, fy=_fy,
                        cx=_CAM_W/2.0 + (_CAM_W * _hao) / _ha,
                        cy=_CAM_H/2.0 + (_CAM_H * _vao) / _va,
                        width=_CAM_W, height=_CAM_H)
print(f"[Intrinsics] fx={_fx:.2f} fy={_fy:.2f} cx={intr.cx:.2f} cy={intr.cy:.2f}")

def world_tf(path):
    p = stage.GetPrimAtPath(path)
    if not p.IsValid():
        return None
    return np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)).T

T_wc  = world_tf(CAMERA_PRIM)
T_wb  = world_tf("/Root/Ref_Xform/Ref/base_link")
T_bc  = np.linalg.inv(T_wb) @ T_wc
T_cb  = np.linalg.inv(T_bc)
T_bw  = np.linalg.inv(T_wb)

# ── Part prim paths ───────────────────────────────────────────────────────────
num_parts = cfg["part"].get("num_parts", 2)
raw_paths = scene.parts_prim_paths
part_prims = [(p, "part_A" if i < num_parts else "part_B")
              for i, p in enumerate(raw_paths)]
print(f"[Parts] {len(part_prims)} parts: {[p for p,_ in part_prims]}")

# Accumulate H, S, V lists per class
samples = {"part_A": {"h": [], "s": [], "v": []},
           "part_B": {"h": [], "s": [], "v": []}}

# ── N iterations ──────────────────────────────────────────────────────────────
last_bgr = None
last_hits = []   # (u, v, class_id) of sampled centroids in final iteration

for it in range(N_ITER):
    # Re-scatter parts
    apply_scatter_config(cfg)
    rep.orchestrator.step()
    settle = int(1.0 / world.get_physics_dt())
    for _ in range(settle):
        world.step(render=False)
    for _ in range(15):
        world.step(render=True)
    rep.orchestrator.step()

    rgb = np.array(_rgb_ann.get_data(), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
    hsv_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    hit = 0
    iter_hits = []
    for prim_path, class_id in part_prims:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue

        T_wobj = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
        pos_w  = T_wobj[:3, 3]
        p_base = (T_bw @ np.r_[pos_w, 1.0])[:3]
        p_cam  = (T_cb @ np.r_[p_base, 1.0])[:3]

        # Isaac Sim camera: -Z forward. Depth = -p_cam[2].
        z = -p_cam[2]
        if z <= 0.05:
            continue

        u = int(intr.fx * p_cam[0] / z + intr.cx)
        v = int(intr.fy * (-p_cam[1]) / z + intr.cy)

        if not (RADIUS <= u < _CAM_W - RADIUS and RADIUS <= v < _CAM_H - RADIUS):
            print(f"  [iter {it+1}] {class_id} projected OOB: u={u} v={v} z={z:.3f}m")
            continue

        patch   = hsv_img[v-RADIUS:v+RADIUS+1, u-RADIUS:u+RADIUS+1]
        h_med   = int(np.median(patch[:, :, 0]))
        s_med   = int(np.median(patch[:, :, 1]))
        v_med   = int(np.median(patch[:, :, 2]))

        samples[class_id]["h"].extend(patch[:, :, 0].flatten().tolist())
        samples[class_id]["s"].extend(patch[:, :, 1].flatten().tolist())
        samples[class_id]["v"].extend(patch[:, :, 2].flatten().tolist())
        iter_hits.append((u, v, class_id, h_med, s_med, v_med))
        hit += 1

    print(f"[iter {it+1}/{N_ITER}] sampled {hit}/{len(part_prims)} parts")
    for u, v, cid, hm, sm, vm in iter_hits:
        print(f"  {cid}  u={u} v={v}  HSV_median=({hm},{sm},{vm})")

    # Keep last iteration for debug overlay
    if it == N_ITER - 1:
        last_bgr  = bgr.copy()
        last_hits = iter_hits

# ── Debug overlay — last iteration ───────────────────────────────────────────
if last_bgr is not None:
    dbg = last_bgr.copy()
    for u, v, cid, hm, sm, vm in last_hits:
        color = (0, 100, 220) if cid == "part_A" else (220, 140, 0)
        cv2.rectangle(dbg, (u - RADIUS, v - RADIUS),
                      (u + RADIUS, v + RADIUS), color, 2)
        cv2.circle(dbg, (u, v), 3, color, -1)
        cv2.putText(dbg, f"{cid[-1]}({hm},{sm},{vm})",
                    (u + RADIUS + 2, v), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
    cv2.imwrite(os.path.join(OUT, "calib_overlay.png"), dbg)
    cv2.imwrite(os.path.join(OUT, "calib_rgb.png"), last_bgr)
    print(f"[debug] calib_rgb.png + calib_overlay.png saved")

# ── Compute aggregated stats ──────────────────────────────────────────────────
print("\n=== Aggregated HSV stats ===")
for cls in ("part_A", "part_B"):
    h = np.array(samples[cls]["h"])
    s = np.array(samples[cls]["s"])
    v = np.array(samples[cls]["v"])
    if len(h) == 0:
        print(f"  {cls}: NO SAMPLES")
        continue
    print(f"  {cls} (n={len(h)} pixels):")
    print(f"    H: median={int(np.median(h))}  "
          f"p5={int(np.percentile(h,5))}  p95={int(np.percentile(h,95))}  "
          f"p1={int(np.percentile(h,1))}  p99={int(np.percentile(h,99))}")
    print(f"    S: median={int(np.median(s))}  "
          f"p5={int(np.percentile(s,5))}  p95={int(np.percentile(s,95))}")
    print(f"    V: median={int(np.median(v))}  "
          f"p5={int(np.percentile(v,5))}  p95={int(np.percentile(v,95))}")

print("\n=== Suggested HSV ranges ===")
for cls in ("part_A", "part_B"):
    h = np.array(samples[cls]["h"])
    s = np.array(samples[cls]["s"])
    v = np.array(samples[cls]["v"])
    if len(h) == 0:
        continue
    h_lo = max(0,   int(np.percentile(h, 5))  - 8)
    h_hi = min(179, int(np.percentile(h, 95)) + 8)
    s_lo = max(0,   int(np.percentile(s, 5))  - 20)
    v_lo = max(0,   int(np.percentile(v, 5))  - 20)
    print(f"  {cls}: lower=[{h_lo},{s_lo},{v_lo}]  upper=[{h_hi},255,255]")
    if h_lo <= 10 or h_hi >= 165:
        print(f"  {cls} red-wrap: lower2=[{max(0,170-(8-(h_lo))},{s_lo},{v_lo}]"
              f"  upper2=[179,255,255]")

logger.close()
kit.close()
