"""
calibrate_hsv.py — Chạy N scatter iterations, dùng semantic mask để sample HSV per class.

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

ROOT   = "/home/ubuntu/tai"
OUT    = os.path.join(ROOT, "lab_outputs/perception")
N_ITER = 8
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "src"))
from baseline_source.config_loader import load_config, apply_scatter_config
from baseline_source.SceneBuilder import SceneBuilder
from baseline_source.RobotArticulation import RobotArticulation
from baseline_source.DataLogger import DataLogger

# ── Label normaliser (same as perception.py) ──────────────────────────────────
_LABEL_MAP = {
    "part_a": "part_A", "part_b": "part_B",
    "parta":  "part_A", "partb":  "part_B",
    "part a": "part_A", "part b": "part_B",
}

def _norm_label(val) -> str | None:
    if isinstance(val, dict):
        raw = val.get("class", val.get("name", "")).lower().strip()
    else:
        raw = str(val).lower().strip()
    return _LABEL_MAP.get(raw)

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
_rp      = rep.create.render_product(CAMERA_PRIM, (_CAM_W, _CAM_H))
_rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
_sem_ann = rep.AnnotatorRegistry.get_annotator("semantic_segmentation",
                                               init_params={"colorize": False})
_rgb_ann.attach(_rp)
_sem_ann.attach(_rp)

world.play()
for _ in range(30):
    world.step(render=True)
rep.orchestrator.step()

# ── HSV sample accumulator ────────────────────────────────────────────────────
samples = {"part_A": {"h": [], "s": [], "v": []},
           "part_B": {"h": [], "s": [], "v": []}}

last_bgr = None
last_masks = {}   # class_id → binary mask for debug overlay

# ── N iterations ──────────────────────────────────────────────────────────────
for it in range(N_ITER):
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

    sem_data = _sem_ann.get_data()
    if sem_data is None or not isinstance(sem_data, dict):
        print(f"[iter {it+1}] no sem_data, skip")
        continue

    mask_arr   = sem_data.get("data")
    id_to_labs = sem_data.get("info", {}).get("idToLabels", {})

    if mask_arr is None or mask_arr.size == 0:
        print(f"[iter {it+1}] empty mask, skip")
        continue

    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    mask_arr = mask_arr.astype(np.int32)

    sem_class = {int(k): _norm_label(v) for k, v in id_to_labs.items()
                 if _norm_label(v) is not None}

    if not sem_class:
        print(f"[iter {it+1}] no recognised labels in idToLabels: {id_to_labs}")
        continue

    iter_counts = {}
    iter_masks  = {}
    for sem_id, class_id in sem_class.items():
        px_mask = (mask_arr == sem_id)
        n_px = int(px_mask.sum())
        if n_px == 0:
            continue
        h_vals = hsv_img[:, :, 0][px_mask].tolist()
        s_vals = hsv_img[:, :, 1][px_mask].tolist()
        v_vals = hsv_img[:, :, 2][px_mask].tolist()
        samples[class_id]["h"].extend(h_vals)
        samples[class_id]["s"].extend(s_vals)
        samples[class_id]["v"].extend(v_vals)
        iter_counts[class_id] = iter_counts.get(class_id, 0) + n_px
        if class_id not in iter_masks:
            iter_masks[class_id] = px_mask.copy()
        else:
            iter_masks[class_id] |= px_mask

    print(f"[iter {it+1}/{N_ITER}] pixels per class: "
          + ", ".join(f"{k}={v}" for k, v in iter_counts.items()))

    if it == N_ITER - 1:
        last_bgr   = bgr.copy()
        last_masks = iter_masks

# ── Debug overlay ─────────────────────────────────────────────────────────────
if last_bgr is not None:
    dbg = last_bgr.copy()
    colors = {"part_A": (0, 100, 220), "part_B": (220, 140, 0)}
    for cls, mask in last_masks.items():
        c = colors.get(cls, (255, 255, 255))
        dbg[mask] = (dbg[mask].astype(int) // 2 + np.array(c) // 2).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, "calib_overlay.png"), dbg)
    cv2.imwrite(os.path.join(OUT, "calib_rgb.png"),    last_bgr)
    print(f"[debug] calib_overlay.png saved")

# ── Aggregated stats ──────────────────────────────────────────────────────────
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
        print(f"  {cls} red-wrap: lower2=[170,{s_lo},{v_lo}]  upper2=[179,255,255]")

logger.close()
kit.close()
