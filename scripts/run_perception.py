"""
run_perception.py — Standalone perception runner for HRC2026 Task 1 (N1).

Chạy:
    /isaac-sim/python.sh scripts/run_perception.py

Output (lab_outputs/perception/):
    sample_rgb.png, sample_depth.npy, sample_depth_preview.png
    depth_sanity_report.md, camera_config_sheet.csv
    transform_sanity_report.md
    overlay_detection.png, overlay_mask.png, overlay_centroid.png
    confusion_matrix_task1.csv, pose_estimator_report.csv
    yaw_report.csv, perception_interface.json, failure_cases_perception.jsonl
"""

from isaacsim import SimulationApp
kit = SimulationApp(launch_config={"width": 1280, "height": 720, "headless": True})

from isaacsim.core.api import World
import omni
import omni.replicator.core as rep
import numpy as np
import cv2 as cv
import os, sys

ROOT      = "/home/ubuntu/tai"
OUT_DIR   = os.path.join(ROOT, "lab_outputs/perception")
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "src"))

from baseline_source.config_loader import load_config, apply_scatter_config
from baseline_source.SceneBuilder import SceneBuilder
from baseline_source.RobotArticulation import RobotArticulation
from baseline_source.DataLogger import DataLogger
from task1.camera_utils import (CameraIntrinsics, depth_sanity,
                                write_depth_sanity_report, save_camera_config_csv)
from task1.transform_utils import run_transform_sanity
from task1.perception import (run_perception, save_perception_json,
                              save_pose_report_csv, save_yaw_report_csv,
                              save_failure_cases_jsonl)
from task1.perception_debug import (save_overlays, save_confusion_matrix_csv,
                                    validate_perception_output)

# ── Config ──────────────────────────────────────────────────────────────────
cfg = load_config(os.path.join(ROOT, "configs/Part_Sorting.yaml"))
cfg["root_path"] = os.path.join(ROOT, "assets/resources/")

# ── Scene ───────────────────────────────────────────────────────────────────
omni.usd.get_context().open_stage(os.path.join(cfg["root_path"], cfg["scene_usd"]))
world = World(stage_units_in_meters=1.0, physics_dt=1/60, rendering_dt=1/20)
world.initialize_physics()

logger = DataLogger(enabled=False, csv_path="/tmp/perc.csv",
                    camera_enabled=False, camera_hdf5_path="/tmp/perc.hdf5")
scene = SceneBuilder(cfg, data_logger=logger)
apply_scatter_config(cfg)
scene.build_all()
rep.orchestrator.step()

print("[1/5] Settling physics...")
world.play()
settle = int(cfg.get("grasp", {}).get("settle_time", 2.0) / world.get_physics_dt())
for _ in range(settle):
    world.step(render=False)
print(f"      Done ({settle} steps)")

# ── Robot ───────────────────────────────────────────────────────────────────
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
print("[2/5] Robot ready")

# ── Camera ──────────────────────────────────────────────────────────────────
CAMERA_PRIM = "/Root/Ref_Xform/Ref/head_pitch_link/head_stereo_left/head_stereo_left_Camera_01"
W, H = 640, 480
_rp        = rep.create.render_product(CAMERA_PRIM, (W, H))
_rgb_ann   = rep.AnnotatorRegistry.get_annotator("rgb")
_depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
_sem_ann   = rep.AnnotatorRegistry.get_annotator("semantic_segmentation",
                                                  init_params={"colorize": False})
_rgb_ann.attach(_rp)
_depth_ann.attach(_rp)
_sem_ann.attach(_rp)

for _ in range(10):
    world.step(render=True)
rep.orchestrator.step()

# ── Intrinsics ───────────────────────────────────────────────────────────────
stage = omni.usd.get_context().get_stage()
from pxr import UsdGeom
_cam = stage.GetPrimAtPath(CAMERA_PRIM)
_fl  = _cam.GetAttribute("focalLength").Get()
_ha  = _cam.GetAttribute("horizontalAperture").Get()
_va  = _cam.GetAttribute("verticalAperture").Get()
intr = CameraIntrinsics(
    fx=(W * _fl) / _ha, fy=(H * _fl) / _va,
    cx=W / 2.0, cy=H / 2.0,
    width=W, height=H, depth_unit="meter",
)
print(f"[3/5] Intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f}")

# ── Transforms ───────────────────────────────────────────────────────────────
def _world_tf(path):
    p = stage.GetPrimAtPath(path)
    return np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)).T if p.IsValid() else None

T_wc = _world_tf(CAMERA_PRIM)
T_wb = _world_tf("/Root/Ref_Xform/Ref/base_link")
# Isaac Sim camera: Y-up, -Z forward. OpenCV (pixel_to_camera_point): Y-down, +Z forward.
# Absorb the frame difference into T_base_camera so the rest of the pipeline is unchanged.
_R_cam = np.diag([1., -1., -1., 1.])
T_base_camera = np.linalg.inv(T_wb) @ T_wc @ _R_cam
print("[4/5] Transforms ready")

# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Grab raw samples
# ═══════════════════════════════════════════════════════════════════════════
rgb_raw   = np.array(_rgb_ann.get_data(),   dtype=np.uint8)
depth_raw = np.array(_depth_ann.get_data(), dtype=np.float32)
sem_raw   = _sem_ann.get_data()

if depth_raw.ndim == 3:
    depth_raw = depth_raw[:, :, 0]

# RGB: Isaac Sim returns RGBA — drop alpha, save as BGR for OpenCV
bgr = cv.cvtColor(rgb_raw[:, :, :3], cv.COLOR_RGB2BGR)
cv.imwrite(os.path.join(OUT_DIR, "sample_rgb.png"), bgr)
np.save(os.path.join(OUT_DIR, "sample_depth.npy"), depth_raw)

# depth preview — normalise to 8-bit
valid_d = depth_raw[np.isfinite(depth_raw) & (depth_raw > 0)]
if valid_d.size > 0:
    d_vis = np.clip((depth_raw - valid_d.min()) / (valid_d.max() - valid_d.min() + 1e-8), 0, 1)
    cv.imwrite(os.path.join(OUT_DIR, "sample_depth_preview.png"),
               (d_vis * 255).astype(np.uint8))
print("[sample] rgb, depth, depth_preview saved")

# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Depth sanity
# ═══════════════════════════════════════════════════════════════════════════
dsanity = depth_sanity(depth_raw)
write_depth_sanity_report(dsanity, os.path.join(OUT_DIR, "depth_sanity_report.md"))
print(f"[depth] median={dsanity.get('median')}  unit={dsanity.get('guessed_unit')}")

# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Camera config CSV
# ═══════════════════════════════════════════════════════════════════════════
save_camera_config_csv(
    intr,
    T_source="ComputeLocalToWorldTransform(USD)",
    camera_name="head_stereo_left",
    notes="Task 1 perception camera",
    path=os.path.join(OUT_DIR, "camera_config_sheet.csv"),
)
print("[camera] camera_config_sheet.csv saved")

# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Transform sanity
# ═══════════════════════════════════════════════════════════════════════════
run_transform_sanity(
    T_base_camera,
    report_path=os.path.join(OUT_DIR, "transform_sanity_report.md"),
)
print("[transform] transform_sanity_report.md saved")

# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — Run full perception pipeline
# ═══════════════════════════════════════════════════════════════════════════
has_sem = (sem_raw is not None
           and isinstance(sem_raw, dict)
           and sem_raw.get("info", {}).get("idToLabels"))
method = "annotation" if has_sem else "depth_fg"
print(f"[perception] detection_method={method}")

perc_state = run_perception(
    rgb_bgr=bgr,
    depth=depth_raw,
    intr=intr,
    T_base_camera=T_base_camera,
    hsv_ranges={},
    frame_id=0,
    camera_name="head_stereo_left",
    detection_method=method,
    sem_ann_data=sem_raw,
)

# ═══════════════════════════════════════════════════════════════════════════
# Step 6 — Validate + save all artifacts
# ═══════════════════════════════════════════════════════════════════════════
is_valid, errors = validate_perception_output(perc_state)
print(f"[validate] valid={is_valid}  errors={errors}")

save_perception_json(perc_state,
                     os.path.join(OUT_DIR, "perception_interface.json"))

save_pose_report_csv(perc_state["objects"],
                     os.path.join(OUT_DIR, "pose_estimator_report.csv"))

save_yaw_report_csv(perc_state["objects"],
                    os.path.join(OUT_DIR, "yaw_report.csv"))

save_failure_cases_jsonl(perc_state["objects"],
                         os.path.join(OUT_DIR, "failure_cases_perception.jsonl"))

save_confusion_matrix_csv(perc_state["objects"],
                          path=os.path.join(OUT_DIR, "confusion_matrix_task1.csv"))

overlay_paths = save_overlays(bgr, perc_state, output_dir=OUT_DIR, intr=intr)

print(f"\n[5/5] Done. Artifacts in {OUT_DIR}/")
print(f"  Objects detected: {perc_state['summary']['num_objects']}")
print(f"  Valid objects:    {perc_state['summary']['num_valid_objects']}")
for o in perc_state["objects"]:
    status = "OK" if o["failure_reason"] is None else o["failure_reason"]
    pose   = o["pose_base"]["position_m"] if o["pose_base"] else None
    print(f"  {o['object_id']}  {o['class_id']}  conf={o['confidence']:.2f}"
          f"  pose_base={np.round(pose, 3).tolist() if pose else None}  [{status}]")

# ═══════════════════════════════════════════════════════════════════════════
# Step 7 — Verify: compare camera pose vs USD stage ground truth
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Pose verification: camera vs stage ──────────────────────────────")

# GT từ USD stage
num_per_class = cfg["part"].get("num_parts", 2)
gt_parts = []
for i, prim_path in enumerate(scene.parts_prim_paths):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        continue
    T_p = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
    gt_parts.append({
        "class_id":  "part_A" if i < num_per_class else "part_B",
        "pos_world": T_p[:3, 3],
    })

# Perception: convert pose_base → world qua T_wb
perc_world = []
for o in perc_state["objects"]:
    if o["pose_base"] is None:
        continue
    pos_base  = np.array(o["pose_base"]["position_m"])
    pos_world = (T_wb @ np.append(pos_base, 1.0))[:3]
    perc_world.append({"class_id": o["class_id"], "pos_world": pos_world})

print(f"  GT parts   ({len(gt_parts)}):")
for p in gt_parts:
    print(f"    {p['class_id']}  world={np.round(p['pos_world'], 3).tolist()}")

print(f"  Cam parts  ({len(perc_world)}):")
for p in perc_world:
    print(f"    {p['class_id']}  world={np.round(p['pos_world'], 3).tolist()}")

# Match nearest GT→perception và tính error
if gt_parts and perc_world:
    print("\n  Nearest-match errors (GT → best camera match):")
    total_err = []
    for gt in gt_parts:
        best_err  = 999.0
        best_perc = None
        for pc in perc_world:
            if pc["class_id"] != gt["class_id"]:
                continue
            err = float(np.linalg.norm(gt["pos_world"] - pc["pos_world"]))
            if err < best_err:
                best_err  = err
                best_perc = pc
        if best_perc is not None:
            total_err.append(best_err)
            print(f"    {gt['class_id']}  GT={np.round(gt['pos_world'],3).tolist()}"
                  f"  cam={np.round(best_perc['pos_world'],3).tolist()}"
                  f"  err={best_err:.4f}m")
        else:
            print(f"    {gt['class_id']}  NO MATCH")
    if total_err:
        print(f"\n  Mean position error: {np.mean(total_err):.4f}m"
              f"  Max: {np.max(total_err):.4f}m")

logger.close()
kit.close()
