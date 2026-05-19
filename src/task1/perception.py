"""
perception.py — Task 1 Perception Module
Author: Thanh Tai (N1)
Chạy: /isaac-sim/python.sh src/task1/perception.py
"""
from isaacsim import SimulationApp
CONFIG = {"width": 1280, "height": 720, "headless": False}
kit = SimulationApp(launch_config=CONFIG)

import os, sys, json, csv, numpy as np, cv2 as cv
sys.path.insert(0, "/home/ubuntu/tai/src")
sys.path.insert(0, "/home/ubuntu/tai/src/baseline_source")

from isaacsim.core.api import World
import omni, omni.replicator.core as rep
from baseline_source.config_loader import load_config, apply_scatter_config
from baseline_source.SceneBuilder import SceneBuilder
from baseline_source.RobotArticulation import RobotArticulation
from baseline_source.DataLogger import DataLogger
from baseline_source.coordinate_utils import CoordinateTransform

OUT = "/home/ubuntu/tai/lab_outputs/perception"
os.makedirs(OUT, exist_ok=True)

# ── Config ────────────────────────────────────────────────────
cfg = load_config("/home/ubuntu/tai/configs/Part_Sorting.yaml")
cfg["root_path"] = "/home/ubuntu/tai/assets/resources/"
grasp_cfg = cfg.get("grasp", {})

# Intrinsics hardcoded head_left 128x128
FX = FY = (5.0 / 2.0955) * 128
CX, CY = 64.0, 64.0
CONF_THRESH = 0.60
MIN_AREA = 15

# HSV ranges
PART_A_RED_L1 = np.array([  0, 120, 50])
PART_A_RED_U1 = np.array([  8, 255, 255])
PART_A_RED_L2 = np.array([155, 120, 50])
PART_A_RED_U2 = np.array([179, 255, 255])
PART_A_ORI_L  = np.array([  5, 150, 30])
PART_A_ORI_U  = np.array([ 22, 255, 200])
PART_B_BLUE_L = np.array([ 90,  80, 30])
PART_B_BLUE_U = np.array([130, 255, 255])

# ── Scene ─────────────────────────────────────────────────────
omni.usd.get_context().open_stage(
    os.path.join(cfg["root_path"], cfg["scene_usd"])
)
world = World(stage_units_in_meters=1.0, physics_dt=1/60, rendering_dt=1/20)
world.initialize_physics()

logger = DataLogger(enabled=False, csv_path=f"{OUT}/poses.csv",
                    camera_enabled=False, camera_hdf5_path=f"{OUT}/cam.hdf5")
scene = SceneBuilder(cfg, data_logger=logger)
apply_scatter_config(cfg)
scene.build_all()
rep.orchestrator.step()

world.play()
settle_steps = int(grasp_cfg.get("settle_time", 2.0) / world.get_physics_dt())
for _ in range(settle_steps):
    world.step(render=False)
print("[Init] Physics settled")

world.pause()
scene.build_robot()
robot = RobotArticulation(prim_path="/Root/Ref_Xform/Ref", name="walkerS2")
robot.initialize()
world.play()
for _ in range(30):
    world.step(render=True)
print("[Init] Robot ready")

# ── Transform ─────────────────────────────────────────────────
urdf_path = os.path.join(cfg["root_path"], "s2.urdf")
robot.initialize_ik(urdf_path)
js = robot.get_joint_states()
if js:
    robot.ik_solver.sync_joint_positions(js["names"], js["positions"][0])
coord = CoordinateTransform.from_torso_link(ik_solver=robot.ik_solver)

from pxr import UsdGeom
import omni.usd as ousd
stage = ousd.get_context().get_stage()
xc = UsdGeom.XformCache()
cam_prim = stage.GetPrimAtPath(
    "/Root/Ref_Xform/Ref/head_pitch_link/head_stereo_left/head_stereo_left_Camera_01"
)
if cam_prim.IsValid():
    tf = xc.GetLocalToWorldTransform(cam_prim)
    t_cw = np.array(tf.ExtractTranslation())
    R_gf = tf.ExtractRotationMatrix()
    R_cw = np.array([[R_gf[i][j] for j in range(3)] for i in range(3)]).T
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = R_cw
    T_world_cam[:3, 3]  = t_cw
    R_bw = coord.robot_world_R_inv
    T_base_world = np.eye(4)
    T_base_world[:3, :3] = R_bw
    T_base_world[:3, 3]  = -R_bw @ coord.robot_world_pos
    T_BASE_CAM = T_base_world @ T_world_cam
    print(f"[Transform] OK. Camera world={t_cw.round(3)}")
else:
    T_BASE_CAM = np.eye(4)
    print("[Transform] WARNING: identity")

# ── Helper functions ───────────────────────────────────────────
def robust_depth(depth, u, v, mask=None, radius=4):
    if mask is not None:
        vals = depth[(mask > 0) & np.isfinite(depth) & (depth > 0) & (depth < 3.0)]
        if len(vals) > 0:
            return float(np.median(vals))
    h, w = depth.shape
    u, v = int(round(u)), int(round(v))
    x1,x2 = max(0,u-radius), min(w,u+radius+1)
    y1,y2 = max(0,v-radius), min(h,v+radius+1)
    patch = depth[y1:y2, x1:x2]
    vals = patch[np.isfinite(patch) & (patch > 0) & (patch < 3.0)]
    return float(np.median(vals)) if len(vals) > 0 else None

def pixel_to_base(u, v, z):
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    p_h = np.array([x, y, z, 1.0])
    return (T_BASE_CAM @ p_h)[:3], np.array([x, y, z])

def detect_hsv(bgr, lower, upper, lower2=None, upper2=None):
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    if lower2 is not None:
        mask = cv.bitwise_or(mask, cv.inRange(hsv, lower2, upper2))
    k = np.ones((3,3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k)
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in cnts:
        area = cv.contourArea(c)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv.boundingRect(c)
        m = cv.moments(c)
        if m["m00"] == 0:
            continue
        obj_mask = np.zeros(mask.shape, np.uint8)
        cv.drawContours(obj_mask, [c], -1, 255, -1)
        dets.append({
            "bbox": [int(x), int(y), int(x+w), int(y+h)],
            "centroid": [float(m["m10"]/m["m00"]), float(m["m01"]/m["m00"])],
            "area": float(area),
            "contour": c,
            "mask": obj_mask,
        })
    return dets, mask

def make_state(det, class_id, depth, idx):
    u, v = det["centroid"]
    z = robust_depth(depth, u, v, mask=det["mask"])
    if z is None:
        return {"object_id": f"obj_{idx:03d}", "class_id": class_id,
                "confidence": 0.1, "bbox_xyxy": det["bbox"],
                "centroid_px": det["centroid"], "centroid_camera_m": None,
                "pose_base": None, "grasp_hint": None,
                "failure_reason": "INVALID_DEPTH"}
    p_base, p_cam = pixel_to_base(u, v, z)
    x_ok = 0.2 <= p_base[0] <= 1.3
    y_ok = -0.2 <= p_base[1] <= 0.7
    z_ok = 0.7 <= p_base[2] <= 1.4
    fail = None if (x_ok and y_ok and z_ok) else "POSE_OUT_OF_RANGE"
    conf = 0.85 if fail is None else 0.35
    if det["area"] < MIN_AREA * 2:
        conf -= 0.1
    yaw = float(np.deg2rad(cv.minAreaRect(det["contour"])[-1]))
    x1,y1,x2,y2 = det["bbox"]
    gw = float(np.clip((x2-x1)/FX*z*0.7, 0.02, 0.08))
    return {"object_id": f"obj_{idx:03d}", "class_id": class_id,
            "confidence": round(conf,3), "bbox_xyxy": det["bbox"],
            "centroid_px": [round(u,2), round(v,2)],
            "centroid_camera_m": [round(x,4) for x in p_cam.tolist()],
            "pose_base": {"position_m": [round(x,4) for x in p_base.tolist()],
                          "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
            "grasp_hint": {"approach_axis": "z_down",
                           "yaw_rad": round(yaw,4),
                           "grasp_width_m": round(gw,4)},
            "failure_reason": fail}

# ── Capture — dùng baseline method ────────────────────────────
print("\n[Perception] Capturing...")
rgbd  = robot.get_camera_rgbd("head_left")
rgb   = rgbd["rgb"]
depth = rgbd["depth"]
if depth is not None:
    depth = np.array(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:,:,0]
bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
print(f"[Capture] RGB={rgb.shape} depth={depth.shape if depth is not None else None}")

# Save samples
cv.imwrite(f"{OUT}/sample_rgb.png", bgr)
if depth is not None:
    np.save(f"{OUT}/sample_depth.npy", depth)
    valid = np.isfinite(depth) & (depth > 0)
    print(f"[Depth] valid={valid.mean():.3f} min={depth[valid].min():.3f} "
          f"median={np.median(depth[valid]):.3f} max={depth[valid].max():.3f}")
    d_vis = depth.copy(); d_vis[~valid] = 0
    d_norm = cv.normalize(d_vis, None, 0, 255, cv.NORM_MINMAX)
    cv.imwrite(f"{OUT}/sample_depth_preview.png",
               cv.applyColorMap(d_norm.astype(np.uint8), cv.COLORMAP_JET))

# ── Detect ────────────────────────────────────────────────────
dets_A_red, mask_ar = detect_hsv(bgr, PART_A_RED_L1, PART_A_RED_U1,
                                  PART_A_RED_L2, PART_A_RED_U2)
dets_A_ori, mask_ao = detect_hsv(bgr, PART_A_ORI_L, PART_A_ORI_U)
dets_B,     mask_b  = detect_hsv(bgr, PART_B_BLUE_L, PART_B_BLUE_U)
dets_A = dets_A_red + dets_A_ori
print(f"[Detect] A={len(dets_A)} (red={len(dets_A_red)} ori={len(dets_A_ori)}) "
      f"B={len(dets_B)}")

# ── Build states ───────────────────────────────────────────────
objects, idx = [], 1
for d in dets_A:
    objects.append(make_state(d, "part_A", depth, idx)); idx += 1
for d in dets_B:
    objects.append(make_state(d, "part_B", depth, idx)); idx += 1

valid_objs = [o for o in objects
              if o["confidence"] >= CONF_THRESH and o["failure_reason"] is None]
print(f"[Result] Total={len(objects)} Valid={len(valid_objs)}")
for o in objects:
    pos = o["pose_base"]["position_m"] if o["pose_base"] else None
    print(f"  {o['object_id']} {o['class_id']} conf={o['confidence']} "
          f"fail={o['failure_reason']} pos={pos}")

# ── Overlay ────────────────────────────────────────────────────
overlay = bgr.copy()
for o in objects:
    if not o["bbox_xyxy"]: continue
    x1,y1,x2,y2 = o["bbox_xyxy"]
    u,v = int(o["centroid_px"][0]), int(o["centroid_px"][1])
    col = (0,255,0) if o["class_id"]=="part_A" else (255,128,0)
    cv.rectangle(overlay,(x1,y1),(x2,y2),col,1)
    cv.circle(overlay,(u,v),3,(0,0,255),-1)
    cv.putText(overlay,f"{o['class_id'][-1]}{o['confidence']:.2f}",
               (x1,max(0,y1-4)),cv.FONT_HERSHEY_SIMPLEX,0.3,(255,255,255),1)
cv.imwrite(f"{OUT}/overlay_detection.png", overlay)
mask_all = cv.bitwise_or(cv.bitwise_or(mask_ar,mask_ao),mask_b)
cv.imwrite(f"{OUT}/overlay_mask.png", mask_all)

# ── Save JSON ──────────────────────────────────────────────────
result = {"frame_id":1,"timestamp":0.05,"camera_name":"head_left",
          "objects":objects,
          "summary":{"num_objects":len(objects),
                     "num_valid_objects":len(valid_objs),
                     "num_invalid_depth":sum(1 for o in objects
                                              if o.get("failure_reason")=="INVALID_DEPTH")}}
with open(f"{OUT}/perception_interface.json","w") as f:
    json.dump(result,f,indent=2)

# pose_estimator_report.csv
with open(f"{OUT}/pose_estimator_report.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["object_id","class_id","u","v","depth_m",
                "x_cam","y_cam","z_cam","x_base","y_base","z_base","status"])
    for o in objects:
        u,v=o["centroid_px"]
        cam=o.get("centroid_camera_m") or [None,None,None]
        base=o["pose_base"]["position_m"] if o["pose_base"] else [None,None,None]
        w.writerow([o["object_id"],o["class_id"],round(u,1),round(v,1),
                    cam[2],cam[0],cam[1],cam[2],base[0],base[1],base[2],
                    o.get("failure_reason") or "ok"])

print(f"\n[Done] lab_outputs/perception/")
print(f"  overlay_detection.png")
print(f"  perception_interface.json")

logger.close()
kit.close()
