"""
perception.py — End-to-end perception pipeline for HRC2026 Task 1.

Pipeline:  RGB-D → detect → classify A/B → centroid/yaw → camera-to-base → ObjectState JSON
"""

import cv2 as cv
import numpy as np
import json
import csv
import time
from typing import Any

from camera_utils import (
    CameraIntrinsics,
    pixel_to_camera_point,
    robust_depth_from_patch,
    robust_depth_from_mask,
)
from transform_utils import (
    transform_point,
    rotation_matrix_to_quaternion,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Detection (colour-based MVP)
# ═══════════════════════════════════════════════════════════════════════════
def detect_by_color(rgb_bgr: np.ndarray,
                    lower_hsv: list | np.ndarray,
                    upper_hsv: list | np.ndarray,
                    min_area: int = 100) -> tuple[list[dict], np.ndarray]:
    """
    Detect objects via HSV colour thresholding + morphology + contour analysis.

    Returns
    -------
    detections : list of dicts with bbox_xyxy, centroid_px, area_px, contour
    mask       : binary mask after morphology
    """
    hsv = cv.cvtColor(rgb_bgr, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, np.array(lower_hsv, dtype=np.uint8),
                      np.array(upper_hsv, dtype=np.uint8))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv.boundingRect(cnt)
        m = cv.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        detections.append({
            "bbox_xyxy": [int(x), int(y), int(x + w), int(y + h)],
            "centroid_px": [float(cx), float(cy)],
            "area_px": float(area),
            "contour": cnt,
        })
    return detections, mask


# ═══════════════════════════════════════════════════════════════════════════
# 1b. Detection by depth foreground (MVP for Task 1)
# ═══════════════════════════════════════════════════════════════════════════
def detect_by_depth_foreground(depth: np.ndarray,
                               table_z_m: float = 1.04,
                               fg_threshold_m: float = 0.02,
                               min_area: int = 100) -> tuple[list[dict], np.ndarray]:
    """
    Detect objects that are ABOVE the table surface using depth.

    In Isaac Sim with a top-down or angled camera, objects on the table
    have depth values LESS than the table depth (they are closer to camera).

    For a head camera looking forward, we use the known table z and
    compare with the 3D z coordinate of each pixel. But simpler:
    create a foreground mask where depth is significantly different from
    the table plane depth.

    Parameters
    ----------
    depth          : depth map (H, W) in metres (distance_to_image_plane)
    table_z_m      : known z of table surface in world frame
    fg_threshold_m : minimum height above table to count as object
    min_area       : minimum contour area in pixels

    Returns
    -------
    detections : list of dicts
    fg_mask    : binary foreground mask
    """
    valid = np.isfinite(depth) & (depth > 0)

    # Compute median depth (≈ table/background depth for most of the image)
    if valid.sum() == 0:
        return [], np.zeros(depth.shape[:2], dtype=np.uint8)

    median_depth = np.median(depth[valid])

    # Foreground = pixels significantly CLOSER than background
    fg_mask = np.zeros(depth.shape[:2], dtype=np.uint8)
    fg_mask[valid & (depth < median_depth - fg_threshold_m)] = 255

    # Morphology cleanup
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)
    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv.boundingRect(cnt)
        m = cv.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        detections.append({
            "bbox_xyxy": [int(x), int(y), int(x + w), int(y + h)],
            "centroid_px": [float(cx), float(cy)],
            "area_px": float(area),
            "contour": cnt,
        })
    return detections, fg_mask


# ═══════════════════════════════════════════════════════════════════════════
# 1c. Shape-based classification
# ═══════════════════════════════════════════════════════════════════════════
def extract_shape_features(contour: np.ndarray) -> dict:
    """
    Extract shape features from a contour for A/B classification.

    Since Part A (Task1_PartA.usd) and Part B (Part_B.usd) have different
    3D shapes, their 2D projections will differ in aspect ratio, solidity,
    hu moments, etc.
    """
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    x, y, w, h = cv.boundingRect(contour)
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull) if len(hull) >= 3 else area

    # Aspect ratio of bounding rect
    aspect_ratio = float(w) / h if h > 0 else 0.0

    # Solidity = contour area / convex hull area
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Circularity = 4π × area / perimeter²
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0

    # Hu moments (log-transformed, shape-invariant)
    moments = cv.moments(contour)
    hu = cv.HuMoments(moments).flatten()
    # Log transform (sign-preserving)
    hu_log = np.array([
        -np.sign(h) * np.log10(abs(h) + 1e-20) for h in hu
    ])

    # minAreaRect aspect ratio (rotation-invariant)
    rect = cv.minAreaRect(contour)
    rect_w, rect_h = rect[1]
    min_rect_aspect = min(rect_w, rect_h) / max(rect_w, rect_h) if max(rect_w, rect_h) > 0 else 0

    return {
        "area_px": float(area),
        "perimeter_px": float(perimeter),
        "bbox_aspect_ratio": round(aspect_ratio, 3),
        "min_rect_aspect": round(min_rect_aspect, 3),
        "solidity": round(solidity, 3),
        "circularity": round(circularity, 3),
        "hu_moments": hu_log.tolist(),
    }


def classify_by_color_hint(rgb_bgr: np.ndarray,
                           contour: np.ndarray,
                           hsv_ranges: dict) -> tuple[str | None, float]:
    """
    Try to classify using colour. Returns (class_id, confidence).
    If colour is ambiguous (ori_color), returns (None, 0.0).

    hsv_ranges format from YAML:
      red:   {lower, upper, lower2, upper2, implies_class: "part_A"}
      blue:  {lower, upper, implies_class: "part_B"}
    """
    hsv = cv.cvtColor(rgb_bgr, cv.COLOR_BGR2HSV)
    mask_contour = np.zeros(rgb_bgr.shape[:2], dtype=np.uint8)
    cv.drawContours(mask_contour, [contour], -1, 255, -1)

    for color_name, cfg in hsv_ranges.items():
        if cfg.get("implies_class") is None:
            continue  # skip ori_color — ambiguous

        m1 = cv.inRange(hsv, np.array(cfg["lower"]), np.array(cfg["upper"]))
        if "lower2" in cfg:
            m2 = cv.inRange(hsv, np.array(cfg["lower2"]), np.array(cfg["upper2"]))
            m1 = m1 | m2

        overlap = cv.bitwise_and(m1, mask_contour)
        overlap_ratio = overlap.sum() / (mask_contour.sum() + 1e-8)

        if overlap_ratio > 0.3:  # >30% of object pixels match this color
            return cfg["implies_class"], min(0.95, 0.7 + overlap_ratio * 0.3)

    return None, 0.0  # ambiguous — need shape classifier


def classify_by_shape(features: dict,
                      part_A_aspect_range: tuple = (0.0, 999.0),
                      part_B_aspect_range: tuple = (0.0, 999.0)) -> tuple[str, float]:
    """
    Classify using shape features when colour is ambiguous.

    TODO: Tài cần capture sample, đo features của Part A và Part B,
    rồi điền thresholds vào đây.

    Fallback: dùng aspect ratio hoặc Hu moments.
    """
    ar = features["min_rect_aspect"]

    # Placeholder logic — THAY bằng thresholds thật sau khi đo
    # Ý tưởng: nếu Part A dài hẹp (aspect ratio thấp) còn Part B vuông hơn
    if part_A_aspect_range[0] <= ar <= part_A_aspect_range[1]:
        return "part_A", 0.65  # lower confidence because shape-only
    if part_B_aspect_range[0] <= ar <= part_B_aspect_range[1]:
        return "part_B", 0.65

    return "unknown", 0.40  # cannot determine
def classify_detections(detections_A: list[dict],
                        detections_B: list[dict],
                        base_confidence: float = 0.85) -> list[dict]:
    """Tag each detection with class_id and initial confidence."""
    all_dets = []
    for d in detections_A:
        d["class_id"] = "part_A"
        d["confidence"] = base_confidence
        all_dets.append(d)
    for d in detections_B:
        d["class_id"] = "part_B"
        d["confidence"] = base_confidence
        all_dets.append(d)
    return all_dets


# ═══════════════════════════════════════════════════════════════════════════
# 3. Yaw / grasp direction
# ═══════════════════════════════════════════════════════════════════════════
def estimate_yaw_minrect(contour: np.ndarray) -> float:
    """Yaw from minAreaRect (radians). Quick MVP approach."""
    rect = cv.minAreaRect(contour)
    angle_deg = rect[-1]
    return float(np.deg2rad(angle_deg))


def estimate_yaw_pca(contour: np.ndarray) -> float:
    """Yaw from PCA on contour points (more stable for elongated objects)."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    mean = pts.mean(axis=0)
    pts_c = pts - mean
    cov = np.cov(pts_c, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Principal axis = eigenvector with largest eigenvalue
    principal = eigvecs[:, -1]
    yaw = float(np.arctan2(principal[1], principal[0]))
    return yaw


# ═══════════════════════════════════════════════════════════════════════════
# 4. Confidence scoring
# ═══════════════════════════════════════════════════════════════════════════
def compute_confidence(area_px: float,
                       depth_valid: bool,
                       mask_quality: float = 1.0,
                       class_ambiguous: bool = False,
                       min_area: float = 200) -> float:
    """
    Rule-based confidence in [0, 1].
    Start at 1.0 and subtract penalties.
    """
    conf = 1.0
    if area_px < min_area:
        conf -= 0.30
    if not depth_valid:
        conf -= 0.40
    if mask_quality < 0.5:
        conf -= 0.20
    if class_ambiguous:
        conf -= 0.20
    return max(0.0, min(1.0, conf))


# ═══════════════════════════════════════════════════════════════════════════
# 5. Full object state builder
# ═══════════════════════════════════════════════════════════════════════════
FAILURE_REASONS = [
    "NO_OBJECT_DETECTED",
    "LOW_CONFIDENCE",
    "INVALID_DEPTH",
    "CLASS_AMBIGUOUS",
    "MASK_TOO_SMALL",
    "MASK_FRAGMENTED",
    "POSE_OUT_OF_RANGE",
    "TRANSFORM_NOT_AVAILABLE",
    "GRASP_HINT_UNSTABLE",
]


def make_object_state(det: dict,
                      depth: np.ndarray,
                      intr: CameraIntrinsics,
                      T_base_camera: np.ndarray | None,
                      object_id: str = "obj_000",
                      yaw_method: str = "minrect",
                      default_grasp_width: float = 0.045,
                      confidence_threshold: float = 0.50) -> dict:
    """
    Build a single ObjectState dict from a detection + depth + transforms.

    Returns a dict matching the perception_interface.json schema.
    """
    u, v = det["centroid_px"]
    class_id = det.get("class_id", "unknown")
    contour = det.get("contour")

    # --- depth ---
    z = robust_depth_from_patch(depth, u, v, radius=3)
    depth_valid = z is not None

    # --- confidence ---
    conf = compute_confidence(
        area_px=det.get("area_px", 0),
        depth_valid=depth_valid,
    )

    # --- failure reason ---
    failure = None
    if not depth_valid:
        failure = "INVALID_DEPTH"
    elif conf < confidence_threshold:
        failure = "LOW_CONFIDENCE"
    elif T_base_camera is None:
        failure = "TRANSFORM_NOT_AVAILABLE"

    # --- 3-D centroid in camera frame ---
    centroid_camera = None
    if depth_valid:
        z_m = z * intr.depth_scale()
        centroid_camera = pixel_to_camera_point(u, v, z_m, intr)

    # --- base frame ---
    pose_base = None
    if centroid_camera is not None and T_base_camera is not None:
        p_base = transform_point(T_base_camera, centroid_camera)
        pose_base = {
            "position_m": p_base.tolist(),
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],  # identity until rotation is needed
        }

    # --- yaw ---
    yaw = 0.0
    if contour is not None and len(contour) >= 5:
        if yaw_method == "pca":
            yaw = estimate_yaw_pca(contour)
        else:
            yaw = estimate_yaw_minrect(contour)

    # --- assemble ---
    obj_state = {
        "object_id": object_id,
        "class_id": class_id,
        "confidence": round(conf, 3),
        "bbox_xyxy": det["bbox_xyxy"],
        "centroid_px": det["centroid_px"],
        "centroid_camera_m": centroid_camera.tolist() if centroid_camera is not None else None,
        "pose_base": pose_base,
        "grasp_hint": {
            "approach_axis": "z_down",
            "yaw_rad": round(yaw, 4),
            "grasp_width_m": default_grasp_width,
        },
        "failure_reason": failure,
    }
    return obj_state


# ═══════════════════════════════════════════════════════════════════════════
# 6. Full-frame perception
# ═══════════════════════════════════════════════════════════════════════════
def run_perception(rgb_bgr: np.ndarray,
                   depth: np.ndarray,
                   intr: CameraIntrinsics,
                   T_base_camera: np.ndarray | None,
                   hsv_ranges: dict,
                   frame_id: int = 0,
                   camera_name: str = "head_stereo_left",
                   detection_method: str = "color") -> dict:
    """
    Run full perception pipeline on one RGB-D frame.

    Parameters
    ----------
    rgb_bgr          : BGR image (H, W, 3)
    depth            : depth map (H, W), raw unit
    intr             : CameraIntrinsics
    T_base_camera    : 4×4 or None
    hsv_ranges       : colour config from YAML
    frame_id         : sequential frame number
    camera_name      : identifier string
    detection_method : "color" (legacy) or "depth_fg" (recommended for Task 1)

    Returns
    -------
    dict matching perception_interface.json schema
    """
    timestamp = round(time.time(), 3)

    if detection_method == "depth_fg":
        # ── NEW: depth-foreground detection + color/shape classification ──
        all_dets, fg_mask = detect_by_depth_foreground(depth)

        for det in all_dets:
            contour = det["contour"]

            # Step 1: try color hint (red → A, blue → B)
            cls, cls_conf = classify_by_color_hint(rgb_bgr, contour, hsv_ranges)

            if cls is not None:
                det["class_id"] = cls
                det["confidence"] = cls_conf
            else:
                # Step 2: fallback to shape
                features = extract_shape_features(contour)
                det["shape_features"] = features
                cls, cls_conf = classify_by_shape(features)
                det["class_id"] = cls
                det["confidence"] = cls_conf
                if cls == "unknown":
                    det["failure_reason_hint"] = "CLASS_AMBIGUOUS"

    else:
        # ── Legacy: pure color-based detection ──
        dets_A, mask_A = detect_by_color(
            rgb_bgr,
            hsv_ranges.get("part_A", hsv_ranges.get("red", {})).get("lower", [0, 100, 100]),
            hsv_ranges.get("part_A", hsv_ranges.get("red", {})).get("upper", [10, 255, 255]),
        )
        dets_B, mask_B = detect_by_color(
            rgb_bgr,
            hsv_ranges.get("part_B", hsv_ranges.get("blue", {})).get("lower", [100, 100, 100]),
            hsv_ranges.get("part_B", hsv_ranges.get("blue", {})).get("upper", [130, 255, 255]),
        )
        all_dets = classify_detections(dets_A, dets_B)

    # --- build object states ---
    objects = []
    for idx, det in enumerate(all_dets):
        oid = f"obj_{idx:03d}"
        obj = make_object_state(det, depth, intr, T_base_camera, object_id=oid)
        objects.append(obj)

    num_valid = sum(1 for o in objects if o["failure_reason"] is None)

    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "camera_name": camera_name,
        "objects": objects,
        "summary": {
            "num_objects": len(objects),
            "num_valid_objects": num_valid,
            "num_invalid_depth": sum(
                1 for o in objects if o["failure_reason"] == "INVALID_DEPTH"
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. Export helpers
# ═══════════════════════════════════════════════════════════════════════════
def save_perception_json(state: dict, path: str = "perception_interface.json"):
    """Write perception output to JSON (strips non-serializable contours)."""
    clean = json.loads(json.dumps(state, default=str))
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    return path


def save_pose_report_csv(objects: list[dict],
                         path: str = "pose_estimator_report.csv"):
    """Write per-object pose report CSV."""
    header = ["object_id", "class_id", "u", "v",
              "depth_m", "x_cam", "y_cam", "z_cam",
              "x_base", "y_base", "z_base", "status"]
    rows = []
    for o in objects:
        u, v = o["centroid_px"]
        cam = o["centroid_camera_m"]
        base = o["pose_base"]["position_m"] if o["pose_base"] else [None]*3
        depth_m = cam[2] if cam else None
        status = "ok" if o["failure_reason"] is None else o["failure_reason"]
        rows.append([
            o["object_id"], o["class_id"],
            round(u, 1), round(v, 1),
            round(depth_m, 4) if depth_m else None,
            round(cam[0], 4) if cam else None,
            round(cam[1], 4) if cam else None,
            round(cam[2], 4) if cam else None,
            round(base[0], 4) if base[0] is not None else None,
            round(base[1], 4) if base[1] is not None else None,
            round(base[2], 4) if base[2] is not None else None,
            status,
        ])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def save_yaw_report_csv(objects: list[dict],
                        path: str = "yaw_report.csv"):
    """Write per-object yaw/grasp direction report."""
    header = ["object_id", "class_id", "yaw_rad", "yaw_deg",
              "grasp_width_m", "approach_axis"]
    rows = []
    for o in objects:
        gh = o.get("grasp_hint", {})
        yaw_r = gh.get("yaw_rad", 0)
        rows.append([
            o["object_id"], o["class_id"],
            round(yaw_r, 4), round(np.degrees(yaw_r), 2),
            gh.get("grasp_width_m"), gh.get("approach_axis"),
        ])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def save_failure_cases_jsonl(objects: list[dict],
                             path: str = "failure_cases_perception.jsonl"):
    """Append failed objects to a JSONL log."""
    with open(path, "a") as f:
        for o in objects:
            if o["failure_reason"] is not None:
                line = {k: v for k, v in o.items() if k != "contour"}
                f.write(json.dumps(line, default=str) + "\n")
    return path
