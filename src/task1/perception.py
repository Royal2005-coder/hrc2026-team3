"""
perception.py — End-to-end perception pipeline for HRC2026 Task 1.

Pipeline:  RGB-D → detect → classify A/B → centroid/yaw → camera-to-base → ObjectState JSON

Author: Thanh Tai (N1)
"""

import cv2 as cv
import numpy as np
import json
import csv
import time

from .camera_utils import (
    CameraIntrinsics,
    pixel_to_camera_point,
    robust_depth_from_patch,
    robust_depth_from_mask,
    median_depth_in_mask,
    valid_depth_mask,
)
from .transform_utils import (
    transform_point,
    rotation_matrix_to_quaternion,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Detection — colour-based
# ═══════════════════════════════════════════════════════════════════════════
def detect_by_color(rgb_bgr: np.ndarray,
                    lower_hsv: list | np.ndarray,
                    upper_hsv: list | np.ndarray,
                    min_area: int = 15) -> tuple[list[dict], np.ndarray]:
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

    # Restrict to lower 60% of image — parts only on table, not walls/ceiling
    h_img = mask.shape[0]
    mask[:int(h_img * 0.40), :] = 0

    kernel = np.ones((3, 3), np.uint8)
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
# 1a. Detection — Isaac Sim annotators (best accuracy, sim-only)
# ═══════════════════════════════════════════════════════════════════════════
_LABEL_MAP = {
    "part_a": "part_A",
    "part_b": "part_B",
    "parta":  "part_A",
    "partb":  "part_B",
    "part a": "part_A",
    "part b": "part_B",
}


def _parse_sem_label(label_val) -> str | None:
    """Normalise Isaac Sim semantic label (str or dict) to 'part_A'/'part_B'."""
    if isinstance(label_val, dict):
        raw = label_val.get("class", label_val.get("name", "")).lower().strip()
    else:
        raw = str(label_val).lower().strip()
    return _LABEL_MAP.get(raw)


def detect_by_annotation(bbox_data: dict,
                          sem_data: dict,
                          depth: np.ndarray,
                          min_area: int = 4) -> list[dict]:
    """
    Detect objects from Isaac Sim semantic_segmentation annotator.

    Uses the semantic mask (H×W) + idToLabels to find part_A/part_B instances
    via connected components — no bbox annotator needed.

    Parameters
    ----------
    bbox_data : ignored (kept for API compat)
    sem_data  : output of semantic_segmentation.get_data()
    depth     : (H, W) depth array in metres
    min_area  : minimum contour area in pixels

    Returns
    -------
    list of detection dicts
    """
    if sem_data is None:
        return []

    # Extract mask array and label map
    if isinstance(sem_data, dict):
        mask_arr   = sem_data.get("data")
        id_to_labels = sem_data.get("info", {}).get("idToLabels", {})
    else:
        return []

    if mask_arr is None or mask_arr.size == 0:
        return []

    # mask_arr may be (H, W) uint32 or (H, W, 4) RGBA — take first channel
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    mask_arr = mask_arr.astype(np.int32)

    # Build sem_id → class_id mapping
    sem_class = {}
    for key, val in id_to_labels.items():
        cid = _parse_sem_label(val)
        if cid is not None:
            sem_class[int(key)] = cid

    if not sem_class:
        return []

    detections = []
    for sem_id, class_id in sem_class.items():
        binary = np.uint8(mask_arr == sem_id) * 255

        # Find connected components (each instance separately)
        num_labels, labels_cc, stats, _ = cv.connectedComponentsWithStats(
            binary, connectivity=8)

        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv.CC_STAT_AREA])
            if area < min_area:
                continue

            x1 = int(stats[lbl, cv.CC_STAT_LEFT])
            y1 = int(stats[lbl, cv.CC_STAT_TOP])
            w  = int(stats[lbl, cv.CC_STAT_WIDTH])
            h  = int(stats[lbl, cv.CC_STAT_HEIGHT])
            x2, y2 = x1 + w, y1 + h

            component_mask = np.uint8(labels_cc == lbl) * 255
            contours, _ = cv.findContours(component_mask,
                                          cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv.contourArea) if contours else None

            m = cv.moments(component_mask)
            if m["m00"] > 0:
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
            else:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            detections.append({
                "bbox_xyxy":   [x1, y1, x2, y2],
                "centroid_px": [cx, cy],
                "area_px":     float(area),
                "class_id":    class_id,
                "confidence":  0.99,
                "contour":     contour,
            })

    return detections


# ═══════════════════════════════════════════════════════════════════════════
# 1b. Detection — depth foreground (recommended for Task 1)
# ═══════════════════════════════════════════════════════════════════════════
def detect_by_depth_foreground(depth: np.ndarray,
                               fg_threshold_m: float = 0.015,
                               max_height_m: float = 0.12,
                               min_area: int = 15,
                               reference_depth: float | None = None,
                               search_bbox: tuple | None = None) -> tuple[list[dict], np.ndarray]:
    """
    Detect objects above the table surface using depth.

    Only pixels in the band (table_depth - max_height_m, table_depth - fg_threshold_m)
    are foreground — this excludes robot arms which are much farther above the table.
    """
    valid = np.isfinite(depth) & (depth > 0)
    if valid.sum() == 0:
        return [], np.zeros(depth.shape[:2], dtype=np.uint8)

    if reference_depth is not None:
        table_depth = reference_depth
    else:
        h, w = depth.shape
        ry1, ry2 = int(h * 0.55), h
        rx1, rx2 = int(w * 0.15), int(w * 0.85)
        roi = depth[ry1:ry2, rx1:rx2]
        roi_valid = np.isfinite(roi) & (roi > 0)
        table_depth = float(np.median(roi[roi_valid])) if roi_valid.sum() > 0 \
                      else float(np.median(depth[valid]))

    fg_mask = np.zeros(depth.shape[:2], dtype=np.uint8)
    fg_mask[valid
            & (depth < table_depth - fg_threshold_m)
            & (depth > table_depth - max_height_m)] = 255

    if search_bbox is not None:
        x1, y1, x2, y2 = search_bbox
        border_mask = np.zeros_like(fg_mask)
        border_mask[y1:y2, x1:x2] = 255
        fg_mask = cv.bitwise_and(fg_mask, border_mask)

    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)
    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel)

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
# 2. Shape features + classification
# ═══════════════════════════════════════════════════════════════════════════
def extract_shape_features(contour: np.ndarray) -> dict:
    """Extract shape features from a contour for A/B classification."""
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    x, y, w, h = cv.boundingRect(contour)
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull) if len(hull) >= 3 else area

    aspect_ratio = float(w) / h if h > 0 else 0.0
    solidity = area / hull_area if hull_area > 0 else 0.0
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0

    moments = cv.moments(contour)
    hu = cv.HuMoments(moments).flatten()
    hu_log = np.array([-np.sign(h_) * np.log10(abs(h_) + 1e-20) for h_ in hu])

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
    Returns (None, 0.0) if colour is ambiguous.

    hsv_ranges format:
      red:   {lower, upper, lower2, upper2, implies_class: "part_A"}
      blue:  {lower, upper, implies_class: "part_B"}
    """
    hsv = cv.cvtColor(rgb_bgr, cv.COLOR_BGR2HSV)
    mask_contour = np.zeros(rgb_bgr.shape[:2], dtype=np.uint8)
    cv.drawContours(mask_contour, [contour], -1, 255, -1)

    for color_name, cfg in hsv_ranges.items():
        if cfg.get("implies_class") is None:
            continue

        m1 = cv.inRange(hsv, np.array(cfg["lower"]), np.array(cfg["upper"]))
        if "lower2" in cfg:
            m2 = cv.inRange(hsv, np.array(cfg["lower2"]), np.array(cfg["upper2"]))
            m1 = m1 | m2

        overlap = cv.bitwise_and(m1, mask_contour)
        overlap_ratio = overlap.sum() / (mask_contour.sum() + 1e-8)

        if overlap_ratio > 0.3:
            return cfg["implies_class"], min(0.95, 0.7 + overlap_ratio * 0.3)

    return None, 0.0


def classify_by_shape(features: dict,
                      part_A_aspect_range: tuple = (0.0, 999.0),
                      part_B_aspect_range: tuple = (0.0, 999.0)) -> tuple[str, float]:
    """
    Classify using shape when colour is ambiguous.
    TODO: measure real Part A / Part B aspect ratios and fill in thresholds.
    """
    ar = features["min_rect_aspect"]
    if part_A_aspect_range[0] <= ar <= part_A_aspect_range[1]:
        return "part_A", 0.65
    if part_B_aspect_range[0] <= ar <= part_B_aspect_range[1]:
        return "part_B", 0.65
    return "unknown", 0.40


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
    return float(np.deg2rad(rect[-1]))


def estimate_yaw_pca(contour: np.ndarray) -> float:
    """Yaw from PCA on contour points (more stable for elongated objects)."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    mean = pts.mean(axis=0)
    pts_c = pts - mean
    cov = np.cov(pts_c, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]
    return float(np.arctan2(principal[1], principal[0]))


# ═══════════════════════════════════════════════════════════════════════════
# 4. Confidence scoring
# ═══════════════════════════════════════════════════════════════════════════
def compute_confidence(area_px: float,
                       depth_valid: bool,
                       mask_quality: float = 1.0,
                       class_ambiguous: bool = False,
                       min_area: float = 200) -> float:
    """Rule-based confidence in [0, 1]. Start at 1.0 and subtract penalties."""
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
    """Build a single ObjectState dict from a detection + depth + transforms."""
    u, v = det["centroid_px"]
    class_id = det.get("class_id", "unknown")
    contour = det.get("contour")

    # Prefer mask-based depth (more robust) over patch if contour available
    if contour is not None and len(contour) >= 3:
        mask_tmp = np.zeros(depth.shape[:2], dtype=np.uint8)
        cv.drawContours(mask_tmp, [contour], -1, 255, -1)
        z = median_depth_in_mask(depth, mask_tmp)
        mask_quality = float((mask_tmp > 0).sum()) / max(depth.shape[0] * depth.shape[1], 1)
    else:
        z = robust_depth_from_patch(depth, u, v, radius=3)
        mask_quality = 0.5

    if z is not None:
        z = z * intr.depth_scale()

    depth_valid = z is not None and z > 0

    conf = compute_confidence(
        area_px=det.get("area_px", 0),
        depth_valid=depth_valid,
        mask_quality=mask_quality,
    )

    failure = None
    if not depth_valid:
        failure = "INVALID_DEPTH"
    elif conf < confidence_threshold:
        failure = "LOW_CONFIDENCE"
    elif T_base_camera is None:
        failure = "TRANSFORM_NOT_AVAILABLE"

    centroid_camera = None
    if depth_valid:
        centroid_camera = pixel_to_camera_point(u, v, z, intr)

    pose_base = None
    if centroid_camera is not None and T_base_camera is not None:
        p_base = transform_point(T_base_camera, centroid_camera)
        pose_base = {
            "position_m": p_base.tolist(),
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }

    yaw = 0.0
    if contour is not None and len(contour) >= 5:
        if yaw_method == "pca":
            yaw = estimate_yaw_pca(contour)
        else:
            yaw = estimate_yaw_minrect(contour)

    return {
        "object_id": object_id,
        "class_id": class_id,
        "confidence": round(conf, 3),
        "bbox_xyxy": det["bbox_xyxy"],
        "centroid_px": det["centroid_px"],
        "mask_quality": round(mask_quality, 4),
        "centroid_camera_m": centroid_camera.tolist() if centroid_camera is not None else None,
        "pose_base": pose_base,
        "grasp_hint": {
            "approach_axis": "z_down",
            "yaw_rad": round(yaw, 4),
            "grasp_width_m": default_grasp_width,
        },
        "failure_reason": failure,
    }


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
                   detection_method: str = "color",
                   reference_depth: float | None = None,
                   bbox_ann_data: dict | None = None,
                   sem_ann_data: dict | None = None) -> dict:
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

    if detection_method == "annotation":
        all_dets = detect_by_annotation(bbox_ann_data, sem_ann_data, depth)

    elif detection_method == "depth_fg":
        h, w = depth.shape[:2]
        # Exclude robot arms at left/right edges — focus on centre table area
        search_bbox = (int(w * 0.20), int(h * 0.40), int(w * 0.80), h)
        all_dets, _ = detect_by_depth_foreground(
            depth, reference_depth=reference_depth, search_bbox=search_bbox)
        for det in all_dets:
            contour = det["contour"]
            cls, cls_conf = classify_by_color_hint(rgb_bgr, contour, hsv_ranges)
            if cls is not None:
                det["class_id"] = cls
                det["confidence"] = cls_conf
            else:
                features = extract_shape_features(contour)
                det["shape_features"] = features
                cls, cls_conf = classify_by_shape(features)
                det["class_id"] = cls
                det["confidence"] = cls_conf
                if cls == "unknown":
                    det["failure_reason_hint"] = "CLASS_AMBIGUOUS"
    else:
        # Colour-based detection: red (with wrap-around), blue, ori
        r = hsv_ranges.get("red", {})
        dets_A, mask_r1 = detect_by_color(rgb_bgr, r.get("lower", [0,80,80]),
                                           r.get("upper", [15,255,255]))
        if "lower2" in r:
            dets_A2, mask_r2 = detect_by_color(rgb_bgr, r["lower2"], r["upper2"])
            dets_A = dets_A + dets_A2
        for d in dets_A:
            d["class_id"] = "part_A"
            d["confidence"] = 0.9

        b = hsv_ranges.get("blue", {})
        dets_B, _ = detect_by_color(rgb_bgr, b.get("lower", [95,120,80]),
                                     b.get("upper", [135,255,255]))
        for d in dets_B:
            d["class_id"] = "part_B"
            d["confidence"] = 0.9

        ori = hsv_ranges.get("ori_color", {})
        dets_ori, _ = detect_by_color(rgb_bgr, ori.get("lower", [10,80,80]),
                                       ori.get("upper", [35,255,255]))
        ori_class = ori.get("implies_class", "part_B")
        for d in dets_ori:
            d["class_id"] = ori_class
            d["confidence"] = 0.85

        # copper-colored part_A (brownish metallic)
        cop = hsv_ranges.get("copper", {})
        dets_cop, _ = detect_by_color(rgb_bgr, cop.get("lower", [8,80,60]),
                                       cop.get("upper", [20,200,200]))
        for d in dets_cop:
            d["class_id"] = cop.get("implies_class", "part_A")
            d["confidence"] = 0.85

        all_dets = dets_A + dets_B + dets_ori + dets_cop

    objects = []
    for idx, det in enumerate(all_dets):
        obj = make_object_state(det, depth, intr, T_base_camera, object_id=f"obj_{idx:03d}")
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
# 7. Public interface for task1_runner (N2 calls this)
# ═══════════════════════════════════════════════════════════════════════════

# Default HSV ranges — có thể override khi gọi detect_parts()
_DEFAULT_HSV_RANGES = {
    "red": {
        "lower": [0, 100, 100], "upper": [10, 255, 255],
        "lower2": [170, 100, 100], "upper2": [179, 255, 255],
        "implies_class": "part_A",
    },
    "copper": {
        # brownish-metallic copper colour in Isaac Sim
        "lower": [8, 80, 60], "upper": [20, 200, 200],
        "implies_class": "part_A",
    },
    "blue": {
        "lower": [100, 100, 100], "upper": [130, 255, 255],
        "implies_class": "part_B",
    },
    "ori_color": {
        # orange/beige part_B variant
        "lower": [10, 80, 80], "upper": [35, 255, 255],
        "implies_class": "part_B",
    },
}


def detect_parts(rgb: np.ndarray,
                 depth: np.ndarray,
                 intr: "CameraIntrinsics",
                 T_base_camera: np.ndarray,
                 confidence_threshold: float = 0.60,
                 detection_method: str = "annotation",
                 hsv_ranges: dict = None,
                 bbox_ann_data: dict | None = None,
                 sem_ann_data: dict | None = None) -> list[dict]:
    """
    Interface chính cho task1_runner — N2 gọi hàm này.

    Parameters
    ----------
    rgb               : RGB image từ robot.get_camera_rgbd() (H, W, 3)
    depth             : depth map (H, W) float32, đơn vị metre
    intr              : CameraIntrinsics của head_left
    T_base_camera     : 4×4 transform camera → robot base
    confidence_threshold : lọc vật có confidence thấp
    detection_method  : "depth_fg" (recommended) | "color"
    hsv_ranges        : override HSV config nếu cần

    Returns
    -------
    list[dict] — chỉ những vật hợp lệ, mỗi vật có:
        object_id    : "obj_000"
        class_id     : "part_A" | "part_B"
        confidence   : float
        pose_base    : {"position_m": [x, y, z], "quaternion_xyzw": [...]}
        grasp_hint   : {"yaw_rad": float, "grasp_width_m": float, ...}
        failure_reason: None
    """
    import cv2 as cv

    # RGB → BGR cho OpenCV
    if rgb.shape[2] == 4:
        bgr = cv.cvtColor(rgb[:, :, :3], cv.COLOR_RGB2BGR)
    else:
        bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

    # Normalise depth
    depth = np.array(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    state = run_perception(
        bgr, depth, intr, T_base_camera,
        hsv_ranges=hsv_ranges or _DEFAULT_HSV_RANGES,
        detection_method=detection_method,
        bbox_ann_data=bbox_ann_data,
        sem_ann_data=sem_ann_data,
    )

    return [
        o for o in state["objects"]
        if o["failure_reason"] is None
        and o["confidence"] >= confidence_threshold
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 8. Export helpers
# ═══════════════════════════════════════════════════════════════════════════
def save_perception_json(state: dict, path: str = "perception_interface.json"):
    """Write perception output to JSON (strips non-serialisable contours)."""
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
        base = o["pose_base"]["position_m"] if o["pose_base"] else [None] * 3
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
