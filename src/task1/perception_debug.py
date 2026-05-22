"""
perception_debug.py — Overlay drawing and debug artifact generation.

Produces: overlay_detection.png, overlay_mask.png, overlay_centroid.png
as required by Task 1 Người 1 rubric.
"""

import cv2 as cv
import numpy as np
import json
import csv
import os


# ── colour palette per class ─────────────────────────────────────────────────
_CLASS_COLORS = {
    "part_A":  (0,   165, 255),   # orange
    "part_B":  (255,  50,  50),   # blue
    "unknown": (180, 180, 180),   # grey
}
_DEFAULT_COLOR = (0, 255, 0)


def _color(class_id: str) -> tuple:
    return _CLASS_COLORS.get(class_id, _DEFAULT_COLOR)


# ── core overlay helpers ──────────────────────────────────────────────────────
def draw_detections(rgb_bgr: np.ndarray, objects: list[dict]) -> np.ndarray:
    """
    Draw bbox, centroid and label for every object.
    Returns a copy of the image with overlays.
    """
    img = rgb_bgr.copy()
    for obj in objects:
        x1, y1, x2, y2 = [int(v) for v in obj["bbox_xyxy"]]
        u, v = obj["centroid_px"]
        class_id = obj.get("class_id", "unknown")
        conf = obj.get("confidence", 0.0)
        fail = obj.get("failure_reason")

        col = _color(class_id)
        thickness = 1 if fail else 2

        cv.rectangle(img, (x1, y1), (x2, y2), col, thickness)
        cv.circle(img, (int(u), int(v)), 4, (0, 0, 255), -1)

        label = f'{class_id} {conf:.2f}'
        if fail:
            label += f' [{fail}]'
        cv.putText(img, label, (x1, max(0, y1 - 5)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def draw_masks(rgb_bgr: np.ndarray, objects: list[dict]) -> np.ndarray:
    """Draw filled contours as a semi-transparent mask overlay."""
    img = rgb_bgr.copy()
    overlay = img.copy()
    for obj in objects:
        contour = obj.get("contour")
        if contour is None:
            continue
        class_id = obj.get("class_id", "unknown")
        col = _color(class_id)
        cv.drawContours(overlay, [contour], -1, col, -1)
    cv.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    return img


def draw_centroids_3d(rgb_bgr: np.ndarray, objects: list[dict],
                      intr, T_base_camera=None) -> np.ndarray:
    """
    Project centroid_camera_m back to pixel and draw a cross.
    Annotates with z_cam depth value.
    """
    img = rgb_bgr.copy()
    for obj in objects:
        cam = obj.get("centroid_camera_m")
        if cam is None:
            continue
        x_c, y_c, z_c = cam
        if z_c <= 0:
            continue
        u_proj = intr.fx * x_c / z_c + intr.cx
        v_proj = intr.fy * y_c / z_c + intr.cy
        col = _color(obj.get("class_id", "unknown"))
        p = (int(u_proj), int(v_proj))
        cv.drawMarker(img, p, col, cv.MARKER_CROSS, 12, 2)
        cv.putText(img, f'z={z_c:.2f}m', (p[0] + 6, p[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    return img


# ── save helpers ─────────────────────────────────────────────────────────────
def save_overlays(rgb_bgr: np.ndarray,
                  perception_state: dict,
                  output_dir: str = "lab_outputs/perception",
                  intr=None) -> dict:
    """
    Generate and save all required debug overlays.

    Returns dict of {name: path}.
    """
    os.makedirs(output_dir, exist_ok=True)
    objects = perception_state.get("objects", [])
    paths = {}

    # overlay_detection.png
    img_det = draw_detections(rgb_bgr, objects)
    p = os.path.join(output_dir, "overlay_detection.png")
    cv.imwrite(p, img_det)
    paths["overlay_detection"] = p

    # overlay_mask.png
    img_mask = draw_masks(rgb_bgr, objects)
    p = os.path.join(output_dir, "overlay_mask.png")
    cv.imwrite(p, img_mask)
    paths["overlay_mask"] = p

    # overlay_centroid.png — with 3D projection if intrinsics available
    if intr is not None:
        img_cent = draw_centroids_3d(rgb_bgr, objects, intr)
    else:
        img_cent = rgb_bgr.copy()
        for obj in objects:
            u, v = obj["centroid_px"]
            cv.circle(img_cent, (int(u), int(v)), 6, (0, 0, 255), 2)
    p = os.path.join(output_dir, "overlay_centroid.png")
    cv.imwrite(p, img_cent)
    paths["overlay_centroid"] = p

    return paths


def save_confusion_matrix_csv(objects: list[dict],
                              gt_labels: list[str] | None = None,
                              path: str = "lab_outputs/perception/confusion_matrix_task1.csv"):
    """
    Write per-object predicted class (and GT if available).
    Produces confusion_matrix_task1.csv required by rubric.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["object_id", "predicted_class", "confidence", "failure_reason"]
        if gt_labels is not None:
            header.insert(2, "gt_class")
        w.writerow(header)
        for i, obj in enumerate(objects):
            row = [obj["object_id"], obj.get("class_id", "unknown"),
                   round(obj.get("confidence", 0), 3), obj.get("failure_reason")]
            if gt_labels is not None:
                gt = gt_labels[i] if i < len(gt_labels) else "?"
                row.insert(2, gt)
            w.writerow(row)
    return path


def validate_perception_output(perception_state: dict) -> tuple[bool, list[str]]:
    """
    Validate a perception_interface.json against required schema.
    Returns (is_valid, list_of_errors).
    """
    required_top = ["frame_id", "timestamp", "camera_name", "objects", "summary"]
    required_obj = ["object_id", "class_id", "confidence", "bbox_xyxy", "centroid_px",
                    "centroid_camera_m", "pose_base", "grasp_hint", "failure_reason"]
    errors = []

    for k in required_top:
        if k not in perception_state:
            errors.append(f"missing top-level field: {k}")

    for i, obj in enumerate(perception_state.get("objects", [])):
        for k in required_obj:
            if k not in obj:
                errors.append(f"object[{i}]: missing field '{k}'")
        conf = obj.get("confidence", -1)
        if not (0.0 <= conf <= 1.0):
            errors.append(f"object[{i}]: confidence={conf} out of [0,1]")
        if len(obj.get("bbox_xyxy", [])) != 4:
            errors.append(f"object[{i}]: bbox_xyxy must have 4 numbers")

    return len(errors) == 0, errors
