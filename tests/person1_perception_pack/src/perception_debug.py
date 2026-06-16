"""
perception_debug.py — Visualization & evaluation helpers for Task 1 perception.

Produces overlay images, confusion matrices, and depth previews
so the team can verify detection quality visually.
"""

import cv2 as cv
import numpy as np
import csv
import os
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# 1. Overlay — detection boxes + labels
# ═══════════════════════════════════════════════════════════════════════════
COLOR_MAP = {
    "part_A": (0, 255, 0),   # green
    "part_B": (255, 128, 0), # orange
    "unknown": (128, 128, 128),
}


def draw_detection_overlay(rgb_bgr: np.ndarray,
                           objects: list[dict],
                           save_path: str = "overlay_detection.png") -> np.ndarray:
    """Draw bounding boxes, centroids, and labels on the image."""
    img = rgb_bgr.copy()
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        u, v = obj["centroid_px"]
        cid = obj.get("class_id", "unknown")
        conf = obj.get("confidence", 0)
        color = COLOR_MAP.get(cid, (255, 255, 255))
        label = f'{cid} {conf:.2f}'

        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv.circle(img, (int(u), int(v)), 5, (0, 0, 255), -1)
        cv.putText(img, label, (x1, max(0, y1 - 8)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                   cv.LINE_AA)

        # failure flag
        fr = obj.get("failure_reason")
        if fr:
            cv.putText(img, fr, (x1, y2 + 15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                       cv.LINE_AA)

    if save_path:
        cv.imwrite(save_path, img)
    return img


# ═══════════════════════════════════════════════════════════════════════════
# 2. Overlay — mask
# ═══════════════════════════════════════════════════════════════════════════
def draw_mask_overlay(rgb_bgr: np.ndarray,
                      mask_A: np.ndarray,
                      mask_B: np.ndarray,
                      alpha: float = 0.4,
                      save_path: str = "overlay_mask.png") -> np.ndarray:
    """Blend coloured masks on top of the RGB image."""
    overlay = rgb_bgr.copy()

    green = np.zeros_like(overlay)
    green[:, :] = COLOR_MAP["part_A"]
    overlay[mask_A > 0] = cv.addWeighted(
        overlay, 1 - alpha, green, alpha, 0
    )[mask_A > 0]

    orange = np.zeros_like(overlay)
    orange[:, :] = COLOR_MAP["part_B"]
    overlay[mask_B > 0] = cv.addWeighted(
        overlay, 1 - alpha, orange, alpha, 0
    )[mask_B > 0]

    if save_path:
        cv.imwrite(save_path, overlay)
    return overlay


# ═══════════════════════════════════════════════════════════════════════════
# 3. Overlay — centroid + yaw arrow
# ═══════════════════════════════════════════════════════════════════════════
def draw_centroid_overlay(rgb_bgr: np.ndarray,
                          objects: list[dict],
                          arrow_length: int = 40,
                          save_path: str = "overlay_centroid.png") -> np.ndarray:
    """Draw centroid dots and yaw-direction arrows."""
    img = rgb_bgr.copy()
    for obj in objects:
        u, v = obj["centroid_px"]
        u_int, v_int = int(u), int(v)
        cid = obj.get("class_id", "unknown")
        color = COLOR_MAP.get(cid, (255, 255, 255))

        # centroid dot
        cv.circle(img, (u_int, v_int), 6, color, -1)
        cv.circle(img, (u_int, v_int), 8, (255, 255, 255), 1)

        # yaw arrow
        gh = obj.get("grasp_hint", {})
        yaw = gh.get("yaw_rad", 0)
        dx = int(arrow_length * np.cos(yaw))
        dy = int(arrow_length * np.sin(yaw))
        cv.arrowedLine(img, (u_int, v_int), (u_int + dx, v_int + dy),
                       (0, 255, 255), 2, tipLength=0.3)

        # label
        label = f'{obj.get("object_id", "")} yaw={np.degrees(yaw):.1f}°'
        cv.putText(img, label, (u_int + 10, v_int - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                   cv.LINE_AA)

    if save_path:
        cv.imwrite(save_path, img)
    return img


# ═══════════════════════════════════════════════════════════════════════════
# 4. Depth preview
# ═══════════════════════════════════════════════════════════════════════════
def save_depth_preview(depth: np.ndarray,
                       save_path: str = "sample_depth_preview.png") -> str:
    """Normalize depth to 0-255 and save a colormapped preview."""
    valid = np.isfinite(depth) & (depth > 0)
    vis = np.zeros_like(depth, dtype=np.uint8)
    if valid.any():
        d_min, d_max = depth[valid].min(), depth[valid].max()
        if d_max > d_min:
            normed = (depth - d_min) / (d_max - d_min)
            normed = np.clip(normed, 0, 1)
            vis = (normed * 255).astype(np.uint8)
        vis[~valid] = 0
    coloured = cv.applyColorMap(vis, cv.COLORMAP_TURBO)
    cv.imwrite(save_path, coloured)
    return save_path


# ═══════════════════════════════════════════════════════════════════════════
# 5. Confusion matrix
# ═══════════════════════════════════════════════════════════════════════════
def build_confusion_matrix(predictions: list[str],
                           ground_truths: list[str],
                           classes: list[str] = None) -> dict:
    """
    Build a confusion matrix dict from predicted vs ground-truth labels.

    Returns
    -------
    dict with keys: "classes", "matrix" (list of lists), "accuracy"
    """
    if classes is None:
        classes = sorted(set(predictions + ground_truths))
    n = len(classes)
    c2i = {c: i for i, c in enumerate(classes)}
    mat = [[0] * n for _ in range(n)]
    for gt, pred in zip(ground_truths, predictions):
        mat[c2i[gt]][c2i[pred]] += 1
    total = sum(sum(row) for row in mat)
    correct = sum(mat[i][i] for i in range(n))
    acc = correct / total if total > 0 else 0.0
    return {"classes": classes, "matrix": mat, "accuracy": acc}


def save_confusion_matrix_csv(cm: dict,
                              path: str = "confusion_matrix_task1.csv"):
    """Save confusion matrix as CSV. Rows = ground truth, cols = predicted."""
    classes = cm["classes"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["GT \\ Pred"] + classes)
        for i, cls in enumerate(classes):
            w.writerow([cls] + cm["matrix"][i])
        w.writerow([])
        w.writerow(["accuracy", f'{cm["accuracy"]:.4f}'])
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 6. Quick "generate all overlays" helper
# ═══════════════════════════════════════════════════════════════════════════
def generate_all_debug_outputs(rgb_bgr: np.ndarray,
                               depth: np.ndarray,
                               objects: list[dict],
                               mask_A: np.ndarray,
                               mask_B: np.ndarray,
                               output_dir: str = "outputs"):
    """One-call helper to dump every debug artefact into output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    p = lambda name: os.path.join(output_dir, name)

    cv.imwrite(p("sample_rgb.png"), rgb_bgr)
    save_depth_preview(depth, p("sample_depth_preview.png"))
    draw_detection_overlay(rgb_bgr, objects, p("overlay_detection.png"))
    draw_mask_overlay(rgb_bgr, mask_A, mask_B, save_path=p("overlay_mask.png"))
    draw_centroid_overlay(rgb_bgr, objects, save_path=p("overlay_centroid.png"))
    np.save(p("sample_depth.npy"), depth)
    print(f"[DEBUG] All overlays saved to {output_dir}/")
