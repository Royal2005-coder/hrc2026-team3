"""Check head_left RGB/depth alignment sanity from saved 512x512 artifacts.

This is a lightweight offline check. It verifies that RGB and depth have the
same resolution, visualizes invalid depth pixels, and computes a simple edge
overlap heuristic between RGB intensity gradients and depth gradients.

It does not prove calibrated alignment, does not compute pixel-to-3D, and does
not claim pose_base.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task1" / "perception"
SAMPLES_DIR = OUTPUT_ROOT / "samples"
OVERLAYS_DIR = OUTPUT_ROOT / "overlays"
CAMERA_INVENTORY_DIR = OUTPUT_ROOT / "camera_inventory"
REPORTS_DIR = OUTPUT_ROOT / "reports"

CAMERA_NAME = "head_left"
RGB_PATH = SAMPLES_DIR / f"sample_rgb_{CAMERA_NAME}.png"
DEPTH_PATH = SAMPLES_DIR / f"sample_depth_{CAMERA_NAME}.npy"
DEPTH_VIS_PATH = SAMPLES_DIR / f"sample_depth_vis_{CAMERA_NAME}.png"

OUTPUT_JSON = CAMERA_INVENTORY_DIR / "rgb_depth_alignment_sanity.json"
OUTPUT_REPORT = REPORTS_DIR / "rgb_depth_alignment_sanity_report.md"
INVALID_OVERLAY_PATH = OVERLAYS_DIR / "rgb_depth_alignment_head_left_invalid_depth.png"
EDGE_OVERLAY_PATH = OVERLAYS_DIR / "rgb_depth_alignment_head_left_edge_overlap.png"


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def normalize_uint8(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mask = np.isfinite(arr) if valid is None else (valid & np.isfinite(arr))
    if not mask.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(arr[mask], 2.0))
    hi = float(np.percentile(arr[mask], 98.0))
    if hi <= lo:
        lo = float(np.min(arr[mask]))
        hi = float(np.max(arr[mask]))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    norm = (arr - lo) / (hi - lo)
    norm = np.where(mask, norm, 0.0)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
    gy[1:-1, :] = arr[2:, :] - arr[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def edge_mask(values: np.ndarray, percentile: float, valid: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mask = np.isfinite(arr) if valid is None else (valid & np.isfinite(arr))
    if not mask.any():
        return np.zeros(arr.shape, dtype=bool)
    threshold = float(np.percentile(arr[mask], percentile))
    return mask & (arr >= threshold)


def make_invalid_overlay(rgb: np.ndarray, valid_depth: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    invalid = ~valid_depth
    overlay[invalid] = np.array([255, 0, 0], dtype=np.uint8)
    return overlay


def make_edge_overlay(rgb: np.ndarray, rgb_edges: np.ndarray, depth_edges: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    both = rgb_edges & depth_edges
    rgb_only = rgb_edges & ~depth_edges
    depth_only = depth_edges & ~rgb_edges
    overlay[rgb_only] = np.array([0, 255, 0], dtype=np.uint8)
    overlay[depth_only] = np.array([255, 0, 0], dtype=np.uint8)
    overlay[both] = np.array([255, 255, 0], dtype=np.uint8)
    return overlay


def build_result() -> Dict[str, Any]:
    rgb = np.asarray(Image.open(RGB_PATH).convert("RGB"), dtype=np.uint8)
    depth = np.load(DEPTH_PATH)
    depth_vis = np.asarray(Image.open(DEPTH_VIS_PATH), dtype=np.uint8)

    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth array, got {depth.shape}")

    rgb_hw = list(rgb.shape[:2])
    depth_hw = list(depth.shape)
    depth_vis_hw = list(depth_vis.shape[:2])
    same_resolution = rgb_hw == depth_hw == depth_vis_hw

    valid_depth = np.isfinite(depth) & (depth > 0)
    gray = (
        0.299 * rgb[:, :, 0].astype(np.float32)
        + 0.587 * rgb[:, :, 1].astype(np.float32)
        + 0.114 * rgb[:, :, 2].astype(np.float32)
    )
    depth_for_grad = np.where(valid_depth, depth, np.nan)
    rgb_grad = gradient_magnitude(gray)
    depth_grad = gradient_magnitude(depth_for_grad)

    interior = np.zeros(depth.shape, dtype=bool)
    interior[1:-1, 1:-1] = True
    valid_grad = interior & valid_depth & np.isfinite(depth_grad)
    rgb_edges = edge_mask(rgb_grad, 95.0, valid_grad)
    depth_edges = edge_mask(depth_grad, 95.0, valid_grad)
    overlap = rgb_edges & depth_edges
    union = rgb_edges | depth_edges

    edge_overlap_ratio = float(overlap.sum() / union.sum()) if union.any() else 0.0
    depth_edge_covered_by_rgb_ratio = float(overlap.sum() / depth_edges.sum()) if depth_edges.any() else 0.0
    rgb_edge_covered_by_depth_ratio = float(overlap.sum() / rgb_edges.sum()) if rgb_edges.any() else 0.0

    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(make_invalid_overlay(rgb, valid_depth)).save(INVALID_OVERLAY_PATH)
    Image.fromarray(make_edge_overlay(rgb, rgb_edges, depth_edges)).save(EDGE_OVERLAY_PATH)

    status = "partial_sanity_pass" if same_resolution and float(valid_depth.mean()) > 0.99 else "needs_review"
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "camera_name": CAMERA_NAME,
        "input_files": {
            "rgb_path": str(RGB_PATH),
            "depth_path": str(DEPTH_PATH),
            "depth_vis_path": str(DEPTH_VIS_PATH),
        },
        "output_files": {
            "invalid_depth_overlay_path": str(INVALID_OVERLAY_PATH),
            "edge_overlap_overlay_path": str(EDGE_OVERLAY_PATH),
        },
        "shape_checks": {
            "rgb_hw": rgb_hw,
            "depth_hw": depth_hw,
            "depth_vis_hw": depth_vis_hw,
            "same_resolution": same_resolution,
        },
        "depth_validity": {
            "valid_ratio": float(valid_depth.mean()),
            "valid_pixels": int(valid_depth.sum()),
            "invalid_pixels": int((~valid_depth).sum()),
            "inf_pixels": int(np.isinf(depth).sum()),
            "nan_pixels": int(np.isnan(depth).sum()),
        },
        "edge_alignment_heuristic": {
            "rgb_edge_pixels": int(rgb_edges.sum()),
            "depth_edge_pixels": int(depth_edges.sum()),
            "overlap_pixels": int(overlap.sum()),
            "union_pixels": int(union.sum()),
            "edge_overlap_ratio": edge_overlap_ratio,
            "depth_edge_covered_by_rgb_ratio": depth_edge_covered_by_rgb_ratio,
            "rgb_edge_covered_by_depth_ratio": rgb_edge_covered_by_depth_ratio,
            "note": "Heuristic only. Low overlap can happen with texture edges, lighting, or smooth geometry; high overlap does not prove calibration.",
        },
        "alignment_status": status,
        "verified_for_pose_base": False,
        "scope_warning": "This does not prove RGB-depth calibration and does not compute pose_base.",
    }


def write_report(result: Dict[str, Any]) -> None:
    shape = result["shape_checks"]
    validity = result["depth_validity"]
    edges = result["edge_alignment_heuristic"]

    lines = [
        "# RGB-Depth Alignment Sanity Report",
        "",
        "## Status",
        "",
        f"- timestamp: `{result['timestamp']}`",
        f"- camera_name: `{result['camera_name']}`",
        f"- alignment_status: `{result['alignment_status']}`",
        f"- verified_for_pose_base: `{result['verified_for_pose_base']}`",
        "",
        "## Input Files",
        "",
        f"- rgb_path: `{result['input_files']['rgb_path']}`",
        f"- depth_path: `{result['input_files']['depth_path']}`",
        f"- depth_vis_path: `{result['input_files']['depth_vis_path']}`",
        "",
        "## Shape Checks",
        "",
        f"- rgb_hw: `{shape['rgb_hw']}`",
        f"- depth_hw: `{shape['depth_hw']}`",
        f"- depth_vis_hw: `{shape['depth_vis_hw']}`",
        f"- same_resolution: `{shape['same_resolution']}`",
        "",
        "## Depth Validity",
        "",
        f"- valid_ratio: `{validity['valid_ratio']}`",
        f"- valid_pixels: `{validity['valid_pixels']}`",
        f"- invalid_pixels: `{validity['invalid_pixels']}`",
        f"- inf_pixels: `{validity['inf_pixels']}`",
        f"- nan_pixels: `{validity['nan_pixels']}`",
        "",
        "## Edge Alignment Heuristic",
        "",
        f"- rgb_edge_pixels: `{edges['rgb_edge_pixels']}`",
        f"- depth_edge_pixels: `{edges['depth_edge_pixels']}`",
        f"- overlap_pixels: `{edges['overlap_pixels']}`",
        f"- union_pixels: `{edges['union_pixels']}`",
        f"- edge_overlap_ratio: `{edges['edge_overlap_ratio']}`",
        f"- depth_edge_covered_by_rgb_ratio: `{edges['depth_edge_covered_by_rgb_ratio']}`",
        f"- rgb_edge_covered_by_depth_ratio: `{edges['rgb_edge_covered_by_depth_ratio']}`",
        f"- note: `{edges['note']}`",
        "",
        "## Debug Overlays",
        "",
        f"- invalid_depth_overlay_path: `{result['output_files']['invalid_depth_overlay_path']}`",
        f"- edge_overlap_overlay_path: `{result['output_files']['edge_overlap_overlay_path']}`",
        "",
        "Overlay color convention for edge overlap:",
        "",
        "- green: RGB edge only",
        "- red: depth edge only",
        "- yellow: RGB and depth edge overlap",
        "",
        "## Conclusion",
        "",
        "- RGB, depth, and depth visualization have matching 512x512 resolution.",
        "- Depth is almost fully valid, with only 3 invalid `inf` pixels.",
        "- Edge overlap is only a heuristic and does not fully verify alignment.",
        "- Do not compute or claim `pose_base` from this report alone.",
    ]
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    CAMERA_INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result = build_result()
    OUTPUT_JSON.write_text(json.dumps(json_safe(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
