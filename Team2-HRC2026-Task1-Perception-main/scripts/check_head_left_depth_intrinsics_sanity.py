"""Check head_left depth and provisional intrinsics sanity without Isaac Sim.

This script reads the current 512x512 RGB-D capture artifacts and
camera_geometry.json. It does not compute pixel-to-3D, T_base_camera, or
pose_base because depth unit, RGB-depth alignment, and T_base_camera remain
unverified.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task1" / "perception"
SAMPLES_DIR = OUTPUT_ROOT / "samples"
CAMERA_INVENTORY_DIR = OUTPUT_ROOT / "camera_inventory"
REPORTS_DIR = OUTPUT_ROOT / "reports"

CAMERA_NAME = "head_left"
DEPTH_PATH = SAMPLES_DIR / f"sample_depth_{CAMERA_NAME}.npy"
RGB_PATH = SAMPLES_DIR / f"sample_rgb_{CAMERA_NAME}.png"
DEPTH_VIS_PATH = SAMPLES_DIR / f"sample_depth_vis_{CAMERA_NAME}.png"
GEOMETRY_PATH = CAMERA_INVENTORY_DIR / "camera_geometry.json"
OUTPUT_JSON_PATH = CAMERA_INVENTORY_DIR / "depth_intrinsics_sanity.json"
OUTPUT_REPORT_PATH = REPORTS_DIR / "depth_intrinsics_sanity_report.md"


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


def load_camera_geometry() -> Dict[str, Any]:
    data = json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))
    for camera in data.get("cameras", []):
        if camera.get("camera_name") == CAMERA_NAME:
            return camera
    raise KeyError(f"{CAMERA_NAME} not found in {GEOMETRY_PATH}")


def sample_depth_points(depth: np.ndarray) -> List[Dict[str, Any]]:
    height, width = depth.shape
    cx = width // 2
    cy = height // 2
    points: List[Tuple[str, int, int]] = [
        ("center", cx, cy),
        ("center_left_10px", cx - 10, cy),
        ("center_right_10px", cx + 10, cy),
        ("center_up_10px", cx, cy - 10),
        ("center_down_10px", cx, cy + 10),
        ("center_up_left_10px", cx - 10, cy - 10),
        ("center_down_right_10px", cx + 10, cy + 10),
    ]

    samples = []
    for label, x, y in points:
        value = float(depth[y, x])
        samples.append(
            {
                "label": label,
                "x": x,
                "y": y,
                "depth_value": value if np.isfinite(value) else str(value),
                "is_valid": bool(np.isfinite(value) and value > 0),
            }
        )
    return samples


def summarize_depth(depth: np.ndarray) -> Dict[str, Any]:
    valid = np.isfinite(depth) & (depth > 0)
    invalid = ~valid
    finite = np.isfinite(depth)

    result: Dict[str, Any] = {
        "shape": list(depth.shape),
        "dtype": str(depth.dtype),
        "total_pixels": int(depth.size),
        "finite_pixels": int(finite.sum()),
        "valid_pixels": int(valid.sum()),
        "invalid_pixels": int(invalid.sum()),
        "inf_pixels": int(np.isinf(depth).sum()),
        "nan_pixels": int(np.isnan(depth).sum()),
        "valid_ratio": float(valid.mean()),
        "center_samples": sample_depth_points(depth),
    }

    if valid.any():
        values = depth[valid]
        result.update(
            {
                "depth_min": float(np.min(values)),
                "depth_p01": float(np.percentile(values, 1.0)),
                "depth_p05": float(np.percentile(values, 5.0)),
                "depth_median": float(np.median(values)),
                "depth_p95": float(np.percentile(values, 95.0)),
                "depth_p99": float(np.percentile(values, 99.0)),
                "depth_max": float(np.max(values)),
            }
        )
    else:
        result.update(
            {
                "depth_min": None,
                "depth_p01": None,
                "depth_p05": None,
                "depth_median": None,
                "depth_p95": None,
                "depth_p99": None,
                "depth_max": None,
            }
        )
    return result


def build_result() -> Dict[str, Any]:
    depth = np.load(DEPTH_PATH)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth array, got shape {depth.shape}")

    camera_geometry = load_camera_geometry()
    intrinsics = camera_geometry.get("intrinsics", {})

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "camera_name": CAMERA_NAME,
        "input_files": {
            "depth_path": str(DEPTH_PATH),
            "rgb_path": str(RGB_PATH),
            "depth_vis_path": str(DEPTH_VIS_PATH),
            "camera_geometry_path": str(GEOMETRY_PATH),
        },
        "depth_summary": summarize_depth(depth),
        "intrinsics_summary": {
            "status": intrinsics.get("status", "unknown") if isinstance(intrinsics, dict) else "unknown",
            "fx": intrinsics.get("fx", "unknown") if isinstance(intrinsics, dict) else "unknown",
            "fy": intrinsics.get("fy", "unknown") if isinstance(intrinsics, dict) else "unknown",
            "cx": intrinsics.get("cx", "unknown") if isinstance(intrinsics, dict) else "unknown",
            "cy": intrinsics.get("cy", "unknown") if isinstance(intrinsics, dict) else "unknown",
            "matrix_3x3": intrinsics.get("matrix_3x3", "unknown") if isinstance(intrinsics, dict) else "unknown",
            "warning": "Intrinsics are provisional and not fully verified by projection sanity test.",
        },
        "geometry_known_unknowns": {
            "depth_unit": camera_geometry.get("depth_unit", "unknown"),
            "rgb_depth_alignment": camera_geometry.get("rgb_depth_alignment", "unknown"),
            "T_base_camera": camera_geometry.get("T_base_camera", "unknown"),
            "pose_base_claimed": False,
        },
        "scope_warning": (
            "This sanity check reads existing RGB-D artifacts only. It does not compute pixel-to-3D, "
            "T_base_camera, object detection, or pose_base."
        ),
    }


def write_report(result: Dict[str, Any]) -> None:
    depth = result["depth_summary"]
    intr = result["intrinsics_summary"]
    unknowns = result["geometry_known_unknowns"]

    sample_lines = [
        "| label | x | y | depth_value | valid |",
        "|---|---:|---:|---:|---:|",
    ]
    for sample in depth["center_samples"]:
        sample_lines.append(
            f"| `{sample['label']}` | {sample['x']} | {sample['y']} | `{sample['depth_value']}` | {sample['is_valid']} |"
        )

    lines = [
        "# Depth + Intrinsics Sanity Report",
        "",
        "## Status",
        "",
        f"- timestamp: `{result['timestamp']}`",
        f"- camera_name: `{result['camera_name']}`",
        f"- depth_path: `{result['input_files']['depth_path']}`",
        f"- rgb_path: `{result['input_files']['rgb_path']}`",
        f"- depth_vis_path: `{result['input_files']['depth_vis_path']}`",
        f"- camera_geometry_path: `{result['input_files']['camera_geometry_path']}`",
        "",
        "## Depth Raw Sanity",
        "",
        f"- shape: `{depth['shape']}`",
        f"- dtype: `{depth['dtype']}`",
        f"- total_pixels: `{depth['total_pixels']}`",
        f"- finite_pixels: `{depth['finite_pixels']}`",
        f"- valid_pixels: `{depth['valid_pixels']}`",
        f"- invalid_pixels: `{depth['invalid_pixels']}`",
        f"- inf_pixels: `{depth['inf_pixels']}`",
        f"- nan_pixels: `{depth['nan_pixels']}`",
        f"- valid_ratio: `{depth['valid_ratio']}`",
        f"- valid_depth_min: `{depth['depth_min']}`",
        f"- valid_depth_median: `{depth['depth_median']}`",
        f"- valid_depth_max: `{depth['depth_max']}`",
        f"- valid_depth_p01: `{depth['depth_p01']}`",
        f"- valid_depth_p99: `{depth['depth_p99']}`",
        "",
        "## Center Pixel Samples",
        "",
        *sample_lines,
        "",
        "## Intrinsics Sanity",
        "",
        f"- status: `{intr['status']}`",
        f"- fx: `{intr['fx']}`",
        f"- fy: `{intr['fy']}`",
        f"- cx: `{intr['cx']}`",
        f"- cy: `{intr['cy']}`",
        f"- matrix_3x3: `{intr['matrix_3x3']}`",
        f"- warning: `{intr['warning']}`",
        "",
        "## Known Unknowns",
        "",
        f"- depth_unit: `{unknowns['depth_unit']}`",
        f"- rgb_depth_alignment: `{unknowns['rgb_depth_alignment']}`",
        f"- T_base_camera: `{unknowns['T_base_camera']}`",
        f"- pose_base_claimed: `{unknowns['pose_base_claimed']}`",
        "",
        "## Conclusion",
        "",
        "- Depth raw file is readable and mostly valid for `head_left` at 512x512.",
        "- Intrinsics are available only as provisional values computed from USD focal/aperture fields.",
        "- This report does not claim `pose_base`.",
        "- Next safe step is to verify depth unit and RGB-depth alignment before pixel-to-3D.",
    ]
    OUTPUT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CAMERA_INVENTORY_DIR.mkdir(parents=True, exist_ok=True)

    result = build_result()
    OUTPUT_JSON_PATH.write_text(json.dumps(json_safe(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
