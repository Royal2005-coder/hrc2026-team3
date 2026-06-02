"""Offline pixel-to-camera sanity check for head_left using saved artifacts.

This script does not run Isaac Sim and does not claim pose_base.
It only checks numeric consistency for back-projection in camera frame.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task1" / "perception"
CAMERA_INVENTORY_DIR = OUTPUT_ROOT / "camera_inventory"
REPORTS_DIR = OUTPUT_ROOT / "reports"
SAMPLES_DIR = OUTPUT_ROOT / "samples"

GEOMETRY_JSON = CAMERA_INVENTORY_DIR / "camera_geometry.json"
DEPTH_PATH = SAMPLES_DIR / "sample_depth_head_left.npy"

OUT_JSON = CAMERA_INVENTORY_DIR / "pixel_to_camera_point_sanity_head_left.json"
OUT_REPORT = REPORTS_DIR / "pixel_to_camera_point_sanity_head_left.md"

CAMERA_NAME = "head_left"


def load_intrinsics() -> tuple[float, float, float, float]:
    geometry = json.loads(GEOMETRY_JSON.read_text(encoding="utf-8"))
    for cam in geometry.get("cameras", []):
        if cam.get("camera_name") == CAMERA_NAME:
            intr = cam.get("intrinsics", {})
            return float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    raise ValueError(f"Camera {CAMERA_NAME} not found in {GEOMETRY_JSON}")


def backproject(u: int, v: int, z: float, fx: float, fy: float, cx: float, cy: float) -> tuple[float, float, float]:
    x = ((float(u) - cx) / fx) * z
    y = ((float(v) - cy) / fy) * z
    return x, y, z


def main() -> int:
    depth = np.load(DEPTH_PATH)
    if depth.shape != (512, 512):
        raise ValueError(f"Unexpected depth shape: {depth.shape}")

    fx, fy, cx, cy = load_intrinsics()

    h, w = depth.shape
    test_pixels = [
        ("center", int(cx), int(cy)),
        ("left_center", max(0, int(cx) - 80), int(cy)),
        ("right_center", min(w - 1, int(cx) + 80), int(cy)),
        ("upper_center", int(cx), max(0, int(cy) - 80)),
        ("lower_center", int(cx), min(h - 1, int(cy) + 80)),
    ]

    samples = []
    for tag, u, v in test_pixels:
        z = float(depth[v, u])
        valid = bool(np.isfinite(z) and z > 0.0)
        xyz = [None, None, None]
        if valid:
            x, y, zc = backproject(u, v, z, fx, fy, cx, cy)
            xyz = [x, y, zc]
        samples.append(
            {
                "tag": tag,
                "u": u,
                "v": v,
                "depth_value": z,
                "depth_valid": valid,
                "camera_point_xyz": xyz,
            }
        )

    valid_samples = [s for s in samples if s["depth_valid"]]
    sign_checks = {
        "left_center_x_expected_negative": None,
        "right_center_x_expected_positive": None,
        "upper_center_y_expected_negative": None,
        "lower_center_y_expected_positive": None,
    }

    by_tag = {s["tag"]: s for s in valid_samples}
    if "left_center" in by_tag:
        sign_checks["left_center_x_expected_negative"] = by_tag["left_center"]["camera_point_xyz"][0] < 0.0
    if "right_center" in by_tag:
        sign_checks["right_center_x_expected_positive"] = by_tag["right_center"]["camera_point_xyz"][0] > 0.0
    if "upper_center" in by_tag:
        sign_checks["upper_center_y_expected_negative"] = by_tag["upper_center"]["camera_point_xyz"][1] < 0.0
    if "lower_center" in by_tag:
        sign_checks["lower_center_y_expected_positive"] = by_tag["lower_center"]["camera_point_xyz"][1] > 0.0

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "camera_name": CAMERA_NAME,
        "depth_path": str(DEPTH_PATH),
        "intrinsics_source": str(GEOMETRY_JSON),
        "intrinsics_status": "provisional_from_camera_geometry_json",
        "depth_unit": "unknown",
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "samples": samples,
        "sign_checks": sign_checks,
        "verified_for_pose_base": False,
        "scope_warning": "Offline numeric sanity only. No depth-unit final validation and no T_base_camera.",
    }

    CAMERA_INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_lines = [
        "# Pixel To Camera Point Sanity (head_left)",
        "",
        "## Status",
        f"- timestamp: `{result['timestamp']}`",
        f"- camera_name: `{CAMERA_NAME}`",
        f"- depth_unit: `{result['depth_unit']}`",
        f"- verified_for_pose_base: `{result['verified_for_pose_base']}`",
        "",
        "## Intrinsics Used",
        f"- fx: `{fx}`",
        f"- fy: `{fy}`",
        f"- cx: `{cx}`",
        f"- cy: `{cy}`",
        "",
        "## Sampled Pixels",
    ]
    for s in samples:
        report_lines.append(
            f"- {s['tag']}: uv=({s['u']},{s['v']}), depth={s['depth_value']}, valid={s['depth_valid']}, xyz={s['camera_point_xyz']}"
        )

    report_lines += [
        "",
        "## Sign Checks",
        f"- left_center_x_expected_negative: `{sign_checks['left_center_x_expected_negative']}`",
        f"- right_center_x_expected_positive: `{sign_checks['right_center_x_expected_positive']}`",
        f"- upper_center_y_expected_negative: `{sign_checks['upper_center_y_expected_negative']}`",
        f"- lower_center_y_expected_positive: `{sign_checks['lower_center_y_expected_positive']}`",
        "",
        "## Conclusion",
        "- This is an offline numeric sanity check only.",
        "- It does not validate depth unit for pose estimation.",
        "- It does not compute or validate `T_base_camera`.",
        "- Do not claim `pose_base` from this result.",
    ]
    OUT_REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

