"""Validate the final Task 1 perception handoff package.

This script is intentionally offline: it does not start Isaac Sim.
It checks that the current final semantic interface is internally
consistent and has the artifacts needed by planner/motion.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task1" / "perception"
JSON_DIR = OUTPUT_ROOT / "json"
REPORTS_DIR = OUTPUT_ROOT / "reports"
SAMPLES_DIR = OUTPUT_ROOT / "samples"
OVERLAYS_DIR = OUTPUT_ROOT / "overlays"
LOGS_DIR = OUTPUT_ROOT / "logs"

INTERFACE_PATH = JSON_DIR / "perception_interface_semantic_pose_base.json"
VALIDATION_JSON = JSON_DIR / "final_perception_validation.json"
VALIDATION_REPORT = REPORTS_DIR / "final_perception_validation_report.md"


REQUIRED_ARTIFACTS = [
    INTERFACE_PATH,
    JSON_DIR / "semantic_bboxes_head_left.json",
    LOGS_DIR / "semantic_pose_pipeline_log.json",
    OVERLAYS_DIR / "overlay_semantic_pose_base_head_left.png",
    OVERLAYS_DIR / "overlay_semantic_bboxes_head_left.png",
    SAMPLES_DIR / "semantic_pose_rgb_head_left.png",
    SAMPLES_DIR / "semantic_pose_depth_head_left.npy",
    SAMPLES_DIR / "semantic_pose_depth_vis_head_left.png",
    OUTPUT_ROOT / "camera_inventory" / "camera_config_sheet.csv",
    OUTPUT_ROOT / "camera_inventory" / "camera_geometry.json",
    OUTPUT_ROOT / "camera_inventory" / "base_transform_runtime.json",
    REPORTS_DIR / "task1_perception_planner_handoff_report.md",
    REPORTS_DIR / "semantic_pose_base_pipeline_report.md",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_object(obj: Dict[str, Any], idx: int) -> List[str]:
    errors: List[str] = []
    class_id = obj.get("class_id")
    if class_id not in {"part_a", "part_b"}:
        errors.append(f"object {idx}: class_id must be part_a/part_b, got {class_id!r}")

    bbox = obj.get("bbox_xyxy")
    if not (isinstance(bbox, list) and len(bbox) == 4 and all(is_number(v) for v in bbox)):
        errors.append(f"object {idx}: invalid bbox_xyxy")
    elif not (bbox[2] > bbox[0] and bbox[3] > bbox[1]):
        errors.append(f"object {idx}: bbox_xyxy has non-positive area")

    centroid = obj.get("centroid_px")
    if not (isinstance(centroid, list) and len(centroid) == 2 and all(is_number(v) for v in centroid)):
        errors.append(f"object {idx}: invalid centroid_px")

    depth = obj.get("depth_median_m")
    if not (is_number(depth) and depth > 0):
        errors.append(f"object {idx}: invalid depth_median_m")

    pose = obj.get("pose_base", {})
    if pose.get("frame") != "/Root/Ref_Xform/Ref":
        errors.append(f"object {idx}: pose_base.frame must be /Root/Ref_Xform/Ref")
    pos = pose.get("position_m")
    if not (isinstance(pos, list) and len(pos) == 3 and all(is_number(v) for v in pos)):
        errors.append(f"object {idx}: invalid pose_base.position_m")
    quat = pose.get("orientation_xyzw")
    if not (isinstance(quat, list) and len(quat) == 4 and all(is_number(v) for v in quat)):
        errors.append(f"object {idx}: invalid pose_base.orientation_xyzw")

    yaw = obj.get("grasp_hint", {}).get("yaw_rad")
    if not is_number(yaw):
        errors.append(f"object {idx}: invalid grasp_hint.yaw_rad")

    return errors


def main() -> int:
    result: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "unknown",
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    missing = [str(path) for path in REQUIRED_ARTIFACTS if not path.exists()]
    result["checks"]["required_artifacts_missing"] = missing
    if missing:
        result["errors"].append("required artifacts are missing")

    if not INTERFACE_PATH.exists():
        result["status"] = "fail"
        write_outputs(result)
        return 1

    data = load_json(INTERFACE_PATH)
    objects = data.get("objects", [])
    part_a = sum(1 for obj in objects if obj.get("class_id") == "part_a")
    part_b = sum(1 for obj in objects if obj.get("class_id") == "part_b")
    result["checks"].update(
        {
            "interface_path": str(INTERFACE_PATH),
            "pipeline": data.get("pipeline"),
            "camera_name": data.get("camera_name"),
            "base_frame": data.get("base_frame"),
            "depth_unit": data.get("depth_unit"),
            "object_count": len(objects),
            "part_a_count": part_a,
            "part_b_count": part_b,
            "four_object_pass": data.get("four_object_pass"),
            "pose_base_available": data.get("pose_base_available"),
            "orientation_available": data.get("orientation_available"),
            "yaw_available": data.get("yaw_available"),
            "seed": data.get("reproducibility", {}).get("seed"),
        }
    )

    if data.get("camera_name") != "head_left":
        result["errors"].append("camera_name is not head_left")
    if data.get("base_frame") != "/Root/Ref_Xform/Ref":
        result["errors"].append("base_frame is not /Root/Ref_Xform/Ref")
    if data.get("depth_unit") != "meters_runtime_distance_to_image_plane_stage_units":
        result["warnings"].append("depth_unit is not the expected runtime meters value")
    if len(objects) != 4 or part_a != 2 or part_b != 2:
        result["errors"].append("expected exactly 4 objects: 2 part_a and 2 part_b")
    if data.get("four_object_pass") is not True:
        result["errors"].append("four_object_pass is not true")
    if data.get("pose_base_available") is not True:
        result["errors"].append("pose_base_available is not true")
    if data.get("orientation_available") is not True:
        result["errors"].append("orientation_available is not true")
    if data.get("yaw_available") is not True:
        result["errors"].append("yaw_available is not true")

    for idx, obj in enumerate(objects):
        result["errors"].extend(validate_object(obj, idx))

    result["status"] = "pass" if not result["errors"] else "fail"
    write_outputs(result)
    return 0 if result["status"] == "pass" else 1


def write_outputs(result: Dict[str, Any]) -> None:
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    checks = result.get("checks", {})
    lines = [
        "# Báo Cáo Kiểm Tra Final Perception",
        "",
        "## Trạng thái",
        f"- `status`: `{result.get('status')}`",
        f"- `timestamp`: `{result.get('timestamp')}`",
        f"- `interface`: `{checks.get('interface_path')}`",
        f"- `camera`: `{checks.get('camera_name')}`",
        f"- `base_frame`: `{checks.get('base_frame')}`",
        f"- `object_count`: `{checks.get('object_count')}`",
        f"- `part_a_count`: `{checks.get('part_a_count')}`",
        f"- `part_b_count`: `{checks.get('part_b_count')}`",
        f"- `pose_base_available`: `{checks.get('pose_base_available')}`",
        f"- `orientation_available`: `{checks.get('orientation_available')}`",
        f"- `yaw_available`: `{checks.get('yaw_available')}`",
        f"- `seed`: `{checks.get('seed')}`",
        "",
        "## Lỗi",
    ]
    errors = result.get("errors", [])
    lines.extend([f"- {err}" for err in errors] if errors else ["- không có"])
    lines += ["", "## Cảnh báo"]
    warnings = result.get("warnings", [])
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- không có"])
    lines += [
        "",
        "## Ghi chú kỹ thuật",
        "- Đây là bước kiểm tra offline, không khởi động Isaac Sim.",
        "- Mục tiêu của validator là kiểm tra tính nhất quán của package final trước khi bàn giao cho Planner, Motion và Evaluation.",
        "- `orientation_xyzw` được kiểm tra theo quy ước `xyzw`.",
        "- `grasp_hint.yaw_rad` được kiểm tra theo đơn vị `radian`.",
        "- Quy ước `yaw` hiện tại là trục local `+X` của object sau khi chiếu xuống mặt phẳng `XY` của base frame.",
        "- Reproducibility ở mức pixel tuyệt đối giữa các máy vẫn phụ thuộc vào Isaac Sim, GPU, driver và timing của physics/render.",
    ]
    VALIDATION_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
