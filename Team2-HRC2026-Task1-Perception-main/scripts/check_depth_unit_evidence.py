"""Collect evidence for Task 1 depth unit semantics without running Isaac Sim.

This script reads local Isaac/baseline/script source files and writes a report
about what is known versus still unverified. It does not modify baseline files
and does not claim pose_base.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = Path("/workspace/GlobalHumanoidRobotChallenge_2026_Baseline")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task1" / "perception"
CAMERA_INVENTORY_DIR = OUTPUT_ROOT / "camera_inventory"
REPORTS_DIR = OUTPUT_ROOT / "reports"

ISAAC_CAMERA_SOURCE = Path("/isaac-sim/exts/isaacsim.sensors.camera/isaacsim/sensors/camera/camera.py")
BASELINE_MAIN = BASELINE_ROOT / "Ubtech_sim" / "main.py"
CAPTURE_SCRIPT = PROJECT_ROOT / "scripts" / "capture_task1_rgbd_once.py"
GEOMETRY_SCRIPT = PROJECT_ROOT / "scripts" / "inspect_task1_camera_geometry.py"
PREVIEW_SCRIPT = PROJECT_ROOT / "scripts" / "preview_task1_cameras_gui.py"
CAMERA_GEOMETRY_JSON = CAMERA_INVENTORY_DIR / "camera_geometry.json"

OUTPUT_JSON = CAMERA_INVENTORY_DIR / "depth_unit_evidence.json"
OUTPUT_REPORT = REPORTS_DIR / "depth_unit_evidence_note.md"


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def find_lines(path: Path, patterns: List[str]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    if not path.exists():
        return [{"path": str(path), "line": None, "text": "missing_file"}]
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        for pattern in patterns:
            if pattern in line:
                hits.append({"path": str(path), "line": lineno, "pattern": pattern, "text": line.strip()})
    return hits


def build_result() -> Dict[str, Any]:
    source_hits = {
        "isaac_camera_depth": find_lines(
            ISAAC_CAMERA_SOURCE,
            [
                "Gets the depth data from the camera sensor as distance to image plane",
                "distance_to_image_plane",
                "in stage units",
            ],
        ),
        "world_stage_units": find_lines(
            BASELINE_MAIN,
            [
                "World(",
                "stage_units_in_meters=1.0",
            ],
        )
        + find_lines(CAPTURE_SCRIPT, ["World(", "stage_units_in_meters=1.0"])
        + find_lines(GEOMETRY_SCRIPT, ["World(", "stage_units_in_meters=1.0"])
        + find_lines(PREVIEW_SCRIPT, ["World(", "stage_units_in_meters=1.0"]),
    }

    geometry_depth_unit = "unknown"
    if CAMERA_GEOMETRY_JSON.exists():
        data = json.loads(CAMERA_GEOMETRY_JSON.read_text(encoding="utf-8"))
        for camera in data.get("cameras", []):
            if camera.get("camera_name") == "head_left":
                geometry_depth_unit = camera.get("depth_unit", "unknown")
                break

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "camera_name": "head_left",
        "depth_api_semantics": "distance_to_image_plane",
        "source_evidence": source_hits,
        "stage_units_in_meters_configured": True,
        "stage_units_in_meters_value": 1.0,
        "working_depth_unit_assumption": "meters",
        "verified_for_pose_base": False,
        "camera_geometry_depth_unit_field": geometry_depth_unit,
        "conclusion": (
            "Local source evidence supports treating depth values as distance_to_image_plane in stage units, "
            "and the baseline/perception scripts configure stage_units_in_meters=1.0. For planning, the working "
            "assumption can be meters, but this has not been validated by a controlled geometry/raycast test, so "
            "pose_base remains unclaimed."
        ),
        "remaining_unknowns": {
            "controlled_depth_unit_validation": "not_done",
            "rgb_depth_alignment": "unknown",
            "T_base_camera": "unknown",
            "pose_base": "not_claimed",
        },
    }


def write_report(result: Dict[str, Any]) -> None:
    depth_hits = result["source_evidence"]["isaac_camera_depth"]
    stage_hits = result["source_evidence"]["world_stage_units"]

    lines = [
        "# Depth Unit Evidence Note",
        "",
        "## Status",
        "",
        f"- timestamp: `{result['timestamp']}`",
        f"- camera_name: `{result['camera_name']}`",
        f"- depth_api_semantics: `{result['depth_api_semantics']}`",
        f"- stage_units_in_meters_configured: `{result['stage_units_in_meters_configured']}`",
        f"- stage_units_in_meters_value: `{result['stage_units_in_meters_value']}`",
        f"- working_depth_unit_assumption: `{result['working_depth_unit_assumption']}`",
        f"- verified_for_pose_base: `{result['verified_for_pose_base']}`",
        f"- camera_geometry_depth_unit_field: `{result['camera_geometry_depth_unit_field']}`",
        "",
        "## Source Evidence",
        "",
        "### Isaac Camera Source",
        "",
        "| path | line | evidence |",
        "|---|---:|---|",
    ]
    for hit in depth_hits:
        lines.append(f"| `{hit['path']}` | {hit['line']} | `{hit['text']}` |")
    lines.extend(
        [
            "",
            "### World Stage Units",
            "",
            "| path | line | evidence |",
            "|---|---:|---|",
        ]
    )
    for hit in stage_hits:
        lines.append(f"| `{hit['path']}` | {hit['line']} | `{hit['text']}` |")

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            result["conclusion"],
            "",
            "## Important Constraint",
            "",
            "- Keep `depth_unit` as `unknown` in geometry/config outputs until a controlled validation step is done.",
            "- Do not compute or claim `pose_base` from this evidence alone.",
            "- Next safe step: verify RGB-depth alignment and design a controlled depth-unit validation using known scene geometry or a raycast/intersection check.",
        ]
    )
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
