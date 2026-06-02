"""Inspect Task 1 runtime camera geometry.

This script builds the Task 1 scene and robot, initializes the four baseline
runtime cameras, and logs geometry/intrinsic/extrinsic fields that Isaac exposes.
It does not compute or claim pose_base.

Default verification command only:

    python -m py_compile /home/ubuntu/Team2/Task1-Perception/scripts/inspect_task1_camera_geometry.py

Runtime command, only when explicitly requested:

    cd /workspace/GlobalHumanoidRobotChallenge_2026_Baseline
    /isaac-sim/python.sh /home/ubuntu/Team2/Task1-Perception/scripts/inspect_task1_camera_geometry.py \
      --config-path /workspace/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim/config/Part_Sorting.yaml \
      --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task1" / "perception"
CAMERA_INVENTORY_DIR = OUTPUT_ROOT / "camera_inventory"
REPORTS_DIR = OUTPUT_ROOT / "reports"
GEOMETRY_JSON_PATH = CAMERA_INVENTORY_DIR / "camera_geometry.json"
GEOMETRY_NOTE_PATH = REPORTS_DIR / "camera_geometry_note.md"

DEFAULT_BASELINE_PATH = Path("/workspace/GlobalHumanoidRobotChallenge_2026_Baseline")
DEFAULT_TASK1_CONFIG = DEFAULT_BASELINE_PATH / "Ubtech_sim" / "config" / "Part_Sorting.yaml"
CAMERA_NAMES = ("head_left", "head_right", "wrist_left", "wrist_right")


def parse_args() -> argparse.Namespace:
    cwd = Path.cwd()
    baseline_default = cwd if (cwd / "Ubtech_sim").exists() else DEFAULT_BASELINE_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-path", default=str(baseline_default))
    parser.add_argument("--config-path", default=str(DEFAULT_TASK1_CONFIG))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--settle-time", type=float, default=None)
    parser.add_argument("--render-steps", type=int, default=20)
    parser.add_argument("--camera-width", type=int, default=512)
    parser.add_argument("--camera-height", type=int, default=512)
    return parser.parse_args()


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return make_json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    return str(value)


def value_or_unknown(value: Any) -> Any:
    return "unknown" if value is None else make_json_safe(value)


def try_call(obj: Any, method_name: str, *args: Any) -> Dict[str, Any]:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return {"available": False, "value": "unknown", "error": "method_not_found"}
    try:
        return {"available": True, "value": make_json_safe(method(*args)), "error": None}
    except Exception as exc:
        return {"available": True, "value": "unknown", "error": str(exc)}


def read_usd_attr(prim: Any, attr_name: str) -> Dict[str, Any]:
    attr = prim.GetAttribute(attr_name)
    if not attr or not attr.IsValid():
        return {"available": False, "value": "unknown"}
    try:
        return {"available": True, "value": make_json_safe(attr.Get())}
    except Exception as exc:
        return {"available": True, "value": "unknown", "error": str(exc)}


def matrix_to_list(matrix: Any) -> Any:
    try:
        return [[float(matrix[i][j]) for j in range(4)] for i in range(4)]
    except Exception:
        return "unknown"


def extract_translation_quat_xyzw(matrix: Any) -> Dict[str, Any]:
    try:
        t = matrix.ExtractTranslation()
        q = matrix.ExtractRotationQuat()
        qi = q.GetImaginary()
        return {
            "position_m": [float(t[0]), float(t[1]), float(t[2])],
            "quaternion_xyzw": [float(qi[0]), float(qi[1]), float(qi[2]), float(q.GetReal())],
        }
    except Exception as exc:
        return {"position_m": "unknown", "quaternion_xyzw": "unknown", "error": str(exc)}


def compute_intrinsic_from_usd(width: Any, height: Any, focal_length: Any, horizontal_aperture: Any, vertical_aperture: Any) -> Dict[str, Any]:
    try:
        w = float(width)
        h = float(height)
        fl = float(focal_length)
        ha = float(horizontal_aperture)
        va = float(vertical_aperture)
        if w <= 0 or h <= 0 or fl <= 0 or ha <= 0 or va <= 0:
            raise ValueError("non_positive_camera_parameter")
        fx = fl / ha * w
        fy = fl / va * h
        cx = w / 2.0
        cy = h / 2.0
        return {
            "status": "computed_from_usd_focal_aperture_assuming_center_principal_point",
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "matrix_3x3": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            "warning": "Computed intrinsics need runtime/API validation before use for pose_base.",
        }
    except Exception as exc:
        return {"status": "unknown", "matrix_3x3": "unknown", "error": str(exc)}


def instantiate_scene_builder(scene_builder_cls: Any, cfg: dict, world: Any) -> Any:
    try:
        return scene_builder_cls(cfg, data_logger=None, world=world)
    except TypeError:
        return scene_builder_cls(cfg, data_logger=None)


def set_camera_resolution(robot: Any, width: int, height: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "requested_resolution": [width, height],
        "cameras": {},
    }
    if not hasattr(robot, "cameras") or not isinstance(robot.cameras, dict):
        result["failure_reason"] = "robot.cameras unavailable"
        return result
    for camera_name in CAMERA_NAMES:
        camera_result: Dict[str, Any] = {
            "before": "unknown",
            "after": "unknown",
            "set_resolution_available": False,
            "failure_reason": None,
        }
        camera = robot.cameras.get(camera_name)
        if camera is None:
            camera_result["failure_reason"] = "camera not found"
            result["cameras"][camera_name] = camera_result
            continue
        try:
            if hasattr(camera, "get_resolution"):
                camera_result["before"] = list(camera.get_resolution())
        except Exception as exc:
            camera_result["before"] = f"unknown: {exc}"
        if not hasattr(camera, "set_resolution"):
            camera_result["failure_reason"] = "set_resolution not available"
            result["cameras"][camera_name] = camera_result
            continue
        camera_result["set_resolution_available"] = True
        try:
            camera.set_resolution((int(width), int(height)))
            if hasattr(camera, "get_resolution"):
                camera_result["after"] = list(camera.get_resolution())
        except Exception as exc:
            camera_result["failure_reason"] = str(exc)
        result["cameras"][camera_name] = camera_result
    return result


def inspect_camera(camera_name: str, camera: Any, stage: Any) -> Dict[str, Any]:
    from pxr import Usd, UsdGeom

    prim_path = str(getattr(camera, "prim_path", "unknown"))
    prim = stage.GetPrimAtPath(prim_path) if prim_path != "unknown" else None

    entry: Dict[str, Any] = {
        "camera_name": camera_name,
        "prim_path": prim_path,
        "prim_valid": bool(prim and prim.IsValid()),
        "resolution": "unknown",
        "depth_unit": "unknown",
        "rgb_depth_alignment": "unknown",
        "T_base_camera": "unknown",
        "pose_base_claimed": False,
        "camera_api_methods": {},
        "usd_attributes": {},
        "camera_world_pose": "unknown",
        "intrinsics": "unknown",
        "notes": [],
    }

    for method_name in (
        "get_resolution",
        "get_intrinsics_matrix",
        "get_projection_matrix",
        "get_focal_length",
        "get_horizontal_aperture",
        "get_vertical_aperture",
        "get_clipping_range",
        "get_world_pose",
        "get_local_pose",
    ):
        entry["camera_api_methods"][method_name] = try_call(camera, method_name)

    resolution_value = entry["camera_api_methods"].get("get_resolution", {}).get("value")
    if isinstance(resolution_value, list) and len(resolution_value) >= 2:
        entry["resolution"] = resolution_value

    if prim and prim.IsValid():
        attr_names = (
            "focalLength",
            "horizontalAperture",
            "verticalAperture",
            "horizontalApertureOffset",
            "verticalApertureOffset",
            "clippingRange",
            "projection",
            "focusDistance",
            "fStop",
        )
        for attr_name in attr_names:
            entry["usd_attributes"][attr_name] = read_usd_attr(prim, attr_name)

        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        world_matrix = xform_cache.GetLocalToWorldTransform(prim)
        entry["camera_world_pose"] = {
            "T_world_camera_matrix4x4": matrix_to_list(world_matrix),
            **extract_translation_quat_xyzw(world_matrix),
            "source": "UsdGeom.XformCache.GetLocalToWorldTransform(camera_prim)",
        }
    else:
        entry["notes"].append("USD camera prim is invalid; cannot read USD attributes or world pose.")

    intrinsics_api = entry["camera_api_methods"].get("get_intrinsics_matrix", {})
    if intrinsics_api.get("available") and intrinsics_api.get("value") != "unknown":
        entry["intrinsics"] = {
            "status": "from_camera_api_get_intrinsics_matrix",
            "matrix_3x3": intrinsics_api["value"],
            "warning": "Still validate depth unit and T_base_camera before pose_base.",
        }
    else:
        width = height = None
        if isinstance(entry["resolution"], list) and len(entry["resolution"]) >= 2:
            width, height = entry["resolution"][0], entry["resolution"][1]
        focal = entry["usd_attributes"].get("focalLength", {}).get("value")
        ha = entry["usd_attributes"].get("horizontalAperture", {}).get("value")
        va = entry["usd_attributes"].get("verticalAperture", {}).get("value")
        entry["intrinsics"] = compute_intrinsic_from_usd(width, height, focal, ha, va)

    if entry["depth_unit"] == "unknown":
        entry["notes"].append("Depth unit was not verified by geometry inspection.")
    if entry["T_base_camera"] == "unknown":
        entry["notes"].append("T_base_camera is not derived here; only camera world pose is logged if available.")
    return entry


def write_json(data: Dict[str, Any]) -> None:
    CAMERA_INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    with GEOMETRY_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(data), f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_note(data: Dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Camera Geometry Note",
        "",
        "## Status",
        "",
        f"- timestamp: `{data.get('timestamp', 'unknown')}`",
        f"- baseline_path: `{data.get('baseline_path', 'unknown')}`",
        f"- config_path: `{data.get('config_path', 'unknown')}`",
        f"- scene_path: `{data.get('scene_path', 'unknown')}`",
        f"- runtime_success: `{data.get('failure_reason') is None}`",
        f"- failure_reason: `{data.get('failure_reason')}`",
        "",
        "## Camera Summary",
        "",
        "| camera_name | prim_valid | resolution | intrinsics_status | world_pose | depth_unit | T_base_camera |",
        "|---|---:|---|---|---|---|---|",
    ]
    for cam in data.get("cameras", []):
        intr = cam.get("intrinsics", {})
        intr_status = intr.get("status", "unknown") if isinstance(intr, dict) else "unknown"
        world_pose = "available" if isinstance(cam.get("camera_world_pose"), dict) else "unknown"
        lines.append(
            f"| `{cam.get('camera_name')}` | {cam.get('prim_valid')} | `{cam.get('resolution')}` | `{intr_status}` | `{world_pose}` | `{cam.get('depth_unit')}` | `{cam.get('T_base_camera')}` |"
        )
    lines.extend([
        "",
        "## Important Notes",
        "",
        "- This report does not claim `pose_base`.",
        "- `T_world_camera` may be available, but `T_base_camera` remains unknown until robot/base frame convention is verified.",
        "- Intrinsics computed from USD focal length/aperture are provisional unless confirmed by Isaac camera API or a projection sanity test.",
        "- Depth unit remains unknown until validated separately.",
        "",
        "## Next Step",
        "",
        "- Review `camera_geometry.json` fields for `head_left` first.",
        "- If intrinsics and camera world pose look usable, plan a transform sanity step to derive/verify `T_base_camera`.",
        "- Do not compute pixel-to-3D or `pose_base` until depth unit and `T_base_camera` are verified.",
    ])
    GEOMETRY_NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    sys.argv = [sys.argv[0]]

    baseline_path = Path(args.baseline_path).resolve()
    config_path = Path(args.config_path).resolve()
    data: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_path": str(baseline_path),
        "config_path": str(config_path),
        "headless": bool(args.headless),
        "requested_camera_resolution": [int(args.camera_width), int(args.camera_height)],
        "render_steps": int(args.render_steps),
        "resolution_config": None,
        "scene_path": "unknown",
        "cameras": [],
        "failure_reason": None,
    }

    if not baseline_path.exists():
        data["failure_reason"] = f"baseline path does not exist: {baseline_path}"
        write_json(data)
        write_note(data)
        return 2
    if not config_path.exists():
        data["failure_reason"] = f"config path does not exist: {config_path}"
        write_json(data)
        write_note(data)
        return 2

    os.chdir(str(baseline_path))
    sys.path.insert(0, str(baseline_path / "Ubtech_sim"))

    from isaacsim import SimulationApp

    kit = SimulationApp({"width": 1280, "height": 720, "headless": bool(args.headless)})
    return_code = 0
    try:
        from isaacsim.core.api import World
        import omni
        import omni.replicator.core as rep
        from source.config_loader import apply_scatter_config, load_config
        from source.RobotArticulation import RobotArticulation
        from source.SceneBuilder import SceneBuilder

        cfg = load_config(str(config_path))
        grasp_cfg = cfg.get("grasp", {})
        scene_path = os.path.join(cfg["root_path"], cfg["scene_usd"])
        data["scene_path"] = scene_path

        omni.usd.get_context().open_stage(scene_path)
        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 20.0)
        world.initialize_physics()

        scene = instantiate_scene_builder(SceneBuilder, cfg, world)
        apply_scatter_config(cfg)
        scene.build_all()
        rep.orchestrator.step()

        world.play()
        settle_time = args.settle_time if args.settle_time is not None else grasp_cfg.get("settle_time", 2.0)
        for _ in range(int(float(settle_time) / world.get_physics_dt())):
            world.step(render=False)

        world.pause()
        scene.build_robot()
        robot = RobotArticulation(prim_path="/Root/Ref_Xform/Ref", name="walkerS2")
        robot.initialize()
        data["resolution_config"] = set_camera_resolution(robot, int(args.camera_width), int(args.camera_height))
        world.play()
        for _ in range(max(0, int(args.render_steps))):
            world.step(render=True)

        stage = omni.usd.get_context().get_stage()
        for camera_name in CAMERA_NAMES:
            camera = robot.cameras.get(camera_name)
            if camera is None:
                data["cameras"].append({"camera_name": camera_name, "failure_reason": "camera_not_found"})
                continue
            data["cameras"].append(inspect_camera(camera_name, camera, stage))

    except Exception as exc:
        data["failure_reason"] = str(exc)
        return_code = 1
    finally:
        write_json(data)
        write_note(data)
        try:
            kit.close()
        except Exception:
            pass
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
