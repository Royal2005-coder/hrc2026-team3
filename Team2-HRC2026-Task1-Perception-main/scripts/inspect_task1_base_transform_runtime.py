"""Inspect Task 1 robot base transform at runtime without Pinocchio/IK.

This script builds the Task 1 scene and robot, reads the USD world transform of
/Root/Ref_Xform/Ref directly, compares it with Part_Sorting.yaml robot root
pose, and writes base transform evidence for perception pose_base.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task1" / "perception"
CAMERA_INVENTORY_DIR = OUTPUT_ROOT / "camera_inventory"
REPORTS_DIR = OUTPUT_ROOT / "reports"
OUT_JSON = CAMERA_INVENTORY_DIR / "base_transform_runtime.json"
OUT_REPORT = REPORTS_DIR / "base_transform_runtime_report.md"

DEFAULT_BASELINE_PATH = Path("/workspace/GlobalHumanoidRobotChallenge_2026_Baseline")
DEFAULT_TASK1_CONFIG = DEFAULT_BASELINE_PATH / "Ubtech_sim" / "config" / "Part_Sorting.yaml"
ROBOT_BASE_PRIM_PATH = "/Root/Ref_Xform/Ref"


def parse_args() -> argparse.Namespace:
    cwd = Path.cwd()
    baseline_default = cwd if (cwd / "Ubtech_sim").exists() else DEFAULT_BASELINE_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-path", default=str(baseline_default))
    parser.add_argument("--config-path", default=str(DEFAULT_TASK1_CONFIG))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--settle-time", type=float, default=None)
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


def rot_z(deg: float) -> List[List[float]]:
    rad = math.radians(float(deg))
    c, s = math.cos(rad), math.sin(rad)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def config_transform(cfg: Dict[str, Any]) -> Dict[str, Any]:
    robot_cfg = cfg.get("robot", {})
    pos = [float(v) for v in robot_cfg.get("robot_position", [0.7, -0.2, 0.9])]
    rot = [float(v) for v in robot_cfg.get("robot_rotation", [0.0, 0.0, 90.0])]
    rz = rot_z(rot[2])
    mat = [
        [rz[0][0], rz[0][1], rz[0][2], pos[0]],
        [rz[1][0], rz[1][1], rz[1][2], pos[1]],
        [rz[2][0], rz[2][1], rz[2][2], pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    yaw_rad = math.radians(rot[2])
    quat_xyzw = [0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)]
    return {
        "position_m": pos,
        "rotation_deg_xyz": rot,
        "quaternion_xyzw": quat_xyzw,
        "T_world_base_matrix4x4": mat,
        "source": "Part_Sorting.yaml robot.robot_position + robot.robot_rotation",
    }


def compare_positions(a: Any, b: Any) -> Dict[str, Any]:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != 3 or len(b) != 3:
        return {"status": "unknown", "error": "position unavailable"}
    delta = [float(a[i]) - float(b[i]) for i in range(3)]
    norm = math.sqrt(sum(v * v for v in delta))
    return {"delta_runtime_minus_config_m": delta, "norm_m": norm, "status": "pass" if norm < 1e-5 else "review"}


def instantiate_scene_builder(scene_builder_cls: Any, cfg: dict, world: Any) -> Any:
    try:
        return scene_builder_cls(cfg, data_logger=None, world=world)
    except TypeError:
        return scene_builder_cls(cfg, data_logger=None)


def write_outputs(data: Dict[str, Any]) -> None:
    CAMERA_INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(make_json_safe(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Runtime Base Transform Report",
        "",
        "## Status",
        f"- timestamp: `{data.get('timestamp', 'unknown')}`",
        f"- runtime_success: `{data.get('failure_reason') is None}`",
        f"- failure_reason: `{data.get('failure_reason')}`",
        f"- base_prim_path: `{ROBOT_BASE_PRIM_PATH}`",
        f"- headless: `{data.get('headless')}`",
        "- Pinocchio/IK used: `no`",
        "",
        "## Comparison",
        f"- config_position_m: `{data.get('config_transform', {}).get('position_m', 'unknown')}`",
        f"- runtime_position_m: `{data.get('runtime_transform', {}).get('position_m', 'unknown')}`",
        f"- position_delta: `{data.get('position_comparison', {}).get('delta_runtime_minus_config_m', 'unknown')}`",
        f"- position_delta_norm_m: `{data.get('position_comparison', {}).get('norm_m', 'unknown')}`",
        f"- comparison_status: `{data.get('position_comparison', {}).get('status', 'unknown')}`",
        "",
        "## Interpretation",
        "",
        "- This verifies the robot root/base prim world transform by direct USD transform lookup.",
        "- It avoids `CoordinateTransform.from_torso_link()` and Pinocchio, which previously crashed in this environment.",
        "- If comparison_status is `pass`, the config-derived base transform used by the offline pose_base MVP matches the runtime USD base prim translation.",
        "- This still does not resolve full torso-link convention beyond `/Root/Ref_Xform/Ref`; use that exact frame name in outputs.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        "base_prim_path": ROBOT_BASE_PRIM_PATH,
        "config_transform": "unknown",
        "runtime_transform": "unknown",
        "position_comparison": "unknown",
        "failure_reason": None,
    }

    if not baseline_path.exists():
        data["failure_reason"] = f"baseline path does not exist: {baseline_path}"
        write_outputs(data)
        return 2
    if not config_path.exists():
        data["failure_reason"] = f"config path does not exist: {config_path}"
        write_outputs(data)
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
        from pxr import Usd, UsdGeom
        from source.config_loader import apply_scatter_config, load_config
        from source.RobotArticulation import RobotArticulation
        from source.SceneBuilder import SceneBuilder

        cfg = load_config(str(config_path))
        data["config_transform"] = config_transform(cfg)
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
        robot = RobotArticulation(prim_path=ROBOT_BASE_PRIM_PATH, name="walkerS2")
        robot.initialize()
        world.play()
        for _ in range(5):
            world.step(render=False)

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(ROBOT_BASE_PRIM_PATH)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"base prim invalid: {ROBOT_BASE_PRIM_PATH}")
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        matrix = cache.GetLocalToWorldTransform(prim)
        runtime_pose = extract_translation_quat_xyzw(matrix)
        runtime_pose["T_world_base_matrix4x4_raw_usd"] = matrix_to_list(matrix)
        runtime_pose["source"] = "UsdGeom.XformCache.GetLocalToWorldTransform(/Root/Ref_Xform/Ref)"
        data["runtime_transform"] = runtime_pose
        data["position_comparison"] = compare_positions(runtime_pose.get("position_m"), data["config_transform"].get("position_m"))
    except Exception as exc:
        data["failure_reason"] = str(exc)
        return_code = 1
    finally:
        write_outputs(data)
        try:
            kit.close()
        except Exception:
            pass
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
