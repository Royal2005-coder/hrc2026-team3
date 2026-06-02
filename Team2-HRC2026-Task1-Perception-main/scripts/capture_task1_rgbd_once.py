"""Capture one RGB-D sample from the four Task 1 runtime cameras.

Run from the baseline repository:

    /isaac-sim/python.sh /home/ubuntu/Team2/Task1-Perception/scripts/capture_task1_rgbd_once.py

This script intentionally does not modify baseline files. It mirrors the
minimum scene/robot setup from Ubtech_sim/main.py, captures once, writes sample
files under the Task1-Perception project, then exits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task1" / "perception"
SAMPLES_DIR = OUTPUT_ROOT / "samples"
LOG_PATH = OUTPUT_ROOT / "logs" / "frame_capture_log.json"

DEFAULT_BASELINE_PATH = Path("/workspace/GlobalHumanoidRobotChallenge_2026_Baseline")
CAMERA_NAMES = ("head_left", "head_right", "wrist_left", "wrist_right")


def parse_args() -> argparse.Namespace:
    cwd = Path.cwd()
    baseline_default = cwd if (cwd / "Ubtech_sim" / "config" / "task1.yaml").exists() else DEFAULT_BASELINE_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-path",
        default=str(baseline_default),
        help="Path to GlobalHumanoidRobotChallenge_2026_Baseline.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Task config path. Defaults to <baseline-path>/Ubtech_sim/config/task1.yaml.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Launch Isaac Sim in headless mode. Default keeps baseline behavior: headless=False.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=None,
        help="Optional override for physics settle time before capture.",
    )
    parser.add_argument(
        "--render-steps",
        type=int,
        default=20,
        help="Number of rendered world steps after camera initialization before capture.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=512,
        help="Requested runtime camera render width for RGB-D capture.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=512,
        help="Requested runtime camera render height for RGB-D capture.",
    )
    return parser.parse_args()


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    return str(value)


def as_array(data: Any) -> Optional["np.ndarray"]:
    if data is None:
        return None
    try:
        return np.asarray(data)
    except Exception:
        return None


def normalize_rgb(rgb: "np.ndarray") -> "np.ndarray":
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        finite = np.isfinite(arr)
        if finite.any() and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def depth_to_vis(depth: "np.ndarray") -> Optional["np.ndarray"]:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        return None

    valid = np.isfinite(arr) & (arr > 0)
    if not valid.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    lo = float(np.nanpercentile(arr[valid], 2.0))
    hi = float(np.nanpercentile(arr[valid], 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(arr[valid]))
        hi = float(np.nanmax(arr[valid]))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    norm = (arr - lo) / (hi - lo)
    norm = np.where(valid, norm, 0.0)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


def save_png(array: "np.ndarray", path: Path) -> None:
    from PIL import Image

    Image.fromarray(array).save(path)


def summarize_depth(depth: Optional["np.ndarray"]) -> Dict[str, Any]:
    if depth is None:
        return {
            "depth_shape": None,
            "depth_dtype": None,
            "depth_min": None,
            "depth_median": None,
            "depth_max": None,
            "depth_valid_ratio": None,
        }

    arr = np.asarray(depth)
    valid = np.isfinite(arr) & (arr > 0)
    if valid.any():
        valid_values = arr[valid]
        return {
            "depth_shape": list(arr.shape),
            "depth_dtype": str(arr.dtype),
            "depth_min": float(np.nanmin(valid_values)),
            "depth_median": float(np.nanmedian(valid_values)),
            "depth_max": float(np.nanmax(valid_values)),
            "depth_valid_ratio": float(valid.mean()),
        }

    return {
        "depth_shape": list(arr.shape),
        "depth_dtype": str(arr.dtype),
        "depth_min": None,
        "depth_median": None,
        "depth_max": None,
        "depth_valid_ratio": 0.0,
    }


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


def capture_camera(camera_name: str, robot: Any) -> Dict[str, Any]:
    rgb_path = SAMPLES_DIR / f"sample_rgb_{camera_name}.png"
    depth_path = SAMPLES_DIR / f"sample_depth_{camera_name}.npy"
    depth_vis_path = SAMPLES_DIR / f"sample_depth_vis_{camera_name}.png"

    entry: Dict[str, Any] = {
        "camera_name": camera_name,
        "rgb_available": False,
        "depth_available": False,
        "rgb_shape": None,
        "depth_shape": None,
        "rgb_dtype": None,
        "depth_dtype": None,
        "depth_min": None,
        "depth_median": None,
        "depth_max": None,
        "depth_valid_ratio": None,
        "rgb_path": None,
        "depth_path": None,
        "depth_vis_path": None,
        "failure_reason": None,
    }

    try:
        rgbd = robot.get_camera_rgbd(camera_name)
    except Exception as exc:
        entry["failure_reason"] = f"get_camera_rgbd failed: {exc}"
        return entry

    rgb = as_array(rgbd.get("rgb") if isinstance(rgbd, dict) else None)
    depth = as_array(rgbd.get("depth") if isinstance(rgbd, dict) else None)

    failures = []
    if rgb is not None:
        entry["rgb_available"] = True
        entry["rgb_shape"] = list(rgb.shape)
        entry["rgb_dtype"] = str(rgb.dtype)
        try:
            save_png(normalize_rgb(rgb), rgb_path)
            entry["rgb_path"] = str(rgb_path)
        except Exception as exc:
            failures.append(f"rgb save failed: {exc}")

    if depth is not None:
        entry["depth_available"] = True
        np.save(depth_path, depth)
        entry["depth_path"] = str(depth_path)
        entry.update(summarize_depth(depth))
        try:
            depth_vis = depth_to_vis(depth)
            if depth_vis is not None:
                save_png(depth_vis, depth_vis_path)
                entry["depth_vis_path"] = str(depth_vis_path)
        except Exception as exc:
            failures.append(f"depth vis save failed: {exc}")
    else:
        entry.update(summarize_depth(None))

    if not entry["rgb_available"] and not entry["depth_available"]:
        failures.append("no rgb or depth returned")
    entry["failure_reason"] = "; ".join(failures) if failures else None
    return entry


def write_log(log: Dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(log), f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    args = parse_args()
    # Prevent custom script args from being forwarded into Kit/SimulationApp.
    sys.argv = [sys.argv[0]]
    baseline_path = Path(args.baseline_path).resolve()
    config_path = Path(args.config_path).resolve() if args.config_path else baseline_path / "Ubtech_sim" / "config" / "task1.yaml"

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    log: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_path": str(baseline_path),
        "config_path": str(config_path),
        "headless": bool(args.headless),
        "requested_camera_resolution": [int(args.camera_width), int(args.camera_height)],
        "render_steps": int(args.render_steps),
        "resolution_config": None,
        "cameras": [],
        "failure_reason": None,
    }

    if not baseline_path.exists():
        log["failure_reason"] = f"baseline path does not exist: {baseline_path}"
        write_log(log)
        return 2
    if not config_path.exists():
        log["failure_reason"] = f"config path does not exist: {config_path}"
        write_log(log)
        return 2

    os.chdir(str(baseline_path))
    sys.path.insert(0, str(baseline_path / "Ubtech_sim"))

    from isaacsim import SimulationApp

    kit = SimulationApp(
        launch_config={
            "width": 1280,
            "height": 720,
            "headless": bool(args.headless),
        }
    )

    try:
        global np
        import numpy as np

        from isaacsim.core.api import World
        import omni
        import omni.replicator.core as rep

        from source.config_loader import apply_scatter_config, load_config
        from source.RobotArticulation import RobotArticulation
        from source.SceneBuilder import SceneBuilder

        cfg = load_config(str(config_path))
        grasp_cfg = cfg.get("grasp", {})

        omni.usd.get_context().open_stage(os.path.join(cfg["root_path"], cfg["scene_usd"]))
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 20.0,
        )
        world.initialize_physics()

        scene = instantiate_scene_builder(SceneBuilder, cfg, world)
        apply_scatter_config(cfg)
        scene.build_all()
        rep.orchestrator.step()

        world.play()
        settle_time = args.settle_time if args.settle_time is not None else grasp_cfg.get("settle_time", 2.0)
        settle_steps = int(float(settle_time) / world.get_physics_dt())
        for _ in range(settle_steps):
            world.step(render=False)

        world.pause()
        scene.build_robot()
        robot = RobotArticulation(prim_path="/Root/Ref_Xform/Ref", name="walkerS2")
        robot.initialize()
        log["resolution_config"] = set_camera_resolution(robot, int(args.camera_width), int(args.camera_height))
        world.play()

        for _ in range(max(0, int(args.render_steps))):
            world.step(render=True)

        for camera_name in CAMERA_NAMES:
            log["cameras"].append(capture_camera(camera_name, robot))

    except Exception as exc:
        log["failure_reason"] = str(exc)
        return_code = 1
    else:
        return_code = 0
    finally:
        write_log(log)
        try:
            kit.close()
        except Exception:
            pass

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
