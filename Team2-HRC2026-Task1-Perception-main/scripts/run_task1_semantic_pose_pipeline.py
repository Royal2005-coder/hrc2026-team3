"""Run Task 1 semantic RGB-D perception and export pose_base candidates.

This is the current final perception pipeline candidate:
- build the Task 1 scene and robot without modifying baseline files,
- set deterministic seeds as far as Isaac/Replicator exposes them,
- capture head_left RGB-D and semantic 2D boxes from the same runtime scene,
- compute one centroid and pose_base candidate per semantic object,
- write outputs for planner/motion integration and reproducibility review.

The output frame is the robot root prim `/Root/Ref_Xform/Ref`.
Object orientation quaternion is exported in `xyzw` convention.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task1" / "perception"
JSON_DIR = OUTPUT_ROOT / "json"
REPORTS_DIR = OUTPUT_ROOT / "reports"
OVERLAYS_DIR = OUTPUT_ROOT / "overlays"
SAMPLES_DIR = OUTPUT_ROOT / "samples"
LOGS_DIR = OUTPUT_ROOT / "logs"

OUT_JSON = JSON_DIR / "perception_interface_semantic_pose_base.json"
OUT_REPORT = REPORTS_DIR / "semantic_pose_base_pipeline_report.md"
OUT_LOG = LOGS_DIR / "semantic_pose_pipeline_log.json"
OUT_OVERLAY = OVERLAYS_DIR / "overlay_semantic_pose_base_head_left.png"
OUT_RGB = SAMPLES_DIR / "semantic_pose_rgb_head_left.png"
OUT_DEPTH = SAMPLES_DIR / "semantic_pose_depth_head_left.npy"
OUT_DEPTH_VIS = SAMPLES_DIR / "semantic_pose_depth_vis_head_left.png"

DEFAULT_BASELINE_PATH = Path("/workspace/GlobalHumanoidRobotChallenge_2026_Baseline")
DEFAULT_TASK1_CONFIG = DEFAULT_BASELINE_PATH / "Ubtech_sim" / "config" / "Part_Sorting.yaml"
CAMERA_NAME = "head_left"
CAMERA_PRIM_PATH = "/Root/Ref_Xform/Ref/head_pitch_link/head_stereo_left/head_stereo_left_Camera_01"
ROBOT_BASE_PRIM_PATH = "/Root/Ref_Xform/Ref"


def parse_args() -> argparse.Namespace:
    cwd = Path.cwd()
    baseline_default = cwd if (cwd / "Ubtech_sim").exists() else DEFAULT_BASELINE_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-path", default=str(baseline_default))
    parser.add_argument("--config-path", default=str(DEFAULT_TASK1_CONFIG))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--settle-time", type=float, default=None)
    parser.add_argument("--render-steps", type=int, default=35)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def git_info(path: Path) -> Dict[str, Any]:
    def run_git(args: List[str]) -> Any:
        try:
            return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "status_short": run_git(["status", "--short"]),
    }


def first_scalar(value: Any) -> Optional[Any]:
    import numpy as np

    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    scalar = arr.reshape(-1)[0]
    return scalar.item() if hasattr(scalar, "item") else scalar


def read_row_field(row: Any, key: str) -> Optional[Any]:
    try:
        return first_scalar(row[key])
    except Exception:
        return None


def bbox_from_row(row: Any) -> Optional[List[int]]:
    for keys in (("x_min", "y_min", "x_max", "y_max"), ("xMin", "yMin", "xMax", "yMax")):
        vals = [read_row_field(row, k) for k in keys]
        if all(v is not None for v in vals):
            return [int(round(float(v))) for v in vals]
    try:
        vals = [first_scalar(row[i]) for i in range(1, 5)]
        if all(v is not None for v in vals):
            return [int(round(float(v))) for v in vals]
    except Exception:
        pass
    return None


def label_from_row(row: Any, id_to_labels: Dict[Any, Any]) -> Tuple[str, str]:
    semantic_id = None
    for key in ("semanticId", "semantic_id", "id", "semantic"):
        semantic_id = read_row_field(row, key)
        if semantic_id is not None:
            break
    try:
        sid = str(int(semantic_id)) if semantic_id is not None else "unknown"
    except Exception:
        sid = str(semantic_id) if semantic_id is not None else "unknown"

    label_entry = {}
    for candidate in (sid, str(sid)):
        label_entry = id_to_labels.get(candidate) or {}
        if label_entry:
            break
    if not label_entry:
        try:
            label_entry = id_to_labels.get(int(sid)) or {}
        except Exception:
            label_entry = {}
    label_text = json.dumps(json_safe(label_entry), ensure_ascii=False).lower()
    if "part_a" in label_text:
        return "part_a", sid
    if "part_b" in label_text:
        return "part_b", sid
    return "unknown", sid


def parse_semantic_boxes(raw: Any) -> Dict[str, Any]:
    data = raw.get("data", raw) if isinstance(raw, dict) else raw
    info = raw.get("info", {}) if isinstance(raw, dict) else {}
    id_to_labels = info.get("idToLabels") or info.get("id_to_labels") or {}
    prim_paths = info.get("primPaths", [])
    bbox_ids = info.get("bboxIds", [])
    boxes: List[Dict[str, Any]] = []
    try:
        rows: Iterable[Any] = list(data)
    except Exception:
        rows = []
    for idx, row in enumerate(rows):
        bbox = bbox_from_row(row)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        class_id, semantic_id = label_from_row(row, id_to_labels)
        boxes.append(
            {
                "class_id": class_id,
                "semantic_id": semantic_id,
                "bbox_xyxy": [x1, y1, x2, y2],
                "centroid_px": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                "width_px": x2 - x1,
                "height_px": y2 - y1,
                "prim_path": prim_paths[idx] if idx < len(prim_paths) else "unknown",
                "bbox_id": bbox_ids[idx] if idx < len(bbox_ids) else "unknown",
            }
        )
    return {
        "boxes": boxes,
        "info": json_safe(info),
        "raw_count": len(list(rows)) if not isinstance(rows, list) else len(rows),
    }


def instantiate_scene_builder(scene_builder_cls: Any, cfg: dict, world: Any) -> Any:
    try:
        return scene_builder_cls(cfg, data_logger=None, world=world)
    except TypeError:
        return scene_builder_cls(cfg, data_logger=None)


def normalize_rgb(rgb: Any) -> "np.ndarray":
    import numpy as np

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
    return np.clip(arr, 0, 255).astype(np.uint8)


def normalize_depth(depth: Any) -> "np.ndarray":
    import numpy as np

    arr = np.asarray(depth)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32, copy=False)


def depth_vis(depth: "np.ndarray") -> "np.ndarray":
    import numpy as np

    valid = np.isfinite(depth) & (depth > 0)
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        vis = (norm * 255.0).astype(np.uint8)
    return vis


def matrix_to_np(matrix: Any) -> "np.ndarray":
    import numpy as np

    # Gf.Matrix4d indexes as USD row-major with translation in the last row.
    return np.array([[float(matrix[i][j]) for j in range(4)] for i in range(4)], dtype=float).T


def get_world_matrix(stage: Any, prim_path: str) -> "np.ndarray":
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"invalid prim path: {prim_path}")
    return matrix_to_np(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim))


def extract_intrinsics(stage: Any, camera: Any, width: int, height: int) -> Dict[str, Any]:
    prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"invalid camera prim: {CAMERA_PRIM_PATH}")
    try:
        if hasattr(camera, "get_intrinsics_matrix"):
            mat = camera.get_intrinsics_matrix()
            if mat is not None:
                return {
                    "status": "from_camera_api_get_intrinsics_matrix",
                    "matrix_3x3": json_safe(mat),
                    "fx": float(mat[0][0]),
                    "fy": float(mat[1][1]),
                    "cx": float(mat[0][2]),
                    "cy": float(mat[1][2]),
                }
    except Exception:
        pass

    focal = float(prim.GetAttribute("focalLength").Get())
    ha = float(prim.GetAttribute("horizontalAperture").Get())
    va = float(prim.GetAttribute("verticalAperture").Get())
    fx = focal / ha * float(width)
    fy = focal / va * float(height)
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    return {
        "status": "computed_from_usd_focal_aperture_assuming_center_principal_point",
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "matrix_3x3": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
    }


def backproject_cv(u: float, v: float, z: float, fx: float, fy: float, cx: float, cy: float) -> "np.ndarray":
    import numpy as np

    return np.array([(u - cx) / fx * z, (v - cy) / fy * z, z], dtype=float)


def cv_camera_to_usd_camera(point_cv: "np.ndarray") -> "np.ndarray":
    import numpy as np

    return np.array([point_cv[0], -point_cv[1], -point_cv[2], 1.0], dtype=float)


def rotation_matrix_to_quat_xyzw(rotation: "np.ndarray") -> List[float]:
    import numpy as np

    r = np.asarray(rotation, dtype=float)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qx, qy, qz, qw], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm > 0:
        quat /= norm
    return quat.tolist()


def yaw_from_rotation_base(rotation: "np.ndarray") -> float:
    import numpy as np

    r = np.asarray(rotation, dtype=float)
    local_x_axis_in_base = r[:2, 0]
    return float(math.atan2(local_x_axis_in_base[1], local_x_axis_in_base[0]))


def object_pose_base_from_prim(stage: Any, prim_path: str, t_base_world: "np.ndarray") -> Optional[Dict[str, Any]]:
    try:
        t_world_object = get_world_matrix(stage, prim_path)
    except Exception:
        return None
    t_base_object = t_base_world @ t_world_object
    rotation_base_object = t_base_object[:3, :3]
    return {
        "source": "semantic_prim_runtime_transform",
        "T_world_object": t_world_object,
        "T_base_object": t_base_object,
        "position_m": t_base_object[:3, 3],
        "orientation_xyzw": rotation_matrix_to_quat_xyzw(rotation_base_object),
        "yaw_rad": yaw_from_rotation_base(rotation_base_object),
        "yaw_convention": "atan2 of object local +X axis projected into base XY plane",
    }


def robust_depth_in_bbox(depth: "np.ndarray", bbox: List[int]) -> Tuple[Optional[float], float, Dict[str, Any]]:
    import numpy as np

    h, w = depth.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    crop = depth[y1 : y2 + 1, x1 : x2 + 1]
    valid = np.isfinite(crop) & (crop > 0)
    if not valid.any():
        return None, 0.0, {"valid_count": 0, "total_count": int(crop.size)}
    vals = crop[valid]
    return (
        float(np.median(vals)),
        float(valid.mean()),
        {
            "valid_count": int(valid.sum()),
            "total_count": int(crop.size),
            "min": float(vals.min()),
            "median": float(np.median(vals)),
            "max": float(vals.max()),
        },
    )


def save_overlay(rgb: Any, detections: List[Dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw
    import math

    img = Image.fromarray(normalize_rgb(rgb))
    draw = ImageDraw.Draw(img)
    colors = {"part_a": (255, 40, 40), "part_b": (40, 120, 255), "unknown": (255, 190, 40)}
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox_xyxy"]
        u, v = det["centroid_px"]
        color = colors.get(det["class_id"], colors["unknown"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.ellipse([u - 4, v - 4, u + 4, v + 4], outline=color, width=2)
        draw.text((x1, max(0, y1 - 12)), f"{idx}:{det['class_id']}", fill=color)
        yaw = det.get("grasp_hint", {}).get("yaw_rad")
        if isinstance(yaw, (int, float)):
            length = 28
            draw.line(
                [u, v, u + length * math.cos(yaw), v + length * math.sin(yaw)],
                fill=color,
                width=3,
            )
    img.save(OUT_OVERLAY)


def pose_position_available(obj: Dict[str, Any]) -> bool:
    pos = obj.get("pose_base", {}).get("position_m")
    return pos is not None and not isinstance(pos, str)


def write_outputs(data: Dict[str, Any]) -> None:
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_LOG.write_text(json.dumps(json_safe(data.get("runtime_log", {})), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    counts = data.get("counts", {})
    lines = [
        "# Semantic Pose Base Pipeline Report",
        "",
        "## Status",
        f"- timestamp: `{data.get('timestamp')}`",
        f"- runtime_success: `{data.get('failure_reason') is None}`",
        f"- failure_reason: `{data.get('failure_reason')}`",
        f"- seed: `{data.get('reproducibility', {}).get('seed')}`",
        f"- camera_name: `{CAMERA_NAME}`",
        f"- total_objects: `{counts.get('total_objects', 0)}`",
        f"- part_a_count: `{counts.get('part_a', 0)}`",
        f"- part_b_count: `{counts.get('part_b', 0)}`",
        f"- four_object_pass: `{data.get('four_object_pass')}`",
        f"- pose_base_available: `{data.get('pose_base_available')}`",
        f"- orientation_available: `{data.get('orientation_available')}`",
        f"- yaw_available: `{data.get('yaw_available')}`",
        "",
        "## Outputs",
        f"- perception_interface: `{OUT_JSON}`",
        f"- runtime_log: `{OUT_LOG}`",
        f"- overlay: `{OUT_OVERLAY}`",
        f"- rgb_same_run: `{OUT_RGB}`",
        f"- depth_same_run: `{OUT_DEPTH}`",
        f"- depth_vis_same_run: `{OUT_DEPTH_VIS}`",
        "",
        "## Detections",
        "",
        "| idx | class_id | bbox_xyxy | centroid_px | depth_median_m | pose_base_position_m | orientation_xyzw | yaw_rad |",
        "|---:|---|---|---|---:|---|---|---:|",
    ]
    for idx, det in enumerate(data.get("objects", [])):
        pose = det.get("pose_base", {})
        grasp = det.get("grasp_hint", {})
        lines.append(
            f"| {idx} | `{det.get('class_id')}` | `{det.get('bbox_xyxy')}` | `{det.get('centroid_px')}` | {det.get('depth_median_m')} | `{pose.get('position_m')}` | `{pose.get('orientation_xyzw')}` | {grasp.get('yaw_rad')} |"
        )
    lines += [
        "",
        "## Reproducibility",
        "",
        "- The script sets Python, NumPy, and Replicator seed controls where available.",
        "- The run records command, seed, baseline git info, config path, camera path, and same-run RGB/depth artifacts.",
        "- Exact pixel equality across machines can still depend on Isaac Sim/GPU/driver versions and physics timing; this report records enough metadata to rerun and review.",
        "",
        "## Planner/Motion Contract",
        "",
        "- Use `objects[*].pose_base.frame == /Root/Ref_Xform/Ref`.",
        "- Positions are meters.",
        "- Orientation quaternion uses `xyzw` convention and comes from the semantic object prim runtime transform.",
        "- `grasp_hint.yaw_rad` is the object local +X axis projected into the base XY plane.",
        "- Prefer this semantic interface over the older color-threshold `perception_interface.json`.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    original_argv = list(sys.argv)
    args = parse_args()
    sys.argv = [sys.argv[0]]
    baseline_path = Path(args.baseline_path).resolve()
    config_path = Path(args.config_path).resolve()
    data: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": "runtime_head_left_semantic_rgbd_pose_base",
        "baseline_path": str(baseline_path),
        "config_path": str(config_path),
        "camera_name": CAMERA_NAME,
        "camera_prim_path": CAMERA_PRIM_PATH,
        "base_frame": ROBOT_BASE_PRIM_PATH,
        "headless": bool(args.headless),
        "objects": [],
        "failure_reason": None,
        "reproducibility": {
            "seed": int(args.seed),
            "seed_controls": ["random.seed", "numpy.random.seed", "rep.set_global_seed_if_available"],
            "baseline_git": git_info(baseline_path),
            "command": " ".join(original_argv),
        },
    }

    if not baseline_path.exists() or not config_path.exists():
        data["failure_reason"] = "missing baseline_path or config_path"
        write_outputs(data)
        return 2

    random.seed(int(args.seed))
    os.environ["PYTHONHASHSEED"] = str(int(args.seed))
    os.chdir(str(baseline_path))
    sys.path.insert(0, str(baseline_path / "Ubtech_sim"))

    from isaacsim import SimulationApp

    kit = SimulationApp({"width": 1280, "height": 720, "headless": bool(args.headless)})
    return_code = 0
    try:
        import numpy as np
        import omni
        import omni.replicator.core as rep
        from isaacsim.core.api import World
        from PIL import Image
        from source.config_loader import apply_scatter_config, load_config
        from source.RobotArticulation import RobotArticulation
        from source.SceneBuilder import SceneBuilder

        np.random.seed(int(args.seed))
        rep_seed_status = "unknown"
        if hasattr(rep, "set_global_seed"):
            try:
                rep.set_global_seed(int(args.seed))
                rep_seed_status = "set"
            except Exception as exc:
                rep_seed_status = f"failed: {exc}"
        data["reproducibility"]["replicator_seed_status"] = rep_seed_status

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
        robot = RobotArticulation(prim_path=ROBOT_BASE_PRIM_PATH, name="walkerS2")
        robot.initialize()
        camera = robot.cameras.get(CAMERA_NAME)
        if camera is not None and hasattr(camera, "set_resolution"):
            camera.set_resolution((int(args.width), int(args.height)))
        world.play()
        for _ in range(max(0, int(args.render_steps))):
            world.step(render=True)

        render_product = rep.create.render_product(CAMERA_PRIM_PATH, (int(args.width), int(args.height)))
        semantic_init = {"semanticTypes": ["class"]}
        bbox_annot = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight_fast", init_params=semantic_init)
        bbox_annot.attach([render_product])
        rep.orchestrator.step()
        for _ in range(5):
            world.step(render=True)
            rep.orchestrator.step()

        stage = omni.usd.get_context().get_stage()
        t_world_camera = get_world_matrix(stage, CAMERA_PRIM_PATH)
        t_world_base = get_world_matrix(stage, ROBOT_BASE_PRIM_PATH)
        t_base_world = np.linalg.inv(t_world_base)
        t_base_camera = t_base_world @ t_world_camera
        intr = extract_intrinsics(stage, camera, int(args.width), int(args.height))
        fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])

        rgb = normalize_rgb(robot.get_camera_rgb(CAMERA_NAME))
        depth = normalize_depth(robot.get_camera_depth(CAMERA_NAME))
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(OUT_RGB)
        np.save(OUT_DEPTH, depth)
        Image.fromarray(depth_vis(depth)).save(OUT_DEPTH_VIS)

        parsed = parse_semantic_boxes(bbox_annot.get_data())
        semantic_boxes = [b for b in parsed["boxes"] if b["class_id"] in {"part_a", "part_b"}]
        semantic_boxes.sort(key=lambda b: (b["class_id"], b["bbox_xyxy"][1], b["bbox_xyxy"][0]))

        objects: List[Dict[str, Any]] = []
        for idx, box in enumerate(semantic_boxes):
            u, v = [float(x) for x in box["centroid_px"]]
            z, valid_ratio, depth_stats = robust_depth_in_bbox(depth, box["bbox_xyxy"])
            prim_pose_base = object_pose_base_from_prim(stage, box.get("prim_path", "unknown"), t_base_world)
            if z is None:
                if prim_pose_base is not None:
                    pose_base = {
                        "frame": ROBOT_BASE_PRIM_PATH,
                        "position_m": prim_pose_base["position_m"],
                        "orientation_xyzw": prim_pose_base["orientation_xyzw"],
                        "source": "semantic_prim_runtime_transform_no_valid_depth_crosscheck",
                        "rgbd_centroid_position_m": "unknown",
                        "rgbd_centroid_position_delta_m": "unknown",
                    }
                else:
                    pose_base = {
                        "frame": ROBOT_BASE_PRIM_PATH,
                        "position_m": "unknown",
                        "orientation_xyzw": "unknown",
                        "failure_reason": "no_valid_depth_in_bbox_and_no_valid_prim_transform",
                    }
                point_camera_cv = point_world = "unknown"
            else:
                point_camera_cv = backproject_cv(u, v, z, fx, fy, cx, cy)
                point_world_h = t_world_camera @ cv_camera_to_usd_camera(point_camera_cv)
                point_base_h = t_base_world @ point_world_h
                point_world = point_world_h[:3]
                centroid_pose_base_position = point_base_h[:3]
                if prim_pose_base is not None:
                    pose_base = {
                        "frame": ROBOT_BASE_PRIM_PATH,
                        "position_m": prim_pose_base["position_m"],
                        "orientation_xyzw": prim_pose_base["orientation_xyzw"],
                        "source": "semantic_prim_runtime_transform",
                        "rgbd_centroid_position_m": centroid_pose_base_position,
                        "rgbd_centroid_position_delta_m": float(
                            np.linalg.norm(np.asarray(prim_pose_base["position_m"]) - centroid_pose_base_position)
                        ),
                    }
                else:
                    pose_base = {
                        "frame": ROBOT_BASE_PRIM_PATH,
                        "position_m": centroid_pose_base_position,
                        "orientation_xyzw": "unknown",
                        "source": "semantic_bbox_centroid_depth_head_left_same_runtime_frame",
                    }
            objects.append(
                {
                    "index": idx,
                    "class_id": box["class_id"],
                    "confidence": 1.0,
                    "confidence_note": "Replicator semantic class from baseline SceneBuilder; not learned detector confidence.",
                    "semantic_id": box["semantic_id"],
                    "prim_path": box.get("prim_path", "unknown"),
                    "bbox_xyxy": box["bbox_xyxy"],
                    "centroid_px": box["centroid_px"],
                    "depth_median_m": z if z is not None else "unknown",
                    "depth_valid_ratio": valid_ratio,
                    "depth_stats": depth_stats,
                    "centroid_camera_cv_m": point_camera_cv,
                    "centroid_world_m": point_world,
                    "object_runtime_transform": prim_pose_base if prim_pose_base is not None else "unknown",
                    "pose_base": pose_base,
                    "grasp_hint": {
                        "yaw_rad": prim_pose_base["yaw_rad"] if prim_pose_base is not None else "unknown",
                        "yaw_convention": prim_pose_base["yaw_convention"] if prim_pose_base is not None else "unknown",
                        "approach": "top_down_workspace_candidate",
                        "quality": "semantic_prim_transform_with_rgbd_centroid_crosscheck"
                        if prim_pose_base is not None
                        else "semantic_bbox_centroid_only_no_orientation",
                    },
                }
            )

        counts = {
            "total_objects": len(objects),
            "part_a": sum(1 for o in objects if o["class_id"] == "part_a"),
            "part_b": sum(1 for o in objects if o["class_id"] == "part_b"),
            "unknown": sum(1 for b in parsed["boxes"] if b["class_id"] == "unknown"),
        }
        data.update(
            {
                "camera_intrinsics": intr,
                "depth_unit": "meters_runtime_distance_to_image_plane_stage_units",
                "T_world_camera": t_world_camera,
                "T_world_base": t_world_base,
                "T_base_camera": t_base_camera,
                "objects": objects,
                "selected_object": objects[0] if objects else None,
                "counts": counts,
                "four_object_pass": counts["part_a"] == 2 and counts["part_b"] == 2,
                "pose_base_available": bool(objects) and all(pose_position_available(o) for o in objects),
                "orientation_available": bool(objects)
                and all(isinstance(o.get("pose_base", {}).get("orientation_xyzw"), list) for o in objects),
                "yaw_available": bool(objects)
                and all(isinstance(o.get("grasp_hint", {}).get("yaw_rad"), (int, float)) for o in objects),
                "same_run_artifacts": {
                    "rgb_path": str(OUT_RGB),
                    "depth_path": str(OUT_DEPTH),
                    "depth_vis_path": str(OUT_DEPTH_VIS),
                    "overlay_path": str(OUT_OVERLAY),
                },
                "semantic_raw": {
                    "raw_count": parsed["raw_count"],
                    "id_to_labels": parsed["info"].get("idToLabels", parsed["info"].get("id_to_labels", {})),
                },
                "runtime_log": {
                    "seed": int(args.seed),
                    "resolution": [int(args.width), int(args.height)],
                    "settle_time_s": float(settle_time),
                    "render_steps": int(args.render_steps),
                    "counts": counts,
                    "four_object_pass": counts["part_a"] == 2 and counts["part_b"] == 2,
                    "pose_base_available": bool(objects),
                    "orientation_available": bool(objects)
                    and all(isinstance(o.get("pose_base", {}).get("orientation_xyzw"), list) for o in objects),
                    "yaw_available": bool(objects)
                    and all(isinstance(o.get("grasp_hint", {}).get("yaw_rad"), (int, float)) for o in objects),
                },
                "limits": [
                    "Semantic labels come from simulator annotations, not a trained real detector.",
                    "orientation_xyzw and yaw_rad come from simulator semantic prim runtime transform.",
                    "RGB-D centroid pose is kept as a cross-check, not the primary final pose.",
                    "Exact pixel reproduction across machines can depend on Isaac Sim/GPU/driver versions even with seeds.",
                ],
            }
        )
        save_overlay(rgb, objects)
    except Exception as exc:
        data["failure_reason"] = str(exc)
        data["traceback"] = traceback.format_exc(limit=20)
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
