"""
run_perception.py — Main entry point for Task 1 perception pipeline.

Usage:
    python run_perception.py --config configs/task1_perception.yaml
    python run_perception.py --rgb sample_rgb.png --depth sample_depth.npy
    python run_perception.py --method depth_fg   (recommended for Task 1)

This script:
1. Loads config / camera params
2. Reads RGB + depth
3. Runs detection → classification → pose estimation
4. Exports all required artifacts
"""

import argparse
import os
import sys
import json
import numpy as np
import cv2 as cv
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from camera_utils import (
    CameraIntrinsics,
    depth_sanity,
    write_depth_sanity_report,
    save_camera_config_csv,
)
from transform_utils import (
    run_transform_sanity,
    make_transform,
    quaternion_to_rotation_matrix,
)
from perception import (
    run_perception,
    save_perception_json,
    save_pose_report_csv,
    save_yaw_report_csv,
    save_failure_cases_jsonl,
    detect_by_color,
    detect_by_depth_foreground,
    extract_shape_features,
)
from perception_debug import (
    draw_detection_overlay,
    draw_mask_overlay,
    draw_centroid_overlay,
    save_depth_preview,
    save_confusion_matrix_csv,
    build_confusion_matrix,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_intrinsics(cfg: dict) -> CameraIntrinsics:
    intr_cfg = cfg["intrinsics"]
    cam_cfg = cfg["camera"]
    return CameraIntrinsics(
        fx=intr_cfg["fx"], fy=intr_cfg["fy"],
        cx=intr_cfg["cx"], cy=intr_cfg["cy"],
        width=cam_cfg["rgb_resolution"][0],
        height=cam_cfg["rgb_resolution"][1],
        depth_unit=cam_cfg["depth_unit"],
    )


def build_T_base_camera(cfg: dict) -> np.ndarray:
    rows = cfg["extrinsics"]["T_base_camera"]
    return np.array(rows, dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Task 1 Perception Pipeline")
    parser.add_argument("--config", default="configs/task1_perception.yaml")
    parser.add_argument("--rgb", default=None, help="Path to RGB image")
    parser.add_argument("--depth", default=None, help="Path to depth .npy")
    parser.add_argument("--output-dir", default="lab_outputs/perception")
    parser.add_argument("--method", default="depth_fg",
                        choices=["color", "depth_fg"],
                        help="Detection method: 'depth_fg' (recommended) or 'color' (legacy)")
    args = parser.parse_args()

    # ── Load config ──
    cfg = load_config(args.config)
    intr = build_intrinsics(cfg)
    T_base_camera = build_T_base_camera(cfg)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    p = lambda name: os.path.join(out_dir, name)

    print("=" * 60)
    print("  HRC2026 Task 1 — Perception Pipeline")
    print(f"  Camera: {cfg['camera']['name']}")
    print(f"  Method: {args.method}")
    print(f"  Output: {out_dir}/")
    print("=" * 60)

    # ── Scene info ──
    if "world" in cfg:
        w = cfg["world"]
        print(f"\n[SCENE] Robot: {w.get('robot_position')}, "
              f"Table z≈{w.get('table_position', [0,0,0])[2] * 2:.2f}m, "
              f"Scatter: x{w.get('scatter_area', {}).get('x_range')}, "
              f"y{w.get('scatter_area', {}).get('y_range')}")

    # ── Load images ──
    rgb_path = args.rgb or p("sample_rgb.png")
    depth_path = args.depth or p("sample_depth.npy")

    if not os.path.exists(rgb_path):
        print(f"\n[ERROR] RGB not found: {rgb_path}")
        print("  → Chạy scripts/capture_rgbd.py trong Isaac Sim trước.")
        sys.exit(1)
    if not os.path.exists(depth_path):
        print(f"\n[ERROR] Depth not found: {depth_path}")
        print("  → Chạy scripts/capture_rgbd.py trong Isaac Sim trước.")
        sys.exit(1)

    rgb_bgr = cv.imread(rgb_path)
    depth = np.load(depth_path)
    print(f"\n[DATA] RGB: {rgb_bgr.shape}, Depth: {depth.shape}")

    # ══════════════════════════════════════════════════════════
    # Step A: Depth sanity
    # ══════════════════════════════════════════════════════════
    info = depth_sanity(depth)
    write_depth_sanity_report(info, p("depth_sanity_report.md"))
    save_depth_preview(depth, p("sample_depth_preview.png"))
    print(f"[DEPTH] valid={info['valid_ratio']:.1%}, "
          f"median={info['median']}, unit={info['guessed_unit']}")

    if info["guessed_unit"] == "millimeter" and cfg["camera"]["depth_unit"] == "meter":
        print("  ⚠ WARNING: depth looks like mm but config says meter!")
        print("  → Change depth_unit in YAML or scale depth manually")

    # ══════════════════════════════════════════════════════════
    # Step B: Camera config CSV
    # ══════════════════════════════════════════════════════════
    save_camera_config_csv(
        intr,
        T_source=cfg["extrinsics"]["T_base_camera_source"],
        camera_name=cfg["camera"]["name"],
        notes=f"Task 1 — {args.method} detection",
        path=p("camera_config_sheet.csv"),
    )

    # ══════════════════════════════════════════════════════════
    # Step C: Transform sanity
    # ══════════════════════════════════════════════════════════
    run_transform_sanity(T_base_camera, report_path=p("transform_sanity_report.md"))

    # ══════════════════════════════════════════════════════════
    # Step D: Run perception
    # ══════════════════════════════════════════════════════════
    hsv_ranges = cfg["detection"]["hsv_ranges"]
    state = run_perception(
        rgb_bgr, depth, intr, T_base_camera,
        hsv_ranges=hsv_ranges,
        frame_id=0,
        camera_name=cfg["camera"]["name"],
        detection_method=args.method,
    )

    objects = state["objects"]
    n_total = len(objects)
    n_valid = state["summary"]["num_valid_objects"]
    n_A = sum(1 for o in objects if o["class_id"] == "part_A")
    n_B = sum(1 for o in objects if o["class_id"] == "part_B")
    n_unk = sum(1 for o in objects if o["class_id"] == "unknown")

    print(f"\n[PERCEPTION] Detected {n_total} objects ({n_valid} valid)")
    print(f"  Part A: {n_A}, Part B: {n_B}, Unknown: {n_unk}")
    if n_total != 4:
        print(f"  ⚠ Expected 4 objects but detected {n_total}")

    for obj in objects:
        status = "✓" if obj["failure_reason"] is None else f"✗ {obj['failure_reason']}"
        print(f"  {obj['object_id']}: {obj['class_id']} "
              f"conf={obj['confidence']:.2f} {status}")

    # ══════════════════════════════════════════════════════════
    # Step E: Save all outputs
    # ══════════════════════════════════════════════════════════
    save_perception_json(state, p("perception_interface.json"))
    save_pose_report_csv(objects, p("pose_estimator_report.csv"))
    save_yaw_report_csv(objects, p("yaw_report.csv"))
    save_failure_cases_jsonl(objects, p("failure_cases_perception.jsonl"))

    # ══════════════════════════════════════════════════════════
    # Step F: Shape features report (for tuning classifier)
    # ══════════════════════════════════════════════════════════
    print(f"\n[SHAPE] Extracting shape features for classifier tuning...")
    shape_rows = []
    for obj in objects:
        # Re-detect to get contour (not stored in JSON)
        # For now, log from detection results
        if "shape_features" in obj:
            feat = obj["shape_features"]
            shape_rows.append({
                "object_id": obj["object_id"],
                "class_id": obj["class_id"],
                **feat,
            })
            print(f"  {obj['object_id']} ({obj['class_id']}): "
                  f"aspect={feat['min_rect_aspect']:.3f}, "
                  f"solidity={feat['solidity']:.3f}, "
                  f"circularity={feat['circularity']:.3f}")

    if shape_rows:
        import csv
        with open(p("shape_features_report.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=shape_rows[0].keys())
            writer.writeheader()
            writer.writerows(shape_rows)
        print(f"  → Saved to shape_features_report.csv")
        print(f"  → Use these values to tune shape_classifier thresholds in YAML")

    # ══════════════════════════════════════════════════════════
    # Step G: Debug overlays
    # ══════════════════════════════════════════════════════════
    draw_detection_overlay(rgb_bgr, objects, p("overlay_detection.png"))
    draw_centroid_overlay(rgb_bgr, objects, save_path=p("overlay_centroid.png"))

    # Depth foreground mask overlay
    if args.method == "depth_fg":
        _, fg_mask = detect_by_depth_foreground(depth)
        empty_mask = np.zeros(depth.shape[:2], dtype=np.uint8)
        draw_mask_overlay(rgb_bgr, fg_mask, empty_mask,
                          save_path=p("overlay_mask.png"))
    else:
        r = cfg["detection"]["hsv_ranges"]
        _, mask_A = detect_by_color(rgb_bgr,
                                     r.get("red", r.get("part_A", {})).get("lower", [0,100,100]),
                                     r.get("red", r.get("part_A", {})).get("upper", [10,255,255]))
        _, mask_B = detect_by_color(rgb_bgr,
                                     r.get("blue", r.get("part_B", {})).get("lower", [100,100,100]),
                                     r.get("blue", r.get("part_B", {})).get("upper", [130,255,255]))
        draw_mask_overlay(rgb_bgr, mask_A, mask_B, save_path=p("overlay_mask.png"))

    # ══════════════════════════════════════════════════════════
    # Step H: Workspace validation
    # ══════════════════════════════════════════════════════════
    if "world" in cfg:
        scatter = cfg["world"].get("scatter_area", {})
        x_range = scatter.get("x_range", [0.50, 0.80])
        y_range = scatter.get("y_range", [0.10, 0.30])
        print(f"\n[WORKSPACE] Checking pose_base against scatter area...")
        for obj in objects:
            if obj["pose_base"]:
                px, py, pz = obj["pose_base"]["position_m"]
                in_x = x_range[0] - 0.1 <= px <= x_range[1] + 0.1
                in_y = y_range[0] - 0.1 <= py <= y_range[1] + 0.1
                ok = "✓" if (in_x and in_y) else "⚠ OUT OF RANGE"
                print(f"  {obj['object_id']}: base=({px:.3f}, {py:.3f}, {pz:.3f}) {ok}")

    print(f"\n{'='*60}")
    print(f"  DONE — All artifacts saved to {out_dir}/")
    print(f"  perception_interface.json → sẵn sàng cho Người 2/3/4")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
