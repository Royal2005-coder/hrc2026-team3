"""
measure_shape_features.py
=========================
Chạy SAU KHI đã capture sample_rgb.png + sample_depth.npy.

Script này:
1. Detect tất cả vật bằng depth foreground
2. Hiển thị từng vật → Tài nhập nhãn thủ công (A hoặc B)
3. Đo shape features (aspect ratio, solidity, circularity, hu moments)
4. In ra thresholds để điền vào YAML

Dùng trên máy local (cần GUI OpenCV).

Usage:
    python measure_shape_features.py --rgb sample_rgb.png --depth sample_depth.npy
    python measure_shape_features.py --rgb sample_rgb.png --depth sample_depth.npy --no-gui
"""

import cv2 as cv
import numpy as np
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from perception import detect_by_depth_foreground, extract_shape_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--output", default="shape_features_report.json")
    parser.add_argument("--no-gui", action="store_true",
                        help="Skip GUI labeling, just print features")
    args = parser.parse_args()

    rgb = cv.imread(args.rgb)
    depth = np.load(args.depth)

    if rgb is None:
        print(f"[ERROR] Cannot read RGB: {args.rgb}")
        sys.exit(1)

    print(f"RGB: {rgb.shape}, Depth: {depth.shape}")
    print(f"Detecting objects by depth foreground...\n")

    detections, fg_mask = detect_by_depth_foreground(depth)
    print(f"Found {len(detections)} objects\n")

    if len(detections) == 0:
        print("[WARNING] No objects detected!")
        print("  -> Check: is the depth valid? Is the camera seeing the table?")
        print("  -> Try adjusting fg_threshold_m in detect_by_depth_foreground()")
        sys.exit(1)

    features_A = []
    features_B = []
    all_features = []

    for i, det in enumerate(detections):
        cnt = det["contour"]
        feat = extract_shape_features(cnt)
        feat["detection_index"] = i
        feat["centroid_px"] = det["centroid_px"]

        print(f"{'~'*50}")
        print(f"Object {i}:")
        print(f"  centroid_px:       {det['centroid_px']}")
        print(f"  area_px:           {feat['area_px']:.0f}")
        print(f"  bbox_aspect_ratio: {feat['bbox_aspect_ratio']}")
        print(f"  min_rect_aspect:   {feat['min_rect_aspect']}")
        print(f"  solidity:          {feat['solidity']}")
        print(f"  circularity:       {feat['circularity']}")

        if not args.no_gui:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            pad = 20
            crop = rgb[max(0,y1-pad):min(rgb.shape[0],y2+pad),
                       max(0,x1-pad):min(rgb.shape[1],x2+pad)].copy()
            scale = max(1, 200 // max(crop.shape[0], crop.shape[1], 1))
            if scale > 1:
                crop = cv.resize(crop, None, fx=scale, fy=scale,
                                 interpolation=cv.INTER_NEAREST)

            cv.imshow(f"Object {i} - Press A or B to label, S to skip", crop)
            print(f"  -> Press A (Part A), B (Part B), or S (skip)")

            while True:
                key = cv.waitKey(0) & 0xFF
                if key == ord('a') or key == ord('A'):
                    feat["label"] = "part_A"
                    features_A.append(feat)
                    print(f"  -> Labeled: part_A")
                    break
                elif key == ord('b') or key == ord('B'):
                    feat["label"] = "part_B"
                    features_B.append(feat)
                    print(f"  -> Labeled: part_B")
                    break
                elif key == ord('s') or key == ord('S'):
                    feat["label"] = "skipped"
                    print(f"  -> Skipped")
                    break
            cv.destroyAllWindows()
        else:
            feat["label"] = "unknown"

        all_features.append(feat)

    # Summary
    print(f"\n{'='*50}")
    print(f"SHAPE FEATURE SUMMARY")
    print(f"{'='*50}")

    for label, group in [("part_A", features_A), ("part_B", features_B)]:
        if not group:
            print(f"\n  {label}: no samples labeled")
            continue
        print(f"\n  {label} ({len(group)} samples):")
        for key in ["area_px", "bbox_aspect_ratio", "min_rect_aspect",
                     "solidity", "circularity"]:
            vals = [f[key] for f in group]
            print(f"    {key:25s}: min={min(vals):.3f}  max={max(vals):.3f}  "
                  f"mean={np.mean(vals):.3f}")

    if features_A and features_B:
        print(f"\n{'~'*50}")
        print(f"SUGGESTED THRESHOLDS FOR YAML:")
        print(f"{'~'*50}")

        for key in ["min_rect_aspect", "bbox_aspect_ratio", "solidity", "circularity"]:
            a_vals = [f[key] for f in features_A]
            b_vals = [f[key] for f in features_B]
            a_min, a_max = min(a_vals), max(a_vals)
            b_min, b_max = min(b_vals), max(b_vals)

            if a_max < b_min or b_max < a_min:
                separator = "SEPARABLE"
            else:
                separator = "overlapping"

            print(f"  {key}:")
            print(f"    part_A: [{a_min:.3f}, {a_max:.3f}]")
            print(f"    part_B: [{b_min:.3f}, {b_max:.3f}]")
            print(f"    -> {separator}")

        print(f"\n  Chon feature co 'SEPARABLE' de dung trong classify_by_shape()")
        print(f"  Dien vao task1_perception.yaml -> shape_classifier section")

    for f in all_features:
        f.pop("hu_moments", None)

    with open(args.output, "w") as fp:
        json.dump({
            "total_objects": len(all_features),
            "part_A_count": len(features_A),
            "part_B_count": len(features_B),
            "features": all_features,
        }, fp, indent=2)
    print(f"\n[SAVED] {args.output}")


if __name__ == "__main__":
    main()
