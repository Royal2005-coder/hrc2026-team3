"""
test_pipeline_integration.py — Integration test with synthetic data.

Creates a fake RGB + depth with known objects, runs the full pipeline,
and verifies outputs are correct.

Run:  python -m pytest tests/test_pipeline_integration.py -v
"""

import numpy as np
import cv2 as cv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from camera_utils import CameraIntrinsics, depth_sanity
from perception import (
    detect_by_color,
    detect_by_depth_foreground,
    extract_shape_features,
    classify_detections,
    make_object_state,
    run_perception,
    save_perception_json,
)
from perception_debug import (
    draw_detection_overlay,
    build_confusion_matrix,
)


def make_synthetic_scene():
    """
    Create a synthetic 640×480 RGB + depth scene with:
    - Background at depth 1.0m (table)
    - 2 red rectangles (Part A) at depth 0.95m
    - 2 blue squares (Part B) at depth 0.95m
    """
    H, W = 480, 640
    rgb = np.full((H, W, 3), (200, 200, 200), dtype=np.uint8)  # gray BG
    depth = np.full((H, W), 1.0, dtype=np.float32)  # table at 1.0m

    # Part A: red rectangles (tall and narrow)
    objects_gt = []

    # A1: red rectangle at (150, 200)
    cv.rectangle(rgb, (130, 180), (170, 250), (0, 0, 220), -1)  # BGR red
    depth[180:250, 130:170] = 0.95
    objects_gt.append({"class": "part_A", "cx": 150, "cy": 215})

    # A2: red rectangle at (300, 200)
    cv.rectangle(rgb, (280, 180), (320, 250), (0, 0, 200), -1)
    depth[180:250, 280:320] = 0.95
    objects_gt.append({"class": "part_A", "cx": 300, "cy": 215})

    # Part B: blue squares (wide)
    # B1: blue square at (450, 200)
    cv.rectangle(rgb, (420, 180), (480, 240), (220, 100, 0), -1)  # BGR blue
    depth[180:240, 420:480] = 0.95
    objects_gt.append({"class": "part_B", "cx": 450, "cy": 210})

    # B2: blue square at (450, 350)
    cv.rectangle(rgb, (420, 330), (480, 390), (200, 80, 0), -1)
    depth[330:390, 420:480] = 0.95
    objects_gt.append({"class": "part_B", "cx": 450, "cy": 360})

    return rgb, depth, objects_gt


class TestDepthForegroundDetection:
    def setup_method(self):
        self.rgb, self.depth, self.gt = make_synthetic_scene()

    def test_detects_4_objects(self):
        dets, mask = detect_by_depth_foreground(self.depth, fg_threshold_m=0.02)
        assert len(dets) == 4, f"Expected 4, got {len(dets)}"

    def test_foreground_mask_not_empty(self):
        _, mask = detect_by_depth_foreground(self.depth, fg_threshold_m=0.02)
        assert mask.sum() > 0

    def test_centroids_reasonable(self):
        dets, _ = detect_by_depth_foreground(self.depth, fg_threshold_m=0.02)
        for det in dets:
            cx, cy = det["centroid_px"]
            assert 0 < cx < 640
            assert 0 < cy < 480


class TestColorDetection:
    def setup_method(self):
        self.rgb, self.depth, self.gt = make_synthetic_scene()

    def test_detect_red(self):
        dets, mask = detect_by_color(self.rgb, [0, 80, 80], [10, 255, 255])
        assert len(dets) >= 2, f"Expected >=2 red, got {len(dets)}"

    def test_detect_blue(self):
        dets, mask = detect_by_color(self.rgb, [100, 60, 60], [130, 255, 255])
        assert len(dets) >= 2, f"Expected >=2 blue, got {len(dets)}"


class TestShapeFeatures:
    def test_rectangle_vs_square(self):
        # Tall rectangle
        rect_cnt = np.array([[[0,0]], [[40,0]], [[40,100]], [[0,100]]], dtype=np.int32)
        rect_feat = extract_shape_features(rect_cnt)

        # Square
        sq_cnt = np.array([[[0,0]], [[60,0]], [[60,60]], [[0,60]]], dtype=np.int32)
        sq_feat = extract_shape_features(sq_cnt)

        # Rectangle should have lower min_rect_aspect than square
        assert rect_feat["min_rect_aspect"] < sq_feat["min_rect_aspect"]
        # Square should have higher circularity
        assert sq_feat["circularity"] > rect_feat["circularity"]


class TestMakeObjectState:
    def setup_method(self):
        self.rgb, self.depth, _ = make_synthetic_scene()
        self.intr = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240,
                                      width=640, height=480)
        self.T = np.eye(4)

    def test_valid_object(self):
        det = {
            "bbox_xyxy": [130, 180, 170, 250],
            "centroid_px": [150.0, 215.0],
            "area_px": 2800.0,
            "contour": np.array([[[130,180]], [[170,180]], [[170,250]], [[130,250]]]),
            "class_id": "part_A",
        }
        obj = make_object_state(det, self.depth, self.intr, self.T)
        assert obj["failure_reason"] is None
        assert obj["class_id"] == "part_A"
        assert obj["confidence"] > 0.5
        assert obj["pose_base"] is not None
        assert obj["grasp_hint"]["approach_axis"] == "z_down"

    def test_invalid_depth(self):
        bad_depth = np.full_like(self.depth, np.nan)
        det = {
            "bbox_xyxy": [130, 180, 170, 250],
            "centroid_px": [150.0, 215.0],
            "area_px": 2800.0,
            "contour": np.array([[[130,180]], [[170,180]], [[170,250]], [[130,250]]]),
            "class_id": "part_A",
        }
        obj = make_object_state(det, bad_depth, self.intr, self.T)
        assert obj["failure_reason"] == "INVALID_DEPTH"


class TestFullPipeline:
    def setup_method(self):
        self.rgb, self.depth, _ = make_synthetic_scene()
        self.intr = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240,
                                      width=640, height=480)
        self.T = np.eye(4)

    def test_run_depth_fg(self):
        hsv = {
            "red": {"lower": [0, 80, 80], "upper": [10, 255, 255],
                    "implies_class": "part_A"},
            "blue": {"lower": [100, 60, 60], "upper": [130, 255, 255],
                     "implies_class": "part_B"},
        }
        state = run_perception(
            self.rgb, self.depth, self.intr, self.T,
            hsv_ranges=hsv, detection_method="depth_fg",
        )
        assert state["summary"]["num_objects"] == 4
        assert all(key in state for key in ["frame_id", "timestamp", "objects", "summary"])

    def test_run_color_legacy(self):
        hsv = {
            "part_A": {"lower": [0, 80, 80], "upper": [10, 255, 255]},
            "part_B": {"lower": [100, 60, 60], "upper": [130, 255, 255]},
        }
        state = run_perception(
            self.rgb, self.depth, self.intr, self.T,
            hsv_ranges=hsv, detection_method="color",
        )
        assert state["summary"]["num_objects"] >= 4

    def test_json_serializable(self, tmp_path):
        hsv = {
            "red": {"lower": [0, 80, 80], "upper": [10, 255, 255],
                    "implies_class": "part_A"},
            "blue": {"lower": [100, 60, 60], "upper": [130, 255, 255],
                     "implies_class": "part_B"},
        }
        state = run_perception(
            self.rgb, self.depth, self.intr, self.T,
            hsv_ranges=hsv, detection_method="depth_fg",
        )
        out = str(tmp_path / "test_output.json")
        save_perception_json(state, out)

        with open(out) as f:
            loaded = json.load(f)
        assert loaded["summary"]["num_objects"] == state["summary"]["num_objects"]


class TestConfusionMatrix:
    def test_perfect(self):
        preds = ["part_A", "part_A", "part_B", "part_B"]
        gts = ["part_A", "part_A", "part_B", "part_B"]
        cm = build_confusion_matrix(preds, gts)
        assert cm["accuracy"] == 1.0

    def test_one_wrong(self):
        preds = ["part_A", "part_B", "part_B", "part_B"]
        gts = ["part_A", "part_A", "part_B", "part_B"]
        cm = build_confusion_matrix(preds, gts)
        assert cm["accuracy"] == 0.75
