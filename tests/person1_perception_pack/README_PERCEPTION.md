# README — Perception Pack (Người 1)

**HRC2026 Training Camp — Task 1: Precise Desktop Sorting of Workpieces**

## What it does

Detects 4 workpieces on a desktop, classifies them as `part_A` or `part_B`, estimates 3D centroid and grasp hint in robot-base frame, and exports a structured JSON for the Planner/FSM (Người 2) and Motion Primitive (Người 3) to consume directly.

## Scene info (from Part_Sorting.yaml)

- **Robot**: Walker S2 at (0.7, -0.2, 0.9), facing +Y
- **Camera**: `head_stereo_left_Camera_01`
- **Table**: surface z ≈ 1.0m
- **Scatter area**: x:[0.50, 0.80], y:[0.10, 0.30], z: 1.04
- **Box (bin)**: (1.2, 0.3, 1.05) — fixed
- **Part A**: 2 variants (ori_color, red) — shape = Task1_PartA.usd
- **Part B**: 2 variants (blue, ori_color) — shape = Part_B.usd
- ⚠ Both have `ori_color` variant → **cannot classify by colour alone!**

## Quick start

```bash
# 1. Install dependencies
pip install numpy opencv-python pyyaml pytest

# 2. In Isaac Sim Script Editor — run these scripts:
#    scripts/extract_camera_params.py  → fills intrinsics + T_base_camera
#    scripts/capture_rgbd.py           → captures sample_rgb.png + sample_depth.npy

# 3. Copy captured images to lab_outputs/perception/

# 4. Run the full pipeline (depth_fg = recommended detection method)
python run_perception.py --config configs/task1_perception.yaml --method depth_fg

# 5. (Optional) Tune shape classifier
python scripts/measure_shape_features.py \
    --rgb lab_outputs/perception/sample_rgb.png \
    --depth lab_outputs/perception/sample_depth.npy

# 6. Run unit tests
python -m pytest tests/ -v
```

## Classification strategy

Since Part A and Part B both have an `ori_color` variant (same colour), classification uses a **two-step approach**:

1. **Colour hint**: red object → Part A (high confidence), blue object → Part B (high confidence)
2. **Shape fallback**: ori_color objects are distinguished by contour shape (aspect ratio, solidity, circularity) since Task1_PartA.usd and Part_B.usd have different 3D geometry

## Inputs

| Input | Source |
|-------|--------|
| RGB image | Isaac Sim head_stereo_left camera |
| Depth image (.npy) | Isaac Sim `distance_to_image_plane` annotator |
| Camera intrinsics | `scripts/extract_camera_params.py` → `configs/camera.yaml` |
| T_base_camera | `scripts/extract_camera_params.py` → `configs/task1_perception.yaml` |
| HSV colour ranges | `scripts/measure_shape_features.py` + HSV picker |

## Outputs

| File | Purpose | Consumer |
|------|---------|----------|
| `perception_interface.json` | **Main output** — all object states | Người 2, 3, 4 |
| `pose_estimator_report.csv` | Per-object centroid/pose data | Debug, Người 4 |
| `yaw_report.csv` | Yaw/grasp direction per object | Người 3 |
| `camera_config_sheet.csv` | Camera params (shared reference) | Whole team |
| `depth_sanity_report.md` | Depth quality check | Self-check |
| `transform_sanity_report.md` | Transform validity check | Self-check, Người 2 |
| `confusion_matrix_task1.csv` | Classification accuracy | Người 4 |
| `failure_cases_perception.jsonl` | Failed detection log | Người 4 |
| Overlay PNGs | Visual debug evidence | Whole team |

## Output JSON schema

```json
{
  "frame_id": 0,
  "timestamp": 1234567890.123,
  "camera_name": "front_rgbd",
  "objects": [
    {
      "object_id": "obj_001",
      "class_id": "part_A",
      "confidence": 0.88,
      "bbox_xyxy": [310, 220, 380, 292],
      "centroid_px": [345.0, 256.0],
      "centroid_camera_m": [0.031, 0.019, 0.740],
      "pose_base": {
        "position_m": [0.46, -0.15, 0.82],
        "quaternion_xyzw": [0.0, 0.0, 0.707, 0.707]
      },
      "grasp_hint": {
        "approach_axis": "z_down",
        "yaw_rad": 1.57,
        "grasp_width_m": 0.045
      },
      "failure_reason": null
    }
  ],
  "summary": {
    "num_objects": 4,
    "num_valid_objects": 4,
    "num_invalid_depth": 0
  }
}
```

## Directory structure

```
person1_perception_pack/
├── README_PERCEPTION.md
├── run_perception.py              # main entry point
├── task1_perception_start_checklist.md
├── scripts/
│   ├── extract_camera_params.py   # run in Isaac Sim → fills YAML
│   ├── capture_rgbd.py            # run in Isaac Sim → captures images
│   └── measure_shape_features.py  # run locally → tunes shape classifier
├── src/
│   ├── camera_utils.py            # intrinsics, pixel-to-3D, depth sanity
│   ├── transform_utils.py         # frame transforms, quaternions, sanity
│   ├── perception.py              # detection, classification, pose, JSON export
│   └── perception_debug.py        # overlays, confusion matrix, depth preview
├── configs/
│   ├── task1_perception.yaml      # full pipeline config (with official scene info)
│   └── camera.yaml                # camera hardware/sim params
├── tests/
│   └── test_camera_and_transform.py
└── lab_outputs/perception/        # generated artifacts go here
```

## Known limitations

- **Colour-only classification is insufficient** — both Part A and Part B have `ori_color` variant. Shape classifier must be tuned from actual samples.
- Detection uses depth foreground — sensitive to `fg_threshold_m`; tune if table depth varies.
- Yaw estimation via `minAreaRect` may be noisy for near-square objects; switch to PCA.
- `quaternion_xyzw` in `pose_base` defaults to identity; update when rotation estimation is implemented.
- Confidence scoring is rule-based; calibrate penalties to actual scene conditions.
- `T_base_camera` must be extracted via `scripts/extract_camera_params.py` (identity placeholder will produce wrong poses).
- Head stereo camera is farther from objects than wrist cameras — depth precision lower.

## Failure reasons

| Code | Meaning |
|------|---------|
| `INVALID_DEPTH` | Depth at object centroid is NaN/0/negative |
| `LOW_CONFIDENCE` | Confidence below grasp threshold |
| `CLASS_AMBIGUOUS` | Cannot distinguish part_A from part_B |
| `MASK_TOO_SMALL` | Detected region smaller than minimum |
| `MASK_FRAGMENTED` | Mask split into disconnected pieces |
| `POSE_OUT_OF_RANGE` | 3D position outside robot workspace |
| `TRANSFORM_NOT_AVAILABLE` | T_base_camera not yet configured |
| `GRASP_HINT_UNSTABLE` | Yaw varies excessively across frames |
