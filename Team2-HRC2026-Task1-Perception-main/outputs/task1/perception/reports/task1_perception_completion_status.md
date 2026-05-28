# Trạng Thái Hoàn Thiện Task 1 Perception

## Trạng thái chung

- `date`: `2026-05-22`
- `project`: `Task 1 - Precise Desktop Sorting of Workpieces`
- `current_decision`: `semantic RGB-D perception là output bàn giao hiện tại`

## Những gì đã hoàn thành

### 1. Camera và RGB-D readiness

- Đã xác minh 4 camera runtime trong baseline:
  - `head_left`
  - `head_right`
  - `wrist_left`
  - `wrist_right`
- Đã capture sample RGB-D thành công.
- Resolution chuẩn hiện tại của package chính là `512x512`.
- Evidence liên quan camera nằm tại:
  - `outputs/task1/perception/camera_inventory/`
  - `outputs/task1/perception/logs/`
  - `outputs/task1/perception/reports/`

### 2. Semantic final handoff

- File downstream chính thức:
  - `outputs/task1/perception/json/perception_interface_semantic_pose_base.json`
- Kết quả semantic runtime hiện tại:
  - `object_count = 4`
  - `class_split = 2 part_a + 2 part_b`
  - `pose_base.position_m` có sẵn
  - `pose_base.orientation_xyzw` có sẵn
  - `grasp_hint.yaw_rad` có sẵn
  - overlay và log đầy đủ

### 3. Offline validation

- Script validator:
  - `scripts/validate_task1_perception_final.py`
- Kết quả validator hiện tại:
  - `pass`
- Artifact kiểm tra:
  - `outputs/task1/perception/json/final_perception_validation.json`
  - `outputs/task1/perception/reports/final_perception_validation_report.md`

## Những gì được coi là final hiện tại

Các artifact sau là output perception chính thức để bàn giao:

- `json/perception_interface_semantic_pose_base.json`
- `json/semantic_bboxes_head_left.json`
- `overlays/overlay_semantic_bboxes_head_left.png`
- `overlays/overlay_semantic_pose_base_head_left.png`
- `camera_inventory/camera_config_sheet.csv`
- `camera_inventory/camera_geometry.json`
- `camera_inventory/base_transform_runtime.json`
- `reports/semantic_pose_base_pipeline_report.md`
- `reports/task1_perception_planner_handoff_report.md`
- `reports/task1_perception_reproducibility_runbook.md`
- `reports/final_perception_validation_report.md`

## Định nghĩa “Perception done”

Task 1 perception được coi là hoàn thành để bàn giao khi thỏa các điều kiện:

1. Có một JSON downstream chính thức và rõ ràng.
2. Có mô tả đầy đủ về frame, unit, quaternion convention, yaw convention.
3. Có overlay và log để kiểm tra lại kết quả.
4. Validator offline `pass`.
5. Teammate khác có thể rerun theo runbook mà không phụ thuộc thao tác GUI thủ công.

Package hiện tại đã đạt các điều kiện trên trong Isaac Sim baseline.

## Khuyến nghị hiện tại

- Với mục tiêu integration, perception đã đủ để bàn giao cho Planner, Motion và Evaluation thông qua file JSON chính.
- Guide sau nên dùng `outputs/task1/perception/json/perception_interface_semantic_pose_base.json`.
