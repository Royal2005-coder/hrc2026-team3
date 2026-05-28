# Runbook Tái Lập Kết Quả Cho Task 1 Perception

## Mục tiêu

Tài liệu này mô tả cách chạy lại kết quả perception hiện tại và cách guide sau nên tiêu thụ output đó.

Nhánh final hiện tại là:

- semantic annotation + RGB-D cùng runtime

Đây là pipeline perception dùng trong Isaac Sim baseline, không phải detector production ngoài simulator.

## Đường dẫn quan trọng

Project:

`/home/ubuntu/Team2/Task1-Perception`

Baseline:

`/workspace/GlobalHumanoidRobotChallenge_2026_Baseline`

Interface chính:

`/home/ubuntu/Team2/Task1-Perception/outputs/task1/perception/json/perception_interface_semantic_pose_base.json`

## Cách chạy lại perception runtime

Chạy từ baseline repo:

```bash
cd /workspace/GlobalHumanoidRobotChallenge_2026_Baseline
/isaac-sim/python.sh /home/ubuntu/Team2/Task1-Perception/scripts/run_task1_semantic_pose_pipeline.py \
  --config-path /workspace/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim/config/Part_Sorting.yaml \
  --headless \
  --seed 20260521
```

## Kết quả mong đợi ở mức high-level

- `camera`: `head_left`
- `resolution`: `512x512`
- `object_count`: `4`
- `class_split`: `2 part_a`, `2 part_b`
- `pose_base_available`: `true`
- `orientation_available`: `true`
- `yaw_available`: `true`
- `pose_base.frame`: `/Root/Ref_Xform/Ref`
- `position unit`: `meters`
- `orientation_xyzw`: quaternion theo quy ước `xyzw`
- `grasp_hint.yaw_rad`: đơn vị `radian`

## Cách validate offline

Chạy từ project repo:

```bash
cd /home/ubuntu/Team2/Task1-Perception
python -m py_compile scripts/validate_task1_perception_final.py
python scripts/validate_task1_perception_final.py
python -m json.tool outputs/task1/perception/json/final_perception_validation.json
```

Kết quả mong đợi:

`pass`

Output của validator:

- `outputs/task1/perception/json/final_perception_validation.json`
- `outputs/task1/perception/reports/final_perception_validation_report.md`

## Contract cho Planner/Motion

Planner hoặc Motion nên đọc:

`outputs/task1/perception/json/perception_interface_semantic_pose_base.json`

Field nên dùng:

- `objects[*].class_id`
- `objects[*].bbox_xyxy`
- `objects[*].centroid_px`
- `objects[*].depth_median_m`
- `objects[*].pose_base.frame`
- `objects[*].pose_base.position_m`
- `objects[*].pose_base.orientation_xyzw`
- `objects[*].grasp_hint.yaw_rad`

Field không nên giả định cứng:

- pixel coordinate cố định giữa các lần chạy

## Quy ước orientation và yaw

- `pose_base.orientation_xyzw` lấy từ semantic object prim transform ở runtime
- `grasp_hint.yaw_rad` được tính từ trục local `+X` sau khi chiếu xuống mặt phẳng `XY` của base frame
- centroid/depth từ RGB-D được giữ như dữ liệu đối chiếu

## Artifact nên đưa vào review

- `outputs/task1/perception/overlays/overlay_semantic_pose_base_head_left.png`
- `outputs/task1/perception/overlays/overlay_semantic_bboxes_head_left.png`
- `outputs/task1/perception/reports/semantic_pose_base_pipeline_report.md`
- `outputs/task1/perception/reports/final_perception_validation_report.md`
- `outputs/task1/perception/reports/task1_perception_planner_handoff_report.md`

## Ghi chú reproducibility

Script runtime hiện ghi lại:

- `command line`
- `seed`
- `config_path`
- baseline git commit/status
- camera path
- RGB cùng lần chạy
- depth `.npy` cùng lần chạy
- depth visualization
- semantic overlay
- JSON final

Script cũng cố gắng set:

- Python random seed
- NumPy random seed
- Replicator global seed khi API cho phép

Lưu ý:

Isaac Sim không bảo đảm pixel giống tuyệt đối giữa các máy. Mức reproducibility mục tiêu ở đây là:

- cùng lệnh
- cùng config
- cùng seed
- cùng schema output
- cùng metadata truy vết
- không sửa baseline trong pipeline perception này

## Unknown còn lại

- generalization ngoài simulator semantic annotation: chưa claim
- pixel identity tuyệt đối giữa các máy: chưa cam kết

## Chính sách baseline

Pipeline perception hiện tại không cần sửa:

- `Ubtech_sim/main.py`
- `Ubtech_sim/source/DataLogger.py`
- `Ubtech_sim/source/RobotArticulation.py`
