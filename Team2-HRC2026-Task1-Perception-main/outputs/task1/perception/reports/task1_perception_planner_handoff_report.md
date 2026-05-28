# Báo Cáo Bàn Giao Perception Cho Planner Và Motion

## Interface tốt nhất ở thời điểm hiện tại

Sử dụng file:

`outputs/task1/perception/json/perception_interface_semantic_pose_base.json`

File này được tạo bằng lệnh:

```bash
cd /workspace/GlobalHumanoidRobotChallenge_2026_Baseline
/isaac-sim/python.sh /home/ubuntu/Team2/Task1-Perception/scripts/run_task1_semantic_pose_pipeline.py \
  --config-path /workspace/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim/config/Part_Sorting.yaml \
  --headless \
  --seed 20260521
```

## Kết quả runtime đã xác nhận

- `status`: `pass`
- `camera`: `head_left`
- `resolution`: `512x512`
- `seed`: `20260521`
- `detected_objects`: `4`
- `class_split`: `2 part_a`, `2 part_b`
- `pose_base_available`: `yes`
- `orientation_available`: `yes`
- `yaw_available`: `yes`
- `pose_base.frame`: `/Root/Ref_Xform/Ref`
- `position unit`: `meters`
- `orientation quaternion convention`: `xyzw`
- `yaw unit`: `radians`

## Contract cho Planner/Motion

Planner hoặc Motion nên đọc các field sau:

- `objects[*].class_id`
- `objects[*].bbox_xyxy`
- `objects[*].centroid_px`
- `objects[*].depth_median_m`
- `objects[*].pose_base.frame`
- `objects[*].pose_base.position_m`
- `objects[*].pose_base.orientation_xyzw`
- `objects[*].grasp_hint.yaw_rad`

Không nên dùng làm nguồn final:

- các output cũ theo hướng color-threshold
- giả định pixel centroid cố định giữa các lần chạy

## Nguồn gốc orientation và yaw

- `pose_base.orientation_xyzw` lấy từ runtime semantic object prim transform.
- `grasp_hint.yaw_rad` được tính từ `atan2` của trục local `+X` sau khi chiếu xuống mặt phẳng `XY` của base frame.
- RGB-D centroid và depth được giữ lại như dữ liệu đối chiếu, không phải nguồn pose chính cuối cùng.

## Kiểm soát reproducibility

Script runtime hiện ghi lại:

- Python random seed
- NumPy random seed
- Replicator global seed khi khả dụng
- baseline git commit/status
- `config_path`
- `command line`
- RGB cùng lần chạy
- depth `.npy` cùng lần chạy
- depth visualization
- overlay semantic pose

Lưu ý quan trọng:

Reproducibility ở đây là mức vận hành. Isaac Sim vẫn có thể sinh sai khác nhỏ theo GPU, driver và timing render/physics giữa các máy.

## File minh chứng nên đưa cho guide sau

- `outputs/task1/perception/overlays/overlay_semantic_pose_base_head_left.png`
- `outputs/task1/perception/samples/semantic_pose_rgb_head_left.png`
- `outputs/task1/perception/samples/semantic_pose_depth_head_left.npy`
- `outputs/task1/perception/samples/semantic_pose_depth_vis_head_left.png`
- `outputs/task1/perception/logs/semantic_pose_pipeline_log.json`
- `outputs/task1/perception/reports/semantic_pose_base_pipeline_report.md`

## Phần còn mở

- semantic annotation là nguồn class trong Isaac Sim, không phải detector production ngoài simulator
- pixel identity tuyệt đối giữa các máy: không cam kết
