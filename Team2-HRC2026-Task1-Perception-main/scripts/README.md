# Scripts

## Mục tiêu

Thư mục này chứa các script cần để chạy lại semantic perception pipeline và kiểm tra output final.

Các script trong repo công khai này chỉ phục vụ nhánh semantic chính.

## Script chính

- `run_task1_semantic_pose_pipeline.py`
  - dựng scene Task 1 trong Isaac Sim
  - capture RGB-D từ `head_left`
  - lấy semantic bounding box/class
  - xuất `perception_interface_semantic_pose_base.json`

- `validate_task1_perception_final.py`
  - kiểm tra offline JSON final và artifact cần thiết
  - không khởi động Isaac Sim

- `capture_task1_rgbd_once.py`
  - capture RGB-D sample từ 4 camera runtime
  - dùng cho bước camera readiness

- `inspect_task1_camera_geometry.py`
  - ghi camera intrinsics và camera pose runtime

- `inspect_task1_base_transform_runtime.py`
  - ghi base transform dùng cho chuỗi transform sang `pose_base`

- `check_depth_unit_evidence.py`
  - kiểm tra evidence liên quan depth unit

- `check_head_left_depth_intrinsics_sanity.py`
- `check_head_left_rgb_depth_alignment.py`
- `pixel_to_camera_point_sanity.py`
  - các sanity check trước khi xuất pose

## Quy tắc sử dụng

- muốn chạy lại output final: dùng `run_task1_semantic_pose_pipeline.py`
- muốn kiểm tra package trước khi bàn giao: dùng `validate_task1_perception_final.py`
- muốn đọc nhanh trình tự chạy: xem `outputs/task1/perception/reports/semantic_mvp_end_to_end_workflow.md`
