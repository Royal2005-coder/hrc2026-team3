# Task 1 - Perception

## Mục tiêu

Thư mục này chứa output và tài liệu kỹ thuật của module `Task 1 - Perception`.

Pipeline hiện tại thực hiện:

- capture RGB-D từ camera runtime
- lấy semantic bounding box và class trong Isaac Sim
- tạo `centroid_px` và `depth_median_m`
- đổi dữ liệu camera sang `pose_base`
- xuất JSON cho Planner/Motion

## Cấu trúc thư mục

- `camera_inventory/`
- `json/`
- `logs/`
- `overlays/`
- `reports/`
- `samples/`

### `camera_inventory/`

Thông tin camera, camera geometry, base transform và camera config sheet.

### `json/`

JSON output chính và JSON kiểm tra final package.

### `logs/`

Log runtime để truy vết lệnh chạy, seed, camera và output.

### `overlays/`

Ảnh minh chứng bounding box, semantic segmentation và pose overlay.

### `reports/`

Tài liệu kỹ thuật, hướng dẫn chạy lại và handoff cho Planner/Motion.

### `samples/`

Sample RGB, depth `.npy` và depth visualization.

## File đầu ra chính thức

Planner/Motion/Evaluation nên đọc:

`json/perception_interface_semantic_pose_base.json`

## Cách chạy end to end

### Bước 1 - Chạy pipeline semantic

```bash
cd /workspace/GlobalHumanoidRobotChallenge_2026_Baseline
/isaac-sim/python.sh /home/ubuntu/Team2/Task1-Perception/scripts/run_task1_semantic_pose_pipeline.py \
  --config-path /workspace/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim/config/Part_Sorting.yaml \
  --headless \
  --seed 20260521
```

Giải thích:

- `cd` vào baseline để import đúng scene/config của Isaac Sim
- `/isaac-sim/python.sh` dùng Python runtime của Isaac Sim
- `run_task1_semantic_pose_pipeline.py` là script chính của semantic MVP
- `--headless` giúp chạy ổn định trên server, không phụ thuộc GUI
- `--seed` cố định scene để có reproducibility tốt hơn

Output chính sinh ra:

- `json/perception_interface_semantic_pose_base.json`
- `overlays/overlay_semantic_pose_base_head_left.png`
- `logs/semantic_pose_pipeline_log.json`

### Bước 2 - Validate offline

```bash
cd /home/ubuntu/Team2/Task1-Perception (Thay bằng đường dẫn của Team)
python scripts/validate_task1_perception_final.py
python -m json.tool outputs/task1/perception/json/final_perception_validation.json
```

Giải thích:

- validator không khởi động Isaac Sim
- bước này chỉ kiểm tra tính nhất quán của package final

Kết quả mong đợi:

- `status = pass`
- `object_count = 4`
- `part_a_count = 2`
- `part_b_count = 2`

### Bước 3 - Đọc giải thích end-to-end

Đọc:

- `reports/semantic_mvp_end_to_end_workflow.md`

Mục tiêu:

- hiểu từ scan baseline đến semantic MVP final
- hiểu cách tăng `128x128` lên `512x512`
- hiểu cách lấy `depth`, `centroid_px`, `orientation_xyzw`, `yaw_rad`

## Quy ước làm việc

- nếu cần file final để bàn giao: dùng `json/perception_interface_semantic_pose_base.json`
- nếu cần hiểu vai trò từng script: đọc `scripts/README.md`
