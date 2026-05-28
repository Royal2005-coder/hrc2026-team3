# HRC2026 Task 1 - Perception

Repo này chứa phần `Task 1 Perception` cho bài `Precise Desktop Sorting of Workpieces`.

Mục tiêu của module là lấy dữ liệu camera RGB-D trong Isaac Sim, nhận diện các workpiece bằng semantic annotation của simulator, và xuất interface cho Planner/Motion.

## Output chính

File Planner/Motion cần đọc:

```text
outputs/task1/perception/json/perception_interface_semantic_pose_base.json
```

Nội dung chính:

- `class_id`: `part_a` hoặc `part_b`
- `bbox_xyxy`: bounding box trên ảnh
- `centroid_px`: tâm object trên ảnh
- `depth_median_m`: depth đại diện
- `pose_base.position_m`: vị trí object trong base frame
- `pose_base.orientation_xyzw`: hướng object theo quaternion `xyzw`
- `grasp_hint.yaw_rad`: yaw hint cho thao tác gắp
- `failure_reason`: lý do lỗi nếu object không hợp lệ

## Cấu trúc repo

```text
scripts/
outputs/task1/perception/
```

`scripts/` chứa script chạy pipeline và validator.

`outputs/task1/perception/` chứa output đã kiểm chứng, log, overlay, sample RGB-D và report kỹ thuật.

## Cách chạy lại semantic pipeline

Chạy từ baseline repo:

```bash
cd /workspace/GlobalHumanoidRobotChallenge_2026_Baseline
/isaac-sim/python.sh /home/ubuntu/Team2/Task1-Perception/scripts/run_task1_semantic_pose_pipeline.py \
  --config-path /workspace/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim/config/Part_Sorting.yaml \
  --headless \
  --seed 20260521 \
  --width 512 \
  --height 512 \
  --render-steps 35
```
cd /workspace/GlobalHumanoidRobotChallenge_2026_Baseline
vào baseline repo để script import đúng scene, config và module của Isaac Sim baseline

/isaac-sim/python.sh
dùng Python runtime của Isaac Sim, không dùng Python hệ thống

scripts/run_task1_semantic_pose_pipeline.py
script chạy pipeline perception semantic chính

--config-path /workspace/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim/config/Part_Sorting.yaml
đường dẫn tới file config của task Part Sorting, chứa thông tin scene và tham số chạy

--headless
chạy không mở GUI, phù hợp server và tránh phụ thuộc màn hình

--seed 20260521
cố định seed để kết quả ổn định hơn giữa các lần chạy

--width 512 --height 512
đặt độ phân giải camera runtime là 512x512

--render-steps 35
cho scene render thêm 35 bước trước khi chụp và trích output, để hình và depth ổn định hơn

## Cách validate offline

Chạy từ repo này:

```bash
cd /home/ubuntu/Team2/Task1-Perception (Thay bằng đường dẫn của team)
python scripts/validate_task1_perception_final.py
python -m json.tool outputs/task1/perception/json/final_perception_validation.json
```

Kỳ vọng:

- `status = pass`
- `object_count = 4`
- `part_a_count = 2`
- `part_b_count = 2`
- `pose_base_available = true`
- `orientation_available = true`
- `yaw_available = true`

## Tài liệu nên đọc

1. `outputs/task1/perception/README.md`
2. `outputs/task1/perception/reports/reading_order.md`
3. `outputs/task1/perception/reports/semantic_mvp_end_to_end_workflow.md`
4. `outputs/task1/perception/reports/task1_perception_reproducibility_runbook.md`
5. `outputs/task1/perception/reports/task1_perception_planner_handoff_report.md`

## Phạm vi hiện tại

Pipeline hiện tại dùng semantic annotation của Isaac Sim để tạo output ổn định cho Task 1 trong simulator baseline.

Repo này không claim là detector production ngoài simulator.
## Lưu ý
Phải thay các đường dẫn tuyệt đối thành đường dẫn tuyệt đối trong server của mỗi team để tránh xung đột location
