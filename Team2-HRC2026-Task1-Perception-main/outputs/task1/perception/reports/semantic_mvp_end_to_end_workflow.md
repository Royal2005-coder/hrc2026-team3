# Quy Trình End-to-End Tạo Semantic MVP Cho Task 1 Perception

## Mục tiêu

Tài liệu này mô tả toàn bộ luồng kỹ thuật đã đi từ bước đầu tiên đến khi tạo được semantic MVP handoff hoàn chỉnh cho Task 1.

Mục tiêu của package này là xuất ra:

- `class_id`
- `bbox_xyxy`
- `centroid_px`
- `depth_median_m`
- `pose_base.position_m`
- `pose_base.orientation_xyzw`
- `grasp_hint.yaw_rad`
- `failure_reason` nếu object không hợp lệ

Output final downstream dùng là:

- `outputs/task1/perception/json/perception_interface_semantic_pose_base.json`

## Bước 0 - Tổ chức output ngay từ đầu

### Mục tiêu

Tách toàn bộ artifact perception ra khỏi baseline để:

- không sửa trực tiếp baseline
- dễ quản lý provenance
- dễ bàn giao cho guide sau

### Output gốc được gom vào

- `outputs/task1/perception/camera_inventory/`
- `outputs/task1/perception/json/`
- `outputs/task1/perception/logs/`
- `outputs/task1/perception/overlays/`
- `outputs/task1/perception/reports/`
- `outputs/task1/perception/samples/`

## Bước 1 - Scan baseline để xác định camera runtime

### Mục tiêu

Tìm chính xác baseline đang setup camera ở đâu và expose camera nào ra runtime API.

### Cách làm

- đọc `Ubtech_sim/source/RobotArticulation.py`
- xác minh hàm `_setup_cameras()`
- xác minh wrapper:
  - `get_camera_rgb`
  - `get_camera_depth`
  - `get_camera_rgbd`

### Kết quả

Đã xác minh 4 camera runtime:

- `head_left`
- `head_right`
- `wrist_left`
- `wrist_right`

Camera chính sau khi review sample runtime:

- `head_left`

Camera backup:

- `head_right`

### Artifact liên quan

- `camera_inventory/camera_inventory.csv`
- `reports/camera_geometry_note.md`
- `reports/rgbd_capture_result_note.md`

## Bước 2 - Capture RGB-D một lần bằng script riêng

### Mục tiêu

Lấy sample RGB-D từ 4 camera mà không sửa:

- `Ubtech_sim/main.py`
- `Ubtech_sim/source/DataLogger.py`
- `Ubtech_sim/source/RobotArticulation.py`

### Script dùng

- `scripts/capture_task1_rgbd_once.py`

### Lệnh chạy

```bash
cd /workspace/GlobalHumanoidRobotChallenge_2026_Baseline
/isaac-sim/python.sh /home/ubuntu/Team2/Task1-Perception/scripts/capture_task1_rgbd_once.py \
  --headless \
  --camera-width 512 \
  --camera-height 512 \
  --render-steps 20
```

### Vì sao phải có script riêng

- `main.py` có vòng lặp runtime dài
- logger mặc định không lưu depth
- output của baseline không nằm trong project perception
- script một lần giúp kiểm tra readiness nhanh, ít rủi ro hơn

## Bước 3 - Tăng resolution từ 128x128 lên 512x512

### Vấn đề ban đầu

Sample cũ ở mức `128x128` quá nhỏ, không phù hợp để:

- kiểm tra object rõ ràng
- so sánh depth
- dùng làm evidence cho downstream review

### Cách tăng resolution

Không tăng bằng cửa sổ GUI.

Thay vào đó, script trực tiếp yêu cầu camera runtime đổi resolution:

```python
camera.set_resolution((int(width), int(height)))
```

Trong `scripts/capture_task1_rgbd_once.py`:

- `--camera-width` mặc định `512`
- `--camera-height` mặc định `512`

Trong `scripts/run_task1_semantic_pose_pipeline.py`:

- `--width` mặc định `512`
- `--height` mặc định `512`

### Ý nghĩa kỹ thuật

- `SimulationApp` window size chỉ là kích thước cửa sổ hiển thị
- camera sensor resolution mới quyết định ảnh runtime mà perception lấy được

## Bước 4 - Lưu RGB, depth và depth visualization

### RGB

- lưu dưới dạng `.png`
- dùng để người review nhìn trực tiếp ảnh màu

### depth

- lưu dưới dạng `.npy`
- đây là dữ liệu số gốc, cần cho pixel-to-3D

### depth visualization

- lưu dưới dạng `.png`
- chỉ để quan sát nhanh vùng gần/xa
- không dùng thay cho depth số thật

### Vì sao cần cả ba

- `RGB`: để nhìn object
- `depth .npy`: để tính toán
- `depth vis`: để kiểm tra nhanh bằng mắt

## Bước 5 - Trích camera geometry và base transform

### Mục tiêu

Chuẩn bị đủ thông tin để đổi từ pixel 2D sang tọa độ 3D trong base frame.

### Camera geometry cần có

- `fx`, `fy`, `cx`, `cy`
- `T_world_camera`
- `T_base_world`

### Artifact

- `camera_inventory/camera_geometry.json`
- `camera_inventory/base_transform_runtime.json`
- `camera_inventory/camera_config_sheet.csv`

### Giải thích

- `fx`, `fy`: tiêu cự quy đổi theo pixel
- `cx`, `cy`: tâm ảnh
- `T_world_camera`: biến đổi từ camera frame sang world frame
- `T_base_world`: biến đổi từ world frame sang base frame

## Bước 6 - Dùng semantic annotation để detect/classify

### Mục tiêu

Tạo object list ổn định trong simulator mà không phụ thuộc màu sắc.

### Script dùng

- `scripts/run_task1_semantic_pose_pipeline.py`

### Cách làm

Script gắn annotator:

- `bounding_box_2d_tight_fast`

với semantic type:

- `class`

Sau đó lọc object thuộc:

- `part_a`
- `part_b`

### Vì sao chọn semantic trước

- random màu mỗi episode làm color-threshold không ổn định
- raw vendor dataset chưa có nhãn detector
- semantic annotation trong simulator cho class source đáng tin để hoàn thành MVP nhanh

## Bước 7 - Tạo bbox và centroid

### bbox_xyxy là gì

Là bounding box theo dạng:

- `x1, y1, x2, y2`

Trong đó:

- `x1, y1`: góc trên bên trái
- `x2, y2`: góc dưới bên phải

### centroid_px là gì

Là tâm 2D của bbox:

```text
centroid_x = (x1 + x2) / 2
centroid_y = (y1 + y2) / 2
```

### Vì sao cần centroid

- đây là pixel đại diện để tra depth
- là điểm đầu vào cho bước back-project sang camera frame

## Bước 8 - Lấy depth median thay vì depth tại 1 pixel duy nhất

### depth_median_m là gì

Là giá trị depth trung vị quanh vùng object, đơn vị hiện đang dùng trong pipeline là `meters` theo stage/runtime evidence.

### Vì sao dùng median

- depth tại đúng 1 pixel có thể nhiễu hoặc invalid
- median trên vùng nhỏ ổn định hơn

### Ý nghĩa

`depth_median_m` là khoảng cách đại diện từ camera tới object tại vùng centroid/object.

## Bước 9 - Từ pixel sang 3D camera frame

### Công thức back-project

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth
```

Trong đó:

- `u, v`: pixel 2D
- `fx, fy, cx, cy`: intrinsics
- `Z`: depth
- `X, Y, Z`: tọa độ 3D trong camera frame kiểu OpenCV

## Bước 10 - Đổi camera OpenCV frame sang USD camera frame

### Vì sao cần đổi

Quy ước trục camera của OpenCV và USD/Isaac không giống nhau.

Nếu không đổi, điểm 3D sẽ sai hướng hoặc lật trục.

### Script đang làm

- tạo `point_camera_cv`
- đổi sang `point_camera_usd`

## Bước 11 - Đổi từ camera sang world, rồi sang base

### Transform chain

```text
p_camera_opencv
-> p_camera_usd
-> p_world
-> p_base
```

### Công thức đồng nhất

```text
p_base = T_base_world @ T_world_camera @ T_camera_usd_from_opencv @ p_camera_opencv_h
```

### Ý nghĩa từng thành phần

- `T_camera_usd_from_opencv`: đổi quy ước trục camera
- `T_world_camera`: đưa điểm từ camera sang world
- `T_base_world`: đưa điểm từ world sang robot base

Base frame dùng trong package final là:

- `/Root/Ref_Xform/Ref`

## Bước 12 - Lấy orientation_xyzw và yaw_rad

### orientation_xyzw

Lấy từ semantic object prim transform ở runtime simulator.

Đây là orientation final trong base frame theo quy ước quaternion `xyzw`.

### yaw_rad

Lấy từ trục local `+X` của object sau khi chiếu xuống mặt phẳng `XY` của base frame.

Đơn vị là `radian`.

### Vì sao cần yaw_rad

- Planner/Motion thường không cần full 3D grasp orientation ngay từ đầu
- yaw quanh trục đứng là thông tin đủ hữu ích để tạo pre-grasp và hướng tiếp cận ban đầu

## Bước 13 - Tạo JSON final cho downstream

### Script sinh output final

- `scripts/run_task1_semantic_pose_pipeline.py`

### Lệnh chạy

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

### Output chính

- `json/perception_interface_semantic_pose_base.json`
- `logs/semantic_pose_pipeline_log.json`
- `overlays/overlay_semantic_pose_base_head_left.png`
- `overlays/overlay_semantic_bboxes_head_left.png`
- `samples/semantic_pose_rgb_head_left.png`
- `samples/semantic_pose_depth_head_left.npy`
- `samples/semantic_pose_depth_vis_head_left.png`

## Bước 14 - Validate package final

### Script validator

- `scripts/validate_task1_perception_final.py`

### Lệnh chạy

```bash
cd /home/ubuntu/Team2/Task1-Perception
python -m py_compile scripts/validate_task1_perception_final.py
python scripts/validate_task1_perception_final.py
python -m json.tool outputs/task1/perception/json/final_perception_validation.json
```

### Kết quả mong đợi

- `status = pass`
- `object_count = 4`
- `part_a_count = 2`
- `part_b_count = 2`
- `pose_base_available = true`
- `orientation_available = true`
- `yaw_available = true`

## Kết luận

Semantic MVP hiện tại đã đi đủ chuỗi:

```text
scan baseline
-> xác định camera runtime
-> capture RGB-D 512x512
-> trích geometry và transform
-> semantic bbox/class
-> centroid + depth median
-> pixel-to-3D
-> world-to-base
-> orientation + yaw
-> perception_interface_semantic_pose_base.json
-> validator pass
```

Đây là nhánh bàn giao chính thức hiện tại cho Planner, Motion và Evaluation trong môi trường Isaac Sim baseline.
