# Ghi Chú Kết Quả Runtime Capture RGB-D

## Timestamp

`2026-05-20T08:01:28.758908+00:00`

## Script được kiểm tra

`/home/ubuntu/Team2/Task1-Perception/scripts/capture_task1_rgbd_once.py`

## Tóm tắt runtime

Lệnh chạy thành công:

```bash
cd /workspace/GlobalHumanoidRobotChallenge_2026_Baseline
/isaac-sim/python.sh /home/ubuntu/Team2/Task1-Perception/scripts/capture_task1_rgbd_once.py \
  --config-path /workspace/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim/config/Part_Sorting.yaml \
  --headless \
  --camera-width 512 \
  --camera-height 512 \
  --render-steps 20
```

Kết quả:

- Isaac Sim startup: `pass`
- scene build: `pass`
- robot build và camera setup: `pass`
- resolution runtime đổi từ `128x128` lên `512x512`
- RGB-D capture cho 4 camera runtime: `pass`

## Output chính

Thư mục sample:

- `outputs/task1/perception/samples/`

Log JSON:

- `outputs/task1/perception/logs/frame_capture_log.json`

## Kết quả theo camera

| camera_name | RGB | Depth | rgb_shape | depth_shape | depth_dtype | depth_min | depth_median | depth_max | valid_ratio | đánh giá nhanh |
|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| `head_left` | pass | pass | `[512, 512, 3]` | `[512, 512]` | `float32` | 0.2464 | 2.7270 | 15.7431 | 0.999989 | góc nhìn tổng quan tốt nhất, ứng viên chính |
| `head_right` | pass | pass | `[512, 512, 3]` | `[512, 512]` | `float32` | 0.2476 | 2.7269 | 15.7431 | 0.999989 | góc nhìn tổng quan tốt, ứng viên dự phòng |
| `wrist_left` | pass | pass | `[512, 512, 3]` | `[512, 512]` | `float32` | 0.0562 | 8.2184 | 17.7464 | 1.000000 | chủ yếu thấy gripper và near field |
| `wrist_right` | pass | pass | `[512, 512, 3]` | `[512, 512]` | `float32` | 0.0563 | 7.5104 | 17.7409 | 1.000000 | chủ yếu thấy gripper và near field |

## Quyết định hiện tại

Ứng viên overview perception:

1. `head_left`
2. `head_right`

Wrist cameras khả dụng, nhưng không phải lựa chọn đầu cho object overview detection.
