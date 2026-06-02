# Ghi Chú Camera Config Sheet

## Đường dẫn

- `outputs/task1/perception/camera_inventory/camera_config_sheet.csv`

## Mục đích

- Gom lại các artifact camera đã được verify cho Task 1 perception.
- Giữ rõ các trường còn `unknown`, tránh over-claim về geometry hoặc pose readiness.

## Nguồn dữ liệu

- `outputs/task1/perception/camera_inventory/camera_inventory.csv`
- `outputs/task1/perception/camera_inventory/camera_geometry.json`
- `outputs/task1/perception/logs/frame_capture_log.json`

## Ý nghĩa các cột quan trọng

- `selection_role`:
  - `primary_overview`: camera chính cho overview perception
  - `backup_overview`: camera dự phòng
  - `close_view_candidate`: camera close-view
- `intrinsics_status`:
  - hiện tại là `provisional_computed_from_usd_focal_aperture`
- `depth_unit`:
  - hiện tại chưa chốt cuối cùng
- `rgb_depth_alignment_status`:
  - mới ở mức sanity heuristic
- `T_base_camera_status`:
  - hiện tại chưa chốt thành final transform

## Quyết định hiện tại

- camera chính: `head_left`
- camera dự phòng: `head_right`
- wrist cameras khả dụng nhưng không phải lựa chọn đầu cho overview detection

## Giới hạn

- Không dùng riêng sheet này để claim `pose_base`.
- `depth_unit`, intrinsics cuối cùng và `T_base_camera` vẫn cần thêm bằng chứng nếu đi theo hướng hình học thuần RGB-D.
