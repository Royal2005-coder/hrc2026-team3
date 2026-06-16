# Báo Cáo Runtime Base Transform

## Trạng thái

- `timestamp`: `2026-05-20T11:12:12.178687+00:00`
- `runtime_success`: `True`
- `failure_reason`: `None`
- `base_prim_path`: `/Root/Ref_Xform/Ref`
- `headless`: `True`
- `Pinocchio/IK used`: `no`

## Kết quả so sánh

- `config_position_m`: `[0.7, -0.2, 0.9]`
- `runtime_position_m`: `[0.7, -0.2, 0.9]`
- `position_delta`: `[0.0, 0.0, 0.0]`
- `position_delta_norm_m`: `0.0`
- `comparison_status`: `pass`

## Diễn giải

- Báo cáo này xác minh `robot root/base prim` trong world frame bằng cách đọc trực tiếp USD transform ở runtime.
- Hướng này tránh dùng `CoordinateTransform.from_torso_link()` và Pinocchio, vốn đã từng gây crash trong môi trường hiện tại.
- Khi `comparison_status = pass`, base transform suy ra từ config phù hợp với base prim runtime ở mức vị trí.
- Frame downstream cần dùng chính xác là `/Root/Ref_Xform/Ref`.
