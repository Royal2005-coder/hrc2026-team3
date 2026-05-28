# Ghi Chú Camera Geometry

## Trạng thái

- `timestamp`: `2026-05-20T08:09:18.047536+00:00`
- `baseline_path`: `/workspace/GlobalHumanoidRobotChallenge_2026_Baseline`
- `config_path`: `/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim/config/Part_Sorting.yaml`
- `scene_path`: `/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/assets/resources/Collected_Task4/SubUSDs/2_small_warehouse2.usd`
- `runtime_success`: `True`
- `failure_reason`: `None`

## Tóm tắt camera

| camera_name | prim_valid | resolution | intrinsics_status | world_pose | depth_unit | T_base_camera |
|---|---:|---|---|---|---|---|
| `head_left` | True | `[512, 512]` | `computed_from_usd_focal_aperture_assuming_center_principal_point` | `available` | `unknown` | `unknown` |
| `head_right` | True | `[512, 512]` | `computed_from_usd_focal_aperture_assuming_center_principal_point` | `available` | `unknown` | `unknown` |
| `wrist_left` | True | `[512, 512]` | `computed_from_usd_focal_aperture_assuming_center_principal_point` | `available` | `unknown` | `unknown` |
| `wrist_right` | True | `[512, 512]` | `computed_from_usd_focal_aperture_assuming_center_principal_point` | `available` | `unknown` | `unknown` |

## Ghi chú kỹ thuật

- Báo cáo này chưa claim `pose_base`.
- `T_world_camera` có thể đọc được, nhưng `T_base_camera` chưa được chốt thành final transform cho pipeline thuần hình học.
- Intrinsics hiện là giá trị suy ra từ USD focal/aperture, chưa phải projection validation đầy đủ.
- `depth_unit` vẫn cần đối chiếu thêm nếu muốn dùng cho pipeline ngoài semantic final.
