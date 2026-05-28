# Báo Cáo Kiểm Tra Final Perception

## Trạng thái
- `status`: `pass`
- `timestamp`: `2026-05-22T07:56:14.482039+00:00`
- `interface`: `/home/ubuntu/Team2/Task1-Perception/outputs/task1/perception/json/perception_interface_semantic_pose_base.json`
- `camera`: `head_left`
- `base_frame`: `/Root/Ref_Xform/Ref`
- `object_count`: `4`
- `part_a_count`: `2`
- `part_b_count`: `2`
- `pose_base_available`: `True`
- `orientation_available`: `True`
- `yaw_available`: `True`
- `seed`: `20260521`

## Lỗi
- không có

## Cảnh báo
- không có

## Ghi chú kỹ thuật
- Đây là bước kiểm tra offline, không khởi động Isaac Sim.
- Mục tiêu của validator là kiểm tra tính nhất quán của package final trước khi bàn giao cho Planner, Motion và Evaluation.
- `orientation_xyzw` được kiểm tra theo quy ước `xyzw`.
- `grasp_hint.yaw_rad` được kiểm tra theo đơn vị `radian`.
- Quy ước `yaw` hiện tại là trục local `+X` của object sau khi chiếu xuống mặt phẳng `XY` của base frame.
- Reproducibility ở mức pixel tuyệt đối giữa các máy vẫn phụ thuộc vào Isaac Sim, GPU, driver và timing của physics/render.
