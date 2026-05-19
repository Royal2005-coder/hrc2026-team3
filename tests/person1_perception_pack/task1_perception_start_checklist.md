# Task 1 — Perception Start Checklist (Người 1)

**Người thực hiện:** Tài
**Ngày bắt đầu:** ____

## Thông tin scene (✓ từ Part_Sorting.yaml)

- [x] Scene USD: `Collected_Task4/SubUSDs/2_small_warehouse2.usd`
- [x] Robot USD: `Collected_s2_v1_ecbg/s2_v1.usd` tại (0.7, -0.2, 0.9) rotation (0,0,90)
- [ ] Camera prim path: `/Root/Ref_Xform/Ref/head_pitch_link/head_stereo_left/head_stereo_left_Camera_01`
- [ ] RGB resolution: 640×480 (xác nhận bằng capture)
- [ ] Depth resolution: 640×480 (xác nhận bằng capture)
- [ ] Depth unit: ☐ meter  ☐ millimeter  (xác nhận bằng capture_rgbd.py)
- [x] Part A: Task1_PartA.usd — 2 variants: ori_color, red
- [x] Part B: Part_B.usd — 2 variants: blue, ori_color
- [x] ⚠ CẢ HAI đều có ori_color → KHÔNG phân biệt chỉ bằng màu!
- [x] Scatter area: center (0.75, 0.28, 1.04), x:[0.50,0.80], y:[0.10,0.30]
- [x] Box (bin): (1.2, 0.3, 1.05) — cố định
- [ ] T_base_camera: chạy extract_camera_params.py

## Thông tin cần lấy từ Isaac Sim

- [ ] Chạy `scripts/extract_camera_params.py` → intrinsics + T_base_camera
- [ ] Chạy `scripts/capture_rgbd.py` → RGB + depth + sanity report
- [ ] Chạy `scripts/measure_shape_features.py` → shape thresholds cho A/B

### Người 2 — Planner/FSM
- [ ] Class names agreed: `part_A`, `part_B`
- [ ] Pose frame agreed: `pose_base` (robot-base frame)
- [ ] Confidence threshold agreed: 0.70
- [ ] Failure handling agreed: skip objects with `failure_reason != null`

### Người 3 — Motion Primitive
- [ ] Yaw precision requirement: _____ rad
- [ ] Need `grasp_width_m`: ☐ yes  ☐ no
- [ ] Default approach axis: `z_down`
- [ ] Pose target: ☐ object centre  ☐ object edge

### Người 4 — Evaluation
- [ ] Debug overlays format agreed
- [ ] Failure log format agreed (JSONL)

## Checklist trước khi bắt đầu code

- [ ] Isaac Sim mở được, scene load được
- [ ] Camera nhìn thấy bàn và workpieces
- [ ] Biết cách chạy baseline script
- [ ] Đã tạo thư mục `lab_outputs/perception/`
- [ ] Đã đọc hết guide Người 1 v1.0
