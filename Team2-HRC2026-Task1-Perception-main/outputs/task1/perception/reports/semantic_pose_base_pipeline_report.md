# Báo Cáo Pipeline Semantic Pose Base

## Trạng thái chạy

- `timestamp`: `2026-05-21T02:46:16.775774+00:00`
- `runtime_success`: `True`
- `failure_reason`: `None`
- `seed`: `20260521`
- `camera_name`: `head_left`
- `total_objects`: `4`
- `part_a_count`: `2`
- `part_b_count`: `2`
- `four_object_pass`: `True`
- `pose_base_available`: `True`
- `orientation_available`: `True`
- `yaw_available`: `True`

## Output sinh ra

- `perception_interface`: `/home/ubuntu/Team2/Task1-Perception/outputs/task1/perception/json/perception_interface_semantic_pose_base.json`
- `runtime_log`: `/home/ubuntu/Team2/Task1-Perception/outputs/task1/perception/logs/semantic_pose_pipeline_log.json`
- `overlay`: `/home/ubuntu/Team2/Task1-Perception/outputs/task1/perception/overlays/overlay_semantic_pose_base_head_left.png`
- `rgb_same_run`: `/home/ubuntu/Team2/Task1-Perception/outputs/task1/perception/samples/semantic_pose_rgb_head_left.png`
- `depth_same_run`: `/home/ubuntu/Team2/Task1-Perception/outputs/task1/perception/samples/semantic_pose_depth_head_left.npy`
- `depth_vis_same_run`: `/home/ubuntu/Team2/Task1-Perception/outputs/task1/perception/samples/semantic_pose_depth_vis_head_left.png`

## Kết quả detect

| idx | class_id | bbox_xyxy | centroid_px | depth_median_m | pose_base_position_m | orientation_xyzw | yaw_rad |
|---:|---|---|---|---:|---|---|---:|
| 0 | `part_a` | `[291, 323, 305, 338]` | `[298.0, 330.5]` | 0.6778058409690857 | `[ 0.58140842 -0.09570717  0.11134551]` | `[-0.6465827477872981, 0.2862346156393701, -0.2870364568688184, 0.6462279532036698]` | -0.8347538254761588 |
| 1 | `part_a` | `[288, 338, 300, 355]` | `[294.0, 346.5]` | 0.6190624237060547 | `[ 0.52225952 -0.089406    0.13361654]` | `[0.3419777124358208, 0.6150953230242305, -0.3510417093167732, 0.6176396247900866]` | -1.8092814042883714 |
| 2 | `part_b` | `[232, 349, 248, 373]` | `[240.0, 361.0]` | 0.588841438293457 | `[0.48224292 0.07563741 0.11538157]` | `[-0.06074671561464545, -0.019732168301973444, -0.715006554899777, 0.6961940135669712]` | -1.5941994073106363 |
| 3 | `part_b` | `[310, 365, 332, 389]` | `[321.0, 377.0]` | 0.5467031002044678 | `[ 0.42774345 -0.13953166  0.13454006]` | `[0.6057616543912454, -0.36445622702102887, -0.45987195950178783, 0.5373474272002835]` | -1.2495692638502005 |

## Ghi chú về reproducibility

- Script có ghi lại `command`, `seed`, `config_path`, baseline git info, camera path và artifact RGB/depth cùng lần chạy.
- Script có set seed cho Python, NumPy và Replicator khi API cho phép.
- Reproducibility mục tiêu ở đây là reproducibility vận hành: cùng lệnh, cùng config, cùng schema output, cùng metadata truy vết.
- Không cam kết pixel giống tuyệt đối giữa các máy vì Isaac Sim còn phụ thuộc GPU, driver và timing runtime.

## Hợp đồng với Planner/Motion

- Dùng `objects[*].pose_base.frame == /Root/Ref_Xform/Ref`
- `position_m` theo đơn vị `meter`
- `orientation_xyzw` theo quy ước `xyzw`
- `grasp_hint.yaw_rad` là góc của trục local `+X` của object sau khi chiếu xuống mặt phẳng `XY` của base frame
- Với giai đoạn hiện tại, ưu tiên interface semantic này thay cho các JSON cũ theo hướng color-threshold
