# Báo Cáo Semantic Bounding Boxes Của Head Left

## Trạng thái

- `timestamp`: `2026-05-20T11:48:19.057752+00:00`
- `runtime_success`: `True`
- `failure_reason`: `None`
- `camera_name`: `head_left`
- `total_boxes`: `4`
- `part_a_count`: `2`
- `part_b_count`: `2`
- `unknown_count`: `42`
- `four_object_semantic_pass`: `True`

## Output

- `json`: `/home/ubuntu/Team2/Task1-Perception/outputs/task1/perception/json/semantic_bboxes_head_left.json`
- `overlay`: `/home/ubuntu/Team2/Task1-Perception/outputs/task1/perception/overlays/overlay_semantic_bboxes_head_left.png`
- `semantic_vis`: `/home/ubuntu/Team2/Task1-Perception/outputs/task1/perception/overlays/overlay_semantic_segmentation_head_left.png`

## Bounding boxes

| idx | class_id | semantic_id | bbox_xyxy | centroid_px |
|---:|---|---|---|---|
| 0 | `part_a` | `3` | `[236, 389, 253, 415]` | `[244.5, 402.0]` |
| 1 | `part_a` | `3` | `[261, 321, 274, 337]` | `[267.5, 329.0]` |
| 2 | `part_b` | `4` | `[294, 367, 319, 386]` | `[306.5, 376.5]` |
| 3 | `part_b` | `4` | `[298, 392, 322, 423]` | `[310.0, 407.5]` |

## Diễn giải

- Nếu báo cáo này giữ được `2 part_a` và `2 part_b`, semantic class evidence là đủ tốt để thay thế color hints trong nhánh final.
- Nếu lần chạy sau lệch số lượng, cần xem lại annotator metadata trong file JSON tương ứng.
