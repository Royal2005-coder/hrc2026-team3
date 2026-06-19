# ACT Training — Part Sorting (2026-06-16)

## Lý do chuyển từ SmolVLA sang ACT

SmolVLA (30K steps) thất bại vì MEAN_STD normalization kết hợp với dataset có left arm STD ≈ 0.001
(tay trái không di chuyển trong 756 episodes). Model cần predict -4.43σ để đạt góc tay thấp nhất
(-1.842 rad shoulder pitch) — gần như không thể với neural network.

**Fix chính**: Đổi sang ACT với **MIN_MAX normalization** cho ACTION và STATE (sửa trong
`src/lerobot/policies/act/configuration_act.py`). MIN_MAX cho phép model học full workspace range
từ dataset min/max thay vì bị giới hạn bởi standard deviation.

## Dataset

`Part_Sorting/part_sorting_long_756_episode/` (756 episodes, 580,608 frames, ~768 frames/episode)
- 4 cameras: head_left, head_right, wrist_left, wrist_right (480×640, 30fps)
- action dim: 20 (14 arm joints + 4 finger joints + 2 gripper commands)
- state dim: 48 (20 joint + 28 object poses)

## Lệnh training

```bash
cd /home/team3/hrc2026-team3
nohup /home/team3/Isaac-GR00T/.venv/bin/python -m src.lerobot.scripts.lerobot_train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --dataset.repo_id=team3/part_sorting_long_756_episode \
  --dataset.root=Part_Sorting/part_sorting_long_756_episode \
  --dataset.video_backend=pyav \
  --output_dir=outputs/train/act_part_sorting \
  --batch_size=32 \
  --steps=100000 \
  --save_freq=5000 \
  --eval_freq=0 \
  --log_freq=200 \
  --wandb.enable=false \
  > /tmp/act_train.log 2>&1 &
```

- Output: `outputs/train/act_part_sorting/checkpoints/`
- Log: `/tmp/act_train.log`
- ~1.3s/step, batch=32 → **100K steps ≈ 36 giờ**
- Checkpoint mỗi 5000 steps (~1.8 giờ)
- GPU: ~36GB/49GB

## Theo dõi log

```bash
# Xem loss
grep "step:" /tmp/act_train.log | tail -20

# Realtime
tail -f /tmp/act_train.log | grep "step:"
```

## Thông số ACT

- `chunk_size=50`: predict 50 bước tương lai (~1.67s ở 30fps) mỗi lần
- `n_action_steps=50`: thực thi toàn bộ 50 bước trước khi query lại
- `kl_weight=10`: VAE KL loss weight (default)
- `use_vae=True`: dùng VAE (CVAE architecture của ACT gốc)
- Architecture: ResNet18 vision + Transformer encoder-decoder (52M params)

## Loss ban đầu

| step | epoch | loss   |
|------|-------|--------|
| 200  | 0.01  | 4.420  |
