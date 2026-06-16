# Kế hoạch RL cho Pick-and-Place Task 1 — Isaac Lab + PPO

## Bối cảnh

Task 1: Robot humanoid S2 dual-arm gắp 4 vật (PartA/PartB) trải ngẫu nhiên trên bàn, đặt vào hộp.

Môi trường: Isaac Sim 5.1, server có GPU RTX 5880 Ada (50GB VRAM).

### Lưu ý về tài nguyên server

> Admin thông báo: mọi training log và process có **Disk IO lớn** phải ghi vào `~/work` thay vì workspace, sau đó backup lại. Lý do: nhiều project dùng chung tài nguyên disk.

Áp dụng cho toàn bộ code RL:
- Checkpoints → `~/work/sac_her/checkpoints/` (hoặc `~/work/<tên_run>/checkpoints/`)
- TensorBoard logs → `~/work/<tên_run>/checkpoints/tb_logs/`
- DataLogger CSV/HDF5 → `~/work/<tên_run>/logs/`
- Replay buffer (nếu lưu xuống) → `~/work/<tên_run>/`

Sau khi train xong → copy về workspace để backup:
```bash
cp -r ~/work/sac_her/ ~/hrc2026/hrc2026-team3/robot_arm_training/task1/checkpoints/
```

Đã thử:
- **IL (LSTM)**: Fail vì thiếu data (91 episodes, cần 300–500+)
- **SAC + HER (raw Isaac Sim)**: Đúng hướng nhưng chỉ chạy 1 env → chậm, HER cần thiết vì sparse reward

---

## Tại sao chuyển sang Isaac Lab + PPO

### Vấn đề của SAC + HER với raw Isaac Sim

| Vấn đề | Giải thích |
|---|---|
| 1 env duy nhất | Ít signal học, sparse reward rất khó hội tụ |
| Chậm | ~1–10 steps/giây → 500k steps mất vài tiếng |
| HER cần thiết | Vì sparse reward, phải relabel goals của episode thất bại |
| Không tận dụng GPU | Raw Isaac Sim không vectorize được |

### Lợi thế của Isaac Lab

| Tiêu chí | Raw Isaac Sim + SB3 | Isaac Lab + PPO |
|---|---|---|
| Số envs song song | 1 | 4096+ |
| Tốc độ | ~1–10 steps/s | ~100k+ steps/s |
| Sparse reward | Cần HER | Không cần — đủ env để có signal |
| GPU utilization | Thấp | Tối đa |
| Thời gian train | Vài ngày | Vài tiếng |

**Lý do HER không cần trong Isaac Lab**: Khi chạy 4096 envs song song, dù tỉ lệ thành công ngẫu nhiên chỉ 0.1%, mỗi batch vẫn có ~4 episode thành công → đủ reward signal tự nhiên.

---

## Thiết kế môi trường Isaac Lab

### Observation space (38-dim, giữ nguyên từ IL)

```
[0:7]   right arm joints (radian)
[7:9]   right finger joints (radian)
[9]     right gripper control (-1=open, +1=close)
[10:38] 4 objects × 7 = [x, y, z, qx, qy, qz, qw]
```

### Action space (10-dim, continuous [-1, 1])

```
[0:7]  right arm joint targets
[7:9]  right finger targets
[9]    gripper command (-1=open, +1=close)
```

Cánh tay trái cố định (không điều khiển).

### Reward function — Dense reward (không cần HER)

```python
reward = 0.0

# 1. Kéo tay phải đến gần vật gần nhất
dist_hand_to_obj = distance(tcp_position, nearest_object_position)
reward += -dist_hand_to_obj * 1.0

# 2. Kéo vật đến gần hộp (chỉ tính khi đang cầm vật)
if is_grasping:
    dist_obj_to_box = distance(grasped_object_position, box_center)
    reward += -dist_obj_to_box * 2.0

# 3. Thưởng khi gắp được vật
reward += 5.0 * is_grasped

# 4. Thưởng khi đặt vật vào hộp thành công
reward += 20.0 * n_objects_in_box

# 5. Phạt mỗi step (khuyến khích hoàn thành nhanh)
reward += -0.01
```

### Episode termination

- **Success**: tất cả 4 vật trong hộp (trong ngưỡng 8cm)
- **Timeout**: 500 steps
- **Reset**: vật rơi khỏi bàn (z < 0.9m)

---

## Cấu trúc file cần viết

```
robot_arm_training/
  task1_isaaclab/
    env.py          ← PickPlaceEnv(DirectRLEnv) — định nghĩa obs/act/reward
    env_cfg.py      ← PickPlaceEnvCfg — cấu hình scene, robot, physics
    train.py        ← entry point training (PPO via skrl hoặc rsl-rl)
    agent_cfg.py    ← PPO hyperparameters
```

### Framework RL nên dùng

Theo thứ tự ưu tiên:
1. **skrl** — hỗ trợ SAC, PPO, TD3; API đơn giản; tích hợp tốt với Isaac Lab
2. **RSL-RL** — PPO tối ưu cho Isaac Lab (dùng trong legged_gym, Isaac Lab tutorials)
3. **RL-Games** — PPO/SAC, phức tạp hơn

---

## Hyperparameters PPO gợi ý (starting point)

```yaml
# PPO
learning_rate: 3e-4
n_steps: 24          # steps per env per update
batch_size: 48384    # n_steps × n_envs / n_minibatches
n_epochs: 5
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
entropy_coef: 0.01
value_loss_coef: 1.0

# Network
policy_net: [256, 128, 64]
value_net:  [256, 128, 64]

# Env
n_envs: 4096
max_episode_steps: 500
```

---

## Thứ tự implement

1. `env_cfg.py` — scene config (robot, table, box, parts scatter area)
2. `env.py` — observation, action, reward, reset logic
3. `agent_cfg.py` — PPO config
4. `train.py` — main training loop
5. Test với 512 envs trước, sau đó scale lên 4096

---

## Thông tin kỹ thuật Isaac Sim / Scene

- Box position: `[1.2, 0.3, 1.05]` (world frame)
- Scatter area: center `[0.75, 0.28, 1.04]`, size `[0.30, 0.23]`
- Robot prim path: `/Root/Ref_Xform/Ref`
- Robot position: `[0.7, -0.2, 0.9]`, rotation: `[0, 0, 90]`
- Config file: `configs/Part_Sorting.yaml`
- SceneBuilder có `scatter_after_reset()` để re-scatter vật mỗi episode

## Tham khảo

- Isaac Lab docs: https://isaac-sim.github.io/IsaacLab
- skrl + Isaac Lab: https://skrl.readthedocs.io/en/latest/intro/examples.html
- Isaac Lab manipulation examples: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/`
