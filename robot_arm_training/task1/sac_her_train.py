"""
SAC + HER pick-and-place training for HRC2026 Task 1.

Requirements (cài thêm, không có trong requirements.txt mặc định):
    pip install "stable-baselines3[extra]>=2.3" "gymnasium>=0.29"

Cách chạy:
    # Train (headless, 500k steps)
    python robot_arm_training/task1/sac_her_train.py --mode train --headless

    # Train nhiều hơn
    python robot_arm_training/task1/sac_her_train.py --mode train --headless --timesteps 1000000

    # Inference (có GUI)
    python robot_arm_training/task1/sac_her_train.py \\
        --mode infer \\
        --model-path robot_arm_training/task1/checkpoints/sac_her/sac_her_final

Lý do dùng SAC + HER:
    - SAC (Soft Actor-Critic): off-policy, sample-efficient, continuous action space
    - HER (Hindsight Experience Replay): giải quyết sparse reward bằng cách relabel
      goal của các episode thất bại → mỗi episode thất bại đều tạo ra signal học

Thiết kế:
    - observation  (38,): right arm joints + fingers + gripper + 4 obj poses
    - achieved_goal (12,): xyz của 4 objects hiện tại
    - desired_goal  (12,): xyz target trong hộp (cố định)
    - action        (10,): right arm joints + fingers + gripper, normalized [-1, 1]
    - reward: sparse (0 thành công, -1 thất bại) — HER handle phần còn lại
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ── Project paths ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src", "baseline_source"))
sys.path.insert(0, os.path.join(_HERE, "..", "shared"))
sys.path.insert(0, _HERE)

# ── SB3 + Gymnasium ───────────────────────────────────────────────────────────
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import SAC
    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
    from stable_baselines3.common.callbacks import CheckpointCallback
except ImportError as e:
    raise ImportError(
        f"{e}\n\nCài dependencies:\n"
        '    pip install "stable-baselines3[extra]>=2.3" "gymnasium>=0.29"'
    ) from e

# ── Task 1 constants ──────────────────────────────────────────────────────────
STATE_DIM  = 38
ACTION_DIM = 10
GOAL_DIM   = 12   # 4 objects × (x, y, z)

# Right arm joint range (radian) — conservative ±2.5 rad cho S2
R_ARM_MIN = np.full(7, -2.5, dtype=np.float32)
R_ARM_MAX = np.full(7,  2.5, dtype=np.float32)

# Right finger range (radian)
FINGER_MIN = np.zeros(2, dtype=np.float32)
FINGER_MAX = np.full(2, 0.04, dtype=np.float32)

# Box ở [1.2, 0.3, 1.05] — 4 target positions trải đều bên trong hộp
_BX, _BY, _BZ = 1.2, 0.3, 1.10
DESIRED_GOAL = np.array([
    _BX - 0.05, _BY - 0.05, _BZ,
    _BX + 0.05, _BY - 0.05, _BZ,
    _BX - 0.05, _BY + 0.05, _BZ,
    _BX + 0.05, _BY + 0.05, _BZ,
], dtype=np.float32)   # shape (12,)

GRIPPER_CLOSE_THRESHOLD = 0.0
SETTLE_STEPS = 20   # bước vật lý sau scatter để vật ổn định trên bàn


# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium GoalEnv
# ─────────────────────────────────────────────────────────────────────────────

class PickPlaceGoalEnv(gym.Env):
    """
    Gymnasium GoalEnv bọc Isaac Sim cho SAC+HER pick-and-place Task 1.

    Nhận các object Isaac Sim đã khởi tạo từ bên ngoài (train() / run_inference())
    để tránh vấn đề thứ tự import với SimulationApp.

    Observation space (Dict):
        observation   (38,): right arm joints (7) + fingers (2) + gripper (1) + obj poses (28)
        achieved_goal (12,): xyz của 4 objects
        desired_goal  (12,): xyz targets trong hộp (luôn là DESIRED_GOAL)

    Action space (10,) in [-1, 1]:
        [0:7]  right arm joints   → unnormalize to [R_ARM_MIN, R_ARM_MAX]
        [7:9]  right finger joints → unnormalize to [FINGER_MIN, FINGER_MAX]
        [9]    gripper cmd         → > 0 = close, <= 0 = open
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        world,
        robot,
        scene_builder,
        max_steps: int = 500,
        success_threshold: float = 0.08,
        physics_steps_per_control: int = 3,
    ):
        super().__init__()
        self._world   = world
        self._robot   = robot
        self._scene   = scene_builder
        self._max_steps  = max_steps
        self._threshold  = success_threshold
        self._phys_steps = physics_steps_per_control
        self._step_count = 0
        self._gripper_ctrl = -1.0   # -1 = open

        # ── Observation / action spaces ──────────────────────────────────────
        obs_low  = np.concatenate([R_ARM_MIN, FINGER_MIN, [-1.0], np.full(28, -5.0)])
        obs_high = np.concatenate([R_ARM_MAX, FINGER_MAX, [ 1.0], np.full(28,  5.0)])

        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(obs_low, obs_high, dtype=np.float32),
            "achieved_goal": spaces.Box(-5.0, 5.0, shape=(GOAL_DIM,), dtype=np.float32),
            "desired_goal":  spaces.Box(-5.0, 5.0, shape=(GOAL_DIM,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)

    # ── Reward (GoalEnv contract, called by HER with batched arrays) ──────────

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Any,
    ) -> np.ndarray:
        """Sparse: 0.0 khi tất cả 4 vật trong ngưỡng, -1.0 khi chưa."""
        ag = np.asarray(achieved_goal).reshape(-1, 4, 3)
        dg = np.asarray(desired_goal).reshape(-1, 4, 3)
        dist = np.linalg.norm(ag - dg, axis=-1)                    # (batch, 4)
        success = np.all(dist < self._threshold, axis=-1)          # (batch,)
        return (success.astype(np.float32) - 1.0).squeeze()        # 0 or -1

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        self._world.reset()
        # Scatter vật ngẫu nhiên sau reset, rồi step vật lý để vật ổn định
        self._scene.scatter_after_reset()
        for _ in range(SETTLE_STEPS):
            self._world.step(render=False)

        self._step_count   = 0
        self._gripper_ctrl = -1.0
        return self._get_obs(), {"is_success": False}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        self._apply_action(action)
        for _ in range(self._phys_steps):
            self._world.step(render=False)

        self._step_count += 1
        obs = self._get_obs()

        reward = float(
            self.compute_reward(obs["achieved_goal"], DESIRED_GOAL, {}).item()
            if obs["achieved_goal"].ndim == 1
            else self.compute_reward(obs["achieved_goal"], DESIRED_GOAL, {})
        )

        is_success = reward == 0.0
        terminated = is_success
        truncated  = self._step_count >= self._max_steps
        return obs, reward, terminated, truncated, {"is_success": is_success}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_obs(self) -> Dict[str, np.ndarray]:
        joint_states = self._robot.get_joint_states()
        if joint_states is None:
            # fallback khi sim chưa ready
            dummy = np.zeros(STATE_DIM, dtype=np.float32)
            return {
                "observation":   dummy,
                "achieved_goal": np.zeros(GOAL_DIM, dtype=np.float32),
                "desired_goal":  DESIRED_GOAL.copy(),
            }

        arm_pos    = np.array(joint_states["arm_positions"],    dtype=np.float32)  # (14,)
        finger_pos = np.array(joint_states["finger_positions"], dtype=np.float32)  # (4,)
        r_arm     = arm_pos[7:14]      # indices 7-13 = right arm
        r_fingers = finger_pos[2:4]    # indices 2-3  = right fingers

        part_poses = self._scene.get_parts_world_poses()
        obj_vec = np.zeros(28, dtype=np.float32)
        for i, part in enumerate(part_poses[:4]):
            pos = np.array(part["position"],    dtype=np.float32)  # (3,)
            ori = np.array(part["orientation"], dtype=np.float32)  # [qw, qx, qy, qz]
            qw, qx, qy, qz = ori
            obj_vec[i * 7: i * 7 + 7] = [pos[0], pos[1], pos[2], qx, qy, qz, qw]

        observation = np.concatenate([
            r_arm, r_fingers,
            np.array([self._gripper_ctrl], dtype=np.float32),
            obj_vec,
        ])
        achieved_goal = obj_vec.reshape(4, 7)[:, :3].flatten().astype(np.float32)

        return {
            "observation":   observation,
            "achieved_goal": achieved_goal,
            "desired_goal":  DESIRED_GOAL.copy(),
        }

    def _apply_action(self, action: np.ndarray):
        # Unnormalize: [-1, 1] → actual joint ranges
        r_arm = (action[:7] + 1.0) / 2.0 * (R_ARM_MAX - R_ARM_MIN) + R_ARM_MIN
        r_fingers = (action[7:9] + 1.0) / 2.0 * (FINGER_MAX - FINGER_MIN) + FINGER_MIN

        joint_states = self._robot.get_joint_states()
        if joint_states is None:
            return
        l_arm     = np.array(joint_states["arm_positions"][:7],    dtype=np.float32)
        l_fingers = np.array(joint_states["finger_positions"][:2], dtype=np.float32)

        self._robot.set_arm_joint_positions(
            np.concatenate([l_arm, r_arm]).tolist()
        )
        self._robot.set_finger_positions(
            np.concatenate([l_fingers, r_fingers]).tolist()
        )

        self._gripper_ctrl = float(action[9])
        if action[9] > GRIPPER_CLOSE_THRESHOLD:
            self._robot.close_gripper(side="right")
        else:
            self._robot.open_gripper(side="right")


# ─────────────────────────────────────────────────────────────────────────────
# Isaac Sim bootstrap (gọi trước khi tạo env)
# ─────────────────────────────────────────────────────────────────────────────

def _init_isaac(config_path: str, headless: bool):
    """Khởi tạo Isaac Sim và trả về (sim_app, world, robot, scene_builder)."""
    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": headless, "anti_aliasing": 0})

    # Import omni sau khi SimulationApp đã ready
    import yaml
    from omni.isaac.core import World
    from SceneBuilder import SceneBuilder
    from isaac_sim_robot_interface import IsaacSimRobotInterface

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    world = World(physics_dt=1 / 60.0, rendering_dt=1 / 20.0)
    world.scene.add_default_ground_plane()

    scene_builder = SceneBuilder(cfg, world)
    scene_builder.build_all()

    robot = IsaacSimRobotInterface(cfg, world)
    world.reset()

    return sim_app, world, robot, scene_builder


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def train(
    config_path:     str = "configs/Part_Sorting.yaml",
    total_timesteps: int = 500_000,
    save_dir:        str = "robot_arm_training/task1/checkpoints/sac_her",
    headless:        bool = True,
    seed:            int = 42,
):
    """
    Chạy SAC+HER training trong Isaac Sim.

    Gợi ý timesteps:
        500k   → ~vài tiếng, model bắt đầu học di chuyển
        1–2M   → model học tiếp cận vật
        5M+    → model có thể gắp và đặt được

    Checkpoints lưu mỗi 10k steps ở save_dir/sac_her_XXXXX_steps.zip
    TensorBoard logs ở save_dir/tb_logs/ — xem bằng:
        tensorboard --logdir robot_arm_training/task1/checkpoints/sac_her/tb_logs
    """
    os.makedirs(save_dir, exist_ok=True)

    sim_app, world, robot, scene = _init_isaac(config_path, headless)

    env = PickPlaceGoalEnv(
        world=world,
        robot=robot,
        scene_builder=scene,
        max_steps=500,
        success_threshold=0.08,
        physics_steps_per_control=3,
    )

    model = SAC(
        policy="MultiInputPolicy",   # xử lý Dict observation space
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,                   # 4 HER goals per real transition
            goal_selection_strategy="future",   # relabel bằng các state tương lai trong ep
        ),
        verbose=1,
        seed=seed,
        # ── SAC hyperparameters ──────────────────────────────────────────────
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=2_000,       # random exploration trước khi update
        batch_size=256,
        tau=0.005,                   # soft update target network
        gamma=0.98,                  # discount factor (cao để khuyến khích long-term)
        train_freq=1,
        gradient_steps=1,
        # ── Network architecture ─────────────────────────────────────────────
        policy_kwargs=dict(
            net_arch=[256, 256, 256],
            n_critics=2,
        ),
        tensorboard_log=os.path.join(save_dir, "tb_logs"),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=save_dir,
        name_prefix="sac_her",
        verbose=1,
    )

    print(f"[SAC+HER] Starting training — {total_timesteps:,} steps")
    print(f"[SAC+HER] Checkpoints → {save_dir}")
    print(f"[SAC+HER] TensorBoard → tensorboard --logdir {os.path.join(save_dir, 'tb_logs')}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    final_path = os.path.join(save_dir, "sac_her_final")
    model.save(final_path)
    print(f"[SAC+HER] Done. Final model → {final_path}.zip")
    sim_app.close()


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model_path:  str,
    config_path: str = "configs/Part_Sorting.yaml",
    n_episodes:  int = 10,
    headless:    bool = False,
):
    """Chạy model đã train, in kết quả từng episode."""
    sim_app, world, robot, scene = _init_isaac(config_path, headless)
    env = PickPlaceGoalEnv(world=world, robot=robot, scene_builder=scene)

    model = SAC.load(model_path, env=env)
    print(f"[SAC+HER] Loaded model from {model_path}")

    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        ok = info.get("is_success", False)
        successes += int(ok)
        print(
            f"  ep {ep + 1:2d}/{n_episodes} | "
            f"steps={steps:3d} | reward={total_reward:.1f} | "
            f"{'SUCCESS ✓' if ok else 'fail'}"
        )

    print(f"\nSuccess rate: {successes}/{n_episodes} = {100 * successes / n_episodes:.0f}%")
    sim_app.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="SAC+HER pick-and-place — HRC2026 Task 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",       choices=["train", "infer"], default="train")
    p.add_argument("--config",     default="configs/Part_Sorting.yaml")
    p.add_argument("--timesteps",  type=int, default=500_000)
    p.add_argument("--save-dir",   default="robot_arm_training/task1/checkpoints/sac_her")
    p.add_argument("--model-path", default=None,
                   help="Path to .zip checkpoint (required for --mode infer)")
    p.add_argument("--episodes",   type=int, default=10,
                   help="Number of episodes to run in infer mode")
    p.add_argument("--headless",   action="store_true", default=False)
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    if args.mode == "train":
        train(
            config_path=args.config,
            total_timesteps=args.timesteps,
            save_dir=args.save_dir,
            headless=args.headless,
            seed=args.seed,
        )
    else:
        if not args.model_path:
            p.error("--model-path is required for --mode infer")
        run_inference(
            model_path=args.model_path,
            config_path=args.config,
            n_episodes=args.episodes,
            headless=args.headless,
        )
