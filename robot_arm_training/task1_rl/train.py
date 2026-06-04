"""Train PPO cho Task 1 pick-and-place (Isaac Lab 0.54.2 + skrl 2.1.0).

Cách chạy:
    cd ~/hrc2026/hrc2026-team3
    python robot_arm_training/task1_rl/train.py --num_envs 512 --headless
    python robot_arm_training/task1_rl/train.py --num_envs 4096 --headless
    python robot_arm_training/task1_rl/train.py --num_envs 512 --headless --checkpoint ~/work/task1_ppo/run1/checkpoints/best_agent.pt

Logs và checkpoints ghi vào ~/work/task1_ppo/ (tránh IO lớn trên workspace).
"""

from __future__ import annotations

import argparse
import os
import sys

# ── BƯỚC 1: Khởi động Isaac Sim TRƯỚC KHI import bất kỳ thứ gì liên quan Isaac ─
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PPO train Task 1 pick-and-place")
parser.add_argument("--num_envs",   type=int,   default=512)
parser.add_argument("--max_steps",  type=int,   default=50_000_000)
parser.add_argument("--checkpoint", type=str,   default=None, help="Resume từ checkpoint path")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── BƯỚC 2: Import phần còn lại sau khi sim đã khởi động ────────────────────────
import torch
import torch.nn as nn

# Thêm thư mục task1_rl vào sys.path để import env_cfg, env
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_cfg import PickPlaceEnvCfg
from env import PickPlaceEnv

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer

OBS_DIM = 38
ACT_DIM = 10


# ── Model definitions ────────────────────────────────────────────────────────────

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=False)

        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 256), nn.ELU(),
            nn.Linear(256, 128),     nn.ELU(),
            nn.Linear(128, 64),      nn.ELU(),
        )
        self.mean_layer = nn.Linear(64, ACT_DIM)
        self.log_std    = nn.Parameter(torch.zeros(ACT_DIM))

    def compute(self, inputs, role=""):
        x = inputs["states"] if isinstance(inputs, dict) else inputs
        feat = self.net(x)
        mean = self.mean_layer(feat)
        log_std = self.log_std.expand(x.shape[0], -1)
        return mean, log_std, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 256), nn.ELU(),
            nn.Linear(256, 128),     nn.ELU(),
            nn.Linear(128, 64),      nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role=""):
        x = inputs["states"] if isinstance(inputs, dict) else inputs
        return self.net(x), {}


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    # ── Env ──────────────────────────────────────────────────────────────────
    env_cfg = PickPlaceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    raw_env = PickPlaceEnv(cfg=env_cfg)
    env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")

    device   = raw_env.device
    num_envs = args_cli.num_envs
    rollouts = 24  # steps per env per update

    print(f"[train] device={device}, num_envs={num_envs}, rollouts={rollouts}")
    print(f"[train] obs_space={env.observation_space}, act_space={env.action_space}")

    # ── Models ───────────────────────────────────────────────────────────────
    obs_space = env.observation_space
    act_space = env.action_space

    policy = Policy(obs_space, act_space, device)
    value  = Value(obs_space,  act_space, device)

    # ── Memory ───────────────────────────────────────────────────────────────
    memory = RandomMemory(memory_size=rollouts, num_envs=num_envs, device=device)

    # ── PPO config (skrl 2.x — pass dict trực tiếp, không dùng PPO_DEFAULT_CONFIG) ──
    ppo_cfg = {
        "rollouts":          rollouts,
        "learning_epochs":   5,
        "mini_batches":      4,
        "discount_factor":   0.99,
        "lambda_":           0.95,   # chú ý underscore cho skrl 2.x
        "learning_rate":     3e-4,
        "grad_norm_clip":    1.0,
        "ratio_clip":        0.2,
        "value_clip":        0.2,
        "clip_predicted_values": True,
        "entropy_loss_scale":    0.005,
        "value_loss_scale":      1.0,
        "state_preprocessor":        RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": OBS_DIM, "device": device},
        "value_preprocessor":        RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1,       "device": device},
        "experiment": {
            "directory":         os.path.expanduser("~/work/task1_ppo"),
            "experiment_name":   "run1",
            "write_interval":    500,
            "checkpoint_interval": 5000,
        },
    }

    agent = PPO(
        models={"policy": policy, "value": value},
        memory=memory,
        cfg=ppo_cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )

    # ── Resume từ checkpoint nếu có ───────────────────────────────────────────
    if args_cli.checkpoint:
        print(f"[train] Loading checkpoint: {args_cli.checkpoint}")
        agent.load(args_cli.checkpoint)

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = SequentialTrainer(
        cfg={"timesteps": args_cli.max_steps, "headless": True},
        env=env,
        agents=agent,
    )
    trainer.train()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
