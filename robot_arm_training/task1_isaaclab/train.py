"""
Training entry point: PPO + skrl cho Task 1 pick-and-place với Isaac Lab.

Cách chạy — dùng isaaclab.sh (recommended) hoặc python trực tiếp:
    # Headless, 512 envs (debug trước)
    /opt/IsaacLab/isaaclab.sh -p robot_arm_training/task1_isaaclab/train.py \
        --headless --num_envs 512

    # Scale lên 4096 envs
    /opt/IsaacLab/isaaclab.sh -p robot_arm_training/task1_isaaclab/train.py \
        --headless --num_envs 4096

    # Tiếp tục từ checkpoint
    /opt/IsaacLab/isaaclab.sh -p robot_arm_training/task1_isaaclab/train.py \
        --headless --checkpoint ~/work/ppo_task1/checkpoints/agent_50000.pt

    # Inference (GUI)
    /opt/IsaacLab/isaaclab.sh -p robot_arm_training/task1_isaaclab/train.py \
        --checkpoint ~/work/ppo_task1/checkpoints/agent_best.pt \
        --eval_only --num_envs 16

TensorBoard:
    tensorboard --logdir ~/work/ppo_task1

Lưu ý server: checkpoints + logs ghi vào ~/work/ theo yêu cầu admin
(giảm Disk IO trên workspace chung).
"""

from __future__ import annotations

import argparse
import os
import sys

# Thêm thư mục chứa train.py vào sys.path để import env_cfg / env / agent_cfg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Isaac Sim / Isaac Lab launcher phải được gọi TRƯỚC khi import bất kỳ thứ gì ─
# AppLauncher xử lý SimulationApp + các extension cần thiết
from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI arguments — phải parse trước AppLauncher
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="PPO pick-and-place Task 1 (Isaac Lab + skrl)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--num_envs",        type=int,   default=4096,
                    help="Số parallel environments")
parser.add_argument("--total_timesteps", type=int,   default=100_000_000,
                    help="Tổng số timesteps training")
parser.add_argument("--seed",            type=int,   default=42)
parser.add_argument("--checkpoint",      type=str,   default=None,
                    help="Path đến checkpoint .pt để load (resume / eval)")
parser.add_argument("--eval_only",       action="store_true",
                    help="Chỉ chạy inference, không train")
# AppLauncher thêm --headless, --device, v.v.
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

# Khởi động Isaac Sim (PHẢI trước khi import omni/isaaclab assets)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Import sau khi SimulationApp đã chạy
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from env_cfg import PickPlaceEnvCfg
from env import PickPlaceEnv
from agent_cfg import (
    PPO_CFG,
    POLICY_NET_ARCH,
    VALUE_NET_ARCH,
    TOTAL_TIMESTEPS,
)


# ---------------------------------------------------------------------------
# Neural network models
# ---------------------------------------------------------------------------

class Policy(GaussianMixin, Model):
    """Gaussian policy (stochastic) cho PPO.

    skrl 2.x: dùng super().__init__() thay vì gọi từng base class trực tiếp,
    vì Model.__init__ không nhận positional args nữa.
    """

    def __init__(self, obs_space, act_space, device, clip_actions: bool = False, **kwargs):
        super().__init__(obs_space, act_space, device,
                         clip_actions=clip_actions, **kwargs)

        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for out_dim in POLICY_NET_ARCH:
            layers += [nn.Linear(in_dim, out_dim), nn.ELU()]
            in_dim = out_dim

        self.net        = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(in_dim, act_dim)
        self.log_std    = nn.Parameter(torch.zeros(act_dim))

    def compute(self, inputs: dict, role: str = ""):
        x    = self.net(inputs["states"])
        mean = self.mean_layer(x)
        return mean, self.log_std, {}


class Value(DeterministicMixin, Model):
    """Value function (critic) cho PPO."""

    def __init__(self, obs_space, act_space, device, clip_actions: bool = False, **kwargs):
        super().__init__(obs_space, act_space, device,
                         clip_actions=clip_actions, **kwargs)

        obs_dim = obs_space.shape[0]

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for out_dim in VALUE_NET_ARCH:
            layers += [nn.Linear(in_dim, out_dim), nn.ELU()]
            in_dim = out_dim

        self.net         = nn.Sequential(*layers)
        self.value_layer = nn.Linear(in_dim, 1)

    def compute(self, inputs: dict, role: str = ""):
        x = self.net(inputs["states"])
        return self.value_layer(x), {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(args.seed)

    # ── Tạo môi trường Isaac Lab ──────────────────────────────────────────
    env_cfg = PickPlaceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # Khi eval: tắt randomization mạnh, dùng ít envs hơn
    if args.eval_only:
        env_cfg.scene.num_envs = min(args.num_envs, 16)

    env_raw = PickPlaceEnv(cfg=env_cfg, render_mode="human" if not args.headless else None)

    # ── Wrap cho skrl ─────────────────────────────────────────────────────
    # skrl auto-detect Isaac Lab DirectRLEnv (không cần chỉ định wrapper type)
    env = wrap_env(env_raw)

    device = env.device

    # ── Tạo models ────────────────────────────────────────────────────────
    models = {
        "policy": Policy(env.observation_space, env.action_space, device),
        "value":  Value(env.observation_space,  env.action_space, device),
    }

    # ── Memory (rollout buffer) ───────────────────────────────────────────
    rollout_size = PPO_CFG["rollouts"]
    memory = RandomMemory(
        memory_size=rollout_size,
        num_envs=env.num_envs,
        device=device,
    )

    # ── Override config ───────────────────────────────────────────────────
    cfg = PPO_CFG.copy()

    # Observation normalizer (chạy online, giúp training ổn định)
    cfg["state_preprocessor"]        = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"]        = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # Ghi vào ~/work/ theo yêu cầu admin
    run_dir = os.path.expanduser("~/work/ppo_task1")
    cfg["experiment"]["directory"] = run_dir

    # ── Tạo PPO agent ─────────────────────────────────────────────────────
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # Load checkpoint nếu có
    if args.checkpoint:
        ckpt = os.path.expanduser(args.checkpoint)
        agent.load(ckpt)
        print(f"[PPO] Loaded checkpoint: {ckpt}")

    # ── Eval only ─────────────────────────────────────────────────────────
    if args.eval_only:
        _run_eval(env, agent, n_episodes=20)
        env.close()
        simulation_app.close()
        return

    # ── Training ──────────────────────────────────────────────────────────
    timesteps = args.total_timesteps or TOTAL_TIMESTEPS

    trainer_cfg = {
        "timesteps": timesteps,
        "headless":  args.headless,
    }
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    print(f"[PPO] Training — {timesteps:,} steps on {env.num_envs} envs")
    print(f"[PPO] Logs + checkpoints → {run_dir}")
    print(f"[PPO] TensorBoard: tensorboard --logdir {run_dir}")

    trainer.train()

    # Backup về workspace sau khi xong
    _backup_checkpoints(run_dir)

    env.close()
    simulation_app.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_eval(env, agent, n_episodes: int = 20):
    """Chạy inference và in success rate."""
    successes = 0
    obs, _ = env.reset()
    ep_count = 0
    ep_reward = torch.zeros(env.num_envs, device=env.device)

    print(f"\n[EVAL] Running {n_episodes} episodes...")
    while ep_count < n_episodes:
        with torch.no_grad():
            actions, _, _ = agent.act(obs, timestep=0, timesteps=0)
        obs, rewards, terminated, truncated, info = env.step(actions)
        ep_reward += rewards

        done = terminated | truncated
        for i in done.nonzero(as_tuple=True)[0]:
            ep_count += 1
            success = info.get("success", torch.zeros(env.num_envs, device=env.device))
            successes += int(success[i].item()) if isinstance(success, torch.Tensor) else 0
            print(
                f"  ep {ep_count:3d} | "
                f"reward={ep_reward[i].item():.1f} | "
                f"{'SUCCESS' if terminated[i] else 'timeout'}"
            )
            ep_reward[i] = 0.0
            if ep_count >= n_episodes:
                break

    print(f"\n[EVAL] Success rate: {successes}/{n_episodes} = "
          f"{100 * successes / max(n_episodes, 1):.0f}%")


def _backup_checkpoints(run_dir: str):
    """Backup checkpoints về workspace sau khi train xong."""
    dst = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "task1", "checkpoints", "ppo_isaaclab",
    )
    os.makedirs(dst, exist_ok=True)
    os.system(f"cp -r {run_dir}/checkpoints/. {dst}/")
    print(f"[PPO] Backup → {dst}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
