"""
Training entry point: PPO + skrl cho Part Sorting Long.

Cách chạy:
    # Headless, 512 envs (debug trước)
    /opt/IsaacLab/isaaclab.sh -p robot_arm_training/part_sorting_long_rl/train.py \
        --headless --num_envs 512

    # Scale lên
    /opt/IsaacLab/isaaclab.sh -p robot_arm_training/part_sorting_long_rl/train.py \
        --headless --num_envs 4096

    # Resume từ checkpoint
    /opt/IsaacLab/isaaclab.sh -p robot_arm_training/part_sorting_long_rl/train.py \
        --headless --checkpoint ~/work/ppo_part_sorting/checkpoints/agent_best.pt

    # Inference GUI
    /opt/IsaacLab/isaaclab.sh -p robot_arm_training/part_sorting_long_rl/train.py \
        --checkpoint ~/work/ppo_part_sorting/checkpoints/agent_best.pt \
        --eval_only --num_envs 4

TensorBoard:
    tensorboard --logdir ~/work/ppo_part_sorting
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PPO Part Sorting Long (Isaac Lab + skrl)")
parser.add_argument("--num_envs",        type=int, default=4096)
parser.add_argument("--total_timesteps", type=int, default=100_000_000)
parser.add_argument("--seed",            type=int, default=42)
parser.add_argument("--checkpoint",      type=str, default=None)
parser.add_argument("--eval_only",       action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from env_cfg import PartSortingEnvCfg
from env import PartSortingEnv
from agent_cfg import PPO_CFG, POLICY_NET_ARCH, VALUE_NET_ARCH, TOTAL_TIMESTEPS


class Policy(GaussianMixin, Model):
    def __init__(self, obs_space, act_space, device, clip_actions=False):
        Model.__init__(self)
        GaussianMixin.__init__(self)
        self.observation_space = obs_space
        self.action_space      = act_space
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        layers, in_dim = [], obs_space.shape[0]
        for out_dim in POLICY_NET_ARCH:
            layers += [nn.Linear(in_dim, out_dim), nn.ELU()]
            in_dim = out_dim
        self.net        = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(in_dim, act_space.shape[0])
        self.log_std    = nn.Parameter(torch.zeros(act_space.shape[0]))

    def compute(self, inputs, role=""):
        states = inputs if isinstance(inputs, torch.Tensor) else (
            inputs.get("states") or next(v for v in inputs.values() if isinstance(v, torch.Tensor))
        )
        return self.mean_layer(self.net(states)), {"log_std": self.log_std}


class Value(DeterministicMixin, Model):
    def __init__(self, obs_space, act_space, device, clip_actions=False):
        Model.__init__(self)
        DeterministicMixin.__init__(self)
        self.observation_space = obs_space
        self.action_space      = act_space
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        layers, in_dim = [], obs_space.shape[0]
        for out_dim in VALUE_NET_ARCH:
            layers += [nn.Linear(in_dim, out_dim), nn.ELU()]
            in_dim = out_dim
        self.net         = nn.Sequential(*layers)
        self.value_layer = nn.Linear(in_dim, 1)

    def compute(self, inputs, role=""):
        states = inputs if isinstance(inputs, torch.Tensor) else (
            inputs.get("states") or next(v for v in inputs.values() if isinstance(v, torch.Tensor))
        )
        return self.value_layer(self.net(states)), {}


def main():
    set_seed(args.seed)

    env_cfg = PartSortingEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.eval_only:
        env_cfg.scene.num_envs = min(args.num_envs, 4)

    env_raw = PartSortingEnv(
        cfg=env_cfg,
        render_mode="human" if not args.headless else None,
    )
    env = wrap_env(env_raw)

    # Unwrap dict obs → flat tensor cho skrl PPO
    _orig_reset, _orig_step = env.reset, env.step

    def _unpack(obs):
        return obs.get("policy", next(iter(obs.values()))) if isinstance(obs, dict) else obs

    env.reset = lambda *a, **kw: (_unpack(r[0]), r[1]) if (r := _orig_reset(*a, **kw)) else r
    env.step  = lambda *a, **kw: (_unpack(r[0]), *r[1:]) if (r := _orig_step(*a, **kw)) else r

    device = env.device
    models = {
        "policy": Policy(env.observation_space, env.action_space, device),
        "value":  Value(env.observation_space,  env.action_space, device),
    }

    memory = RandomMemory(
        memory_size=PPO_CFG["rollouts"],
        num_envs=env.num_envs,
        device=device,
    )

    cfg = PPO_CFG.copy()
    cfg["state_preprocessor"]        = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"]        = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    run_dir = os.path.expanduser("~/work/ppo_part_sorting")
    cfg["experiment"]["directory"] = run_dir

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    if args.checkpoint:
        agent.load(os.path.expanduser(args.checkpoint))
        print(f"[PPO] Loaded: {args.checkpoint}")

    if args.eval_only:
        _run_eval(env, agent)
        env.close()
        simulation_app.close()
        return

    timesteps = args.total_timesteps or TOTAL_TIMESTEPS
    trainer = SequentialTrainer(
        cfg={"timesteps": timesteps, "headless": args.headless},
        env=env,
        agents=agent,
    )
    print(f"[PPO] Training — {timesteps:,} steps | {env.num_envs} envs")
    print(f"[PPO] Logs → {run_dir}  |  tensorboard --logdir {run_dir}")
    trainer.train()

    env.close()
    simulation_app.close()


def _run_eval(env, agent, n_episodes=20):
    successes, ep_count = 0, 0
    obs, _ = env.reset()
    ep_reward = torch.zeros(env.num_envs, device=env.device)

    print(f"\n[EVAL] {n_episodes} episodes...")
    while ep_count < n_episodes:
        with torch.no_grad():
            actions, _, _ = agent.act(obs, timestep=0, timesteps=0)
        obs, rewards, terminated, truncated, info = env.step(actions)
        ep_reward += rewards
        done = terminated | truncated
        for i in done.nonzero(as_tuple=True)[0]:
            ep_count += 1
            successes += int(terminated[i].item())
            print(f"  ep {ep_count:3d} | reward={ep_reward[i].item():.1f} | "
                  f"{'SUCCESS' if terminated[i] else 'timeout'}")
            ep_reward[i] = 0.0
            if ep_count >= n_episodes:
                break

    print(f"\n[EVAL] Success: {successes}/{n_episodes} = {100*successes//max(n_episodes,1)}%")


if __name__ == "__main__":
    main()
