"""
Training entry point: GA (neuroevolution) cho Part Sorting Long — chia sẻ đúng 1
simulation context với 100 cá thể (xem GA_policy_evolution.py ở repo root).

Khác với train.py (PPO): không có gradient/value-network, mỗi cá thể GA "mang"
toàn bộ trọng số mạng Policy (mã hoá thành 10 đoạn gen 0-9 / trọng số), được nạp
lần lượt vào CÙNG 1 instance Policy để rollout trên dải env riêng của nó.

Cách chạy:
    /opt/IsaacLab/isaaclab.sh -p robot_arm_training/part_sorting_long_rl/train_ga.py \
        --headless --num_envs 4096 --generations 1000
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GA Part Sorting Long (Isaac Lab, neuroevolution)")
parser.add_argument("--num_envs",     type=int, default=4096)
parser.add_argument("--generations",  type=int, default=1000)
parser.add_argument("--rollout_steps", type=int, default=24)   # số bước mô phỏng / lần đánh giá fitness
parser.add_argument("--seed",         type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed

from env_cfg import PartSortingEnvCfg
from env import PartSortingEnv
from train import Policy   # tái dùng đúng kiến trúc Policy đã định nghĩa trong train.py

from GA_policy_evolution import (
    pop_size, NUM_ENVS, OBS_DIM, ACT_DIM, HIDDEN, npar,
    run_evolution,
)


def main():
    set_seed(args.seed)

    if args.num_envs != pop_size and args.num_envs % pop_size != 0:
        print(f"[GA] Lưu ý: {args.num_envs} env không chia hết cho {pop_size} cá thể "
              f"-> sẽ rải phần dư đều (xem split_envs).")

    env_cfg = PartSortingEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env_raw = PartSortingEnv(cfg=env_cfg, render_mode=None)
    env = wrap_env(env_raw)

    # Unwrap dict obs -> flat tensor, giống train.py
    _orig_reset, _orig_step = env.reset, env.step

    def _unpack(obs):
        return obs.get("policy", next(iter(obs.values()))) if isinstance(obs, dict) else obs

    env.reset = lambda *a, **kw: (_unpack(r[0]), r[1]) if (r := _orig_reset(*a, **kw)) else r
    env.step  = lambda *a, **kw: (_unpack(r[0]), *r[1:]) if (r := _orig_step(*a, **kw)) else r

    device = env.device
    assert env.observation_space.shape[0] == OBS_DIM and env.action_space.shape[0] == ACT_DIM, (
        "OBS_DIM/ACT_DIM trong GA_policy_evolution.py không khớp env hiện tại — "
        "cập nhật lại để genome_to_state_dict() map đúng kiến trúc."
    )

    # 1 instance Policy dùng làm "khung" — nạp lại trọng số của từng cá thể trước khi forward
    policy_model = Policy(env.observation_space, env.action_space, device)

    run_dir = os.path.expanduser("~/work/ga_part_sorting")
    print(f"[GA] {pop_size} cá thể chia sẻ {env.num_envs} env (~{env.num_envs // pop_size} env/cá thể)")
    print(f"[GA] npar (tổng tham số Policy {OBS_DIM}->{HIDDEN}->{ACT_DIM}) = {npar}")
    print(f"[GA] Train {args.generations} thế hệ x {args.rollout_steps} bước rollout/thế hệ")
    print(f"[GA] Lưu best genome + lịch sử fitness -> {run_dir}")

    final_pop, history = run_evolution(
        env=env,
        policy_model=policy_model,
        num_generations=args.generations,
        n_steps=args.rollout_steps,
        log_every=1,
        save_dir=run_dir,
    )

    print(f"[GA] Xong. Best fitness cuối cùng = {history[-1]:.2f} "
          f"(genome lưu tại {os.path.join(run_dir, 'best_genome.npy')})")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
