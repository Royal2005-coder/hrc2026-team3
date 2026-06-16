"""
Debug entry point: kiểm tra trực tiếp 3 nghi vấn khiến fitness của GA/ES không
phụ thuộc vào genome — triệu chứng quan sát được: `worst` (cá thể tệ nhất mỗi
thế hệ) ra ĐÚNG -289.44, lặp lại y hệt đến 2 chữ số thập phân, suốt > 600 thế
hệ, ở CẢ HAI cách sinh trọng số hoàn toàn khác nhau (digit-GA ngẫu nhiên lớn
U[-1,1] và ES nhiễu nhỏ quanh init). Một thuật toán tiến hoá không thể tạo ra
hiện tượng "đứng yên tuyệt đối" như vậy nếu fitness thực sự là hàm của genome
— do đó cần loại trừ khả năng có bug trong pipeline action -> env -> reward
TRƯỚC KHI tiếp tục chỉnh tham số GA/ES.

3 phép kiểm tra:
  A. load_state_dict() có thực sự làm policy_model xuất ra action khác nhau
     giữa 2 genome khác nhau không? (nếu KHÔNG -> mọi cá thể đang chạy chung
     1 bộ trọng số, fitness chỉ khác nhau do dải env khác nhau)
  B. action có thực sự ảnh hưởng đến reward không? (so sánh fitness khi ép
     action = 0 mọi lúc vs action ngẫu nhiên biên độ lớn — nếu fitness gần như
     giống hệt nhau -> bug nằm ở env.step()/action processing, không phải GA)
  C. env.reset() có thực sự random lại vị trí 4 vật mỗi lần gọi không? (nếu
     KHÔNG -> 1 số dải env "đóng băng" ở cùng 1 trạng thái mỗi thế hệ)

Cách chạy (vài chục giây, không cần generations dài):
    /opt/IsaacLab/isaaclab.sh -p robot_arm_training/part_sorting_long_rl/debug_ga_pipeline.py \
        --headless --num_envs 200
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug pipeline GA Part Sorting Long")
parser.add_argument("--num_envs", type=int, default=200)
parser.add_argument("--seed",     type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import torch.nn as nn

from skrl.envs.wrappers.torch import wrap_env
from skrl.models.torch import GaussianMixin, Model
from skrl.utils import set_seed

from env_cfg import PartSortingEnvCfg
from env import PartSortingEnv

from GA_policy_evolution import (
    OBS_DIM, ACT_DIM, HIDDEN, npar,
    split_envs, decode, pop_init, genome_to_state_dict,
)


class Policy(GaussianMixin, Model):
    def __init__(self, obs_space, act_space, device, clip_actions=False):
        Model.__init__(self)
        GaussianMixin.__init__(self)
        self.observation_space = obs_space
        self.action_space      = act_space
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        layers, in_dim = [], obs_space.shape[0]
        for out_dim in HIDDEN:
            layers += [nn.Linear(in_dim, out_dim), nn.Tanh()]
            in_dim = out_dim
        self.net        = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(in_dim, act_space.shape[0])
        self.log_std    = nn.Parameter(torch.zeros(act_space.shape[0]))

    def compute(self, inputs, role=""):
        if isinstance(inputs, torch.Tensor):
            states = inputs
        elif "states" in inputs:
            states = inputs["states"]
        else:
            states = next(v for v in inputs.values() if isinstance(v, torch.Tensor))
        return torch.tanh(self.mean_layer(self.net(states))), {"log_std": self.log_std}


def _load_genome(policy_model, genome, device):
    sd = genome_to_state_dict(genome)
    policy_model.load_state_dict(
        {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in sd.items()},
        strict=True,
    )


def main():
    set_seed(args.seed)

    env_cfg = PartSortingEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env_raw = PartSortingEnv(cfg=env_cfg, render_mode=None)
    env = wrap_env(env_raw)

    _orig_reset, _orig_step = env.reset, env.step

    def _unpack(obs):
        return obs.get("policy", next(iter(obs.values()))) if isinstance(obs, dict) else obs

    env.reset = lambda *a, **kw: (_unpack(r[0]), r[1]) if (r := _orig_reset(*a, **kw)) else r
    env.step  = lambda *a, **kw: (_unpack(r[0]), *r[1:]) if (r := _orig_step(*a, **kw)) else r

    device = env.device
    policy_model = Policy(env.observation_space, env.action_space, device).to(device)

    n_groups = 4
    env_slices = split_envs(env.num_envs, n_groups)

    print("=" * 70)
    print("[DEBUG-A] load_state_dict() có thực sự đổi output của policy_model?")
    print("=" * 70)
    g1 = decode(pop_init(1, npar, 10))[0]
    g2 = decode(pop_init(1, npar, 10))[0]
    obs, _ = env.reset()
    _load_genome(policy_model, g1, device)
    with torch.no_grad():
        out1, _ = policy_model.compute({"states": obs})
    _load_genome(policy_model, g2, device)
    with torch.no_grad():
        out2, _ = policy_model.compute({"states": obs})
    diff = (out1 - out2).abs().mean().item()
    print(f"  ||action(genome_1) - action(genome_2)||_mean = {diff:.6f}")
    print(f"  -> {'CÓ tác dụng (load_state_dict OK)' if diff > 1e-5 else '!!! KHÔNG có tác dụng -> mọi cá thể đang dùng CHUNG 1 bộ trọng số -> BUG ở load_state_dict/Policy'}")

    print()
    print("=" * 70)
    print("[DEBUG-B] action có thực sự ảnh hưởng đến reward / quỹ đạo env?")
    print("=" * 70)
    n_steps = 24

    def _rollout_fixed_action(make_action):
        obs, _ = env.reset()
        fit = np.zeros(n_groups)
        for _ in range(n_steps):
            actions = make_action()
            obs, rewards, *_ = env.step(actions)
            r = rewards.detach().cpu().numpy()
            for i, ids in enumerate(env_slices):
                fit[i] += r[ids].sum()
        return fit

    fit_zero = _rollout_fixed_action(lambda: torch.zeros((env.num_envs, ACT_DIM), device=device))
    fit_rand = _rollout_fixed_action(lambda: torch.rand((env.num_envs, ACT_DIM), device=device) * 2 - 1)
    fit_ones = _rollout_fixed_action(lambda: torch.ones((env.num_envs, ACT_DIM), device=device))
    print(f"  fitness 4 nhóm | action = 0 mọi lúc        : {np.round(fit_zero, 2)}")
    print(f"  fitness 4 nhóm | action ngẫu nhiên U[-1,1] : {np.round(fit_rand, 2)}")
    print(f"  fitness 4 nhóm | action = 1 mọi lúc        : {np.round(fit_ones, 2)}")
    spread = max(fit_zero.max(), fit_rand.max(), fit_ones.max()) - min(fit_zero.min(), fit_rand.min(), fit_ones.min())
    print(f"  chênh lệch lớn nhất giữa 3 kiểu action     : {spread:.2f}")
    print(f"  -> {'action CÓ ảnh hưởng rõ rệt đến reward' if spread > 5.0 else '!!! 3 kiểu action SO SÁNH GẦN NHƯ GIỐNG NHAU -> action KHÔNG ảnh hưởng đến reward -> BUG ở env.step()/action processing/wrapper'}")

    print()
    print("=" * 70)
    print("[DEBUG-C] env.reset() có thực sự random lại vị trí 4 vật mỗi lần gọi?")
    print("=" * 70)
    env.reset()
    pos_1 = torch.stack([p.data.root_pos_w for p in env_raw.parts], dim=1).clone()
    env.reset()
    pos_2 = torch.stack([p.data.root_pos_w for p in env_raw.parts], dim=1).clone()
    same = torch.allclose(pos_1, pos_2)
    max_diff = (pos_1 - pos_2).abs().max().item()
    print(f"  vị trí vật giống hệt giữa 2 lần reset liên tiếp: {same}  (chênh lệch lớn nhất = {max_diff:.6f} m)")
    print(f"  -> {'!!! reset KHÔNG random lại -> 1 số dải env có thể đóng băng ở cùng 1 trạng thái mỗi thế hệ' if same else 'reset CÓ random lại vị trí — OK'}")

    print()
    print("=" * 70)
    print("KẾT LUẬN: nếu cả A, B, C đều OK mà fitness vẫn đứng yên qua nhiều thế hệ,")
    print("vấn đề khả năng cao nằm ở chính thuật toán tiến hoá (GA/ES) — lúc đó mới")
    print("nên quay lại tinh chỉnh mutation_rate / sigma / selection / npar...")
    print("Nếu MỘT trong 3 test trên báo lỗi (!!!), hãy fix đúng chỗ đó trước —")
    print("không thuật toán tiến hoá nào học được trên 1 fitness không phụ thuộc genome.")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
