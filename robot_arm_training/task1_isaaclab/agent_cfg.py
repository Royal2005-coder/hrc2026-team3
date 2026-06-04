"""
PPO hyperparameters cho skrl — Task 1 pick-and-place.

Tham khảo: rl_isaac_lab_plan.md + skrl docs
  https://skrl.readthedocs.io/en/latest/api/agents/ppo.html
"""

from __future__ import annotations

from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG

# ---------------------------------------------------------------------------
# PPO config (copy từ default rồi override)
# ---------------------------------------------------------------------------
PPO_CFG: dict = PPO_DEFAULT_CONFIG.copy()
PPO_CFG.update(
    {
        # ── Rollout ──────────────────────────────────────────────────────────
        "rollouts": 24,          # số steps thu thập mỗi env trước mỗi update
        "learning_epochs": 5,    # số epochs gradient trên mỗi rollout
        "mini_batches": 4,       # batch_size ≈ 24 × 4096 / 4 ≈ 24.576k transitions

        # ── Discount & GAE ───────────────────────────────────────────────────
        "discount_factor": 0.99,
        "lambda": 0.95,          # GAE lambda

        # ── Learning rate ────────────────────────────────────────────────────
        "learning_rate": 3e-4,
        "learning_rate_scheduler": None,  # có thể đổi sang CosineAnnealingLR

        # ── Gradient ─────────────────────────────────────────────────────────
        "grad_norm_clip": 1.0,

        # ── PPO clip ─────────────────────────────────────────────────────────
        "ratio_clip": 0.2,              # clipping epsilon ε
        "value_clip": 0.2,              # clipping cho value loss
        "clip_predicted_values": True,  # khuyến khích dùng khi clip value

        # ── Loss coefficients ────────────────────────────────────────────────
        "entropy_loss_scale": 0.01,     # khuyến khích exploration
        "value_loss_scale": 1.0,

        # ── Normalization ────────────────────────────────────────────────────
        "state_preprocessor": None,          # sẽ override ở train.py
        "state_preprocessor_kwargs": {},
        "value_preprocessor": None,
        "value_preprocessor_kwargs": {},

        # ── Logging ──────────────────────────────────────────────────────────
        "experiment": {
            "directory": "~/work/ppo_task1",
            "experiment_name": "pick_place_ppo",
            "write_interval": 1000,       # log mỗi 1000 steps
            "checkpoint_interval": 50000, # save checkpoint mỗi 50k steps
            "store_separately": False,
        },
    }
)

# ---------------------------------------------------------------------------
# Network architecture (dùng trong train.py để xây Policy và Value models)
# ---------------------------------------------------------------------------
POLICY_NET_ARCH: list[int] = [256, 128, 64]
VALUE_NET_ARCH:  list[int] = [256, 128, 64]

# ---------------------------------------------------------------------------
# Env config override (có thể override từ CLI)
# ---------------------------------------------------------------------------
NUM_ENVS_TRAIN: int = 4096
NUM_ENVS_EVAL:  int = 16
SEED:           int = 42
TOTAL_TIMESTEPS: int = 100_000_000   # 100M steps (khoảng vài tiếng với 4096 envs)
