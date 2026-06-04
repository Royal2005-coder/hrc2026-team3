"""
PPO hyperparameters cho skrl — Task 1 pick-and-place.

Không import PPO_DEFAULT_CONFIG (đã bị xóa trong skrl 2.x).
Định nghĩa dict trực tiếp với đầy đủ keys mà PPO agent cần.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# PPO config dict — tương thích skrl 1.x và 2.x
# ---------------------------------------------------------------------------
PPO_CFG: dict = {
    # ── Rollout ──────────────────────────────────────────────────────────────
    "rollouts": 24,           # n_steps per env trước mỗi update
    "learning_epochs": 5,     # số epochs gradient trên mỗi rollout batch
    "mini_batches": 4,        # effective batch ≈ 24 × num_envs / 4

    # ── Discount & GAE ───────────────────────────────────────────────────────
    "discount_factor": 0.99,
    "lambda": 0.95,           # GAE lambda

    # ── Learning rate ────────────────────────────────────────────────────────
    "learning_rate": 3e-4,
    "learning_rate_scheduler": None,
    "learning_rate_scheduler_kwargs": {},

    # ── Preprocessors (sẽ được override trong train.py) ──────────────────────
    "state_preprocessor": None,
    "state_preprocessor_kwargs": {},
    "value_preprocessor": None,
    "value_preprocessor_kwargs": {},

    # ── Gradient clipping ────────────────────────────────────────────────────
    "grad_norm_clip": 1.0,

    # ── PPO clip ─────────────────────────────────────────────────────────────
    "ratio_clip": 0.2,
    "value_clip": 0.2,
    "clip_predicted_values": True,

    # ── Loss coefficients ────────────────────────────────────────────────────
    "entropy_loss_scale": 0.01,
    "value_loss_scale": 1.0,

    # ── Misc ─────────────────────────────────────────────────────────────────
    "kl_threshold": 0,
    "rewards_shaper": None,
    "time_limit_bootstrap": False,

    # ── Logging / checkpointing ──────────────────────────────────────────────
    "experiment": {
        "directory": "~/work/ppo_task1",
        "experiment_name": "pick_place_ppo",
        "write_interval": 1000,
        "checkpoint_interval": 50000,
        "store_separately": False,
    },
}

# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------
POLICY_NET_ARCH: list[int] = [256, 128, 64]
VALUE_NET_ARCH:  list[int] = [256, 128, 64]

# ---------------------------------------------------------------------------
# Env / training defaults (có thể override từ CLI)
# ---------------------------------------------------------------------------
NUM_ENVS_TRAIN:  int = 4096
NUM_ENVS_EVAL:   int = 16
SEED:            int = 42
TOTAL_TIMESTEPS: int = 100_000_000
