"""PPO hyperparameters cho Part Sorting Long RL."""

from __future__ import annotations

PPO_CFG: dict = {
    "rollouts": 24,
    "learning_epochs": 5,
    "mini_batches": 4,

    "discount_factor": 0.99,
    "gae_lambda": 0.95,

    "learning_rate": 3e-4,
    "learning_rate_scheduler": None,
    "learning_rate_scheduler_kwargs": {},

    "state_preprocessor": None,
    "state_preprocessor_kwargs": {},
    "value_preprocessor": None,
    "value_preprocessor_kwargs": {},

    "grad_norm_clip": 1.0,
    "ratio_clip": 0.2,
    "value_clip": 0.2,

    "entropy_loss_scale": 0.01,
    "value_loss_scale": 1.0,

    "kl_threshold": 0,
    "rewards_shaper": None,
    "time_limit_bootstrap": False,

    "experiment": {
        "directory": "~/work/ppo_part_sorting",
        "experiment_name": "part_sorting_ppo",
        "write_interval": 1000,
        "checkpoint_interval": 50000,
        "store_separately": False,
    },
}

POLICY_NET_ARCH: list[int] = [256, 128, 64]
VALUE_NET_ARCH:  list[int] = [256, 128, 64]

NUM_ENVS_TRAIN:  int = 4096
NUM_ENVS_EVAL:   int = 16
SEED:            int = 42
TOTAL_TIMESTEPS: int = 100_000_000
