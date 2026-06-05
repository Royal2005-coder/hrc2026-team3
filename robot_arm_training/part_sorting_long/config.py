"""
Config cho Behavioral Cloning trên Part_Sorting Long dataset (756 episodes, 768 frames/ep).

Dataset: part_sorting_long_756_episode (LeRobot parquet format)
  - observation.state: 48-dim
  - action:            20-dim
  - 4 grasps/episode (right arm only, left arm constant)
"""

import os

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(ROOT_DIR, "Part_Sorting", "part_sorting_long_756_episode")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

# ── Feature selection ─────────────────────────────────────────────────────────
# observation.state layout (48 dims):
#   0-6:  L arm joints (constant → excluded)
#   7-13: R arm joints
#   14-15: L fingers (constant → excluded)
#   16-17: R fingers
#   18:   left_gripper_control  (constant → excluded)
#   19:   right_gripper_control
#   20-47: obj0..3 poses (x,y,z,qx,qy,qz,qw)
STATE_IDX = list(range(7, 14)) + [16, 17, 19] + list(range(20, 48))

# action layout (20 dims): same joint ordering as state (no object poses)
#   0-6:  L arm (excluded)
#   7-13: R arm
#   14-15: L fingers (excluded)
#   16-17: R fingers
#   18:   left_gripper_control (excluded)
#   19:   right_gripper_control
ACTION_IDX = list(range(7, 14)) + [16, 17, 19]

STATE_DIM  = len(STATE_IDX)    # 38
ACTION_DIM = len(ACTION_IDX)   # 10

# ── Dataset split ─────────────────────────────────────────────────────────────
SEED       = 42
VAL_RATIO  = 0.10   # ~75 episodes
TEST_RATIO = 0.10   # ~75 episodes

# ── MLP hyperparameters ───────────────────────────────────────────────────────
MLP_HIDDEN_DIMS = [256, 256, 128]
MLP_DROPOUT     = 0.3

# ── LSTM hyperparameters ──────────────────────────────────────────────────────
LSTM_HIDDEN_DIM  = 256
LSTM_NUM_LAYERS  = 2
LSTM_DROPOUT     = 0.2
# Episode = 768 frames; 4 grasps → 192 frames/grasp → window = 1 full grasp cycle
LSTM_WINDOW_SIZE = 192

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE      = 512
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 2e-4
NUM_EPOCHS      = 300
PATIENCE        = 30
GRAD_CLIP       = 1.0
STATE_NOISE_STD = 0.01
VAL_EMA_ALPHA   = 0.3
LOG_EVERY_N_EPOCHS = 5
