"""
Cấu hình hyperparameters và đường dẫn cho robot arm training.
"""

import os

# Đường dẫn
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "part_sorting_sample.csv")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

# Cột state (input): 48 features
STATE_JOINT_COLS = [
    "state.L_shoulder_pitch_joint.pos", "state.L_shoulder_roll_joint.pos",
    "state.L_shoulder_yaw_joint.pos",   "state.L_elbow_roll_joint.pos",
    "state.L_elbow_yaw_joint.pos",      "state.L_wrist_pitch_joint.pos",
    "state.L_wrist_roll_joint.pos",
    "state.R_shoulder_pitch_joint.pos", "state.R_shoulder_roll_joint.pos",
    "state.R_shoulder_yaw_joint.pos",   "state.R_elbow_roll_joint.pos",
    "state.R_elbow_yaw_joint.pos",      "state.R_wrist_pitch_joint.pos",
    "state.R_wrist_roll_joint.pos",
    "state.L_finger1_joint.pos",        "state.L_finger2_joint.pos",
    "state.R_finger1_joint.pos",        "state.R_finger2_joint.pos",
    "state.left_gripper_control",       "state.right_gripper_control",
]

STATE_OBJ_COLS = []
for i in range(4):
    for attr in ["x", "y", "z", "qx", "qy", "qz", "qw"]:
        STATE_OBJ_COLS.append(f"state.obj{i}_{attr}")

STATE_COLS = STATE_JOINT_COLS + STATE_OBJ_COLS   # 20 + 28 = 48

# Cột action (output): 10 features — chỉ cánh tay phải (task 1: cánh tay trái cố định)
ACTION_COLS = [
    "action.R_shoulder_pitch_joint.pos", "action.R_shoulder_roll_joint.pos",
    "action.R_shoulder_yaw_joint.pos",   "action.R_elbow_roll_joint.pos",
    "action.R_elbow_yaw_joint.pos",      "action.R_wrist_pitch_joint.pos",
    "action.R_wrist_roll_joint.pos",
    "action.R_finger1_joint.pos",        "action.R_finger2_joint.pos",
    "action.right_gripper_control",
]

STATE_DIM  = len(STATE_COLS)   # 48
ACTION_DIM = len(ACTION_COLS)  # 10

# Hyperparameters chung
SEED         = 42
VAL_RATIO    = 0.15   # 15% episodes dùng validation
TEST_RATIO   = 0.10   # 10% episodes dùng test

# MLP config
MLP_HIDDEN_DIMS = [512, 512, 256]
MLP_DROPOUT     = 0.2    # tăng từ 0.1 → 0.2 để chống overfitting

# LSTM config
LSTM_HIDDEN_DIM    = 256
LSTM_NUM_LAYERS    = 2
LSTM_DROPOUT       = 0.2  # tăng từ 0.1 → 0.2
LSTM_WINDOW_SIZE   = 10

# Training config
BATCH_SIZE       = 256
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-4   # tăng từ 1e-5 → 1e-4
NUM_EPOCHS       = 200
PATIENCE         = 20
LR_STEP_SIZE     = 30     # không dùng nữa (cosine scheduler)
LR_GAMMA         = 0.5
GRAD_CLIP        = 1.0

# Data augmentation: thêm Gaussian noise vào state khi train
STATE_NOISE_STD  = 0.01   # std noise tương đối so với normalized state

# Logging
LOG_EVERY_N_EPOCHS = 5
