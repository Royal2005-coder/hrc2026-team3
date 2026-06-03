"""
Cấu hình hyperparameters và đường dẫn cho robot arm training.
"""

import os

# Đường dẫn — task1 nằm ở robot_arm_training/task1/, root là 2 cấp trên
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH     = os.path.join(ROOT_DIR, "data", "task1", "part_sorting_sample.csv")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
LOG_DIR       = os.path.join(ROOT_DIR, "logs", "task1")

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
VAL_RATIO    = 0.20   # 20% episodes → ~18 eps, giảm noise trong val metrics
TEST_RATIO   = 0.10   # 10% episodes dùng test

# MLP config — thu nhỏ network (424k→112k params) để giảm overfitting
MLP_HIDDEN_DIMS = [256, 256, 128]
MLP_DROPOUT     = 0.3

# LSTM config
LSTM_HIDDEN_DIM    = 256
LSTM_NUM_LAYERS    = 2
LSTM_DROPOUT       = 0.2
LSTM_WINDOW_SIZE   = 15   # tăng 10→15 để nắm thêm ngữ cảnh thời gian

# Training config
BATCH_SIZE       = 256
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 2e-4   # tăng L2 regularization
NUM_EPOCHS       = 300    # tăng 200→300: với regularization tốt hơn model có thể train lâu hơn
PATIENCE         = 30     # scale theo NUM_EPOCHS
LR_STEP_SIZE     = 30     # không dùng nữa (cosine scheduler)
LR_GAMMA         = 0.5
GRAD_CLIP        = 1.0

# Data augmentation: thêm Gaussian noise vào state khi train
STATE_NOISE_STD  = 0.01   # std noise tương đối so với normalized state

# EMA smoothing cho early stopping (alpha: trọng số của epoch hiện tại)
VAL_EMA_ALPHA    = 0.3    # ema = 0.3*val + 0.7*ema_prev

# Logging
LOG_EVERY_N_EPOCHS = 5
