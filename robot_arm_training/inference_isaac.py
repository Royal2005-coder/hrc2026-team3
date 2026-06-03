"""
Tích hợp model đã train vào Isaac Sim.

Class ILPolicyRunner nhận state từ robot + scene,
chạy model predict action, rồi gửi xuống robot.

Cách dùng trong simulation loop:
    from robot_arm_training.inference_isaac import ILPolicyRunner

    runner = ILPolicyRunner(model_type="mlp")   # hoặc "lstm"
    runner.load()

    # Trong mỗi physics step:
    action = runner.step(robot, part_poses)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch

# Thêm thư mục robot_arm_training vào path để import được config, model, data_loader
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import config as cfg
from model import build_model, BCMlpPolicy, BCLstmPolicy
from data_loader import Normalizer


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: cánh tay trái cố định, model chỉ điều khiển cánh tay phải.
# ARM_JOINT_NAMES liệt kê đủ 14 khớp theo thứ tự robot interface yêu cầu.
# ─────────────────────────────────────────────────────────────────────────────
ARM_JOINT_NAMES = [
    "L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
    "L_elbow_roll_joint",     "L_elbow_yaw_joint",
    "L_wrist_pitch_joint",    "L_wrist_roll_joint",          # indices 0-6 (fixed)
    "R_shoulder_pitch_joint", "R_shoulder_roll_joint", "R_shoulder_yaw_joint",
    "R_elbow_roll_joint",     "R_elbow_yaw_joint",
    "R_wrist_pitch_joint",    "R_wrist_roll_joint",          # indices 7-13 (from model)
]

FINGER_JOINT_NAMES = [
    "L_finger1_joint", "L_finger2_joint",   # indices 0-1 (fixed)
    "R_finger1_joint", "R_finger2_joint",   # indices 2-3 (from model)
]

# Threshold để quyết định open/close gripper (giá trị từ CSV: -1=open, +1=close)
GRIPPER_CLOSE_THRESHOLD = 0.0
GRIPPER_CLOSE_WIDTH      =  0.01     # từ RobotArticulation
GRIPPER_OPEN_WIDTH       = -0.0215  # từ RobotArticulation


# ─────────────────────────────────────────────────────────────────────────────
# ILPolicyRunner
# ─────────────────────────────────────────────────────────────────────────────

class ILPolicyRunner:
    """
    Chạy Imitation Learning policy trong Isaac Sim.

    Args:
        model_type:  "mlp" hoặc "lstm"
        ckpt_dir:    Thư mục chứa best.pt và normalizer (mặc định tự tìm)
        device:      "cuda" / "cpu" (mặc định tự chọn)
    """

    def __init__(
        self,
        model_type: str = "mlp",
        ckpt_dir:   str | None = None,
        device:     str | None = None,
    ):
        self.model_type = model_type
        self.ckpt_dir   = ckpt_dir or os.path.join(cfg.CHECKPOINT_DIR, model_type)
        self.device     = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model:    BCMlpPolicy | BCLstmPolicy | None = None
        self.s_norm:   Normalizer | None = None
        self.a_norm:   Normalizer | None = None

        # Buffer LSTM window
        self._state_buffer: list[np.ndarray] = []
        self._lstm_hidden   = None

    # ── Load model từ checkpoint ──────────────────────────────────────────────

    def load(self):
        best_ckpt = os.path.join(self.ckpt_dir, "best.pt")
        if not os.path.exists(best_ckpt):
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint: {best_ckpt}\n"
                f"Hãy train trước: python train.py --model {self.model_type}"
            )

        self.model = build_model(self.model_type).to(self.device)
        ckpt = torch.load(best_ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.s_norm = Normalizer.load(os.path.join(self.ckpt_dir, "state_norm.npz"))
        self.a_norm = Normalizer.load(os.path.join(self.ckpt_dir, "action_norm.npz"))

        print(f"[ILPolicy] Loaded {self.model_type.upper()} (epoch {ckpt['epoch']}) "
              f"on {self.device}")

    def reset(self):
        """Reset LSTM hidden state giữa các episode."""
        self._state_buffer = []
        self._lstm_hidden  = None

    # ── Build state vector từ robot + scene ──────────────────────────────────

    @staticmethod
    def build_state_vector(
        robot_joint_states: dict,
        part_poses: list[dict],
        gripper_control: list[float] | None = None,
    ) -> np.ndarray:
        """
        Ghép 48-dim state vector từ dữ liệu Isaac Sim.

        Args:
            robot_joint_states: Output của robot.get_joint_states()
            part_poses:         Output của scene.get_parts_world_poses()
                                mỗi item có 'position' (3,) và 'orientation' (4,) [qw,qx,qy,qz]
            gripper_control:    [left_gripper, right_gripper] = [-1.0, -1.0] (open) hoặc [1.0, 1.0]

        Returns:
            state: np.ndarray shape (48,)
        """
        # 14 arm joints (theo thứ tự ARM_JOINT_NAMES)
        arm_pos = np.array(robot_joint_states["arm_positions"], dtype=np.float32)  # (14,)

        # 4 finger joints
        finger_pos = np.array(robot_joint_states["finger_positions"], dtype=np.float32)  # (4,)

        # 2 gripper control (-1 = open, 1 = close)
        if gripper_control is None:
            gripper_control = [-1.0, -1.0]
        gripper = np.array(gripper_control, dtype=np.float32)  # (2,)

        # 4 objects × 7 = 28 values  [x, y, z, qx, qy, qz, qw]
        obj_vec = np.zeros(28, dtype=np.float32)
        for i, part in enumerate(part_poses[:4]):
            pos = np.array(part["position"], dtype=np.float32)          # (3,)
            ori = np.array(part["orientation"], dtype=np.float32)       # (4,) [qw,qx,qy,qz]
            # CSV lưu theo [qx,qy,qz,qw], Isaac trả [qw,qx,qy,qz] → convert
            qw, qx, qy, qz = ori
            obj_vec[i * 7: i * 7 + 7] = [
                pos[0], pos[1], pos[2],
                qx, qy, qz, qw,
            ]

        state = np.concatenate([arm_pos, finger_pos, gripper, obj_vec])  # (48,)
        assert state.shape == (cfg.STATE_DIM,), f"State dim mismatch: {state.shape}"
        return state

    # ── Predict action từ state ───────────────────────────────────────────────

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Nhận state (48,), trả về action (10,) ở đơn vị radian (unnormalized).
        Thứ tự: R_shoulder_pitch..R_wrist_roll (7), R_finger1, R_finger2, right_gripper.
        """
        assert self.model is not None, "Gọi runner.load() trước"

        state_norm = self.s_norm.transform(state.reshape(1, -1))  # (1, 48)

        with torch.no_grad():
            if self.model_type == "mlp":
                s_t = torch.from_numpy(state_norm).float().to(self.device)
                a_n = self.model(s_t).cpu().numpy()   # (1, 10)

            else:  # lstm
                # Thêm vào buffer
                self._state_buffer.append(state_norm[0])
                if len(self._state_buffer) > cfg.LSTM_WINDOW_SIZE:
                    self._state_buffer = self._state_buffer[-cfg.LSTM_WINDOW_SIZE:]

                window = np.stack(self._state_buffer, axis=0)  # (T, 48)
                # Pad nếu chưa đủ window
                if len(window) < cfg.LSTM_WINDOW_SIZE:
                    pad    = np.repeat(window[:1], cfg.LSTM_WINDOW_SIZE - len(window), axis=0)
                    window = np.concatenate([pad, window], axis=0)

                w_t = torch.from_numpy(window).float().unsqueeze(0).to(self.device)  # (1, W, 48)
                a_n, self._lstm_hidden = self.model(w_t, self._lstm_hidden)
                a_n = a_n.cpu().numpy()   # (1, 10)

        action = self.a_norm.inverse_transform(a_n)[0]   # (20,) đơn vị radian
        return action

    # ── Parse action vector thành các phần ───────────────────────────────────

    @staticmethod
    def parse_action(action: np.ndarray) -> dict:
        """
        Tách action (10,) — chỉ cánh tay phải.

        Returns dict:
            r_arm_joints:  np.ndarray (7,) radian  — R_shoulder_pitch..R_wrist_roll
            r_fingers:     np.ndarray (2,) radian  — R_finger1, R_finger2
            right_gripper: float  (-1=open, +1=close)
        """
        r_arm_joints  = action[:7]       # indices 0..6
        r_fingers     = action[7:9]      # indices 7..8
        right_gripper = float(action[9]) # index 9
        return {
            "r_arm_joints":  r_arm_joints,
            "r_fingers":     r_fingers,
            "right_gripper": right_gripper,
        }

    # ── Apply action lên robot ────────────────────────────────────────────────

    @staticmethod
    def apply_action_to_robot(
        robot,
        parsed_action: dict,
        left_arm_positions: np.ndarray,
        left_finger_positions: np.ndarray,
    ):
        """
        Gửi action xuống robot trong Isaac Sim.
        Cánh tay trái giữ nguyên vị trí hiện tại (left_arm_positions từ get_joint_states).

        robot: IsaacSimRobotInterface hoặc RobotArticulation
        """
        # 1. Ghép 7 khớp trái (cố định) + 7 khớp phải (từ model) → 14 joints
        full_arm = np.concatenate([left_arm_positions, parsed_action["r_arm_joints"]])
        robot.set_arm_joint_positions(full_arm.tolist())

        # 2. Ghép 2 ngón trái (cố định) + 2 ngón phải (từ model) → 4 fingers
        full_fingers = np.concatenate([left_finger_positions, parsed_action["r_fingers"]])
        robot.set_finger_positions(full_fingers.tolist())

        # 3. Chỉ điều khiển gripper phải
        if parsed_action["right_gripper"] > GRIPPER_CLOSE_THRESHOLD:
            robot.close_gripper(side="right")
        else:
            robot.open_gripper(side="right")

    # ── One-call convenience ──────────────────────────────────────────────────

    def step(
        self,
        robot,
        part_poses: list[dict],
        gripper_control: list[float] | None = None,
        apply: bool = True,
    ) -> np.ndarray:
        """
        Full pipeline: đọc state → predict → apply.

        Args:
            robot:           Robot interface (IsaacSimRobotInterface)
            part_poses:      scene.get_parts_world_poses()
            gripper_control: Trạng thái gripper hiện tại [-1/-1=open]
            apply:           Nếu True, gửi action xuống robot luôn

        Returns:
            action (10,) ở đơn vị radian (chỉ cánh tay phải)
        """
        joint_states = robot.get_joint_states()
        if joint_states is None:
            return np.zeros(cfg.ACTION_DIM, dtype=np.float32)

        state  = self.build_state_vector(joint_states, part_poses, gripper_control)
        action = self.predict(state)

        if apply:
            parsed = self.parse_action(action)
            left_arm     = np.array(joint_states["arm_positions"][:7],    dtype=np.float32)
            left_fingers = np.array(joint_states["finger_positions"][:2], dtype=np.float32)
            self.apply_action_to_robot(robot, parsed, left_arm, left_fingers)

        return action
