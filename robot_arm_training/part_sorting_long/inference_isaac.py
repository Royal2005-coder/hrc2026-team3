"""
Tích hợp LSTM/MLP đã train (part_sorting_long) vào Isaac Sim.

Cách dùng trong simulation loop:
    from robot_arm_training.part_sorting_long.inference_isaac import ILPolicyRunner

    runner = ILPolicyRunner(model_type="lstm")
    runner.load()

    # Trong mỗi physics step:
    action = runner.step(robot, part_poses, gripper_control=gripper_state)
"""

from __future__ import annotations

import importlib.util as _ilu
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "shared"))  # model.py
sys.path.insert(0, _HERE)                                 # data_loader.py

# Load part_sorting_long config qua explicit path để tránh xung đột
# với config.py của các package khác trong sys.path (Isaac Sim, cv2, …)
_cfg_spec = _ilu.spec_from_file_location("config", os.path.join(_HERE, "config.py"))
cfg = _ilu.module_from_spec(_cfg_spec)
sys.modules["config"] = cfg
_cfg_spec.loader.exec_module(cfg)
del _ilu, _cfg_spec

from model import BCLstmPolicy, BCMlpPolicy, build_model
from data_loader import Normalizer

# ─────────────────────────────────────────────────────────────────────────────
# Tên joint theo thứ tự robot interface
# ─────────────────────────────────────────────────────────────────────────────

ARM_JOINT_NAMES = [
    "L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
    "L_elbow_roll_joint",     "L_elbow_yaw_joint",
    "L_wrist_pitch_joint",    "L_wrist_roll_joint",          # 0-6  (cố định)
    "R_shoulder_pitch_joint", "R_shoulder_roll_joint", "R_shoulder_yaw_joint",
    "R_elbow_roll_joint",     "R_elbow_yaw_joint",
    "R_wrist_pitch_joint",    "R_wrist_roll_joint",          # 7-13 (từ model)
]

FINGER_JOINT_NAMES = [
    "L_finger1_joint", "L_finger2_joint",   # 0-1 (cố định)
    "R_finger1_joint", "R_finger2_joint",   # 2-3 (từ model)
]

GRIPPER_CLOSE_THRESHOLD = 0.0
GRIPPER_CLOSE_WIDTH     =  0.01
GRIPPER_OPEN_WIDTH      = -0.0215


# ─────────────────────────────────────────────────────────────────────────────
# ILPolicyRunner
# ─────────────────────────────────────────────────────────────────────────────

class ILPolicyRunner:
    """
    Chạy IL policy (MLP hoặc LSTM) cho Part Sorting Long task.

    Args:
        model_type: "mlp" hoặc "lstm"
        ckpt_dir:   Thư mục chứa best.pt + normalizer (mặc định tự tìm)
        device:     "cuda" / "cpu" (mặc định tự chọn)
    """

    def __init__(
        self,
        model_type: str = "lstm",
        ckpt_dir:   str | None = None,
        device:     str | None = None,
    ):
        self.model_type = model_type
        self.ckpt_dir   = ckpt_dir or os.path.join(cfg.CHECKPOINT_DIR, model_type)
        self.device     = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model:  BCMlpPolicy | BCLstmPolicy | None = None
        self.s_norm: Normalizer | None = None
        self.a_norm: Normalizer | None = None

        self._state_buffer: list[np.ndarray] = []
        self._lstm_hidden = None

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

        print(f"[ILPolicy/part_sorting_long] Loaded {self.model_type.upper()} "
              f"(epoch {ckpt['epoch']}) — val_loss={ckpt.get('best_val_loss', '?'):.5f} "
              f"on {self.device}")

    def reset(self):
        """Reset LSTM hidden state giữa các episode."""
        self._state_buffer = []
        self._lstm_hidden  = None

    # ── Build state vector (38-dim) từ Isaac Sim ──────────────────────────────

    @staticmethod
    def build_state_vector(
        robot_joint_states: dict,
        part_poses: list[dict],
        gripper_control: list[float] | None = None,
    ) -> np.ndarray:
        """
        Ghép 38-dim state vector từ dữ liệu Isaac Sim.

        Layout:
            R arm joints (7) + R fingers (2) + right_gripper (1) + obj0..3 poses (28)

        Args:
            robot_joint_states: output của robot.get_joint_states()
            part_poses:         output của scene.get_parts_world_poses()
                                mỗi item có 'position' (3,) và 'orientation' (4,) [qw,qx,qy,qz]
            gripper_control:    [left_gripper, right_gripper], mặc định [-1, -1] (mở)
        """
        arm_pos    = np.array(robot_joint_states["arm_positions"],    dtype=np.float32)  # (14,)
        finger_pos = np.array(robot_joint_states["finger_positions"], dtype=np.float32)  # (4,)

        if gripper_control is None:
            gripper_control = [-1.0, -1.0]

        r_arm     = arm_pos[7:14]                                     # (7,)
        r_fingers = finger_pos[2:4]                                   # (2,)
        r_gripper = np.array([gripper_control[1]], dtype=np.float32)  # (1,)

        # 4 objects × 7 = 28 values  [x, y, z, qx, qy, qz, qw]
        obj_vec = np.zeros(28, dtype=np.float32)
        for i, part in enumerate(part_poses[:4]):
            pos = np.array(part["position"],    dtype=np.float32)   # (3,)
            ori = np.array(part["orientation"], dtype=np.float32)   # (4,) [qw,qx,qy,qz]
            qw, qx, qy, qz = ori
            obj_vec[i * 7: i * 7 + 7] = [pos[0], pos[1], pos[2], qx, qy, qz, qw]

        state = np.concatenate([r_arm, r_fingers, r_gripper, obj_vec])  # (38,)
        assert state.shape == (cfg.STATE_DIM,), f"State dim mismatch: {state.shape}"
        return state

    # ── Predict action (10-dim) từ state ────────────────────────────────────

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Nhận state (38,), trả về action (10,) ở đơn vị radian (unnormalized).
        Thứ tự: R_shoulder_pitch..R_wrist_roll (7), R_finger1, R_finger2, right_gripper.
        """
        assert self.model is not None, "Gọi runner.load() trước"

        state_norm = self.s_norm.transform(state.reshape(1, -1))  # (1, 38)

        with torch.no_grad():
            if self.model_type == "mlp":
                s_t = torch.from_numpy(state_norm).float().to(self.device)
                a_n = self.model(s_t).cpu().numpy()

            else:  # lstm
                self._state_buffer.append(state_norm[0])
                if len(self._state_buffer) > cfg.LSTM_WINDOW_SIZE:
                    self._state_buffer = self._state_buffer[-cfg.LSTM_WINDOW_SIZE:]

                window = np.stack(self._state_buffer, axis=0)  # (T, 38)
                # Pad đầu bằng frame đầu tiên nếu chưa đủ window
                if len(window) < cfg.LSTM_WINDOW_SIZE:
                    pad    = np.repeat(window[:1], cfg.LSTM_WINDOW_SIZE - len(window), axis=0)
                    window = np.concatenate([pad, window], axis=0)

                w_t = torch.from_numpy(window).float().unsqueeze(0).to(self.device)  # (1, 30, 38)
                a_n, self._lstm_hidden = self.model(w_t, self._lstm_hidden)
                a_n = a_n.cpu().numpy()

        return self.a_norm.inverse_transform(a_n)[0]  # (10,) radian

    # ── Parse action thành các phần ─────────────────────────────────────────

    @staticmethod
    def parse_action(action: np.ndarray) -> dict:
        return {
            "r_arm_joints":  action[:7],
            "r_fingers":     action[7:9],
            "right_gripper": float(action[9]),
        }

    # ── Apply action lên robot ───────────────────────────────────────────────

    @staticmethod
    def apply_action_to_robot(robot, parsed_action: dict,
                               left_arm_positions: np.ndarray,
                               left_finger_positions: np.ndarray):
        """Gửi action xuống robot. Cánh tay trái giữ nguyên."""
        full_arm     = np.concatenate([left_arm_positions, parsed_action["r_arm_joints"]])
        full_fingers = np.concatenate([left_finger_positions, parsed_action["r_fingers"]])

        robot.set_arm_joint_positions(full_arm.tolist())
        robot.set_finger_positions(full_fingers.tolist())

        if parsed_action["right_gripper"] > GRIPPER_CLOSE_THRESHOLD:
            robot.close_gripper(side="right")
        else:
            robot.open_gripper(side="right")

    # ── One-call convenience ─────────────────────────────────────────────────

    def step(
        self,
        robot,
        part_poses: list[dict],
        gripper_control: list[float] | None = None,
        apply: bool = True,
    ) -> np.ndarray:
        """
        Full pipeline: đọc state → predict → (tùy chọn) apply.

        Returns:
            action (10,) ở đơn vị radian
        """
        joint_states = robot.get_joint_states()
        if joint_states is None:
            return np.zeros(cfg.ACTION_DIM, dtype=np.float32)

        state  = self.build_state_vector(joint_states, part_poses, gripper_control)
        action = self.predict(state)

        if apply:
            parsed       = self.parse_action(action)
            left_arm     = np.array(joint_states["arm_positions"][:7],    dtype=np.float32)
            left_fingers = np.array(joint_states["finger_positions"][:2], dtype=np.float32)
            self.apply_action_to_robot(robot, parsed, left_arm, left_fingers)

        return action
