"""
PickPlaceEnv — Isaac Lab DirectRLEnv cho Task 1 pick-and-place.

Cách chạy (qua train.py, không chạy trực tiếp):
    python robot_arm_training/task1_isaaclab/train.py \
        --headless --num_envs 512

Observation (38-dim):
    [0:7]   right arm joint positions (rad)
    [7:9]   right finger joint positions (rad)
    [9]     gripper control state (-1=open, +1=close)
    [10:38] 4 objects × [x, y, z, qx, qy, qz, qw]

Action (10-dim, [-1, 1]):
    [0:7]  right arm targets → map sang [-2.5, 2.5] rad
    [7:9]  right finger targets → map sang [0, 0.04] rad
    [9]    gripper command (>0 = close, ≤0 = open)
"""

from __future__ import annotations

import math
import torch
import numpy as np

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from env_cfg import PickPlaceEnvCfg, USD_TABLE, USD_BOX


class PickPlaceEnv(DirectRLEnv):
    """Pick-and-place environment sử dụng Isaac Lab DirectRLEnv.

    Phiên bản này điều khiển cánh tay PHẢI của robot S2 để gắp 4 vật
    (2 PartA + 2 PartB) trải ngẫu nhiên trên bàn và đặt vào hộp.
    Cánh tay trái được giữ cố định ở vị trí init.
    """

    cfg: PickPlaceEnvCfg

    def __init__(self, cfg: PickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ── Giá trị unnormalize cho action ─────────────────────────────────
        r_min, r_max = cfg.r_arm_joint_min, cfg.r_arm_joint_max
        f_min, f_max = cfg.r_finger_min,    cfg.r_finger_max
        self._arm_scale  = torch.tensor((r_max - r_min) / 2.0, device=self.device)
        self._arm_bias   = torch.tensor((r_max + r_min) / 2.0, device=self.device)
        self._fing_scale = torch.tensor((f_max - f_min) / 2.0, device=self.device)
        self._fing_bias  = torch.tensor((f_max + f_min) / 2.0, device=self.device)

        # ── Target positions (4 điểm trong hộp) ────────────────────────────
        self._box_targets = torch.tensor(
            cfg.box_targets, dtype=torch.float32, device=self.device
        )  # (4, 3)

        # ── Scatter params ──────────────────────────────────────────────────
        sc = cfg.scatter_center
        self._scatter_center = torch.tensor(sc, dtype=torch.float32, device=self.device)
        self._scatter_half_x = cfg.scatter_half_x
        self._scatter_half_y = cfg.scatter_half_y

        # ── Gripper state buffer: -1 = open, +1 = close ────────────────────
        self._gripper_state = torch.full(
            (self.num_envs,), -1.0, dtype=torch.float32, device=self.device
        )

        # ── Left arm joint targets (cố định, giữ nguyên suốt episode) ──────
        l_arm_init = [
            cfg.robot.init_state.joint_pos.get(j, 0.0)
            for j in [
                "L_shoulder_pitch_joint", "L_shoulder_roll_joint",
                "L_shoulder_yaw_joint",   "L_elbow_roll_joint",
                "L_elbow_yaw_joint",      "L_wrist_pitch_joint",
                "L_wrist_roll_joint",
            ]
        ]
        self._l_arm_targets = torch.tensor(
            l_arm_init, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)  # (num_envs, 7)

        # ── Resolve joint indices (physics view đã sẵn sàng sau super().__init__()) ──
        self._resolve_joint_ids()

    # ── Scene setup ──────────────────────────────────────────────────────────

    def _setup_scene(self):
        # Robot
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        # 4 Parts
        self.parts: list[RigidObject] = []
        for i in range(4):
            part = RigidObject(getattr(self.cfg, f"part{i}"))
            self.scene.rigid_objects[f"part{i}"] = part
            self.parts.append(part)

        # Table (static collider, không cần tracking)
        table_spawn = sim_utils.UsdFileCfg(
            usd_path=USD_TABLE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
        table_spawn.func(
            "/World/envs/env_.*/Table",
            table_spawn,
            translation=(0.75, 0.3, 0.5),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        # Box (static collider)
        box_spawn = sim_utils.UsdFileCfg(
            usd_path=USD_BOX,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
        box_spawn.func(
            "/World/envs/env_.*/Box",
            box_spawn,
            translation=(1.2, 0.3, 1.05),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        # Ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # Ánh sáng
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/skyLight", light_cfg)
        # _resolve_joint_ids() gọi sau super().__init__() vì physics view
        # chưa sẵn sàng ở đây (cần sim.play() + scene.initialize() trước)

    def _resolve_joint_ids(self):
        """Map tên khớp sang chỉ số Isaac Lab."""
        r_arm_names = [
            "R_shoulder_pitch_joint", "R_shoulder_roll_joint",
            "R_shoulder_yaw_joint",   "R_elbow_roll_joint",
            "R_elbow_yaw_joint",      "R_wrist_pitch_joint",
            "R_wrist_roll_joint",
        ]
        l_arm_names = [
            "L_shoulder_pitch_joint", "L_shoulder_roll_joint",
            "L_shoulder_yaw_joint",   "L_elbow_roll_joint",
            "L_elbow_yaw_joint",      "L_wrist_pitch_joint",
            "L_wrist_roll_joint",
        ]
        r_finger_names = ["R_finger1_joint", "R_finger2_joint"]
        l_finger_names = ["L_finger1_joint", "L_finger2_joint"]

        self._r_arm_ids,    _ = self.robot.find_joints(r_arm_names)
        self._l_arm_ids,    _ = self.robot.find_joints(l_arm_names)
        self._r_finger_ids, _ = self.robot.find_joints(r_finger_names)
        self._l_finger_ids, _ = self.robot.find_joints(l_finger_names)

        # TCP = link cuối tay phải (dùng body R_sixforce_link nếu có, else wrist link)
        tcp_candidates = ["R_sixforce_link", "R_wrist_roll_link", "R_wrist_link"]
        self._tcp_body_idx = None
        for name in tcp_candidates:
            idxs, names_found = self.robot.find_bodies(name)
            if idxs:
                self._tcp_body_idx = idxs[0]
                break

        # Khởi tạo target joints = init positions
        joint_pos_init = self.robot.data.default_joint_pos.clone()  # (num_envs, n_joints)
        self._joint_targets = joint_pos_init.clone()

    # ── Physics step ─────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Lưu action và chuẩn bị joint targets."""
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Unnormalize: [-1, 1] → actual joint space
        r_arm_targets = (
            self._actions[:, :7] * self._arm_scale + self._arm_bias
        )                                                              # (N, 7)
        r_fing_targets = (
            self._actions[:, 7:9] * self._fing_scale + self._fing_bias
        )                                                              # (N, 2)

        # Cập nhật gripper state từ action[:, 9]
        self._gripper_state = torch.where(
            self._actions[:, 9] > 0.0,
            torch.ones_like(self._gripper_state),    # close
            -torch.ones_like(self._gripper_state),   # open
        )

        # Gripper → finger target (close → max, open → min)
        gripper_finger = torch.where(
            self._gripper_state > 0,
            torch.full_like(self._gripper_state, self.cfg.r_finger_max),
            torch.full_like(self._gripper_state, self.cfg.r_finger_min),
        ).unsqueeze(1).expand(-1, 2)   # (N, 2)

        # Override r_fing_targets với gripper command (gripper command ưu tiên)
        r_fing_targets = gripper_finger

        # Ghi vào joint_targets theo đúng indices
        self._joint_targets[:, self._r_arm_ids]    = r_arm_targets
        self._joint_targets[:, self._r_finger_ids] = r_fing_targets
        # Cánh tay trái giữ nguyên init (đã set trong _reset_idx)

    def _apply_action(self) -> None:
        """Áp dụng joint position targets lên robot."""
        self.robot.set_joint_position_target(self._joint_targets)
        self.robot.write_data_to_sim()

    # ── Observations ─────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        """Trả về obs dict với key 'policy', shape (num_envs, 38)."""
        joint_pos = self.robot.data.joint_pos  # (N, n_joints)

        # Right arm (7) + right fingers (2) + gripper state (1)
        r_arm   = joint_pos[:, self._r_arm_ids]     # (N, 7)
        r_fing  = joint_pos[:, self._r_finger_ids]  # (N, 2)
        gripper = self._gripper_state.unsqueeze(1)  # (N, 1)

        # 4 object poses: [x, y, z, qx, qy, qz, qw] × 4 = 28-dim
        obj_vecs = []
        for part in self.parts:
            pos  = part.data.root_pos_w          # (N, 3)
            quat = part.data.root_quat_w         # (N, 4) [w, x, y, z] Isaac convention
            # Chuyển sang [x, y, z, qx, qy, qz, qw] theo format IL
            qx = quat[:, 1:2]
            qy = quat[:, 2:3]
            qz = quat[:, 3:4]
            qw = quat[:, 0:1]
            obj_vecs.append(torch.cat([pos, qx, qy, qz, qw], dim=-1))  # (N, 7)

        obj_obs = torch.cat(obj_vecs, dim=-1)  # (N, 28)

        obs = torch.cat([r_arm, r_fing, gripper, obj_obs], dim=-1)  # (N, 38)
        return {"policy": obs}

    # ── Rewards ──────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        """Dense reward function từ plan."""
        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Vị trí TCP (tay phải)
        tcp_pos = self._get_tcp_pos()  # (N, 3)

        # Vị trí 4 objects
        obj_positions = torch.stack(
            [p.data.root_pos_w for p in self.parts], dim=1
        )  # (N, 4, 3)

        # 1. Kéo tay phải đến gần vật gần nhất
        dist_tcp_to_objs = torch.linalg.norm(
            tcp_pos.unsqueeze(1) - obj_positions, dim=-1
        )  # (N, 4)
        min_dist_to_obj, nearest_idx = dist_tcp_to_objs.min(dim=-1)  # (N,)
        reward += -min_dist_to_obj * 1.0

        # 2. Kéo vật đến gần hộp khi đang gắp (proxy: tcp gần vật và gripper đóng)
        is_grasping = (min_dist_to_obj < 0.05) & (self._gripper_state > 0.0)  # (N,)
        nearest_obj_pos = obj_positions[
            torch.arange(self.num_envs, device=self.device), nearest_idx
        ]  # (N, 3)

        # Khoảng cách từ vật gần nhất đến target tương ứng trong hộp
        # (dùng nearest_idx để map vật → target slot)
        target_pos = self._box_targets[
            nearest_idx % 4
        ]  # (N, 3)
        dist_obj_to_box = torch.linalg.norm(nearest_obj_pos - target_pos, dim=-1)  # (N,)
        reward += is_grasping.float() * (-dist_obj_to_box * 2.0)

        # 3. Bonus khi gắp được vật
        reward += is_grasping.float() * 5.0

        # 4. Bonus khi đặt vật vào hộp thành công (từng vật)
        n_in_box = self._count_objects_in_box(obj_positions)  # (N,)
        reward += n_in_box.float() * 20.0

        # 5. Step penalty
        reward += -0.01

        return reward

    def _get_tcp_pos(self) -> torch.Tensor:
        """Vị trí TCP tay phải: dùng body cuối nếu có, fallback về wrist joint pos."""
        if self._tcp_body_idx is not None:
            return self.robot.data.body_pos_w[:, self._tcp_body_idx, :]  # (N, 3)
        # Fallback: dùng vị trí forward kinematics của wrist roll joint
        # (xấp xỉ, không chính xác nhưng đủ cho signal học)
        joint_pos = self.robot.data.joint_pos
        r_wrist_rad = joint_pos[:, self._r_arm_ids[-1]]   # R_wrist_roll_joint
        # Trả về pos gần đúng của wrist dựa trên body index nếu có
        return self.robot.data.body_pos_w[:, -1, :]  # last body as fallback

    def _count_objects_in_box(self, obj_positions: torch.Tensor) -> torch.Tensor:
        """Đếm số vật đang trong ngưỡng success_threshold so với target.

        Args:
            obj_positions: (N, 4, 3) positions của 4 objects.
        Returns:
            (N,) số vật thành công, dtype long.
        """
        targets = self._box_targets.unsqueeze(0)   # (1, 4, 3)
        dists   = torch.linalg.norm(obj_positions - targets, dim=-1)  # (N, 4)
        in_box  = dists < self.cfg.success_threshold  # (N, 4) bool
        return in_box.sum(dim=-1)                     # (N,) int

    # ── Termination ──────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            terminated: (N,) bool — success (tất cả 4 vật trong hộp)
            truncated:  (N,) bool — timeout (episode quá dài)
        """
        obj_positions = torch.stack(
            [p.data.root_pos_w for p in self.parts], dim=1
        )  # (N, 4, 3)

        # Success: tất cả 4 vật vào đúng hộp
        n_in_box   = self._count_objects_in_box(obj_positions)  # (N,)
        terminated = n_in_box >= 4                               # (N,) bool

        # Timeout
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        # Optional: vật rơi khỏi bàn → reset (tính là truncated để không penalize nặng)
        min_z = obj_positions[:, :, 2].min(dim=-1).values  # (N,)
        fallen = min_z < self.cfg.table_z_min
        truncated = truncated | fallen

        return terminated, truncated

    # ── Reset ────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset các environment chỉ định: robot về vị trí init, scatter parts."""
        n = len(env_ids)
        if n == 0:
            return

        super()._reset_idx(env_ids)

        # ── Reset robot ──────────────────────────────────────────────────────
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Cập nhật joint targets buffer
        self._joint_targets[env_ids] = joint_pos

        # Reset gripper state
        self._gripper_state[env_ids] = -1.0  # open

        # ── Scatter parts ngẫu nhiên trên bàn ───────────────────────────────
        for part in self.parts:
            rand_x = (torch.rand(n, device=self.device) * 2 - 1) * self._scatter_half_x
            rand_y = (torch.rand(n, device=self.device) * 2 - 1) * self._scatter_half_y
            scatter_pos = self._scatter_center.unsqueeze(0).expand(n, -1).clone()
            scatter_pos[:, 0] += rand_x
            scatter_pos[:, 1] += rand_y

            # Orientation: ngẫu nhiên quay quanh trục Z
            rand_yaw = torch.rand(n, device=self.device) * 2 * math.pi
            half_yaw = rand_yaw * 0.5
            quat = torch.stack([
                torch.cos(half_yaw),           # w
                torch.zeros(n, device=self.device),  # x
                torch.zeros(n, device=self.device),  # y
                torch.sin(half_yaw),           # z
            ], dim=-1)  # (N, 4) [w, x, y, z]

            root_state = torch.cat(
                [scatter_pos, quat, torch.zeros(n, 6, device=self.device)],
                dim=-1,
            )  # (N, 13): pos(3) + quat_wxyz(4) + lin_vel(3) + ang_vel(3)
            part.write_root_state_to_sim(root_state, env_ids=env_ids)

        # Write changes to sim
        self.robot.write_data_to_sim()
        for part in self.parts:
            part.write_data_to_sim()
