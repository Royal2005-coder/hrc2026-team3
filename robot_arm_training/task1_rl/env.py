"""PickPlaceEnv — Isaac Lab DirectRLEnv cho Task 1 pick-and-place.

Cách chạy (qua train.py):
    python robot_arm_training/task1_rl/train.py --num_envs 512 --headless
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from env_cfg import PickPlaceEnvCfg


class PickPlaceEnv(DirectRLEnv):
    cfg: PickPlaceEnvCfg

    def __init__(self, cfg: PickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ── Joint indices (resolved sau super().__init__()) ─────────────────
        self._r_arm_ids, _ = self.robot.find_joints([
            "R_shoulder_pitch_joint", "R_shoulder_roll_joint",
            "R_shoulder_yaw_joint",   "R_elbow_roll_joint",
            "R_elbow_yaw_joint",      "R_wrist_pitch_joint",
            "R_wrist_roll_joint",
        ])
        self._l_arm_ids, _ = self.robot.find_joints([
            "L_shoulder_pitch_joint", "L_shoulder_roll_joint",
            "L_shoulder_yaw_joint",   "L_elbow_roll_joint",
            "L_elbow_yaw_joint",      "L_wrist_pitch_joint",
            "L_wrist_roll_joint",
        ])
        self._r_fing_ids, _ = self.robot.find_joints(["R_finger1_joint", "R_finger2_joint"])
        self._l_fing_ids, _ = self.robot.find_joints(["L_finger1_joint", "L_finger2_joint"])

        # ── TCP body index (tay phải) ────────────────────────────────────────
        self._tcp_body_idx = None
        for name in ["R_sixforce_link", "R_wrist_roll_link", "R_wrist_pitch_link"]:
            idxs, _ = self.robot.find_bodies(name)
            if len(idxs) > 0:
                self._tcp_body_idx = idxs[0]
                print(f"[PickPlaceEnv] TCP body: {name} (idx={idxs[0]})")
                break
        if self._tcp_body_idx is None:
            print("[PickPlaceEnv] WARNING: TCP body not found, using last body as fallback")

        # ── Action scaling ───────────────────────────────────────────────────
        r_min, r_max = cfg.r_arm_joint_min, cfg.r_arm_joint_max
        f_min, f_max = cfg.r_finger_open,   cfg.r_finger_close
        self._arm_scale  = (r_max - r_min) / 2.0
        self._arm_bias   = (r_max + r_min) / 2.0
        self._fing_scale = (f_max - f_min) / 2.0
        self._fing_bias  = (f_max + f_min) / 2.0

        # ── Gripper state buffer ─────────────────────────────────────────────
        self._gripper_state = torch.full(
            (self.num_envs,), -1.0, dtype=torch.float32, device=self.device
        )

        # ── Joint target buffer (khởi tạo ở init position) ─────────────────
        self._joint_tgt = self.robot.data.default_joint_pos.clone()  # (N, n_joints)

        # ── Box world positions: local + env_origins ────────────────────────
        box_local = torch.tensor(cfg.box_local_pos, dtype=torch.float32, device=self.device)
        self._box_pos_w = self.scene.env_origins + box_local.unsqueeze(0)  # (N, 3)

        # ── Scatter params (local frame) ─────────────────────────────────────
        self._scatter_local = torch.tensor(
            cfg.scatter_center, dtype=torch.float32, device=self.device
        )

    # ── Scene setup ──────────────────────────────────────────────────────────

    def _setup_scene(self):
        # 1. Tạo robot và 4 parts (staged tại env_0)
        self.robot = Articulation(self.cfg.robot)

        self.parts: list[RigidObject] = []
        for i in range(4):
            self.parts.append(RigidObject(getattr(self.cfg, f"part{i}")))

        # 2. Spawn table và box (kinematic, per-env, staged tại env_0)
        table_cfg = sim_utils.UsdFileCfg(
            usd_path=self.cfg.table_usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
        table_cfg.func(
            "/World/envs/env_0/Table", table_cfg,
            translation=(0.75, 0.3, 0.5),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        box_cfg = sim_utils.UsdFileCfg(
            usd_path=self.cfg.box_usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
        box_cfg.func(
            "/World/envs/env_0/Box", box_cfg,
            translation=(1.2, 0.3, 1.05),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        # 3. Ground plane (global, không nhân bản)
        spawn_ground_plane("/World/ground", GroundPlaneCfg())

        # 4. Clone tất cả envs từ env_0
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # 5. Đăng ký vào scene manager
        self.scene.articulations["robot"] = self.robot
        for i, part in enumerate(self.parts):
            self.scene.rigid_objects[f"part{i}"] = part

        # 6. Ánh sáng
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── Physics step ─────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Unnormalize right arm targets
        r_arm = self._actions[:, :7] * self._arm_scale + self._arm_bias   # (N, 7)

        # Gripper command → finger targets (action[:, 9] > 0 = close)
        self._gripper_state = torch.where(
            self._actions[:, 9] > 0.0,
            torch.ones(self.num_envs, device=self.device),
            -torch.ones(self.num_envs, device=self.device),
        )
        fing_tgt = torch.where(
            self._gripper_state.unsqueeze(1) > 0.0,
            torch.full((self.num_envs, 2), self.cfg.r_finger_close, device=self.device),
            torch.full((self.num_envs, 2), self.cfg.r_finger_open,  device=self.device),
        )  # (N, 2)

        # Chỉ cập nhật right arm và right finger; left arm giữ nguyên init target
        self._joint_tgt[:, self._r_arm_ids]  = r_arm
        self._joint_tgt[:, self._r_fing_ids] = fing_tgt

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_tgt)

    # ── Observations ─────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        jp = self.robot.data.joint_pos  # (N, n_joints)

        r_arm   = jp[:, self._r_arm_ids]    # (N, 7)
        r_fing  = jp[:, self._r_fing_ids]   # (N, 2)
        gripper = self._gripper_state.unsqueeze(1)  # (N, 1)

        # 4 object poses: [x,y,z, qx,qy,qz,qw] — Isaac convention: quat=[w,x,y,z]
        obj_vecs = []
        for part in self.parts:
            pos  = part.data.root_pos_w   # (N, 3)
            quat = part.data.root_quat_w  # (N, 4) [w,x,y,z]
            # Reorder sang [qx, qy, qz, qw]
            obj_vecs.append(torch.cat([pos, quat[:, 1:], quat[:, :1]], dim=-1))  # (N, 7)
        obj_obs = torch.cat(obj_vecs, dim=-1)  # (N, 28)

        obs = torch.cat([r_arm, r_fing, gripper, obj_obs], dim=-1)  # (N, 38)
        return {"policy": obs}

    # ── Rewards ──────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # TCP position
        if self._tcp_body_idx is not None:
            tcp = self.robot.data.body_pos_w[:, self._tcp_body_idx, :]  # (N, 3)
        else:
            tcp = self.robot.data.body_pos_w[:, -1, :]

        # Object world positions
        obj_pos = torch.stack([p.data.root_pos_w for p in self.parts], dim=1)  # (N, 4, 3)

        # 1. Kéo TCP đến vật gần nhất
        dist_tcp = torch.linalg.norm(tcp.unsqueeze(1) - obj_pos, dim=-1)  # (N, 4)
        min_dist, near_idx = dist_tcp.min(dim=-1)  # (N,)
        reward += -min_dist * 1.0

        # 2. Khi đang gắp: kéo vật về hộp
        is_grasping = (min_dist < 0.05) & (self._gripper_state > 0.0)
        near_obj = obj_pos[torch.arange(self.num_envs, device=self.device), near_idx]  # (N, 3)
        dist_to_box = torch.linalg.norm(near_obj - self._box_pos_w, dim=-1)  # (N,)
        reward += is_grasping.float() * (-dist_to_box * 2.0 + 5.0)

        # 3. Bonus per vật đã vào hộp
        dists_all = torch.linalg.norm(
            obj_pos - self._box_pos_w.unsqueeze(1), dim=-1
        )  # (N, 4)
        n_in_box = (dists_all < self.cfg.success_threshold).sum(dim=-1).float()
        reward += n_in_box * 20.0

        # 4. Step penalty
        reward -= 0.01

        return reward

    # ── Termination ──────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_pos = torch.stack([p.data.root_pos_w for p in self.parts], dim=1)  # (N, 4, 3)

        # Success: tất cả 4 vật trong hộp
        dists = torch.linalg.norm(obj_pos - self._box_pos_w.unsqueeze(1), dim=-1)  # (N, 4)
        terminated = (dists < self.cfg.success_threshold).all(dim=-1)              # (N,)

        # Timeout
        timed_out = self.episode_length_buf >= self.max_episode_length - 1

        # Vật rơi khỏi bàn → truncate
        fallen = obj_pos[:, :, 2].min(dim=-1).values < self.cfg.table_z_min
        truncated = timed_out | fallen

        return terminated, truncated

    # ── Reset ────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)

        # ── Reset robot ──────────────────────────────────────────────────────
        default_root = self.robot.data.default_root_state[env_ids].clone()
        default_root[:, :3] += self.scene.env_origins[env_ids]  # local → world

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)

        self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Sync joint target buffer
        self._joint_tgt[env_ids] = joint_pos
        self._gripper_state[env_ids] = -1.0  # open

        # ── Scatter parts ngẫu nhiên trên bàn ───────────────────────────────
        origins = self.scene.env_origins[env_ids]  # (n, 3)
        for part in self.parts:
            rand_x = (torch.rand(n, device=self.device) * 2 - 1) * self.cfg.scatter_half_x
            rand_y = (torch.rand(n, device=self.device) * 2 - 1) * self.cfg.scatter_half_y
            pos_local = self._scatter_local.unsqueeze(0).expand(n, -1).clone()
            pos_local[:, 0] += rand_x
            pos_local[:, 1] += rand_y
            pos_world = pos_local + origins  # (n, 3)

            # Random yaw rotation
            yaw = torch.rand(n, device=self.device) * 2.0 * math.pi
            quat = torch.stack([
                torch.cos(yaw * 0.5),                       # w
                torch.zeros(n, device=self.device),          # x
                torch.zeros(n, device=self.device),          # y
                torch.sin(yaw * 0.5),                       # z
            ], dim=-1)  # (n, 4) [w,x,y,z]

            pose = torch.cat([pos_world, quat], dim=-1)         # (n, 7)
            vel  = torch.zeros(n, 6, device=self.device)
            part.write_root_pose_to_sim(pose, env_ids)
            part.write_root_velocity_to_sim(vel, env_ids)
