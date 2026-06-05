"""
PartSortingEnv — Isaac Lab DirectRLEnv cho Part Sorting Long.

Task: gắp 4 parts và đặt đúng bin:
  - part0, part1 (PartA) → BinA (y ≈ 0.10)
  - part2, part3 (PartB) → BinB (y ≈ 0.50)

Observation (38-dim):
    [0:7]   right arm joints (rad)
    [7:9]   right finger joints (rad)
    [9]     gripper state (-1=open, +1=close)
    [10:38] 4 objects × [x, y, z, qx, qy, qz, qw]

Action (10-dim, [-1, 1]):
    [0:7]  right arm targets → [-2.5, 2.5] rad
    [7:9]  right finger targets → [0, 0.04] rad
    [9]    gripper command
"""

from __future__ import annotations

import math
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from env_cfg import PartSortingEnvCfg, USD_TABLE, USD_BOX


class PartSortingEnv(DirectRLEnv):
    """Part Sorting environment: sort 4 parts vào đúng 2 bins."""

    cfg: PartSortingEnvCfg

    def __init__(self, cfg: PartSortingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ── Unnormalize scales ───────────────────────────────────────────────
        r_min, r_max = cfg.r_arm_joint_min, cfg.r_arm_joint_max
        f_min, f_max = cfg.r_finger_min,    cfg.r_finger_max
        self._arm_scale  = torch.tensor((r_max - r_min) / 2.0, device=self.device)
        self._arm_bias   = torch.tensor((r_max + r_min) / 2.0, device=self.device)
        self._fing_scale = torch.tensor((f_max - f_min) / 2.0, device=self.device)
        self._fing_bias  = torch.tensor((f_max + f_min) / 2.0, device=self.device)

        # ── Target positions per part type ───────────────────────────────────
        # part0, part1 (PartA) → bin_a_targets[0], bin_a_targets[1]
        # part2, part3 (PartB) → bin_b_targets[0], bin_b_targets[1]
        bin_a = torch.tensor(cfg.bin_a_targets, dtype=torch.float32, device=self.device)  # (2, 3)
        bin_b = torch.tensor(cfg.bin_b_targets, dtype=torch.float32, device=self.device)  # (2, 3)

        # _part_targets[i]: target position của part i, shape (3,)
        self._part_targets = torch.stack([
            bin_a[0], bin_a[1],  # part0 → binA slot 0, part1 → binA slot 1
            bin_b[0], bin_b[1],  # part2 → binB slot 0, part3 → binB slot 1
        ], dim=0)  # (4, 3)

        # ── Scatter ──────────────────────────────────────────────────────────
        sc = cfg.scatter_center
        self._scatter_center = torch.tensor(sc, dtype=torch.float32, device=self.device)
        self._scatter_half_x = cfg.scatter_half_x
        self._scatter_half_y = cfg.scatter_half_y

        # ── Gripper state ────────────────────────────────────────────────────
        self._gripper_state = torch.full(
            (self.num_envs,), -1.0, dtype=torch.float32, device=self.device
        )

        # ── Grasp signal from wrist camera (0=open/empty, 1=grasped) ─────────
        self._grasp_signal      = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._prev_grasp_signal = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # ── Left arm fixed targets ────────────────────────────────────────────
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
        ).unsqueeze(0).expand(self.num_envs, -1)

        self._resolve_joint_ids()

    # ── Scene setup ──────────────────────────────────────────────────────────

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.parts: list[RigidObject] = []
        for i in range(4):
            part = RigidObject(getattr(self.cfg, f"part{i}"))
            self.scene.rigid_objects[f"part{i}"] = part
            self.parts.append(part)

        self.finger_contact = ContactSensor(self.cfg.finger_contact)
        self.scene.sensors["finger_contact"] = self.finger_contact

        # Table
        table_spawn = sim_utils.UsdFileCfg(
            usd_path=USD_TABLE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
        table_spawn.func(
            "/World/envs/env_.*/Table", table_spawn,
            translation=(0.75, 0.3, 0.5),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        # BinA (cho PartA)
        bin_a_spawn = sim_utils.UsdFileCfg(
            usd_path=USD_BOX,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
        bin_a_spawn.func(
            "/World/envs/env_.*/BinA", bin_a_spawn,
            translation=self.cfg.bin_a_pos,
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        # BinB (cho PartB)
        bin_b_spawn = sim_utils.UsdFileCfg(
            usd_path=USD_BOX,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
        bin_b_spawn.func(
            "/World/envs/env_.*/BinB", bin_b_spawn,
            translation=self.cfg.bin_b_pos,
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/skyLight", light_cfg)

    def _resolve_joint_ids(self):
        r_arm_names   = ["R_shoulder_pitch_joint", "R_shoulder_roll_joint",
                         "R_shoulder_yaw_joint",   "R_elbow_roll_joint",
                         "R_elbow_yaw_joint",      "R_wrist_pitch_joint",
                         "R_wrist_roll_joint"]
        l_arm_names   = ["L_shoulder_pitch_joint", "L_shoulder_roll_joint",
                         "L_shoulder_yaw_joint",   "L_elbow_roll_joint",
                         "L_elbow_yaw_joint",      "L_wrist_pitch_joint",
                         "L_wrist_roll_joint"]
        r_finger_names = ["R_finger1_joint", "R_finger2_joint"]
        l_finger_names = ["L_finger1_joint", "L_finger2_joint"]

        self._r_arm_ids,    _ = self.robot.find_joints(r_arm_names)
        self._l_arm_ids,    _ = self.robot.find_joints(l_arm_names)
        self._r_finger_ids, _ = self.robot.find_joints(r_finger_names)
        self._l_finger_ids, _ = self.robot.find_joints(l_finger_names)

        tcp_candidates = ["R_sixforce_link", "R_wrist_roll_link", "R_wrist_link"]
        self._tcp_body_idx = None
        for name in tcp_candidates:
            idxs, _ = self.robot.find_bodies(name)
            if idxs:
                self._tcp_body_idx = idxs[0]
                break

        self._r_finger1_body_idx = None
        for name in ["R_finger1_link", "R_finger1"]:
            idxs, _ = self.robot.find_bodies(name)
            if idxs:
                self._r_finger1_body_idx = idxs[0]
                break

        self._r_finger2_body_idx = None
        for name in ["R_finger2_link", "R_finger2"]:
            idxs, _ = self.robot.find_bodies(name)
            if idxs:
                self._r_finger2_body_idx = idxs[0]
                break

        self._joint_targets = self.robot.data.default_joint_pos.clone()

    # ── Physics step ─────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

        r_arm_targets  = self._actions[:, :7] * self._arm_scale  + self._arm_bias
        r_fing_targets = self._actions[:, 7:9] * self._fing_scale + self._fing_bias

        self._gripper_state = torch.where(
            self._actions[:, 9] > 0.0,
            torch.ones_like(self._gripper_state),
            -torch.ones_like(self._gripper_state),
        )

        gripper_finger = torch.where(
            self._gripper_state > 0,
            torch.full_like(self._gripper_state, self.cfg.r_finger_max),
            torch.full_like(self._gripper_state, self.cfg.r_finger_min),
        ).unsqueeze(1).expand(-1, 2)

        self._joint_targets[:, self._r_arm_ids]    = r_arm_targets
        self._joint_targets[:, self._r_finger_ids] = gripper_finger

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_targets)
        self.robot.write_data_to_sim()

    # ── Observations ─────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos

        r_arm   = joint_pos[:, self._r_arm_ids]
        r_fing  = joint_pos[:, self._r_finger_ids]
        gripper = self._gripper_state.unsqueeze(1)
        tcp_pos = self._get_tcp_pos()  # (N, 3)

        # Fingertip positions — fallback to TCP nếu không tìm thấy body
        body_pos = self.robot.data.body_pos_w
        finger1_pos = body_pos[:, self._r_finger1_body_idx, :] if self._r_finger1_body_idx is not None else tcp_pos
        finger2_pos = body_pos[:, self._r_finger2_body_idx, :] if self._r_finger2_body_idx is not None else tcp_pos

        obj_vecs = []
        for part in self.parts:
            pos  = part.data.root_pos_w
            quat = part.data.root_quat_w  # [w, x, y, z]
            obj_vecs.append(torch.cat([pos, quat[:, 1:2], quat[:, 2:3], quat[:, 3:4], quat[:, 0:1]], dim=-1))

        # Grasp detection: contact force trên ngón tay phải với bất kỳ part nào
        self._prev_grasp_signal = self._grasp_signal.clone()
        contact_forces = self.finger_contact.data.net_forces_w  # (N, 1, 3)
        contact_mag = torch.linalg.norm(contact_forces[:, 0, :], dim=-1)  # (N,)
        self._grasp_signal = (contact_mag > self.cfg.grasp_contact_threshold).float()
        grasp_obs = self._grasp_signal.unsqueeze(1)  # (N, 1)

        # Sorted status: 4 bits cho policy biết vật nào đã xong
        obj_positions_now = torch.stack([p.data.root_pos_w for p in self.parts], dim=1)
        sorted_status = self._get_sorted_mask(obj_positions_now).float()  # (N, 4)

        obs = torch.cat([r_arm, r_fing, gripper, tcp_pos, finger1_pos, finger2_pos, *obj_vecs, grasp_obs, sorted_status], dim=-1)  # (N, 52)
        return {"policy": obs}

    # ── Rewards ──────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        tcp_pos = self._get_tcp_pos()  # (N, 3)

        obj_positions = torch.stack(
            [p.data.root_pos_w for p in self.parts], dim=1
        )  # (N, 4, 3)

        # 1. Kéo TCP đến gần vật nào chưa vào đúng bin
        not_sorted = ~self._get_sorted_mask(obj_positions)  # (N, 4) bool
        dist_tcp_to_objs = torch.linalg.norm(
            tcp_pos.unsqueeze(1) - obj_positions, dim=-1
        )  # (N, 4)
        # Chỉ tính khoảng cách đến vật chưa sort xong; vật đã sort thì ignore
        dist_tcp_to_unsorted = dist_tcp_to_objs.clone()
        dist_tcp_to_unsorted[~not_sorted] = 1e3  # loại ra nếu đã vào bin
        min_dist_to_obj, nearest_idx = dist_tcp_to_unsorted.min(dim=-1)
        reward += -min_dist_to_obj.clamp(max=2.0) * 1.0

        # 2. Khi đang gắp: dùng wrist camera signal thay vì proximity heuristic
        is_grasping = self._grasp_signal.bool()  # (N,)
        nearest_obj_pos = obj_positions[
            torch.arange(self.num_envs, device=self.device), nearest_idx
        ]  # (N, 3)
        correct_target = self._part_targets[nearest_idx]  # (N, 3) — target đúng loại
        dist_obj_to_target = torch.linalg.norm(nearest_obj_pos - correct_target, dim=-1)
        reward += is_grasping.float() * (-dist_obj_to_target * 2.0 + 5.0)

        # 3. Release bonus: vừa thả (prev=1 → curr=0) và vật gần đúng bin
        just_released = self._prev_grasp_signal.bool() & ~self._grasp_signal.bool()
        in_bin_radius = dist_obj_to_target < (self.cfg.success_threshold * 2.0)
        reward += just_released.float() * in_bin_radius.float() * self.cfg.release_bonus

        # 4. Bonus per vật đã vào đúng bin (mỗi step)
        n_sorted = self._get_sorted_mask(obj_positions).sum(dim=-1).float()  # (N,)
        reward += n_sorted * 20.0

        # 5. Step penalty
        reward += -0.01

        return reward

    def _get_sorted_mask(self, obj_positions: torch.Tensor) -> torch.Tensor:
        """Trả về (N, 4) bool: True nếu part i đã vào đúng bin."""
        targets = self._part_targets.unsqueeze(0)  # (1, 4, 3)
        dists   = torch.linalg.norm(obj_positions - targets, dim=-1)  # (N, 4)
        return dists < self.cfg.success_threshold

    def _get_tcp_pos(self) -> torch.Tensor:
        if self._tcp_body_idx is not None:
            return self.robot.data.body_pos_w[:, self._tcp_body_idx, :]
        return self.robot.data.body_pos_w[:, -1, :]

    # ── Termination ──────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_positions = torch.stack(
            [p.data.root_pos_w for p in self.parts], dim=1
        )

        # Success: tất cả 4 vật đúng bin
        terminated = self._get_sorted_mask(obj_positions).all(dim=-1)

        # Timeout
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        # Vật rơi khỏi bàn
        min_z = obj_positions[:, :, 2].min(dim=-1).values
        truncated = truncated | (min_z < self.cfg.table_z_min)

        return terminated, truncated

    # ── Reset ────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return

        super()._reset_idx(env_ids)

        # Reset robot
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._joint_targets[env_ids]      = joint_pos
        self._gripper_state[env_ids]      = -1.0
        self._grasp_signal[env_ids]       = 0.0
        self._prev_grasp_signal[env_ids]  = 0.0

        # Scatter parts ngẫu nhiên (không chồng lên nhau)
        for part in self.parts:
            rand_x = (torch.rand(n, device=self.device) * 2 - 1) * self._scatter_half_x
            rand_y = (torch.rand(n, device=self.device) * 2 - 1) * self._scatter_half_y
            scatter_pos = self._scatter_center.unsqueeze(0).expand(n, -1).clone()
            scatter_pos[:, 0] += rand_x
            scatter_pos[:, 1] += rand_y

            rand_yaw = torch.rand(n, device=self.device) * 2 * math.pi
            half_yaw = rand_yaw * 0.5
            quat = torch.stack([
                torch.cos(half_yaw),
                torch.zeros(n, device=self.device),
                torch.zeros(n, device=self.device),
                torch.sin(half_yaw),
            ], dim=-1)

            root_state = torch.cat(
                [scatter_pos, quat, torch.zeros(n, 6, device=self.device)], dim=-1
            )
            part.write_root_state_to_sim(root_state, env_ids=env_ids)

        self.robot.write_data_to_sim()
        for part in self.parts:
            part.write_data_to_sim()
