"""
PartSortingEnvCfg — Scene + physics config cho Part Sorting Long (Isaac Lab + PPO).

Task: sort 4 parts (2×PartA, 2×PartB) vào đúng bin.
  - part0, part1 (PartA) → BinA
  - part2, part3 (PartB) → BinB
"""

from __future__ import annotations

import os
import yaml

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Resolve asset paths từ configs/Part_Sorting.yaml
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CONFIG_FILE  = os.path.join(_PROJECT_ROOT, "configs", "Part_Sorting.yaml")


def _resolve_assets_root() -> str:
    with open(_CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)
    return os.path.abspath(os.path.join(os.path.dirname(_CONFIG_FILE), cfg["root_path"]))


ASSETS_ROOT = _resolve_assets_root()

USD_ROBOT  = f"{ASSETS_ROOT}/Collected_s2_v1_ecbg/s2_v1.usd"
USD_PART_A = f"{ASSETS_ROOT}/Collected_Task1_PartA_ori_color/Task1_PartA.usd"
USD_PART_B = f"{ASSETS_ROOT}/Collected_Part_B_ori_color/Part_B.usd"
USD_TABLE  = f"{ASSETS_ROOT}/Collected_table_v2/table_v2.usd"
USD_BOX    = f"{ASSETS_ROOT}/Box_blank/box_60_40_23_cut_0.usd"

# ---------------------------------------------------------------------------
# Joint init positions
# ---------------------------------------------------------------------------
_ROBOT_INIT_JOINTS: dict[str, float] = {
    "L_shoulder_pitch_joint":  0.09322,
    "L_shoulder_roll_joint":  -0.59332,
    "L_shoulder_yaw_joint":   -1.59588,
    "L_elbow_roll_joint":     -1.89636,
    "L_elbow_yaw_joint":       1.40005,
    "L_wrist_pitch_joint":    -0.00049,
    "L_wrist_roll_joint":      0.09987,
    "R_shoulder_pitch_joint": -0.09322,
    "R_shoulder_roll_joint":  -0.59335,
    "R_shoulder_yaw_joint":    1.59587,
    "R_elbow_roll_joint":     -1.89636,
    "R_elbow_yaw_joint":      -1.40009,
    "R_wrist_pitch_joint":     0.00048,
    "R_wrist_roll_joint":      0.09985,
    "head_pitch_joint":       -0.60095,
    "head_yaw_joint":          0.0,
    "L_finger1_joint":         0.0,
    "L_finger2_joint":         0.0,
    "R_finger1_joint":         0.0,
    "R_finger2_joint":         0.0,
}

_PART_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=4,
    solver_velocity_iteration_count=0,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)
_PART_MASS_PROPS = sim_utils.MassPropertiesCfg(mass=0.1)


def _make_part_cfg(prim_suffix: str, usd_path: str, init_pos: tuple) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{prim_suffix}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=_PART_RIGID_PROPS,
            mass_props=_PART_MASS_PROPS,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos),
    )


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------
@configclass
class PartSortingEnvCfg(DirectRLEnvCfg):
    """Config cho Part Sorting task với 2 bins riêng biệt.

    Observation (38-dim):
        [0:7]   right arm joints (rad)
        [7:9]   right finger joints (rad)
        [9]     gripper control (-1=open, +1=close)
        [10:38] 4 objects × 7 = [x, y, z, qx, qy, qz, qw]

    Action (10-dim, [-1, 1]):
        [0:7]  right arm joint targets → unnormalize to ±2.5 rad
        [7:9]  right finger targets → unnormalize to [0, 0.04] rad
        [9]    gripper command (>0 = close, ≤0 = open)

    Sorting rule:
        part0, part1 (PartA) → BinA tại y ≈ 0.10
        part2, part3 (PartB) → BinB tại y ≈ 0.50
    """

    # --- Simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=3,
        physx=sim_utils.PhysxCfg(
            gpu_collision_stack_size=2**30,  # 1 GB — prevents buffer overflow with 512+ envs
        ),
    )
    decimation: int = 3
    episode_length_s: float = 30.0  # dài hơn task1 vì 4 grasps riêng biệt

    # --- Obs / act ---
    observation_space: int = 38
    action_space: int = 10
    state_space: int = 0

    # --- Scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # --- Robot ---
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_ROBOT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.7, -0.2, 0.9),
            rot=(0.0, 0.0, 0.7071, 0.7071),
            joint_pos=_ROBOT_INIT_JOINTS,
        ),
        actuators={
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=[
                    "[LR]_shoulder_pitch_joint", "[LR]_shoulder_roll_joint",
                    "[LR]_shoulder_yaw_joint",   "[LR]_elbow_roll_joint",
                    "[LR]_elbow_yaw_joint",      "[LR]_wrist_pitch_joint",
                    "[LR]_wrist_roll_joint",
                ],
                stiffness=800.0,
                damping=40.0,
            ),
            "finger_joints": ImplicitActuatorCfg(
                joint_names_expr=["[LR]_finger1_joint", "[LR]_finger2_joint"],
                stiffness=200.0,
                damping=10.0,
            ),
        },
    )

    # --- 4 Parts: 2x PartA (part0,1), 2x PartB (part2,3) ---
    part0: RigidObjectCfg = _make_part_cfg("Part0", USD_PART_A, (0.68, 0.22, 1.10))
    part1: RigidObjectCfg = _make_part_cfg("Part1", USD_PART_A, (0.72, 0.22, 1.10))
    part2: RigidObjectCfg = _make_part_cfg("Part2", USD_PART_B, (0.78, 0.34, 1.10))
    part3: RigidObjectCfg = _make_part_cfg("Part3", USD_PART_B, (0.82, 0.34, 1.10))

    # --- Task parameters ---
    success_threshold: float = 0.08   # 8cm — vật coi như trong bin
    table_z_min: float = 0.90         # z thấp hơn = vật rơi

    # BinA (cho PartA - part0, part1): vị trí x=1.2, y=0.10
    bin_a_pos: tuple = (1.2, 0.10, 1.05)
    bin_a_targets: tuple = (
        (1.15, 0.08, 1.10),
        (1.25, 0.08, 1.10),
    )

    # BinB (cho PartB - part2, part3): vị trí x=1.2, y=0.50
    bin_b_pos: tuple = (1.2, 0.50, 1.05)
    bin_b_targets: tuple = (
        (1.15, 0.52, 1.10),
        (1.25, 0.52, 1.10),
    )

    # Vùng scatter ngẫu nhiên khi reset
    scatter_center: tuple = (0.75, 0.28, 1.10)
    scatter_half_x: float = 0.15   # x ∈ [0.60, 0.90]
    scatter_half_y: float = 0.10   # y ∈ [0.18, 0.38]

    # Joint range để unnormalize action
    r_arm_joint_min: float = -2.5
    r_arm_joint_max: float = 2.5
    r_finger_min: float = 0.0
    r_finger_max: float = 0.04
