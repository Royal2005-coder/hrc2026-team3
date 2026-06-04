"""PickPlaceEnvCfg — scene + physics config cho Task 1 (Isaac Lab 0.54.2)."""

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
# Asset paths (resolved from configs/Part_Sorting.yaml)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))


def _assets_root() -> str:
    cfg_path = os.path.join(_PROJECT_ROOT, "configs", "Part_Sorting.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return os.path.abspath(os.path.join(os.path.dirname(cfg_path), cfg["root_path"]))


_R = _assets_root()  # /home/ubuntu/hrc2026/assets/resources

# ---------------------------------------------------------------------------
# Shared rigid body props for graspable parts
# ---------------------------------------------------------------------------
_PART_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)
_PART_MASS_PROPS = sim_utils.MassPropertiesCfg(mass=0.1)


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------
@configclass
class PickPlaceEnvCfg(DirectRLEnvCfg):
    """Config cho pick-and-place Task 1 với Isaac Lab DirectRLEnv.

    Observation (38-dim):
        [0:7]   right arm joint positions (rad)
        [7:9]   right finger joint positions (rad)
        [9]     gripper state (-1=open, +1=close)
        [10:38] 4 objects × [x,y,z, qx,qy,qz,qw]

    Action (10-dim, [-1,1]):
        [0:7]  right arm targets → [-2.5, 2.5] rad
        [7:9]  right finger targets → [0, 0.04] rad
        [9]    gripper command (>0=close, ≤0=open)
    """

    # Simulation
    decimation: int = 4
    episode_length_s: float = 20.0
    action_space: int = 10
    observation_space: int = 38
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=4)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.5, replicate_physics=True
    )

    # ── Robot ──────────────────────────────────────────────────────────────
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_R}/Collected_s2_v1_ecbg/s2_v1.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.7, -0.2, 0.9),
            rot=(0.0, 0.0, 0.7071, 0.7071),  # 90° quanh trục Z
            joint_pos={
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
            },
        ),
        actuators={
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    "[LR]_shoulder_pitch_joint", "[LR]_shoulder_roll_joint",
                    "[LR]_shoulder_yaw_joint",   "[LR]_elbow_roll_joint",
                    "[LR]_elbow_yaw_joint",      "[LR]_wrist_pitch_joint",
                    "[LR]_wrist_roll_joint",
                ],
                stiffness=400.0,
                damping=20.0,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["[LR]_finger1_joint", "[LR]_finger2_joint"],
                stiffness=100.0,
                damping=5.0,
            ),
        },
    )

    # ── 4 Parts: 2x PartA + 2x PartB ───────────────────────────────────────
    part0: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Part0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_R}/Collected_Task1_PartA_ori_color/Task1_PartA.usd",
            rigid_props=_PART_RIGID_PROPS,
            mass_props=_PART_MASS_PROPS,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.65, 0.25, 1.10)),
    )

    part1: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Part1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_R}/Collected_Task1_PartA_ori_color/Task1_PartA.usd",
            rigid_props=_PART_RIGID_PROPS,
            mass_props=_PART_MASS_PROPS,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.72, 0.30, 1.10)),
    )

    part2: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Part2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_R}/Collected_Part_B_ori_color/Part_B.usd",
            rigid_props=_PART_RIGID_PROPS,
            mass_props=_PART_MASS_PROPS,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.80, 0.25, 1.10)),
    )

    part3: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Part3",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_R}/Collected_Part_B_ori_color/Part_B.usd",
            rigid_props=_PART_RIGID_PROPS,
            mass_props=_PART_MASS_PROPS,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.85, 0.30, 1.10)),
    )

    # ── Static asset USD paths (spawned trực tiếp trong _setup_scene) ──────
    table_usd: str = f"{_R}/Collected_table_v2/table_v2.usd"
    box_usd: str = f"{_R}/Box_blank/box_60_40_23_cut_0.usd"

    # ── Task parameters ──────────────────────────────────────────────────────
    success_threshold: float = 0.08   # 8 cm
    table_z_min: float = 0.90         # vật rơi khỏi bàn nếu z < giá trị này

    # Local-frame positions (cộng env_origins trong env.py để ra world frame)
    box_local_pos: tuple = (1.2, 0.3, 1.10)
    scatter_center: tuple = (0.75, 0.28, 1.10)
    scatter_half_x: float = 0.15      # x ∈ [0.60, 0.90]
    scatter_half_y: float = 0.10      # y ∈ [0.18, 0.38]

    # Action unnormalization range
    r_arm_joint_min: float = -2.5
    r_arm_joint_max: float = 2.5
    r_finger_open: float = 0.0
    r_finger_close: float = 0.04
