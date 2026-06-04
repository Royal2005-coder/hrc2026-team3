"""
PickPlaceEnvCfg — Scene + physics config cho Task 1 (Isaac Lab + PPO).

Thứ tự import: file này chỉ dùng @configclass + dataclass primitives,
không import trực tiếp torch/numpy để tránh vấn đề khi đọc config
trước khi SimulationApp khởi động.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Đường dẫn assets  (root_path từ configs/Part_Sorting.yaml: ../../assets/resources)
# ---------------------------------------------------------------------------
ASSETS_ROOT = "/home/ncd/workspace/assets/resources"

USD_ROBOT  = f"{ASSETS_ROOT}/Collected_s2_v1_ecbg/s2_v1.usd"
USD_PART_A = f"{ASSETS_ROOT}/Collected_Task1_PartA_ori_color/Task1_PartA.usd"
USD_PART_B = f"{ASSETS_ROOT}/Collected_Part_B_ori_color/Part_B.usd"
USD_TABLE  = f"{ASSETS_ROOT}/Collected_table_v2/table_v2.usd"
USD_BOX    = f"{ASSETS_ROOT}/Box_blank/box_60_40_23_cut_0.usd"

# ---------------------------------------------------------------------------
# Giá trị khớp ban đầu của robot S2 (trích từ IsaacSimRobotInterface)
# ---------------------------------------------------------------------------
_ROBOT_INIT_JOINTS: dict[str, float] = {
    # Left arm (cố định trong task 1)
    "L_shoulder_pitch_joint":  0.09322,
    "L_shoulder_roll_joint":  -0.59332,
    "L_shoulder_yaw_joint":   -1.59588,
    "L_elbow_roll_joint":     -1.89636,
    "L_elbow_yaw_joint":       1.40005,
    "L_wrist_pitch_joint":    -0.00049,
    "L_wrist_roll_joint":      0.09987,
    # Right arm (được điều khiển bởi RL policy)
    "R_shoulder_pitch_joint": -0.09322,
    "R_shoulder_roll_joint":  -0.59335,
    "R_shoulder_yaw_joint":    1.59587,
    "R_elbow_roll_joint":     -1.89636,
    "R_elbow_yaw_joint":      -1.40009,
    "R_wrist_pitch_joint":     0.00048,
    "R_wrist_roll_joint":      0.09985,
    # Head (giữ nguyên)
    "head_pitch_joint":       -0.60095,
    "head_yaw_joint":          0.0,
    # Fingers (mở hoàn toàn ban đầu)
    "L_finger1_joint":         0.0,
    "L_finger2_joint":         0.0,
    "R_finger1_joint":         0.0,
    "R_finger2_joint":         0.0,
}

# ---------------------------------------------------------------------------
# Vật lý chung cho rigid parts (có thể gắp)
# ---------------------------------------------------------------------------
_PART_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=4,
    solver_velocity_iteration_count=0,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)
_PART_MASS_PROPS = sim_utils.MassPropertiesCfg(mass=0.1)  # 100g mỗi part


# ---------------------------------------------------------------------------
# Helper: tạo RigidObjectCfg cho một part
# ---------------------------------------------------------------------------
def _make_part_cfg(prim_suffix: str, usd_path: str, init_pos: tuple) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_suffix}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=_PART_RIGID_PROPS,
            mass_props=_PART_MASS_PROPS,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos),
    )


# ---------------------------------------------------------------------------
# Cấu hình môi trường chính
# ---------------------------------------------------------------------------
@configclass
class PickPlaceEnvCfg(DirectRLEnvCfg):
    """Config cho pick-and-place Task 1 với Isaac Lab DirectRLEnv.

    Observation (38-dim):
        [0:7]   right arm joints (rad)
        [7:9]   right finger joints (rad)
        [9]     gripper control (-1=open, +1=close)
        [10:38] 4 objects × 7 = [x, y, z, qx, qy, qz, qw]

    Action (10-dim, [-1, 1]):
        [0:7]  right arm joint targets → unnormalize to ±2.5 rad
        [7:9]  right finger targets → unnormalize to [0, 0.04] rad
        [9]    gripper command (>0 = close, ≤0 = open)
    """

    # --- Simulation ---
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=3)
    decimation: int = 3           # 3 physics steps per RL step → ~20 Hz control
    episode_length_s: float = 25.0  # 500 RL steps × 3 × (1/60) ≈ 25 s

    # --- Obs / act dimensions (Isaac Lab 2.x dùng observation_space / action_space) ---
    observation_space: int = 38
    action_space: int = 10
    state_space: int = 0    # không dùng asymmetric actor-critic

    # --- Scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # --- Robot ---
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_ROBOT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,   # robot được gắn cố định vào ground
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.7, -0.2, 0.9),
            rot=(0.0, 0.0, 0.7071, 0.7071),  # 90° xung quanh trục Z
            joint_pos=_ROBOT_INIT_JOINTS,
        ),
        actuators={
            # Cánh tay (left + right) — position-controlled, stiffness cao
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
            # Ngón tay — position-controlled, nhẹ hơn
            "finger_joints": ImplicitActuatorCfg(
                joint_names_expr=["[LR]_finger1_joint", "[LR]_finger2_joint"],
                stiffness=200.0,
                damping=10.0,
            ),
        },
    )

    # --- 4 Parts: 2x PartA, 2x PartB ---
    # Vị trí ban đầu trải đều trong scatter area; sẽ bị reset ngẫu nhiên khi train
    part0: RigidObjectCfg = _make_part_cfg("Part0", USD_PART_A, (0.68, 0.28, 1.10))
    part1: RigidObjectCfg = _make_part_cfg("Part1", USD_PART_A, (0.72, 0.28, 1.10))
    part2: RigidObjectCfg = _make_part_cfg("Part2", USD_PART_B, (0.78, 0.28, 1.10))
    part3: RigidObjectCfg = _make_part_cfg("Part3", USD_PART_B, (0.82, 0.28, 1.10))

    # --- Task parameters ---
    success_threshold: float = 0.08     # 8cm — vật coi như trong hộp
    table_z_min: float = 0.90           # z thấp hơn ngưỡng này = vật rơi khỏi bàn

    # 4 vị trí target bên trong hộp [1.2, 0.3, 1.05]
    box_targets: tuple = (
        (1.15, 0.25, 1.10),
        (1.25, 0.25, 1.10),
        (1.15, 0.35, 1.10),
        (1.25, 0.35, 1.10),
    )

    # Vùng scatter (center ± half_size) cho reset ngẫu nhiên
    scatter_center: tuple = (0.75, 0.28, 1.10)
    scatter_half_x: float = 0.15   # x ∈ [0.60, 0.90]
    scatter_half_y: float = 0.10   # y ∈ [0.18, 0.38]

    # Joint range của right arm dùng để unnormalize action
    r_arm_joint_min: float = -2.5   # rad
    r_arm_joint_max: float = 2.5    # rad
    r_finger_min: float = 0.0       # rad
    r_finger_max: float = 0.04      # rad
