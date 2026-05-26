
import csv
import math
import numpy as np
import robot_math_utils as rmu
import json
import yaml
import sys
import time
from pathlib import Path
from enum import Enum
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

sys.path.append('/home/ubuntu/vinh/Ubtech_sim_ref/source')
from DualArmIK import DualArmIK

# 1. ENUMS & CONFIGURATION

class PrimitiveEvent(Enum):
    SUCCESS = "PRIMITIVE_SUCCESS"              
    GRASP_FAIL = "PRIMITIVE_GRASP_FAIL"        
    DROP = "PRIMITIVE_DROP"                    
    WRONG_BIN = "PRIMITIVE_WRONG_BIN"          
    COLLISION = "PRIMITIVE_COLLISION"          
    TIMEOUT = "PRIMITIVE_TIMEOUT"              

@dataclass
class PrimitiveResult:
    primitive_name: str
    success: bool
    elapsed_s: float
    retry_count: int
    failure_reason: Optional[str]
    metrics: dict

EVENT_RETRY_POLICIES = {
    PrimitiveEvent.GRASP_FAIL: {"max_attempts": 3, "speed": "slow", "timeout_s": 25.0, "on_exceed": "skip"},
    PrimitiveEvent.DROP: {"max_attempts": 2, "speed": "very_slow", "timeout_s": 30.0, "on_exceed": "skip"},
    PrimitiveEvent.TIMEOUT: {"max_attempts": 2, "speed": "slow", "timeout_s": 35.0, "on_exceed": "abort"},
    PrimitiveEvent.COLLISION: {"max_attempts": 1, "speed": "normal", "timeout_s": 20.0, "on_exceed": "abort"}
}

ENABLE_WORKSPACE_VALIDATION = True

def load_workspace_bounds_from_yaml(config_path: Path) -> dict:
    default_bounds = {"x": (0.4, 1.0), "y": (-0.5, 0.5), "z": (0.9, 1.5)} 
    try:
        if not config_path.exists(): return default_bounds
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        bounds_cfg = config.get("workspace_bounds", {})
        if not bounds_cfg: return default_bounds
        return {
            "x": tuple(bounds_cfg.get("x", default_bounds["x"])),
            "y": tuple(bounds_cfg.get("y", default_bounds["y"])),
            "z": tuple(bounds_cfg.get("z", default_bounds["z"])),
        }
    except Exception: return default_bounds

WORKSPACE_BOUNDS = load_workspace_bounds_from_yaml(Path("/home/ubuntu/thu/configs/planner.yaml"))

def validate_position_in_workspace(position_m: list, bounds: dict = None) -> tuple:
    if bounds is None: bounds = WORKSPACE_BOUNDS
    x, y, z = position_m
    errors = []
    if not (bounds["x"][0] <= x <= bounds["x"][1]): errors.append(f"X={x:.4f}m out of bounds")
    if not (bounds["y"][0] <= y <= bounds["y"][1]): errors.append(f"Y={y:.4f}m out of bounds")
    if not (bounds["z"][0] <= z <= bounds["z"][1]): errors.append(f"Z={z:.4f}m out of bounds")
    if errors: return False, "; ".join(errors)
    return True, "Valid"

# 2. CORE MOTION & INTERPOLATION
def load_action_plans_from_person2(file_path: str = None) -> list:
    if file_path is None: file_path = "/home/ubuntu/thu/lab_outputs/planner_outputs/action_plans_for_person3_demo.json"
    with open(file_path, "r", encoding="utf-8") as f: return json.load(f)

def load_action_plan(yaml_file_path):
    with open(yaml_file_path, 'r') as f: return yaml.safe_load(f)['action_plan']

def get_base_matrix(position, quaternion) -> np.ndarray:
    return rmu.make_T(np.array(quaternion), np.array(position))

def convert_matrix_to_ik_input(T: np.ndarray) -> list:
    import pinocchio as pin
    R_mat = T[:3, :3]
    p = T[:3, 3]
    return DualArmIK.se3_to_xyzrpy(pin.SE3(R_mat, p)).tolist()

def apply_grasp_rotation(T_pose: np.ndarray, yaw_rad: float, approach_axis: str = "z_down") -> np.ndarray:
    T_rotated = T_pose.copy()
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
    R_yaw = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])
    if approach_axis == "z_down":
        R_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        R_final = R_yaw @ R_flip
    else: R_final = R_yaw
    T_rotated[:3, :3] = T_rotated[:3, :3] @ R_final
    return T_rotated

def execute_stage(robot, world, stage_name, target_pose, gripper_side, gripper_state="open", step_size=0.016, max_steps=2000, pos_tol=0.02, rot_tol=0.1, timeout_sec=60.0, verbose=True):
    # gripper_state is accepted but not used - gripper is managed separately by FSM
    left_target = target_pose if gripper_side == "left" else None
    right_target = target_pose if gripper_side == "right" else None
    start_time = time.time()
    steps = 0
    while steps < max_steps:
        is_reached = robot.control_dual_arm_ik(step_size, left_target_xyzrpy=left_target, right_target_xyzrpy=right_target, pos_tol=pos_tol, rot_tol=rot_tol)
        world.step(render=True)
        steps += 1
        elapsed = time.time() - start_time
        if is_reached:
            settle_frames = 10 if verbose else 1
            for _ in range(settle_frames): world.step(render=True)
            if verbose: print(f"  ✓ [{stage_name}] reached in {steps} steps")
            return True, elapsed, None
        if elapsed > timeout_sec:
            if verbose: print(f"  ⚠️ [{stage_name}] TIMEOUT")
            return False, elapsed, "timeout"
    if verbose: print(f"  ✗ [{stage_name}] MAX STEPS EXCEEDED")
    return False, elapsed, "max_steps_exceeded"

def move_interpolated(robot, world, T_start, T_end, side, stage_name, num_steps=20, max_sim_steps=100, writer=None, object_id="") -> tuple:
    print(f"  [{stage_name}] Nội suy {num_steps} điểm (Đường thẳng tuyệt đối)...")
    for i in range(1, num_steps + 1):
        t = i / float(num_steps)
        T_curr = T_start.copy()
        
        # KHÓA CHẶT TUYẾN TÍNH: Bắt buộc đi theo đường thẳng tắp trong hệ World
        T_curr[:3, 3] = T_start[:3, 3] + t * (T_end[:3, 3] - T_start[:3, 3])
        T_curr[:3, :3] = T_end[:3, :3] 

        ik_input_world = convert_matrix_to_ik_input(T_curr)
        x_w, y_w, z_w, roll_w, pitch_w, yaw_w = ik_input_world
        
        x_base, y_base, z_base = y_w + 0.20, -x_w + 0.70, z_w - 0.9005
        yaw_base = (yaw_w - (math.pi / 2.0) + math.pi) % (2 * math.pi) - math.pi
        ik_input_base = [x_base, y_base, z_base, roll_w, pitch_w, yaw_base]
        
        is_success, _, _ = execute_stage_fsm(
            robot, world, stage_name=f"{stage_name}_pt{i}", target_pose=ik_input_base, 
            gripper_side=side, max_steps=150, pos_tol=0.08, rot_tol=0.20, verbose=False
        )
        if not is_success:
            print(f"  ✗ [{stage_name}] Kẹt vật lý tại điểm {i}/{num_steps}")
            return False, "interpolation_stuck"
            
    if writer: writer.writerow([object_id, stage_name, "active"] + ik_input_base)
    return True, None


# ======================================================================
# 3. Industrial-optimized FSM: L-shape trajectory & grip force verification
# ======================================================================

def execute_stage_fsm(robot, world, stage_name, target_pose, gripper_side, 
                      step_size=0.016, max_steps=120, pos_tol=0.04, rot_tol=0.2, 
                      timeout_sec=60.0, verbose=True):
    """Wrapper for FSM state machine - doesn't require gripper_state"""
    left_target = target_pose if gripper_side == "left" else None
    right_target = target_pose if gripper_side == "right" else None
    start_time = time.time()
    steps = 0
    while steps < max_steps:
        is_reached = robot.control_dual_arm_ik(step_size, left_target_xyzrpy=left_target, 
                                               right_target_xyzrpy=right_target, 
                                               pos_tol=pos_tol, rot_tol=rot_tol)
        world.step(render=True)
        steps += 1
        elapsed = time.time() - start_time
        if is_reached:
            settle_frames = 10 if verbose else 1
            for _ in range(settle_frames): world.step(render=True)
            if verbose: print(f"  ✓ [{stage_name}] reached in {steps} steps")
            return True, elapsed, None
        if elapsed > timeout_sec:
            if verbose: print(f"  ⚠️ [{stage_name}] TIMEOUT")
            return False, elapsed, "timeout"
    if verbose: print(f"  ✗ [{stage_name}] MAX STEPS EXCEEDED")
    return False, elapsed, "max_steps_exceeded"

class PickAndPlaceStateMachine:
    def __init__(self, robot, world, plan, motion_specs, coord_transform=None, writer=None):
        self.robot, self.world = robot, world
        self.plan, self.specs = plan, motion_specs
        self.coord_transform, self.writer = coord_transform, writer
        
        self.state = "INIT"
        self.start_time = time.time()
        self.metrics = {}
        self.failure_reason = None
        self.object_id = plan['object_id']
        self.grasp_width = plan.get('grasp_hint', {}).get('grasp_width_m', 0.05)

    def run(self) -> PrimitiveResult:
        print(f"\n================ FSM: PICK & PLACE [{self.object_id}] ================")
        while self.state not in ["DONE", "FAIL"]:
            if self.state == "INIT": self._state_init()
            elif self.state == "APPROACH": self._state_approach()
            elif self.state == "CONTACT_GRASP": self._state_contact_grasp()
            elif self.state == "VERIFY_GRASP": self._state_verify_grasp()  # <--- BƯỚC MỚI
            elif self.state == "LIFT_TRANSFER": self._state_lift_transfer()
            elif self.state == "PLACE": self._state_place()
            elif self.state == "VERIFY": self._state_verify()
                
        return PrimitiveResult(
            primitive_name="pick_place", success=(self.state == "DONE"),
            elapsed_s=time.time() - self.start_time, retry_count=0,
            failure_reason=self.failure_reason, metrics=self.metrics
        )

    def _fail(self, reason: str):
        self.failure_reason = reason
        self.state = "FAIL"
        print(f"  [FSM ERROR] Failed: {reason}")

    def _state_init(self):
        print("[FSM] State: INIT")
        try:
            self.yaw_rad = 0.0 
            
            raw_obj = np.array(self.plan['object_pose_base']['position_m'])
            raw_bin = np.array(self.plan['bin_pose_base']['position_m'])
            
            if self.coord_transform is not None:
                world_obj = self.coord_transform.robot_to_world(raw_obj)
                world_bin = self.coord_transform.robot_to_world(raw_bin)
            else:
                world_obj = np.array([-raw_obj[1] + 0.70, raw_obj[0] - 0.20, 1.0400])
                world_bin = np.array([-raw_bin[1] + 0.70, raw_bin[0] - 0.20, 1.0400])
                
            world_obj[2] = 1.0400
            world_bin[2] = 1.0400
            
            obj_valid, msg = validate_position_in_workspace(world_obj.tolist())
            if not obj_valid and ENABLE_WORKSPACE_VALIDATION: return self._fail("target_out_of_workspace")

            self.side = "right" if raw_obj[1] < 0 else "left"
            
            tcp_offset = np.array([0.0, 0.0, 0.0], dtype=float)
            config_paths = ['configs/Part_Sorting.yaml', '../configs/Part_Sorting.yaml', '../../configs/Part_Sorting.yaml']
            for cfg_path in config_paths:
                if Path(cfg_path).exists():
                    try:
                        with open(cfg_path, 'r') as f:
                            cfg = yaml.safe_load(f)
                            if cfg and 'grasp' in cfg and 'tcp_offset' in cfg['grasp']:
                                tcp_offset = np.array(cfg['grasp']['tcp_offset'], dtype=float)
                                print(f"[FSM] Loaded TCP offset from {cfg_path}: {tcp_offset.tolist()}")
                                break
                    except Exception as e:
                        print(f"[FSM] Warning: failed to load {cfg_path}: {e}")
                        continue
            if np.allclose(tcp_offset, [0, 0, 0]):
                print(f"[FSM] Warning: TCP offset is zero (default) - gripper may not touch target!")
            
            T_obj = get_base_matrix(world_obj.tolist(), self.plan['object_pose_base']['quaternion_xyzw'])
            T_bin = get_base_matrix(world_bin.tolist(), self.plan['bin_pose_base']['quaternion_xyzw'])
            
            app_offset = self.specs.get("approach_offset_m", 0.08)
            lift_height = self.specs.get("lift_height_m", 0.17)
            Z_OFFSET = 0.035 
            SAFE_FLY_HEIGHT = 1.25 
            
            T_grasp_base = apply_grasp_rotation(T_obj, self.yaw_rad)
            
            # 1. Điểm Grasp (Chạm vật)
            self.T_grasp = T_grasp_base.copy()
            self.T_grasp[:3, 3] += T_grasp_base[:3, :3] @ tcp_offset
            self.T_grasp[2, 3] += Z_OFFSET

            print(f"[FSM] T_grasp position (world): {self.T_grasp[:3,3].tolist()}")
            print(f"[FSM] T_pre_grasp position (world) will be computed next")
            
            # 2. Điểm Pre-grasp (Nằm NGAY TRÊN ĐỈNH ĐẦU vật)
            self.T_pre_grasp = self.T_grasp.copy()
            self.T_pre_grasp[2, 3] += app_offset
            
            # 3. Điểm High Approach (Nằm TÍT TRÊN KHÔNG TRUNG, ngay trên đầu vật)
            self.T_high_approach = self.T_pre_grasp.copy()
            self.T_high_approach[2, 3] = SAFE_FLY_HEIGHT
            
            # Khởi tạo tương tự cho khay Bin (Bảo đảm đi cắm thẳng đứng)
            T_bin_rot = apply_grasp_rotation(T_bin, self.yaw_rad)
            self.T_place = T_bin_rot.copy()
            self.T_place[:3, 3] += T_bin_rot[:3, :3] @ tcp_offset
            self.T_place[2, 3] += Z_OFFSET
            
            self.T_pre_place = self.T_place.copy()
            self.T_pre_place[2, 3] += app_offset
            
            self.T_high_place = self.T_pre_place.copy()
            self.T_high_place[2, 3] = SAFE_FLY_HEIGHT
            
            self.state = "APPROACH"
        except Exception as e: return self._fail(f"init_error: {e}")

    def _state_approach(self):
        print("[FSM] State: APPROACH (Quỹ đạo chữ L)")
        if self.robot: self.robot.open_gripper(side=self.side)
        
        # Nhịp 1: Đưa tay tới không gian an toàn, không vặn cổ tay
        ik_input_world = convert_matrix_to_ik_input(self.T_high_approach)
        x_w, y_w, z_w, _, _, _ = ik_input_world
        x_base, y_base, z_base = y_w + 0.20, -x_w + 0.70, z_w - 0.9005
        
        print("  -> Lướt đến tọa độ thẳng đứng trên vật (Chưa xoay cổ tay)...")
        print(f"    target base coords: x={x_base:.3f}, y={y_base:.3f}, z={z_base:.3f}")
        success, elapsed, reason = execute_stage(self.robot, self.world, "fly_pos_only", [x_base, y_base, z_base, 0, 0, 0], 
                     self.side, "open", max_steps=400, pos_tol=0.20, rot_tol=3.14)
        if not success:
            print(f"  [FSM] fly_pos_only failed: {reason} (elapsed={elapsed}) -- will continue to alignment attempt")

        # Nhịp 2: Căn chỉnh xoay cổ tay vuông góc (z_down)
        print("  -> Vặn úp bàn tay xuống...")
        yaw_base = (0.0 - (math.pi / 2.0) + math.pi) % (2 * math.pi) - math.pi
        print(f"    align target yaw (base): {yaw_base:.3f}")
        success2, elapsed2, reason2 = execute_stage(self.robot, self.world, "fly_align_rot", [x_base, y_base, z_base, 3.1416, 0.0, yaw_base], 
                     self.side, "open", max_steps=250, pos_tol=0.12, rot_tol=0.6)
        if not success2:
            print(f"  [FSM] fly_align_rot failed: {reason2} (elapsed={elapsed2}) -- proceeding to vertical descent to observe behavior")
        
        # Phase 3: Vertical linear descent (ensure no arc motion)
        print("  -> Vertical linear descent to pre-grasp point...")
        success, err = move_interpolated(self.robot, self.world, self.T_high_approach, self.T_pre_grasp, self.side, "drop_vertical", num_steps=15, max_sim_steps=150)
        
        if success: self.state = "CONTACT_GRASP"
        else: self._fail(err)

    def _state_contact_grasp(self):
        print("[FSM] State: CONTACT_GRASP (Lower and grasp)")
        success, err = move_interpolated(self.robot, self.world, self.T_pre_grasp, self.T_grasp, self.side, "grasp_down", num_steps=15, max_sim_steps=120)
        if not success: return self._fail("collision_risk")
        
        print("  -> Close gripper...")
        if self.robot: self.robot.close_gripper(side=self.side, width=self.grasp_width)
        if self.world: self.world.step(steps=50) # Wait for fingers to close
        
        self.state = "VERIFY_GRASP" # Move to verification step

    def _state_verify_grasp(self):
        print("[FSM] State: VERIFY_GRASP (Grip force verification - Lift Test)")
        
        # 1. Perform a small lift (5cm) to test the grasp
        print("  -> Performing lift test (raise 5cm)...")
        T_lift_test = self.T_grasp.copy()
        T_lift_test[2, 3] += 0.05
        
        success, err = move_interpolated(self.robot, self.world, self.T_grasp, T_lift_test, self.side, "lift_test", num_steps=10)
        if not success: return self._fail("lift_test_failed")
        
        # 2. SENSOR CHECK (prevent false grasps)
        print("  -> Reading joint states from fingers...")
        is_grasped = True # Mặc định Pass nếu không có API
        if self.robot and hasattr(self.robot, 'get_joint_states'):
            joints = self.robot.get_joint_states()
            if joints:
                names = joints['names']
                positions = joints['positions'][0]
                
                # Find the fingers for the active side
                finger_prefix = "L_finger" if self.side == "left" else "R_finger"
                for i, name in enumerate(names):
                    if finger_prefix in name:
                        # If finger is nearly fully closed (e.g., < 0.012) -> likely empty grasp
                        # If finger is obstructed (e.g., > 0.015) -> likely holding an object
                        finger_pos = positions[i]
                        print(f"     * {name} = {finger_pos:.4f}")
                        if finger_pos < 0.012: 
                            is_grasped = False
                            
        if not is_grasped:
            print("  ⚠️ Detected failed grasp (fingers empty)!")
            self.robot.open_gripper(side=self.side)
            self.world.step(steps=20)
            return self._fail("gripper_not_closed") # Report failure to orchestrator for retry
            
        print("  ✓ Confirmed object is held securely!")
        self.state = "LIFT_TRANSFER"

    def _state_lift_transfer(self):
        print("[FSM] State: LIFT_TRANSFER (Đưa vật sang khay)")
        T_lift_test = self.T_grasp.copy()
        T_lift_test[2, 3] += 0.05 # Bắt đầu từ điểm Lift Test
        
        print("  -> Nhấc bổng thẳng đứng...")
        success, err = move_interpolated(self.robot, self.world, T_lift_test, self.T_high_approach, self.side, "lift_vertical", num_steps=15)
        if not success: return self._fail("dropped_object")
        
        print("  -> Lướt trên không trung qua Bin...")
        success, err = move_interpolated(self.robot, self.world, self.T_high_approach, self.T_high_place, self.side, "fly_to_bin", num_steps=15)
        if success: self.state = "PLACE"
        else: self._fail(err)

    def _state_place(self):
        print("[FSM] State: PLACE (Hạ và nhả vật)")
        print("  -> Đâm thẳng đứng xuống khay Bin...")
        success, err = move_interpolated(self.robot, self.world, self.T_high_place, self.T_place, self.side, "drop_to_bin", num_steps=15)
        if not success: return self._fail("collision_risk")
        
        print("  -> Mở tay nhả vật...")
        if self.robot: self.robot.open_gripper(side=self.side)
        if self.world: self.world.step(steps=40)
        
        print("  -> Rút tay lên khỏi khay...")
        move_interpolated(self.robot, self.world, self.T_place, self.T_high_place, self.side, "post_place_up", num_steps=10)
        self.state = "VERIFY"

    def _state_verify(self):
        print("[FSM] State: VERIFY -> Hoàn tất vòng lặp!")
        self.state = "DONE"

# ======================================================================
# 4. ORCHESTRATOR 
# ======================================================================

def emit_primitive_event(event: PrimitiveEvent, plan_id: str, object_id: str, details: dict = None) -> dict:
    if details is None: details = {}
    return {"event_type": event.value, "plan_id": plan_id, "object_id": object_id, "details": details}

def run_pipeline_from_person2(action_plan_yaml: str = None, robot=None, world=None, coord_transform=None):
    if action_plan_yaml is None: action_plan_yaml = 'src/task1/primitive_spec_task1.yaml'
    motion_specs = load_action_plan(action_plan_yaml)
    action_plans = load_action_plans_from_person2()

    print(f"\n[System] Khởi chạy {len(action_plans)} plans với Cartesian Interpolation...")

    FILE_OUTPUT_CSV = 'waypoints_task1.csv'
    retry_counts = {plan['plan_id']: 0 for plan in action_plans}

    with open(FILE_OUTPUT_CSV, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Object_id', 'Stage', "Gripper", 'x', 'y', 'z', 'Roll', "Pitch", "Yaw"])

        plan_idx = 0
        while plan_idx < len(action_plans):
            plan = action_plans[plan_idx]
            plan_id = plan['plan_id']
            retry_count = retry_counts[plan_id]

            is_first_plan = (plan_idx == 0)
            is_recovering = (retry_count > 0)
            
            if robot is not None and (is_first_plan or is_recovering):
                print(f"[Init] Teleport tay về Neutral...")
                try:
                    import torch
                    if robot._articulation is not None and robot.inital_joint_positions is not None:
                        s2_joint_names = robot._articulation.dof_names
                        s2_joint_indices = [robot._articulation.get_dof_index(n) for n in s2_joint_names]
                        robot._articulation.set_joint_positions(
                            torch.tensor(robot.inital_joint_positions, dtype=torch.float32), 
                            joint_indices=torch.tensor(s2_joint_indices, dtype=torch.int32)
                        )
                    if hasattr(robot, 'ik_solver') and hasattr(robot.ik_solver, 'q_initial'):
                        robot.ik_solver.q = robot.ik_solver.q_initial.copy()
                    if world is not None:
                        for _ in range(60): world.step(render=True)
                except Exception as e: print(f"⚠️ Lỗi reset: {e}")

            fsm = PickAndPlaceStateMachine(robot, world, plan, motion_specs, coord_transform, writer)
            result = fsm.run()
            
            if result.success: final_event = PrimitiveEvent.SUCCESS
            else:
                if result.failure_reason in ["ik_fail", "timeout", "target_out_of_workspace", "collision_risk", "interpolation_stuck"]:
                    final_event = PrimitiveEvent.COLLISION
                elif result.failure_reason in ["dropped_object", "gripper_not_closed"]:
                    final_event = PrimitiveEvent.GRASP_FAIL
                else: final_event = PrimitiveEvent.GRASP_FAIL

            emit_primitive_event(final_event, plan_id, plan['object_id'], details={"execution_time_s": result.elapsed_s})

            if final_event == PrimitiveEvent.SUCCESS:
                print(f"✅ [SUCCESS] Kế hoạch {plan_id} hoàn tất.")
                plan_idx += 1
                retry_counts[plan_id] = 0
            else:
                print(f"❌ [FAILED] {plan_id} lỗi: {result.failure_reason}")
                policy = EVENT_RETRY_POLICIES.get(final_event, {})
                max_attempts = policy.get("max_attempts", 1)
                retry_count += 1
                retry_counts[plan_id] = retry_count
                
                if retry_count < max_attempts: print(f"🔄 Retry {retry_count}/{max_attempts}")
                else:
                    print(f"🛑 Abort. Bỏ qua {plan_id}.")
                    plan_idx += 1

    print(f'\n🎉 Toàn bộ Motion Pipeline hoàn tất!')