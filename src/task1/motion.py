
import csv
import numpy as np
import robot_math_utils as rmu
import json
import yaml
import sys
import time
sys.path.append('/home/ubuntu/vinh/Ubtech_sim_ref/source')
from DualArmIK import DualArmIK


def load_action_plan(yaml_file_path) :
    """ Load primitive specifications from a YAML file."""
    with open(yaml_file_path, 'r') as f:
        spec = yaml.safe_load(f)['action_plan']
    return spec

def run_pipeline(json_file_path, action_plan, robot=None, world=None, coord_transform=None):
    with open(json_file_path, 'r') as f:
        workpieces = json.load(f).get('workpieces', [])
    print(f"Calculating IK for {len(workpieces)} workpieces...")
    FILE_OUTPUT_CSV = 'waypoints_task1.csv'

    with open(FILE_OUTPUT_CSV, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Object_id', 'Stage', "Gripper", 'x', 'y', 'z', 'Roll', "Pitch", "Yaw"]) 
        for index,  item in enumerate(workpieces):
            T_object = get_base_matrix(item['position'], item['quaternion'])
            T_bin = get_base_matrix(item['bin_position'], item['quaternion'])
            R_flip = rmu.make_R(0, np.pi, 0)
            T_object[:3, :3] = T_object[:3, :3] @ R_flip
            T_bin[:3, :3] = T_bin[:3, :3] @ R_flip
            excute_pick_and_place(T_object, T_bin, item['id'], action_plan, writer, robot, world, coord_transform)
    print('Run successfully! Waypoints saved to', FILE_OUTPUT_CSV)  
# Phan tinh toan 
def get_base_matrix(position, quaternion) -> np.ndarray:
    """ Convert Pose3D to a homogeneous transformation matrix."""
    return rmu.make_T(np.array(quaternion), np.array(position))

def convert_matrix_to_ik_input(T: np.ndarray) -> list:
    """ Convert a homogeneous transformation matrix to [x,y,z,roll,pitch,yaw] for IK input."""
    import pinocchio as pin
    R_mat = T[:3, :3]
    p = T[:3, 3]
    se3_obj = pin.SE3(R_mat, p)
    return DualArmIK.se3_to_xyzrpy(se3_obj).tolist()

def execute_stage(robot, world, stage_name, target_pose, gripper_side, gripper_state, 
                  step_size=0.016, max_steps=2000, pos_tol=3e-2, timeout_sec=60.0):

    

    print(f"\n[Stage] {stage_name}: gripper={gripper_state}, side={gripper_side}")
    print(f"   => Mục tiêu IK (target_pose): {target_pose}")

    

    # Set up target based on side

    if gripper_side == "left":

        left_target = target_pose

        right_target = None

    else:

        left_target = None

        right_target = target_pose

        

    # Gửi tín hiệu đóng/mở ngay từ đầu stage

    if gripper_state == "close":

        robot.close_gripper(side=gripper_side)

    else:  # "open"

        robot.open_gripper(side=gripper_side)

    

    start_time = time.time()

    steps = 0

    success = False

    failure_reason = None

    

    # 2. VÒNG LẶP CHẠY MÔ PHỎNG (Vừa di chuyển tay, vừa là Settle Time cho Gripper)

    while steps < max_steps:
        # Call IK control every step
        is_reached =robot.control_dual_arm_ik(
            step_size=step_size,
            left_target_xyzrpy=left_target,
            right_target_xyzrpy=right_target,
            pos_tol=pos_tol
        )
        
        # Step simulation
        world.step(render=True)
        steps += 1
        elapsed = time.time() - start_time

        if is_reached:
            for _ in range(15):  # Extra steps to ensure stability at target
                world.step(render=True)
                success = True
                break
        if elapsed > timeout_sec:
            failure_reason = "timeout"
            print(f"  ⚠️ Stage {stage_name} Bị Quá Thời Gian (Timeout)!")
            break

    if not success and failure_reason is None:
        print(f"  ⚙️ Stage {stage_name} đạt giới hạn bước ({steps} steps) - Tiếp tục stage tiếp theo.")
        success = True

    if success:        
        print(f"  ✓ Stage {stage_name} complete ({steps} steps, {elapsed:.2f}s)")

    return success, elapsed, failure_reason
# Phan thuc hien pick and place
def excute_pick_and_place(T_object, T_bin, object_id, action_plan, writer, robot=None, world=None, coord_transform=None):

    approach_offset = action_plan['approach_offset_m']
    lift_height = action_plan['lift_height_m']
    
    # Calculate all 8 stages
    # 1. Pre-grasp pose (approach from above)
    T_pre_grasp = T_object.copy()
    T_pre_grasp[2, 3] += approach_offset
    # 2. Grasp pose (at object)
    T_grasp = T_object.copy()
    # 3. Post-grasp pose (same as grasp but with gripper closed)
    T_post_grasp = T_grasp.copy()
    # 4. Lift pose
    T_lift = T_grasp.copy()
    T_lift[2, 3] += lift_height
    # 5. Pre-place pose
    T_pre_place = T_bin.copy()
    T_pre_place[2, 3] += approach_offset
    # 6-8. Place poses
    T_place = T_bin.copy()
    
    # Define 8-step sequence: (stage_name, pose, gripper_state)
    stages = [
        ("pre-grasp", T_pre_grasp, "open"),
        ("grasp", T_grasp, "open"),
        ("post-grasp", T_post_grasp, "close"),
        ("lift", T_lift, "close"),
        ("pre-place", T_pre_place, "close"),
        ("place", T_place, "close"),
        ("release", T_place, "open"),
        ("post-place", T_pre_place, "open"),
    ]
    
    # Determine which arm to use based on object position
    gripper_side = "left" if T_object[1, 3] < 0 else "right"
    total_time = 0.0
    # Log to CSV and execute each stage
    for stage_name, target_T, gripper_state in stages:
        if coord_transform is not None:
            Target_T_local = coord_transform.world_to_local(target_T)
        else:
            Target_T_local = target_T
        ik_input = convert_matrix_to_ik_input(Target_T_local)

        writer.writerow([object_id, stage_name, gripper_state] + ik_input)
        
        # Execute on simulator if robot and world provided
        if robot is not None and world is not None:
            is_success, stage_time, failure_reason = execute_stage(robot, world, stage_name, ik_input, gripper_side, gripper_state)
            total_time += stage_time
            if not is_success:
                print(f"Aborting remaining stages for {object_id} due to failure at stage {stage_name}.")
                break
    print(f"Total execution time for {object_id}: {total_time:.2f} seconds")