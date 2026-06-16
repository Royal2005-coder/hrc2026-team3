from src.task1.planner import validate_objects, select_object, map_bin, build_action_plan
from src.task1.state import ObjectState, Pose, GraspHint, ObjectStatus, BinState

def test_validate_objects_rejects_low_confidence():
    pose = Pose(position_m=[-0.5, 0.0, 0.8], quaternion_xyzw=[0.0, 0.0, 0.0, 1.0])
    hint = GraspHint(yaw_rad=0.0, grasp_width_m=0.05, approach_axis="z_down")
    
    obj_low = ObjectState("obj_1", "part_A", 0.5, pose, hint, ObjectStatus.PENDING)
    obj_high = ObjectState("obj_2", "part_A", 0.9, pose, hint, ObjectStatus.PENDING)
    
    accepted, rejected = validate_objects([obj_low, obj_high])
    
    assert len(accepted) == 1
    assert accepted[0].object_id == "obj_2"
    
    assert len(rejected) == 1
    assert rejected[0].object_id == "obj_1"
    assert "confidence" in rejected[0].failure_reason

def test_map_bin_valid_class():
    bin_state = map_bin("part_A")
    assert bin_state is not None
    assert bin_state.bin_id == "bin_A"

def test_map_bin_invalid_class():
    bin_state = map_bin("unknown_part")
    assert bin_state is None

def test_select_object():
    pose = Pose(position_m=[-0.5, 0.0, 0.8], quaternion_xyzw=[0.0, 0.0, 0.0, 1.0])
    hint = GraspHint(yaw_rad=0.0, grasp_width_m=0.05, approach_axis="z_down")

    obj_1 = ObjectState("obj_1", "part_A", 0.8, pose, hint, ObjectStatus.PENDING)
    obj_2 = ObjectState("obj_2", "part_B", 0.95, pose, hint, ObjectStatus.PENDING)
    obj_3 = ObjectState("obj_3", "part_A", 0.99, pose, hint, ObjectStatus.PICKED) 
    
    selected = select_object([obj_1, obj_2, obj_3], strategy="highest_confidence")
    
    assert selected is not None
    assert selected.object_id == "obj_2"
    assert selected.status == ObjectStatus.SELECTED

def test_build_action_plan():
    pose = Pose(position_m=[-0.5, 0.0, 0.8], quaternion_xyzw=[0.0, 0.0, 0.0, 1.0])
    hint = GraspHint(yaw_rad=0.0, grasp_width_m=0.05, approach_axis="z_down")
    obj = ObjectState("obj_1", "part_A", 0.9, pose, hint, ObjectStatus.SELECTED)
    
    bin_state = BinState("bin_A", "part_A", Pose([-0.45, 0.25, 0.8], [0.0, 0.0, 0.0, 1.0]))
    
    plan = build_action_plan(obj, bin_state)
    
    assert plan.object_id == "obj_1"
    assert plan.target_bin == "bin_A"
    assert plan.class_id == "part_A"
    assert plan.retry_policy == "retry_once_slow"
