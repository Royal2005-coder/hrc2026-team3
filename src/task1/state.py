from dataclasses import dataclass
from enum import Enum
from typing import List, Optional



class ObjectStatus(Enum):
    """Tracks where an object is in its pick-place lifecycle."""
    PENDING   = "PENDING"    # detected, not yet handled
    SELECTED  = "SELECTED"   # chosen by planner, plan being built
    PICKING   = "PICKING"    # robot arm moving to pick
    PICKED    = "PICKED"     # successfully grasped
    PLACING   = "PLACING"    # robot arm moving to bin
    PLACED    = "PLACED"     # successfully dropped in bin
    FAILED    = "FAILED"     # could not pick after retries
    SKIPPED   = "SKIPPED"    # rejected during validation


class PrimitiveStatus(Enum):
    """Result codes returned by Person 3 after executing a move."""
    SUCCESS     = "SUCCESS"      # pick and place completed
    GRASP_FAIL  = "GRASP_FAIL"   # gripper could not grab object
    DROP        = "DROP"         # object dropped mid-move
    WRONG_BIN   = "WRONG_BIN"    # placed in wrong bin
    COLLISION   = "COLLISION"    # arm hit something
    TIMEOUT     = "TIMEOUT"      # move took too long


class FSMState(Enum):
    """All possible states in the pick-place FSM."""
    IDLE          = "IDLE"
    VALIDATE      = "VALIDATE"
    SELECT_OBJECT = "SELECT_OBJECT"
    PLAN          = "PLAN"
    PICKING       = "PICKING"
    MOVING        = "MOVING"
    PLACING       = "PLACING"
    VERIFYING     = "VERIFYING"
    RETRY         = "RETRY"
    DONE          = "DONE"
    FAIL          = "FAIL"


# ObjectState
@dataclass
class GraspHint:
    """How the gripper should approach and grab the object."""
    yaw_rad: float        
    grasp_width_m: float    
    approach_axis: str      


@dataclass
class Pose:
    """Position + orientation in robot base frame."""
    position_m: List[float]        
    quaternion_xyzw: List[float]   


@dataclass
class ObjectState:
    """
    Represents one detected object from Person 1's PerceptionFrame.
    Status is updated as the robot works through the pick-place cycle.

    Example:
        obj = ObjectState(
            object_id="obj_001",
            class_id="part_A",
            confidence=0.91,
            pose=Pose([0.46, -0.15, 0.82], [0.0, 0.0, 0.707, 0.707]),
            grasp_hint=GraspHint(1.57, 0.045, "z_down")
        )
    """
    object_id:    str
    class_id:     str
    confidence:   float
    pose:         Pose
    grasp_hint:   GraspHint
    status:       ObjectStatus = ObjectStatus.PENDING
    failure_reason: Optional[str] = None
    retry_count:  int = 0



# BinState
@dataclass
class BinState:
    """
    Represents one target bin.
    count tracks how many objects have been placed inside.

    Example:
        bin_a = BinState(
            bin_id="bin_A",
            assigned_class="part_A",
            pose=Pose([0.55, 0.20, 0.80], [0.0, 0.0, 0.0, 1.0])
        )
    """
    bin_id:          str
    assigned_class:  str         
    pose:            Pose
    count:           int = 0    


# ActionPlan
@dataclass
class ActionPlan:
    """
    Instruction sent to Person 3 (Motion Primitive).
    Contains everything Person 3 needs to execute one pick-place move.

    Example:
        plan = ActionPlan(
            plan_id="plan_0007",
            object_id="obj_001",
            class_id="part_A",
            object_pose=Pose([0.46, -0.15, 0.82], [0.0, 0.0, 0.707, 0.707]),
            target_bin="bin_A",
            bin_pose=Pose([0.55, 0.20, 0.80], [0.0, 0.0, 0.0, 1.0]),
            grasp_hint=GraspHint(1.57, 0.045, "z_down"),
            retry_policy="retry_once_slow",
            timeout_s=20.0
        )
    """
    plan_id:       str
    object_id:     str
    class_id:      str
    object_pose:   Pose
    target_bin:    str
    bin_pose:      Pose
    grasp_hint:    GraspHint
    retry_policy:  str = "retry_once_slow"
    timeout_s:     float = 20.0

    def to_dict(self) -> dict:
        """Convert to dict for sending to Person 3 as JSON."""
        return {
            "plan_id": self.plan_id,
            "primitive": "pick_place",
            "object_id": self.object_id,
            "class_id": self.class_id,
            "object_pose_base": {
                "position_m": self.object_pose.position_m,
                "quaternion_xyzw": self.object_pose.quaternion_xyzw,
            },
            "target_bin": self.target_bin,
            "bin_pose_base": {
                "position_m": self.bin_pose.position_m,
                "quaternion_xyzw": self.bin_pose.quaternion_xyzw,
            },
            "grasp_hint": {
                "yaw_rad": self.grasp_hint.yaw_rad,
                "grasp_width_m": self.grasp_hint.grasp_width_m,
                "approach_axis": self.grasp_hint.approach_axis,
            },
            "retry_policy": self.retry_policy,
            "timeout_s": self.timeout_s,
        }



# PrimitiveResult
@dataclass
class PrimitiveResult:
    """
    Result received from Person 3 after executing an ActionPlan.
    FSM uses this to decide next action: verify / retry / fail.

    Example:
        result = PrimitiveResult(
            plan_id="plan_0007",
            object_id="obj_001",
            status=PrimitiveStatus.SUCCESS
        )
    """
    plan_id:        str
    object_id:      str
    status:         PrimitiveStatus
    failure_reason: Optional[str] = None
    actual_bin:     Optional[str] = None 

