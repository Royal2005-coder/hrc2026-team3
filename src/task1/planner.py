import yaml
import math
from typing import List, Optional, Tuple
from src.task1.state import (
    ObjectState, BinState, ActionPlan,
    ObjectStatus, GraspHint, Pose
)
from src.task1.transform_utils import validate_transform
from pathlib import Path


PLANNER_CONFIG_PATH = "configs/planner.yaml"
BINS_CONFIG_PATH = "configs/bins.yaml"



# Default fallback configs
DEFAULT_WORKSPACE_BOUNDS = {
    "x": (-0.7, -0.4),
    "y": (-0.4, 0.4),
    "z": (0.5, 1.0),
}

DEFAULT_PLANNER = {
    "confidence_threshold": 0.75,
    "selection_strategy": "highest_confidence"
}

DEFAULT_BINS = {
    "part_A": {
        "bin_id": "bin_A",
        "position_m": [-0.45, 0.25, 0.80],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "grasp_width_m": 0.045,
        "description": "Blue bin — left side of table"
    },
    "part_B": {
        "bin_id": "bin_B",
        "position_m": [-0.45, -0.25, 0.80],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "grasp_width_m": 0.060,
        "description": "Red bin — right side of table"
    }
}


# YAML loader
def load_yaml_config(path: Path) -> dict:

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Load planner config
def load_planner_config():
    try:
        config = load_yaml_config(PLANNER_CONFIG_PATH)

        planner_cfg = config.get("planner", {})
        bounds_cfg = config.get("workspace_bounds", {})

        planner = {
            "confidence_threshold":
                planner_cfg.get(
                    "confidence_threshold",
                    DEFAULT_PLANNER["confidence_threshold"]
                ),

            "selection_strategy":
                planner_cfg.get(
                    "selection_strategy",
                    DEFAULT_PLANNER["selection_strategy"]
                )
        }

        workspace_bounds = {
            "x": tuple(bounds_cfg.get("x", DEFAULT_WORKSPACE_BOUNDS["x"])),
            "y": tuple(bounds_cfg.get("y", DEFAULT_WORKSPACE_BOUNDS["y"])),
            "z": tuple(bounds_cfg.get("z", DEFAULT_WORKSPACE_BOUNDS["z"])),
        }

        return planner, workspace_bounds

    except Exception as e:
        print(f"[WARN] Failed to load planner.yaml: {e}")
        print("[WARN] Using default planner config.")

        return DEFAULT_PLANNER, DEFAULT_WORKSPACE_BOUNDS

# Load bins config
def load_bins_config():
    """
    Load bins from bins.yaml
    """

    try:
        config = load_yaml_config(BINS_CONFIG_PATH)

        bins = config.get("bins")

        if not bins:
            raise ValueError("Missing 'bins' section")

        return bins

    except Exception as e:
        print(f"[WARN] Failed to load bins.yaml: {e}")
        print("[WARN] Using default bins config.")

        return DEFAULT_BINS


PLANNER_CONFIG, WORKSPACE_BOUNDS = load_planner_config()

CONFIDENCE_THRESHOLD = PLANNER_CONFIG["confidence_threshold"]

SELECTION_STRATEGY = PLANNER_CONFIG["selection_strategy"]

BINS = load_bins_config()

GRASP_WIDTHS = {
    class_id: bin_cfg["grasp_width_m"]
    for class_id, bin_cfg in BINS.items()
}

# 1. parse_perception_frame
def parse_perception_frame(frame: dict) -> List[ObjectState]:
    """
    Convert raw PerceptionFrame JSON from Person 1
    into a list of ObjectState instances.

    Args:
        frame: raw dict from Person 1

    Returns:
        list of ObjectState (all PENDING)

    Example:
        objects = parse_perception_frame(frame)
        # → [ObjectState(obj_001, part_A, ...), ObjectState(obj_002, ...)]
    """
    objects = []
    for obj in frame.get("objects", []):
        pose = Pose(
            position_m=obj["pose_base"]["position_m"],
            quaternion_xyzw=obj["pose_base"]["quaternion_xyzw"]
        )
        hint = GraspHint(
            yaw_rad=obj["grasp_hint"]["yaw_rad"],
            grasp_width_m=obj["grasp_hint"].get(
                "grasp_width_m",
                GRASP_WIDTHS.get(obj["class_id"], 0.05)
            ),
            approach_axis=obj["grasp_hint"]["approach_axis"]
        )
        state = ObjectState(
            object_id=obj["object_id"],
            class_id=obj["class_id"],
            confidence=obj["confidence"],
            pose=pose,
            grasp_hint=hint,
            status=ObjectStatus.PENDING,
            failure_reason=obj.get("failure_reason")
        )
        objects.append(state)
    return objects


# 2. validate_objects

def validate_objects(objects: List[ObjectState]) -> Tuple[List[ObjectState], List[ObjectState]]:
    """
    Split objects into accepted (valid) and rejected (invalid) lists.

    Checks:
      - confidence >= threshold
      - failure_reason is None
      - pose passes transform sanity (quaternion + bounds)
      - class_id exists in bins.yaml

    Args:
        objects: list of ObjectState

    Returns:
        (accepted, rejected)

    Example:
        good, bad = validate_objects(objects)
        # good → ready for selection
        # bad  → logged and skipped
    """
    accepted = []
    rejected = []

    for obj in objects:

        # Check 1: already failed by Person 1
        if obj.failure_reason is not None:
            obj.status = ObjectStatus.SKIPPED
            obj.failure_reason = f"Person1 flagged: {obj.failure_reason}"
            rejected.append(obj)
            continue

        # Check 2: confidence threshold
        if obj.confidence < CONFIDENCE_THRESHOLD:
            obj.status = ObjectStatus.SKIPPED
            obj.failure_reason = f"confidence {obj.confidence} < {CONFIDENCE_THRESHOLD}"
            rejected.append(obj)
            continue

        # Check 3: pose sanity (quaternion valid + in bounds)
        pose_ok, reason = validate_transform(
            obj.pose.position_m,
            obj.pose.quaternion_xyzw
        )
        if not pose_ok:
            obj.status = ObjectStatus.SKIPPED
            obj.failure_reason = f"pose invalid: {reason}"
            rejected.append(obj)
            continue

        # Check 4: class_id known in bins
        if obj.class_id not in BINS:
            obj.status = ObjectStatus.SKIPPED
            obj.failure_reason = f"unknown class_id: {obj.class_id}"
            rejected.append(obj)
            continue

        accepted.append(obj)

    return accepted, rejected


# 3. select_object
def _distance_to_robot(position: List[float]) -> float:
    """Euclidean distance from robot base origin to object position."""
    x, y, z = position
    return math.sqrt(x**2 + y**2 + z**2)


def select_object(
    objects: List[ObjectState],
    strategy: str = "highest_confidence"
) -> Optional[ObjectState]:
    """
    Select the best next object from validated candidates.

    Strategies:
      "highest_confidence" → pick the object Person 1 is most sure about
      "closest_distance"   → pick the object closest to robot base

    Only considers objects with status == PENDING.

    Args:
        objects:  list of validated ObjectState
        strategy: selection strategy string

    Returns:
        selected ObjectState, or None if list is empty

    Example:
        obj = select_object(accepted, strategy="highest_confidence")
        # → ObjectState(obj_001, part_A, confidence=0.91)
    """
    candidates = [o for o in objects if o.status == ObjectStatus.PENDING]

    if not candidates:
        return None

    if strategy == "highest_confidence":
        selected = max(candidates, key=lambda o: o.confidence)

    elif strategy == "closest_distance":
        selected = min(
            candidates,
            key=lambda o: _distance_to_robot(o.pose.position_m)
        )

    else:
        selected = max(candidates, key=lambda o: o.confidence)

    selected.status = ObjectStatus.SELECTED
    return selected


# 4. map_bin
def map_bin(class_id: str) -> Optional[BinState]:
    """
    Convert bin configuration into BinState object.

    Args:
        class_id: object class id (e.g. part_A)

    Returns:
        BinState or None if not found
    """

    bin_cfg = BINS.get(class_id)

    if bin_cfg is None:
        return None

    pose = Pose(
        position_m=bin_cfg["position_m"],
        quaternion_xyzw=bin_cfg["quaternion_xyzw"]
    )

    return BinState(
        bin_id=bin_cfg["bin_id"],
        assigned_class=class_id,
        pose=pose
    )

# 5. build_action_plan
_plan_counter = 0

def build_action_plan(obj: ObjectState, bin_state: BinState) -> ActionPlan:
    """
    Assemble a complete ActionPlan to send to Person 3.

    Args:
        obj:       selected ObjectState
        bin_state: target BinState from map_bin()

    Returns:
        ActionPlan ready to send

    Example:
        plan = build_action_plan(obj_001, bin_a)
        # → ActionPlan(plan_id="plan_0001", object_id="obj_001", ...)
    """
    global _plan_counter
    _plan_counter += 1
    plan_id = f"plan_{_plan_counter:04d}"

    known_width = GRASP_WIDTHS.get(obj.class_id, obj.grasp_hint.grasp_width_m)
    grasp = GraspHint(
        yaw_rad=obj.grasp_hint.yaw_rad,
        grasp_width_m=known_width,
        approach_axis=obj.grasp_hint.approach_axis
    )

    return ActionPlan(
        plan_id=plan_id,
        object_id=obj.object_id,
        class_id=obj.class_id,
        object_pose=obj.pose,
        target_bin=bin_state.bin_id,
        bin_pose=bin_state.pose,
        grasp_hint=grasp,
        retry_policy="retry_once_slow",
        timeout_s=20.0
    )

# 6. log_decision
def log_decision(
    step_id: int,
    fsm_state: str,
    obj: ObjectState,
    bin_state: BinState,
    reason: str = "highest_confidence_reachable",
    failure_reason: Optional[str] = None
) -> dict:
    """
    Build an eval log entry for Person 4.

    Returns:
        dict log entry
    """
    return {
        "event": "planner_decision",
        "episode_id": "ep_0001",
        "step_id": step_id,
        "fsm_state": fsm_state,
        "selected_object_id": obj.object_id,
        "selected_class_id": obj.class_id,
        "selected_bin": bin_state.bin_id,
        "object_confidence": obj.confidence,
        "decision_reason": reason,
        "failure_reason": failure_reason
    }


def run_planner(frame: dict, step_id: int = 0):
    """
    Full pipeline: frame → validate → select → map bin → ActionPlan + log.

    Returns:
        (ActionPlan, log_entry, rejected_list)
        or (None, None, rejected_list) if no valid object found
    """
    objects = parse_perception_frame(frame)
    accepted, rejected = validate_objects(objects)

    obj = select_object(accepted, strategy="highest_confidence")
    if obj is None:
        return None, None, rejected

    bin_state = map_bin(obj.class_id)
    if bin_state is None:
        obj.status = ObjectStatus.FAILED
        obj.failure_reason = f"No bin mapped for {obj.class_id}"
        return None, None, rejected

    plan = build_action_plan(obj, bin_state)
    log  = log_decision(step_id, "SELECT_OBJECT", obj, bin_state)

    return plan, log, rejected

