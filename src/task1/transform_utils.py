import math
import numpy as np
from typing import List, Tuple, Optional
import yaml

DEFAULT_WORKSPACE_BOUNDS = {
    "x": (-0.7, -0.4),
    "y": (-0.4, 0.4),
    "z": (0.5, 1.0),
}


def load_workspace_bounds() -> dict:
    """
    Load workspace bounds from configs/planner.yaml.

    Returns:
        dict:
        {
            "x": (min_x, max_x),
            "y": (min_y, max_y),
            "z": (min_z, max_z)
        }

    Falls back to DEFAULT_WORKSPACE_BOUNDS if:
      - file does not exist
      - yaml invalid
      - required keys missing
    """

    try:

        with open("configs/planner.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        bounds = config.get("workspace_bounds", {})

        return {
            "x": tuple(bounds["x"]),
            "y": tuple(bounds["y"]),
            "z": tuple(bounds["z"]),
        }

    except Exception as e:
        print(f"[WARN] Failed to load planner.yaml: {e}")
        print("[WARN] Using default workspace bounds.")

        return DEFAULT_WORKSPACE_BOUNDS


WORKSPACE_BOUNDS = load_workspace_bounds()
QUATERNION_NORM_TOLERANCE = 0.01


# 1. validate_quaternion
def validate_quaternion(q: List[float]) -> Tuple[bool, str]:
    """
    Check if a quaternion [x, y, z, w] is valid.
    A valid quaternion must satisfy: x² + y² + z² + w² ≈ 1.0

    Args:
        q: [x, y, z, w] quaternion values

    Returns:
        (is_valid, reason_if_invalid)

    Example:
        >>> validate_quaternion([0.0, 0.0, 0.707, 0.707])
        (True, "OK")
        >>> validate_quaternion([1.0, 1.0, 1.0, 1.0])
        (False, "Quaternion norm 2.0000 is not 1.0")
    """
    if len(q) != 4:
        return False, f"Quaternion must have 4 values, got {len(q)}"

    x, y, z, w = q
    norm = math.sqrt(x**2 + y**2 + z**2 + w**2)

    if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
        return False, f"Quaternion norm {norm:.4f} is not 1.0 (tolerance ±{QUATERNION_NORM_TOLERANCE})"

    return True, "OK"


# 2. validate_position
def validate_position(position: List[float], bounds: dict = None) -> Tuple[bool, str]:
    """
    Check if a 3D position [x, y, z] is within robot workspace bounds.

    Args:
        position: [x, y, z] in meters (robot base frame)
        bounds: optional custom bounds dict, uses WORKSPACE_BOUNDS if None

    Returns:
        (is_valid, reason_if_invalid)
    """

    if len(position) != 3:
        return False, f"Position must have 3 values [x,y,z], got {len(position)}"

    b = bounds if bounds else WORKSPACE_BOUNDS
    labels = ["x", "y", "z"]

    for label, val in zip(labels, position):
        lo, hi = b[label]

        if not (lo <= val <= hi):
            return False, f"{label}={val} out of bounds [{lo}, {hi}]"

    return True, "OK"



# 3. validate_transform
def validate_transform(position: List[float], quaternion: List[float]) -> Tuple[bool, str]:
    """
    Run both position and quaternion validation together.

    Args:
        position:   [x, y, z] in robot base frame
        quaternion: [x, y, z, w]

    Returns:
        (is_valid, reason_if_invalid)

    Example:
        >>> validate_transform([0.46, -0.15, 0.82], [0.0, 0.0, 0.707, 0.707])
        (True, "OK")
    """
    q_valid, q_reason = validate_quaternion(quaternion)
    if not q_valid:
        return False, f"Quaternion invalid: {q_reason}"

    p_valid, p_reason = validate_position(position)
    if not p_valid:
        return False, f"Position invalid: {p_reason}"

    return True, "OK"


# 4. quaternion_to_yaw
def quaternion_to_yaw(q: List[float]) -> float:
    """
    Extract yaw angle (rotation around Z axis) from quaternion [x, y, z, w].
    This is the angle the gripper needs to rotate before closing.

    Formula: yaw = atan2(2*(w*z + x*y), 1 - 2*(y² + z²))

    Args:
        q: [x, y, z, w] quaternion

    Returns:
        yaw angle in radians

    Example:
        >>> quaternion_to_yaw([0.0, 0.0, 0.707, 0.707])
        1.5707...   ← 90 degrees
        >>> quaternion_to_yaw([0.0, 0.0, 0.0, 1.0])
        0.0         ← no rotation
    """
    x, y, z, w = q
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw


# 5. camera_to_base
def camera_to_base(
    position_camera: List[float],
    T_cam_to_base: Optional[np.ndarray] = None
) -> List[float]:
    """
    Convert a position from camera frame to robot base frame.

    In the real system, T_cam_to_base comes from camera calibration.
    For Task 1, Person 1 already provides pose_base, so this is used
    to verify or re-apply the transform if needed.

    Args:
        position_camera: [x, y, z] in camera frame (meters)
        T_cam_to_base:   4x4 homogeneous transform matrix (optional)
                         If None, uses a default identity-like transform
                         (assumes camera is aligned with robot base)

    Returns:
        [x, y, z] in robot base frame

    Example:
        >>> camera_to_base([0.12, -0.04, 0.74])
        [0.12, -0.04, 0.74]   ← with identity transform
    """
    if T_cam_to_base is None:
        T_cam_to_base = np.eye(4)

    p_cam = np.array([position_camera[0], position_camera[1], position_camera[2], 1.0])

    p_base = T_cam_to_base @ p_cam

    return [round(float(p_base[0]), 4),
            round(float(p_base[1]), 4),
            round(float(p_base[2]), 4)]


# 6. world_to_robot
def world_to_robot(
    position_world: List[float],
    robot_origin_in_world: List[float] = None
) -> List[float]:
    """
    Convert a position from world frame to robot base frame.

    Robot base frame = world frame shifted by robot's origin position.

    Args:
        position_world:        [x, y, z] in world frame
        robot_origin_in_world: [x, y, z] of robot base in world frame
                               Default: [0, 0, 0] (robot IS the world origin)

    Returns:
        [x, y, z] in robot base frame

    Example:
        >>> world_to_robot([1.46, 0.85, 0.82], robot_origin_in_world=[1.0, 1.0, 0.0])
        [0.46, -0.15, 0.82]
    """
    if robot_origin_in_world is None:
        robot_origin_in_world = [0.0, 0.0, 0.0]

    result = [
        round(position_world[0] - robot_origin_in_world[0], 4),
        round(position_world[1] - robot_origin_in_world[1], 4),
        round(position_world[2] - robot_origin_in_world[2], 4),
    ]
    return result
