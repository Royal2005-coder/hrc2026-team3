"""
transform_utils.py — Coordinate-frame transforms for HRC2026 Task 1.

Convention:
    p_base = T_base_camera @ p_camera   (homogeneous)

T_base_camera is a 4×4 matrix that maps points from the camera frame
into the robot-base frame.
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def transform_point(T_ab: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """
    Apply a 4×4 homogeneous transform to a 3-D point.

    Parameters
    ----------
    T_ab : (4, 4) — transform from frame B to frame A.
    p_b  : (3,)   — point expressed in frame B.

    Returns
    -------
    np.ndarray (3,) — point expressed in frame A.
    """
    p_h = np.array([p_b[0], p_b[1], p_b[2], 1.0])
    return (T_ab @ p_h)[:3]


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Efficient inverse of a rigid-body 4×4 transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# ---------------------------------------------------------------------------
# Quaternion ↔ Rotation matrix  (xyzw convention)
# ---------------------------------------------------------------------------
def quaternion_to_rotation_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] → 3×3 rotation matrix."""
    x, y, z, w = q_xyzw
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix → quaternion [x, y, z, w]."""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4×4 homogeneous transform from R (3×3) and t (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# Transform validation (Module 4 spec)
# ---------------------------------------------------------------------------
def check_transform(T, atol: float = 1e-3) -> dict:
    """
    Validate a 4×4 homogeneous transform.

    Returns dict with det_R, ortho_err, is_valid — mirrors Module 4 skeleton.
    """
    T = np.asarray(T, dtype=float)
    assert T.shape == (4, 4), f"Expected 4×4, got {T.shape}"
    R = T[:3, :3]
    det_R = float(np.linalg.det(R))
    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
    is_valid = abs(det_R - 1.0) < atol and ortho_err < atol
    return {"det_R": det_R, "ortho_err": ortho_err, "is_valid": is_valid}


# ---------------------------------------------------------------------------
# Sanity tests
# ---------------------------------------------------------------------------
def sanity_identity(p: np.ndarray) -> bool:
    """T = I should leave point unchanged."""
    p_out = transform_point(np.eye(4), p)
    return bool(np.allclose(p, p_out))


def sanity_workspace_bounds(p_base: np.ndarray,
                            x_range: Tuple[float, float] = (-0.5, 1.0),
                            y_range: Tuple[float, float] = (-0.5, 0.5),
                            z_range: Tuple[float, float] = (0.0, 1.5)) -> bool:
    """Check whether a base-frame point lies within the robot workspace."""
    x, y, z = p_base
    return (x_range[0] <= x <= x_range[1] and
            y_range[0] <= y <= y_range[1] and
            z_range[0] <= z <= z_range[1])


def run_transform_sanity(T_base_camera: np.ndarray,
                         test_points_cam: list[np.ndarray] | None = None,
                         report_path: str = "transform_sanity_report.md") -> str:
    """
    Run standard sanity checks and write a markdown report.

    Parameters
    ----------
    T_base_camera   : 4×4 transform (camera → base)
    test_points_cam : list of 3-D points in camera frame to test
    report_path     : output file path

    Returns
    -------
    str — path to generated report
    """
    if test_points_cam is None:
        test_points_cam = [
            np.array([0.0, 0.0, 0.7]),
            np.array([0.1, 0.0, 0.7]),
            np.array([-0.1, 0.0, 0.7]),
            np.array([0.0, 0.05, 0.5]),
        ]

    lines = ["# Transform Sanity Report\n"]

    ok = sanity_identity(np.array([1.0, 2.0, 3.0]))
    lines.append(f"- Identity test: {'PASS' if ok else 'FAIL'}")

    det = np.linalg.det(T_base_camera[:3, :3])
    lines.append(f"- det(R) = {det:.6f} (should be ≈ 1.0): "
                 f"{'PASS' if abs(det - 1.0) < 1e-4 else 'FAIL'}")

    lines.append("\n## Test points\n")
    lines.append("| p_camera | p_base | in_workspace |")
    lines.append("|----------|--------|-------------|")
    for p_cam in test_points_cam:
        p_base = transform_point(T_base_camera, p_cam)
        in_ws = sanity_workspace_bounds(p_base)
        lines.append(f"| {np.round(p_cam, 4).tolist()} "
                     f"| {np.round(p_base, 4).tolist()} "
                     f"| {'YES' if in_ws else 'NO'} |")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return report_path
