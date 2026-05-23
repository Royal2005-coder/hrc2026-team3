"""
camera_utils.py — Camera intrinsics, pixel-to-3D conversion, robust depth extraction.

HRC2026 Task 1 — Người 1: Camera/RGB-D + Perception
"""

import numpy as np
import csv
import os


# ---------------------------------------------------------------------------
# Camera Intrinsics
# ---------------------------------------------------------------------------
class CameraIntrinsics:
    """Store and validate camera intrinsic parameters."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 width: int = None, height: int = None, depth_unit: str = "meter"):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = width
        self.height = height
        self.depth_unit = depth_unit  # "meter" | "millimeter"

    def K(self) -> np.ndarray:
        """Return 3×3 intrinsic matrix."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=float)

    def depth_scale(self) -> float:
        """Factor to convert raw depth to metres."""
        if self.depth_unit == "millimeter":
            return 1e-3
        return 1.0

    def __repr__(self):
        return (f"CameraIntrinsics(fx={self.fx}, fy={self.fy}, "
                f"cx={self.cx}, cy={self.cy}, "
                f"{self.width}x{self.height}, unit={self.depth_unit})")


# ---------------------------------------------------------------------------
# Pixel → 3-D (camera frame)
# ---------------------------------------------------------------------------
def pixel_to_camera_point(u: float, v: float, depth_z: float,
                          intr: CameraIntrinsics) -> np.ndarray:
    """
    Back-project a pixel (u, v) + depth → 3-D point in camera frame.

    Parameters
    ----------
    u, v      : pixel coordinates (column, row)
    depth_z   : depth in metres (caller must scale if raw unit differs)
    intr      : CameraIntrinsics

    Returns
    -------
    np.ndarray  shape (3,) — [X_cam, Y_cam, Z_cam]
    """
    if depth_z is None or depth_z <= 0 or not np.isfinite(depth_z):
        raise ValueError(f"Invalid depth value: {depth_z}")

    x = (u - intr.cx) * depth_z / intr.fx
    y = -(v - intr.cy) * depth_z / intr.fy  # Isaac Sim camera Y-up, image v-down
    z = depth_z
    return np.array([x, y, z], dtype=float)


# ---------------------------------------------------------------------------
# Depth validity helpers (Module 4 spec)
# ---------------------------------------------------------------------------
def valid_depth_mask(depth: np.ndarray,
                     min_depth: float = 0.05,
                     max_depth: float = 5.0) -> np.ndarray:
    """Boolean mask: True where depth is finite and within [min_depth, max_depth]."""
    depth = np.asarray(depth, dtype=float)
    return np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)


def median_depth_in_mask(depth: np.ndarray,
                         mask: np.ndarray,
                         min_depth: float = 0.05,
                         max_depth: float = 5.0) -> float | None:
    """Median of valid depth values inside a binary mask."""
    valid = valid_depth_mask(depth, min_depth, max_depth) & (mask.astype(bool))
    values = depth[valid]
    if values.size == 0:
        return None
    return float(np.median(values))


# ---------------------------------------------------------------------------
# Robust depth sampling
# ---------------------------------------------------------------------------
def robust_depth_from_patch(depth: np.ndarray, u: float, v: float,
                            radius: int = 3) -> float | None:
    """Median depth in a (2*radius+1) square patch around (u, v)."""
    h, w = depth.shape[:2]
    u_int, v_int = int(round(u)), int(round(v))

    x1, x2 = max(0, u_int - radius), min(w, u_int + radius + 1)
    y1, y2 = max(0, v_int - radius), min(h, v_int + radius + 1)

    patch = depth[y1:y2, x1:x2]
    valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 5.0)]
    return None if len(valid) == 0 else float(np.median(valid))


def robust_depth_from_mask(depth: np.ndarray, mask: np.ndarray) -> float | None:
    """Median depth over a binary mask region."""
    valid = depth[(mask > 0) & np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)]
    return None if len(valid) == 0 else float(np.median(valid))


# ---------------------------------------------------------------------------
# Depth sanity check
# ---------------------------------------------------------------------------
def depth_sanity(depth: np.ndarray) -> dict:
    """Return a sanity dict: shape, valid_ratio, min/median/max."""
    valid_mask = np.isfinite(depth) & (depth > 0)
    info = {
        "shape": list(depth.shape),
        "dtype": str(depth.dtype),
        "valid_ratio": float(valid_mask.mean()),
    }
    if valid_mask.any():
        vals = depth[valid_mask]
        info["min"] = float(vals.min())
        info["median"] = float(np.median(vals))
        info["max"] = float(vals.max())
        med = info["median"]
        info["guessed_unit"] = "meter" if med < 10 else "millimeter"
    else:
        info["min"] = info["median"] = info["max"] = None
        info["guessed_unit"] = "unknown"
    return info


def write_depth_sanity_report(info: dict, path: str = "depth_sanity_report.md"):
    """Write a concise markdown depth-sanity report."""
    lines = [
        "# Depth Sanity Report\n",
        f"- **Shape**: {info['shape']}",
        f"- **Dtype**: {info['dtype']}",
        f"- **Valid ratio**: {info['valid_ratio']:.4f}",
        f"- **Min depth**: {info['min']}",
        f"- **Median depth**: {info['median']}",
        f"- **Max depth**: {info['max']}",
        f"- **Guessed unit**: {info['guessed_unit']}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Camera config CSV
# ---------------------------------------------------------------------------
def save_camera_config_csv(intr: CameraIntrinsics,
                           T_source: str = "unknown",
                           camera_name: str = "front_rgbd",
                           notes: str = "",
                           path: str = "camera_config_sheet.csv"):
    """Persist camera config so the whole team shares one source of truth."""
    header = ["camera_name", "width", "height", "fx", "fy", "cx", "cy",
              "depth_unit", "T_base_camera_source", "notes"]
    row = [camera_name, intr.width, intr.height,
           intr.fx, intr.fy, intr.cx, intr.cy,
           intr.depth_unit, T_source, notes]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(row)
    return path
