"""
test_camera_and_transform.py — Unit tests for Người 1 core modules.

Run:  python -m pytest test_camera_and_transform.py -v
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from camera_utils import (
    CameraIntrinsics,
    pixel_to_camera_point,
    robust_depth_from_patch,
    robust_depth_from_mask,
    depth_sanity,
)
from transform_utils import (
    transform_point,
    invert_transform,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    make_transform,
    sanity_identity,
    sanity_workspace_bounds,
)


# ── CameraIntrinsics ──────────────────────────────────────────────────────
class TestCameraIntrinsics:
    def test_K_matrix(self):
        intr = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240)
        K = intr.K()
        assert K.shape == (3, 3)
        assert K[0, 0] == 600
        assert K[1, 2] == 240

    def test_depth_scale_meter(self):
        intr = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240, depth_unit="meter")
        assert intr.depth_scale() == 1.0

    def test_depth_scale_mm(self):
        intr = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240, depth_unit="millimeter")
        assert abs(intr.depth_scale() - 1e-3) < 1e-9


# ── pixel_to_camera_point ─────────────────────────────────────────────────
class TestPixelTo3D:
    intr = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240)

    def test_optical_center(self):
        """Pixel at optical centre with depth 1.0 → (0, 0, 1)."""
        p = pixel_to_camera_point(320, 240, 1.0, self.intr)
        np.testing.assert_allclose(p, [0.0, 0.0, 1.0], atol=1e-6)

    def test_off_centre(self):
        p = pixel_to_camera_point(320 + 60, 240, 1.0, self.intr)
        expected_x = 60 / 600  # 0.1
        np.testing.assert_allclose(p[0], expected_x, atol=1e-6)
        assert p[2] == 1.0

    def test_invalid_depth_raises(self):
        import pytest
        with pytest.raises(ValueError):
            pixel_to_camera_point(320, 240, 0.0, self.intr)
        with pytest.raises(ValueError):
            pixel_to_camera_point(320, 240, -1.0, self.intr)
        with pytest.raises(ValueError):
            pixel_to_camera_point(320, 240, float('nan'), self.intr)

    def test_depth_scaling(self):
        """Doubling depth should double all coordinates proportionally."""
        p1 = pixel_to_camera_point(380, 300, 1.0, self.intr)
        p2 = pixel_to_camera_point(380, 300, 2.0, self.intr)
        np.testing.assert_allclose(p2, p1 * 2.0, atol=1e-6)


# ── robust_depth_from_patch ───────────────────────────────────────────────
class TestRobustDepth:
    def test_uniform_depth(self):
        depth = np.full((480, 640), 0.75)
        z = robust_depth_from_patch(depth, 320, 240, radius=3)
        assert abs(z - 0.75) < 1e-6

    def test_handles_nan(self):
        depth = np.full((480, 640), np.nan)
        z = robust_depth_from_patch(depth, 320, 240, radius=3)
        assert z is None

    def test_median_filters_outlier(self):
        depth = np.full((480, 640), 0.75)
        depth[240, 320] = 100.0  # outlier at centre
        z = robust_depth_from_patch(depth, 320, 240, radius=3)
        assert abs(z - 0.75) < 0.01  # median ignores outlier

    def test_mask_depth(self):
        depth = np.full((480, 640), 0.5)
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[200:260, 300:360] = 255
        z = robust_depth_from_mask(depth, mask)
        assert abs(z - 0.5) < 1e-6


# ── depth_sanity ──────────────────────────────────────────────────────────
class TestDepthSanity:
    def test_normal_depth(self):
        depth = np.random.uniform(0.3, 1.5, (480, 640))
        info = depth_sanity(depth)
        assert info["valid_ratio"] > 0.99
        assert info["guessed_unit"] == "meter"

    def test_mm_depth(self):
        depth = np.random.uniform(300, 1500, (480, 640))
        info = depth_sanity(depth)
        assert info["guessed_unit"] == "millimeter"


# ── transform_point ───────────────────────────────────────────────────────
class TestTransform:
    def test_identity(self):
        p = np.array([1.0, 2.0, 3.0])
        p_out = transform_point(np.eye(4), p)
        np.testing.assert_allclose(p, p_out)

    def test_pure_translation(self):
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        p = np.array([0.0, 0.0, 0.0])
        p_out = transform_point(T, p)
        np.testing.assert_allclose(p_out, [1.0, 2.0, 3.0])

    def test_invert_roundtrip(self):
        T = np.eye(4)
        T[:3, :3] = quaternion_to_rotation_matrix([0.0, 0.0, np.sqrt(2)/2, np.sqrt(2)/2])
        T[:3, 3] = [0.5, -0.3, 0.8]
        T_inv = invert_transform(T)
        result = T @ T_inv
        np.testing.assert_allclose(result, np.eye(4), atol=1e-6)


# ── quaternion roundtrip ──────────────────────────────────────────────────
class TestQuaternion:
    def test_identity_quaternion(self):
        R = quaternion_to_rotation_matrix([0, 0, 0, 1])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)

    def test_roundtrip(self):
        q_in = np.array([0.0, 0.0, 0.707107, 0.707107])
        R = quaternion_to_rotation_matrix(q_in)
        q_out = rotation_matrix_to_quaternion(R)
        # quaternion sign ambiguity: q and -q represent same rotation
        if np.dot(q_in, q_out) < 0:
            q_out = -q_out
        np.testing.assert_allclose(q_in, q_out, atol=1e-4)


# ── sanity helpers ────────────────────────────────────────────────────────
class TestSanity:
    def test_identity_sanity(self):
        assert sanity_identity(np.array([5.0, 6.0, 7.0]))

    def test_workspace_in(self):
        assert sanity_workspace_bounds(np.array([0.3, 0.1, 0.8]))

    def test_workspace_out(self):
        assert not sanity_workspace_bounds(np.array([10.0, 10.0, 10.0]))
