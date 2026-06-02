import math
from src.task1.transform_utils import (
    validate_quaternion,
    validate_position,
    validate_transform,
    quaternion_to_yaw,
    camera_to_base,
    world_to_robot,
)



# validate_quaternion
class TestValidateQuaternion:

    def test_valid_identity(self):
        """No rotation — quaternion [0,0,0,1] (used by all real objects)"""
        valid, reason = validate_quaternion([0.0, 0.0, 0.0, 1.0])
        assert valid is True
        assert reason == "OK"

    def test_valid_90_degree_rotation(self):
        """Standard 90° Z rotation"""
        valid, reason = validate_quaternion([0.0, 0.0, 0.707, 0.707])
        assert valid is True
        assert reason == "OK"

    def test_invalid_unnormalized(self):
        """All ones — norm = 2.0, should fail"""
        valid, reason = validate_quaternion([1.0, 1.0, 1.0, 1.0])
        assert valid is False
        assert "norm" in reason

    def test_invalid_zero_quaternion(self):
        """All zeros — norm = 0, should fail"""
        valid, reason = validate_quaternion([0.0, 0.0, 0.0, 0.0])
        assert valid is False

    def test_invalid_wrong_length(self):
        """Only 3 values given"""
        valid, reason = validate_quaternion([0.0, 0.0, 1.0])
        assert valid is False
        assert "4 values" in reason

    def test_valid_near_tolerance(self):
        """Slightly imprecise but within tolerance"""
        valid, reason = validate_quaternion([0.0, 0.0, 0.7071, 0.7071])
        assert valid is True



# validate_position — updated for real coordinate frame
class TestValidatePosition:

    def test_valid_real_object_000(self):
        """obj_000: part_B at [-0.5788, -0.1683, 0.8270]"""
        valid, reason = validate_position([-0.5788, -0.1683, 0.8270])
        assert valid is True
        assert reason == "OK"

    def test_valid_real_object_003(self):
        """obj_003: part_A at [-0.5163, 0.0244, 0.6198] — lowest Z"""
        valid, reason = validate_position([-0.5163, 0.0244, 0.6198])
        assert valid is True
        assert reason == "OK"

    def test_x_too_far_negative(self):
        """Object too far in -X direction"""
        valid, reason = validate_position([-0.9, 0.0, 0.75])
        assert valid is False
        assert "x" in reason

    def test_x_too_close_to_robot(self):
        """Object too close to robot (X > -0.4)"""
        valid, reason = validate_position([-0.3, 0.0, 0.75])
        assert valid is False
        assert "x" in reason

    def test_y_out_of_range(self):
        """Object too far to the side"""
        valid, reason = validate_position([-0.5, 0.9, 0.75])
        assert valid is False
        assert "y" in reason

    def test_z_too_low(self):
        """Object below workspace floor"""
        valid, reason = validate_position([-0.5, 0.0, 0.3])
        assert valid is False
        assert "z" in reason

    def test_boundary_values(self):
        """Exactly at workspace boundary — should pass"""
        valid, reason = validate_position([-0.7, -0.4, 0.5])
        assert valid is True

    def test_wrong_length(self):
        """Only 2D position given"""
        valid, reason = validate_position([-0.5, 0.0])
        assert valid is False


# validate_transform — updated for real coordinate frame
class TestValidateTransform:

    def test_valid_real_object(self):
        """Real object with identity quaternion"""
        valid, reason = validate_transform(
            [-0.5788, -0.1683, 0.8270],
            [0.0, 0.0, 0.0, 1.0]
        )
        assert valid is True

    def test_invalid_position_valid_quat(self):
        """Bad position (positive X — old frame), good quaternion"""
        valid, reason = validate_transform(
            [0.46, -0.15, 0.82],
            [0.0, 0.0, 0.0, 1.0]
        )
        assert valid is False
        assert "Position" in reason

    def test_valid_position_invalid_quat(self):
        """Good position, bad quaternion"""
        valid, reason = validate_transform(
            [-0.55, -0.15, 0.82],
            [1.0, 1.0, 1.0, 1.0]
        )
        assert valid is False
        assert "Quaternion" in reason



# quaternion_to_yaw
class TestQuaternionToYaw:

    def test_identity_zero_yaw(self):
        """Identity quaternion [0,0,0,1] → yaw = 0 (used by all real objects)"""
        yaw = quaternion_to_yaw([0.0, 0.0, 0.0, 1.0])
        assert abs(yaw - 0.0) < 0.001

    def test_90_degrees(self):
        """[0,0,0.707,0.707] should give yaw = π/2 ≈ 1.5708"""
        yaw = quaternion_to_yaw([0.0, 0.0, 0.707, 0.707])
        assert abs(yaw - math.pi / 2) < 0.01

    def test_180_degrees(self):
        """[0,0,1,0] → yaw = π (180°)"""
        yaw = quaternion_to_yaw([0.0, 0.0, 1.0, 0.0])
        assert abs(abs(yaw) - math.pi) < 0.01


# camera_to_base
class TestCameraToBase:

    def test_identity_transform(self):
        """With identity matrix, output should equal input"""
        result = camera_to_base([0.2003, -0.3131, 0.6795])
        assert result == [0.2003, -0.3131, 0.6795]

    def test_returns_3_values(self):
        """Output must always be [x, y, z]"""
        result = camera_to_base([0.1, 0.2, 0.5])
        assert len(result) == 3

    def test_with_custom_transform(self):
        """With an offset transform, position should shift"""
        import numpy as np
        T = np.eye(4)
        T[0, 3] = 0.5   # shift x by +0.5
        result = camera_to_base([0.2003, -0.3131, 0.6795], T_cam_to_base=T)
        assert abs(result[0] - 0.7003) < 0.001


# world_to_robot
class TestWorldToRobot:

    def test_with_robot_offset(self):
        """Subtract robot origin from world position"""
        result = world_to_robot([0.0, 0.0, 0.82], [0.55, 0.15, 0.0])
        assert abs(result[0] - (-0.55)) < 0.001
        assert abs(result[1] - (-0.15)) < 0.001
        assert abs(result[2] - 0.82) < 0.001

    def test_no_offset(self):
        """Robot at world origin → output equals input"""
        result = world_to_robot([-0.5, 0.1, 0.8])
        assert result == [-0.5, 0.1, 0.8]

    def test_returns_3_values(self):
        result = world_to_robot([1.0, 1.0, 1.0], [0.5, 0.5, 0.0])
        assert len(result) == 3
