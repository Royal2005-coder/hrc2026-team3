import numpy as np
from scipy.spatial.transform import Rotation as R
def quat_xyzw_to_R(quat_xyzw):
    """ Convert quaternion from xyzw format to a rotation matrix."""
    q = np.array(quat_xyzw, dtype = float) 
    q = q/ np.linalg.norm(q)  # Normalize the quaternion
    return R.from_quat(q).as_matrix()
def make_T(quat_xyzw, position_m):
    """ Create a homogeneous transformation matrix from quaternion and position."""
    R_mat = quat_xyzw_to_R(quat_xyzw)
    p = np.asarray(position_m, dtype=float).reshape(3)
    T = np.eye(4, dtype = float)
    T[:3, :3] = R_mat
    T[:3, 3] = p
    return T
# def make_R(roll, pitch, yaw):
#     """ Create a rotation matrix from roll, pitch, yaw angles."""
#     return R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

