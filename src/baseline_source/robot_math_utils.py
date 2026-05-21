import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------- rotation ----------
def quat_xyzw_to_R(q):
    q = np.asarray(q, dtype=float).reshape(4)
    q = q / np.linalg.norm(q)
    return R.from_quat(q).as_matrix()

def R_to_quat_xyzw(R_mat):
    return R.from_matrix(R_mat).as_quat()

# ---------- SE(3) ----------
def make_T(R_mat, p):
    T = np.eye(4)
    T[:3,:3] = R_mat
    T[:3,3] = np.asarray(p).reshape(3)
    return T

def inv_T(T):
    R_mat, p = T[:3,:3], T[:3,3]
    out = np.eye(4)
    out[:3,:3] = R_mat.T
    out[:3,3] = -R_mat.T @ p
    return out

# ---------- 2R FK/IK ----------
def fk_2r(q, L1=1.0, L2=0.75):
    t1, t2 = q
    return np.array([
        L1*np.cos(t1) + L2*np.cos(t1+t2),
        L1*np.sin(t1) + L2*np.sin(t1+t2),
        t1+t2
    ])

def jacobian_2r(q, L1=1.0, L2=0.75):
    t1, t2 = q
    return np.array([
        [-L1*np.sin(t1)-L2*np.sin(t1+t2), -L2*np.sin(t1+t2)],
        [ L1*np.cos(t1)+L2*np.cos(t1+t2),  L2*np.cos(t1+t2)]
    ])
