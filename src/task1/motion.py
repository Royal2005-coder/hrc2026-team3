import csv
import math
import numpy as np
import robot_math_utils as rmu
import json
import yaml
import sys
import time
import types
import torch
from pathlib import Path
from enum import Enum
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

# DualArmIK lives in baseline_source alongside main_fixed.py.
# Add both the relative path (for standalone use) and legacy dev-machine path as fallback.
sys.path.append(str(Path(__file__).parent.parent / 'baseline_source'))
sys.path.append('/home/ubuntu/vinh/Ubtech_sim_ref/source')
from DualArmIK import DualArmIK

# ======================================================================
# 1. ENUMS & CONFIGURATION
# ======================================================================

class PrimitiveEvent(Enum):
    SUCCESS = "PRIMITIVE_SUCCESS"
    GRASP_FAIL = "PRIMITIVE_GRASP_FAIL"
    DROP = "PRIMITIVE_DROP"
    WRONG_BIN = "PRIMITIVE_WRONG_BIN"
    COLLISION = "PRIMITIVE_COLLISION"
    TIMEOUT = "PRIMITIVE_TIMEOUT"

@dataclass
class PrimitiveResult:
    primitive_name: str
    success: bool
    elapsed_s: float
    retry_count: int
    failure_reason: Optional[str]
    metrics: dict

EVENT_RETRY_POLICIES = {
    PrimitiveEvent.GRASP_FAIL: {"max_attempts": 3, "speed": "slow", "timeout_s": 25.0, "on_exceed": "skip"},
    PrimitiveEvent.DROP: {"max_attempts": 2, "speed": "very_slow", "timeout_s": 30.0, "on_exceed": "skip"},
    PrimitiveEvent.TIMEOUT: {"max_attempts": 2, "speed": "slow", "timeout_s": 35.0, "on_exceed": "abort"},
    PrimitiveEvent.COLLISION: {"max_attempts": 1, "speed": "normal", "timeout_s": 20.0, "on_exceed": "abort"}
}

ENABLE_WORKSPACE_VALIDATION = True

def load_workspace_bounds_from_yaml(config_path: Path) -> dict:
    default_bounds = {"x": (0.4, 1.0), "y": (-0.5, 0.5), "z": (0.9, 1.5)}
    try:
        if not config_path.exists(): return default_bounds
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        bounds_cfg = config.get("workspace_bounds", {})
        if not bounds_cfg: return default_bounds
        return {
            "x": tuple(bounds_cfg.get("x", default_bounds["x"])),
            "y": tuple(bounds_cfg.get("y", default_bounds["y"])),
            "z": tuple(bounds_cfg.get("z", default_bounds["z"])),
        }
    except Exception: return default_bounds

WORKSPACE_BOUNDS = load_workspace_bounds_from_yaml(Path("/home/ubuntu/thu/configs/planner.yaml"))

# Bin world positions per part type [x, y, z] in Isaac Sim world frame.
# Part_Sorting.yaml defines ONE physical box at [1.2, 0.3, 1.05].
# Both PartA and PartB go to this same box; PartB is offset slightly within.
_PART_BIN_WORLD = {
    "PartA": [1.2,  0.3, 1.05],
    "PartB": [1.1,  0.3, 1.05],  # same physical bin, slightly less reach (XY=0.640m)
}

def validate_position_in_workspace(position_m: list, bounds: dict = None) -> tuple:
    if bounds is None: bounds = WORKSPACE_BOUNDS
    x, y, z = position_m
    errors = []
    if not (bounds["x"][0] <= x <= bounds["x"][1]): errors.append(f"X={x:.4f}m out of bounds")
    if not (bounds["y"][0] <= y <= bounds["y"][1]): errors.append(f"Y={y:.4f}m out of bounds")
    if not (bounds["z"][0] <= z <= bounds["z"][1]): errors.append(f"Z={z:.4f}m out of bounds")
    if errors: return False, "; ".join(errors)
    return True, "Valid"


# ======================================================================
# 2. WORKAROUND HELPERS — fixes cho BTC code issues
# ======================================================================

def _check_reached_via_fk(robot, target_xyzrpy, side, pos_tol, rot_tol):
    """[FIX #1] Tự dùng FK của solver để check arm đã reach target chưa.
    Không phụ thuộc vào return value của control_dual_arm_ik (vốn return None).
    """
    if robot is None or robot.ik_solver is None or target_xyzrpy is None:
        return True
    try:
        import pinocchio as pin
        # Sync current joint state vào solver
        joints = robot.get_joint_states()
        if joints is None:
            return False
        # joints['positions'] có thể là [[...]] hoặc [...] tùy version Isaac
        positions = joints['positions']
        if positions and isinstance(positions[0], list):
            positions = positions[0]
        robot.ik_solver.sync_joint_positions(joints['names'], positions)
        
        # FK → current EE pose
        current_pose = robot.ik_solver.get_ee_pose(side)
        target_se3 = robot.ik_solver.xyzrpy_to_se3(target_xyzrpy)
        
        # Error in SE3
        err = pin.log(current_pose.actInv(target_se3)).vector
        pos_err = float(np.linalg.norm(err[:3]))
        rot_err = float(np.linalg.norm(err[3:]))
        
        return pos_err < pos_tol and rot_err < rot_tol, pos_err, rot_err
    except Exception as e:
        print(f"  [_check_reached] error: {e}")
        return False, 999.0, 999.0


def patch_robot_workarounds(robot):
    """[FIX #3] Áp các workaround lên robot instance:
    - Tăng EMA smoothing alpha cho responsive hơn
    - (Optional) Disable smoothing nếu cần debug
    """
    if robot is None:
        return
    try:
        if hasattr(robot, '_smooth_alpha'):
            old = robot._smooth_alpha
            robot._smooth_alpha = 0.8
            print(f"[Patch] EMA alpha: {old} → {robot._smooth_alpha}")
    except Exception as e:
        print(f"[Patch] Failed to patch alpha: {e}")


def reset_robot_state_full(robot, world, verbose=True):
    """[FIX #2] Reset toàn bộ runtime state của robot trước khi start plan mới:
    - Ensure _physics_view is valid (reinit if needed)
    - Teleport joints về initial
    - Clear _last_arm_positions (EMA state)
    - Reset IK solver runtime state
    - Re-sync joint state vào solver
    """
    if robot is None:
        return
    try:
        import torch
        # 0. Ensure articulation physics view is valid after any FixedJoint mutations
        if hasattr(robot, '_reinitialize_physics'):
            if not hasattr(robot._articulation, '_physics_view') or robot._articulation._physics_view is None:
                if verbose: print("  → _physics_view missing — reinitializing articulation...")
                robot._reinitialize_physics()

        # 1. Teleport joints về initial
        if robot._articulation is not None and robot.inital_joint_positions is not None:
            try:
                s2_joint_names = robot._articulation.dof_names
                s2_joint_indices = [robot._articulation.get_dof_index(n) for n in s2_joint_names]
                robot._articulation.set_joint_positions(
                    torch.tensor(robot.inital_joint_positions, dtype=torch.float32),
                    joint_indices=torch.tensor(s2_joint_indices, dtype=torch.int32)
                )
                if verbose: print("  → Teleported joints to initial")
            except AttributeError as _ae:
                if "_physics_view" in str(_ae):
                    if verbose: print("  ⚠ set_joint_positions failed (_physics_view) — skipping teleport")
                else:
                    raise
        
        # 2. Clear EMA state
        if hasattr(robot, '_last_arm_positions'):
            robot._last_arm_positions = {}
            if verbose: print("  → Cleared _last_arm_positions (EMA state)")
        
        # 3. Step world để physics settle
        if world is not None:
            for _ in range(30): world.step(render=True)
        
        # 4. Reset IK solver state + re-sync
        if hasattr(robot, 'ik_solver') and robot.ik_solver is not None:
            if hasattr(robot.ik_solver, 'reset_runtime_state'):
                robot.ik_solver.reset_runtime_state()
                if verbose: print("  → IK solver runtime state reset")
            
            joints = robot.get_joint_states()
            if joints is not None:
                positions = joints['positions']
                if positions and isinstance(positions[0], list):
                    positions = positions[0]
                robot.ik_solver.sync_joint_positions(joints['names'], positions)
                if hasattr(robot.ik_solver, 'save_initial_q'):
                    robot.ik_solver.save_initial_q()
                if verbose: print("  → Re-synced IK solver with current joints")
        
        # 5. Settle thêm
        if world is not None:
            for _ in range(30): world.step(render=True)
    except Exception as e:
        print(f"⚠️ reset_robot_state_full error: {e}")


def close_gripper_with_width(robot, side, width):
    """[FIX #4] Close gripper với width tùy chỉnh (BTC's close_gripper không nhận width)."""
    if not robot or not robot._articulation:
        return
    try:
        import torch
        from isaacsim.core.utils.types import ArticulationActions
        
        finger_names = (['L_finger1_joint', 'L_finger2_joint'] if side == 'left' 
                        else ['R_finger1_joint', 'R_finger2_joint'])
        dof_names = robot._articulation.dof_names
        indices = [robot._articulation.get_dof_index(n) 
                   for n in finger_names if n in dof_names]
        if indices:
            positions = [width] * len(indices)
            robot._articulation.apply_action(
                ArticulationActions(
                    joint_positions=torch.tensor([positions], dtype=torch.float32),
                    joint_indices=torch.tensor(indices, dtype=torch.int32),
                )
            )
    except Exception as e:
        print(f"  close_gripper_with_width error: {e} — fallback to default close_gripper")
        try:
            robot.close_gripper(side=side)
        except Exception:
            pass


def verify_grasp_success(robot, side, min_finger_gap_m: float = 0.004) -> bool:
    """Check if gripper is holding an object by reading actual finger joint positions.

    Returns True if fingers stopped short of fully closing (object between them).
    Returns False if fingers closed completely (grasped nothing).
    """
    if not robot or not robot._articulation:
        return True
    try:
        finger_names = (['L_finger1_joint', 'L_finger2_joint'] if side == 'left'
                        else ['R_finger1_joint', 'R_finger2_joint'])
        dof_names = robot._articulation.dof_names
        indices = [robot._articulation.get_dof_index(n)
                   for n in finger_names if n in dof_names]
        if not indices:
            print("  [VERIFY_GRASP] No finger joints found — assuming success")
            return True
        joint_pos = robot._articulation.get_joint_positions()
        positions = joint_pos[0] if (hasattr(joint_pos, 'shape') and len(joint_pos.shape) > 1) else joint_pos
        finger_positions = [float(positions[i]) for i in indices]
        avg_pos = sum(finger_positions) / len(finger_positions)
        print(f"  [VERIFY_GRASP] Finger positions: {[f'{p:.4f}m' for p in finger_positions]} | avg={avg_pos:.4f}m")
        # For this robot: close_width=0.01, open_width=-0.0215
        # Fingers at close limit (≈0.01) = empty gripper
        # Fingers stopped before close limit = object is blocking
        fully_closed = getattr(robot, 'gripper_close_width', 0.01)
        margin = 0.003  # 3mm noise margin
        if avg_pos >= (fully_closed - margin):
            print(f"  [VERIFY_GRASP] ✗ Gripper fully closed ({avg_pos:.4f}m ≈ close_limit {fully_closed:.4f}m) — empty")
            return False
        else:
            print(f"  [VERIFY_GRASP] ✓ Fingers stopped at {avg_pos:.4f}m (close_limit={fully_closed:.4f}m) — object detected")
            return True
    except Exception as e:
        print(f"  [VERIFY_GRASP] error: {e} — assuming success")
        return True



def _create_grasp_joint(world, robot, object_prim_path: str, side: str):
    """Attach object to robot gripper via USD FixedJoint for reliable sim grasping.
    Uses pre-cached gripper link path (set at initialize_ik time) to avoid
    stage.TraverseAll during active simulation.
    Returns joint_path string on success, None on failure.
    """
    try:
        from pxr import UsdPhysics, UsdGeom, Usd, Sdf, Gf
        from isaacsim.core.utils.stage import get_current_stage
        stage = get_current_stage()

        # Use cached gripper link (populated by initialize_ik at startup)
        grip_link = None
        if hasattr(robot, '_gripper_link_cache'):
            grip_link = robot._gripper_link_cache.get(side)
        if grip_link is None:
            print(f"  [GRASP_JOINT] ⚠ No cached gripper link for side={side} — skip attachment")
            return None

        obj_prim = stage.GetPrimAtPath(object_prim_path)
        if not obj_prim.IsValid():
            print(f"  [GRASP_JOINT] ⚠ Object prim not found: {object_prim_path}")
            return None

        joint_path = f"{grip_link}/grasp_attach"
        existing = stage.GetPrimAtPath(joint_path)
        if existing.IsValid():
            stage.RemovePrim(existing.GetPath())

        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(grip_link)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(object_prim_path)])

        # Set joint local frames so the constraint already holds at current poses.
        # Without this, USD physics sees "disjointed body transforms" (wrist and object
        # are at different world positions) and SNAPS both bodies together — this
        # violently jerks the arm out of its z_down grasp orientation, causing
        # S3_LIFT to face rot_err≈1.0 rad and stall for the full timeout.
        #
        # Formula: L0=identity (joint frame at body0/wrist origin),
        #          L1 = W0 × W1⁻¹  (wrist world transform expressed in object local frame)
        # Verification: L0×W0 = W0  and  L1×W1 = (W0×W1⁻¹)×W1 = W0  → frames coincide ✓
        try:
            wrist_prim = stage.GetPrimAtPath(grip_link)
            W0 = UsdGeom.Xformable(wrist_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            W1 = UsdGeom.Xformable(obj_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            L1 = W0 * W1.GetInverse()

            t = L1.ExtractTranslation()
            q = L1.ExtractRotationQuat()
            im = q.GetImaginary()

            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(
                float(q.GetReal()), Gf.Vec3f(float(im[0]), float(im[1]), float(im[2]))))
            print(f"  [GRASP_JOINT] No-snap frames set: L1_t=({float(t[0]):.3f},{float(t[1]):.3f},{float(t[2]):.3f})")
        except Exception as _fe:
            print(f"  [GRASP_JOINT] ⚠ Could not set joint frames ({_fe}) — snap may occur")

        print(f"  [GRASP_JOINT] ✓ Attached: {object_prim_path} → {grip_link}")
        return joint_path
    except Exception as e:
        print(f"  [GRASP_JOINT] Error creating joint: {e}")
        return None


def _remove_grasp_joint(world, joint_path: str):
    """Remove grasp fixed joint to release object."""
    if not joint_path:
        return
    try:
        from isaacsim.core.utils.stage import get_current_stage
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(joint_path)
        if prim.IsValid():
            stage.RemovePrim(prim.GetPath())
            print(f"  [GRASP_JOINT] ✓ Released: {joint_path}")
    except Exception as e:
        print(f"  [GRASP_JOINT] Error removing joint: {e}")


# ======================================================================
# 3. CORE MOTION & INTERPOLATION
# ======================================================================

def load_action_plans_from_person2(file_path: str = None) -> list:
    if file_path is None: file_path = "/home/ubuntu/thu/lab_outputs/planner_outputs/action_plans_for_person3_demo.json"
    with open(file_path, "r", encoding="utf-8") as f: return json.load(f)


def load_action_plans_from_scene(template_file: str = None) -> list:
    """Build action plans using ACTUAL object positions from USD scene at runtime.

    Reads /Replicator/Ref_Xform_NN for objects and /Root/Box for bin.
    Uses template JSON only for class_id, grasp_hint, etc.
    Falls back to JSON if Isaac Sim is unavailable.
    """
    if template_file is None:
        template_file = "/home/ubuntu/thu/lab_outputs/planner_outputs/action_plans_for_person3_demo.json"
    try:
        with open(template_file, "r") as f:
            template_plans = json.load(f)
    except Exception as e:
        print(f"[SCENE_LOAD] Cannot load template: {e}")
        template_plans = []

    try:
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import Usd, UsdGeom
    except ImportError:
        print("[SCENE_LOAD] Isaac Sim unavailable — using JSON fallback")
        return template_plans

    try:
        stage = get_current_stage()

        def _prim_world_pos(prim_path: str):
            p = stage.GetPrimAtPath(prim_path)
            if not p.IsValid():
                return None
            t = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()).ExtractTranslation()
            return np.array([float(t[0]), float(t[1]), float(t[2])])

        def _world_to_base(w):
            return np.array([w[1] + 0.20, -w[0] + 0.70, w[2] - 0.9005])

        # Read bin position — try multiple candidate prim paths, then scan by name
        _BIN_CANDIDATES = [
            "/Root/Box", "/World/Box", "/Root/Bin", "/World/Bin",
            "/Root/sorting_bin", "/World/sorting_bin",
            "/Root/Tray", "/World/Tray", "/Root/Container", "/World/Container",
            "/Root/basket", "/World/basket",
        ]
        bin_world = None
        bin_found_at = None
        for _bp in _BIN_CANDIDATES:
            _w = _prim_world_pos(_bp)
            if _w is not None:
                bin_world = _w
                bin_found_at = _bp
                break

        if bin_world is None:
            # Broad scan: find any prim with bin/box/tray in its name at table height
            _bin_kw = ["bin", "box", "tray", "basket", "container", "sort"]
            for prim in stage.TraverseAll():
                _pname = prim.GetName().lower()
                if any(_k in _pname for _k in _bin_kw):
                    _path = str(prim.GetPath())
                    _w = _prim_world_pos(_path)
                    if _w is not None and 0.85 < _w[2] < 1.30:
                        bin_world = _w
                        bin_found_at = _path
                        print(f"[SCENE_LOAD] Bin found by name-scan: {_path}")
                        break

        if bin_world is None:
            print("[SCENE_LOAD] ⚠ Bin prim not found in stage — using JSON fallback")
            bin_base = None
        else:
            bin_base = _world_to_base(bin_world)
            _reach_xy = math.sqrt(bin_base[0]**2 + bin_base[1]**2)
            print(f"[SCENE_LOAD] Bin at '{bin_found_at}': "
                  f"world=({bin_world[0]:.3f},{bin_world[1]:.3f},{bin_world[2]:.3f})"
                  f"  base=({bin_base[0]:.3f},{bin_base[1]:.3f},{bin_base[2]:.3f})"
                  f"  XY-reach={_reach_xy:.3f}m")

        # Read object positions: /Replicator/Ref_Xform_01 … Ref_Xform_20
        obj_list = []
        for i in range(1, 21):
            path = f"/Replicator/Ref_Xform_{i:02d}"
            w = _prim_world_pos(path)
            if w is None:
                break
            b = _world_to_base(w)
            obj_list.append((path, w, b))
            print(f"[SCENE_LOAD] {path}  world=({w[0]:.4f},{w[1]:.4f},{w[2]:.4f})"
                  f"  base=({b[0]:.4f},{b[1]:.4f},{b[2]:.4f})")

        if not obj_list:
            print("[SCENE_LOAD] No Replicator objects found — using JSON fallback")
            return template_plans

        # Load part type map from task1_workpieces.json (written by main_fixed.py after settle)
        # Maps prim_path or prim_name → "PartA" | "PartB"
        _type_map = {}
        _wp_type_paths = [
            Path(__file__).parent.parent.parent / "task1_workpieces.json",
            Path("/home/ubuntu/thu/task1_workpieces.json"),
            Path("task1_workpieces.json"),
        ]
        for _wp_t in _wp_type_paths:
            if _wp_t.exists():
                try:
                    import json as _json
                    with open(_wp_t) as _f:
                        _wdata = _json.load(_f)
                    for _w in _wdata.get("workpieces", []):
                        _pp = _w.get("prim_path") or f"/Replicator/{_w.get('id', '')}"
                        _tp = _w.get("type", "PartA")
                        _type_map[_pp] = _tp
                        if _w.get("id"):
                            _type_map[_w["id"]] = _tp
                    print(f"[SCENE_LOAD] Type map from {_wp_t.name}: "
                          f"{ {k.split('/')[-1]: v for k, v in _type_map.items()} }")
                except Exception as _e:
                    print(f"[SCENE_LOAD] Type map error: {_e}")
                break

        plans = []
        for idx, (path, w_obj, b_obj) in enumerate(obj_list):
            tmpl = template_plans[idx] if idx < len(template_plans) else {}

            # Determine part type: type map → index fallback (first half PartA, rest PartB)
            prim_name = path.split("/")[-1]
            part_type = _type_map.get(path) or _type_map.get(prim_name)
            if part_type is None:
                part_type = "PartA" if idx < max(len(obj_list) // 2, 1) else "PartB"
                print(f"[SCENE_LOAD] {prim_name}: type unknown — index fallback → {part_type}")
            else:
                print(f"[SCENE_LOAD] {prim_name}: type={part_type}")

            # Bin position: use stage-scanned bin for PartA; _PART_BIN_WORLD for PartB.
            # Both types target the same physical box; PartB uses a slightly less reach offset.
            if bin_base is not None and part_type == "PartA":
                bin_pos = bin_base.tolist()
            else:
                _bin_world_typed = np.array(_PART_BIN_WORLD.get(part_type, _PART_BIN_WORLD["PartA"]))
                bin_pos = _world_to_base(_bin_world_typed).tolist()
            print(f"[SCENE_LOAD] {prim_name} [{part_type}] bin_base={[round(v,3) for v in bin_pos]}")

            bin_quat = tmpl.get("bin_pose_base", {}).get("quaternion_xyzw", [0, 0, 0, 1])
            obj_quat = tmpl.get("object_pose_base", {}).get("quaternion_xyzw", [0, 0, 0, 1])
            grasp_hint = dict(tmpl.get("grasp_hint", {}))
            grasp_hint.setdefault("approach_axis", "diagonal_45")
            grasp_hint.setdefault("yaw_rad", 0.0)
            grasp_hint.setdefault("grasp_width_m", 0.05)
            grasp_hint["approach_axis"] = "diagonal_45"  # always override
            grasp_hint["yaw_rad"] = 0.0  # JSON yaw was for z_down — large yaw pushes wrist out of reach for diagonal_45

            plans.append({
                "plan_id": tmpl.get("plan_id", f"plan_{idx+1:04d}"),
                "primitive": "pick_place",
                "object_id": tmpl.get("object_id", prim_name),
                "class_id": part_type,
                "prim_path": path,
                "object_pose_base": {"position_m": b_obj.tolist(), "quaternion_xyzw": obj_quat},
                "bin_pose_base":    {"position_m": bin_pos,         "quaternion_xyzw": bin_quat},
                "grasp_hint": grasp_hint,
                "retry_policy": tmpl.get("retry_policy", "retry_once_slow"),
                "timeout_s": tmpl.get("timeout_s", 20.0),
            })

        print(f"[SCENE_LOAD] ✓ Built {len(plans)} plans from actual USD scene positions")
        return plans

    except Exception as e:
        print(f"[SCENE_LOAD] Error ({e}) — using JSON fallback")
        return template_plans

def load_action_plan(yaml_file_path):
    with open(yaml_file_path, 'r') as f: return yaml.safe_load(f)['action_plan']

def get_base_matrix(position, quaternion) -> np.ndarray:
    return rmu.make_T(np.array(quaternion), np.array(position))

def convert_matrix_to_ik_input(T: np.ndarray) -> list:
    import pinocchio as pin
    R_mat = T[:3, :3]
    p = T[:3, 3]
    return DualArmIK.se3_to_xyzrpy(pin.SE3(R_mat, p)).tolist()

def _make_diagonal_R(tilt_deg: float) -> np.ndarray:
    """Build R = Rx(tilt) @ R_flip: tilt z_down by tilt_deg around +X_world axis.

    tilt_deg=0   → z_down: tool_Z_world=[0,0,-1]           reach≈0.46m
    tilt_deg=45  → tool_Z_world=[0,+0.707,-0.707]           reach≈0.34m
    tilt_deg=60  → tool_Z_world=[0,+0.866,-0.500]           reach≈0.30m

    Tilting around +X_world pulls the wrist in −Y_world (back toward robot),
    because the arm extends primarily in +Y_world. Verified: wrist ends up
    between robot (Y=−0.20) and object (Y=+0.10) for all tilt_deg > 0.
    """
    a = math.radians(tilt_deg)
    ca, sa = math.cos(a), math.sin(a)
    R_flip = np.array([[1., 0.,  0.], [0., -1., 0.], [0., 0., -1.]])
    Rx     = np.array([[1., 0.,  0.], [0.,  ca, -sa], [0., sa,  ca]])
    return Rx @ R_flip


def apply_grasp_rotation(T_pose: np.ndarray, yaw_rad: float, approach_axis: str = "z_down") -> np.ndarray:
    T_rotated = T_pose.copy()
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
    R_yaw = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])
    if approach_axis == "z_down":
        R_final = R_yaw @ _make_diagonal_R(0)
    elif approach_axis == "diagonal_45":
        R_final = R_yaw @ _make_diagonal_R(45)
    elif approach_axis == "diagonal_60":
        R_final = R_yaw @ _make_diagonal_R(60)
    else:
        R_final = R_yaw
    T_rotated[:3, :3] = T_rotated[:3, :3] @ R_final
    return T_rotated

_R_W2B = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])  # world→base: 90° Z rotation

def _matrix_to_base_ik(T: np.ndarray) -> list:
    """Convert 4x4 world-frame SE3 → [x,y,z,roll,pitch,yaw] in robot base frame.

    Rotates the matrix directly (world→base) before Euler extraction to avoid
    gimbal lock from intermediate world-frame Euler decomposition.
    """
    import pinocchio as pin
    x_w, y_w, z_w = T[0, 3], T[1, 3], T[2, 3]
    p_base = np.array([y_w + 0.20, -x_w + 0.70, z_w - 0.9005])
    R_base = _R_W2B @ T[:3, :3]
    return DualArmIK.se3_to_xyzrpy(pin.SE3(R_base, p_base)).tolist()


def execute_stage(robot, world, stage_name, target_pose, gripper_side,
                  gripper_state="open", step_size=0.016, max_steps=2000,
                  pos_tol=0.05, rot_tol=0.15, timeout_sec=60.0, verbose=True,
                  ik_pos_tol=0.005, ik_rot_tol=0.01,
                  ik_rot_weight=1.0, ik_null_weight=0.1, ik_max_iter=150):
    """[FIX #1] Tự check is_reached qua FK thay vì dựa vào return của control_dual_arm_ik.

    Key insight: solver dùng tolerance chặt (5e-3) bên trong để converge sâu,
    còn mình check ở ngoài với tolerance lỏng hơn để break sớm khi đủ tốt.

    ik_rot_weight: IK orientation weight (1.0=full, 0.01≈position-only). Lowering
      this stops the solver wasting iterations on orientation, which is critical
      for transit stages (S3/S4/S7) where exact orientation is irrelevant.
    ik_null_weight: null-space pull toward neutral (0=disabled). Setting to 0 prevents
      the solver from dragging the arm back to the neutral "hip" pose during transit.
    ik_max_iter: IK iterations per call (default 150; use 300 for hard targets).
    """
    left_target = target_pose if gripper_side == "left" else None
    right_target = target_pose if gripper_side == "right" else None

    start_time = time.time()
    steps = 0
    last_pos_err = None
    stuck_counter = 0
    stall_counter = 0   # counts checks where improvement exists but is too slow to matter
    check_interval = 5  # check FK mỗi 5 steps
    # Minimum useful improvement per check (5 steps). Below this the arm is "stalled":
    # making progress in principle but will never reach the goal before timeout.
    MIN_USEFUL_RATE = 2e-3

    while steps < max_steps:
        # Gọi controller — apply_action xảy ra bên trong, ignore return value
        robot.control_dual_arm_ik(
            step_size,
            left_target_xyzrpy=left_target,
            right_target_xyzrpy=right_target,
            pos_tol=ik_pos_tol,
            rot_tol=ik_rot_tol,
            rot_weight=ik_rot_weight,
            null_weight=ik_null_weight,
            max_iter=ik_max_iter,
        )
        world.step(render=True)
        steps += 1
        elapsed = time.time() - start_time

        # Tự check is_reached qua FK
        if steps % check_interval == 0:
            result = _check_reached_via_fk(robot, target_pose, gripper_side, pos_tol, rot_tol)
            if isinstance(result, tuple):
                is_reached, pos_err, rot_err = result
            else:
                is_reached = result
                pos_err, rot_err = None, None

            if is_reached:
                settle_frames = 10 if verbose else 1
                for _ in range(settle_frames): world.step(render=True)
                if verbose:
                    print(f"  ✓ [{stage_name}] reached in {steps} steps "
                          f"(pos_err={pos_err:.4f}m, rot_err={rot_err:.4f}rad)" if pos_err is not None
                          else f"  ✓ [{stage_name}] reached in {steps} steps")
                return True, elapsed, None

            # Detect stuck/stall: exit early rather than waiting for full timeout.
            if pos_err is not None:
                improvement = (last_pos_err - pos_err) if last_pos_err is not None else MIN_USEFUL_RATE
                if last_pos_err is not None and abs(improvement) < 1e-4:
                    stuck_counter += 1   # no movement at all
                else:
                    stuck_counter = 0
                # Stall: improvement exists but below useful rate and still outside tolerance
                if last_pos_err is not None and pos_err > pos_tol and improvement < MIN_USEFUL_RATE:
                    stall_counter += 1
                else:
                    stall_counter = 0
                last_pos_err = pos_err

                # Exit on stuck (300 steps no change) or stall (400 steps slow crawl)
                if stuck_counter > 60 or stall_counter > 80:
                    reason_out = "stuck" if stuck_counter > 60 else "stalled"
                    if verbose:
                        print(f"  ✗ [{stage_name}] {reason_out.upper()} at step {steps} "
                              f"(pos_err={pos_err:.4f}m, rot_err={rot_err:.4f}rad)")
                    return False, elapsed, reason_out

                # Log progress every 50 steps
                if verbose and steps % 50 == 0:
                    print(f"  [{stage_name}] step {steps}: pos_err={pos_err:.4f}m, rot_err={rot_err:.4f}rad")
        
        if elapsed > timeout_sec:
            if verbose: 
                msg = f"  ⚠️ [{stage_name}] TIMEOUT after {steps} steps"
                if last_pos_err is not None:
                    msg += f" (last pos_err={last_pos_err:.4f}m)"
                print(msg)
            return False, elapsed, "timeout"
    
    if verbose: 
        msg = f"  ✗ [{stage_name}] MAX STEPS EXCEEDED"
        if last_pos_err is not None:
            msg += f" (last pos_err={last_pos_err:.4f}m)"
        print(msg)
    return False, elapsed, "max_steps_exceeded"


# Alias: execute_stage_fsm dùng cùng logic
def execute_stage_fsm(robot, world, stage_name, target_pose, gripper_side,
                      step_size=0.016, max_steps=200, pos_tol=0.04, rot_tol=0.2,
                      timeout_sec=60.0, verbose=True, ik_rot_tol=0.01,
                      ik_rot_weight=1.0, ik_null_weight=0.1, ik_max_iter=150):
    """Wrapper for FSM state machine - same logic as execute_stage."""
    return execute_stage(robot, world, stage_name, target_pose, gripper_side,
                         step_size=step_size, max_steps=max_steps,
                         pos_tol=pos_tol, rot_tol=rot_tol,
                         timeout_sec=timeout_sec, verbose=verbose,
                         ik_rot_tol=ik_rot_tol,
                         ik_rot_weight=ik_rot_weight,
                         ik_null_weight=ik_null_weight,
                         ik_max_iter=ik_max_iter)


def _slerp_R(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between two rotation matrices using pinocchio log3/exp3."""
    import pinocchio as pin
    R_rel = R0.T @ R1
    log_R = pin.log3(R_rel)
    return R0 @ pin.exp3(t * log_R)


def move_interpolated(robot, world, T_start, T_end, side, stage_name,
                      num_steps=20, max_sim_steps=100,
                      pos_tol=0.08, rot_tol=0.20,
                      step_size=0.016,
                      writer=None, object_id="",
                      interp_rotation=False,
                      ik_rot_tol=0.01,
                      ik_rot_weight=1.0, ik_null_weight=0.1, ik_max_iter=150) -> tuple:
    """Interpolate from T_start to T_end in Cartesian space.

    interp_rotation=True: SLERP rotation from T_start to T_end (for orientation transitions).
    interp_rotation=False: use T_end rotation for all waypoints (original behaviour).
    """
    print(f"  [{stage_name}] {num_steps} pts | tol=({pos_tol},{rot_tol}) | step={step_size} | max_sim={max_sim_steps} | interp_rot={interp_rotation}")
    for i in range(1, num_steps + 1):
        t = i / float(num_steps)
        T_curr = T_start.copy()
        T_curr[:3, 3] = T_start[:3, 3] + t * (T_end[:3, 3] - T_start[:3, 3])
        if interp_rotation:
            T_curr[:3, :3] = _slerp_R(T_start[:3, :3], T_end[:3, :3], t)
        else:
            T_curr[:3, :3] = T_end[:3, :3]

        ik_input_base = _matrix_to_base_ik(T_curr)

        is_success, _, _ = execute_stage_fsm(
            robot, world, stage_name=f"{stage_name}_pt{i}", target_pose=ik_input_base,
            gripper_side=side, step_size=step_size,
            max_steps=max_sim_steps, pos_tol=pos_tol, rot_tol=rot_tol, verbose=False,
            ik_rot_tol=ik_rot_tol,
            ik_rot_weight=ik_rot_weight, ik_null_weight=ik_null_weight, ik_max_iter=ik_max_iter,
        )
        if not is_success:
            print(f"  ✗ [{stage_name}] stuck at pt {i}/{num_steps}")
            return False, "interpolation_stuck"

    if writer: writer.writerow([object_id, stage_name, "active"] + ik_input_base)
    return True, None


# ======================================================================
# 4. Industrial-optimized FSM: L-shape trajectory & grip force verification
# ======================================================================

class PickAndPlaceStateMachine:
    def __init__(self, robot, world, plan, motion_specs, coord_transform=None, writer=None,
                 remaining_plans=None):
        self.robot, self.world = robot, world
        self.plan, self.specs = plan, motion_specs
        self.coord_transform, self.writer = coord_transform, writer
        self.remaining_plans = remaining_plans or []
        
        self.state = "INIT"
        self.start_time = time.time()
        self.metrics = {}
        self.failure_reason = None
        self.object_id = plan['object_id']
        self.grasp_width = plan.get('grasp_hint', {}).get('grasp_width_m', 0.05)
        self._grasp_joint = None
        self.object_prim_path = plan.get('prim_path', f"/Replicator/{self.object_id}")

    def run(self) -> PrimitiveResult:
        print(f"\n================ FSM: PICK & PLACE [{self.object_id}] ================")
        while self.state not in ["DONE", "FAIL"]:
            if   self.state == "INIT":         self._state_init()
            elif self.state == "S1_PREGRASP":  self._state_s1_pregrasp()
            elif self.state == "S2_GRASP":     self._state_s2_grasp()
            elif self.state == "S3_LIFT":      self._state_s3_lift()
            elif self.state == "S4_TRANSFER":  self._state_s4_transfer()
            elif self.state == "S5_LOWER_BIN": self._state_s5_lower_bin()
            elif self.state == "S6_RELEASE":   self._state_s6_release()
            elif self.state == "S7_RETREAT":   self._state_s7_retreat()
            elif self.state == "VERIFY":       self._state_verify()
                
        return PrimitiveResult(
            primitive_name="pick_place", success=(self.state == "DONE"),
            elapsed_s=time.time() - self.start_time, retry_count=0,
            failure_reason=self.failure_reason, metrics=self.metrics
        )

    def _fail(self, reason: str):
        self.failure_reason = reason
        self.state = "FAIL"
        print(f"  [FSM ERROR] Failed: {reason}")

    def _state_init(self):
        print("[FSM] State: INIT")
        try:
            grasp_hint = self.plan.get('grasp_hint', {})
            self.yaw_rad = float(grasp_hint.get('yaw_rad', 0.0))
            self.approach_axis = grasp_hint.get('approach_axis', 'z_down')
            print(f"[FSM] Grasp hint: yaw_rad={self.yaw_rad:.4f}, approach_axis={self.approach_axis}")

            raw_obj = np.array(self.plan['object_pose_base']['position_m'])
            raw_bin = np.array(self.plan['bin_pose_base']['position_m'])
            
            if self.coord_transform is not None:
                world_obj = self.coord_transform.robot_to_world(raw_obj)
                world_bin = self.coord_transform.robot_to_world(raw_bin)
            else:
                world_obj = np.array([-raw_obj[1] + 0.70, raw_obj[0] - 0.20, 1.0400])
                world_bin = np.array([-raw_bin[1] + 0.70, raw_bin[0] - 0.20, 1.0400])
            
            # Sanity check transform roundtrip
            back_to_base = np.array([world_obj[1] + 0.20, -world_obj[0] + 0.70, world_obj[2] - 0.9005])
            roundtrip_err = float(np.linalg.norm(raw_obj[:2] - back_to_base[:2]))
            print(f"[SANITY] raw_obj (base):     {np.round(raw_obj, 4).tolist()}")
            print(f"[SANITY] world_obj:          {np.round(world_obj, 4).tolist()}")
            print(f"[SANITY] back_to_base:       {np.round(back_to_base, 4).tolist()}")
            print(f"[SANITY] roundtrip error XY: {roundtrip_err*100:.2f}cm")
            print(f"[SANITY] ⚠️ Hãy so sánh world_obj với [DEBUG_SCENE] output bên trên để xác nhận tọa độ khớp!")
                
            # Clamp Z to valid table range instead of hardcoding 1.0400.
            # Objects rest on table at ~1.04m world Z; allow ±5cm tolerance.
            _TABLE_Z_MIN, _TABLE_Z_MAX = 0.98, 1.15
            if not (_TABLE_Z_MIN <= world_obj[2] <= _TABLE_Z_MAX):
                print(f"[FSM] ⚠ world_obj Z={world_obj[2]:.4f} outside [{_TABLE_Z_MIN},{_TABLE_Z_MAX}] "
                      f"— clamping to table range")
                world_obj[2] = max(_TABLE_Z_MIN, min(_TABLE_Z_MAX, world_obj[2]))

            # Clamp world_obj into workspace bounds before validation.
            # Scene-derived positions are physically valid but may fall slightly outside
            # configured bounds (e.g. Z_min set for old hardcoded 1.04m).
            # Clamping avoids false-positive failures while keeping downstream math sane.
            if ENABLE_WORKSPACE_VALIDATION:
                _wb = WORKSPACE_BOUNDS
                _pre_clamp = world_obj.copy()
                world_obj[0] = max(_wb["x"][0], min(_wb["x"][1], world_obj[0]))
                world_obj[1] = max(_wb["y"][0], min(_wb["y"][1], world_obj[1]))
                world_obj[2] = max(_wb["z"][0], min(_wb["z"][1], world_obj[2]))
                _delta = np.linalg.norm(world_obj - _pre_clamp)
                if _delta > 0.001:
                    print(f"[FSM] Workspace clamp: {np.round(_pre_clamp,4).tolist()} "
                          f"→ {np.round(world_obj,4).tolist()} (Δ={_delta*100:.1f}cm)")
            obj_valid, msg = validate_position_in_workspace(world_obj.tolist())
            if not obj_valid and ENABLE_WORKSPACE_VALIDATION:
                print(f"[FSM] ⚠ Workspace validation still failed after clamp: {msg} — proceeding anyway")

            # Arm selection: both bins are at base_y ≈ -0.5 (right side of robot).
            # Only the right arm shoulder (base ≈ (0, -0.3)) can reach the bin
            # (shoulder→bin ≈ 0.60m).  Left arm shoulder (base ≈ (0, +0.3)) would
            # need to cross 0.94m — beyond reach.
            # Objects in scatter area (base_y ≈ [-0.10, +0.20]) are all reachable by
            # the right arm.  Switch to left only if object is very far left (>0.35).
            if raw_obj[1] > 0.35:
                self.side = "left"
            else:
                self.side = "right"
            
            tcp_offset = np.array([0.0, 0.0, 0.0], dtype=float)
            config_paths = [
                '/home/ubuntu/vinh/configs/Part_Sorting.yaml',
                '/home/ubuntu/thu/configs/Part_Sorting.yaml',
                'configs/Part_Sorting.yaml',
                '../configs/Part_Sorting.yaml',
                '../../configs/Part_Sorting.yaml',
            ]
            for cfg_path in config_paths:
                if Path(cfg_path).exists():
                    try:
                        with open(cfg_path, 'r') as f:
                            cfg = yaml.safe_load(f)
                            if cfg and 'grasp' in cfg and 'tcp_offset' in cfg['grasp']:
                                tcp_offset = np.array(cfg['grasp']['tcp_offset'], dtype=float)
                                print(f"[FSM] Loaded TCP offset from {cfg_path}: {tcp_offset.tolist()}")
                                break
                    except Exception as e:
                        print(f"[FSM] Warning: failed to load {cfg_path}: {e}")
                        continue
            if np.allclose(tcp_offset, [0, 0, 0]):
                print(f"[FSM] Warning: TCP offset is zero (default) - gripper may not touch target!")

            # tcp_z: distance from sixforce_link (EE) to fingertip along tool-Z.
            tcp_z = float(tcp_offset[2]) if tcp_offset[2] > 0.01 else 0.22
            print(f"[FSM] tcp_z={tcp_z:.3f}m")

            Z_DOWN_R = _make_diagonal_R(0)  # z_down: tool-Z=[0,0,-1]; kept for bin-place targets
            app_offset = self.specs.get("approach_offset_m", 0.08)
            Z_OFFSET = self.specs.get("grasp_z_offset_m", 0.0)
            SAFE_FLY_HEIGHT = 1.40

            # Choose grasp orientation based on wrist XY reach to object.
            # z_down reach limit ≈ 0.46m. Objects in the scatter area (world_y up to 0.30m)
            # land at base_x = world_y + 0.20 ≈ 0.50m — past the limit — causing the IK to
            # find an elbow-up solution (joints 1&2 at limit, joint 3 past 90°) that cannot
            # descend to grasp. diagonal_45 tilts tool-Z to [0,+0.707,-0.707], moving the
            # wrist 0.707*tcp_z ≈ 0.156m closer to the robot body, cutting XY reach to ~0.34m.
            _obj_base_x = world_obj[1] + 0.20   # base_x = world_y + 0.20
            _obj_base_y = -world_obj[0] + 0.70  # base_y = -world_x + 0.70
            _obj_xy_reach = math.sqrt(_obj_base_x**2 + _obj_base_y**2)
            # Always z_down: gripper points straight down (wrist_pitch bent 90°).
            # S1 uses an overhead approach (fly directly above → descend) so the
            # arm never reaches the XY joint limit before the wrist is pointing down.
            tilt_deg = 0
            print(f"[FSM] Object XY reach={_obj_xy_reach:.3f}m → z_down approach (gripper pointing straight down)")
            APPROACH_R = _make_diagonal_R(tilt_deg)
            tool_Z_dir = APPROACH_R[:, 2]  # world frame: [0,0,-1] for z_down, [0,+0.707,-0.707] for 45°

            # GRASP wrist: fingertip = wrist + tcp_z*tool_Z_dir  →  wrist = obj - tcp_z*tool_Z_dir
            self.T_grasp = np.eye(4)
            self.T_grasp[:3, :3] = APPROACH_R
            self.T_grasp[0, 3] = world_obj[0] - tcp_z * tool_Z_dir[0]
            self.T_grasp[1, 3] = world_obj[1] - tcp_z * tool_Z_dir[1]
            self.T_grasp[2, 3] = world_obj[2] - tcp_z * tool_Z_dir[2] + Z_OFFSET

            # PRE-GRASP: retract app_offset along approach axis (opposite to tool_Z_dir)
            self.T_pre_grasp = self.T_grasp.copy()
            self.T_pre_grasp[0, 3] -= app_offset * tool_Z_dir[0]
            self.T_pre_grasp[1, 3] -= app_offset * tool_Z_dir[1]
            self.T_pre_grasp[2, 3] -= app_offset * tool_Z_dir[2]

            # HIGH APPROACH: safe fly height above grasp XY
            self.T_high_approach = self.T_grasp.copy()
            self.T_high_approach[2, 3] = SAFE_FLY_HEIGHT

            # BIN PLACE: z_down, wrist above bin
            # Clamp bin XY only when truly out of reach.
            # Right arm shoulder ≈ (0, -0.3) → bin at (0.5, -0.5) is ~0.60m away.
            # Arm length ≈ 0.70m, so 0.72m XY from base origin is achievable.
            _MAX_BIN_XY_BASE = 0.72  # max arm XY reach from base origin (m)
            _bin_base_x = world_bin[1] + 0.20   # base x = world_y + 0.20
            _bin_base_y = -world_bin[0] + 0.70  # base y = -world_x + 0.70
            _bin_xy = math.sqrt(_bin_base_x**2 + _bin_base_y**2)
            if _bin_xy > _MAX_BIN_XY_BASE:
                _scale = _MAX_BIN_XY_BASE / _bin_xy
                _bin_base_x_c = _bin_base_x * _scale
                _bin_base_y_c = _bin_base_y * _scale
                # Convert clamped base coords back to world
                world_bin[0] = -_bin_base_y_c + 0.70   # world_x = -base_y + 0.70
                world_bin[1] = _bin_base_x_c - 0.20    # world_y = base_x - 0.20
                print(f"[FSM] ⚠ Bin XY reach {_bin_xy:.3f}m > {_MAX_BIN_XY_BASE}m limit — "
                      f"clamped base:({_bin_base_x:.3f},{_bin_base_y:.3f}) → "
                      f"({_bin_base_x_c:.3f},{_bin_base_y_c:.3f})")
            else:
                print(f"[FSM] Bin XY reach {_bin_xy:.3f}m (OK)")

            # Clamp bin Z to bin height range (bin center ~1.05m, accept 0.95–1.20m)
            _BIN_Z_MIN, _BIN_Z_MAX = 0.95, 1.20
            if not (_BIN_Z_MIN <= world_bin[2] <= _BIN_Z_MAX):
                world_bin[2] = max(_BIN_Z_MIN, min(_BIN_Z_MAX, world_bin[2]))
                print(f"[FSM] ⚠ Bin Z clamped to {world_bin[2]:.4f}m")
            print(f"[FSM] Bin world Z={world_bin[2]:.4f}m  place fingertip_z={world_bin[2]+Z_OFFSET:.4f}m")

            self.T_place = np.eye(4)
            self.T_place[:3, :3] = Z_DOWN_R
            self.T_place[0, 3] = world_bin[0]
            self.T_place[1, 3] = world_bin[1]
            self.T_place[2, 3] = world_bin[2] + tcp_z + Z_OFFSET

            # PRE-PLACE: directly above place, same XY
            self.T_pre_place = self.T_place.copy()
            self.T_pre_place[2, 3] += app_offset

            # HIGH PLACE: safe fly height
            self.T_high_place = self.T_place.copy()
            self.T_high_place[2, 3] = SAFE_FLY_HEIGHT

            # Debug + reach check
            ik_grasp_debug = _matrix_to_base_ik(self.T_grasp)
            ik_pre_debug   = _matrix_to_base_ik(self.T_pre_grasp)
            ik_place_debug = _matrix_to_base_ik(self.T_high_place)
            x_b, y_b, z_b = ik_pre_debug[0], ik_pre_debug[1], ik_pre_debug[2]
            reach_dist = np.sqrt(x_b**2 + y_b**2 + z_b**2)
            print(f"[FSM] T_grasp wrist world=({self.T_grasp[0,3]:.4f},{self.T_grasp[1,3]:.4f},{self.T_grasp[2,3]:.4f})  fingertip≈obj_z={world_obj[2]+Z_OFFSET:.4f}m")
            print(f"[FSM] T_grasp base: x={ik_grasp_debug[0]:.3f}, y={ik_grasp_debug[1]:.3f}, z={ik_grasp_debug[2]:.3f}")
            print(f"[FSM] Pre-grasp base coords: x={x_b:.3f}, y={y_b:.3f}, z={z_b:.3f}")
            print(f"[FSM] Reach distance from base: {reach_dist:.3f}m")
            print(f"[FSM] Bin high-place base: x={ik_place_debug[0]:.3f}, y={ik_place_debug[1]:.3f}, z={ik_place_debug[2]:.3f}")

            if self.robot: self.robot.open_gripper(side=self.side)

            # Spatial conflict warning
            conflicts = check_spatial_conflicts(
                self.plan, self.remaining_plans, safety_margin_m=0.05)
            if conflicts:
                for oid, d in conflicts:
                    print(f"  ⚠️ [SPATIAL] {oid} nằm cách {d*100:.1f}cm — có thể bị va chạm khi gắp!")

            self.state = "S1_PREGRASP"
        except Exception as e: return self._fail(f"init_error: {e}")

    def _state_s1_pregrasp(self):
        print("[FSM] State: S1_PREGRASP -> overhead approach (fly above object, then descend straight down)")
        # Overhead approach for z_down grasping:
        #   Phase 1 → fly to T_high_approach (same XY as grasp, SAFE_FLY_HEIGHT)
        #             At this height the arm is NOT at joint limits, so the IK CAN
        #             achieve z_down (wrist_pitch bent down) without singularity.
        #   Phase 2 → descend straight down to T_pre_grasp (8 cm above grasp)
        #             Pure Z descent while holding z_down → elbow bends more but
        #             never hits limits.
        # This replaces the old 3-phase column+sweep which left the arm at full XY
        # extension at table height, leaving no joint DOF to keep the wrist down.

        # Pre-bend wrist to -π/2 (z_down) before running any IK.
        # Without this, IK warm-start begins from wrist_pitch≈0 (horizontal) and
        # converges to a local minimum where position is correct but wrist stays
        # horizontal (gripper pointing sideways) instead of pointing straight down.
        # Teleporting the physical joint seeds the warm-start so IK stays in z_down.
        _prefix = "R" if self.side == "right" else "L"
        _wp_name = f"{_prefix}_wrist_pitch_joint"
        if self.robot and self.robot._articulation:
            _dof_names = self.robot._articulation.dof_names
            if _wp_name in _dof_names:
                try:
                    import torch as _torch
                    _wp_idx = self.robot._articulation.get_dof_index(_wp_name)
                    self.robot._articulation.set_joint_positions(
                        _torch.tensor([[-math.pi / 2]], dtype=_torch.float32),
                        joint_indices=_torch.tensor([_wp_idx], dtype=_torch.int32)
                    )
                    for _ in range(10):
                        self.world.step(render=True)
                    print(f"  [S1] Wrist {_wp_name} pre-bent to z_down ({-math.pi/2:.3f} rad)")
                except Exception as _e:
                    print(f"  [S1] Wrist pre-bend failed ({_e}) — IK may not achieve z_down")

        # Read current EE world-frame pose as 4×4 matrix (interpolation start point).
        T_current = self.T_high_approach.copy()
        T_current[2, 3] += 0.20   # safe fallback: slightly above T_high_approach
        if self.robot and self.robot.ik_solver:
            try:
                joints = self.robot.get_joint_states()
                positions = joints['positions']
                if positions and isinstance(positions[0], list):
                    positions = positions[0]
                self.robot.ik_solver.sync_joint_positions(joints['names'], positions)
                ee_se3 = self.robot.ik_solver.get_ee_pose(self.side)
                p_b = np.array(ee_se3.translation)
                R_b = np.array(ee_se3.rotation)
                p_w = np.array([-p_b[1] + 0.70, p_b[0] - 0.20, p_b[2] + 0.9005])
                T_current = np.eye(4)
                T_current[:3, :3] = _R_W2B.T @ R_b
                T_current[:3, 3]  = p_w
            except Exception as e:
                print(f"  [S1] FK read failed ({e}) — using T_high_approach+0.2 as start")

        ik_high = _matrix_to_base_ik(self.T_high_approach)
        ik_pre  = _matrix_to_base_ik(self.T_pre_grasp)
        xh, yh, zh = ik_high[0], ik_high[1], ik_high[2]
        xb, yb, zb = ik_pre[0], ik_pre[1], ik_pre[2]
        print(f"  high_approach base: x={xh:.3f}, y={yh:.3f}, z={zh:.3f}")
        print(f"  pre_grasp base:     x={xb:.3f}, y={yb:.3f}, z={zb:.3f}")

        # Phase 1: Fly directly above object with z_down orientation.
        # interp_rotation=True slerps from current arm rotation to z_down, guiding the
        # IK to achieve wrist-down before the arm is at full forward extension.
        success1, err1 = move_interpolated(
            self.robot, self.world, T_current, self.T_high_approach,
            self.side, "s1_fly_above",
            num_steps=20, max_sim_steps=300, pos_tol=0.06, rot_tol=0.35,
            step_size=0.018, interp_rotation=True,
            ik_rot_weight=0.8, ik_null_weight=0.0, ik_max_iter=250,
        )

        if not success1:
            alt_side = "left" if self.side == "right" else "right"
            print(f"  [S1/P1] Primary arm '{self.side}' failed ({err1}). Trying '{alt_side}'...")
            if self.robot:
                self.robot.open_gripper(side=alt_side)
            # Pre-bend alt_side wrist to z_down as well
            _alt_wp_name = f"{'R' if alt_side == 'right' else 'L'}_wrist_pitch_joint"
            if self.robot and self.robot._articulation and _alt_wp_name in self.robot._articulation.dof_names:
                try:
                    import torch as _torch
                    _alt_wp_idx = self.robot._articulation.get_dof_index(_alt_wp_name)
                    self.robot._articulation.set_joint_positions(
                        _torch.tensor([[-math.pi / 2]], dtype=_torch.float32),
                        joint_indices=_torch.tensor([_alt_wp_idx], dtype=_torch.int32)
                    )
                    for _ in range(5):
                        self.world.step(render=True)
                except Exception:
                    pass
            success_alt, err_alt = move_interpolated(
                self.robot, self.world, T_current, self.T_high_approach,
                alt_side, "s1_fly_above_alt",
                num_steps=20, max_sim_steps=300, pos_tol=0.06, rot_tol=0.35,
                step_size=0.018, interp_rotation=True,
                ik_rot_weight=0.8, ik_null_weight=0.0, ik_max_iter=250,
            )
            if success_alt:
                self.side = alt_side
                print(f"  [S1/P1] ✓ Fallback to '{alt_side}' succeeded")
            else:
                # Last resort: direct IK (no interpolation, loose tolerances)
                ok_d, _, r_d = execute_stage(
                    self.robot, self.world, "s1_direct_high", ik_high, self.side,
                    step_size=0.025, max_steps=3000,
                    pos_tol=0.08, rot_tol=0.50, timeout_sec=90.0,
                    ik_rot_weight=0.5, ik_null_weight=0.0, ik_max_iter=300,
                )
                if not ok_d:
                    return self._fail(f"s1_fly_above_fail: {err1} / {err_alt} / {r_d}")

        # Phase 2: Descend straight down to pre-grasp height (pure Z drop, z_down held).
        print(f"  [Phase2] Descend: Z {self.T_high_approach[2,3]:.3f}m → {self.T_pre_grasp[2,3]:.3f}m")
        success2, err2 = move_interpolated(
            self.robot, self.world, self.T_high_approach, self.T_pre_grasp,
            self.side, "s1_descend",
            num_steps=15, max_sim_steps=250, pos_tol=0.04, rot_tol=0.30,
            step_size=0.015, interp_rotation=False,
            ik_rot_weight=0.9, ik_null_weight=0.0, ik_max_iter=250,
        )
        if not success2:
            print(f"  [S1/P2] Descent failed ({err2}) — direct IK fallback to pre-grasp...")
            ok2, _, r2 = execute_stage(
                self.robot, self.world, "s1_direct_pre", ik_pre, self.side,
                step_size=0.015, max_steps=2000,
                pos_tol=0.06, rot_tol=0.40, timeout_sec=45.0,
                ik_null_weight=0.0, ik_rot_weight=0.8,
            )
            if not ok2:
                return self._fail(f"s1_pregrasp_fail: {r2}")

        try:
            joints = self.robot.get_joint_states()
            positions = joints['positions']
            if positions and isinstance(positions[0], list):
                positions = positions[0]
            self.robot.ik_solver.sync_joint_positions(joints['names'], positions)
            actual_se3 = self.robot.ik_solver.get_ee_pose(self.side)
            actual_z_ax = actual_se3.rotation[:, 2]
            print(f"  [S1 POST] EE Z-axis in base: {actual_z_ax.round(3).tolist()} (expect ≈[0,0,-1])")
        except Exception as e:
            print(f"  [S1 POST] Could not read EE state: {e}")
        self.state = "S2_GRASP"

    def _state_s2_grasp(self):
        print("[FSM] State: S2_GRASP -> sweep to above-grasp, then pure vertical descent")
        # With diagonal_45, T_pre_grasp is BEHIND the object (wrist Y < object Y).
        # Descending directly along tool_Z_dir=[0,+0.707,-0.707] sweeps the forearm
        # forward-and-down over the table surface, causing the wrist body to collide
        # with the table before the fingertip reaches the object.
        #
        # Fix: split into two phases to eliminate the diagonal forward sweep:
        #   Phase 1 — horizontal sweep from T_pre_grasp to T_above_grasp
        #             (same XY as T_grasp, same Z as T_pre_grasp — arm stays high)
        #   Phase 2 — pure vertical descent from T_above_grasp to T_grasp
        #             (no forward movement; arm lowers straight down, no table sweep)
        # For z_down, tool_Z_dir=[0,0,-1] → T_above_grasp == T_pre_grasp → Phase 1 is a no-op.
        T_above_grasp = self.T_grasp.copy()
        T_above_grasp[2, 3] = self.T_pre_grasp[2, 3]  # grasp XY, pre_grasp Z

        success1, _err1 = move_interpolated(
            self.robot, self.world, self.T_pre_grasp, T_above_grasp,
            self.side, "s2_sweep_above",
            num_steps=8, max_sim_steps=200, pos_tol=0.025, rot_tol=0.40,
            step_size=0.010, interp_rotation=False,
            ik_rot_weight=0.8, ik_null_weight=0.0, ik_max_iter=200)
        _start_p2 = T_above_grasp if success1 else self.T_pre_grasp

        success, err = move_interpolated(
            self.robot, self.world, _start_p2, self.T_grasp,
            self.side, "s2_descend",
            num_steps=15, max_sim_steps=350, pos_tol=0.010, rot_tol=0.40,
            step_size=0.010, interp_rotation=False,
            ik_rot_weight=0.8, ik_null_weight=0.0, ik_max_iter=250)
        if not success: return self._fail("collision_on_grasp")

        # Settle + verify wrist Z before closing
        if self.world:
            for _ in range(15): self.world.step(render=True)
        try:
            joints = self.robot.get_joint_states()
            positions = joints['positions']
            if positions and isinstance(positions[0], list):
                positions = positions[0]
            self.robot.ik_solver.sync_joint_positions(joints['names'], positions)
            actual_se3 = self.robot.ik_solver.get_ee_pose(self.side)
            actual_z_base = float(actual_se3.translation[2])
            target_z_base = _matrix_to_base_ik(self.T_grasp)[2]
            z_err = actual_z_base - target_z_base
            print(f"  [S2 CHECK] wrist z_base: actual={actual_z_base:.3f} target={target_z_base:.3f} err={z_err:+.3f}m")
            if z_err > 0.010:  # arm still >1cm above target — push down one more time
                print(f"  [S2 CHECK] Arm {z_err*100:.1f}cm above grasp — correction step...")
                ik_grasp = _matrix_to_base_ik(self.T_grasp)
                execute_stage(
                    self.robot, self.world, "s2_correct",
                    ik_grasp, self.side,
                    step_size=0.004, max_steps=800,
                    pos_tol=0.008, rot_tol=0.5, timeout_sec=25.0,
                    ik_null_weight=0.0)
        except Exception as e:
            print(f"  [S2 CHECK] Cannot verify: {e}")

        print("  -> Close gripper...")
        if self.robot:
            self.robot.close_gripper(side=self.side)
        if self.world:
            for _ in range(80): self.world.step(render=True)

        # Attach object via USD FixedJoint for reliable simulation grasping
        # Pass robot object (not prim_path) so cached gripper link is used
        if self.robot and self.world:
            self._grasp_joint = _create_grasp_joint(
                self.world, self.robot, self.object_prim_path, self.side)
            # FixedJoint addition triggers Isaac Sim physics rebuild which clears
            # _physics_view on all Articulation instances. Wait more frames before
            # reinit so the physics scene has time to stabilize first.
            if self._grasp_joint and hasattr(self.robot, '_reinitialize_physics'):
                for _ in range(30): self.world.step(render=True)
                ok = self.robot._reinitialize_physics()
                if ok:
                    # Re-sync IK solver after reinit so arm doesn't jump
                    joints = self.robot.get_joint_states()
                    if joints and self.robot.ik_solver:
                        positions = joints['positions']
                        if positions and isinstance(positions[0], list):
                            positions = positions[0]
                        self.robot.ik_solver.sync_joint_positions(joints['names'], positions)
                # Reset IK fail counters so S3 doesn't inherit a "continuously failing" state
                if self.robot.ik_solver:
                    if hasattr(self.robot.ik_solver, '_right_fail_count'):
                        self.robot.ik_solver._right_fail_count = 0
                    if hasattr(self.robot.ik_solver, '_left_fail_count'):
                        self.robot.ik_solver._left_fail_count = 0

        if self._grasp_joint:
            print("  [S2] ✓ Grasp confirmed via joint attachment")
        else:
            # Fallback: position-based verification
            if not verify_grasp_success(self.robot, self.side):
                return self._fail("grasp_fail_empty_gripper")

        self.state = "S3_LIFT"

    def _state_s3_lift(self):
        print("[FSM] State: S3_LIFT [MoveJ] -> raise to SAFE_FLY_HEIGHT (joint space)")
        # Sync IK warm-start to post-grasp joint state (physics may have reinitialized in S2)
        if self.robot and self.robot.ik_solver:
            try:
                joints = self.robot.get_joint_states()
                if joints:
                    positions = joints['positions']
                    if positions and isinstance(positions[0], list):
                        positions = positions[0]
                    self.robot.ik_solver.sync_joint_positions(joints['names'], positions)
                # Reset fail counters: S2 FixedJoint creation may leave counter elevated
                if hasattr(self.robot.ik_solver, '_right_fail_count'):
                    self.robot.ik_solver._right_fail_count = 0
                if hasattr(self.robot.ik_solver, '_left_fail_count'):
                    self.robot.ik_solver._left_fail_count = 0
            except Exception as _e:
                print(f"  [S3] IK sync warning: {_e}")
        ik_high = _matrix_to_base_ik(self.T_high_approach)
        # rot_weight=0.3: prioritize position (lift height) over exact orientation.
        # null_weight=0.0: disable pull toward neutral — stops arm from going to "hip" mid-lift.
        success, _, reason = execute_stage(
            self.robot, self.world, "s3_lift", ik_high, self.side,
            step_size=0.025, max_steps=4000,
            pos_tol=0.08, rot_tol=0.50,
            timeout_sec=60.0,
            ik_rot_tol=0.15, ik_rot_weight=0.3, ik_null_weight=0.0, ik_max_iter=300)
        if not success:
            print(f"  [S3] ⚠ Lift incomplete ({reason}) — proceeding anyway")
        self.state = "S4_TRANSFER"

    def _state_s4_transfer(self):
        print("[FSM] State: S4_TRANSFER [MoveJ] -> fly to bin (joint space)")

        # Sync IK warm-start to post-lift joint state and reset fail counters.
        if self.robot and self.robot.ik_solver:
            try:
                joints = self.robot.get_joint_states()
                if joints:
                    positions = joints['positions']
                    if positions and isinstance(positions[0], list):
                        positions = positions[0]
                    self.robot.ik_solver.sync_joint_positions(joints['names'], positions)
            except Exception as _e:
                print(f"  [S4] IK sync warning: {_e}")
            if hasattr(self.robot.ik_solver, '_right_fail_count'):
                self.robot.ik_solver._right_fail_count = 0
            if hasattr(self.robot.ik_solver, '_left_fail_count'):
                self.robot.ik_solver._left_fail_count = 0

        # Build transfer target: bin XY position + CURRENT arm orientation.
        # z_down orientation at the bin position is often kinematically infeasible
        # (rot_err stuck at π/2 for PartB bins), causing IK warm-start resets and
        # S4 timeout. The object is held by FixedJoint so orientation during transit
        # doesn't affect the grasp. S5 will re-approach with z_down.
        ik_high_place = _matrix_to_base_ik(self.T_high_place)  # [x,y,z, roll,pitch,yaw] — z_down orientation
        print(f"  [S4] Targeting z_down orientation at bin high-place")

        px, py, pz = ik_high_place[0], ik_high_place[1], ik_high_place[2]
        print(f"  bin high-place target base: x={px:.3f}, y={py:.3f}, z={pz:.3f} "
              f"| XY-reach={math.sqrt(px**2+py**2):.3f}m")

        # pos_tol=0.08m: arm must be within 8cm of bin before S5 descends.
        # ik_rot_weight=0.01: essentially position-only IK — orientation completely relaxed.
        # ik_null_weight=0.0: disable null-space pull toward neutral (stops "arm goes to hip" behavior).
        # ik_max_iter=300: more iterations per call to find solutions for hard configurations.
        success, _, reason = execute_stage(
            self.robot, self.world, "s4_transfer", ik_high_place, self.side,
            step_size=0.030, max_steps=4000,
            pos_tol=0.08, rot_tol=0.80,
            timeout_sec=60.0,
            ik_rot_tol=0.15, ik_rot_weight=0.3, ik_null_weight=0.0, ik_max_iter=300)
        if not success:
            print(f"  [S4] ⚠ Transfer incomplete ({reason}) — proceeding best-effort to S5")
        self.state = "S5_LOWER_BIN"

    def _state_s5_lower_bin(self):
        print("[FSM] State: S5_LOWER_BIN [z_down] -> lower straight to T_place")

        # Get actual EE pose (S4 may leave arm slightly off T_high_place)
        T_start_s5 = self.T_high_place.copy()  # fallback
        if self.robot and self.robot.ik_solver:
            try:
                joints = self.robot.get_joint_states()
                if joints:
                    positions = joints['positions']
                    if positions and isinstance(positions[0], list):
                        positions = positions[0]
                    self.robot.ik_solver.sync_joint_positions(joints['names'], positions)
                ee_se3 = self.robot.ik_solver.get_ee_pose(self.side)
                p_base = ee_se3.translation
                R_base = ee_se3.rotation
                p_world = np.array([-p_base[1] + 0.70, p_base[0] - 0.20, p_base[2] + 0.9005])
                R_world = _R_W2B.T @ R_base
                T_start_s5 = np.eye(4)
                T_start_s5[:3, :3] = R_world
                T_start_s5[:3, 3] = p_world
                z_start = T_start_s5[2, 3]
                z_target = self.T_pre_place[2, 3]
                print(f"  [S5] Actual EE world Z={z_start:.3f}m → pre_place Z={z_target:.3f}m")
            except Exception as _e:
                print(f"  [S5] Cannot get current EE ({_e}) — using T_high_place as start")

        # Phase 1: Descent to pre-place height with z_down orientation.
        # S4 now targets z_down at T_high_place, so T_start_s5 should already be near
        # z_down. Use interp_rotation=True to smoothly complete any remaining transition.
        success, err = move_interpolated(
            self.robot, self.world, T_start_s5, self.T_pre_place,
            self.side, "s5_lower_pre",
            num_steps=10, max_sim_steps=150, pos_tol=0.06, rot_tol=0.50,
            interp_rotation=True, ik_rot_tol=0.15,
            ik_rot_weight=0.8, ik_null_weight=0.0, ik_max_iter=250)
        if not success:
            print("  [S5] Pre-lower stuck — releasing above bin (best-effort)")
            self.state = "S6_RELEASE"
            return

        # Phase 2: Final descent to T_place maintaining z_down
        success, err = move_interpolated(
            self.robot, self.world, self.T_pre_place, self.T_place,
            self.side, "s5_lower_final",
            num_steps=8, max_sim_steps=150, pos_tol=0.03, rot_tol=0.40,
            interp_rotation=False, ik_rot_tol=0.15,
            ik_rot_weight=1.0, ik_null_weight=0.0, ik_max_iter=250)
        if not success:
            print("  [S5] Final lower stuck — releasing at current position (best-effort)")
        self.state = "S6_RELEASE"

    def _state_s6_release(self):
        print("[FSM] State: S6_RELEASE -> open gripper, release object")
        if self._grasp_joint:
            _remove_grasp_joint(self.world, self._grasp_joint)
            self._grasp_joint = None
            # Joint removal triggers physics rebuild; 3 frames is enough for the scene
            # graph to stabilize before reinit — no need for a long settle here.
            if self.world:
                for _ in range(3): self.world.step(render=True)
            if self.robot and hasattr(self.robot, '_reinitialize_physics'):
                self.robot._reinitialize_physics()
            if self.world:
                for _ in range(3): self.world.step(render=True)

        def _try_teleport_open():
            dof_names = self.robot._articulation.dof_names
            prefix = "L" if self.side == "left" else "R"
            f_names = [f"{prefix}_finger1_joint", f"{prefix}_finger2_joint"]
            f_idx = [self.robot._articulation.get_dof_index(n)
                     for n in f_names if n in dof_names]
            if not f_idx:
                print(f"  [S6] ⚠ No {prefix} finger joints found in DOF list")
                return False
            open_w = getattr(self.robot, 'gripper_open_width', -0.0215)
            self.robot._articulation.set_joint_positions(
                torch.tensor([[open_w] * len(f_idx)], dtype=torch.float32),
                joint_indices=torch.tensor(f_idx, dtype=torch.int32)
            )
            print(f"  [S6] ✓ Gripper teleported open: {f_names} → {open_w:.4f}m")
            return True

        # Primary: set_joint_positions teleports fingers directly — bypasses PD drive
        # so it works even if apply_action/drive stiffness is unreliable after reinit.
        _opened = False
        if self.robot and self.robot._articulation:
            try:
                _opened = _try_teleport_open()
            except AttributeError as _ae:
                if "_physics_view" in str(_ae) and hasattr(self.robot, '_reinitialize_physics'):
                    print(f"  [S6] _physics_view still missing — reinitializing again...")
                    self.robot._reinitialize_physics()
                    if self.world:
                        for _ in range(5): self.world.step(render=True)
                    try:
                        _opened = _try_teleport_open()
                    except Exception as _retry_e:
                        print(f"  [S6] set_joint_positions still failed after reinit: {_retry_e}")
                else:
                    print(f"  [S6] set_joint_positions AttributeError: {_ae}")
            except Exception as _e:
                print(f"  [S6] set_joint_positions error: {_e}")

        # Backup: PD drive via open_gripper API
        if self.robot:
            self.robot.open_gripper(side=self.side)
        # Teleport already opened the fingers instantly; 5 frames lets physics
        # register the new finger positions before S7 starts moving the arm.
        if self.world:
            for _ in range(5): self.world.step(render=True)
        self.state = "S7_RETREAT"

    def _state_s7_retreat(self):
        print("[FSM] State: S7_RETREAT [MoveJ] -> raise back to SAFE_FLY_HEIGHT (joint space)")
        # Retreat to T_high_approach (above original object, always reachable since S3 visited it).
        # Avoid T_high_place (above bin): for bins near/at workspace limit the IK fails completely
        # and stalls for the full timeout — adding 45s of dead time per plan.
        ik_retreat = _matrix_to_base_ik(self.T_high_approach)
        if self.robot and self.robot.ik_solver:
            try:
                joints = self.robot.get_joint_states()
                if joints:
                    positions = joints['positions']
                    if positions and isinstance(positions[0], list):
                        positions = positions[0]
                    self.robot.ik_solver.sync_joint_positions(joints['names'], positions)
            except Exception:
                pass
        execute_stage(
            self.robot, self.world, "s7_retreat", ik_retreat, self.side,
            step_size=0.030, max_steps=3000,
            pos_tol=0.10, rot_tol=0.80,
            timeout_sec=30.0,
            ik_rot_tol=0.20, ik_rot_weight=0.2, ik_null_weight=0.0, ik_max_iter=300)
        self.state = "VERIFY"  # luôn tiếp tục kể cả khi retreat không hoàn hảo

    def _state_verify(self):
        print("[FSM] State: VERIFY -> Hoàn tất vòng lặp!")
        self.state = "DONE"


# ======================================================================
# 5. ORCHESTRATOR 
# ======================================================================

def emit_primitive_event(event: PrimitiveEvent, plan_id: str, object_id: str, details: dict = None) -> dict:
    if details is None: details = {}
    return {"event_type": event.value, "plan_id": plan_id, "object_id": object_id, "details": details}

def check_spatial_conflicts(current_plan: dict, remaining_plans: list,
                            safety_margin_m: float = 0.05) -> list:
    cur_pos = np.array(current_plan['object_pose_base']['position_m'][:2])
    cur_width = current_plan.get('grasp_hint', {}).get('grasp_width_m', 0.06)
    danger_r = cur_width / 2.0 + safety_margin_m
    conflicts = []
    for p in remaining_plans:
        if p['object_id'] == current_plan['object_id']:
            continue
        other_pos = np.array(p['object_pose_base']['position_m'][:2])
        dist = float(np.linalg.norm(cur_pos - other_pos))
        if dist < danger_r:
            conflicts.append((p['object_id'], dist))
    return conflicts

def sort_plans_by_confidence(action_plans: list, perception_json_path: str = None) -> list:
    if perception_json_path is None:
        perception_json_path = "/home/ubuntu/thu/src/task1/perception_interface.json"
    try:
        with open(perception_json_path, "r", encoding="utf-8") as f:
            perception = json.load(f)
        confidence_map = {obj["object_id"]: obj.get("confidence", 0.0)
                          for obj in perception.get("objects", [])}
        sorted_plans = sorted(action_plans,
                              key=lambda p: confidence_map.get(p["object_id"], 0.0),
                              reverse=True)
        print("[Planner] Thứ tự gắp theo confidence (cao → thấp):")
        for p in sorted_plans:
            conf = confidence_map.get(p["object_id"], 0.0)
            print(f"  {p['object_id']} ({p['class_id']}): confidence={conf:.3f}")
        return sorted_plans
    except Exception as e:
        print(f"[Planner] Warning: không sort được theo confidence ({e}), giữ thứ tự gốc")
        return action_plans
def debug_scene_objects(world=None):
    """Scan USD stage for all rigid bodies and print their actual world positions.

    Compares world positions with the coordinate transform used in FSM so we
    can detect if JSON plan coordinates differ from actual sim positions.
    """
    try:
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError:
        print("[DEBUG_SCENE] Isaac Sim not available — skipping")
        return

    print("\n" + "="*60)
    print("[DEBUG_SCENE] Scanning stage for all RigidBody prims...")
    print("  Format: world(x,y,z)  →  base(x,y,z)")
    print("  Robot base frame: x_base=y_w+0.20, y_base=-x_w+0.70, z_base=z_w-0.9005")
    print("="*60)

    try:
        stage = get_current_stage()
        count = 0
        for prim in stage.TraverseAll():
            try:
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                path = str(prim.GetPath())
                xf = UsdGeom.Xformable(prim)
                tf = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                t = tf.ExtractTranslation()
                x_w, y_w, z_w = float(t[0]), float(t[1]), float(t[2])
                x_b = y_w + 0.20
                y_b = -x_w + 0.70
                z_b = z_w - 0.9005
                print(f"  {path}")
                print(f"    world: ({x_w:.4f}, {y_w:.4f}, {z_w:.4f})")
                print(f"    base:  ({x_b:.4f}, {y_b:.4f}, {z_b:.4f})")
                count += 1
            except Exception:
                continue
        if count == 0:
            print("  ⚠️ Không tìm thấy rigid body nào. Thử scan tất cả XformPrim...")
            for prim in stage.TraverseAll():
                try:
                    if not UsdGeom.Xformable(prim):
                        continue
                    path = str(prim.GetPath())
                    if any(skip in path for skip in ['/NavMesh', '/PhysicsScene', '/DistantLight',
                                                      '/RectLight', '/GroundPlane', '/NavMesh']):
                        continue
                    xf = UsdGeom.Xformable(prim)
                    tf = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    t = tf.ExtractTranslation()
                    x_w, y_w, z_w = float(t[0]), float(t[1]), float(t[2])
                    if z_w < 0.9 or z_w > 1.2:
                        continue  # chỉ in vật ở độ cao mặt bàn
                    x_b = y_w + 0.20; y_b = -x_w + 0.70; z_b = z_w - 0.9005
                    print(f"  {path}")
                    print(f"    world: ({x_w:.4f}, {y_w:.4f}, {z_w:.4f})")
                    print(f"    base:  ({x_b:.4f}, {y_b:.4f}, {z_b:.4f})")
                    count += 1
                except Exception:
                    continue
        print(f"[DEBUG_SCENE] Tổng: {count} objects tìm thấy")
        print("="*60 + "\n")
    except Exception as e:
        print(f"[DEBUG_SCENE] Error: {e}")


def debug_ee_frame(robot):
    import pinocchio as pin
    import numpy as np
    
    joints = robot.get_joint_states()
    positions = joints['positions']
    if positions and isinstance(positions[0], list):
        positions = positions[0]
    robot.ik_solver.sync_joint_positions(joints['names'], positions)
    
    # FK tại neutral pose
    poses = robot.ik_solver.get_both_ee_poses()
    print(f"[DEBUG] Right EE at neutral (xyzrpy): {poses['right']}")
    print(f"[DEBUG] Left EE at neutral (xyzrpy):  {poses['left']}")
    
    # Rotation matrix của right EE
    right_se3 = robot.ik_solver.get_ee_pose("right")
    print(f"[DEBUG] Right EE rotation matrix:")
    print(right_se3.rotation)
    print(f"[DEBUG] Right EE Z-axis (col 2): {right_se3.rotation[:, 2]}")
    print(f"  → Nếu Z-axis ~ [0,0,-1] thì gripper chĩa xuống (đúng)")
    print(f"  → Nếu Z-axis ~ [1,0,0] hay [0,1,0] thì gripper hướng ngang (sai)")



def run_pipeline_from_person2(action_plan_yaml: str = None, robot=None, world=None,
                               coord_transform=None, perception_json_path: str = None):
    if action_plan_yaml is None: action_plan_yaml = 'src/task1/primitive_spec_task1.yaml'
    motion_specs = load_action_plan(action_plan_yaml)
    # Ưu tiên đọc tọa độ thật từ USD scene; nếu không có thì fallback sang JSON
    action_plans = load_action_plans_from_scene()
    action_plans = sort_plans_by_confidence(action_plans, perception_json_path)

    print(f"\n[System] Khởi chạy {len(action_plans)} plans (sorted by confidence)...")

    # [FIX #3] Áp workaround patches lên robot 1 lần ở đầu pipeline
    if robot is not None:
        patch_robot_workarounds(robot)
        debug_ee_frame(robot)

    # Scan thực tế vị trí vật trong sim để so sánh với JSON plan
    debug_scene_objects(world)

    FILE_OUTPUT_CSV = 'waypoints_task1.csv'
    retry_counts = {plan['plan_id']: 0 for plan in action_plans}

    with open(FILE_OUTPUT_CSV, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Object_id', 'Stage', "Gripper", 'x', 'y', 'z', 'Roll', "Pitch", "Yaw"])

        plan_idx = 0
        while plan_idx < len(action_plans):
            plan = action_plans[plan_idx]
            plan_id = plan['plan_id']
            retry_count = retry_counts[plan_id]

            is_first_plan = (plan_idx == 0)
            is_recovering = (retry_count > 0)

            # Always reset before each plan: ensures _physics_view is valid and
            # joint state is clean after any FixedJoint mutations from prior plans.
            if robot is not None:
                verbose_reset = is_first_plan or is_recovering
                if verbose_reset:
                    print(f"[Init] Teleport tay về Neutral + reset full state...")
                reset_robot_state_full(robot, world, verbose=verbose_reset)

            remaining = action_plans[plan_idx + 1:]
            fsm = PickAndPlaceStateMachine(robot, world, plan, motion_specs,
                                           coord_transform, writer,
                                           remaining_plans=remaining)
            try:
                result = fsm.run()
            except Exception as _fsm_exc:
                import traceback as _tb
                print(f"❌ [FSM CRASH] {plan_id}: {_fsm_exc}")
                _tb.print_exc()
                result = PrimitiveResult(
                    primitive_name="pick_place", success=False,
                    elapsed_s=0.0, retry_count=0,
                    failure_reason=f"fsm_crash: {_fsm_exc}", metrics={})
            
            if result.success:
                final_event = PrimitiveEvent.SUCCESS
            else:
                r = (result.failure_reason or "").lower()
                # Substring match: failure reason may be prefixed with stage name
                # e.g. "s1_pregrasp_fail: timeout" still matches "timeout"
                if any(k in r for k in ["timeout", "stuck", "max_steps", "out_of_workspace",
                                         "too_far", "ik_fail", "collision", "init_error",
                                         "interpolation_stuck"]):
                    final_event = PrimitiveEvent.COLLISION
                elif any(k in r for k in ["empty_gripper", "dropped", "gripper_not_closed",
                                           "grasp_fail"]):
                    final_event = PrimitiveEvent.GRASP_FAIL
                else:
                    final_event = PrimitiveEvent.GRASP_FAIL

            emit_primitive_event(final_event, plan_id, plan['object_id'], 
                                details={"execution_time_s": result.elapsed_s})

            if final_event == PrimitiveEvent.SUCCESS:
                print(f"✅ [SUCCESS] Kế hoạch {plan_id} hoàn tất.")
                plan_idx += 1
                retry_counts[plan_id] = 0
            else:
                print(f"❌ [FAILED] {plan_id} lỗi: {result.failure_reason}")
                policy = EVENT_RETRY_POLICIES.get(final_event, {})
                max_attempts = policy.get("max_attempts", 1)
                retry_count += 1
                retry_counts[plan_id] = retry_count
                
                if retry_count < max_attempts: 
                    print(f"🔄 Retry {retry_count}/{max_attempts}")
                else:
                    print(f"🛑 Abort. Bỏ qua {plan_id}.")
                    plan_idx += 1

    print(f'\n🎉 Toàn bộ Motion Pipeline hoàn tất!')