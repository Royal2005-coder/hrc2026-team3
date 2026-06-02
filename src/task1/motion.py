"""
motion.py — Task 1 N3: Pick-place motion primitive runner (step-based cho Isaac Sim).

N3 nhận ActionPlan từ N2, thực thi chuỗi waypoint:
  pre_grasp → open → grasp → close → lift → pre_place → place → release → retreat

Mỗi step() gọi một lần per physics tick — không blocking.
"""
import json
import time
import numpy as np
import pinocchio as pin
from datetime import datetime, timezone


# ── Failure reasons ───────────────────────────────────────────────────────────
BAD_ACTION_PLAN     = "BAD_ACTION_PLAN"
OUT_OF_WORKSPACE    = "OUT_OF_WORKSPACE_MOTION"
IK_FAIL_GRASP       = "IK_FAIL_GRASP"
GRASP_NOT_CONFIRMED = "GRASP_NOT_CONFIRMED"
TIMEOUT             = "TIMEOUT"
RUNTIME_ERROR       = "RUNTIME_ERROR"


def check_preconditions(action_plan, workspace):
    """Validate ActionPlan schema + positions. Returns (ok, failure_reason)."""
    required = ["object_id", "object_pose_base", "bin_pose_base", "grasp_hint"]
    for key in required:
        if key not in action_plan:
            return False, f"{BAD_ACTION_PLAN}_MISSING_{key.upper()}"

    grasp_w = action_plan["grasp_hint"].get("grasp_width_m", 0)
    if grasp_w < 0:
        return False, f"{BAD_ACTION_PLAN}_INVALID_GRASP_WIDTH"

    # World-frame bounds check (optional — skips if pos_world not provided)
    for label, key in [("object", "object_pos_world"), ("bin", "bin_pos_world")]:
        pos = action_plan.get(key)
        if pos is None:
            continue
        x, y, z = [float(v) for v in pos]
        if not (workspace["x"][0] <= x <= workspace["x"][1]):
            return False, f"{label.upper()}_OUT_OF_WORKSPACE_X"
        if not (workspace["y"][0] <= y <= workspace["y"][1]):
            return False, f"{label.upper()}_OUT_OF_WORKSPACE_Y"
        if not (workspace["z"][0] <= z <= workspace["z"][1]):
            return False, f"{label.upper()}_OUT_OF_WORKSPACE_Z"

    return True, None


def apply_retry_adjustment(params, failure_reason):
    """Return adjusted params dict for retry attempt."""
    p = dict(params)
    if failure_reason in (IK_FAIL_GRASP, GRASP_NOT_CONFIRMED, "COLLISION_ON_DESCEND"):
        p["pre_grasp_height_m"]  = params.get("pre_grasp_height_m", 0.10) + 0.05
        p["gripper_close_steps"] = params.get("gripper_close_steps", 60) + 30
    elif failure_reason in ("DROP_DURING_LIFT", "COLLISION_TRANSFER"):
        p["lift_height_m"] = params.get("lift_height_m", 0.20) + 0.05
    return p


class MotionPrimitiveRunner:
    """
    Step-based motion primitive runner cho Isaac Sim physics callback.

    Cách dùng:
        ok = runner.start_pick_place(action_plan)
        # Mỗi physics tick:
        left_t, right_t = runner.step(step_size)
        if runner.is_done():
            result = runner.get_result()
    """

    IDLE    = "IDLE"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED  = "FAILED"

    def __init__(self, robot, coord_transform, workspace, params,
                 finger_open=0.0, finger_close=0.7, trace_path=None):
        self.robot          = robot
        self.coord          = coord_transform
        self.workspace      = workspace
        self.params         = params
        self.finger_open    = finger_open
        self.finger_close   = finger_close
        self.trace_path     = trace_path

        self._state         = self.IDLE
        self._steps         = []
        self._step_idx      = 0
        self._hold_counter  = 0
        self._arm           = "left"
        self._last_left_t   = None     # keep sending last target when in gripper step
        self._last_right_t  = None
        self._gripper_state = {"left": finger_open, "right": finger_open}
        self._result        = None
        self._start_time    = 0.0
        self._executed      = []
        self._object_id     = "unknown"
        self._attempt       = 0
        self._reach_log_n   = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def start_pick_place(self, action_plan, attempt=0):
        """Queue pick_place. Returns False if preconditions fail."""
        ok, reason = check_preconditions(action_plan, self.workspace)
        if not ok:
            self._result = self._make_result(False, reason, [])
            self._state  = self.FAILED
            self._log("precondition_fail", reason=reason)
            return False

        self._object_id    = action_plan.get("object_id", "unknown")
        self._attempt      = attempt
        self._start_time   = time.time()
        self._executed     = []
        self._step_idx     = 0
        self._hold_counter = 0
        self._state        = self.RUNNING

        # Chọn tay
        obj_world = action_plan.get("object_pos_world")
        if obj_world is not None:
            pb = self.coord.world_to_robot(np.array(obj_world, dtype=float))
            self._arm = "left" if pb[1] > 0 else "right"
        else:
            self._arm = "left"

        self._steps = self._build_steps(action_plan,
                                        apply_retry_adjustment(self.params, IK_FAIL_GRASP)
                                        if attempt > 0 else self.params)
        self._log("pick_place_start", object_id=self._object_id,
                  arm=self._arm, attempt=attempt)
        print(f"[N3] start_pick_place  {self._object_id}  arm={self._arm}  attempt={attempt}")
        return True

    def step(self, step_size):
        """
        Gọi mỗi physics tick.
        Trả về (left_target_6d | None, right_target_6d | None).
        """
        if self._state != self.RUNNING:
            return None, None

        # Timeout guard
        if time.time() - self._start_time > self.params.get("timeout_s", 30.0):
            self._result = self._make_result(False, TIMEOUT, self._executed)
            self._state  = self.FAILED
            self._log("timeout", object_id=self._object_id)
            return None, None

        if self._step_idx >= len(self._steps):
            # Tất cả waypoints xong → SUCCESS
            self._result = self._make_result(True, None, self._executed)
            self._state  = self.SUCCESS
            self._log("success", object_id=self._object_id,
                      duration_s=round(time.time() - self._start_time, 2))
            print(f"[N3] SUCCESS  {self._object_id}  "
                  f"dur={round(time.time()-self._start_time,1)}s")
            return None, None

        s = self._steps[self._step_idx]
        left_t = right_t = None

        if s["type"] == "move":
            arm    = s["arm"]
            target = s["target"]
            tol    = s["tol"]

            if arm == "left":
                left_t = target
                self._last_left_t = target
            else:
                right_t = target
                self._last_right_t = target

            if self._is_reached(arm, target, tol):
                self._executed.append(s["name"])
                self._log("waypoint_reached", name=s["name"])
                print(f"[N3] ✓ {s['name']}")
                self._step_idx   += 1
                self._hold_counter = 0

        elif s["type"] == "gripper":
            # Áp dụng gripper
            side = s["side"]
            pos  = s["pos"]
            self._gripper_state[side] = pos
            self._apply_gripper(side, pos)

            # Giữ nguyên IK target của lần move trước
            left_t  = self._last_left_t
            right_t = self._last_right_t

            self._hold_counter += 1
            if self._hold_counter >= s["hold"]:
                self._executed.append(s["name"])
                self._log("gripper_done", name=s["name"], pos=pos)
                print(f"[N3] ✓ {s['name']}  pos={pos:.2f}")
                self._step_idx   += 1
                self._hold_counter = 0

        elif s["type"] == "confirm":
            # MVP: không có force/contact sensor → luôn pass sau 1 tick
            # Nếu sau này có sensor: kiểm tra gripper width hoặc object still present
            left_t  = self._last_left_t
            right_t = self._last_right_t
            self._executed.append(s["name"])
            self._log("confirm_grasp", name=s["name"], result="ok_mvp")
            print(f"[N3] ✓ {s['name']}  (mvp)")
            self._step_idx   += 1
            self._hold_counter = 0

        return left_t, right_t

    def is_done(self):
        return self._state in (self.SUCCESS, self.FAILED)

    def is_success(self):
        return self._state == self.SUCCESS

    def get_result(self):
        return self._result

    def get_gripper_state(self):
        return self._gripper_state

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_steps(self, action_plan, params):
        """Tạo danh sách step từ ActionPlan."""
        obj_w = np.array(action_plan.get("object_pos_world") or
                         action_plan["object_pose_base"]["position_m"], dtype=float)
        bin_w = np.array(action_plan.get("bin_pos_world") or
                         action_plan["bin_pose_base"]["position_m"], dtype=float)

        h_pre   = params.get("pre_grasp_height_m", 0.10)
        h_lift  = params.get("lift_height_m",       0.20)
        h_place = params.get("pre_place_height_m",  0.12)
        gc_s    = params.get("gripper_close_steps", 60)
        go_s    = params.get("gripper_open_steps",  30)
        arm     = self._arm
        fo      = self.finger_open
        fc      = self.finger_close

        # ── Tính rotation gripper từ grasp_hint.yaw_rad (N3 guide Step 4) ─
        # Z-axis hướng xuống (approach_axis=z_down), X-axis theo yaw của vật.
        # Dùng yaw_rad từ grasp_hint thay vì tự tính reach_dir để đồng nhất với
        # contract N1→N2→N3 và tránh LOG MAP inflate khi rot_weight>0.
        yaw = float(action_plan.get("grasp_hint", {}).get("yaw_rad", 0.0))

        z_world   = np.array([0.0, 0.0, -1.0])
        x_world   = np.array([np.cos(yaw), np.sin(yaw), 0.0])

        base_down = self.coord.robot_world_R_inv @ z_world
        base_down /= np.linalg.norm(base_down)

        x_base    = self.coord.robot_world_R_inv @ x_world
        x_base    = x_base - np.dot(x_base, base_down) * base_down
        if np.linalg.norm(x_base) < 1e-6:
            perp   = np.array([1.0, 0.0, 0.0]) if abs(base_down[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x_base = perp - np.dot(perp, base_down) * base_down
        x_base   /= np.linalg.norm(x_base)
        y_base    = np.cross(base_down, x_base)
        y_base   /= np.linalg.norm(y_base)

        R_grasp   = np.column_stack([x_base, y_base, base_down])
        grasp_rpy = pin.rpy.matrixToRpy(R_grasp)

        def w2_6d(pos_world):
            pos_base = self.coord.world_to_robot(pos_world)
            return np.concatenate([pos_base, grasp_rpy])

        return [
            # 1. Di chuyển tới trên vật
            {"type": "move",    "name": "pre_grasp",    "arm": arm,
             "target": w2_6d(obj_w + [0, 0, h_pre]),  "tol": 0.04},
            # 2. Mở gripper
            {"type": "gripper", "name": "open_gripper", "side": arm,
             "pos": fo, "hold": go_s},
            # 3. Hạ xuống vị trí gắp
            {"type": "move",    "name": "grasp",         "arm": arm,
             "target": w2_6d(obj_w),                   "tol": 0.03},
            # 4. Đóng gripper
            {"type": "gripper", "name": "close_gripper","side": arm,
             "pos": fc, "hold": gc_s},
            # 5. Confirm grasp (MVP: luôn pass; log GRASP_NOT_CONFIRMED nếu sensor phát hiện drop)
            {"type": "confirm", "name": "confirm_grasp"},
            # 6. Nhấc lên
            {"type": "move",    "name": "lift",          "arm": arm,
             "target": w2_6d(obj_w + [0, 0, h_lift]),  "tol": 0.04},
            # 7. Di chuyển tới trên bin
            {"type": "move",    "name": "pre_place",     "arm": arm,
             "target": w2_6d(bin_w + [0, 0, h_place]),  "tol": 0.05},
            # 7. Hạ vào bin
            {"type": "move",    "name": "place",         "arm": arm,
             "target": w2_6d(bin_w + [0, 0, 0.03]),     "tol": 0.04},
            # 8. Thả vật
            {"type": "gripper", "name": "release",       "side": arm,
             "pos": fo, "hold": go_s},
            # 9. Rút tay lên
            {"type": "move",    "name": "retreat",       "arm": arm,
             "target": w2_6d(bin_w + [0, 0, h_place + 0.05]), "tol": 0.06},
        ]

    def _is_reached(self, side, target_6d, tol):
        js = self.robot.get_joint_states()
        if js is None:
            return False
        self.robot.ik_solver.sync_joint_positions(js["names"], js["positions"][0])
        ee_pos = np.array(self.robot.ik_solver.get_ee_pose(side).translation)
        err    = float(np.linalg.norm(ee_pos - np.array(target_6d[:3])))

        self._reach_log_n += 1
        if self._reach_log_n >= 30:
            step_name = self._steps[self._step_idx]["name"] if self._step_idx < len(self._steps) else "?"
            print(f"[N3] {side} err={err:.4f}m tol={tol}  [{step_name}]")
            self._reach_log_n = 0
        return err < tol

    def _apply_gripper(self, side, pos):
        import torch
        JOINTS = {
            "left":  ["L_finger1_joint", "L_finger2_joint"],
            "right": ["R_finger1_joint", "R_finger2_joint"],
        }
        dof = self.robot._articulation.dof_names
        idxs, vals = [], []
        for j in JOINTS.get(side, []):
            if j in dof:
                idxs.append(self.robot._articulation.get_dof_index(j))
                vals.append(pos)
        if idxs:
            self.robot._articulation.set_joint_positions(
                torch.tensor(vals, dtype=torch.float32),
                joint_indices=torch.tensor(idxs, dtype=torch.int32),
            )

    def _make_result(self, success, failure_reason, executed):
        return {
            "object_id":          self._object_id,
            "primitive":          "pick_place",
            "primitive_success":  success,
            "duration_s":         round(time.time() - self._start_time, 2),
            "retry_count":        self._attempt,
            "failure_reason":     failure_reason,
            "waypoints_executed": executed,
        }

    def _log(self, event, **kwargs):
        if self.trace_path is None:
            return
        record = {"time": datetime.now(timezone.utc).isoformat(),
                  "event": event, **kwargs}
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
