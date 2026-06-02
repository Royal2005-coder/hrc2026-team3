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
import cv2
from datetime import datetime, timezone


# ── Failure reasons ───────────────────────────────────────────────────────────
BAD_ACTION_PLAN     = "BAD_ACTION_PLAN"
OUT_OF_WORKSPACE    = "OUT_OF_WORKSPACE_MOTION"
IK_FAIL_GRASP       = "IK_FAIL_GRASP"
GRASP_NOT_CONFIRMED = "GRASP_NOT_CONFIRMED"
TIMEOUT             = "TIMEOUT"
RUNTIME_ERROR       = "RUNTIME_ERROR"


# HSV ranges từ perception.py — dùng lại cho wrist camera detection
_HSV_RANGES = {
    "part_A": [
        (np.array([0,  35, 30]),  np.array([15, 255, 255])),   # red
        (np.array([165,35, 30]),  np.array([179,255, 255])),   # red wrap
        (np.array([15, 35, 80]),  np.array([40, 255, 255])),   # gold
    ],
    "part_B": [
        (np.array([87, 30,150]),  np.array([110,255, 255])),   # blue
    ],
}


def _detect_centroid_wrist(rgb, class_id="part_A"):
    """
    Detect object centroid in wrist camera RGB image using HSV thresholding.
    Returns (cx, cy) pixel coords, or (None, None) if not found.
    """
    if rgb is None:
        return None, None

    img = np.array(rgb, dtype=np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    hsv  = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lo, hi in _HSV_RANGES.get(class_id, _HSV_RANGES["part_A"]):
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

    # Morphological cleanup — loại noise nhỏ
    k    = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:   # quá nhỏ → nhiễu
        return None, None

    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, None

    return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])


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

        elif s["type"] == "servo":
            left_t  = self._last_left_t
            right_t = self._last_right_t

            rgb     = self.robot.get_camera_rgb(s["camera"])
            cx, cy  = _detect_centroid_wrist(rgb, s["class_id"])

            W = rgb.shape[1] if rgb is not None else 640
            H = rgb.shape[0] if rgb is not None else 480

            self._hold_counter += 1

            if cx is None:
                # Không thấy vật — giữ nguyên, chờ tối đa max_steps
                if self._hold_counter >= s["max_steps"]:
                    self._executed.append(s["name"])
                    self._log("servo_done", name=s["name"], reason="timeout_no_object")
                    print(f"[N3] ✓ {s['name']}  timeout (vật không thấy trong wrist cam)")
                    self._step_idx   += 1
                    self._hold_counter = 0
                return left_t, right_t

            err_x   = cx - W / 2   # >0: vật bên phải ảnh
            err_y   = cy - H / 2   # >0: vật bên dưới ảnh
            aligned = abs(err_x) < s["tol_px"] and abs(err_y) < s["tol_px"]

            if not aligned:
                dx, dy = self._servo_delta(err_x, err_y, s["gain"])
                arm    = s["arm"]
                prev   = self._last_left_t if arm == "left" else self._last_right_t
                if prev is not None:
                    tgt    = list(prev)
                    tgt[0] += dx
                    tgt[1] += dy
                    if arm == "left":
                        left_t = tgt;  self._last_left_t  = tgt
                    else:
                        right_t = tgt; self._last_right_t = tgt

            # Log mỗi 30 ticks
            self._reach_log_n += 1
            if self._reach_log_n >= 30:
                print(f"[N3] servo  err=({err_x:+.0f},{err_y:+.0f})px  "
                      f"aligned={aligned}  tick={self._hold_counter}")
                self._reach_log_n = 0

            if aligned or self._hold_counter >= s["max_steps"]:
                reason = "aligned" if aligned else "timeout"
                self._executed.append(s["name"])
                self._log("servo_done", name=s["name"], reason=reason,
                          err_x=round(err_x, 1), err_y=round(err_y, 1))
                print(f"[N3] ✓ {s['name']}  {reason}  "
                      f"err=({err_x:+.0f},{err_y:+.0f})px")
                self._step_idx   += 1
                self._hold_counter = 0

        elif s["type"] == "confirm":
            # MVP: không có force/contact sensor → luôn pass sau 1 tick
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

        h_pre    = params.get("pre_grasp_height_m",  0.10)
        h_lift   = params.get("lift_height_m",        0.20)
        h_place  = params.get("pre_place_height_m",   0.12)
        gc_s     = params.get("gripper_close_steps",  60)
        go_s     = params.get("gripper_open_steps",   30)
        # Offset z để gắp tại TOP SURFACE thay vì centroid (= object_half_height)
        gz_off   = params.get("grasp_z_offset_m",     0.0)
        # Wrist servo params
        cam_name = "wrist_left" if self._arm == "left" else "wrist_right"
        class_id = action_plan.get("class_id", "part_A")
        srv_tol  = params.get("servo_tol_px",         20)
        srv_gain = params.get("servo_gain",            0.0003)
        srv_max  = params.get("servo_max_steps",       300)
        arm      = self._arm
        fo       = self.finger_open
        fc       = self.finger_close

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
            # 1. Di chuyển tới trên vật (head camera pose, ±10cm error OK)
            {"type": "move",    "name": "pre_grasp",    "arm": arm,
             "target": w2_6d(obj_w + [0, 0, h_pre]),  "tol": 0.04},
            # 2. Mở gripper
            {"type": "gripper", "name": "open_gripper", "side": arm,
             "pos": fo, "hold": go_s},
            # 3. Wrist servo — căn chỉnh bằng camera tay đến khi centroid vào center
            #    Tune: servo_gain (0.0002–0.0005), servo_tol_px (15–25)
            #    Nếu robot đi sai chiều: đổi dấu trong _servo_delta()
            {"type": "servo",   "name": "wrist_servo",
             "arm": arm, "camera": cam_name, "class_id": class_id,
             "tol_px": srv_tol, "gain": srv_gain, "max_steps": srv_max},
            # 4. Hạ xuống TOP SURFACE (centroid + grasp_z_offset_m)
            {"type": "move",    "name": "grasp",         "arm": arm,
             "target": w2_6d(obj_w + [0, 0, gz_off]),  "tol": 0.02},
            # 5. Đóng gripper
            {"type": "gripper", "name": "close_gripper","side": arm,
             "pos": fc, "hold": gc_s},
            # 6. Confirm grasp (MVP: luôn pass)
            {"type": "confirm", "name": "confirm_grasp"},
            # 7. Nhấc lên
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

    def _servo_delta(self, err_x, err_y, gain):
        """
        Chuyển pixel error → delta robot base frame (dx, dy).

        Mapping mặc định (wrist camera nhìn xuống, Walker S2):
          image +X (phải)  → base -Y (phải robot)   sign_x = -1
          image +Y (xuống) → base +X (trước robot)  sign_y = +1

        Nếu robot đi sai chiều khi test → đổi dấu sign_x hoặc sign_y.
        Nếu trục x/y bị hoán đổi → swap dx/dy.
        """
        sign_x = -1   # TUNE: +1 hoặc -1
        sign_y = +1   # TUNE: +1 hoặc -1
        dx = sign_y * err_y * gain
        dy = sign_x * err_x * gain
        return dx, dy

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
