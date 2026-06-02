"""
planner.py — Task 1 N2: Object validation, selection, bin mapping, ActionPlan.

N2 nhận perception output từ N1, validate, chọn object tốt nhất,
map class_id → bin, tạo ActionPlan cho N3 thực thi.
"""
import math
import json
import time
from datetime import datetime, timezone


# ── Failure reason enum ───────────────────────────────────────────────────────
MISSING_FIELD        = "MISSING_FIELD"
LOW_CONFIDENCE       = "LOW_CONFIDENCE"
INVALID_POSITION     = "INVALID_POSITION"
INVALID_QUATERNION   = "INVALID_QUATERNION"
BAD_QUATERNION_NORM  = "BAD_QUATERNION_NORM"
OUT_OF_WORKSPACE     = "OUT_OF_WORKSPACE"
CLASS_NOT_MAPPED     = "CLASS_NOT_MAPPED"
ALREADY_HANDLED      = "ALREADY_HANDLED"
NO_VALID_OBJECT      = "NO_VALID_OBJECT"
PLAN_ERROR           = "PLAN_ERROR"


# ── Object state validator (N2 Step 3) ───────────────────────────────────────
def validate_object_state(obj, min_confidence=0.70):
    """Validate required fields + confidence. Returns (ok, failure_reason)."""
    required = ["object_id", "class_id", "confidence", "pose_base", "grasp_hint"]
    for key in required:
        if key not in obj:
            return False, f"{MISSING_FIELD}_{key.upper()}"

    if obj.get("failure_reason"):
        return False, str(obj["failure_reason"])

    if float(obj["confidence"]) < min_confidence:
        return False, LOW_CONFIDENCE

    pos  = obj["pose_base"].get("position_m")
    quat = obj["pose_base"].get("quaternion_xyzw")

    if pos is None or len(pos) != 3:
        return False, INVALID_POSITION
    if quat is None or len(quat) != 4:
        return False, INVALID_QUATERNION
    if any(not isinstance(v, (int, float)) for v in pos):
        return False, INVALID_POSITION

    return True, None


# ── Pose sanity check (N2 Step 4) ────────────────────────────────────────────
def is_pose_sane(obj, workspace):
    """Check quaternion norm and position within workspace."""
    quat = obj["pose_base"]["quaternion_xyzw"]
    norm = math.sqrt(sum(float(x) ** 2 for x in quat))
    if abs(norm - 1.0) > 0.15:
        return False, BAD_QUATERNION_NORM

    pos = obj.get("pos_world") or obj["pose_base"]["position_m"]
    x, y, z = [float(v) for v in pos]

    if not (workspace["x"][0] <= x <= workspace["x"][1]):
        return False, OUT_OF_WORKSPACE
    if not (workspace["y"][0] <= y <= workspace["y"][1]):
        return False, OUT_OF_WORKSPACE
    if not (workspace["z"][0] <= z <= workspace["z"][1]):
        return False, OUT_OF_WORKSPACE

    return True, None


# ── Object scoring (N2 Step 5) ───────────────────────────────────────────────
def score_object(obj):
    """Higher score = better candidate. Prefer high confidence, near centre."""
    conf = float(obj["confidence"])
    pos  = obj.get("pos_world") or obj["pose_base"]["position_m"]
    y    = float(pos[1])
    lateral_penalty = 0.02 * abs(y - 0.3)
    return conf - lateral_penalty


# ── Select next object ────────────────────────────────────────────────────────
def select_next_object(objects, handled_ids, workspace, min_confidence=0.70):
    """Return (best_obj, rejected_list). rejected_list = [(id, reason), ...]."""
    candidates = []
    rejected   = []

    for obj in objects:
        oid = obj.get("object_id", "unknown")

        if oid in handled_ids:
            rejected.append((oid, ALREADY_HANDLED))
            continue

        ok, reason = validate_object_state(obj, min_confidence)
        if not ok:
            rejected.append((oid, reason))
            continue

        ok, reason = is_pose_sane(obj, workspace)
        if not ok:
            rejected.append((oid, reason))
            continue

        candidates.append((score_object(obj), obj))

    if not candidates:
        return None, rejected

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], rejected


# ── Bin mapping ───────────────────────────────────────────────────────────────
def map_class_to_bin(class_id, class_to_bin):
    if class_id not in class_to_bin:
        raise ValueError(f"CLASS_NOT_MAPPED: {class_id}")
    return class_to_bin[class_id]


# ── ActionPlan builder ────────────────────────────────────────────────────────
def make_action_plan(obj, bins_config, retry_policy="retry_once_slow"):
    """Create full ActionPlan for N3 motion primitive."""
    bin_name = map_class_to_bin(obj["class_id"], bins_config["class_to_bin"])
    bin_info = bins_config["bins"][bin_name]

    return {
        "primitive":        "pick_place",
        "object_id":        obj["object_id"],
        "class_id":         obj["class_id"],
        "object_pose_base": obj["pose_base"],
        "object_pos_world": obj.get("pos_world"),   # world frame — N3 dùng cho IK
        "target_bin":       bin_name,
        "bin_pose_base":    bin_info["pose_base"],
        "bin_pos_world":    bin_info["pos_world"],   # world frame — N3 dùng cho IK
        "grasp_hint":       obj["grasp_hint"],
        "retry_policy":     retry_policy,
        "timeout_s":        30.0,
    }


# ── Planner trace logger ──────────────────────────────────────────────────────
def log_event(path, event, **kwargs):
    if path is None:
        return
    record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **kwargs}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Task1Planner class ────────────────────────────────────────────────────────
class Task1Planner:
    """
    N2 Planner: validate → select → plan → log.
    Không tự gọi motion; trả ActionPlan để FSM/motion thực thi.
    """

    def __init__(self, bins_config, workspace, min_confidence=0.70, trace_path=None):
        self.bins_config    = bins_config
        self.workspace      = workspace
        self.min_confidence = min_confidence
        self.trace_path     = trace_path
        self.handled_ids    = set()
        self.retry_counts   = {}        # object_id → số lần retry

    def reset(self):
        self.handled_ids.clear()
        self.retry_counts.clear()
        log_event(self.trace_path, "planner_reset")

    def mark_handled(self, object_id):
        self.handled_ids.add(object_id)
        log_event(self.trace_path, "object_handled", object_id=object_id)

    def can_retry(self, object_id, max_retries=1):
        return self.retry_counts.get(object_id, 0) < max_retries

    def increment_retry(self, object_id):
        self.retry_counts[object_id] = self.retry_counts.get(object_id, 0) + 1

    def plan(self, objects):
        """
        Select best object + create ActionPlan.
        Returns (action_plan | None, info_dict).
        """
        obj, rejected = select_next_object(
            objects,
            handled_ids=self.handled_ids,
            workspace=self.workspace,
            min_confidence=self.min_confidence,
        )

        if rejected:
            for oid, reason in rejected:
                print(f"  [Planner] reject {oid}: {reason}")
                log_event(self.trace_path, "object_rejected",
                          object_id=oid, reason=reason)

        if obj is None:
            info = {"failure_reason": NO_VALID_OBJECT, "rejected": rejected}
            log_event(self.trace_path, "plan_failed", **info)
            return None, info

        try:
            action_plan = make_action_plan(obj, self.bins_config)
        except Exception as e:
            info = {"failure_reason": PLAN_ERROR, "error": str(e),
                    "object_id": obj.get("object_id")}
            log_event(self.trace_path, "plan_error", **info)
            return None, info

        info = {
            "failure_reason": None,
            "object_id":      obj["object_id"],
            "class_id":       obj["class_id"],
            "target_bin":     action_plan["target_bin"],
        }
        log_event(self.trace_path, "plan_created", **info)
        print(f"  [Planner] plan → {obj['object_id']} ({obj['class_id']}) "
              f"→ {action_plan['target_bin']}")
        return action_plan, info
