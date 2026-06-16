from src.task1.transform_utils import validate_transform, quaternion_to_yaw
import yaml
import json
from pathlib import Path


DEFAULT_BINS = {
    "part_A": {
        "bin_id": "bin_A",
        "position_m": [-0.45, 0.25, 0.80],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "grasp_width_m": 0.045,
        "description": "Blue bin — left side of table"
    },
    "part_B": {
        "bin_id": "bin_B",
        "position_m": [-0.45, -0.25, 0.80],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "grasp_width_m": 0.060,
        "description": "Red bin — right side of table"
    }
}


def load_bins_config() -> dict:
    """
    Load bin definitions from configs/bins.yaml.

    Returns:
        dict containing bin configurations.

    Falls back to DEFAULT_BINS if:
      - file missing
      - YAML invalid
      - bins key missing
    """

    try:

        with open("configs/bins.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        bins = config.get("bins")

        if not bins:
            raise ValueError("Missing 'bins' section in bins.yaml")

        return bins

    except Exception as e:
        print(f"[WARN] Failed to load bins.yaml: {e}")
        print("[WARN] Using default bin configuration.")

        return DEFAULT_BINS


BINS = load_bins_config()

CONFIDENCE_THRESHOLD = 0.75


# Build the report
def build_pose_report(perception_frame: dict) -> dict:
    """
    For each object:
      1. Validate confidence
      2. Validate pose (position + quaternion)
      3. Map class_id → bin
      4. Validate bin pose
      5. Record result with status
    """
    report = {
        "frame_id": perception_frame["frame_id"],
        "timestamp": perception_frame["timestamp"],
        "summary": {},
        "objects": []
    }

    total = 0
    accepted = 0
    rejected = 0

    for obj in perception_frame["objects"]:
        total += 1
        obj_id    = obj["object_id"]
        class_id  = obj["class_id"]
        conf      = obj["confidence"]
        pos       = obj["pose_base"]["position_m"]
        quat      = obj["pose_base"]["quaternion_xyzw"]
        hint      = obj["grasp_hint"]

        entry = {
            "object_id":   obj_id,
            "class_id":    class_id,
            "confidence":  conf,
            "status":      None,
            "reject_reason": None,
            "object_pose_robot_base": {
                "position_m":       pos,
                "quaternion_xyzw":  quat,
                "yaw_rad_extracted": None
            },
            "assigned_bin":  None,
            "bin_pose_robot_base": None
        }

        # --- Check 1: confidence threshold ---
        if conf < CONFIDENCE_THRESHOLD:
            entry["status"] = "REJECTED"
            entry["reject_reason"] = f"confidence {conf} < threshold {CONFIDENCE_THRESHOLD}"
            rejected += 1
            report["objects"].append(entry)
            continue

        # --- Check 2: validate object pose ---
        pose_valid, pose_reason = validate_transform(pos, quat)
        if not pose_valid:
            entry["status"] = "REJECTED"
            entry["reject_reason"] = f"pose invalid: {pose_reason}"
            rejected += 1
            report["objects"].append(entry)
            continue

        # --- Extract yaw ---
        yaw = quaternion_to_yaw(quat)
        entry["object_pose_robot_base"]["yaw_rad_extracted"] = round(yaw, 4)

        # --- Check 3: map class → bin ---
        if class_id not in BINS:
            entry["status"] = "REJECTED"
            entry["reject_reason"] = f"class_id '{class_id}' not found in bins.yaml"
            rejected += 1
            report["objects"].append(entry)
            continue

        bin_info = BINS[class_id]
        bin_pos  = bin_info["position_m"]
        bin_quat = bin_info["quaternion_xyzw"]

        # --- Check 4: validate bin pose ---
        bin_valid, bin_reason = validate_transform(bin_pos, bin_quat)
        if not bin_valid:
            entry["status"] = "REJECTED"
            entry["reject_reason"] = f"bin pose invalid: {bin_reason}"
            rejected += 1
            report["objects"].append(entry)
            continue

        # --- All checks passed ---
        entry["status"] = "ACCEPTED"
        entry["assigned_bin"] = bin_info["bin_id"]
        entry["bin_pose_robot_base"] = {
            "position_m":      bin_pos,
            "quaternion_xyzw": bin_quat,
            "description":     bin_info["description"]
        }
        accepted += 1
        report["objects"].append(entry)

    report["summary"] = {
        "total_objects":    total,
        "accepted":         accepted,
        "rejected":         rejected,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }

    return report