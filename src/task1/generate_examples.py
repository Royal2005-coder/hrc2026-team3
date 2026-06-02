import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.task1.planner import run_planner

def generate_examples():
    outputs_dir = "lab_outputs/planner_outputs"
    
    # Example PerceptionFrame (mock)
    frame_success = {
        "frame_id": 1,
        "timestamp": 10.0,
        "camera_name": "front_rgbd",
        "objects": [
            {
                "object_id": "obj_001",
                "class_id": "part_A",
                "confidence": 0.95,
                "bbox_xyxy": [0,0,10,10],
                "centroid_px": [5,5],
                "centroid_camera_m": [0,0,1],
                "pose_base": {
                    "position_m": [-0.5, 0.0, 0.8],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
                },
                "grasp_hint": {
                    "yaw_rad": 0.0,
                    "grasp_width_m": 0.05,
                    "approach_axis": "z_down"
                },
                "failure_reason": None
            }
        ]
    }
    
    frame_fail = {
        "frame_id": 2,
        "timestamp": 11.0,
        "camera_name": "front_rgbd",
        "objects": [
            {
                "object_id": "obj_002",
                "class_id": "part_B",
                "confidence": 0.4, # Too low
                "bbox_xyxy": [0,0,10,10],
                "centroid_px": [5,5],
                "centroid_camera_m": [0,0,1],
                "pose_base": {
                    "position_m": [-0.5, 0.0, 0.8],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
                },
                "grasp_hint": {
                    "yaw_rad": 0.0,
                    "grasp_width_m": 0.05,
                    "approach_axis": "z_down"
                },
                "failure_reason": None
            },
            {
                "object_id": "obj_003",
                "class_id": "unknown_part", # Invalid class
                "confidence": 0.9, 
                "bbox_xyxy": [0,0,10,10],
                "centroid_px": [5,5],
                "centroid_camera_m": [0,0,1],
                "pose_base": {
                    "position_m": [-0.5, 0.0, 0.8],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
                },
                "grasp_hint": {
                    "yaw_rad": 0.0,
                    "grasp_width_m": 0.05,
                    "approach_axis": "z_down"
                },
                "failure_reason": None
            }
        ]
    }
    
    # 1. Action Plan Examples
    plan, log, rejected = run_planner(frame_success, step_id=1)
    out_plan = os.path.join(outputs_dir, "action_plan_examples.jsonl")
    with open(out_plan, "w") as f:
        if plan:
            plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan)
            f.write(json.dumps(plan_dict) + "\n")
            print(f"Generated {out_plan}")
            
    # 2. Failure Cases Planner
    plan_f, log_f, rejected_f = run_planner(frame_fail, step_id=2)
    out_fail = os.path.join(outputs_dir, "failure_cases_planner.jsonl")
    with open(out_fail, "w") as f:
        for r in rejected_f:
            f.write(json.dumps({"object_id": r.object_id, "failure_reason": r.failure_reason}) + "\n")
        print(f"Generated {out_fail}")

if __name__ == "__main__":
    generate_examples()
