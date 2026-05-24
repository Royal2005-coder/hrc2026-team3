import json
import os
import pytest
from src.task1.generate_pose_report import build_pose_report

def test_generate_pose_report_manual():
    input_path = "src/task1/perception_interface.json"
    if not os.path.exists(input_path):
        pytest.skip(f"Input file {input_path} not found")
        
    with open(input_path, "r") as f:
        real_frame = json.load(f)
        
    report = build_pose_report(real_frame)
    os.makedirs("outputs", exist_ok=True)
    out_path = "lab_outputs/planner_outputs/object_bin_pose_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved → {out_path}")
    print(f"\nSummary: {report['summary']}\n")
    for obj in report["objects"]:
        status = obj["status"]
        reason = f"  ← {obj['reject_reason']}" if obj["reject_reason"] else ""
        bin_id = obj.get("assigned_bin") or "—"
        print(f"  {obj['object_id']} ({obj['class_id']}) conf={obj['confidence']}  →  [{status}]  bin={bin_id}{reason}")
