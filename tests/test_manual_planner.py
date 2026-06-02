import json
import os
import pytest
from src.task1.planner import run_planner

def test_planner_manual():
    input_path = "src/task1/perception_interface.json"
    if not os.path.exists(input_path):
        pytest.skip(f"Input file {input_path} not found")
        
    with open(input_path, "r") as f:
        real_frame = json.load(f)

    plan, log, rejected = run_planner(real_frame, step_id=1)

    print("=" * 55)
    print("PLANNER OUTPUT")
    print("=" * 55)
    print(f"\nRejected objects ({len(rejected)}):")
    for r in rejected:
        print(f"  ✗ {r.object_id} → {r.failure_reason}")

    if plan:
        print(f"\nSelected → ActionPlan:")
        plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan)
        print(json.dumps(plan_dict, indent=2))
        print(f"\nEval log:")
        print(json.dumps(log, indent=2))
    else:
        print("\nNo valid object found.")
