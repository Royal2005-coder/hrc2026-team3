import json
import os
import pytest
from src.task1.fsm import Task1FSM, make_mock_person3

def test_fsm_manual():
    input_path = "src/task1/perception_interface.json"
    if not os.path.exists(input_path):
        pytest.skip(f"Input file {input_path} not found")
        
    with open(input_path, "r") as f:
        real_frame = json.load(f)

    scenarios = ["all_success", "first_grasp_fail", "exceed_retries", "drop_on_place"]
    os.makedirs("outputs", exist_ok=True)

    for scenario in scenarios:
        print(f"\n{'#'*55}")
        print(f"  SCENARIO: {scenario}")
        print(f"{'#'*55}")
        fsm = Task1FSM()
        fsm.run(real_frame, mock_person3=make_mock_person3(scenario))

        # Save trace
        out_path = f"lab_outputs/planner_outputs/planner_trace_{scenario}.jsonl"
        with open(out_path, "w") as f:
            for entry in fsm.get_trace():
                f.write(json.dumps(entry) + "\n")
        print(f"\n  Trace saved → {out_path}")
