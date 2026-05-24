# Person 2: Task Planner (HRC_2026)

## Overview
The Task Planner module acts as the logical "brain" of the robotic pick-and-place system. It translates raw 3D perception data (from Person 1) into strictly structured, step-by-step motion plans (for Person 3). It validates object coordinates, selects the best target, maps it to the correct placement bin, and robustly handles any grasp or motion failures through a dynamic retry system.

## Architecture & Core Files

All source code is located in `src/task1/`.

- **`planner.py`** (The Decision Engine)
  Filters out invalid objects based on AI confidence thresholds and reachability boundaries. Selects the optimal target (e.g., highest confidence) and packages it into a ready-to-execute `ActionPlan`.
- **`state.py`** (The Data Dictionary)
  Defines the central data models (`ObjectState`, `ActionPlan`, `BinState`, `FSMState`, `PrimitiveStatus`). Uses strictly typed Dataclasses and Enums to guarantee data integrity across the system.
- **`fsm.py`** (The Workflow Manager)
  A Finite State Machine that controls the entire pick-and-place sequence (`VALIDATE` → `SELECT` → `PICK` → `PLACE` → `VERIFY`). It acts as the central orchestrator.
- **`retry_policy.py`** (The Safety Net)
  Manages error recovery. Analyzes failures from the robot arm (like a slipped grasp or a dropped object) and calculates whether to retry and at what speed (e.g., dynamically slowing the arm down to 60% speed for a second attempt).
- **`transform_utils.py`** (The Math Library)
  Spatial utilities to validate if an object is physically reachable within the 3D workspace, and to convert rotation quaternions into `yaw` angles for the robotic gripper.

## How to Test & Generate Required Outputs

All manual checks have been converted into an automated test suite. To generate all required Person 2 output files, open a terminal at the project root and run the following commands:

```bash
# 1. Run Unit Tests & Export Reports
pytest tests/test_transform.py -v > lab_outputs/planner_outputs/pytest_transform.txt
pytest tests/test_planner.py -v > lab_outputs/planner_outputs/planner_unit_test_report.txt

# 2. Generate Action Plan & Failure Case Examples
python3 src/task1/generate_examples.py

# 3. Generate Object Pose Reports & FSM Black-Box Traces
pytest tests/test_manual_*.py -s
```

## Expected Outputs (`lab_outputs/planner_outputs/`)

After running the commands above, the `lab_outputs/planner_outputs/` folder will be populated with:
- `action_plan_examples.jsonl`: Successfully generated JSON payload examples that Person 3 can consume.
- `failure_cases_planner.jsonl`: Logs of objects rejected due to low confidence, bad geometry, or unknown classes.
- `object_bin_pose_report.json`: Final placement mappings showing exactly which bin each object belongs to.
- `planner_trace_*.jsonl`: Detailed event logs recording the FSM's state transitions. Includes scenarios for perfect success, single grasp failures, dropped objects, and unrecoverable max-retries.

## Integration Note for Motion Team (Person 3)
The Motion Primitive module MUST adhere to the schema defined in `schemas/action_plan_schema.json`. Every field is required. Person 3 is expected to return a `PrimitiveStatus` (e.g., `SUCCESS`, `GRASP_FAIL`, `COLLISION`) upon completion of a move so the FSM knows how to proceed.
