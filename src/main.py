import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Planner functions from Person 2
from src.task1.planner import (
    parse_perception_frame,
    validate_objects,
    map_bin,
    build_action_plan
)

def main():
    # 1. Load Perception Input (from Person 1)
    perception_file = Path("src/task1/perception_interface.json")
    if not perception_file.exists():
        print(f"Error: {perception_file} not found.")
        return
        
    with open(perception_file, "r") as f:
        perception_frame = json.load(f)

    # 2. Parse and Validate Objects
    objects = parse_perception_frame(perception_frame)
    accepted, rejected = validate_objects(objects)
    
    # 3. Generate ActionPlans for Person 3
    action_plans = []
    
    for obj in accepted:
        bin_state = map_bin(obj.class_id)
        if bin_state:
            plan = build_action_plan(obj, bin_state)
            action_plans.append(plan.to_dict())
            
    # 4. Output the exact JSON format for Person 3
    
    output_path = Path("lab_outputs/planner_outputs/action_plans_for_person3.json")
    with open(output_path, "w") as f:
        json.dump(action_plans, f, indent=2)
        
    print(f"Successfully generated {len(action_plans)} ActionPlans.")
    print(f"Saved to: {output_path.absolute()}")

if __name__ == "__main__":
    main()
