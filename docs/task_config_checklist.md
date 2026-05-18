# Task Configuration Checklist

## Task 1: Conveyor Sorting

| Configuration | Details |
|---|---|
| **Task name** | Conveyor Sorting |
| **Config file** | `Ubtech_sim/config/Part_Sorting.yaml` |
| **Scene file** | `assets/resources/Collected_Task4/SubUSDs/2_small_warehouse2.usd` |
| **Robot config** | `assets/resources/Collected_s2_v1_ecbg/s2_v1.usd` |
| **Robot root** | `/Root/walker_s2` (Position: [0.7, -0.2, 0.9], Rotation: [0, 0, 90]°) |
| **Camera** | RGB-D via `dummy_camera_top` and `/Replicator/SDGPipeline` |
| **Target objects** | 4 workpieces (2 types, 2 each): Part A & Part B. Random spawn position and pose on table. Requires object class and pose/yaw recognition |
| **Goal area** | Box location: [1.2, 0.3, 1.05], Plane area: [0.75, 0.28, 1.04] |
| **Controller/policy** | *To be configured* |
| **Logging** | DataLogger available in `Ubtech_sim/source/DataLogger.py` |
| **Known issues** | Box locked in place (lock_boxes: true), Time limit: 100 seconds |

### Configuration Parameters
- **Root path**: `../../assets/resources/`
- **Task number**: 1
- **Time limit**: 100 seconds
- **Parts per category**: 2
- **Grasp target index**: 0
- **Lift height**: 0.17
- **Settle time**: 2.0 seconds
- **Approach time**: 3.0 seconds

### Asset Details
- **Part A assets**: 
  - `Collected_Task1_PartA_ori_color/Task1_PartA.usd`
  - `Collected_Task1_PartA_red/Task1_PartA.usd`
- **Part B assets**:
  - `Collected_Part_B_blue/Part_B.usd`
  - `Collected_Part_B_ori_color/Part_B.usd`
- **Table asset**: `Collected_table_v2/table_v2.usd` at [0.75, 0.3, 0.5]
- **Box asset**: `Box_blank/box_60_40_23_cut_0.usd` at [1.2, 0.3, 1.05]

### Robot Kinematics
- Available inverse kinematics solver: `DualArmIK` in `Ubtech_sim/source/DualArmIK.py`
- Robot articulation handler: `RobotArticulation.py`
- Grasp planner available: `grasp_planner.py`

---

## Notes
- [ ] Verify camera configuration
- [ ] Define controller/policy implementation
- [ ] Test robot movements in simulation
- [ ] Validate target object detection
- [ ] Confirm goal area boundaries
