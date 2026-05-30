# FSM Integration Guide - Person 3 Motion Execution

## Overview
Motion executor (Person 3) phối hợp với FSM qua **Primitive Events**. Khi motion execution xong, Motion phải emit event để FSM biết chuyển trạng thái tiếp theo.

---

## 1. Primitive Events (from `task1_fsm.yaml`)

### Success Flow:
- **PICKING** state → (success) → emit `PRIMITIVE_SUCCESS` → go to **PLACING**
- **PLACING** state → (success) → emit `PRIMITIVE_SUCCESS` → go to **VERIFYING**

### Failure Modes:
| Event | Triggered By | FSM Next State |
|-------|-------------|----------------|
| `PRIMITIVE_SUCCESS` | Motion completed all 8 stages | PLACING or VERIFYING |
| `PRIMITIVE_GRASP_FAIL` | Gripper closed but no object sensor signal | RETRY |
| `PRIMITIVE_DROP` | Object detected falling during lift/transport | RETRY |
| `PRIMITIVE_TIMEOUT` | Motion exceeded 20 seconds | RETRY |
| `PRIMITIVE_COLLISION` | Overcurrent or force sensor exceeded | FAIL (abort) |
| `PRIMITIVE_WRONG_BIN` | Placed object in wrong bin (camera confirms) | FAIL (abort) |

---

## 2. Event Emission in Code

Motion code automatically emits events based on execution results:

```python
from src.task1.motion import PrimitiveEvent, emit_primitive_event

# Example 1: Success
emit_primitive_event(
    PrimitiveEvent.SUCCESS,
    plan_id="plan_0001",
    object_id="obj_000",
    details={"execution_time_s": 5.2}
)

# Example 2: Timeout
emit_primitive_event(
    PrimitiveEvent.TIMEOUT,
    plan_id="plan_0001",
    object_id="obj_000",
    details={
        "stage_failed": "pre-grasp",
        "error_msg": "IK control timed out at pre-grasp stage"
    }
)

# Example 3: Grasp fail
emit_primitive_event(
    PrimitiveEvent.GRASP_FAIL,
    plan_id="plan_0001",
    object_id="obj_000",
    details={"error_msg": "Gripper closed but force sensor shows no object"}
)
```

---

## 3. Retry Policy

When `PRIMITIVE_GRASP_FAIL`, `PRIMITIVE_DROP`, or `PRIMITIVE_TIMEOUT` occurs:
- FSM transitions to **RETRY** state
- Motion receives **apply_retry_policy()** call
- Motion should:
  1. Check if retry_count < max_retries (default: 2)
  2. If allowed: reduce speed to "slow" mode for careful retry
  3. If exceeded: FSM transitions to **FAIL** state

```python
from src.task1.motion import apply_retry_policy

# On retry attempt
retry_result = apply_retry_policy(retry_count=1)

if retry_result["allowed"]:
    print(f"Retrying with speed: {retry_result['speed']}")
    # Retry motion with slower speed
else:
    print("Max retries exceeded - system will go to FAIL")
```

---

## 4. Workspace Validation

Motion executor validates ALL waypoints BEFORE execution:
- Check if object_pose_base is within workspace bounds
- Check if bin_pose_base is within workspace bounds  
- Check if ALL 8 intermediate waypoints are reachable

If ANY waypoint is out of bounds:
- Reject plan immediately
- Emit `PRIMITIVE_COLLISION` event (to prevent hardware damage)
- FSM transitions to **FAIL**

Workspace bounds (from `thu/configs/planner.yaml`):
```yaml
x: [-0.7, -0.4]  meters
y: [-0.4, 0.4]   meters
z: [0.5, 1.0]    meters
```

---

## 5. 8-Stage Execution Sequence

Each pick-place operation has 8 stages:

```
1. pre-grasp  → approach from above (open gripper)
2. grasp      → at object surface (open gripper)
3. post-grasp → same as grasp (close gripper)
4. lift       → move up by lift_height (closed gripper)
5. pre-place  → approach bin from above (closed gripper)
6. place      → at bin surface (closed gripper)
7. release    → same as bin (open gripper)
8. post-place → retract from bin (open gripper)
```

**If ANY stage times out/fails:**
- Stage name is recorded
- Event is emitted with error details
- FSM decides next action (usually RETRY or FAIL)

---

## 6. Current Implementation Status ✓

### Implemented:
- ✓ Load action plans from Person 2
- ✓ Workspace bounds validation (position check)
- ✓ Trajectory waypoint validation (all 8 stages)
- ✓ FSM event emission system
- ✓ Retry policy tracking
- ✓ Error classification (TIMEOUT/GRASP_FAIL/DROP/COLLISION)
- ✓ IK input conversion

### TODO (if needed):
- [ ] Gripper force sensor integration (detect GRASP_FAIL)
- [ ] Drop detection during lift/transport (detect DROP)
- [ ] Overcurrent sensor integration (detect COLLISION)
- [ ] Vision-based bin verification (detect WRONG_BIN)

---

## 7. Usage Example

```python
from src.task1.motion import (
    run_pipeline_from_person2,
    load_action_plans_from_person2,
    apply_retry_policy,
    PrimitiveEvent
)

# Main motion pipeline
if __name__ == "__main__":
    try:
        # Load all plans from Person 2
        plans = load_action_plans_from_person2()
        
        # Run motion pipeline (auto-emits events to FSM)
        run_pipeline_from_person2()
        
    except Exception as e:
        print(f"[FATAL] Motion system failed: {e}")
        print("System will abort and go to FAIL state")
```

---

## 8. FSM Workflow Summary

```
       ┌─────────┐
       │  IDLE   │
       └────┬────┘
            │ (FRAME_RECEIVED)
       ┌────▼────┐
       │VALIDATE │
       └────┬────┘
            │ (VALID_OBJECTS_FOUND)
    ┌───────▼─────────┐
    │ SELECT_OBJECT   │
    └───────┬─────────┘
            │ (OBJECT_SELECTED)
    ┌───────▼─────────┐
    │     PLAN        │
    └───────┬─────────┘
            │ (PLAN_READY)
    ┌───────▼─────────┐
    │   PICKING       │◄─────────────────┐
    │(Motion exec)    │                  │
    └───┬───┬─────────┘                  │
        │   │                            │
        │   └─── PRIMITIVE_TIMEOUT ──────┤
        │   └─── PRIMITIVE_GRASP_FAIL ──┤
        │                                │
        ├─── PRIMITIVE_SUCCESS ───► PLACING
        │                                │
        │                          ┌─────▼──────┐
        │                          │  PLACING   │
        │                          │ (Motion)   │
        │                          └─────┬──────┘
        │                                │
        │                    ┌───────────┼───────────┐
        │                    │           │           │
        │            PRIMITIVE_SUCCESS   │   PRIMITIVE_DROP
        │                    │           │           │
        │            ┌───────▼──────┐   │    │
        │            │  VERIFYING   │   │    │
        │            │ (Camera)     │   │    │
        │            └───────┬──────┘   │    │
        │                    │          │    │
        │            ┌───────┴──────┐   │    │
        │            │              │   │    │
        │       VERIFY_OK      VERIFY_FAIL  │
        │            │              │   │    │
        │       ┌────▼──┐      ┌────▼───┼───▼───┐
        │       │ IDLE  │      │  RETRY │       │
        │       └───────┘      └────┬───┴───────┘
        │                           │
        │            ┌──────────────┤
        │            │              │
        │    RETRY_ALLOWED   RETRY_EXCEEDED
        │            │              │
        │            │         ┌────▼─────┐
        │            │         │   FAIL   │
        │            │         └──────────┘
        └────────────┘
```

---

## 9. Debugging

Enable verbose logging to track FSM events:

```python
# In motion.py - check console output for:
[FSM Event] PRIMITIVE_SUCCESS for obj_000 (plan plan_0001)
[FSM Event] PRIMITIVE_TIMEOUT for obj_001 (plan plan_0002)
            Details: IK control timed out at pre-grasp stage
[Retry Policy] Attempt 1/2
               Speed: slow (careful mode)
```

---

**Version:** 1.0  
**Last Updated:** 2026-05-25  
**Person 3 Motion Executor**
