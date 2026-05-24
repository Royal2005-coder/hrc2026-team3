from typing import Callable, Optional, List
from src.task1.state import (
    ObjectState, ActionPlan, PrimitiveResult,
    ObjectStatus, PrimitiveStatus, FSMState
)
from src.task1.planner import (
    parse_perception_frame, validate_objects,
    select_object, map_bin, build_action_plan
)
from src.task1.retry_policy import RetryManager


class Task1FSM:
    """
    Finite State Machine for Task 1 pick-place cycle.

    Attributes:
        state:        current FSMState
        step_id:      increments every transition (for logging)
        trace:        list of all log entries (sent to Person 4)
        retry_count:  retries for current object
        current_obj:  object currently being picked
        current_plan: ActionPlan currently being executed
    """

    def __init__(self):
        self.state:        FSMState          = FSMState.IDLE
        self.step_id:      int               = 0
        self.trace:        List[dict]        = []
        self.retry_count:  int               = 0
        self.current_obj:  Optional[ObjectState]  = None
        self.current_plan: Optional[ActionPlan]   = None
        self._pending_objects: List[ObjectState]  = []
        self.retry_manager: RetryManager          = RetryManager()

    def _transition(self, new_state: FSMState, reason: str = ""):
        """Log and perform a state transition."""
        self.step_id += 1
        entry = {
            "step_id":    self.step_id,
            "from_state": self.state.value,
            "to_state":   new_state.value,
            "reason":     reason,
        }
        if self.current_obj:
            entry["object_id"] = self.current_obj.object_id
        self.trace.append(entry)
        print(f"  [FSM] {self.state.value:15s} → {new_state.value:15s}  | {reason}")
        self.state = new_state

    def _log_event(self, event: str, detail: dict = None):
        """Log a non-transition event."""
        entry = {"step_id": self.step_id, "event": event}
        if detail:
            entry.update(detail)
        self.trace.append(entry)

    def _handle_idle(self, frame: dict):
        self._transition(FSMState.VALIDATE, "frame received")
        self._handle_validate(frame)

    def _handle_validate(self, frame: dict):
        objects = parse_perception_frame(frame)
        accepted, rejected = validate_objects(objects)

        for r in rejected:
            self._log_event("object_rejected", {
                "object_id": r.object_id,
                "reason": r.failure_reason
            })
            print(f"         ✗ {r.object_id} rejected → {r.failure_reason}")

        if not accepted:
            self._transition(FSMState.DONE, "no valid objects in frame")
            return

        self._pending_objects = accepted
        self._transition(FSMState.SELECT_OBJECT, f"{len(accepted)} valid objects found")

    def _handle_select(self):
        obj = select_object(self._pending_objects, strategy="highest_confidence")

        if obj is None:
            self._transition(FSMState.DONE, "no objects remaining")
            return

        self.current_obj  = obj
        self.retry_count  = 0
        self.retry_manager.reset_object(obj.object_id)

        self._pending_objects = [
            o for o in self._pending_objects
            if o.object_id != obj.object_id
        ]

        self._log_event("object_selected", {
            "object_id":  obj.object_id,
            "class_id":   obj.class_id,
            "confidence": obj.confidence
        })
        print(f"         ✓ selected {obj.object_id} ({obj.class_id}) conf={obj.confidence}")
        self._transition(FSMState.PLAN, f"selected {obj.object_id}")

    def _handle_plan(self):
        bin_state = map_bin(self.current_obj.class_id)

        if bin_state is None:
            self._log_event("bin_not_found", {"class_id": self.current_obj.class_id})
            self._transition(FSMState.FAIL, f"no bin for {self.current_obj.class_id}")
            return

        self.current_plan = build_action_plan(self.current_obj, bin_state)

        self._log_event("plan_built", {
            "plan_id":    self.current_plan.plan_id,
            "target_bin": self.current_plan.target_bin
        })
        print(f"         ✓ plan={self.current_plan.plan_id}  bin={self.current_plan.target_bin}")
        self._transition(FSMState.PICKING, "plan ready")

    def _handle_picking(self, mock_person3: Callable):
        """
        Send ActionPlan to Person 3 and wait for PrimitiveResult.
        In real system: publish to ROS topic / socket.
        Here: call mock_person3(plan) → PrimitiveResult
        """
        if self.retry_count > 0:
            print(f"         ↺ retry #{self.retry_count} (slow speed)")

        result: PrimitiveResult = mock_person3(self.current_plan, self.retry_count)
        print(f"         Person3 → {result.status.value}")

        if result.status == PrimitiveStatus.SUCCESS:
            self.current_obj.status = ObjectStatus.PICKED
            self._transition(FSMState.PLACING, "grasp success")

        elif result.status == PrimitiveStatus.GRASP_FAIL:
            self._handle_retry(PrimitiveStatus.GRASP_FAIL)

        elif result.status == PrimitiveStatus.TIMEOUT:
            self._handle_retry(PrimitiveStatus.TIMEOUT)

        elif result.status == PrimitiveStatus.COLLISION:
            self._handle_retry(PrimitiveStatus.COLLISION)

        else:
            self._handle_retry(result.status)

    def _handle_placing(self, mock_person3: Callable):
        result: PrimitiveResult = mock_person3(self.current_plan, self.retry_count)
        print(f"         Person3 → {result.status.value}")

        if result.status == PrimitiveStatus.SUCCESS:
            self.current_obj.status = ObjectStatus.PLACED
            self._transition(FSMState.VERIFYING, "place success")

        elif result.status == PrimitiveStatus.DROP:
            self._handle_retry(PrimitiveStatus.DROP)

        elif result.status == PrimitiveStatus.WRONG_BIN:
            self._handle_retry(PrimitiveStatus.WRONG_BIN)

        else:
            self._handle_retry(result.status)

    def _handle_verifying(self):
        """
        In real system: ask Person 1 to re-check bin.
        Here: trust the SUCCESS result from placing.
        """
        self._log_event("verify_ok", {"object_id": self.current_obj.object_id})
        print(f"         ✓ {self.current_obj.object_id} verified in {self.current_plan.target_bin}")

        if self._pending_objects:
            self._transition(FSMState.SELECT_OBJECT, "next object")
        else:
            self._transition(FSMState.DONE, "all objects placed")

    def _handle_retry(self, status: PrimitiveStatus):
        decision = self.retry_manager.apply_retry(self.current_obj.object_id, status)
        
        if not decision.allowed:
            self.current_obj.status = ObjectStatus.FAILED
            self.current_obj.failure_reason = decision.reason
            self._log_event("retry_exceeded_or_denied", {
                "object_id": self.current_obj.object_id,
                "reason":    decision.reason
            })
            self._transition(FSMState.FAIL, f"retry denied: {decision.reason}")
        else:
            self.retry_count = decision.attempt_num - 1
            self._log_event("retry_triggered", {
                "object_id":   self.current_obj.object_id,
                "attempt":     decision.attempt_num,
                "speed":       decision.speed_factor,
                "reason":      decision.reason
            })
            self._transition(FSMState.RETRY, f"retry allowed: {decision.reason}")
            self._transition(FSMState.PICKING, "retrying pick")


    def run(self, frame: dict, mock_person3: Callable):
        """
        Run the full FSM cycle for one PerceptionFrame.

        Args:
            frame:        PerceptionFrame dict from Person 1
            mock_person3: fn(ActionPlan, retry_count) → PrimitiveResult
        """
        print("\n" + "=" * 55)
        print("FSM START")
        print("=" * 55)

        self._handle_idle(frame)

        max_steps = 50  # safety limit
        steps = 0

        while self.state not in (FSMState.DONE, FSMState.FAIL):
            steps += 1
            if steps > max_steps:
                print("  [FSM] Safety limit reached — stopping")
                break

            if self.state == FSMState.SELECT_OBJECT:
                self._handle_select()

            elif self.state == FSMState.PLAN:
                self._handle_plan()

            elif self.state == FSMState.PICKING:
                self._handle_picking(mock_person3)

            elif self.state == FSMState.PLACING:
                self._handle_placing(mock_person3)

            elif self.state == FSMState.VERIFYING:
                self._handle_verifying()

            else:
                self._transition(FSMState.FAIL, f"unhandled state: {self.state.value}")
                break

        print("=" * 55)
        print(f"FSM END  → {self.state.value}")
        print("=" * 55)

    def get_trace(self) -> List[dict]:
        return self.trace


def make_mock_person3(scenario: str = "all_success"):
    """
    Returns a mock_person3 function simulating different scenarios.

    Scenarios:
      "all_success"        → every pick-place succeeds
      "first_grasp_fail"   → obj_001 fails once, then succeeds on retry
      "exceed_retries"     → obj_001 always fails (exceeds max retries)
      "drop_on_place"      → obj_001 drops during placing, retries and succeeds
    """
    call_counts = {}

    def mock_person3(plan: ActionPlan, retry_count: int) -> PrimitiveResult:
        oid = plan.object_id
        call_counts[oid] = call_counts.get(oid, 0) + 1
        count = call_counts[oid]

        if scenario == "all_success":
            return PrimitiveResult(plan.plan_id, oid, PrimitiveStatus.SUCCESS)

        elif scenario == "first_grasp_fail":
            if oid == "obj_001" and count == 1:
                return PrimitiveResult(plan.plan_id, oid, PrimitiveStatus.GRASP_FAIL,
                                       failure_reason="gripper slipped")
            return PrimitiveResult(plan.plan_id, oid, PrimitiveStatus.SUCCESS)

        elif scenario == "exceed_retries":
            if oid == "obj_001":
                return PrimitiveResult(plan.plan_id, oid, PrimitiveStatus.GRASP_FAIL,
                                       failure_reason="always fails")
            return PrimitiveResult(plan.plan_id, oid, PrimitiveStatus.SUCCESS)

        elif scenario == "drop_on_place":
            if oid == "obj_001" and count == 2:
                return PrimitiveResult(plan.plan_id, oid, PrimitiveStatus.DROP,
                                       failure_reason="dropped mid-air")
            return PrimitiveResult(plan.plan_id, oid, PrimitiveStatus.SUCCESS)

        return PrimitiveResult(plan.plan_id, oid, PrimitiveStatus.SUCCESS)

    return mock_person3
