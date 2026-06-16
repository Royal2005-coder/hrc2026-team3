from src.task1.state import PrimitiveStatus
from src.task1.retry_policy import RetryManager

def test_retry_policy_manual():
    print("=" * 55)
    print("RETRY POLICY — MANUAL CHECK")
    print("=" * 55)

    manager = RetryManager()

    tests = [
        # (object_id, status, description)
        ("obj_001", PrimitiveStatus.GRASP_FAIL, "1st grasp fail"),
        ("obj_001", PrimitiveStatus.GRASP_FAIL, "2nd grasp fail"),
        ("obj_001", PrimitiveStatus.GRASP_FAIL, "3rd grasp fail → exceeds max"),
        ("obj_002", PrimitiveStatus.DROP,       "drop → retry very slow"),
        ("obj_002", PrimitiveStatus.DROP,       "drop again → exceeds max"),
        ("obj_003", PrimitiveStatus.COLLISION,  "collision → never retry"),
        ("obj_004", PrimitiveStatus.WRONG_BIN,  "wrong bin → never retry"),
    ]

    for obj_id, status, desc in tests:
        decision = manager.apply_retry(obj_id, status)
        allowed  = "RETRY" if decision.allowed else "FAIL "
        print(f"\n  {desc}")
        print(f"  {allowed}  attempt={decision.attempt_num}  "
              f"speed={decision.speed_factor}x  "
              f"timeout={decision.timeout_s}s")
        print(f"           reason: {decision.reason}")
