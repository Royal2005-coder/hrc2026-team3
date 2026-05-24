from dataclasses import dataclass
from typing import Dict
from src.task1.state import PrimitiveStatus


SPEED_FACTORS = {
    "normal":    1.0,    # full speed
    "slow":      0.6,    # 60% speed — safer grasp approach
    "very_slow": 0.3,    # 30% speed — max care, e.g. after DROP
}


@dataclass
class RetryPolicy:
    """
    Rules for how to handle a specific PrimitiveStatus failure.

    Attributes:
        status:       which failure this policy applies to
        retry:        whether retrying is allowed at all
        max_attempts: total tries (1 = no retry, 2 = 1 retry, 3 = 2 retries)
        speed:        speed label ("normal" / "slow" / "very_slow")
        timeout_s:    seconds per attempt on retry
        on_exceed:    what to do after max_attempts ("skip" or "abort")
    """
    status:       PrimitiveStatus
    retry:        bool
    max_attempts: int
    speed:        str
    timeout_s:    float
    on_exceed:    str   # "skip" or "abort"



POLICIES: Dict[PrimitiveStatus, RetryPolicy] = {

    PrimitiveStatus.GRASP_FAIL: RetryPolicy(
        status=PrimitiveStatus.GRASP_FAIL,
        retry=True,
        max_attempts=3,     # 1 original + 2 retries
        speed="slow",
        timeout_s=25.0,
        on_exceed="skip"
    ),

    PrimitiveStatus.DROP: RetryPolicy(
        status=PrimitiveStatus.DROP,
        retry=True,
        max_attempts=2,     # 1 original + 1 retry
        speed="very_slow",
        timeout_s=30.0,
        on_exceed="skip"
    ),

    PrimitiveStatus.TIMEOUT: RetryPolicy(
        status=PrimitiveStatus.TIMEOUT,
        retry=True,
        max_attempts=2,
        speed="slow",
        timeout_s=35.0,
        on_exceed="abort"
    ),

    PrimitiveStatus.WRONG_BIN: RetryPolicy(
        status=PrimitiveStatus.WRONG_BIN,
        retry=False,
        max_attempts=1,
        speed="normal",
        timeout_s=20.0,
        on_exceed="abort"
    ),

    PrimitiveStatus.COLLISION: RetryPolicy(
        status=PrimitiveStatus.COLLISION,
        retry=False,
        max_attempts=1,
        speed="normal",
        timeout_s=20.0,
        on_exceed="abort"
    ),
}


@dataclass
class RetryDecision:
    """
    Answer from RetryManager: should FSM retry or give up?

    Attributes:
        allowed:       True  → FSM should go to RETRY → PICKING
                       False → FSM should go to FAIL
        attempt_num:   which attempt this is (2 = first retry, 3 = second)
        speed_factor:  motion speed multiplier to pass to Person 3
        timeout_s:     timeout for this attempt
        on_exceed:     "skip" or "abort" (used when allowed=False)
        reason:        human-readable explanation
    """
    allowed:      bool
    attempt_num:  int
    speed_factor: float
    timeout_s:    float
    on_exceed:    str
    reason:       str

class RetryManager:
    """
    Tracks retry counts per object and applies retry policies.

    One RetryManager instance lives inside the FSM for the whole episode.
    Call reset_object() when starting a new object.
    """

    def __init__(self):
        # attempt_counts[object_id] = number of attempts so far
        self._attempt_counts: Dict[str, int] = {}

    def reset_object(self, object_id: str):
        """Reset counter when starting a fresh object."""
        self._attempt_counts[object_id] = 0

    def get_attempt_count(self, object_id: str) -> int:
        return self._attempt_counts.get(object_id, 0)

    def apply_retry(
        self,
        object_id: str,
        status: PrimitiveStatus
    ) -> RetryDecision:
        """
        Decide whether to retry after a failure.

        Args:
            object_id: which object failed
            status:    what kind of failure (GRASP_FAIL, DROP, etc.)

        Returns:
            RetryDecision
        """
        policy = get_policy(status)

        # Increment attempt count
        current = self._attempt_counts.get(object_id, 0) + 1
        self._attempt_counts[object_id] = current

        speed_factor = get_speed_factor(policy.speed)

        # Case 1: policy says never retry
        if not policy.retry:
            return RetryDecision(
                allowed=False,
                attempt_num=current,
                speed_factor=speed_factor,
                timeout_s=policy.timeout_s,
                on_exceed=policy.on_exceed,
                reason=f"{status.value} is not retryable → {policy.on_exceed}"
            )

        # Case 2: exceeded max attempts
        if current >= policy.max_attempts:
            return RetryDecision(
                allowed=False,
                attempt_num=current,
                speed_factor=speed_factor,
                timeout_s=policy.timeout_s,
                on_exceed=policy.on_exceed,
                reason=f"max attempts {policy.max_attempts} exceeded → {policy.on_exceed}"
            )

        # Case 3: retry allowed
        return RetryDecision(
            allowed=True,
            attempt_num=current + 1,
            speed_factor=speed_factor,
            timeout_s=policy.timeout_s,
            on_exceed=policy.on_exceed,
            reason=f"retry {current}/{policy.max_attempts - 1} at speed={policy.speed}"
        )


def get_policy(status: PrimitiveStatus) -> RetryPolicy:
    """
    Look up retry policy for a given PrimitiveStatus.
    Falls back to no-retry policy if status not found.
    """
    return POLICIES.get(status, RetryPolicy(
        status=status,
        retry=False,
        max_attempts=1,
        speed="normal",
        timeout_s=20.0,
        on_exceed="abort"
    ))


def get_speed_factor(speed_label: str) -> float:
    """
    Convert speed label → float multiplier.

    Example:
        get_speed_factor("slow") → 0.6
        get_speed_factor("very_slow") → 0.3
    """
    return SPEED_FACTORS.get(speed_label, 1.0)
