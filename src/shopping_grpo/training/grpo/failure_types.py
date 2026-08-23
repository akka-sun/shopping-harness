"""Normalize Reward v3 outcomes into scheduler-only failure categories."""

from enum import Enum


class FailureType(str, Enum):
    STRICT_SUCCESS = "strict_success"
    NEAR_MISS = "near_miss"
    WRONG_COMMIT = "wrong_commit"
    PREMATURE_STOP = "premature_stop"
    EXPLORATION_STALL = "exploration_stall"
    INVALID = "invalid"
    UNKNOWN_VALID = "unknown_valid"


_MAPPING = {
    "gold_purchase": FailureType.STRICT_SUCCESS,
    "valid_alternative_purchase": FailureType.NEAR_MISS,
    "partial_alternative_purchase": FailureType.NEAR_MISS,
    "wrong_purchase": FailureType.WRONG_COMMIT,
    "graceful_stop": FailureType.PREMATURE_STOP,
    "early_abstain": FailureType.PREMATURE_STOP,
    "repeat_loop": FailureType.EXPLORATION_STALL,
    "max_steps": FailureType.EXPLORATION_STALL,
}


def normalize_failure_type(info: dict) -> FailureType:
    if info.get("infrastructure_invalid") or info.get("reward_valid") is False:
        return FailureType.INVALID
    return _MAPPING.get(str(info.get("reward_type") or ""), FailureType.UNKNOWN_VALID)

