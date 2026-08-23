"""Frozen v1 online learnability priority."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PrioritySignals:
    group_variance: float
    strict_success_rate: float
    near_miss_rate: float
    wrong_commit_rate: float
    steps_since_update: int
    probe_difficulty: float
    exploration_stall_rate: float = 0.0
    consecutive_zero_variance: int = 0


def compute_learnability(signals: PrioritySignals) -> float:
    v = max(float(signals.group_variance), 0.0)
    p = min(max(float(signals.strict_success_rate), 0.0), 1.0)
    variance_signal = v / (v + 0.05)
    success_uncertainty = 4.0 * p * (1.0 - p)
    recoverability = min(max(signals.near_miss_rate + 0.75 * signals.wrong_commit_rate, 0.0), 1.0)
    staleness = min(max(signals.steps_since_update / 25.0, 0.0), 1.0)
    difficulty = min(max(float(signals.probe_difficulty), 0.0), 1.0)
    value = (
        0.30 * variance_signal
        + 0.25 * success_uncertainty
        + 0.25 * recoverability
        + 0.10 * staleness
        + 0.10 * difficulty
    )
    if not math.isfinite(value):
        return float("nan")
    return value


def compute_task_priority(signals: PrioritySignals, minimum=0.05, maximum=4.0) -> float:
    priority = compute_learnability(signals)
    priority *= 1.0 - 0.75 * min(max(signals.strict_success_rate, 0.0), 1.0) ** 2
    if signals.exploration_stall_rate > 0 and signals.near_miss_rate == 0:
        priority *= max(0.35, 1.0 - 0.65 * signals.exploration_stall_rate)
    if signals.consecutive_zero_variance >= 4:
        priority *= 0.25
    return min(max(priority, float(minimum)), float(maximum))

