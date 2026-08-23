import pytest

from shopping_grpo.training.grpo.curriculum.state import CurriculumState, GroupOutcome
from shopping_grpo.training.grpo.failure_types import normalize_failure_type


@pytest.mark.parametrize(("reward_type", "expected"), [
    ("gold_purchase", "strict_success"),
    ("valid_alternative_purchase", "near_miss"),
    ("partial_alternative_purchase", "near_miss"),
    ("wrong_purchase", "wrong_commit"),
    ("graceful_stop", "premature_stop"),
    ("early_abstain", "premature_stop"),
    ("repeat_loop", "exploration_stall"),
    ("max_steps", "exploration_stall"),
])
def test_failure_mapping(reward_type, expected):
    assert normalize_failure_type({"reward_type": reward_type, "reward_valid": True}).value == expected


def group(task_id=7, utilities=(0.0, 0.0), failure="exploration_stall", valid=True):
    return GroupOutcome(task_id, "hard", tuple(utilities), tuple([failure] * len(utilities)), valid)


def test_invalid_group_does_not_update_capability_statistics():
    state = CurriculumState.empty("abc", "def", 10)
    state.update_group(group(valid=False), 0)
    assert state.invalid_group_count == 1
    assert state.cells == {}


def test_zero_variance_patience_resets_on_variance():
    state = CurriculumState.empty("abc", "def", 10)
    for step in range(4):
        state.update_group(group(), step)
    assert state.tasks[7].consecutive_zero_variance == 4
    state.update_group(group(utilities=(0.0, 1.0), failure="near_miss"), 4)
    assert state.tasks[7].consecutive_zero_variance == 0


def test_checkpoint_roundtrip_and_hash_guard():
    original = CurriculumState.empty("abc", "def", 10)
    original.update_group(group(utilities=(0.0, 1.0), failure="near_miss"), 1)
    restored = CurriculumState.from_checkpoint(original.to_checkpoint(), "abc", "def", 10)
    assert restored == original
    with pytest.raises(ValueError, match="manifest hash"):
        CurriculumState.from_checkpoint(original.to_checkpoint(), "other", "def", 10)
