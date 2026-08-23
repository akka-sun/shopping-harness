from shopping_grpo.training.grpo.curriculum.sampler import (
    CurriculumSampler,
    SamplerConfig,
    TaskCandidate,
    phase_for_step,
)
from shopping_grpo.training.grpo.curriculum.state import CurriculumState, GroupOutcome
from shopping_grpo.training.grpo.evidence import EvidenceEvent
from shopping_grpo.training.grpo.trajectory_credit import (
    CreditConfig,
    TurnSpan,
    compute_turn_multipliers,
)


def _event(turn, *, positive=False, stalled=False):
    return EvidenceEvent(
        turn, True, positive, int(positive), False, False, False,
        False, stalled, stalled,
    )


def test_single_run_switch_credit_and_resume_contract():
    config = SamplerConfig(seed=17, total_steps=10, switch_ratio=.30, uniform_ratio=.20)
    tasks = [
        TaskCandidate(i, category, i / 10, 1.0 + i / 10)
        for i, category in zip(range(1, 7), ("easy", "easy", "medium", "medium", "hard", "hard"))
    ]
    sampler = CurriculumSampler(tasks, config)
    assert [phase_for_step(i, 10, .30) for i in (0, 2, 3, 9)] == [
        "baseline_30", "baseline_30", "difficulty_70", "difficulty_70"
    ]
    assert sampler.select(2, 0, 2, lambda _: [6, 5]).task_ids == (6, 5)

    state = CurriculumState.empty("manifest", "config", 10)
    state.update_group(GroupOutcome(3, "medium", (0.0, .6), ("wrong_commit", "near_miss")), 2)
    state.update_group(GroupOutcome(5, "hard", (0.0, 0.0), ("exploration_stall",) * 2), 2)
    restored = CurriculumState.from_checkpoint(state.to_checkpoint(), "manifest", "config", 10)
    assert restored == state
    selected = sampler.select(3, 0, 3, lambda _: [])
    replayed = CurriculumSampler(tasks, config).select(3, 0, 3, lambda _: [])
    assert selected.task_ids == replayed.task_ids
    assert len(set(selected.task_ids)) == 3

    credit = compute_turn_multipliers(
        1.0,
        [_event(0, positive=True), _event(1, stalled=True)],
        [TurnSpan(0, 0, 2), TurnSpan(1, 2, 4)],
        CreditConfig(),
    )
    assert credit.fallback_reason is None
    assert .5 <= min(credit.multipliers) < 1 < max(credit.multipliers) <= 1.5
    assert sum(value * 2 for value in credit.multipliers) == 4.0
