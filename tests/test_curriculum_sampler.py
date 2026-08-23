import pytest

from shopping_grpo.training.grpo.curriculum.priority import PrioritySignals, compute_learnability
from shopping_grpo.training.grpo.curriculum.sampler import (
    CurriculumSampler,
    SamplerConfig,
    TaskCandidate,
    phase_for_step,
    switch_step,
)


def test_switch_boundary_uses_committed_step():
    assert switch_step(10, 0.30) == 3
    assert phase_for_step(2, 10, 0.30) == "baseline_30"
    assert phase_for_step(3, 10, 0.30) == "difficulty_70"


def test_priority_formula_is_frozen():
    signals = PrioritySignals(.05, .5, .2, .1, 25, .8)
    expected = .30*.5 + .25*1 + .25*.275 + .10*1 + .10*.8
    assert compute_learnability(signals) == pytest.approx(expected)


def test_selection_is_unique_and_reproducible():
    tasks = [TaskCandidate(i, "medium", i / 10, 1.0 + i) for i in range(1, 6)]
    config = SamplerConfig(seed=9)
    left = CurriculumSampler(tasks, config).select(4, 0, 3, lambda n: [9] * n)
    right = CurriculumSampler(tasks, config).select(4, 0, 3, lambda n: [9] * n)
    assert left.task_ids == right.task_ids
    assert len(set(left.task_ids)) == 3
    assert all(value > 0 for value in left.theoretical_probabilities.values())


def test_baseline_phase_delegates_without_curriculum_rng():
    tasks = [TaskCandidate(1, "easy", .1, 1.0)]
    selected = CurriculumSampler(tasks, SamplerConfig(seed=9)).select(0, 0, 2, lambda n: [8, 7])
    assert selected.task_ids == (8, 7)
    assert selected.phase == "baseline_30"
