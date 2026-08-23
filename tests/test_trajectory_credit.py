import pytest
import torch

from shopping_grpo.training.grpo.evidence import EvidenceEvent
from shopping_grpo.training.grpo.trajectory_credit import (
    CreditConfig,
    TurnSpan,
    compute_turn_multipliers,
    project_token_advantages,
)


def event(turn, *, positive=False, repeat=False):
    return EvidenceEvent(turn, True, positive, int(positive), False, False, False, False, repeat, repeat)


def test_advantage_sign_reverses_responsibility_direction():
    events = [event(0, positive=True), event(1, repeat=True)]
    spans = [TurnSpan(0, 0, 2), TurnSpan(1, 2, 4)]
    positive = compute_turn_multipliers(1.0, events, spans, CreditConfig())
    negative = compute_turn_multipliers(-1.0, events, spans, CreditConfig())
    assert positive.multipliers[0] > positive.multipliers[1]
    assert negative.multipliers[0] < negative.multipliers[1]


def test_projection_preserves_mass_and_non_assistant_zero():
    result = project_token_advantages(
        2.0,
        torch.tensor([1, 1, 0, 1, 1, 0]),
        [TurnSpan(0, 0, 2), TurnSpan(1, 3, 5)],
        [event(0, positive=True), event(1, repeat=True)],
        CreditConfig(),
    )
    assert result.token_advantages[2].item() == 0
    assert result.token_advantages[5].item() == 0
    assert result.token_advantages.sum().item() == pytest.approx(8.0, abs=1e-6)
    assert min(result.multipliers) >= 0.5
    assert max(result.multipliers) <= 1.5


def test_compaction_falls_back_for_whole_trajectory():
    result = project_token_advantages(
        1.0, torch.ones(2), [TurnSpan(0, 0, 2)], [event(0)], CreditConfig(), compacted=True
    )
    assert result.fallback_reason == "context_compaction_alignment_unsupported"
    assert result.multipliers == (1.0,)
