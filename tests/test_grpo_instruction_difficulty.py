from shopping_grpo.training.grpo.difficulty.instruction import (
    INSTRUCTION_SCORER_VERSION,
    score_public_instruction,
)


def test_instruction_score_is_deterministic_and_bounded():
    text = "Buy a Sony WH-1000XM5 under $350 in black, compatible with Android."
    first = score_public_instruction(text)
    second = score_public_instruction(text)
    assert first == second
    assert 0.0 <= first.score <= 1.0
    assert first.scorer_version == INSTRUCTION_SCORER_VERSION
    assert first.features["has_budget"] is True
    assert first.features["brand_model_constraints"] >= 1
    assert first.features["compatibility_constraints"] == 1
    assert text not in repr(first.features)


def test_instruction_length_is_only_a_weak_capped_feature():
    simple = score_public_instruction("Buy a red mug.")
    padded = score_public_instruction("Buy a red mug. " + "Please consider carefully. " * 100)
    constrained = score_public_instruction(
        "Buy a red Contigo mug under $25, 16 oz, dishwasher safe, compatible with car holders."
    )
    assert padded.score - simple.score <= 0.05
    assert constrained.score > padded.score


def test_empty_instruction_is_rejected():
    try:
        score_public_instruction("  ")
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("empty instructions must fail")
