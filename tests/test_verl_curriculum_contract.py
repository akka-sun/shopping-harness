import json
from pathlib import Path

import numpy as np
import pytest
import torch
from verl import DataProto

from shopping_grpo.training.grpo.curriculum.state import CurriculumState
from shopping_grpo.training.grpo.dynamic_sampling import (
    apply_evidence_credit,
    load_curriculum_sidecar,
    save_curriculum_sidecar,
    sanitize_curriculum_diagnostic,
)


PATCH = Path(__file__).resolve().parents[1] / "patches/verl-0.8.0-shopping-dynamic-sampling.patch"


def _batch(advantage=1.0):
    return DataProto.from_dict(
        tensors={
            "advantages": torch.tensor([[advantage, advantage, 0.0, 0.0]]),
            "response_mask": torch.tensor([[1, 1, 0, 0]]),
        },
        non_tensors={
            "shopping": np.array([{
                "evidence_events": [{
                    "assistant_turn_id": 0, "action_attempted": True,
                    "new_search_result_set": True, "new_product_count": 1,
                    "first_product_open": False, "first_info_subpage": False,
                    "first_option_confirmation": False, "guard_rejection": False,
                    "exact_repeat": False, "no_progress": False,
                    "schema_version": "shopping-evidence-v1",
                }],
                "assistant_turn_spans": [{"assistant_turn_id": 0, "start": 0, "end": 2}],
                "credit_alignment_valid": True,
            }], dtype=object),
        },
    )


def test_disabled_credit_preserves_tensor_identity_and_values():
    batch = _batch()
    original = batch.batch["advantages"]
    result = apply_evidence_credit(batch, {"enable": False})
    assert result["applied"] == 0
    assert batch.batch["advantages"] is original


def test_zero_advantage_stays_zero():
    batch = _batch(0.0)
    apply_evidence_credit(batch, {"enable": True})
    assert torch.equal(batch.batch["advantages"], torch.zeros((1, 4)))


def test_curriculum_sidecar_roundtrip_and_required_missing(tmp_path):
    state = CurriculumState.empty("manifest", "config", 10)
    state.committed_global_step = 3
    save_curriculum_sidecar(tmp_path, state)
    restored = load_curriculum_sidecar(tmp_path, "manifest", "config", 10, required=True)
    assert restored == state
    with pytest.raises(FileNotFoundError):
        load_curriculum_sidecar(tmp_path / "missing", "manifest", "config", 10, required=True)


def test_curriculum_diagnostic_drops_sensitive_nested_fields():
    safe = sanitize_curriculum_diagnostic({
        "stage": "difficulty_70", "task_id": 7, "actions": [{"asin": "secret"}],
        "raw_observation": "secret", "gold_purchase": {"asin": "secret"},
        "fallback_reasons": {"bad_span": 1},
    })
    encoded = json.dumps(safe)
    assert safe["stage"] == "difficulty_70"
    assert "secret" not in encoded
    assert "actions" not in safe


def test_patch_uses_thin_project_calls_and_filter_precedes_credit():
    patch = PATCH.read_text(encoding="utf-8")
    for symbol in (
        "CurriculumSampler", "update_group", "project_token_advantages",
        "to_checkpoint", "from_checkpoint",
    ):
        assert symbol in patch
    assert "0.30 * variance_signal" not in patch
    assert "near_miss_rate +" not in patch
    assert patch.index("select_reward_varying_groups(") < patch.index("project_token_advantages(")
