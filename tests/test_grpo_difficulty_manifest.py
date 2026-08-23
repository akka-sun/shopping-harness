import json

import pytest

from shopping_grpo.training.grpo.difficulty.manifest import (
    ProbeCounters,
    assign_categories,
    compute_probe_scores,
    fit_train_boundaries,
    validate_task_isolation,
    write_probe_artifacts,
)


def test_probe_formula_is_exact_and_bounded():
    scores = compute_probe_scores(
        0.8, ProbeCounters(35, 10, 2, 3, 35)
    )
    assert scores.step_score == 1.0
    assert scores.repeat_ratio == 0.2
    assert scores.guard_score == 1.0
    assert scores.probe_difficulty == pytest.approx(0.79, abs=1e-8)


def test_train_boundaries_do_not_split_ties():
    train = [(1, 0.1), (2, 0.2), (3, 0.2), (4, 0.8), (5, 0.9), (6, 1.0)]
    boundaries = fit_train_boundaries(train)
    assigned = assign_categories(train, boundaries)
    assert assigned[2] == assigned[3]
    assert set(assigned.values()) == {"easy", "medium", "hard"}


def test_task_isolation_fails_closed():
    with pytest.raises(ValueError, match="overlap"):
        validate_task_isolation({1, 2}, {3}, {2, 4})


def test_artifacts_are_separate_and_hashed(tmp_path):
    paths = write_probe_artifacts(
        tmp_path / "probe-output",
        valid_records=[{"task_id": 1, "split": "train", "instruction_features": {}}],
        invalid_records=[{"task_id": 2, "split": "validation", "reason": "environment_error"}],
        metadata={"schema_version": "shopping-probe-metadata-v1"},
    )
    assert paths.manifest.name == "difficulty_manifest.jsonl"
    assert paths.invalid.name == "invalid_probes.jsonl"
    metadata = json.loads(paths.metadata.read_text())
    assert len(metadata["manifest_sha256"]) == 64
    assert "full public instruction" not in paths.manifest.read_text()
