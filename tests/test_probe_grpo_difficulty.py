import json
from pathlib import Path

from scripts import probe_grpo_difficulty as probe_cli
from shopping_grpo.training.grpo.difficulty.probe import (
    DETERMINISTIC_DECODE,
    ProbeOutcome,
    ProbeRequest,
    run_probe,
)


def _write(path: Path, ids):
    path.write_text("".join(json.dumps({"task_id": item}) + "\n" for item in ids))


def test_probe_calls_each_task_once_and_separates_invalid(tmp_path):
    train, val, evaluation = (tmp_path / name for name in ("train.jsonl", "val.jsonl", "eval.jsonl"))
    _write(train, [1, 2, 4]); _write(val, [3]); _write(evaluation, [9])
    model = tmp_path / "model"; model.mkdir(); (model / "config.json").write_text("{}")
    calls = []

    def rollout_one(task_id, decode, seed, max_steps):
        calls.append(task_id)
        return ProbeOutcome(
            task_id=task_id,
            instruction=f"Buy task {task_id} under $20",
            valid=task_id != 3,
            invalid_reason=None if task_id != 3 else "reward_unverifiable",
            action_steps=task_id * 4,
            action_attempt_count=4,
            repeat_action_count=0,
            guard_count=0,
            reward_type="gold_purchase" if task_id != 3 else "reward_unverifiable",
        )

    request = ProbeRequest(model, train, val, evaluation, tmp_path / "out", None, "http://shop", 7)
    result = run_probe(request, rollout_one)
    assert calls == [1, 2, 4, 3]
    assert result.valid_task_ids == (1, 2, 4)
    assert result.invalid_task_ids == (3,)
    assert DETERMINISTIC_DECODE == {"temperature": 0.0, "top_p": 1.0, "do_sample": False}


def test_dry_run_does_not_create_output_or_rollout(monkeypatch, tmp_path):
    train, val, evaluation = (tmp_path / name for name in ("train.jsonl", "val.jsonl", "eval.jsonl"))
    _write(train, [1]); _write(val, [2]); _write(evaluation, [9])
    model = tmp_path / "model"; model.mkdir(); (model / "config.json").write_text("{}")
    monkeypatch.setattr(probe_cli, "execute_probe", lambda args: (_ for _ in ()).throw(AssertionError()))
    output = tmp_path / "output"
    audit = probe_cli.main_for_test([
        "--model", str(model), "--train-data", str(train), "--val-data", str(val),
        "--evaluation-data", str(evaluation), "--output", str(output), "--dry-run",
    ])
    assert audit["decode"] == dict(DETERMINISTIC_DECODE)
    assert not output.exists()
