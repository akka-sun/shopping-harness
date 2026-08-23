"""One deterministic Fresh-SFT rollout per GRPO task."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from shopping_grpo.training.grpo.difficulty.instruction import score_public_instruction
from shopping_grpo.training.grpo.difficulty.manifest import (
    ProbeCounters,
    assign_categories,
    compute_probe_scores,
    fit_train_boundaries,
    validate_task_isolation,
    write_probe_artifacts,
)

DETERMINISTIC_DECODE: Mapping[str, object] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": False,
}


@dataclass(frozen=True)
class ProbeRequest:
    model: Path
    train_tasks: Path
    validation_tasks: Path
    evaluation_tasks: Path
    output_dir: Path
    raw_run_dir: Path | None
    environment_url: str
    seed: int
    max_environment_steps: int = 35


@dataclass(frozen=True)
class ProbeOutcome:
    task_id: int
    instruction: str
    valid: bool
    invalid_reason: str | None
    action_steps: int
    action_attempt_count: int
    repeat_action_count: int
    guard_count: int
    reward_type: str | None


@dataclass(frozen=True)
class ProbeRunResult:
    valid_task_ids: tuple[int, ...]
    invalid_task_ids: tuple[int, ...]
    output_dir: Path


def _load_ids(path: Path) -> list[int]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    ids = [int(row["task_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate task_id in {path}")
    return ids


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    candidates = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in candidates:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def run_probe(
    request: ProbeRequest,
    rollout_one: Callable[[int, Mapping[str, object], int, int], ProbeOutcome],
) -> ProbeRunResult:
    train_ids = _load_ids(request.train_tasks)
    validation_ids = _load_ids(request.validation_tasks)
    evaluation_ids = set(_load_ids(request.evaluation_tasks))
    validate_task_isolation(set(train_ids), set(validation_ids), evaluation_ids)
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for split, task_ids in (("train", train_ids), ("validation", validation_ids)):
        for task_id in task_ids:
            outcome = rollout_one(
                task_id,
                DETERMINISTIC_DECODE,
                int(request.seed),
                int(request.max_environment_steps),
            )
            if outcome.task_id != task_id:
                raise ValueError("Probe outcome task_id mismatch")
            if not outcome.valid:
                invalid.append({"task_id": task_id, "split": split, "reason": outcome.invalid_reason})
                continue
            instruction = score_public_instruction(outcome.instruction)
            counters = ProbeCounters(
                outcome.action_steps,
                outcome.action_attempt_count,
                outcome.repeat_action_count,
                outcome.guard_count,
                request.max_environment_steps,
            )
            scores = compute_probe_scores(instruction.score, counters)
            valid.append(
                {
                    "task_id": task_id,
                    "split": split,
                    **asdict(counters),
                    **asdict(scores),
                    "instruction_features": dict(instruction.features),
                    "probe_status": "valid",
                    "reward_type": outcome.reward_type,
                    "scorer_version": instruction.scorer_version,
                }
            )
    train_scores = [(row["task_id"], row["probe_difficulty"]) for row in valid if row["split"] == "train"]
    boundaries = fit_train_boundaries(train_scores)
    categories = assign_categories(
        [(row["task_id"], row["probe_difficulty"]) for row in valid], boundaries
    )
    for row in valid:
        row["difficulty_category"] = categories[row["task_id"]]
    valid_ids = {row["task_id"] for row in valid}
    invalid_ids = {row["task_id"] for row in invalid}
    if valid_ids | invalid_ids != set(train_ids) | set(validation_ids):
        raise ValueError("Probe output has missing or unknown task_id")
    validate_task_isolation(valid_ids & set(train_ids), valid_ids & set(validation_ids), evaluation_ids)
    write_probe_artifacts(
        request.output_dir,
        valid_records=valid,
        invalid_records=invalid,
        metadata={
            "schema_version": "shopping-probe-metadata-v1",
            "model_path": str(request.model.resolve()),
            "model_sha256": _hash_path(request.model),
            "decode": dict(DETERMINISTIC_DECODE),
            "seed": request.seed,
            "max_environment_steps": request.max_environment_steps,
            "boundaries": asdict(boundaries),
            "environment_version": "shopsimulator-environment-v2.1",
            "reward_version": "shopsimulator-reward-v3",
            "observation_version": "shopping-observation-v2",
            "tool_schema_version": "shopping-tools-v2",
            "evaluation_overlap": False,
        },
    )
    return ProbeRunResult(tuple(sorted(valid_ids)), tuple(sorted(invalid_ids)), request.output_dir)
