"""Validated, versioned Fresh-SFT Probe difficulty artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROBE_SCORER_VERSION = "shopping-grpo-probe-scorer-v1"


@dataclass(frozen=True)
class ProbeCounters:
    action_steps: int
    action_attempt_count: int
    repeat_action_count: int
    guard_count: int
    max_environment_steps: int = 35


@dataclass(frozen=True)
class ProbeScores:
    task_difficulty: float
    step_score: float
    repeat_ratio: float
    guard_score: float
    probe_difficulty: float


@dataclass(frozen=True)
class DifficultyBoundaries:
    easy_max: float
    medium_max: float


@dataclass(frozen=True)
class ProbeArtifactPaths:
    manifest: Path
    metadata: Path
    invalid: Path


def compute_probe_scores(
    task_difficulty: float, counters: ProbeCounters
) -> ProbeScores:
    task_score = float(task_difficulty)
    values = (
        counters.action_steps,
        counters.action_attempt_count,
        counters.repeat_action_count,
        counters.guard_count,
    )
    if not math.isfinite(task_score) or not 0.0 <= task_score <= 1.0:
        raise ValueError("task_difficulty must be finite and in [0, 1]")
    if any(int(value) < 0 for value in values):
        raise ValueError("Probe counters must be non-negative")
    if counters.repeat_action_count > counters.action_attempt_count:
        raise ValueError("repeat_action_count cannot exceed action_attempt_count")
    if counters.max_environment_steps <= 0:
        raise ValueError("max_environment_steps must be positive")
    step = min(counters.action_steps / counters.max_environment_steps, 1.0)
    repeat = counters.repeat_action_count / max(counters.action_attempt_count, 1)
    guard = min(counters.guard_count / 3.0, 1.0)
    combined = 0.45 * task_score + 0.30 * step + 0.15 * repeat + 0.10 * guard
    return ProbeScores(task_score, step, repeat, guard, combined)


def fit_train_boundaries(
    train_scores: list[tuple[int, float]],
) -> DifficultyBoundaries:
    if len(train_scores) < 3:
        raise ValueError("at least three train scores are required")
    ordered = sorted((float(score), int(task_id)) for task_id, score in train_scores)
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score, _ in ordered):
        raise ValueError("train scores must be finite and in [0, 1]")
    unique = sorted({score for score, _ in ordered})
    if len(unique) < 3:
        raise ValueError("three categories cannot be formed without splitting tied scores")

    def boundary(nominal: int) -> float:
        nominal = min(max(nominal, 1), len(ordered) - 1)
        left = nominal - 1
        right = nominal
        while right < len(ordered) and ordered[left][0] == ordered[right][0]:
            right += 1
        if right == len(ordered):
            right = nominal
            left = nominal - 1
            while left >= 0 and ordered[left][0] == ordered[right][0]:
                left -= 1
        if left < 0 or right >= len(ordered) or ordered[left][0] == ordered[right][0]:
            raise ValueError("cannot place quantile boundary without splitting ties")
        return (ordered[left][0] + ordered[right][0]) / 2.0

    lower = boundary(math.ceil(len(ordered) * 0.33))
    upper = boundary(math.ceil(len(ordered) * 0.66))
    if lower >= upper:
        raise ValueError("difficulty boundaries must be strictly ordered")
    return DifficultyBoundaries(lower, upper)


def assign_categories(
    task_scores: list[tuple[int, float]], boundaries: DifficultyBoundaries
) -> dict[int, str]:
    result = {}
    for task_id, raw_score in task_scores:
        score = float(raw_score)
        result[int(task_id)] = (
            "easy"
            if score <= boundaries.easy_max
            else "medium"
            if score <= boundaries.medium_max
            else "hard"
        )
    return result


def validate_task_isolation(
    train_ids: set[int], validation_ids: set[int], evaluation_ids: set[int]
) -> None:
    if train_ids & validation_ids:
        raise ValueError("train/validation task_id overlap")
    if (train_ids | validation_ids) & evaluation_ids:
        raise ValueError("GRPO/Evaluation task_id overlap")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_probe_artifacts(
    output_dir: Path,
    *,
    valid_records: list[dict[str, Any]],
    invalid_records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> ProbeArtifactPaths:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = destination / "difficulty_manifest.jsonl"
    invalid = destination / "invalid_probes.jsonl"
    metadata_path = destination / "probe_metadata.json"
    _write_atomic(
        manifest,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in valid_records),
    )
    _write_atomic(
        invalid,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in invalid_records),
    )
    complete_metadata = {
        **metadata,
        "scorer_version": PROBE_SCORER_VERSION,
        "manifest_sha256": sha256_file(manifest),
        "invalid_probes_sha256": sha256_file(invalid),
    }
    _write_atomic(
        metadata_path,
        json.dumps(complete_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return ProbeArtifactPaths(manifest, metadata_path, invalid)
