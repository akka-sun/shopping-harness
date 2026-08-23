"""Cumulative online curriculum statistics and checkpoint schema."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

STATE_VERSION = "shopping-curriculum-state-v1"


@dataclass(frozen=True)
class GroupOutcome:
    task_id: int
    difficulty_category: str
    terminal_utilities: tuple[float, ...]
    failure_types: tuple[str, ...]
    valid: bool = True


@dataclass
class CellStats:
    valid_trajectories: int = 0
    valid_groups: int = 0
    strict_successes: int = 0
    utility_sum: float = 0.0
    group_variance_sum: float = 0.0
    zero_variance_groups: int = 0
    failure_count: int = 0
    last_update_step: int = -1


@dataclass
class TaskStats:
    sample_count: int = 0
    last_sample_step: int = -1
    consecutive_zero_variance: int = 0


@dataclass
class CurriculumState:
    manifest_hash: str
    config_hash: str
    total_steps: int
    committed_global_step: int = 0
    cells: dict[tuple[str, str], CellStats] = field(default_factory=dict)
    tasks: dict[int, TaskStats] = field(default_factory=dict)
    invalid_group_count: int = 0
    schema_version: str = STATE_VERSION

    @classmethod
    def empty(cls, manifest_hash: str, config_hash: str, total_steps: int):
        return cls(str(manifest_hash), str(config_hash), int(total_steps))

    def update_group(self, outcome: GroupOutcome, global_step: int) -> None:
        if not outcome.valid:
            self.invalid_group_count += 1
            return
        if not outcome.terminal_utilities or len(outcome.terminal_utilities) != len(outcome.failure_types):
            raise ValueError("complete group utilities and failure types are required")
        utilities = tuple(float(value) for value in outcome.terminal_utilities)
        if any(not math.isfinite(value) for value in utilities):
            raise ValueError("terminal utilities must be finite")
        mean = sum(utilities) / len(utilities)
        variance = sum((value - mean) ** 2 for value in utilities) / len(utilities)
        zero_variance = variance <= 1e-8
        task = self.tasks.setdefault(int(outcome.task_id), TaskStats())
        task.sample_count += 1
        task.last_sample_step = int(global_step)
        task.consecutive_zero_variance = task.consecutive_zero_variance + 1 if zero_variance else 0
        if "near_miss" in outcome.failure_types:
            task.consecutive_zero_variance = 0
        for failure_type in sorted(set(outcome.failure_types)):
            indices = [i for i, value in enumerate(outcome.failure_types) if value == failure_type]
            cell = self.cells.setdefault((outcome.difficulty_category, failure_type), CellStats())
            cell.valid_trajectories += len(indices)
            cell.valid_groups += 1
            cell.strict_successes += sum(outcome.failure_types[i] == "strict_success" for i in indices)
            cell.utility_sum += sum(utilities[i] for i in indices)
            cell.group_variance_sum += variance
            cell.zero_variance_groups += int(zero_variance)
            cell.failure_count += len(indices)
            cell.last_update_step = int(global_step)

    def to_checkpoint(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "manifest_hash": self.manifest_hash,
            "config_hash": self.config_hash,
            "total_steps": self.total_steps,
            "committed_global_step": self.committed_global_step,
            "invalid_group_count": self.invalid_group_count,
            "cells": {f"{key[0]}::{key[1]}": asdict(value) for key, value in self.cells.items()},
            "tasks": {str(key): asdict(value) for key, value in self.tasks.items()},
        }

    @classmethod
    def from_checkpoint(cls, payload: dict, manifest_hash: str, config_hash: str, total_steps: int):
        if payload.get("schema_version") != STATE_VERSION:
            raise ValueError("curriculum state schema mismatch")
        if payload.get("manifest_hash") != manifest_hash:
            raise ValueError("curriculum manifest hash mismatch")
        if payload.get("config_hash") != config_hash:
            raise ValueError("curriculum config hash mismatch")
        if int(payload.get("total_steps", -1)) != int(total_steps):
            raise ValueError("curriculum total steps mismatch")
        state = cls(
            manifest_hash,
            config_hash,
            int(total_steps),
            committed_global_step=int(payload.get("committed_global_step", 0)),
            invalid_group_count=int(payload.get("invalid_group_count", 0)),
        )
        state.cells = {
            tuple(key.split("::", 1)): CellStats(**value)
            for key, value in payload.get("cells", {}).items()
        }
        state.tasks = {int(key): TaskStats(**value) for key, value in payload.get("tasks", {}).items()}
        return state
