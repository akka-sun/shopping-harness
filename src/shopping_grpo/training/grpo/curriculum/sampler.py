"""Deterministic 30/70 phase and mixed task sampler."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass


def switch_step(total_steps: int, switch_ratio: float) -> int:
    if total_steps <= 0 or not 0 < switch_ratio < 1:
        raise ValueError("total_steps must be positive and switch_ratio must be in (0, 1)")
    return math.ceil(total_steps * switch_ratio)


def phase_for_step(global_step: int, total_steps: int, switch_ratio: float) -> str:
    return "baseline_30" if global_step < switch_step(total_steps, switch_ratio) else "difficulty_70"


@dataclass(frozen=True)
class TaskCandidate:
    task_id: int
    difficulty_category: str
    probe_difficulty: float
    priority: float


@dataclass(frozen=True)
class SamplerConfig:
    seed: int
    total_steps: int = 10
    switch_ratio: float = 0.30
    uniform_ratio: float = 0.20


@dataclass(frozen=True)
class SelectionResult:
    task_ids: tuple[int, ...]
    phase: str
    uniform_count: int
    dynamic_count: int
    theoretical_probabilities: dict[int, float]
    fallback_reason: str | None = None


class CurriculumSampler:
    def __init__(self, tasks: list[TaskCandidate], config: SamplerConfig):
        if not 0 < config.uniform_ratio <= 1:
            raise ValueError("uniform_ratio must be in (0, 1]")
        if len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("candidate task_ids must be unique")
        self.tasks = tuple(tasks)
        self.config = config

    def _rng(self, global_step: int, generation_batch_index: int) -> random.Random:
        payload = json.dumps(
            [self.config.seed, int(global_step), int(generation_batch_index)], separators=(",", ":")
        )
        seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
        return random.Random(seed)

    def select(
        self,
        global_step: int,
        generation_batch_index: int,
        count: int,
        baseline_select: Callable[[int], list[int]],
    ) -> SelectionResult:
        phase = phase_for_step(global_step, self.config.total_steps, self.config.switch_ratio)
        if phase == "baseline_30":
            return SelectionResult(tuple(baseline_select(count)), phase, count, 0, {})
        if count > len(self.tasks):
            raise ValueError("not enough unique legal tasks")
        priorities = [float(task.priority) for task in self.tasks]
        fallback = None
        if any(not math.isfinite(value) or value <= 0 for value in priorities):
            priorities = [1.0] * len(self.tasks)
            fallback = "non_finite_priority"
        total = sum(priorities)
        theoretical = {
            task.task_id: self.config.uniform_ratio / len(self.tasks)
            + (1.0 - self.config.uniform_ratio) * priority / total
            for task, priority in zip(self.tasks, priorities, strict=True)
        }
        rng = self._rng(global_step, generation_batch_index)
        remaining = list(zip(self.tasks, priorities, strict=True))
        selected = []
        uniform_count = 0
        for _ in range(count):
            use_uniform = rng.random() < self.config.uniform_ratio or fallback is not None
            weights = [1.0] * len(remaining) if use_uniform else [item[1] for item in remaining]
            chosen = rng.choices(range(len(remaining)), weights=weights, k=1)[0]
            task, _ = remaining.pop(chosen)
            selected.append(task.task_id)
            uniform_count += int(use_uniform)
        return SelectionResult(
            tuple(selected), phase, uniform_count, count - uniform_count, theoretical, fallback
        )
