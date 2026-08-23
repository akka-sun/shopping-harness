"""Bounded, token-mass-conserving trajectory credit assignment."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from shopping_grpo.training.grpo.evidence import EVIDENCE_VERSION, EvidenceEvent


@dataclass(frozen=True)
class TurnSpan:
    assistant_turn_id: int
    start: int
    end: int


@dataclass(frozen=True)
class CreditConfig:
    positive_evidence_score: float = 1.0
    no_progress_score: float = -0.5
    exact_repeat_score: float = -1.0
    guard_rejection_score: float = -1.0
    strength: float = 0.5
    multiplier_min: float = 0.5
    multiplier_max: float = 1.5
    conservation_tolerance: float = 1e-6


@dataclass(frozen=True)
class CreditResult:
    multipliers: tuple[float, ...]
    mass_error: float
    fallback_reason: str | None = None


@dataclass(frozen=True)
class ProjectionResult:
    token_advantages: Any
    multipliers: tuple[float, ...]
    mass_error: float
    fallback_reason: str | None = None


class CreditFallbackWindow:
    """Fail a run when a full rolling window exceeds the fallback threshold."""

    def __init__(self, size: int, threshold: float):
        if size <= 0 or not 0 <= threshold <= 1:
            raise ValueError("fallback window size/threshold are invalid")
        self.size = int(size)
        self.threshold = float(threshold)
        self._values = deque(maxlen=self.size)

    def record(self, fallback: bool) -> None:
        self._values.append(bool(fallback))
        if len(self._values) == self.size and sum(self._values) / self.size > self.threshold:
            raise RuntimeError(
                f"trajectory credit fallback rate exceeds {self.threshold:.6f} "
                f"over {self.size} trajectories"
            )


def _fallback(count: int, reason: str) -> CreditResult:
    return CreditResult((1.0,) * count, 0.0, reason)


def _score(event: EvidenceEvent, config: CreditConfig) -> float:
    positive = bool(
        event.new_search_result_set
        or event.new_product_count
        or event.first_product_open
        or event.first_info_subpage
        or event.first_option_confirmation
    )
    value = config.positive_evidence_score if positive else 0.0
    value += config.no_progress_score if event.no_progress else 0.0
    value += config.exact_repeat_score if event.exact_repeat else 0.0
    value += config.guard_rejection_score if event.guard_rejection else 0.0
    return min(max(value, -1.0), 1.0)


def compute_turn_multipliers(
    trajectory_advantage: float,
    events: list[EvidenceEvent],
    spans: list[TurnSpan],
    config: CreditConfig,
    *,
    compacted: bool = False,
) -> CreditResult:
    count = len(spans)
    if compacted:
        return _fallback(count, "context_compaction_alignment_unsupported")
    if len(events) != count:
        return _fallback(count, "event_turn_count_mismatch")
    if not math.isfinite(float(trajectory_advantage)):
        return _fallback(count, "non_finite_advantage")
    if trajectory_advantage == 0 or count == 0:
        return CreditResult((1.0,) * count, 0.0)
    if not 0 < config.multiplier_min <= 1 <= config.multiplier_max:
        return _fallback(count, "invalid_multiplier_bounds")
    lengths = []
    scores = []
    previous_end = 0
    for event, span in zip(events, spans, strict=True):
        if event.schema_version != EVIDENCE_VERSION:
            return _fallback(count, "unknown_event_version")
        if event.assistant_turn_id != span.assistant_turn_id:
            return _fallback(count, "turn_id_mismatch")
        if span.start < previous_end or span.end <= span.start:
            return _fallback(count, "invalid_turn_span")
        previous_end = span.end
        lengths.append(span.end - span.start)
        scores.append(_score(event, config) * (1.0 if trajectory_advantage > 0 else -1.0))
    total_tokens = sum(lengths)
    center = sum(length * score for length, score in zip(lengths, scores, strict=True)) / total_tokens
    centered = [score - center for score in scores]
    scale = float(config.strength)
    if not math.isfinite(scale) or scale < 0:
        return _fallback(count, "non_finite_weight")
    for value in centered:
        if value > 0:
            scale = min(scale, (config.multiplier_max - 1.0) / value)
        elif value < 0:
            scale = min(scale, (config.multiplier_min - 1.0) / value)
    multipliers = [1.0 + scale * value for value in centered]
    weighted_mean = sum(n * m for n, m in zip(lengths, multipliers, strict=True)) / total_tokens
    residual = 1.0 - weighted_mean
    multipliers = [m + residual for m in multipliers]
    mass_error = abs(
        sum(n * m for n, m in zip(lengths, multipliers, strict=True)) / total_tokens - 1.0
    )
    if (
        any(not math.isfinite(m) or m < config.multiplier_min or m > config.multiplier_max for m in multipliers)
        or mass_error > config.conservation_tolerance
    ):
        return _fallback(count, "normalization_failed")
    return CreditResult(tuple(multipliers), mass_error)


def project_token_advantages(
    trajectory_advantage,
    response_mask,
    spans: list[TurnSpan],
    events: list[EvidenceEvent],
    config: CreditConfig,
    *,
    compacted: bool = False,
) -> ProjectionResult:
    import torch

    mask = response_mask.bool()
    base = torch.zeros_like(response_mask, dtype=torch.float64)
    base[mask] = float(trajectory_advantage)
    credit = compute_turn_multipliers(
        float(trajectory_advantage), events, spans, config, compacted=compacted
    )
    if credit.fallback_reason:
        return ProjectionResult(base, credit.multipliers, 0.0, credit.fallback_reason)
    output = torch.zeros_like(base)
    covered = torch.zeros_like(mask)
    for span, multiplier in zip(spans, credit.multipliers, strict=True):
        if span.end > mask.numel() or not bool(mask[span.start:span.end].all()):
            fallback = _fallback(len(spans), "response_mask_alignment")
            return ProjectionResult(base, fallback.multipliers, 0.0, fallback.fallback_reason)
        output[span.start:span.end] = float(trajectory_advantage) * multiplier
        covered[span.start:span.end] = True
    if not torch.equal(covered, mask):
        fallback = _fallback(len(spans), "incomplete_response_spans")
        return ProjectionResult(base, fallback.multipliers, 0.0, fallback.fallback_reason)
    before = float(base.sum().item())
    after = float(output.sum().item())
    mass_error = abs(after - before)
    if mass_error > config.conservation_tolerance:
        fallback = _fallback(len(spans), "token_mass_mismatch")
        return ProjectionResult(base, fallback.multipliers, mass_error, fallback.fallback_reason)
    return ProjectionResult(output, credit.multipliers, mass_error)
