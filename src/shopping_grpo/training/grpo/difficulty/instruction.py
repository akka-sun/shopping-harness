"""Deterministic GRPO task difficulty derived from public instructions only."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

INSTRUCTION_SCORER_VERSION = "grpo-public-instruction-v1"

_NUMERIC = re.compile(
    r"(?:[$￥¥€£]\s*\d|\d+(?:\.\d+)?\s*(?:元|美元|usd|cm|mm|inch|英寸|oz|gb|tb|ml|l|kg|g)\b|"
    r"(?:under|below|less than|at most|不超过|低于|以内|至少|between)\s*[$￥¥€£]?\s*\d+)",
    re.IGNORECASE,
)
_BUDGET = re.compile(r"(?:[$￥¥€£]\s*\d|budget|price|预算|价格|不超过|低于|以内)", re.IGNORECASE)
_MODEL = re.compile(r"\b(?:[A-Z]{2,}[A-Z0-9-]*\d[A-Z0-9-]*|[A-Za-z]+-\d[A-Za-z0-9-]*)\b")
_OPTION = re.compile(
    r"\b(?:black|white|red|blue|green|small|medium|large|size|color|capacity|material|"
    r"storage|memory)\b|颜色|尺寸|大小|容量|材质|规格|型号|款式|内存|存储",
    re.IGNORECASE,
)
_COMPATIBILITY = re.compile(r"compatible|works? with|适配|兼容|支持.+(?:系统|设备|接口)|可用于", re.IGNORECASE)
_FEATURE = re.compile(r"dishwasher|waterproof|wireless|noise.?cancel|功能|防水|降噪|续航|可洗", re.IGNORECASE)
_BRAND_HINT = re.compile(r"\b(?:sony|apple|samsung|contigo|nike|adidas|logitech|lenovo|dell|hp)\b", re.IGNORECASE)
_JOINT = re.compile(r"(?:,|，|\band\b|并且|同时|以及|且)", re.IGNORECASE)


@dataclass(frozen=True)
class InstructionDifficulty:
    score: float
    features: Mapping[str, int | float | bool]
    scorer_version: str = INSTRUCTION_SCORER_VERSION


def _count(pattern: re.Pattern[str], text: str, cap: int) -> int:
    return min(len(pattern.findall(text)), cap)


def score_public_instruction(instruction: str) -> InstructionDifficulty:
    """Score only lexical constraints visible in the user's instruction."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    text = instruction.strip()
    numeric = _count(_NUMERIC, text, 3)
    brand_model = min(_count(_BRAND_HINT, text, 2) + _count(_MODEL, text, 2), 2)
    option_axes = _count(_OPTION, text, 4)
    compatibility = _count(_COMPATIBILITY, text, 2)
    functional = _count(_FEATURE, text, 3)
    categories = sum(
        bool(value)
        for value in (numeric, brand_model, option_axes, compatibility, functional)
    )
    joint = categories >= 3 or len(_JOINT.findall(text)) >= 2
    length_weak = min(len(text.split()) / 200.0, 1.0)
    normalized = {
        "constraint_categories": min(categories / 6.0, 1.0),
        "numeric_constraints": numeric / 3.0,
        "brand_model_constraints": brand_model / 2.0,
        "option_axes": option_axes / 4.0,
        "compatibility_constraints": compatibility / 2.0,
        "joint_requirement": float(joint),
        "length_weak": length_weak,
    }
    weights = {
        "constraint_categories": 0.25,
        "numeric_constraints": 0.15,
        "brand_model_constraints": 0.15,
        "option_axes": 0.15,
        "compatibility_constraints": 0.15,
        "joint_requirement": 0.10,
        "length_weak": 0.05,
    }
    score = sum(weights[name] * normalized[name] for name in weights)
    features: dict[str, int | float | bool] = {
        "constraint_categories": categories,
        "numeric_constraints": numeric,
        "brand_model_constraints": brand_model,
        "option_axes": option_axes,
        "compatibility_constraints": compatibility,
        "functional_constraints": functional,
        "joint_requirement": joint,
        "has_budget": bool(_BUDGET.search(text)),
        "length_weak": length_weak,
    }
    return InstructionDifficulty(
        score=min(max(float(score), 0.0), 1.0),
        features=MappingProxyType(features),
    )
