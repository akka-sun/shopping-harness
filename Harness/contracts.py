"""Strict contracts for teacher-produced Memory and Skill artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

from shopping_grpo.evaluation.contracts import ERROR_TAXONOMY

MEMORY_SCHEMA_VERSION = "shopping-memory-v1"
SKILL_SCHEMA_VERSION = "shopping-skill-v1"
MEMORY_STATES = frozenset(
    {"searching", "candidate_review", "verification", "purchase", "termination"}
)
RUNTIME_STATES = frozenset(
    {"start", "searching", "candidate_found", "product_opened", "verification",
     "option_selected", "ready_to_buy", "purchased", "termination"}
)
_ASIN = re.compile(r"(?<!\d)\d{8,12}(?!\d)")
_CONCRETE_VALUE = re.compile(
    r"(?:[¥￥$€£]\s*\d|\d+(?:\.\d+)?\s*(?:元|块|美元|年|月|日|天|小时|分钟|"
    r"kg|g|mg|lb|oz|ml|l|cm|mm|m|gb|tb|w|v|mah|英寸|寸|码|升|毫升|公斤|克))",
    re.IGNORECASE,
)
_FORBIDDEN_KEYS = frozenset(
    {
        "gold",
        "gold_asin",
        "target_asin",
        "hidden_goal",
        "private_goal",
        "gold_purchase",
        "task_id",
        "asin",
        "product_id",
        "answer",
    }
)
_MEMORY_ERROR_TYPES = ERROR_TAXONOMY - {
    "reward_rubric_disagreement",
    "infrastructure_invalid",
}
_RUNTIME_PREDICATE = re.compile(
    r"^(last_tool|available_tools|product_opened|variant_selected|required_fields_verified|"
    r"purchase_available|environment_done|same_query_count|same_action_count|no_progress_steps)"
    r"\s*(==|>=|<=|>|<)\s*(.+)$",
    re.IGNORECASE,
)
_BOOLEAN_PREDICATE_KEYS = frozenset(
    {"product_opened", "variant_selected", "required_fields_verified", "purchase_available", "environment_done"}
)
_COUNT_PREDICATE_KEYS = frozenset({"same_query_count", "same_action_count", "no_progress_steps"})


class HarnessValidationError(ValueError):
    """Raised when teacher output is unsafe or does not match its contract."""


def _text(value: object, path: str, *, max_len: int = 500) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessValidationError(f"{path} must be a non-empty string")
    value = value.strip()
    if len(value) > max_len:
        raise HarnessValidationError(f"{path} exceeds {max_len} characters")
    return value


def _list(value: object, path: str, *, max_items: int = 32) -> list:
    if not isinstance(value, list):
        raise HarnessValidationError(f"{path} must be a list")
    if len(value) > max_items:
        raise HarnessValidationError(f"{path} has too many items")
    return value


def _mapping(value: object, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise HarnessValidationError(f"{path} must be an object")
    return value


def _reject_extra_keys(value: Mapping, allowed: frozenset[str], path: str) -> None:
    unexpected = sorted(str(key) for key in value if str(key) not in allowed)
    if unexpected:
        raise HarnessValidationError(
            f"{path} contains unsupported fields: {', '.join(unexpected)}"
        )


def _check_no_leak(value: object, path: str = "artifact") -> None:
    """Reject likely answer/goal leakage recursively, including suspicious keys."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).casefold()
            if key_text in _FORBIDDEN_KEYS or "hidden" in key_text or "gold" in key_text:
                raise HarnessValidationError(f"{path}.{key} is not allowed")
            _check_no_leak(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _check_no_leak(item, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.casefold()
        if _ASIN.search(value):
            raise HarnessValidationError(f"{path} contains a product identifier")
        if _CONCRETE_VALUE.search(value):
            raise HarnessValidationError(f"{path} contains a concrete task value")
        if re.search(
            r"\b(?:asin|product_id|gold|hidden|target_asin|private_goal|gold_purchase)\b",
            lowered,
        ):
            raise HarnessValidationError(f"{path} contains a forbidden goal marker")


def _trigger(value: object, path: str, allowed_states: frozenset[str]) -> dict:
    trigger = _mapping(value, path)
    state = _text(trigger.get("state"), f"{path}.state", max_len=64).casefold()
    if state not in allowed_states:
        raise HarnessValidationError(f"{path}.state is unsupported: {state}")
    predicates = _list(trigger.get("predicates", []), f"{path}.predicates", max_items=8)
    normalized = []
    for index, predicate in enumerate(predicates):
        predicate_path = f"{path}.predicates[{index}]"
        text = _text(predicate, predicate_path, max_len=160)
        match = _RUNTIME_PREDICATE.fullmatch(text)
        if not match:
            raise HarnessValidationError(
                f"{predicate_path} is not in the runtime predicate grammar"
            )
        key, operator, raw = match.groups()
        key = key.casefold()
        raw = raw.strip().casefold()
        if key in _BOOLEAN_PREDICATE_KEYS:
            if operator != "==" or raw not in {"true", "false"}:
                raise HarnessValidationError(
                    f"{predicate_path} requires == true or == false"
                )
        elif key in _COUNT_PREDICATE_KEYS:
            if not raw.isdigit() or not 0 <= int(raw) <= 10:
                raise HarnessValidationError(
                    f"{predicate_path} requires an integer between 0 and 10"
                )
        elif operator != "==":
            raise HarnessValidationError(f"{predicate_path} only supports ==")
        normalized.append(f"{key} {operator} {raw}")
    return {"state": state, "predicates": normalized}


def _confidence(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarnessValidationError(f"{path} must be a number")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise HarnessValidationError(f"{path} must be between 0 and 1")
    return confidence


def validate_memory(value: object, *, event_ids: set[str] | None = None) -> dict:
    """Validate and return one public-only single-trajectory Memory."""
    payload = _mapping(value, "memory")
    if payload.get("schema_version") != MEMORY_SCHEMA_VERSION:
        raise HarnessValidationError("memory.schema_version is unsupported")
    _reject_extra_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "memory_id",
                "source_trajectory_id",
                "phase",
                "trigger",
                "error_type",
                "first_bad_event_id",
                "lesson",
                "risk",
                "confidence",
            }
        ),
        "memory",
    )
    result = dict(payload)
    _text(result.get("memory_id"), "memory.memory_id", max_len=128)
    _text(result.get("source_trajectory_id"), "memory.source_trajectory_id", max_len=128)
    result["phase"] = _text(result.get("phase"), "memory.phase", max_len=64).casefold()
    if result["phase"] not in MEMORY_STATES:
        raise HarnessValidationError("memory.phase is unsupported")
    result["trigger"] = _trigger(result.get("trigger"), "memory.trigger", RUNTIME_STATES)
    error_type = _text(result.get("error_type"), "memory.error_type", max_len=80)
    if error_type not in _MEMORY_ERROR_TYPES:
        raise HarnessValidationError("memory.error_type is not in the frozen taxonomy")
    event_id = _text(result.get("first_bad_event_id"), "memory.first_bad_event_id", max_len=32)
    if event_ids is not None and event_id not in event_ids:
        raise HarnessValidationError("memory.first_bad_event_id is not present in trajectory")
    result["lesson"] = _text(result.get("lesson"), "memory.lesson", max_len=320)
    result["risk"] = _text(result.get("risk"), "memory.risk", max_len=120)
    result["confidence"] = _confidence(result.get("confidence"), "memory.confidence")
    # IDs are checked against local provenance by the pipeline.  UUIDs can
    # legitimately contain an eight-digit run, so leakage checks belong on the
    # teacher-authored semantic fields rather than opaque identity fields.
    _check_no_leak(
        {
            "trigger": result["trigger"],
            "lesson": result["lesson"],
            "risk": result["risk"],
        }
    )
    return result


def validate_skill(value: object) -> dict:
    """Validate one multi-trajectory procedural Skill candidate."""
    payload = _mapping(value, "skill")
    if payload.get("schema_version") != SKILL_SCHEMA_VERSION:
        raise HarnessValidationError("skill.schema_version is unsupported")
    _reject_extra_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "skill_id",
                "name",
                "scope",
                "trigger",
                "procedure",
                "forbidden_shortcuts",
                "source_trajectory_ids",
                "source_memory_ids",
                "confidence",
                "status",
            }
        ),
        "skill",
    )
    result = dict(payload)
    _text(result.get("skill_id"), "skill.skill_id", max_len=128)
    _text(result.get("name"), "skill.name", max_len=120)
    result["scope"] = _text(result.get("scope"), "skill.scope", max_len=64).casefold()
    if result["scope"] not in MEMORY_STATES:
        raise HarnessValidationError("skill.scope must name exactly one supported phase")
    result["trigger"] = _trigger(result.get("trigger"), "skill.trigger", RUNTIME_STATES)
    procedure = _list(result.get("procedure"), "skill.procedure", max_items=8)
    result["procedure"] = [_text(item, f"skill.procedure[{i}]", max_len=240) for i, item in enumerate(procedure)]
    if not result["procedure"]:
        raise HarnessValidationError("skill.procedure must not be empty")
    forbidden = _list(result.get("forbidden_shortcuts", []), "skill.forbidden_shortcuts", max_items=8)
    result["forbidden_shortcuts"] = [_text(item, f"skill.forbidden_shortcuts[{i}]", max_len=200) for i, item in enumerate(forbidden)]
    trajectory_ids = _list(result.get("source_trajectory_ids"), "skill.source_trajectory_ids", max_items=64)
    result["source_trajectory_ids"] = [_text(item, f"skill.source_trajectory_ids[{i}]", max_len=128) for i, item in enumerate(trajectory_ids)]
    if len(set(result["source_trajectory_ids"])) < 2:
        raise HarnessValidationError("skill requires at least two source trajectories")
    if len(result["source_trajectory_ids"]) != len(set(result["source_trajectory_ids"])):
        raise HarnessValidationError("skill.source_trajectory_ids must not contain duplicates")
    memory_ids = _list(result.get("source_memory_ids", []), "skill.source_memory_ids", max_items=64)
    result["source_memory_ids"] = [_text(item, f"skill.source_memory_ids[{i}]", max_len=128) for i, item in enumerate(memory_ids)]
    result["confidence"] = _confidence(result.get("confidence"), "skill.confidence")
    status = _text(result.get("status"), "skill.status", max_len=32).casefold()
    if status not in {"candidate", "active", "rejected", "retired"}:
        raise HarnessValidationError("skill.status is unsupported")
    result["status"] = status
    _check_no_leak(
        {
            "name": result["name"],
            "scope": result["scope"],
            "trigger": result["trigger"],
            "procedure": result["procedure"],
            "forbidden_shortcuts": result["forbidden_shortcuts"],
        }
    )
    return result
