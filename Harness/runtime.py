"""Deterministic state tracking and runtime knowledge retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re

from .contracts import validate_memory, validate_skill


@dataclass
class HarnessState:
    """Trajectory-local public state; never stores hidden goals or target values."""

    phase: str = "start"
    last_tool: str | None = None
    available_tools: tuple[str, ...] = ()
    product_opened: bool = False
    variant_selected: bool = False
    # None means no public evidence checker supplied a verdict.  Treating that
    # as False would turn uncertainty into a fact and over-trigger guidance.
    required_fields_verified: bool | None = None
    purchase_available: bool = False
    current_product_id: str | None = None
    environment_done: bool = False
    same_query_count: int = 0
    same_action_count: int = 0
    no_progress_steps: int = 0
    _last_query: str | None = field(default=None, repr=False)
    _last_action: str | None = field(default=None, repr=False)

    def update(
        self,
        *,
        tool_name: str | None,
        parameters: Mapping | None,
        observation: str,
        done: bool = False,
        required_fields_verified: bool | None = None,
    ) -> None:
        """Advance state using only public tool/observation data.

        ``required_fields_verified`` is optional because the Harness cannot infer
        a task's requested fields from a generic observation.  An integration may
        pass a value from its public evidence checker; the default remains the
        conservative ``False``.
        """
        previous_action = self._last_action
        previous_query = self._last_query
        self.last_tool = str(tool_name) if tool_name else None
        self.environment_done = bool(done)
        text = str(observation or "")
        self.available_tools = tuple(_buttons(text))
        self.purchase_available = any(item.casefold() == "buy now" for item in self.available_tools)
        if required_fields_verified is not None:
            self.required_fields_verified = bool(required_fields_verified)
        if self.last_tool == "search_products":
            # A new result set invalidates every product-local fact.  Keeping
            # these sticky would make guidance for an earlier product match a
            # later, unrelated candidate.
            self.product_opened = False
            self.current_product_id = None
            self.variant_selected = False
            if required_fields_verified is None:
                self.required_fields_verified = None
            query = str((parameters or {}).get("query", "")).strip().casefold()
            self.same_query_count = self.same_query_count + 1 if query and query == previous_query else 0
            self._last_query = query or self._last_query
            self.phase = "candidate_found" if _product_buttons(self.available_tools) else "searching"
        elif self.last_tool == "open_product":
            self.product_opened = True
            self.current_product_id = str((parameters or {}).get("asin") or "") or None
            self.variant_selected = False
            if required_fields_verified is None:
                self.required_fields_verified = None
            self.phase = "product_opened"
        elif self.last_tool == "select_option":
            self.variant_selected = True
            # ``ready_to_buy`` must describe the pre-purchase decision point,
            # not the aftermath of an attempted purchase.  The old transition
            # made every ready-to-buy rule unreachable on normal successful
            # trajectories because ``buy_now`` terminates immediately.
            self.phase = "ready_to_buy" if self.purchase_available else "option_selected"
        elif self.last_tool == "buy_now":
            self.phase = "purchased" if done else "ready_to_buy"
        elif self.last_tool in {"view_description", "view_features", "view_reviews", "view_attributes"}:
            self.phase = "verification"
        elif self.last_tool == "back_to_search":
            self.product_opened = False
            self.current_product_id = None
            self.variant_selected = False
            if required_fields_verified is None:
                self.required_fields_verified = None
            self.phase = "searching"
        elif self.last_tool == "prev_page":
            # ``prev_page`` is overloaded by ShopSimulator: from an information
            # subpage it returns to the product, while from product_detail it
            # returns to search results.  Reconcile against the authoritative
            # page_type below instead of assuming either destination here.
            if not self.product_opened:
                self.phase = "searching"
            elif self.variant_selected and self.purchase_available:
                self.phase = "ready_to_buy"
            elif self.variant_selected:
                self.phase = "option_selected"
            else:
                self.phase = "product_opened"
        self._reconcile_page_type(text, required_fields_verified)
        if done:
            self.phase = "purchased" if self.last_tool == "buy_now" else "termination"
        action = _action_signature(self.last_tool, parameters)
        self.same_action_count = self.same_action_count + 1 if action and action == previous_action else 0
        self._last_action = action or self._last_action
        # A changed action is progress from the loop detector's perspective.  The
        # previous implementation assigned ``_last_action`` before comparing it,
        # which made every action look unchanged and allowed stale risk counts to
        # survive indefinitely.
        self.no_progress_steps = (
            self.no_progress_steps + 1
            if action and action == previous_action
            else 0
        )

    def _reconcile_page_type(
        self, observation: str, required_fields_verified: bool | None
    ) -> None:
        """Make the latest public page type authoritative over action history."""
        page_type = _page_type(observation)
        if page_type in {"search", "search_home"}:
            self.product_opened = False
            self.current_product_id = None
            self.variant_selected = False
            if required_fields_verified is None:
                self.required_fields_verified = None
            self.phase = "searching"
        elif page_type == "search_results":
            self.product_opened = False
            self.current_product_id = None
            self.variant_selected = False
            if required_fields_verified is None:
                self.required_fields_verified = None
            self.phase = "candidate_found"
        elif page_type == "product_detail":
            self.product_opened = True
            if self.variant_selected and self.purchase_available:
                self.phase = "ready_to_buy"
            elif self.variant_selected:
                self.phase = "option_selected"
            else:
                self.phase = "product_opened"
        elif page_type in {"description", "features", "reviews", "attributes"}:
            self.product_opened = True
            self.phase = "verification"

    def record_rejected(
        self,
        *,
        tool_name: str | None,
        parameters: Mapping | None,
        observation: str,
    ) -> None:
        """Record a guard-rejected attempt without pretending it changed pages."""
        previous_action = self._last_action
        previous_query = self._last_query
        self.last_tool = str(tool_name) if tool_name else None
        self.available_tools = tuple(_buttons(str(observation or "")))
        self.purchase_available = any(
            item.casefold() == "buy now" for item in self.available_tools
        )
        if self.last_tool == "search_products":
            query = str((parameters or {}).get("query", "")).strip().casefold()
            self.same_query_count = (
                self.same_query_count + 1 if query and query == previous_query else 0
            )
            self._last_query = query or self._last_query
        action = _action_signature(self.last_tool, parameters)
        repeated = bool(action and action == previous_action)
        self.same_action_count = self.same_action_count + 1 if repeated else 0
        self.no_progress_steps = self.no_progress_steps + 1 if repeated else 0
        self._last_action = action or self._last_action

    def predicates(self) -> dict[str, object]:
        return {
            "state": self.phase,
            "last_tool": self.last_tool,
            "available_tools": set(self.available_tools),
            "product_opened": self.product_opened,
            "variant_selected": self.variant_selected,
            "required_fields_verified": self.required_fields_verified,
            "purchase_available": self.purchase_available,
            "current_product_id": self.current_product_id,
            "environment_done": self.environment_done,
            "same_query_count": self.same_query_count,
            "same_action_count": self.same_action_count,
            "no_progress_steps": self.no_progress_steps,
        }


@dataclass(frozen=True)
class RuntimeGuidance:
    skill: dict | None
    memories: tuple[dict, ...]
    message: str


class KnowledgeBank:
    """In-memory validated bank; persistence is intentionally append-only outside it."""

    def __init__(self, memories: Sequence[Mapping] = (), skills: Sequence[Mapping] = (), source_queries=None):
        self.memories = [validate_memory(item) for item in memories]
        self.skills = [validate_skill(item) for item in skills]
        self.source_queries = dict(source_queries or {})
        memory_ids = [item["memory_id"] for item in self.memories]
        skill_ids = [item["skill_id"] for item in self.skills]
        if len(memory_ids) != len(set(memory_ids)):
            raise ValueError("KnowledgeBank contains duplicate memory_id")
        if len(skill_ids) != len(set(skill_ids)):
            raise ValueError("KnowledgeBank contains duplicate skill_id")

    def matching_artifacts(self, values):
        return {
            "skill": [item for item in self.skills if item.get("status") == "active" and _matches(item["trigger"], values)],
            "memory": [item for item in self.memories if _matches(item["trigger"], values)],
        }

    def source_text(self, artifact):
        trajectory_ids = artifact.get("source_trajectory_ids") or [artifact.get("source_trajectory_id")]
        return [self.source_queries[str(value)] for value in trajectory_ids if str(value) in self.source_queries]

    def match(self, state: HarnessState, *, memory_limit: int = 2) -> RuntimeGuidance:
        if memory_limit < 0:
            raise ValueError("memory_limit must be non-negative")
        values = state.predicates()
        skills = [item for item in self.skills if item.get("status") == "active" and _matches(item["trigger"], values)]
        skills.sort(key=lambda item: (-float(item.get("confidence", 0)), str(item.get("skill_id"))))
        selected_skill = skills[0] if skills else None
        memories = [item for item in self.memories if _matches(item["trigger"], values)]
        memories.sort(key=lambda item: (-float(item.get("confidence", 0)), str(item.get("memory_id"))))
        # Teacher memories often contain near-duplicate corrections for the
        # same failure class.  Filling both runtime slots from one error type
        # wastes context and permanently shadows complementary risks.  Select
        # one per error type first, then backfill only if the limit remains.
        selected = []
        selected_ids = set()
        seen_error_types = set()
        limit = max(0, int(memory_limit))
        if limit == 0:
            return RuntimeGuidance(selected_skill, (), render_guidance(selected_skill, ()))
        for item in memories:
            error_type = str(item.get("error_type") or "")
            if error_type in seen_error_types:
                continue
            selected.append(item)
            selected_ids.add(item["memory_id"])
            seen_error_types.add(error_type)
            if len(selected) == limit:
                break
        if len(selected) < limit:
            for item in memories:
                if item["memory_id"] in selected_ids:
                    continue
                selected.append(item)
                if len(selected) == limit:
                    break
        selected_memories = tuple(selected)
        return RuntimeGuidance(selected_skill, selected_memories, render_guidance(selected_skill, selected_memories))


def build_guidance(bank: KnowledgeBank, state: HarnessState, *, memory_limit: int = 2) -> RuntimeGuidance:
    return bank.match(state, memory_limit=memory_limit)


def render_guidance(skill: Mapping | None, memories: Sequence[Mapping]) -> str:
    if skill is None and not memories:
        return ""
    lines = ["[HARNESS_GUIDANCE]", "以下是经过验证的运行时提示；仍须遵守系统规则、当前 observation 和 Action Guard。"]
    if skill is not None:
        lines.append(f"流程 Skill：{skill['name']}")
        lines.extend(f"{index}. {step}" for index, step in enumerate(skill["procedure"], 1))
        lines.extend(f"避免捷径：{item}" for item in skill.get("forbidden_shortcuts", []))
    for memory in memories:
        lines.append(f"避错 Memory：{memory['lesson']}")
    return "\n".join(lines)


def _matches(trigger: Mapping, values: Mapping) -> bool:
    if trigger.get("state") != values.get("state"):
        return False
    for predicate in trigger.get("predicates", []):
        if not _predicate_matches(str(predicate), values):
            return False
    return True


_PREDICATE = re.compile(r"^([a-z_]+)\s*(==|>=|<=|>|<)?\s*(.*)$", re.I)


def _predicate_matches(predicate: str, values: Mapping) -> bool:
    match = _PREDICATE.match(predicate.strip())
    if not match:
        return False
    key, operator, raw = match.groups()
    key = key.casefold()
    if key not in values:
        return False
    actual = values[key]
    if operator is None:
        return bool(actual)
    raw = raw.strip().strip('"\'').casefold()
    if isinstance(actual, set):
        return raw in {str(item).casefold() for item in actual}
    if isinstance(actual, bool):
        expected = raw in {"true", "1", "yes"}
    else:
        try:
            expected = float(raw) if isinstance(actual, (int, float)) else raw
        except ValueError:
            expected = raw
    try:
        return {"==": actual == expected, ">=": actual >= expected, "<=": actual <= expected, ">": actual > expected, "<": actual < expected}[operator]
    except (TypeError, KeyError):
        return False


def _buttons(observation: str) -> list[str]:
    match = re.search(r"可点击的按钮:\s*(\[[^\n]*\])", observation)
    if not match:
        return []
    import json
    try:
        values = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in values if isinstance(item, str)]


def _page_type(observation: str) -> str | None:
    match = re.search(r"(?m)^page_type:\s*([^\s]+)\s*$", str(observation or ""))
    return match.group(1).strip().casefold() if match else None


def _product_buttons(buttons: Sequence[str]) -> tuple[str, ...]:
    """Return visible product targets without treating navigation as candidates."""
    return tuple(item for item in buttons if re.fullmatch(r"\d{8,12}", item.strip()))


def _action_signature(tool_name: str | None, parameters: Mapping | None) -> str:
    if not tool_name:
        return ""
    clean = {str(key): value for key, value in (parameters or {}).items() if key != "note"}
    import json
    return json.dumps([tool_name, clean], ensure_ascii=False, sort_keys=True, default=str)
