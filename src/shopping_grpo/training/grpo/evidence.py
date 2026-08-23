"""Public-only environment deltas for Assistant action turns."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

EVIDENCE_VERSION = "shopping-evidence-v1"


def _hash_values(values: Iterable[object]) -> frozenset[str]:
    return frozenset(hashlib.sha256(str(value).encode("utf-8")).hexdigest() for value in values)


@dataclass(frozen=True)
class PublicStateSnapshot:
    product_hashes: frozenset[str]
    search_result_hashes: frozenset[str]
    page_type: str
    visited_subpage_hashes: frozenset[str]
    confirmed_option_hashes: frozenset[str]

    @classmethod
    def from_public_state(cls, products, searches, page_type, subpages, options):
        return cls(
            _hash_values(products),
            _hash_values(searches),
            str(page_type),
            _hash_values(subpages),
            _hash_values(options),
        )

    @property
    def fingerprint(self) -> str:
        payload = "|".join(
            (
                *sorted(self.product_hashes),
                *sorted(self.search_result_hashes),
                self.page_type,
                *sorted(self.visited_subpage_hashes),
                *sorted(self.confirmed_option_hashes),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def capture_public_snapshot(state: object) -> PublicStateSnapshot:
    """Capture hashed navigation evidence from an observation-v2 public object."""
    if not isinstance(state, Mapping):
        state = {}
    products = []
    for product in state.get("products") or []:
        if isinstance(product, Mapping) and product.get("asin") is not None:
            products.append(product["asin"])
    page_type = str(state.get("page_type") or "unknown")
    searches = []
    if page_type == "search_results":
        searches.append(
            (
                state.get("normalized_query", state.get("query", "")),
                state.get("page", 1),
                tuple(products),
            )
        )
    subpages = []
    if page_type == "information_subpage":
        subpages.append(
            (
                state.get("asin", state.get("product_id", "")),
                state.get("subpage", state.get("section", "")),
            )
        )
    raw_options = state.get("selected_options") or state.get("confirmed_options") or {}
    if isinstance(raw_options, Mapping):
        options = sorted(raw_options.items())
    elif isinstance(raw_options, (list, tuple, set, frozenset)):
        options = list(raw_options)
    else:
        options = [raw_options] if raw_options else []
    return PublicStateSnapshot.from_public_state(
        products, searches, page_type, subpages, options
    )


@dataclass(frozen=True)
class EvidenceEvent:
    assistant_turn_id: int
    action_attempted: bool
    new_search_result_set: bool
    new_product_count: int
    first_product_open: bool
    first_info_subpage: bool
    first_option_confirmation: bool
    guard_rejection: bool
    exact_repeat: bool
    no_progress: bool
    schema_version: str = EVIDENCE_VERSION

    def to_dict(self):
        return asdict(self)


def build_evidence_event(
    assistant_turn_id: int,
    before: PublicStateSnapshot,
    after: PublicStateSnapshot,
    *,
    action_attempted: bool,
    guard_rejection: bool = False,
    exact_repeat: bool = False,
    terminal: bool = False,
) -> EvidenceEvent:
    new_products = after.product_hashes - before.product_hashes
    new_search = bool(after.search_result_hashes - before.search_result_hashes)
    first_open = before.page_type != "product_detail" and after.page_type == "product_detail"
    first_subpage = bool(after.visited_subpage_hashes - before.visited_subpage_hashes)
    first_option = bool(after.confirmed_option_hashes - before.confirmed_option_hashes)
    positive = new_search or bool(new_products) or first_open or first_subpage or first_option
    no_progress = bool(
        action_attempted and not positive and before.fingerprint == after.fingerprint and not terminal
    )
    return EvidenceEvent(
        assistant_turn_id=int(assistant_turn_id),
        action_attempted=bool(action_attempted),
        new_search_result_set=new_search,
        new_product_count=len(new_products),
        first_product_open=first_open,
        first_info_subpage=first_subpage,
        first_option_confirmation=first_option,
        guard_rejection=bool(guard_rejection),
        exact_repeat=bool(exact_repeat),
        no_progress=no_progress,
    )


def validate_evidence_event(raw: object) -> EvidenceEvent:
    if not isinstance(raw, dict):
        raise TypeError("evidence event must be an object")
    allowed = {field.name for field in EvidenceEvent.__dataclass_fields__.values()}
    if set(raw) != allowed:
        raise ValueError("evidence event fields do not match shopping-evidence-v1")
    event = EvidenceEvent(**raw)
    if event.schema_version != EVIDENCE_VERSION:
        raise ValueError("unknown evidence event version")
    if event.assistant_turn_id < 0 or event.new_product_count < 0:
        raise ValueError("evidence event counters must be non-negative")
    return event
