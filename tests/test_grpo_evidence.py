from shopping_grpo.training.grpo.evidence import (
    PublicStateSnapshot,
    build_evidence_event,
)


def snapshot(products=(), searches=(), page="search_results", subpages=(), options=()):
    return PublicStateSnapshot.from_public_state(products, searches, page, subpages, options)


def test_new_search_results_are_positive():
    before = snapshot(products=("p1",), searches=("s1",))
    after = snapshot(products=("p1", "p2"), searches=("s1", "s2"))
    event = build_evidence_event(4, before, after, action_attempted=True)
    assert event.new_search_result_set is True
    assert event.new_product_count == 1
    assert event.no_progress is False


def test_repeat_guard_without_state_change_is_negative():
    state = snapshot()
    event = build_evidence_event(
        2, state, state, action_attempted=True, exact_repeat=True, guard_rejection=True
    )
    assert event.exact_repeat and event.guard_rejection and event.no_progress


def test_serialized_event_has_no_hidden_or_raw_fields():
    event = build_evidence_event(1, snapshot(), snapshot(), action_attempted=False)
    payload = event.to_dict()
    forbidden = {"target_asin", "gold", "hidden_constraints", "observation", "parameters"}
    assert forbidden.isdisjoint(payload)
