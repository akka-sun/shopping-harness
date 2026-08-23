import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
from shopping_grpo.training.grpo.adapter.runtime import (
    current_environment,
    current_runtime_state,
    make_runtime_state,
)
from shopping_grpo.training.grpo.adapter.tools import ShopSimulatorTool
from shopping_grpo.training.grpo.evidence import capture_public_snapshot


def _tool(name):
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    try:
        from verl.tools.schemas import OpenAIFunctionToolSchema
    except ImportError:
        tool_schema = schema
    else:
        tool_schema = OpenAIFunctionToolSchema.model_validate(schema)
    return ShopSimulatorTool({}, tool_schema)


def _search_state(*asins):
    return {
        "observation_version": "shopping-observation-v2",
        "page_type": "search_results",
        "normalized_query": "mug",
        "page": 1,
        "products": [{"asin": asin} for asin in asins],
    }


def test_capture_public_snapshot_uses_only_public_state():
    snapshot = capture_public_snapshot(_search_state("12345678", "87654321"))
    assert len(snapshot.product_hashes) == 2
    assert len(snapshot.search_result_hashes) == 1
    assert snapshot.page_type == "search_results"


def test_generation_boundary_records_stable_turn_span_and_neutral_event():
    async def parent(_loop, agent_data, sampling_params, ignore_termination=False):
        agent_data.response_mask.extend([1, 1, 1])
        return AgentState.PROCESSING_TOOLS

    async def run():
        loop = object.__new__(ShoppingToolAgentLoop)
        loop.context_compaction_enable = False
        loop.context_window_tokens = 100
        loop.context_generation_reserve_tokens = 10
        loop.context_safety_margin_tokens = 10
        loop.context_input_budget = 80
        state = make_runtime_state(1, 35)
        token = current_runtime_state.set(state)
        data = SimpleNamespace(
            prompt_ids=[], response_mask=[], response_logprobs=[], routed_experts=None
        )
        try:
            with patch.object(ToolAgentLoop, "_handle_generating_state", parent):
                result = await loop._handle_generating_state(data, {})
            return result, state
        finally:
            current_runtime_state.reset(token)

    result, state = asyncio.run(run())
    assert result == AgentState.PROCESSING_TOOLS
    assert state["assistant_turn_spans"] == [
        {"assistant_turn_id": 0, "start": 0, "end": 3}
    ]
    assert state["evidence_events"][0]["assistant_turn_id"] == 0
    assert state["evidence_events"][0]["action_attempted"] is False
    assert state["active_assistant_turn_id"] == 0


def test_guard_rejection_replaces_neutral_event_for_same_turn():
    state = make_runtime_state(1, 35)
    state["latest_observation"] = "page_type: search_results"
    state["_public_snapshot"] = capture_public_snapshot(_search_state("12345678"))
    state["active_assistant_turn_id"] = 3
    state["evidence_events"] = [{"assistant_turn_id": 3}]
    env_token = current_environment.set(object())
    state_token = current_runtime_state.set(state)
    try:
        asyncio.run(_tool("open_product").execute("x", {"asin": "99999999"}))
    finally:
        current_runtime_state.reset(state_token)
        current_environment.reset(env_token)
    event = state["evidence_events"][0]
    assert event["assistant_turn_id"] == 3
    assert event["action_attempted"] is True
    assert event["guard_rejection"] is True


def test_compaction_marks_credit_alignment_invalid():
    state = make_runtime_state(1, 35)
    state["context_compactions"] = 1
    ShoppingToolAgentLoop._finalize_credit_metadata(state, response_length=4)
    assert state["credit_alignment_valid"] is False
    assert state["credit_alignment_reason"] == "context_compaction_alignment_unsupported"
