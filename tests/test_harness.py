import json

import pytest

from Harness.contracts import HarnessValidationError, validate_memory, validate_skill
from Harness.pipeline import build_skill_request_rows, validate_artifact_rows
from Harness.prompts import actor_visible_trajectory, build_memory_messages, build_skill_messages
from Harness.runtime import HarnessState, KnowledgeBank
from Harness.integration import with_guidance
from Harness.semantic import SemanticThresholdGate


def memory(**overrides):
    value = {
        "schema_version": "shopping-memory-v1",
        "memory_id": "mem-1",
        "source_trajectory_id": "traj-1",
        "phase": "verification",
        "trigger": {"state": "product_opened", "predicates": ["purchase_available == true"]},
        "error_type": "critical_evidence_missing",
        "first_bad_event_id": "e0001",
        "lesson": "不要只看标题，购买前核验所有相关约束字段。",
        "risk": "wrong_purchase",
        "confidence": 0.9,
    }
    value.update(overrides)
    return value


def skill(**overrides):
    value = {
        "schema_version": "shopping-skill-v1",
        "skill_id": "skill-1",
        "name": "purchase_preflight",
        "scope": "verification",
        "trigger": {"state": "product_opened", "predicates": ["purchase_available == true"]},
        "procedure": ["核验约束", "确认预算", "再购买"],
        "forbidden_shortcuts": ["不要只看标题"],
        "source_trajectory_ids": ["traj-1", "traj-2"],
        "source_memory_ids": [],
        "confidence": 0.9,
        "status": "active",
    }
    value.update(overrides)
    return value


def test_memory_is_single_trajectory_and_rejects_leak():
    assert validate_memory(memory(), event_ids={"e0001"})["memory_id"] == "mem-1"
    with pytest.raises(HarnessValidationError):
        validate_memory(memory(lesson="购买 123456789012 商品"))
    with pytest.raises(HarnessValidationError):
        validate_memory(memory(lesson="只选择标有 50 年的选项"))
    with pytest.raises(HarnessValidationError):
        validate_memory(memory(lesson="记住上一次看到的 ASIN 再购买"))
    # Provenance UUIDs are checked against local trajectory indexes and must not
    # be mistaken for product IDs merely because one segment is all digits.
    assert validate_memory(
        memory(source_trajectory_id="06876930-5aea-4d50-adbc-fc1553b9b07e")
    )["source_trajectory_id"].startswith("06876930")


def test_skill_requires_multiple_trajectories():
    assert validate_skill(skill())["status"] == "active"
    with pytest.raises(HarnessValidationError):
        validate_skill(skill(source_trajectory_ids=["traj-1"]))
    with pytest.raises(HarnessValidationError):
        validate_skill(
            skill(trigger={"state": "product_opened", "predicates": ["页面有购买按钮"]})
        )
    for unreachable_state in ("candidate_review", "purchase"):
        with pytest.raises(HarnessValidationError):
            validate_skill(skill(trigger={"state": unreachable_state, "predicates": []}))


def test_skill_request_and_validation_preserve_memory_provenance():
    trajectories = [
        {"schema_version": "shopping-normalized-trajectory-v1", "trajectory_id": "traj-1"},
        {"schema_version": "shopping-normalized-trajectory-v1", "trajectory_id": "traj-2"},
    ]
    rows = build_skill_request_rows(
        trajectories,
        [{"cluster_id": "cluster-1", "trajectory_ids": ["traj-1", "traj-2"],
          "memory_ids": ["mem-1"], "evidence": {"phase": "verification"}}],
        [memory()],
    )
    assert rows[0]["memory_ids"] == ["mem-1"]
    assert rows[0]["expected_scope"] == "verification"

    accepted, rejected = validate_artifact_rows(
        [skill(source_memory_ids=["invented-memory"])],
        kind="skill",
        trajectory_ids={"traj-1", "traj-2"},
        memory_ids={"mem-1"},
    )
    assert not accepted
    assert "source_memory_ids are unknown" in rejected[0]["error"]


def test_runtime_matches_skill_and_memory():
    bank = KnowledgeBank([memory()], [skill()])
    state = HarnessState()
    state.update(tool_name="open_product", parameters={}, observation='可点击的按钮: ["Buy Now"]')
    guidance = bank.match(state)
    assert guidance.skill["name"] == "purchase_preflight"
    assert len(guidance.memories) == 1
    assert "核验约束" in guidance.message


def test_runtime_resets_product_local_state_when_candidate_changes():
    state = HarnessState()
    state.update(tool_name="open_product", parameters={"asin": "visible"}, observation="")
    state.update(tool_name="select_option", parameters={"value": "visible"}, observation="")
    state.update(
        tool_name="view_features", parameters={}, observation="", required_fields_verified=True
    )
    assert state.product_opened and state.variant_selected and state.required_fields_verified

    state.update(tool_name="open_product", parameters={"asin": "another"}, observation="")
    assert state.product_opened
    assert not state.variant_selected
    assert state.required_fields_verified is None

    state.update(tool_name="back_to_search", parameters={}, observation="")
    assert not state.product_opened
    assert not state.variant_selected
    assert state.required_fields_verified is None


def test_runtime_enters_ready_to_buy_before_purchase():
    state = HarnessState()
    state.update(tool_name="open_product", parameters={}, observation='可点击的按钮: ["option"]')
    state.update(
        tool_name="select_option",
        parameters={"value": "option"},
        observation='可点击的按钮: ["Buy Now"]',
    )
    assert state.phase == "ready_to_buy"
    assert state.variant_selected and state.purchase_available


def test_runtime_restores_ready_to_buy_after_information_subpage():
    state = HarnessState()
    state.update(tool_name="open_product", parameters={}, observation='可点击的按钮: ["option"]')
    state.update(
        tool_name="select_option", parameters={"value": "option"},
        observation='可点击的按钮: ["Buy Now", "Features"]',
    )
    state.update(tool_name="view_features", parameters={}, observation='可点击的按钮: ["< Prev"]')
    assert state.phase == "verification"
    state.update(tool_name="prev_page", parameters={}, observation='可点击的按钮: ["Buy Now"]')
    assert state.phase == "ready_to_buy"


def test_runtime_prev_page_to_search_results_clears_product_state():
    state = HarnessState()
    state.update(
        tool_name="open_product", parameters={},
        observation='page_type: product_detail\n可点击的按钮: ["option"]',
    )
    state.update(
        tool_name="select_option", parameters={"value": "option"},
        observation='page_type: product_detail\n可点击的按钮: ["Buy Now"]',
    )
    state.update(
        tool_name="prev_page", parameters={},
        observation='page_type: search_results\n可点击的按钮: ["12345678"]',
    )
    assert state.phase == "candidate_found"
    assert not state.product_opened
    assert state.current_product_id is None
    assert not state.variant_selected
    assert state.required_fields_verified is None


def test_runtime_tracks_current_product_from_public_action():
    state = HarnessState()
    state.update(
        tool_name="open_product", parameters={"asin": "12345678"},
        observation='page_type: product_detail\n可点击的按钮: ["Buy Now"]',
    )
    assert state.current_product_id == "12345678"
    state.update(
        tool_name="back_to_search", parameters={},
        observation='page_type: search_results\n可点击的按钮: ["12345678"]',
    )
    assert state.current_product_id is None


def test_memory_retrieval_prefers_distinct_error_types():
    duplicate = memory(
        memory_id="mem-2", error_type="critical_evidence_missing", confidence=0.8,
    )
    complementary = memory(
        memory_id="mem-3", error_type="budget_violation", confidence=0.7,
    )
    bank = KnowledgeBank([memory(), duplicate, complementary])
    state = HarnessState()
    state.update(tool_name="open_product", parameters={}, observation='可点击的按钮: ["Buy Now"]')
    selected = bank.match(state).memories
    assert [item["memory_id"] for item in selected] == ["mem-1", "mem-3"]
    assert bank.match(state, memory_limit=0).memories == ()

def test_teacher_prompts_are_separate_and_skill_requires_two_inputs():
    messages = build_memory_messages({"schema_version": "shopping-normalized-trajectory-v1"})
    assert "只能分析一条" in messages[0]["content"]
    with pytest.raises(ValueError):
        build_skill_messages(
            [{"trajectory_id": "only-one"}], [], cluster_id="c1",
            expected_scope="verification",
        )


def test_public_projection_drops_audit_and_goal_fields():
    projected = actor_visible_trajectory(
        {
            "schema_version": "shopping-normalized-trajectory-v1",
            "trajectory_id": "traj-1",
            "task_id": 42,
            "final_reward": 1.0,
            "events": [
                {
                    "event_id": "e0001",
                    "actor_visible_observation": "public",
                    "raw_observation": "hidden audit",
                    "reward": 1.0,
                }
            ],
            "terminal": {
                "purchase": {"asin": "12345678"},
                "reward_detail": {"reward_type": "gold_purchase"},
            },
        }
    )
    assert "task_id" not in projected
    assert "raw_observation" not in str(projected)
    assert "purchase" not in projected


def test_runtime_loop_counters_reset_on_new_action():
    state = HarnessState()
    state.update(
        tool_name="search_products",
        parameters={"query": "lamp"},
        observation='可点击的按钮: ["12345678"]',
    )
    state.update(
        tool_name="search_products",
        parameters={"query": "lamp"},
        observation='可点击的按钮: ["12345678"]',
    )
    assert state.same_query_count == 1
    assert state.same_action_count == 1
    assert state.no_progress_steps == 1
    state.update(
        tool_name="next_page",
        parameters={},
        observation='可点击的按钮: ["Back to Search"]',
    )
    assert state.same_action_count == 0
    assert state.no_progress_steps == 0


def test_guidance_injection_is_ephemeral():
    bank = KnowledgeBank([memory()], [skill()])
    state = HarnessState()
    state.update(tool_name="open_product", parameters={}, observation='可点击的按钮: ["Buy Now"]')
    guidance = bank.match(state)
    history = [{"role": "system", "content": "rules"}]
    request = with_guidance(history, guidance)
    assert len(history) == 1
    assert request[-1]["role"] == "user"
    assert "HARNESS_GUIDANCE" in request[-1]["content"]


def test_knowledge_bank_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        KnowledgeBank([memory(), memory()])


def test_semantic_threshold_gate_can_skip_all_candidates():
    bank = KnowledgeBank(
        [memory(source_trajectory_id="traj-1")],
        [skill(source_trajectory_ids=["traj-1", "traj-2"])],
        source_queries={
            "traj-1": "Instruction: 购买红色运动鞋",
            "traj-2": "Instruction: 购买蓝色运动鞋",
        },
    )
    state = HarnessState()
    state.update(
        tool_name="open_product", parameters={"asin": "visible"},
        observation='可点击的按钮: ["option", "Buy Now"]',
    )
    # Make the test artifacts reachable without weakening production triggers.
    state.purchase_available = True
    router = SemanticThresholdGate(threshold=0.95)

    guidance = router.select(
        instruction="Instruction: 购买运动鞋",
        observation='title: 运动鞋\n可点击的按钮: ["Buy Now"]',
        state=state,
        bank=bank,
    )

    assert guidance.message == ""
    assert guidance.skill is None
    assert guidance.memories == ()
    assert router.last_decision["selected"] == {
        "skill_id": None, "memory_id": None,
    }


def test_semantic_threshold_gate_selects_relevant_kinds_independently():
    relevant_memory = memory(source_trajectory_id="shoe-memory")
    unrelated_skill = skill(source_trajectory_ids=["phone-skill-1", "phone-skill-2"])
    bank = KnowledgeBank(
        [relevant_memory],
        [unrelated_skill],
        source_queries={
            "shoe-memory": "Instruction: 购买红色运动鞋，尺码42",
            "phone-skill-1": "Instruction: 购买大屏智能手机，内存256GB",
            "phone-skill-2": "Instruction: 购买折叠屏手机，内存512GB",
        },
    )
    state = HarnessState()
    state.update(
        tool_name="open_product", parameters={"asin": "visible"},
        observation='可点击的按钮: ["option", "Buy Now"]',
    )
    state.purchase_available = True
    router = SemanticThresholdGate(
        threshold=0.15, product_weight=0.25, resolved_threshold=0.95,
        skill_threshold=0.95, skill_task_threshold=0.95,
    )

    guidance = router.select(
        instruction="Instruction: 想买红色运动鞋，尺码42",
        observation=("title: 红色运动鞋\ncategory: 鞋靴\n"
                     'selected_options: {"尺码": "42"}\n'
                     '可点击的按钮: ["Buy Now"]'),
        state=state,
        bank=bank,
    )

    assert guidance.skill is None
    assert guidance.memories[0]["memory_id"] == relevant_memory["memory_id"]
    assert router.last_decision["selected"]["skill_id"] is None
