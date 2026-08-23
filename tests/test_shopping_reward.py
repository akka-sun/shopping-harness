"""Shopping GRPO Reward v3 terminal-contract tests."""

import unittest

from shopping_grpo.training.grpo.adapter.runtime import (
    make_runtime_state,
    record_action_attempt,
    reward_breakdown,
    validate_reward,
)


def reward_detail(
    *,
    reward_type="gold_purchase",
    terminal_utility=1.0,
    reward_valid=True,
    purchase_success=True,
    weighted_score=1.0,
):
    status = "pass" if reward_valid else "unverifiable"
    return {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": reward_type,
        "reward_valid": reward_valid,
        "termination_reason": reward_type,
        "target_asin_match": reward_type == "gold_purchase",
        "hard_gates": {
            "category": {
                "status": status,
                "passed": reward_valid,
                "verifiable": reward_valid,
                "comparator": "category_leaf_ancestor_chain",
                "source_field": "category",
            },
            "budget": {
                "status": status,
                "passed": reward_valid,
                "verifiable": reward_valid,
                "comparator": "numeric_lte",
                "source_field": "price",
            },
        },
        "weighted_score": weighted_score,
        "evidence_coverage": 1.0 if reward_valid else 0.0,
        "dimension_scores": {
            "brand": weighted_score,
            "model": weighted_score,
            "core_functions": weighted_score,
            "key_options": weighted_score,
        },
        "terminal_utility": terminal_utility,
        "purchase_success": purchase_success,
        "sampling_invalid": not reward_valid,
    }


def terminal_state(*, detail=None):
    detail = validate_reward(detail or reward_detail())
    state = make_runtime_state(task_id=1, max_steps=35)
    state.update(
        {
            "done": True,
            "terminal_result": {"done": True, "over": True},
            "final_reward": detail["terminal_utility"],
            "reward_version": detail["reward_version"],
            "reward_type": detail["reward_type"],
            "reward_valid": detail["reward_valid"],
            "reward_unverifiable": not detail["reward_valid"],
            "reward_detail": detail,
        }
    )
    return state


class ShoppingRewardTest(unittest.TestCase):
    def test_gold_purchase_uses_verified_terminal_utility(self):
        result = reward_breakdown(terminal_state())

        self.assertEqual(result["full"], 1.0)
        self.assertEqual(result["strict"], 1.0)
        self.assertEqual(result["semantic"], 1.0)
        self.assertEqual(result["terminal_utility"], 1.0)
        self.assertEqual(result["total"], 1.0)
        self.assertFalse(result["sampling_invalid"])

    def test_valid_alternative_is_not_strict_success(self):
        state = terminal_state(
            detail=reward_detail(
                reward_type="valid_alternative_purchase",
                terminal_utility=0.55,
                purchase_success=True,
                weighted_score=0.8,
            )
        )

        result = reward_breakdown(state)

        self.assertEqual(result["full"], 0.0)
        self.assertEqual(result["strict"], 0.0)
        self.assertEqual(result["purchase_success"], 1.0)
        self.assertEqual(result["terminal_utility"], 0.55)

    def test_reward_v3_requires_complete_finite_bounded_fields(self):
        with self.assertRaisesRegex(ValueError, "reward_type"):
            validate_reward({"reward_version": "shopsimulator-reward-v3"})

        non_finite = reward_detail()
        non_finite["terminal_utility"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_reward(non_finite)

        out_of_range = reward_detail()
        out_of_range["weighted_score"] = 1.1
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            validate_reward(out_of_range)

    def test_reward_unverifiable_is_sampling_invalid(self):
        state = terminal_state(
            detail=reward_detail(
                reward_type="reward_unverifiable",
                terminal_utility=0.0,
                reward_valid=False,
                purchase_success=False,
                weighted_score=0.0,
            )
        )

        result = reward_breakdown(state)

        self.assertEqual(result["total"], 0.0)
        self.assertTrue(result["sampling_invalid"])
        self.assertTrue(result["reward_unverifiable"])

    def test_same_action_on_same_page_within_three_attempts_is_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")
        record_action_attempt(state, "open_product", {"asin": "123"}, "search page")
        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")

        self.assertEqual(state["action_attempt_count"], 3)
        self.assertEqual(state["repeat_action_count"], 1)
        self.assertAlmostEqual(reward_breakdown(state)["repeat_action_rate"], 1 / 3)

    def test_different_parameters_or_page_are_not_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "search_products", {"query": "mug"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "cup"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "mug"}, "page 2")

        self.assertEqual(state["repeat_action_count"], 0)

    def test_think_is_not_an_environment_action_attempt(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "think", {"note": "plan"}, "page")

        self.assertEqual(state["action_attempt_count"], 0)
        self.assertEqual(state["recent_action_signatures"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
