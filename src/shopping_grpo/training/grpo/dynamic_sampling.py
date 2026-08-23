"""Pure reward-group selection used by the bounded veRL sampling patch."""

from __future__ import annotations

import json
import math
import hashlib
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from shopping_grpo.training.grpo.curriculum.state import CurriculumState
from shopping_grpo.training.grpo.curriculum.sampler import (
    CurriculumSampler,
    SamplerConfig,
    TaskCandidate,
)
from shopping_grpo.training.grpo.curriculum.priority import PrioritySignals, compute_task_priority
from shopping_grpo.training.grpo.curriculum.state import GroupOutcome
from shopping_grpo.training.grpo.failure_types import normalize_failure_type
from shopping_grpo.training.grpo.evidence import validate_evidence_event
from shopping_grpo.training.grpo.trajectory_credit import (
    CreditConfig,
    TurnSpan,
    project_token_advantages,
)

CURRICULUM_SIDECAR = "shopping_curriculum_state.json"
_SAFE_DIAGNOSTIC_KEYS = {
    "stage", "switch_step", "category_counts", "category_probabilities",
    "uniform_count", "dynamic_count", "priority_summary", "failure_proportions",
    "fallback_count", "fallback_reasons", "unique_task_rate", "valid_group_yield",
    "skipped_updates", "evidence_proportions", "multiplier_summary", "mass_error",
}


def sanitize_curriculum_diagnostic(payload: Mapping[str, object]) -> dict[str, object]:
    """Allowlist aggregate scheduler diagnostics; never serialize trajectory content."""
    return {key: payload[key] for key in _SAFE_DIAGNOSTIC_KEYS if key in payload}


def save_curriculum_sidecar(checkpoint_dir: str | Path, state: CurriculumState) -> Path:
    destination = Path(checkpoint_dir) / CURRICULUM_SIDECAR
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state.to_checkpoint(), ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_curriculum_sidecar(
    checkpoint_dir: str | Path,
    manifest_hash: str,
    config_hash: str,
    total_steps: int,
    *,
    required: bool,
) -> CurriculumState | None:
    source = Path(checkpoint_dir) / CURRICULUM_SIDECAR
    if not source.is_file():
        if required:
            raise FileNotFoundError(f"required curriculum sidecar is missing: {source}")
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    return CurriculumState.from_checkpoint(payload, manifest_hash, config_hash, total_steps)


def apply_evidence_credit(batch, config: Mapping[str, object]) -> dict[str, object]:
    """Apply public turn credit only to the existing response-token advantages."""
    if not bool(config.get("enable", False)):
        return {"applied": 0, "fallback_count": 0, "fallback_reasons": {}}
    advantages = batch.batch["advantages"]
    response_mask = batch.batch["response_mask"]
    shopping_infos = batch.non_tensor_batch["shopping"].tolist()
    if len(shopping_infos) != advantages.shape[0]:
        raise ValueError("shopping metadata must align with advantage rows")
    credit_config = CreditConfig(
        positive_evidence_score=float(config.get("positive_evidence_score", 1.0)),
        no_progress_score=float(config.get("no_progress_score", -0.5)),
        exact_repeat_score=float(config.get("exact_repeat_score", -1.0)),
        guard_rejection_score=float(config.get("guard_rejection_score", -1.0)),
        strength=float(config.get("strength", 0.5)),
        multiplier_min=float(config.get("multiplier_min", 0.5)),
        multiplier_max=float(config.get("multiplier_max", 1.5)),
        conservation_tolerance=float(config.get("conservation_tolerance", 1e-6)),
    )
    updated = advantages.clone()
    fallback_reasons: dict[str, int] = {}
    applied = 0
    mass_error = 0.0
    for index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping):
            reason = "missing_shopping_metadata"
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
            continue
        try:
            events = [validate_evidence_event(raw) for raw in info.get("evidence_events", [])]
            spans = [TurnSpan(**raw) for raw in info.get("assistant_turn_spans", [])]
            active = response_mask[index].bool()
            values = advantages[index][active]
            trajectory_advantage = float(values[0].item()) if values.numel() else 0.0
            if values.numel() and not bool((values == values[0]).all()):
                raise ValueError("non_scalar_trajectory_advantage")
            result = project_token_advantages(
                trajectory_advantage,
                response_mask[index],
                spans,
                events,
                credit_config,
                compacted=not bool(info.get("credit_alignment_valid", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            reason = str(exc) or exc.__class__.__name__
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
            continue
        if result.fallback_reason:
            fallback_reasons[result.fallback_reason] = fallback_reasons.get(result.fallback_reason, 0) + 1
            continue
        updated[index] = result.token_advantages.to(dtype=advantages.dtype, device=advantages.device)
        applied += 1
        mass_error = max(mass_error, float(result.mass_error))
    if applied:
        batch.batch["advantages"] = updated
    return {
        "applied": applied,
        "fallback_count": sum(fallback_reasons.values()),
        "fallback_reasons": fallback_reasons,
        "mass_error": mass_error,
    }


class EvidenceCurriculumRuntime:
    """Driver-side adapter joining a task dataset to the pure curriculum APIs."""

    def __init__(self, config, dataset, collate_fn, total_steps: int):
        self.config = dict(config)
        self.total_steps = int(total_steps)
        manifest_path = Path(str(self.config["manifest_path"]))
        self.manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        config_payload = json.dumps(self.config, sort_keys=True, default=str)
        self.config_hash = hashlib.sha256(config_payload.encode()).hexdigest()
        self.records = {}
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                self.records[int(row["task_id"])] = row
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.row_by_task_id = {}
        for index in range(len(dataset)):
            row = dataset[index]
            extra = row.get("extra_info", {})
            task_id = int(extra["task_id"])
            if task_id in self.row_by_task_id:
                raise ValueError(f"duplicate train dataset task_id: {task_id}")
            self.row_by_task_id[task_id] = index
        if set(self.row_by_task_id) != set(self.records):
            raise ValueError("difficulty manifest must exactly cover the train dataset")
        self.state = CurriculumState.empty(self.manifest_hash, self.config_hash, total_steps)

    def _candidate(self, task_id: int, global_step: int) -> TaskCandidate:
        row = self.records[task_id]
        category = str(row["difficulty_category"])
        relevant = [cell for (cat, _), cell in self.state.cells.items() if cat == category]
        trajectories = sum(cell.valid_trajectories for cell in relevant)
        strict = sum(cell.strict_successes for cell in relevant)
        failures = {name: 0 for name in ("near_miss", "wrong_commit", "exploration_stall")}
        for (cat, failure), cell in self.state.cells.items():
            if cat == category and failure in failures:
                failures[failure] += cell.failure_count
        task_stats = self.state.tasks.get(task_id)
        signals = PrioritySignals(
            group_variance=(sum(cell.group_variance_sum for cell in relevant) / max(sum(cell.valid_groups for cell in relevant), 1)),
            strict_success_rate=strict / max(trajectories, 1),
            near_miss_rate=failures["near_miss"] / max(trajectories, 1),
            wrong_commit_rate=failures["wrong_commit"] / max(trajectories, 1),
            steps_since_update=(global_step - task_stats.last_sample_step if task_stats else global_step + 1),
            probe_difficulty=float(row["probe_difficulty"]),
            exploration_stall_rate=failures["exploration_stall"] / max(trajectories, 1),
            consecutive_zero_variance=(task_stats.consecutive_zero_variance if task_stats else 0),
        )
        return TaskCandidate(
            task_id, category, float(row["probe_difficulty"]),
            compute_task_priority(signals, self.config.get("priority_min", .05), self.config.get("priority_max", 4.0)),
        )

    def select_batch(self, baseline_batch, global_step: int, generation_batch_index: int):
        count = len(next(iter(baseline_batch.values())))
        sampler = CurriculumSampler(
            [self._candidate(task_id, global_step) for task_id in sorted(self.records)],
            SamplerConfig(
                seed=int(self.config.get("seed", 20260823)),
                total_steps=self.total_steps,
                switch_ratio=float(self.config.get("switch_ratio", .30)),
                uniform_ratio=float(self.config.get("uniform_ratio", .20)),
            ),
        )
        baseline_ids = [int(item["task_id"]) for item in baseline_batch["extra_info"]]
        selection = sampler.select(global_step, generation_batch_index, count, lambda _: baseline_ids)
        if selection.phase == "baseline_30":
            return baseline_batch, selection
        rows = [self.dataset[self.row_by_task_id[task_id]] for task_id in selection.task_ids]
        return self.collate_fn(rows), selection

    def update_groups(self, group_stats, shopping_infos, global_step: int) -> None:
        for group in group_stats["groups"]:
            if not bool(group.get("kept")):
                continue
            indices = group["indices"]
            first = shopping_infos[indices[0]]
            task_id = int(first["task_id"])
            complete = len(indices) > 0 and all(int(shopping_infos[i]["task_id"]) == task_id for i in indices)
            valid = complete
            failures = tuple(normalize_failure_type(shopping_infos[i]).value for i in indices)
            if any(value == "invalid" for value in failures):
                valid = False
            self.state.update_group(GroupOutcome(
                task_id=task_id,
                difficulty_category=str(self.records[task_id]["difficulty_category"]),
                terminal_utilities=tuple(float(group["terminal_utilities"][j]) for j in range(len(indices))),
                failure_types=failures,
                valid=valid,
            ), global_step)

    def save(self, checkpoint_dir: str | Path, committed_step: int) -> Path:
        self.state.committed_global_step = int(committed_step)
        return save_curriculum_sidecar(checkpoint_dir, self.state)

    def restore(self, checkpoint_dir: str | Path) -> None:
        self.state = load_curriculum_sidecar(
            checkpoint_dir, self.manifest_hash, self.config_hash, self.total_steps, required=True
        )


def build_rollout_diagnostics(
    uids: Sequence[Hashable], shopping_infos: Sequence[object]
) -> list[dict[str, Any]]:
    """Attach stable group/rollout identities to public AgentLoop diagnostics."""
    if len(uids) != len(shopping_infos):
        raise ValueError("uids and shopping_infos must have equal length")
    rollout_counts: dict[Hashable, int] = {}
    records = []
    for index, (uid, info) in enumerate(zip(uids, shopping_infos, strict=True)):
        if not isinstance(info, Mapping):
            raise ValueError(f"shopping extra field at index {index} is not an object")
        rollout_index = rollout_counts.get(uid, 0)
        rollout_counts[uid] = rollout_index + 1
        records.append({"uid": uid, "rollout_index": rollout_index, **dict(info)})
    return records


def append_training_diagnostic(
    path: str | Path | None,
    event: str,
    global_step: int,
    **payload: object,
) -> None:
    """Append one driver-side training event; an unset path disables persistence."""
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def scalar(value):
        item = getattr(value, "item", None)
        if callable(item):
            return item()
        raise TypeError(f"{type(value).__name__} is not JSON serializable")

    record = {
        "schema_version": 1,
        "event": str(event),
        "global_step": int(global_step),
        **payload,
    }
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=scalar))
        handle.write("\n")


def aggregate_shopping_metrics(shopping_infos: Sequence[object]) -> dict[str, float]:
    """把 AgentLoop 轨迹诊断聚合为 veRL 每步指标。"""
    if not shopping_infos:
        return {}

    reward_keys = (
        "full",
        "strict",
        "native",
        "semantic",
        "total",
        "efficiency",
        "penalty_overlong",
        "penalty_unfinished",
        "penalty_repeat",
        "repeat_action_rate",
        "r_type",
        "r_att",
        "r_option",
        "r_price",
    )
    rewards = {key: [] for key in reward_keys}
    steps = []
    done = []
    max_steps = []
    overlong = []
    repeat_loop = []
    infrastructure_invalid = []
    reward_unverifiable = []
    terminal_utilities = []
    purchase_success = []
    sampling_invalid = []
    match_scores = []
    evidence_coverage = []
    partial_purchase = []
    for index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping) or not isinstance(info.get("reward"), Mapping):
            raise ValueError(f"shopping extra field at index {index} is missing reward diagnostics")
        reward = info["reward"]
        for key in reward_keys:
            try:
                value = float(reward[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"shopping reward at index {index} is missing numeric {key}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"shopping reward {key} at index {index} is not finite")
            rewards[key].append(value)
        steps.append(float(info.get("steps", 0)))
        done.append(float(info.get("done") is True))
        max_steps.append(float(info.get("termination_reason") == "max_steps"))
        overlong.append(float(bool(info.get("overlong"))))
        repeat_loop.append(float(info.get("reward_type") == "repeat_loop"))
        infrastructure_invalid.append(float(bool(info.get("infrastructure_invalid"))))
        reward_unverifiable.append(float(bool(info.get("reward_unverifiable"))))
        terminal_utilities.append(
            float(reward.get("terminal_utility", reward["total"]))
        )
        purchase_success.append(
            float(bool(reward.get("purchase_success", reward["full"])))
        )
        sampling_invalid.append(
            float(
                bool(
                    reward.get(
                        "sampling_invalid",
                        info.get("infrastructure_invalid")
                        or info.get("reward_unverifiable"),
                    )
                )
            )
        )
        match_scores.append(float(reward.get("match_score", reward["r_att"])))
        evidence_coverage.append(
            float(reward.get("evidence_coverage", 0.0))
        )
        partial_purchase.append(
            float(info.get("reward_type") == "partial_alternative_purchase")
        )

    def mean(values):
        return sum(values) / len(values)

    return {
        "reward/full_mean": mean(rewards["full"]),
        "reward/strict_mean": mean(rewards["strict"]),
        "reward/native_mean": mean(rewards["native"]),
        "reward/semantic_mean": mean(rewards["semantic"]),
        "reward/shaped_min": min(rewards["total"]),
        "reward/shaped_mean": mean(rewards["total"]),
        "reward/shaped_max": max(rewards["total"]),
        "reward/terminal_utility_min": min(terminal_utilities),
        "reward/terminal_utility_mean": mean(terminal_utilities),
        "reward/terminal_utility_max": max(terminal_utilities),
        "reward/purchase_success_rate": mean(purchase_success),
        "reward/partial_purchase_rate": mean(partial_purchase),
        "reward/match_score_mean": mean(match_scores),
        "reward/evidence_coverage_mean": mean(evidence_coverage),
        "reward/efficiency_mean": mean(rewards["efficiency"]),
        "penalty/overlong_mean": mean(rewards["penalty_overlong"]),
        "penalty/unfinished_mean": mean(rewards["penalty_unfinished"]),
        "penalty/repeat_mean": mean(rewards["penalty_repeat"]),
        "component/r_type_mean": mean(rewards["r_type"]),
        "component/r_att_mean": mean(rewards["r_att"]),
        "component/r_option_mean": mean(rewards["r_option"]),
        "component/r_price_mean": mean(rewards["r_price"]),
        "trajectory/average_steps": mean(steps),
        "trajectory/done_rate": mean(done),
        "trajectory/max_steps_rate": mean(max_steps),
        "trajectory/overlong_rate": mean(overlong),
        "trajectory/repeat_loop_rate": mean(repeat_loop),
        "trajectory/repeat_action_rate": mean(rewards["repeat_action_rate"]),
        "trajectory/infrastructure_invalid_rate": mean(infrastructure_invalid),
        "trajectory/reward_unverifiable_rate": mean(reward_unverifiable),
        "trajectory/sampling_invalid_rate": mean(sampling_invalid),
    }


def extract_shopping_group_signals(
    shopping_infos: Sequence[object],
) -> tuple[list[float], list[bool], list[bool], list[tuple[str, ...]]]:
    """Return terminal utility, success metrics, and explicit invalid reasons."""
    terminal_utilities = []
    purchase_success = []
    sampling_invalid = []
    invalid_reasons = []
    for index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping) or not isinstance(info.get("reward"), Mapping):
            raise ValueError(f"shopping extra field at index {index} is missing reward diagnostics")
        try:
            terminal_utility = float(info["reward"]["terminal_utility"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"shopping extra field at index {index} is missing terminal_utility"
            ) from exc
        if not math.isfinite(terminal_utility):
            raise ValueError(
                f"shopping terminal_utility at index {index} is not finite"
            )
        raw_purchase_success = info["reward"].get("purchase_success")
        if not isinstance(raw_purchase_success, (bool, int, float)):
            raise ValueError(
                f"shopping extra field at index {index} is missing purchase_success"
            )
        if "infrastructure_invalid" not in info:
            raise ValueError(
                f"shopping extra field at index {index} is missing infrastructure_invalid"
            )
        reasons = []
        if bool(info["infrastructure_invalid"]):
            reasons.append("infrastructure_invalid")
        if bool(info.get("reward_unverifiable")):
            reasons.append("reward_unverifiable")
        if bool(info.get("overlong")):
            reasons.append("overlong")
        reward_sampling_invalid = bool(
            info["reward"].get("sampling_invalid", False)
        )
        if reward_sampling_invalid and not reasons:
            reasons.append("reward_sampling_invalid")
        terminal_utilities.append(terminal_utility)
        purchase_success.append(bool(raw_purchase_success))
        sampling_invalid.append(bool(reasons))
        invalid_reasons.append(tuple(reasons))
    return (
        terminal_utilities,
        purchase_success,
        sampling_invalid,
        invalid_reasons,
    )


def select_reward_varying_groups(
    uids: Sequence[Hashable],
    seq_rewards: Sequence[float],
    *,
    terminal_utilities: Sequence[float] | None = None,
    purchase_success: Sequence[bool] | None = None,
    sampling_invalid: Sequence[bool] | None = None,
    sampling_invalid_reasons: Sequence[Sequence[str]] | None = None,
    tolerance: float = 1.0e-8,
) -> tuple[list[int], dict[str, Any]]:
    """Return trajectory indices belonging to groups with non-constant reward.

    Group order follows the first occurrence of each uid. Returned trajectory
    indices preserve their original order, so callers can safely apply the same
    selection to every aligned tensor and non-tensor batch field.
    """

    if len(uids) != len(seq_rewards):
        raise ValueError(
            f"uids and seq_rewards must have equal length, got {len(uids)} and {len(seq_rewards)}"
        )
    optional_sequences = {
        "terminal_utilities": terminal_utilities,
        "purchase_success": purchase_success,
        "sampling_invalid": sampling_invalid,
        "sampling_invalid_reasons": sampling_invalid_reasons,
    }
    for name, values in optional_sequences.items():
        if values is not None and len(values) != len(uids):
            raise ValueError(f"{name} must have the same length as uids")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be a finite non-negative number, got {tolerance!r}")

    utility_values = (
        terminal_utilities if terminal_utilities is not None else seq_rewards
    )
    success_values = (
        purchase_success if purchase_success is not None else [False] * len(uids)
    )
    invalid_values = (
        sampling_invalid if sampling_invalid is not None else [False] * len(uids)
    )
    reason_values = (
        sampling_invalid_reasons
        if sampling_invalid_reasons is not None
        else [()] * len(uids)
    )
    grouped: dict[Hashable, dict[str, Any]] = {}
    for index, (
        uid,
        raw_reward,
        raw_utility,
        raw_success,
        raw_invalid,
        raw_reasons,
    ) in enumerate(
        zip(
            uids,
            seq_rewards,
            utility_values,
            success_values,
            invalid_values,
            reason_values,
            strict=True,
        )
    ):
        try:
            hash(uid)
        except TypeError as exc:
            raise ValueError(f"uid at index {index} is not hashable: {uid!r}") from exc

        reward = float(raw_reward)
        if not math.isfinite(reward):
            raise ValueError(f"seq_reward at index {index} is not finite: {raw_reward!r}")
        utility = float(raw_utility)
        if not math.isfinite(utility):
            raise ValueError(
                f"terminal_utility at index {index} is not finite: {raw_utility!r}"
            )

        group = grouped.setdefault(
            uid,
            {
                "uid": uid,
                "indices": [],
                "rewards": [],
                "terminal_utilities": [],
                "purchase_success": [],
                "sampling_invalid": [],
                "sampling_invalid_reasons": [],
            },
        )
        group["indices"].append(index)
        group["rewards"].append(reward)
        group["terminal_utilities"].append(utility)
        group["purchase_success"].append(bool(raw_success))
        group["sampling_invalid"].append(bool(raw_invalid))
        group["sampling_invalid_reasons"].extend(str(reason) for reason in raw_reasons)

    kept_uids: list[Hashable] = []
    dropped_uids: list[Hashable] = []
    groups: list[dict[str, Any]] = []
    for uid, group in grouped.items():
        utilities = group["terminal_utilities"]
        utility_min = min(utilities)
        utility_max = max(utilities)
        utility_varying = utility_max - utility_min > tolerance
        has_sampling_invalid = any(group["sampling_invalid"])
        reasons = tuple(sorted(set(group["sampling_invalid_reasons"])))
        if has_sampling_invalid:
            drop_reason = "sampling_invalid"
        elif not utility_varying:
            drop_reason = "constant_reward"
        else:
            drop_reason = None
        keep = drop_reason is None
        if keep:
            kept_uids.append(uid)
        else:
            dropped_uids.append(uid)
        groups.append(
            {
                "uid": uid,
                "indices": tuple(group["indices"]),
                "rewards": tuple(group["rewards"]),
                "terminal_utilities": tuple(utilities),
                "purchase_success": tuple(group["purchase_success"]),
                "utility_min": utility_min,
                "utility_max": utility_max,
                "reward_varying": utility_varying,
                "sampling_invalid": has_sampling_invalid,
                "sampling_invalid_reasons": reasons,
                "drop_reason": drop_reason,
                "kept": keep,
            }
        )

    kept_uid_set = set(kept_uids)
    trajectory_indices = [index for index, uid in enumerate(uids) if uid in kept_uid_set]
    stats = {
        "num_trajectories": len(uids),
        "num_groups": len(grouped),
        "kept_group_count": len(kept_uids),
        "dropped_group_count": len(dropped_uids),
        "kept_uids": tuple(kept_uids),
        "dropped_uids": tuple(dropped_uids),
        "all_equal_group_count": sum(
            not group["reward_varying"] for group in groups
        ),
        "all_zero_utility_group_count": sum(
            max(abs(value) for value in group["terminal_utilities"]) <= tolerance
            for group in groups
        ),
        "all_purchase_success_group_count": sum(
            all(group["purchase_success"])
            for group in groups
        ),
        "no_purchase_success_group_count": sum(
            not any(group["purchase_success"]) for group in groups
        ),
        "sampling_invalid_group_count": sum(
            group["sampling_invalid"] for group in groups
        ),
        "sampling_invalid_reason_counts": {
            reason: sum(
                reason in group["sampling_invalid_reasons"] for group in groups
            )
            for reason in sorted(
                {
                    reason
                    for group in groups
                    for reason in group["sampling_invalid_reasons"]
                }
            )
        },
        # Compatibility aliases for existing monitoring code.
        "infrastructure_invalid_group_count": sum(
            group["sampling_invalid"] for group in groups
        ),
        "groups": tuple(groups),
    }
    return trajectory_indices, stats
