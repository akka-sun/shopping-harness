#!/usr/bin/env python3
"""Build evidence-balanced, cross-task clusters for Harness Skill induction."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from Harness.contracts import validate_memory
from Harness.pipeline import read_jsonl
from Harness.prompts import actor_visible_trajectory


def _strict_gold(row: dict) -> bool:
    terminal = row.get("terminal") or {}
    detail = terminal.get("reward_detail") or {}
    return (
        row.get("status") == "done"
        and terminal.get("done") is True
        and terminal.get("over") is True
        and detail.get("reward_type") == "gold_purchase"
        and detail.get("reward_valid") is True
    )


def _tools(row: dict) -> set[str]:
    return {
        str(event.get("tool_name"))
        for event in row.get("events", [])
        if isinstance(event, dict) and event.get("tool_name")
    }


def _public_size(row: dict) -> int:
    return len(json.dumps(actor_visible_trajectory(row), ensure_ascii=False))


def build_clusters(
    trajectories: list[dict], memories: list[dict], *, max_failures: int
) -> list[dict]:
    by_trajectory = {str(row.get("trajectory_id")): row for row in trajectories}
    if len(by_trajectory) != len(trajectories):
        raise ValueError("trajectory_id values must be present and unique")
    task_by_trajectory = {
        trajectory_id: str(row.get("task_id") or trajectory_id)
        for trajectory_id, row in by_trajectory.items()
    }

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    seen_memory_ids: set[str] = set()
    for raw in memories:
        memory = validate_memory(raw)
        memory_id = memory["memory_id"]
        if memory_id in seen_memory_ids:
            raise ValueError(f"duplicate memory_id: {memory_id}")
        seen_memory_ids.add(memory_id)
        source = memory["source_trajectory_id"]
        if source not in by_trajectory:
            raise ValueError(f"memory {memory_id!r} has unknown source trajectory {source!r}")
        key = (memory["phase"], memory["error_type"], memory["trigger"]["state"])
        groups[key].append(memory)

    gold = [row for row in trajectories if _strict_gold(row)]
    gold.sort(key=lambda row: (_public_size(row), str(row["trajectory_id"])))
    clusters = []
    for key, group in sorted(groups.items()):
        # A single failure plus an unrelated success is not enough evidence for a
        # reusable procedure.  Require two independently sourced Memories first.
        distinct = []
        used_tasks = set()
        for memory in sorted(group, key=lambda item: (-item["confidence"], item["memory_id"])):
            task = task_by_trajectory[memory["source_trajectory_id"]]
            if task in used_tasks:
                continue
            used_tasks.add(task)
            distinct.append(memory)
            if len(distinct) == max_failures:
                break
        if len(distinct) < 2:
            continue

        source_rows = [by_trajectory[item["source_trajectory_id"]] for item in distinct]
        common_tools = Counter(tool for row in source_rows for tool in _tools(row))
        source_ids = {item["source_trajectory_id"] for item in distinct}
        # Prefer a short strict-success trajectory that exercises the same tools.
        success = max(
            (
                row for row in gold
                if row["trajectory_id"] not in source_ids
                and task_by_trajectory[row["trajectory_id"]] not in used_tasks
            ),
            key=lambda row: (
                sum(common_tools[tool] for tool in _tools(row)),
                -_public_size(row),
                str(row["trajectory_id"]),
            ),
            default=None,
        )
        trajectory_ids = [item["source_trajectory_id"] for item in distinct]
        if success is not None:
            trajectory_ids.append(str(success["trajectory_id"]))
        phase, error_type, state = key
        digest = hashlib.sha256("\0".join(trajectory_ids).encode()).hexdigest()[:10]
        clusters.append(
            {
                "cluster_id": f"{phase}--{error_type}--{state}--{digest}",
                "trajectory_ids": trajectory_ids,
                "memory_ids": [item["memory_id"] for item in distinct],
                "evidence": {
                    "phase": phase,
                    "error_type": error_type,
                    "trigger_state": state,
                    "failure_memory_count": len(distinct),
                    "strict_success_count": int(success is not None),
                },
            }
        )
    return clusters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-failures", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.max_failures < 2:
        raise SystemExit("--max-failures must be at least 2")
    if args.output.exists() and not args.force:
        raise SystemExit(f"output already exists: {args.output}; pass --force to replace it")
    clusters = build_clusters(
        read_jsonl(args.trajectories),
        read_jsonl(args.memories),
        max_failures=args.max_failures,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in clusters:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"clusters": len(clusters), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
