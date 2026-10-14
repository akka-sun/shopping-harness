#!/usr/bin/env python3
"""Compare a Harness qualification run with its no-Harness SFT baseline."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def by_task(path: Path) -> dict[int, dict]:
    result = {}
    for row in rows(path):
        task_id = int(row["task_id"])
        if task_id in result and not _infrastructure_failure(result[task_id]):
            raise SystemExit(f"duplicate non-infrastructure task_id in {path}: {task_id}")
        result[task_id] = row
    return result


def _infrastructure_failure(row: dict) -> bool:
    error_type = str((row.get("error") or {}).get("type") or "")
    return bool(row.get("release_error")) or error_type in {
        "ShopHttpError", "ShopEnvironmentError", "URLError",
        "RemoteDisconnected", "TimeoutError",
    }


def outcome(row: dict) -> str:
    terminal = row.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    return str(detail.get("reward_type") or row.get("status") or "unknown")


def strict(row: dict) -> bool:
    terminal = row.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    return (
        row.get("status") == "done"
        and terminal.get("done") is True
        and terminal.get("over") is True
        and detail.get("reward_type") == "gold_purchase"
        and detail.get("reward_valid") is True
        and detail.get("reward_version") == "shopsimulator-reward-v3"
    )


def guidance_ids(row: dict) -> tuple[set[str], set[str]]:
    skills, memories = set(), set()
    for hit in row.get("harness_guidance") or []:
        if hit.get("skill_id"):
            skills.add(str(hit["skill_id"]))
        memories.update(str(value) for value in hit.get("memory_ids") or [] if value)
    return skills, memories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, harness = by_task(args.baseline), by_task(args.harness)
    if set(baseline) != set(harness):
        raise SystemExit(
            f"task sets differ: baseline_only={sorted(set(baseline)-set(harness))}, "
            f"harness_only={sorted(set(harness)-set(baseline))}"
        )
    gains, regressions, stable_success, stable_failure = [], [], [], []
    transitions = Counter()
    artifact_stats: dict[str, Counter] = defaultdict(Counter)
    detail = []
    for task_id in sorted(baseline):
        before, after = baseline[task_id], harness[task_id]
        before_ok, after_ok = strict(before), strict(after)
        label = (
            "gain" if not before_ok and after_ok else
            "regression" if before_ok and not after_ok else
            "stable_success" if before_ok else "stable_failure"
        )
        {"gain": gains, "regression": regressions,
         "stable_success": stable_success, "stable_failure": stable_failure}[label].append(task_id)
        transitions[(outcome(before), outcome(after))] += 1
        skill_ids, memory_ids = guidance_ids(after)
        for artifact_id in skill_ids | memory_ids:
            artifact_stats[artifact_id][label] += 1
            artifact_stats[artifact_id]["tasks"] += 1
        detail.append({
            "task_id": task_id,
            "classification": label,
            "baseline_outcome": outcome(before),
            "harness_outcome": outcome(after),
            "baseline_status": before.get("status"),
            "harness_status": after.get("status"),
            "baseline_steps": len(before.get("steps") or []),
            "harness_steps": len(after.get("steps") or []),
            "skill_ids": sorted(skill_ids),
            "memory_ids": sorted(memory_ids),
        })
    report = {
        "tasks": len(baseline),
        "baseline_strict_successes": sum(map(strict, baseline.values())),
        "harness_strict_successes": sum(map(strict, harness.values())),
        "net_improvement": len(gains) - len(regressions),
        "gains": gains,
        "regressions": regressions,
        "stable_successes": stable_success,
        "stable_failures": stable_failure,
        "outcome_transitions": [
            {"baseline": before, "harness": after, "count": count}
            for (before, after), count in sorted(transitions.items())
        ],
        "artifact_associations": [
            {"artifact_id": artifact_id, **dict(counts)}
            for artifact_id, counts in sorted(
                artifact_stats.items(),
                key=lambda item: (-item[1]["regression"], -item[1]["gain"], item[0]),
            )
        ],
        "task_comparison": detail,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "tasks", "baseline_strict_successes", "harness_strict_successes",
        "net_improvement", "gains", "regressions")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
