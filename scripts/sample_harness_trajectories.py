#!/usr/bin/env python3
"""Deterministically sample a balanced subset of normalized Harness trajectories."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def strict_gold(row: dict) -> bool:
    terminal = row.get("terminal") or {}
    detail = terminal.get("reward_detail") or {}
    return (
        row.get("status") == "done"
        and terminal.get("done") is True
        and terminal.get("over") is True
        and detail.get("reward_type") == "gold_purchase"
        and detail.get("reward_valid") is True
    )


def outcome(row: dict) -> str:
    detail = (row.get("terminal") or {}).get("reward_detail") or {}
    return str(detail.get("reward_type") or ("gold_purchase" if strict_gold(row) else "unknown"))


def sample(rows: list[dict], size: int, seed: int) -> list[dict]:
    if size <= 0 or size > len(rows):
        raise ValueError(f"size must be between 1 and {len(rows)}")
    if any(not row.get("trajectory_id") for row in rows):
        raise ValueError("every trajectory must have trajectory_id")
    if len({row["trajectory_id"] for row in rows}) != len(rows):
        raise ValueError("trajectory_id values must be unique")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[outcome(row)].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected: list[dict] = []
    # Round-robin strata keeps rare terminal outcomes represented when possible.
    keys = sorted(groups)
    while len(selected) < size and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < size:
                selected.append(groups[key].pop())
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"output already exists: {args.output}; pass --force")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen = sample(rows, args.size, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in chosen), encoding="utf-8")
    counts = defaultdict(int)
    for row in chosen:
        counts[outcome(row)] += 1
    print(json.dumps({"input": len(rows), "sampled": len(chosen), "seed": args.seed, "outcomes": dict(sorted(counts.items()))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
