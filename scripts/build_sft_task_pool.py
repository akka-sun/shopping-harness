#!/usr/bin/env python3
"""Build a deterministic ShopSimulator task pool for future SFT rollouts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCTS_ARCHIVE = (
    ROOT
    / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
)
DEFAULT_EXCLUSIONS = (
    ROOT / "data/grpo/train.jsonl",
    ROOT / "data/grpo/validation.jsonl",
    ROOT / "data/evaluation/tasks.jsonl",
    ROOT / "data/sft/train.jsonl",
    ROOT / "data/sft/validation.jsonl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_count(products_archive: Path) -> int:
    with gzip.open(products_archive, "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    seen_asins = set()
    count = 0
    for product in products:
        asin = product.get("asin")
        if not isinstance(asin, str) or asin == "nan" or len(asin) > 20:
            continue
        if asin in seen_asins:
            continue
        seen_asins.add(asin)
        count += sum(
            1
            for instruction in product.get("instructions", ())
            if instruction.get("attributes")
        )
    return count


def _task_ids(path: Path) -> set[int]:
    task_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                task_id = int(row["task_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid task_id") from exc
            if task_id in task_ids:
                raise ValueError(f"{path}:{line_number}: duplicate task_id {task_id}")
            task_ids.add(task_id)
    return task_ids


def build_task_pool(
    *, products_archive: Path, exclusion_paths: list[Path], count: int, seed: int
) -> tuple[list[int], dict]:
    total_task_count = _task_count(products_archive)
    excluded = set()
    exclusions = []
    for path in exclusion_paths:
        ids = _task_ids(path)
        excluded.update(ids)
        exclusions.append(
            {"path": str(path), "task_count": len(ids), "sha256": _sha256(path)}
        )
    out_of_range = sorted(task_id for task_id in excluded if task_id not in range(total_task_count))
    if out_of_range:
        raise ValueError(
            f"excluded task IDs outside ShopSimulator range: {out_of_range[:10]}"
        )
    eligible = [task_id for task_id in range(total_task_count) if task_id not in excluded]
    if count > len(eligible):
        raise ValueError(
            f"requested {count} tasks, but only {len(eligible)} eligible tasks remain"
        )
    selected = sorted(random.Random(seed).sample(eligible, count))
    metadata = {
        "schema_version": "shopping-sft-task-pool-v1",
        "environment": "shopsimulator-environment-v2.1",
        "products_archive": {
            "path": str(products_archive),
            "sha256": _sha256(products_archive),
        },
        "seed": seed,
        "source_task_count": total_task_count,
        "excluded_task_count": len(excluded),
        "eligible_task_count": len(eligible),
        "task_count": len(selected),
        "exclusions": exclusions,
    }
    return selected, metadata


def _write_outputs(output_dir: Path, task_ids: list[int], metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = output_dir / "tasks.jsonl"
    tasks_path.write_text(
        "".join(json.dumps({"task_id": task_id}, separators=(",", ":")) + "\n" for task_id in task_ids),
        encoding="utf-8",
    )
    metadata = {
        **metadata,
        "tasks": {
            "path": str(tasks_path),
            "rows": len(task_ids),
            "sha256": _sha256(tasks_path),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products-archive", type=Path, default=DEFAULT_PRODUCTS_ARCHIVE)
    parser.add_argument("--exclude", action="append", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/sft_candidates")
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    exclusions = args.exclude if args.exclude is not None else list(DEFAULT_EXCLUSIONS)
    try:
        task_ids, metadata = build_task_pool(
            products_archive=args.products_archive,
            exclusion_paths=exclusions,
            count=args.count,
            seed=args.seed,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    _write_outputs(args.output_dir, task_ids, metadata)
    print(
        f"wrote {len(task_ids)} tasks to {args.output_dir / 'tasks.jsonl'} "
        f"from {metadata['eligible_task_count']} eligible tasks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
