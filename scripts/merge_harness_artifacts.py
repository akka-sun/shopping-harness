#!/usr/bin/env python3
"""Merge artifact JSONL files while requiring globally unique artifact IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("memory", "skill"), required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"output already exists: {args.output}; pass --force to replace it")
    key = f"{args.kind}_id"
    merged = {}
    for path in args.input:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            artifact_id = str(row.get(key) or "")
            if not artifact_id:
                raise SystemExit(f"{path} contains an artifact without {key}")
            if artifact_id in merged:
                raise SystemExit(f"duplicate {key}: {artifact_id}")
            merged[artifact_id] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in merged.values():
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"kind": args.kind, "count": len(merged)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
