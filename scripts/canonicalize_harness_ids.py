#!/usr/bin/env python3
"""Replace teacher-chosen artifact IDs with deterministic provenance IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_harness_knowledge import canonical_artifact_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("memory", "skill"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"output already exists: {args.output}; pass --force to replace it")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    seen = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            if args.kind == "memory":
                source = str(row.get("source_trajectory_id") or "")
            else:
                # Skill requests are clustered, but candidates intentionally do
                # not carry internal cluster IDs.  The sorted evidence set is a
                # stable provenance identity for the final candidate.
                source = "\0".join(sorted(map(str, row.get("source_trajectory_ids", []))))
            if not source:
                raise SystemExit(f"cannot derive {args.kind} ID from artifact provenance")
            artifact_id = canonical_artifact_id(args.kind, source)
            if artifact_id in seen:
                raise SystemExit(f"duplicate {args.kind} provenance: {source}")
            seen.add(artifact_id)
            row[f"{args.kind}_id"] = artifact_id
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"kind": args.kind, "count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
