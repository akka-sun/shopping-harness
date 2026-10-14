#!/usr/bin/env python3
"""Select teacher requests corresponding to validator-rejected artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_harness_knowledge import canonical_artifact_id, request_id


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("memory", "skill"), required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--rejected", type=Path)
    parser.add_argument(
        "--errors",
        type=Path,
        help="generation error JSONL containing _request_id values",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"output already exists: {args.output}; pass --force to replace it")
    if not args.rejected and not args.errors:
        raise SystemExit("at least one of --rejected or --errors is required")
    rejected_ids = set()
    if args.rejected:
        rejected_ids = {
            str((row.get("artifact") or {}).get(f"{args.kind}_id") or "")
            for row in _rows(args.rejected)
        }
        rejected_ids.discard("")
    error_request_ids = set()
    if args.errors:
        error_request_ids = {
            str(row.get("_request_id") or "") for row in _rows(args.errors)
        }
        error_request_ids.discard("")
    requests = _rows(args.requests)
    selected = [
        row for row in requests
        if (
            canonical_artifact_id(args.kind, request_id(row)) in rejected_ids
            or request_id(row) in error_request_ids
        )
    ]
    expected_request_ids = {
        request_id(row)
        for row in requests
        if canonical_artifact_id(args.kind, request_id(row)) in rejected_ids
    } | error_request_ids
    selected_request_ids = {request_id(row) for row in selected}
    if selected_request_ids != expected_request_ids:
        found = {canonical_artifact_id(args.kind, request_id(row)) for row in selected}
        missing_artifacts = sorted(rejected_ids - found)
        missing_requests = sorted(expected_request_ids - selected_request_ids)
        raise SystemExit(
            "cannot map retry inputs: "
            f"artifact_ids={missing_artifacts}, request_ids={missing_requests}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"kind": args.kind, "retry_requests": len(selected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
