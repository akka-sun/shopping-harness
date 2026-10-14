#!/usr/bin/env python3
"""Apply an explicit semantic-review rejection ledger to validated artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("memory", "skill"), required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--rejection-ledger", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for path in (args.accepted, args.rejected):
        if path.exists() and not args.force:
            raise SystemExit(f"output already exists: {path}; pass --force to replace it")
    id_field = f"{args.kind}_id"
    artifacts = [artifact for path in args.input for artifact in _rows(path)]
    artifact_ids = [str(row.get(id_field) or "") for row in artifacts]
    if "" in artifact_ids:
        raise SystemExit(f"input contains an artifact without {id_field}")
    if len(artifact_ids) != len(set(artifact_ids)):
        raise SystemExit(f"inputs contain duplicate {id_field} values")
    ledger = _rows(args.rejection_ledger)
    reasons = {}
    for row in ledger:
        decision = row.get("decision")
        if decision == "accept":
            continue
        if decision not in (None, "reject"):
            raise SystemExit(f"unsupported semantic decision: {decision!r}")
        artifact_id = str(row.get(id_field) or "")
        reason = str(row.get("reason") or "")
        if not artifact_id or not reason:
            raise SystemExit(f"each rejection requires {id_field} and reason")
        if artifact_id in reasons:
            raise SystemExit(f"duplicate rejection ledger ID: {artifact_id}")
        reasons[artifact_id] = reason
    known = {str(row.get(id_field) or "") for row in artifacts}
    unknown = sorted(set(reasons) - known)
    if unknown:
        raise SystemExit(f"rejection ledger references unknown IDs: {unknown}")
    accepted = [row for row in artifacts if str(row.get(id_field)) not in reasons]
    rejected = [
        {"reason": reasons[str(row[id_field])], "artifact": row}
        for row in artifacts if str(row.get(id_field)) in reasons
    ]
    for path, rows in ((args.accepted, accepted), (args.rejected, rejected)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"kind": args.kind, "accepted": len(accepted),
                      "semantic_rejected": len(rejected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
