#!/usr/bin/env python3
"""Select deduplicated Harness artifacts using an evaluation comparison report.

Only artifacts with observed net benefit are activated.  This is deliberately
deterministic: no teacher or semantic model is called during selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm(value):
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def fingerprint(kind, row):
    fields = [row.get("scope"), row.get("trigger"), row.get("procedure"),
              row.get("lesson"), row.get("risk"), row.get("forbidden_shortcuts")]
    return hashlib.sha256(json.dumps([kind, *fields], ensure_ascii=False,
                                     sort_keys=True, default=str).encode()).hexdigest()


def select(artifacts, comparison, *, min_gains=1, max_regression_ratio=1.0):
    stats = {str(x["artifact_id"]): x for x in comparison.get("artifact_associations", [])}
    selected, rejected, seen = [], [], set()
    for row in artifacts:
        kind = "skill" if row.get("skill_id") else "memory"
        aid = str(row.get(f"{kind}_id") or "")
        item = stats.get(aid, {})
        gains = int(item.get("gain", 0))
        regressions = int(item.get("regression", 0))
        reason = None
        if gains < min_gains:
            reason = f"no_observed_gain:{gains}"
        elif regressions > gains * max_regression_ratio:
            reason = f"regression_ratio:{regressions}/{gains}"
        else:
            fp = fingerprint(kind, row)
            if fp in seen:
                reason = "duplicate_behavior"
            else:
                seen.add(fp)
        if reason:
            rejected.append({"artifact_id": aid, "reason": reason, "artifact": row})
        else:
            selected.append(row)
    return selected, rejected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=("memory", "skill"), required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--accepted", type=Path, required=True)
    p.add_argument("--rejected", type=Path, required=True)
    p.add_argument("--min-gains", type=int, default=1)
    p.add_argument("--max-regression-ratio", type=float, default=1.0)
    args = p.parse_args()
    accepted, rejected = select(_rows(args.artifacts), json.loads(args.comparison.read_text()),
                                min_gains=args.min_gains,
                                max_regression_ratio=args.max_regression_ratio)
    for path, rows in ((args.accepted, accepted), (args.rejected, rejected)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
