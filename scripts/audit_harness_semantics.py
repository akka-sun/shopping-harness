#!/usr/bin/env python3
"""Semantically review validated Harness artifacts with a teacher model."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


class TeacherAuthenticationError(RuntimeError):
    """The configured teacher credentials cannot authenticate requests."""


POLICY = """You are the final safety reviewer for runtime guidance used by a shopping agent.
Review exactly one already schema-validated Memory or Skill. Return JSON only:
{"decision":"accept"|"reject","reason":"concise concrete reason","categories":["..."]}.

Accept only guidance that is reusable, actor-visible, executable, causally supported by its
stated source, and consistent with all rules below. Reject on any violation:
1. Search queries should be concise: category plus a few discriminating terms. Never demand
   copying the full request or all hard constraints into one query.
2. Reviews can support subjective use experience only. They cannot establish brand, model,
   official features, technical specifications, composition, design, compatibility, quantity,
   authenticity, category, variant identity, or price. Listing Reviews as an alternative route
   to prove such a field is a violation, even when other evidence sources are also listed.
3. Never permit buying an approximate, substitute, partial-match, closest, or otherwise
   hard-requirement-violating item. If exact requirements cannot be verified, continue useful
   exploration or safely finish without purchase when eligible.
4. Do not require arbitrary exploration quotas such as always opening a second candidate when
   the current candidate is fully evidenced. Comparison is appropriate only while a material
   requirement or best-choice question remains unresolved.
5. Every proposed action must be possible from the latest visible observation. Do not use stale
   product IDs or unavailable controls, invent tools, override the Action Guard/Reward, or treat
   missing evidence as false evidence.
6. Reject task-specific values, answer leakage, contradictory branches, vague slogans, and rules
   whose trigger is materially broader than their advice. A Memory must be one local correction;
   a Skill must be one short phase-specific procedure with observable success and fallback.

Do not reject merely because the text says not to buy a substitute, or says Reviews are
insufficient. Judge the semantic direction. Categories should come from: search_overstuffing,
reviews_as_official_evidence, substitute_purchase, arbitrary_comparison_quota, stale_or_illegal_action,
missing_evidence_assumed, trigger_mismatch, task_specific, vague_or_noncausal, contradictory,
scope_or_completion, other. Use [] when accepting."""


def load_env() -> None:
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def call(item: dict, *, base: str, key: str, model: str, retries: int) -> dict:
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": POLICY},
            {"role": "user", "content": json.dumps(item, ensure_ascii=False, sort_keys=True)},
        ],
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                base + "/chat/completions", payload,
                {"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                content = json.loads(response.read())["choices"][0]["message"]["content"]
            start = content.find("{")
            decision = json.JSONDecoder().raw_decode(content[start:])[0]
            if decision.get("decision") not in {"accept", "reject"}:
                raise ValueError("invalid decision")
            if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
                raise ValueError("missing reason")
            if not isinstance(decision.get("categories"), list):
                raise ValueError("categories must be a list")
            return decision
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 402, 403}:
                raise TeacherAuthenticationError(
                    f"teacher gateway returned HTTP {exc.code}; check credentials or quota"
                ) from exc
            if attempt == retries:
                raise RuntimeError(str(exc)) from exc
            time.sleep(min(2 ** attempt, 20))
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError,
                json.JSONDecodeError) as exc:
            if attempt == retries:
                raise RuntimeError(str(exc)) from exc
            time.sleep(min(2 ** attempt, 20))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("memory", "skill"), required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is required")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "deepseek-chat")
    id_field = args.kind + "_id"
    items = [item for path in args.input for item in rows(path)]
    ids = [str(item.get(id_field) or "") for item in items]
    if "" in ids or len(ids) != len(set(ids)):
        raise SystemExit(f"inputs contain missing or duplicate {id_field}")
    if args.limit > 0:
        items = items[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {str(row.get(id_field)) for row in rows(args.output)} if args.output.exists() else set()
    pending = [item for item in items if str(item.get(id_field)) not in completed]
    # Errors are a current unresolved-work view.  A resumed run retries every
    # item not present in the decision file instead of preserving stale errors.
    args.errors.unlink(missing_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(call, item, base=base, key=key, model=model,
                               retries=args.retries): item for item in pending}
        for index, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            artifact_id = str(item[id_field])
            try:
                decision = future.result()
                record = {id_field: artifact_id, "decision": decision["decision"],
                          "reason": decision["reason"].strip(),
                          "categories": decision["categories"], "teacher_model": model}
                target = args.output
            except TeacherAuthenticationError:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            except Exception as exc:
                record = {id_field: artifact_id, "error": str(exc)}
                target = args.errors
            with target.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if index % 25 == 0 or index == len(pending):
                print(json.dumps({"completed": len(completed) + index,
                                  "total": len(items)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
