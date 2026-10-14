"""Call an OpenAI-compatible teacher and write validator-ready artifacts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


class TeacherAuthenticationError(RuntimeError):
    """The teacher gateway cannot accept further authenticated requests."""


def load_env() -> None:
    path = Path(".env")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()


def request_id(row: dict) -> str:
    return str(row.get("trajectory_id") or row.get("cluster_id") or "")


def normalize_artifact(content: str, kind: str) -> dict | None:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
    start = min((index for index in (content.find("{"), content.find("null")) if index >= 0),
                default=-1)
    if start < 0:
        raise ValueError("teacher response contains no JSON object or null")
    value, _ = json.JSONDecoder().raw_decode(content[start:])
    if isinstance(value, dict) and set(value) == {kind}:
        value = value[kind]
    if value is not None and not isinstance(value, dict):
        raise ValueError("teacher artifact must be an object or null")
    if value is not None:
        value.setdefault("schema_version", f"shopping-{kind}-v1")
        if kind == "skill":
            value.setdefault("status", "candidate")
    return value


def _check_request_provenance(row: dict, artifact: dict | None) -> None:
    """Reject otherwise-valid looking artifacts attributed to another request."""
    if artifact is None:
        return
    kind = str(row["request_type"])
    if kind == "memory":
        expected = request_id(row)
        actual = str(artifact.get("source_trajectory_id") or "")
        if actual != expected:
            raise ValueError(
                f"memory source_trajectory_id {actual!r} does not match request {expected!r}"
            )
        return
    allowed = {str(value) for value in row.get("trajectory_ids", [])}
    actual = artifact.get("source_trajectory_ids")
    if not isinstance(actual, list) or len(set(map(str, actual))) < 2:
        raise ValueError("skill must cite at least two distinct source trajectories")
    unknown = sorted(set(map(str, actual)) - allowed)
    if unknown:
        raise ValueError(f"skill cites trajectories outside its request: {unknown}")
    allowed_memories = {str(value) for value in row.get("memory_ids", [])}
    actual_memories = artifact.get("source_memory_ids", [])
    if not isinstance(actual_memories, list):
        raise ValueError("skill source_memory_ids must be a list")
    unknown_memories = sorted(set(map(str, actual_memories)) - allowed_memories)
    if unknown_memories:
        raise ValueError(
            f"skill cites memories outside its request: {unknown_memories}"
        )


def canonical_artifact_id(kind: str, request_identifier: str) -> str:
    """Return an opaque, stable ID independent of teacher naming choices."""
    digest = hashlib.sha256(request_identifier.encode("utf-8")).hexdigest()[:24]
    return f"{kind}-{digest}"


def call_teacher(row: dict, *, base: str, key: str, model: str, retries: int,
                 use_json_response_format: bool):
    kind = str(row["request_type"])
    body = {"model": model, "messages": row["messages"], "temperature": 0}
    # Some OpenAI-compatible gateways incorrectly echo the response_format
    # descriptor as assistant content.  Prompts already require JSON, so this is
    # opt-in instead of silently corrupting otherwise usable teacher output.
    if use_json_response_format:
        body["response_format"] = {"type": "json_object"}
    payload = json.dumps(body).encode()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(base + "/chat/completions", payload,
                {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as response:
                response_body = json.loads(response.read())
            content = response_body["choices"][0]["message"].get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("teacher response has empty content")
            artifact = normalize_artifact(content, kind)
            _check_request_provenance(row, artifact)
            if artifact is not None:
                artifact[f"{kind}_id"] = canonical_artifact_id(kind, request_id(row))
                if kind == "skill":
                    expected_scope = str(row.get("expected_scope") or "")
                    if not expected_scope:
                        raise ValueError("skill request has no expected_scope")
                    artifact["scope"] = expected_scope
            return request_id(row), artifact
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 402, 403}:
                raise TeacherAuthenticationError(
                    f"teacher gateway returned HTTP {exc.code}; check credentials or quota"
                ) from exc
            if attempt == retries:
                raise RuntimeError(f"{request_id(row)}: {exc}") from exc
            time.sleep(min(2 ** attempt, 20))
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError,
                IndexError, json.JSONDecodeError) as exc:
            if attempt == retries:
                raise RuntimeError(f"{request_id(row)}: {exc}") from exc
            time.sleep(min(2 ** attempt, 20))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--null-output", type=Path)
    parser.add_argument("--error-output", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--json-response-format",
        action="store_true",
        help="request provider-side JSON mode (off by default for gateway compatibility)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is required")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "deepseek-chat")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit > 0:
        rows = rows[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    null_path = args.null_output or args.output.with_name(args.output.stem + "_null.jsonl")
    error_path = args.error_output or args.output.with_name(args.output.stem + "_errors.jsonl")
    if args.force:
        args.output.unlink(missing_ok=True)
        null_path.unlink(missing_ok=True)
        error_path.unlink(missing_ok=True)
    completed = set()
    for path in (args.output, null_path):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    identity = str(item.get("_request_id") or item.get("source_trajectory_id") or "")
                    if not identity and item.get("skill_id"):
                        # Skill IDs are deterministically derived from cluster IDs,
                        # so resume never depends on teacher-selected identifiers.
                        identity = next(
                            (
                                request_id(row)
                                for row in rows
                                if canonical_artifact_id("skill", request_id(row))
                                == str(item["skill_id"])
                            ),
                            "",
                        )
                    if identity:
                        completed.add(identity)
    pending = [row for row in rows if request_id(row) not in completed]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(call_teacher, row, base=base, key=key, model=model,
                               retries=args.retries,
                               use_json_response_format=args.json_response_format)
                   for row in pending]
        for index, future in enumerate(as_completed(futures), 1):
            try:
                rid, artifact = future.result()
            except TeacherAuthenticationError as exc:
                for pending_future in futures:
                    pending_future.cancel()
                raise SystemExit(str(exc)) from exc
            except Exception as exc:
                row = pending[futures.index(future)]
                with error_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"_request_id": request_id(row), "error": str(exc)},
                                            ensure_ascii=False) + "\n")
                continue
            target = null_path if artifact is None else args.output
            record = ({"_request_id": rid, "_teacher_model": model, "artifact": None}
                      if artifact is None else artifact)
            with target.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            completed.add(rid)
            if index % 25 == 0 or index == len(pending):
                print(json.dumps({"completed": len(completed), "total": len(rows)}), flush=True)

    # Errors are an unresolved-work view, not an append-only experiment log.
    # Remove stale records whose request later succeeded during a resumed run.
    if error_path.exists():
        unresolved = {}
        for line in error_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            rid = str(item.get("_request_id") or "")
            if rid and rid not in completed:
                unresolved[rid] = item
        with error_path.open("w", encoding="utf-8") as stream:
            for item in unresolved.values():
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
