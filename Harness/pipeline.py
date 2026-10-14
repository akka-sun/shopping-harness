"""Offline Harness artifact pipeline.

This module only reads JSONL and writes deterministic request/validation
artifacts.  It never calls a teacher model, starts ShopSimulator, or changes
model weights.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import json
import os
from pathlib import Path
import tempfile

from shopping_grpo.evaluation.trajectory import (
    NORMALIZED_TRAJECTORY_VERSION,
    normalize_trajectory,
)

from .contracts import HarnessValidationError, validate_memory, validate_skill
from .prompts import (
    MEMORY_PROMPT_VERSION,
    SKILL_PROMPT_VERSION,
    build_memory_messages,
    build_skill_messages,
)


class PipelineError(ValueError):
    """Raised when an offline Harness input is malformed."""


def read_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    rows = []
    try:
        stream = source.open(encoding="utf-8")
    except OSError as exc:
        raise PipelineError(f"cannot read {source}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineError(f"{source}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise PipelineError(f"{source}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def _write_jsonl(path: str | Path, rows: Iterable[Mapping], *, force: bool) -> None:
    output = Path(path)
    if output.exists() and not force:
        raise FileExistsError(f"output already exists: {output}; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            for row in rows:
                if not isinstance(row, Mapping):
                    raise PipelineError("output row must be an object")
                temporary.write(
                    json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str)
                    + "\n"
                )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
    except BaseException:
        temporary.close()
        temporary_path.unlink(missing_ok=True)
        raise


def _normalized(row: Mapping) -> dict:
    if row.get("schema_version") == NORMALIZED_TRAJECTORY_VERSION:
        return dict(row)
    try:
        return normalize_trajectory(row)
    except (TypeError, KeyError, ValueError) as exc:
        raise PipelineError(f"cannot normalize trajectory: {exc}") from exc


def build_memory_request_rows(trajectories: Sequence[Mapping]) -> list[dict]:
    """Build one model request row per normalized or raw trajectory."""
    rows = []
    for index, raw in enumerate(trajectories):
        normalized = _normalized(raw)
        trajectory_id = str(normalized.get("trajectory_id") or "")
        if not trajectory_id:
            raise PipelineError(f"trajectory at index {index} has no trajectory_id")
        rows.append(
            {
                "request_type": "memory",
                "prompt_version": MEMORY_PROMPT_VERSION,
                "trajectory_id": trajectory_id,
                "messages": build_memory_messages(normalized),
            }
        )
    return rows


def build_skill_request_rows(
    trajectories: Sequence[Mapping],
    clusters: Sequence[Mapping],
    memories: Sequence[Mapping] = (),
) -> list[dict]:
    """Build one multi-trajectory Skill request per explicit cluster."""
    normalized_by_id = {}
    for index, raw in enumerate(trajectories):
        normalized = _normalized(raw)
        trajectory_id = str(normalized.get("trajectory_id") or "")
        if not trajectory_id:
            raise PipelineError(f"trajectory at index {index} has no trajectory_id")
        if trajectory_id in normalized_by_id:
            raise PipelineError(f"duplicate trajectory_id: {trajectory_id}")
        normalized_by_id[trajectory_id] = normalized

    memory_by_id = {}
    for index, raw in enumerate(memories):
        try:
            artifact = validate_memory(raw)
        except HarnessValidationError as exc:
            raise PipelineError(f"invalid memory at index {index}: {exc}") from exc
        memory_by_id[artifact["memory_id"]] = artifact

    rows = []
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, Mapping):
            raise PipelineError(f"cluster at index {index} must be an object")
        cluster_id = str(cluster.get("cluster_id") or "")
        trajectory_ids = cluster.get("trajectory_ids")
        if not cluster_id or not isinstance(trajectory_ids, list):
            raise PipelineError(
                f"cluster at index {index} requires cluster_id and trajectory_ids"
            )
        selected_ids = [str(value) for value in trajectory_ids]
        if len(set(selected_ids)) < 2:
            raise PipelineError(f"cluster {cluster_id!r} requires two trajectories")
        missing = sorted(set(selected_ids) - set(normalized_by_id))
        if missing:
            raise PipelineError(f"cluster {cluster_id!r} references unknown trajectories: {missing}")
        memory_ids = cluster.get("memory_ids", [])
        if not isinstance(memory_ids, list):
            raise PipelineError(f"cluster {cluster_id!r}.memory_ids must be a list")
        missing_memories = sorted(set(str(value) for value in memory_ids) - set(memory_by_id))
        if missing_memories:
            raise PipelineError(
                f"cluster {cluster_id!r} references unknown memories: {missing_memories}"
            )
        selected_trajectories = [normalized_by_id[value] for value in selected_ids]
        selected_memories = [memory_by_id[str(value)] for value in memory_ids]
        evidence = cluster.get("evidence")
        expected_scope = str(evidence.get("phase") if isinstance(evidence, Mapping) else "")
        if expected_scope not in {"searching", "candidate_review", "verification", "purchase", "termination"}:
            raise PipelineError(
                f"cluster {cluster_id!r} requires evidence.phase as a supported scope"
            )
        rows.append(
            {
                "request_type": "skill",
                "prompt_version": SKILL_PROMPT_VERSION,
                "cluster_id": cluster_id,
                "trajectory_ids": selected_ids,
                "memory_ids": [str(value) for value in memory_ids],
                "expected_scope": expected_scope,
                "messages": build_skill_messages(
                    selected_trajectories,
                    selected_memories,
                    cluster_id=cluster_id,
                    expected_scope=expected_scope,
                ),
            }
        )
    return rows


def validate_artifact_rows(
    artifacts: Sequence[Mapping],
    *,
    kind: str,
    trajectory_events: Mapping[str, set[str]] | None = None,
    trajectory_ids: set[str] | None = None,
    memory_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Validate rows without dropping failures; return accepted and rejected."""
    if kind not in {"memory", "skill"}:
        raise PipelineError("kind must be memory or skill")
    accepted = []
    rejected = []
    for line_number, artifact in enumerate(artifacts, start=1):
        try:
            if kind == "memory":
                source_id = str(artifact.get("source_trajectory_id") or "")
                events = (
                    trajectory_events.get(source_id, set())
                    if trajectory_events is not None
                    else None
                )
                value = validate_memory(artifact, event_ids=events)
                if trajectory_ids is not None and source_id not in trajectory_ids:
                    raise HarnessValidationError("source_trajectory_id is unknown")
            else:
                value = validate_skill(artifact)
                if trajectory_ids is not None:
                    unknown = sorted(set(value["source_trajectory_ids"]) - trajectory_ids)
                    if unknown:
                        raise HarnessValidationError(
                            "source_trajectory_ids are unknown: " + ", ".join(unknown)
                        )
                if memory_ids is not None:
                    unknown = sorted(set(value["source_memory_ids"]) - memory_ids)
                    if unknown:
                        raise HarnessValidationError(
                            "source_memory_ids are unknown: " + ", ".join(unknown)
                        )
            accepted.append(value)
        except (HarnessValidationError, AttributeError, TypeError) as exc:
            rejected.append({"line": line_number, "error": str(exc), "artifact": dict(artifact)})
    return accepted, rejected


def _trajectory_indexes(rows: Sequence[Mapping]) -> tuple[dict[str, set[str]], set[str]]:
    events = {}
    ids = set()
    for raw in rows:
        normalized = _normalized(raw)
        trajectory_id = str(normalized.get("trajectory_id") or "")
        if not trajectory_id:
            continue
        ids.add(trajectory_id)
        events[trajectory_id] = {
            str(event.get("event_id"))
            for event in normalized.get("events", [])
            if isinstance(event, Mapping) and event.get("event_id")
        }
    return events, ids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m Harness",
        description="Offline teacher-request and artifact validation utilities",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("memory-requests", help="build single-trajectory Memory requests")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--force", action="store_true")
    command = commands.add_parser("skill-requests", help="build multi-trajectory Skill requests")
    command.add_argument("--trajectories", type=Path, required=True)
    command.add_argument("--clusters", type=Path, required=True)
    command.add_argument("--memories", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--force", action="store_true")
    command = commands.add_parser("validate", help="validate Memory or Skill JSONL artifacts")
    command.add_argument("--kind", choices=("memory", "skill"), required=True)
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--accepted", type=Path, required=True)
    command.add_argument("--rejected", type=Path)
    command.add_argument("--trajectories", type=Path)
    command.add_argument(
        "--memories",
        type=Path,
        help="validated Memory JSONL used to verify Skill source_memory_ids",
    )
    command.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "memory-requests":
        rows = build_memory_request_rows(read_jsonl(args.input))
        _write_jsonl(args.output, rows, force=args.force)
        print(json.dumps({"request_type": "memory", "count": len(rows)}, ensure_ascii=False))
        return
    if args.command == "skill-requests":
        rows = build_skill_request_rows(
            read_jsonl(args.trajectories),
            read_jsonl(args.clusters),
            read_jsonl(args.memories) if args.memories else (),
        )
        _write_jsonl(args.output, rows, force=args.force)
        print(json.dumps({"request_type": "skill", "count": len(rows)}, ensure_ascii=False))
        return
    trajectory_events = trajectory_ids = None
    if args.trajectories:
        trajectory_events, trajectory_ids = _trajectory_indexes(read_jsonl(args.trajectories))
    memory_ids = None
    if args.memories:
        memory_ids = set()
        for index, row in enumerate(read_jsonl(args.memories)):
            try:
                memory_ids.add(validate_memory(row)["memory_id"])
            except HarnessValidationError as exc:
                raise PipelineError(f"invalid source memory at index {index}: {exc}") from exc
    accepted, rejected = validate_artifact_rows(
        read_jsonl(args.input),
        kind=args.kind,
        trajectory_events=trajectory_events,
        trajectory_ids=trajectory_ids,
        memory_ids=memory_ids,
    )
    _write_jsonl(args.accepted, accepted, force=args.force)
    if args.rejected:
        _write_jsonl(args.rejected, rejected, force=args.force)
    print(
        json.dumps(
            {"kind": args.kind, "accepted": len(accepted), "rejected": len(rejected)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
