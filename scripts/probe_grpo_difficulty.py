#!/usr/bin/env python3
"""Probe GRPO task difficulty once with a deterministic fresh SFT policy."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from urllib.request import urlopen
from pathlib import Path

from shopping_grpo.evaluation.rollout import OpenAIChatClient, collect_for_task
from shopping_grpo.training.grpo.difficulty.probe import (
    DETERMINISTIC_DECODE,
    ProbeOutcome,
    ProbeRequest,
    run_probe,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, default=ROOT / "data/grpo/train.jsonl")
    parser.add_argument("--val-data", type=Path, default=ROOT / "data/grpo/validation.jsonl")
    parser.add_argument("--evaluation-data", type=Path, default=ROOT / "data/evaluation/tasks.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-run-dir", type=Path)
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--max-environment-steps", type=int, default=35)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-launch-vllm", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser.parse_args(argv)


def _validate(args):
    for label, path in (
        ("model", args.model), ("train", args.train_data), ("validation", args.val_data),
        ("evaluation", args.evaluation_data),
    ):
        if not path.exists():
            raise SystemExit(f"{label} path does not exist: {path.resolve()}")
    if args.output.exists():
        raise SystemExit(f"output must not exist: {args.output.resolve()}")
    if args.max_environment_steps != 35:
        raise SystemExit("Probe v1 requires --max-environment-steps=35")


def build_audit(args):
    return {
        "model": str(args.model.resolve()),
        "train_data": str(args.train_data.resolve()),
        "validation_data": str(args.val_data.resolve()),
        "evaluation_data": str(args.evaluation_data.resolve()),
        "output": str(args.output.resolve()),
        "environment_url": args.env_url,
        "decode": dict(DETERMINISTIC_DECODE),
        "seed": args.seed,
        "max_environment_steps": args.max_environment_steps,
    }


def execute_probe(args):
    client = OpenAIChatClient(
        model=str(args.model.resolve()),
        base_url=args.llm_base_url,
        api_key=args.api_key,
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        context_window=24576,
        observation_token_budget=1536,
    )

    def rollout_one(task_id, decode, seed, max_steps):
        del decode, seed
        trajectory = collect_for_task(
            {"task_id": task_id}, client, base_url=args.env_url, max_steps=max_steps
        )
        terminal = trajectory.get("terminal_result") or {}
        detail = terminal.get("reward_detail") or {}
        valid = (
            trajectory.get("status") == "done"
            and terminal.get("done") is True
            and terminal.get("over") is True
            and detail.get("reward_version") == "shopsimulator-reward-v3"
            and detail.get("reward_valid") is True
        )
        initial = trajectory.get("initial_result") or {}
        instruction = str(initial.get("instruction") or initial.get("observation") or "")
        blocked = trajectory.get("blocked_tool_calls") or []
        return ProbeOutcome(
            task_id=task_id,
            instruction=instruction,
            valid=valid,
            invalid_reason=None if valid else str(trajectory.get("status") or "incomplete_rollout"),
            action_steps=len(trajectory.get("steps") or []),
            action_attempt_count=len(trajectory.get("steps") or []) + len(blocked),
            repeat_action_count=int(trajectory.get("repeat_action_count", 0)),
            guard_count=len(blocked),
            reward_type=detail.get("reward_type"),
        )

    request = ProbeRequest(
        args.model, args.train_data, args.val_data, args.evaluation_data, args.output,
        args.raw_run_dir, args.env_url, args.seed, args.max_environment_steps,
    )
    return run_probe(request, rollout_one)


def _endpoint_ready(base_url: str) -> bool:
    try:
        with urlopen(base_url.rstrip("/") + "/models", timeout=2):
            return True
    except Exception:
        return False


def _run_with_vllm(args):
    if args.no_launch_vllm or _endpoint_ready(args.llm_base_url):
        return execute_probe(args)
    if not 0.1 <= args.gpu_memory_utilization <= 0.99:
        raise SystemExit("--gpu-memory-utilization must be in [0.1, 0.99]")
    server_env = os.environ.copy()
    # FlashInfer's sampler build in the pinned vLLM stack is not compatible
    # with every GPU/kernel combination (including the observed Blackwell
    # setup); vLLM's native sampler is the safe fallback.
    server_env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    server = subprocess.Popen([
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(args.model.resolve()), "--host", "127.0.0.1",
        "--port", "8000", "--dtype", "bfloat16",
        "--max-model-len", "24576",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ], env=server_env)
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise SystemExit(f"vLLM exited before readiness (code {server.returncode})")
            if _endpoint_ready(args.llm_base_url):
                return execute_probe(args)
            time.sleep(2)
        raise SystemExit("timed out waiting for vLLM at " + args.llm_base_url)
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()


def main_for_test(argv=None):
    args = parse_args(argv)
    _validate(args)
    audit = build_audit(args)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.dry_run:
        return audit
    _run_with_vllm(args)
    return audit


def main():
    main_for_test()


if __name__ == "__main__":
    main()
