#!/usr/bin/env python3
"""Run GRPO with trajectory credit and the 30/70 evidence curriculum."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from scripts import train_grpo as baseline

DEFAULT_CONFIG = baseline.DEFAULT_CONFIG


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=baseline.DEFAULT_MODEL)
    parser.add_argument("--train-data", type=Path, default=baseline.DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data", type=Path, default=baseline.DEFAULT_VAL_DATA)
    parser.add_argument("--difficulty-manifest", type=Path, required=True)
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--output", type=Path, default=Path("outputs/models/grpo-evidence-curriculum"))
    parser.add_argument("--logger", choices=("console", "swanlab"), default="console")
    parser.add_argument("--experiment-name", default="shopping-agent-grpo-evidence-curriculum")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("hydra_overrides", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def build_command(args):
    manifest = baseline._validated_path(args.difficulty_manifest, "difficulty manifest")
    command, environment = baseline.build_command(args)
    command.extend(
        [
            "shopping_curriculum_sampling.enable=true",
            "shopping_curriculum_sampling.switch_ratio=0.30",
            f"shopping_curriculum_sampling.manifest_path={manifest}",
            "shopping_trajectory_credit.enable=true",
        ]
    )
    environment["SHOPPING_DIFFICULTY_MANIFEST"] = str(manifest)
    return command, environment


def main():
    args = parse_args()
    command, environment = build_command(args)
    audit = {
        "command": command,
        "difficulty_manifest": environment["SHOPPING_DIFFICULTY_MANIFEST"],
        "output": environment["GRPO_OUTPUT_DIR"],
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    Path(environment["GRPO_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    preflight = [
        baseline.sys.executable,
        str(baseline.ROOT / "scripts/check_grpo_runtime.py"),
        *baseline._runtime_overrides(args),
    ]
    status = subprocess.call(preflight, cwd=baseline.ROOT, env=environment)
    if status:
        raise SystemExit(status)
    raise SystemExit(subprocess.call(command, cwd=baseline.ROOT, env=environment))


if __name__ == "__main__":
    main()
