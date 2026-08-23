"""Public CPU and parameterized GRPO entry-point tests."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.train_grpo import build_command, main as grpo_main, parse_args
from shopping_grpo.cli import main as cli_main
from shopping_grpo.smoke import run_cpu_smoke


class PublicEntrypointTest(unittest.TestCase):
    def test_cpu_smoke_covers_public_contracts(self):
        result = run_cpu_smoke()

        self.assertEqual(
            result["checks"],
            [
                "action_schema",
                "trajectory_normalization",
                "reward_sample",
                "sft_label_mask",
                "dynamic_sampling_grouping",
            ],
        )

    def test_offline_example_cli_runs_without_models_or_environment(self):
        root = Path(__file__).resolve().parents[1]
        with patch.object(
            sys,
            "argv",
            [
                "shopping-grpo",
                "evaluate",
                str(root / "examples/trajectories.jsonl"),
            ],
        ), patch("builtins.print") as output:
            cli_main()

        summary = json.loads(output.call_args.args[0])
        self.assertEqual(summary["trajectory_count"], 3)
        self.assertEqual(summary["strict_gold_success_count"], 1)

    def test_public_grpo_launcher_accepts_sharded_weights_and_console(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors.index.json").write_text(
                "{}",
                encoding="utf-8",
            )
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            output = temporary / "output"
            with patch.object(
                sys,
                "argv",
                [
                    "train_grpo.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--output",
                    str(output),
                    "--config",
                    str(root / "configs/grpo.yaml"),
                    "--logger",
                    "console",
                    "--dry-run",
                ],
            ):
                args = parse_args()
            command, environment = build_command(args)

        self.assertIn("verl.trainer.main_ppo", command)
        self.assertEqual(environment["GRPO_MODEL_PATH"], str(model.resolve()))
        self.assertEqual(environment["GRPO_TRAIN_FILE"], str(train.resolve()))
        self.assertEqual(environment["GRPO_VAL_FILE"], str(validation.resolve()))
        self.assertEqual(
            environment["SHOPPING_GRPO_DIAGNOSTICS_PATH"],
            str(output.resolve() / "training_diagnostics.jsonl"),
        )
        self.assertIn("trainer.logger=[console]", command)

    def test_grpo_preflight_receives_normalized_hydra_overrides(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            args = argparse.Namespace(
                logger="console",
                config=root / "configs/grpo.yaml",
                dry_run=False,
                experiment_name="regression",
                hydra_overrides=["--", "trainer.total_epochs=2"],
            )
            command = [
                sys.executable,
                "-m",
                "verl.trainer.main_ppo",
                f"--config-path={root / 'configs'}",
                "--config-name=grpo",
                "trainer.logger=[console]",
                "trainer.experiment_name=regression",
                "trainer.total_epochs=2",
            ]
            environment = {
                "GRPO_MODEL_PATH": "/model",
                "GRPO_TRAIN_FILE": "/train.parquet",
                "GRPO_VAL_FILE": "/validation.parquet",
                "SHOPSIM_BASE_URL": "http://127.0.0.1:5700",
                "GRPO_OUTPUT_DIR": str(output),
            }
            with patch("scripts.train_grpo.parse_args", return_value=args), patch(
                "scripts.train_grpo.build_command",
                return_value=(command, environment),
            ), patch("scripts.train_grpo.subprocess.call", side_effect=[0, 0]) as call:
                with self.assertRaisesRegex(SystemExit, "0"):
                    grpo_main()

        preflight = call.call_args_list[0].args[0]
        self.assertEqual(
            preflight[2:],
            [
                "trainer.logger=[console]",
                "trainer.experiment_name=regression",
                "trainer.total_epochs=2",
            ],
        )


if __name__ == "__main__":
    unittest.main()
