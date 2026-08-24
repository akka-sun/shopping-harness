import argparse
from pathlib import Path

import yaml

from scripts.check_grpo_runtime import validate_evidence_curriculum
from scripts.train_grpo_evidence_curriculum import build_command


def test_baseline_config_keeps_new_features_off():
    config = yaml.safe_load(Path("configs/grpo.yaml").read_text())
    assert config["shopping_curriculum_sampling"]["enable"] is False
    assert config["shopping_trajectory_credit"]["enable"] is False


def test_curriculum_entry_enables_features_without_creating_output(tmp_path):
    model = tmp_path / "model"; model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"weights")
    train = tmp_path / "train.parquet"; train.write_bytes(b"train")
    val = tmp_path / "val.parquet"; val.write_bytes(b"val")
    manifest = tmp_path / "difficulty.jsonl"; manifest.write_text("{}\n")
    output = tmp_path / "output"
    args = argparse.Namespace(
        model=model, train_data=train, val_data=val, env_url="http://shop", output=output,
        logger="console", experiment_name="test", config=Path("configs/grpo.yaml"),
        dry_run=True, difficulty_manifest=manifest, hydra_overrides=[],
    )
    command, _ = build_command(args)
    assert "shopping_curriculum_sampling.enable=true" in command
    assert "shopping_curriculum_sampling.switch_ratio=0.30" in command
    assert f"shopping_curriculum_sampling.manifest_path={manifest.resolve()}" in command
    assert "shopping_trajectory_credit.enable=true" in command
    assert not output.exists()


def test_preflight_rejects_zero_uniform_probability(tmp_path):
    config = {
        "shopping_curriculum_sampling": {"enable": True, "switch_ratio": .3, "uniform_ratio": 0},
        "shopping_trajectory_credit": {"enable": False},
    }
    try:
        validate_evidence_curriculum(config, manifest_path=tmp_path / "missing")
    except SystemExit as exc:
        assert "uniform_ratio" in str(exc)
    else:
        raise AssertionError("zero uniform probability must fail")
