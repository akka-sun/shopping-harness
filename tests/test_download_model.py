import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/download_model.sh"


class DownloadModelScriptTest(unittest.TestCase):
    def _fake_hf(self, root: Path, *, complete: bool) -> tuple[Path, Path]:
        executable = root / "fake-hf"
        arguments = root / "arguments.txt"
        body = """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "$FAKE_HF_ARGUMENTS"
output=''
previous=''
for argument in "$@"; do
  if [[ "$previous" == '--local-dir' ]]; then output="$argument"; fi
  previous="$argument"
done
mkdir -p "$output"
printf '{}' > "$output/config.json"
"""
        if complete:
            body += """printf '{"weight_map":{"layer":"model-00001-of-00001.safetensors"}}' > "$output/model.safetensors.index.json"
printf 'weights' > "$output/model-00001-of-00001.safetensors"
"""
        executable.write_text(body, encoding="utf-8")
        executable.chmod(0o755)
        return executable, arguments

    def _run(self, fake_hf: Path, arguments: Path, output: Path):
        environment = {
            **os.environ,
            "HF_CLI": str(fake_hf),
            "FAKE_HF_ARGUMENTS": str(arguments),
            "NETWORK_TURBO_FILE": str(output.parent / "missing-network-turbo"),
        }
        return subprocess.run(
            ["bash", str(SCRIPT), "Example/Model", str(output)],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_downloads_to_requested_directory_and_verifies_weight_shards(self):
        """Catches wrong model/output forwarding and missing artifact verification."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_hf, arguments = self._fake_hf(root, complete=True)
            output = root / "model"

            result = self._run(fake_hf, arguments, output)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                arguments.read_text(encoding="utf-8").splitlines(),
                [
                    "download",
                    "Example/Model",
                    "--local-dir",
                    str(output),
                    "--max-workers",
                    "4",
                ],
            )
            self.assertIn("model download verified", result.stdout)

    def test_rejects_download_without_model_weights(self):
        """Catches false success when the Hub command returns without weights."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_hf, arguments = self._fake_hf(root, complete=False)

            result = self._run(fake_hf, arguments, root / "model")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no model weight files", result.stderr)


if __name__ == "__main__":
    unittest.main()
