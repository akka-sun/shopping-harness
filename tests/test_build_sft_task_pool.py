import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_sft_task_pool.py"


def _write_jsonl(path, task_ids):
    path.write_text(
        "".join(json.dumps({"task_id": task_id}) + "\n" for task_id in task_ids),
        encoding="utf-8",
    )


class BuildSftTaskPoolTest(unittest.TestCase):
    def _fixture(self, root):
        products = []
        for task_id in range(8):
            products.append(
                {
                    "asin": f"asin-{task_id}",
                    "instructions": [
                        {
                            "instruction": f"buy product {task_id}",
                            "attributes": ["color"],
                        }
                    ],
                }
            )
        archive = root / "products.json.gz"
        with gzip.open(archive, "wt", encoding="utf-8") as handle:
            json.dump(products, handle)
        excluded = []
        for name, task_ids in (
            ("grpo.jsonl", [0]),
            ("test.jsonl", [1]),
            ("existing-sft.jsonl", [2]),
        ):
            path = root / name
            _write_jsonl(path, task_ids)
            excluded.append(path)
        return archive, excluded

    def _run(self, archive, excluded, output_dir, *, count=3, seed=17):
        command = [
            sys.executable,
            str(SCRIPT),
            "--products-archive",
            str(archive),
            "--output-dir",
            str(output_dir),
            "--count",
            str(count),
            "--seed",
            str(seed),
        ]
        for path in excluded:
            command.extend(["--exclude", str(path)])
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def test_cli_builds_deterministic_pool_without_excluded_tasks(self):
        """Catches omitted exclusions, unstable sampling, and wrong row counts."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, excluded = self._fixture(root)
            first = root / "first"
            second = root / "second"

            first_run = self._run(archive, excluded, first)
            second_run = self._run(archive, excluded, second)

            self.assertEqual(first_run.returncode, 0, first_run.stderr)
            self.assertEqual(second_run.returncode, 0, second_run.stderr)
            self.assertEqual(
                (first / "tasks.jsonl").read_bytes(),
                (second / "tasks.jsonl").read_bytes(),
            )
            rows = [
                json.loads(line)
                for line in (first / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            task_ids = [row["task_id"] for row in rows]
            self.assertEqual(len(task_ids), 3)
            self.assertEqual(len(set(task_ids)), 3)
            self.assertFalse({0, 1, 2}.intersection(task_ids))
            metadata = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["task_count"], 3)
            self.assertEqual(metadata["eligible_task_count"], 5)
            self.assertEqual(metadata["excluded_task_count"], 3)

    def test_cli_rejects_a_request_larger_than_the_eligible_pool(self):
        """Catches silent undersized datasets when exclusions leave too few tasks."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, excluded = self._fixture(root)

            result = self._run(archive, excluded, root / "output", count=6)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("only 5 eligible tasks", result.stderr)


if __name__ == "__main__":
    unittest.main()
