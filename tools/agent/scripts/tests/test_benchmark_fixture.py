"""The benchmark skill's synthetic source freezes a stable two-case dataset."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src/vllm-sr"))

from cli.main import main  # noqa: E402

SOURCE = ROOT / "tools/agent/skills/vllm-sr-benchmark/fixtures/two-case-mcq.jsonl"


class BenchmarkFixtureTests(unittest.TestCase):
    def test_local_preparation_freezes_two_synthetic_cases_without_calls(self):
        with TemporaryDirectory() as directory:
            args = [
                "benchmark",
                "--store",
                directory,
                "dataset",
                "prepare",
                "--local",
                "--benchmark",
                "mmlu-pro",
                "--profile",
                "smoke",
                "--source-path",
                str(SOURCE),
                "--revision",
                "synthetic-v1",
                "--limit",
                "2",
            ]
            result = CliRunner().invoke(main, args)
            self.assertEqual(result.exit_code, 0, result.output)
            manifest = json.loads(result.output)
            self.assertEqual(manifest["case_count"], 2)
            self.assertEqual(manifest["profile"], "smoke")
            self.assertEqual(manifest["split"], "dev")
            self.assertTrue(manifest["custom_subset"])
            self.assertEqual(
                manifest["sources"]["mmlu-pro"]["files"][0]["sha256"],
                hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            )
            cases = [
                json.loads(line)
                for line in Path(manifest["path"]).read_text().splitlines()
            ]
            self.assertEqual(len(cases), 2)
            self.assertEqual({case["benchmark"] for case in cases}, {"mmlu-pro"})
            self.assertEqual({case["answer"] for case in cases}, {"B", "C"})
