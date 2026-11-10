"""The CLI lane must include the development launcher's actual request result."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime_evidence
from ci_results import collection_errors

CASE = "tools/test/smoke/test-local-up-router.sh"


class LocalSmokeEvidenceTests(unittest.TestCase):
    def write_cli(self, directory):
        report = {
            "expected_cases": ["cli:lifecycle"],
            "cases": [{"id": "cli:lifecycle", "status": "passed"}],
        }
        (directory / "cli-test-report.json").write_text(json.dumps(report))

    def test_cli_requires_both_lifecycle_and_development_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write_cli(directory)
            (directory / "dev-smoke-report.json").write_text(
                json.dumps(
                    {
                        "expected_cases": [CASE],
                        "cases": [{"id": CASE, "status": "passed"}],
                    }
                )
            )
            evidence = runtime_evidence.local_evidence(directory, "cli")
            self.assertEqual(evidence["expected_cases"], ["cli:lifecycle", CASE])
            self.assertEqual(collection_errors(evidence, "test"), [])

    def test_cli_rejects_missing_failed_or_empty_development_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write_cli(directory)
            with self.assertRaises(FileNotFoundError):
                runtime_evidence.local_evidence(directory, "cli")
            for status in ("failed", "skipped", ""):
                with self.subTest(status=status):
                    (directory / "dev-smoke-report.json").write_text(
                        json.dumps(
                            {
                                "expected_cases": [CASE],
                                "cases": [{"id": CASE, "status": status}],
                            }
                        )
                    )
                    with self.assertRaisesRegex(ValueError, "did not complete"):
                        runtime_evidence.local_evidence(directory, "cli")
            (directory / "dev-smoke-report.json").write_text(
                json.dumps({"expected_cases": [], "cases": []})
            )
            with self.assertRaisesRegex(ValueError, "did not complete"):
                runtime_evidence.local_evidence(directory, "cli")

    def test_failed_launcher_replaces_stale_success_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "dev-smoke-report.json"
            report.write_text(
                json.dumps(
                    {
                        "expected_cases": [CASE],
                        "cases": [{"id": CASE, "status": "passed"}],
                    }
                )
            )
            result = subprocess.run(
                ["bash", str(runtime_evidence.ROOT / CASE)],
                cwd=runtime_evidence.ROOT,
                env={
                    **os.environ,
                    "TMPDIR": tmp,
                    "VLLM_SRUN_COMMAND": str(Path(tmp) / "missing-runtime"),
                    "LOCAL_DEV_SMOKE_REPORT_PATH": str(report),
                },
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not found", result.stdout)
            receipt = json.loads(report.read_text())
            self.assertEqual(receipt["cases"], [{"id": CASE, "status": "failed"}])
            self.write_cli(Path(tmp))
            with self.assertRaisesRegex(ValueError, "did not complete"):
                runtime_evidence.local_evidence(Path(tmp), "cli")


if __name__ == "__main__":
    unittest.main()
