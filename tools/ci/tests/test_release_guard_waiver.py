"""Release Guard exception must preserve the original failed E2E report."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_guard_waiver import GUARD_WAIVER
from run_e2e_batch import profile_evidence


class ReleaseGuardWaiverTests(unittest.TestCase):
    def test_only_the_known_failure_qualifies(self):
        record = {
            "id": "e2e.production-stack",
            "profile": "production-stack",
            "runtime": "candle",
            "device": "cpu",
            "known_issue_waiver": GUARD_WAIVER,
        }
        report = {
            "profile": "production-stack",
            "cluster_name": "ci-production-stack-test",
            "expected_cases": [
                *(f"case-{index}" for index in range(9)),
                "jailbreak-detection",
            ],
            "test_results": [
                {"Name": f"case-{index}", "Passed": True} for index in range(9)
            ]
            + [
                {
                    "Name": "jailbreak-detection",
                    "Passed": False,
                    "Details": GUARD_WAIVER["details"],
                }
            ],
            "status": "FAILED",
            "exit_code": 1,
            "total_tests": 10,
            "passed_tests": 9,
            "failed_tests": 1,
        }
        message = (
            "- ❌ **jailbreak-detection** (10s) - Error: `"
            + GUARD_WAIVER["reason"]
            + "`\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp)

            def evidence(candidate: dict, markdown: str = message) -> dict:
                (raw / "test-report.json").write_text(json.dumps(candidate))
                (raw / "test-report.md").write_text(markdown)
                return profile_evidence(
                    record, report["cluster_name"], raw, passed=False
                )

            qualified = evidence(report)
            self.assertEqual(
                qualified["cases"][-1],
                {"id": "jailbreak-detection", "status": "failed"},
            )
            self.assertEqual(qualified["known_issue_waiver"], GUARD_WAIVER)
            with self.assertRaises(ValueError):
                evidence(report, "- ❌ **jailbreak-detection** - Error: `different`\n")
            for field, value in (
                ("failed_tests", 2),
                ("passed_tests", 8),
                ("cluster_name", "wrong"),
                ("test_results", [{"Name": "jailbreak-detection", "Passed": False}]),
            ):
                candidate = copy.deepcopy(report)
                candidate[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    evidence(candidate)
            candidate = copy.deepcopy(report)
            candidate["test_results"][0]["Passed"] = False
            with self.assertRaises(ValueError):
                evidence(candidate)
            candidate = copy.deepcopy(report)
            candidate["test_results"][-1]["Details"]["correct_tests"] = 9
            with self.assertRaises(ValueError):
                evidence(candidate)
            candidate = copy.deepcopy(report)
            candidate["test_results"][-1]["Details"]["failed_cases"][0][
                "description"
            ] = "new attack miss"
            with self.assertRaises(ValueError):
                evidence(candidate)
            candidate = copy.deepcopy(report)
            candidate["test_results"][-1]["Details"]["failed_cases"][0][
                "error"
            ] = "request error"
            with self.assertRaises(ValueError):
                evidence(candidate)


if __name__ == "__main__":
    unittest.main()
