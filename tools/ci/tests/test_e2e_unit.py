"""E2E helper discovery has one owner and cannot hide missing Go results."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))
import run_e2e_unit  # noqa: E402
from ci_plan import make_plan  # noqa: E402
from classify_pr_changes import classify  # noqa: E402
from verification_catalog import full_cpu_ids  # noqa: E402

MODULE = run_e2e_unit.MODULE


class E2EUnitTests(unittest.TestCase):
    def test_source_and_executor_changes_select_units_without_extra_kind_profile(self):
        for path in (
            "e2e/testcases/cache_polarity_test.go",
            "e2e/pkg/framework/inventory.go",
            "e2e/profiles/ai-gateway/values.yaml",
            "e2e/profiles/dashboard/profile_test.go",
            "e2e/go.mod",
            "tools/make/e2e.mk",
            "tools/ci/run_e2e_unit.py",
            "tools/ci/workflow_evidence.py",
            ".github/workflows/test-tools.yml",
        ):
            with self.subTest(path=path):
                self.assertIn("e2e-unit", classify([path]).selected_jobs)
        self.assertNotIn(
            "e2e.envoy-ai-gateway",
            classify(["e2e/profiles/dashboard/profile_test.go"]).selected_jobs,
        )
        self.assertIn("e2e-unit", full_cpu_ids())
        record = make_plan([], source_sha="a" * 40, requested=("e2e-unit",))[
            "verifications"
        ][0]
        self.assertEqual(record["executor"], "tools")
        self.assertEqual(record["boundary"], ["unit"])
        self.assertEqual(record["images"], [])
        self.assertFalse(record["native"])

    def test_dynamic_packages_preserve_dedicated_owners(self):
        evidence, packages, calls = self._run()
        selected = [f"{MODULE}/profiles/new-profile", f"{MODULE}/testcases"]
        self.assertEqual(packages["selected"], selected)
        self.assertEqual(packages["dedicated_owners"], run_e2e_unit.DEDICATED_OWNERS)
        self.assertEqual(len(evidence["cases"]), len(selected))
        self.assertEqual(len(calls), 2)
        self.assertIn("-list", calls[0])
        self.assertIn("-count=1", calls[1])
        for call in calls:
            self.assertEqual(call[-len(selected) :], selected)

    def test_missing_skipped_and_failed_cases_cannot_succeed(self):
        for outcome in ("missing", "skip", "fail"):
            with self.subTest(outcome=outcome), self.assertRaises(ValueError):
                self._run(outcome=outcome)

    def test_discovery_or_execution_failure_propagates(self):
        for failed_phase in ("inventory", "tests"):
            with self.subTest(phase=failed_phase), self.assertRaises(
                subprocess.CalledProcessError
            ):
                self._run(failed_phase=failed_phase)

    def _run(self, outcome="pass", failed_phase=""):
        calls = []
        selected = [f"{MODULE}/profiles/new-profile", f"{MODULE}/testcases"]

        def process(command, *, stdout, **_kwargs):
            calls.append(command)
            phase = "inventory" if "-list" in command else "tests"
            for package in selected:
                if phase == "inventory":
                    event = {"Package": package, "Output": "TestContract\n"}
                elif outcome == "missing":
                    continue
                else:
                    event = {
                        "Package": package,
                        "Test": "TestContract",
                        "Action": outcome,
                    }
                stdout.write(json.dumps(event) + "\n")
            return subprocess.CompletedProcess(command, int(phase == failed_phase))

        discovered = "\n".join([*selected, *run_e2e_unit.DEDICATED_OWNERS])
        with tempfile.TemporaryDirectory() as directory, patch.object(
            run_e2e_unit.subprocess, "check_output", return_value=discovered
        ), patch.object(run_e2e_unit.subprocess, "run", side_effect=process):
            path = Path(directory)
            evidence = run_e2e_unit.run(path)
            return evidence, json.loads((path / "packages.json").read_text()), calls


if __name__ == "__main__":
    unittest.main()
