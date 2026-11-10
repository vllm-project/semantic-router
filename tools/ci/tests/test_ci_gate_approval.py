"""Approval waiting is distinct from failed or completed qualification."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))

from check_ci_gate import evaluate_gate, gate_report, render_summary  # noqa: E402
from ci_plan import component_batches, digest  # noqa: E402
from execution_batches import (  # noqa: E402
    e2e_batches,
    expected_dispatch_jobs,
    platform_batches,
)
from test_ci_artifact_selection import write_builds, write_receipts  # noqa: E402
from test_ci_gate import SHA, completed  # noqa: E402

BASE = "b" * 40
PATHS = ["src/model-runtime/Dockerfile"]


def reseal(plan):
    plan["plan_sha256"] = digest(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )


class GateApprovalTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.receipts, self.builds = completed(
            PATHS, require_approval=True, base_sha=BASE
        )

    def verdict(self, plan=None, receipts=None, builds=None, **options):
        return evaluate_gate(
            self.plan if plan is None else plan,
            self.receipts if receipts is None else receipts,
            builds=self.builds if builds is None else builds,
            **options,
        )

    def test_waiting_preserves_complete_inventory_without_validation_errors(self):
        verdict = self.verdict()
        self.assertEqual(verdict.status, "awaiting-approval")
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.errors, ())
        self.assertEqual(
            verdict.required, tuple(self.plan["required_verification_ids"])
        )
        self.assertEqual(
            verdict.deferred_verifications, tuple(self.plan["deferred_verifications"])
        )
        self.assertEqual(verdict.deferred_images, tuple(self.plan["deferred_images"]))
        self.assertTrue(verdict.deferred_verifications)
        self.assertTrue(verdict.deferred_images)
        summary = render_summary(verdict)
        self.assertIn("## Waiting for committer approval", summary)
        self.assertIn(f"{len(verdict.required)} required verifications", summary)
        self.assertIn("write, maintain, or admin", summary)
        self.assertIn("`/ci run`", summary)
        self.assertNotIn("CI gate failed", summary)
        for name in (*verdict.deferred_verifications, *verdict.deferred_images):
            self.assertIn(f"`{name}`", summary)

    def test_affected_authorization_completes_the_same_required_inventory(self):
        plan, receipts, builds = completed(
            PATHS,
            require_approval=True,
            authorization={"authorized": True, "scopes": ["affected"]},
        )
        verdict = self.verdict(plan, receipts, builds)
        self.assertEqual(verdict.status, "success")
        self.assertTrue(verdict.passed)
        self.assertEqual(
            verdict.required, tuple(self.plan["required_verification_ids"])
        )
        self.assertEqual(verdict.deferred_verifications, ())
        self.assertEqual(verdict.deferred_images, ())

    def test_partial_authorization_never_qualifies_the_remaining_contracts(self):
        automatic, _, _ = completed(PATHS, full=True, require_approval=True)
        plan, receipts, builds = completed(
            PATHS,
            full=True,
            require_approval=True,
            authorization={"authorized": True, "scopes": ["operator"]},
        )
        self.assertIn("operator", automatic["deferred_verifications"])
        self.assertNotIn("operator", plan["deferred_verifications"])
        verdict = self.verdict(plan, receipts, builds)
        self.assertEqual(verdict.status, "awaiting-approval")
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.errors, ())
        self.assertEqual(
            verdict.required, tuple(automatic["required_verification_ids"])
        )

    def test_real_failures_take_priority_over_waiting(self):
        bad_evidence = copy.deepcopy(self.receipts)
        bad_evidence[0]["evidence_sha256"] = "tampered"
        jobs = {
            name: {"result": "success"} for name in self.plan["expected_dispatch_jobs"]
        }
        jobs["plan"]["result"] = "failure"
        for receipts, options, expected in (
            (self.receipts[1:], {}, "missing"),
            (bad_evidence, {}, "evidence digest differs"),
            (self.receipts, {"jobs": jobs}, "prerequisite plan: failure"),
        ):
            with self.subTest(expected=expected):
                verdict = self.verdict(receipts=receipts, **options)
                self.assertEqual(verdict.status, "failure")
                self.assertFalse(verdict.passed)
                self.assertTrue(any(expected in error for error in verdict.errors))
                self.assertEqual(
                    verdict.deferred_verifications,
                    tuple(self.plan["deferred_verifications"]),
                )
                self.assertIn("## CI gate failed", render_summary(verdict))

    def test_malformed_partitions_are_failures_not_waiting(self):
        for field, value in (
            ("expected_verification_ids", [{}]),
            ("deferred_verifications", "not-a-list"),
            ("deferred_verifications", [None]),
            ("deferred_verifications", self.plan["deferred_verifications"] * 2),
            ("deferred_verifications", self.plan["expected_verification_ids"]),
            ("required_verification_ids", self.plan["expected_verification_ids"]),
            ("deferred_images", "not-a-list"),
            ("deferred_images", [None]),
            ("deferred_images", self.plan["deferred_images"] * 2),
            ("deferred_images", []),
            ("deferred_images", ["unselected-image"]),
            ("approval_required", False),
        ):
            with self.subTest(field=field, value=value):
                plan = copy.deepcopy(self.plan)
                plan[field] = value
                reseal(plan)
                verdict = self.verdict(plan)
                self.assertEqual(verdict.status, "failure")
                self.assertTrue(verdict.errors)

    def test_overlapping_image_partitions_fail(self):
        plan = copy.deepcopy(self.plan)
        plan["images"] = plan["deferred_images"]
        reseal(plan)
        self.assertEqual(self.verdict(plan).status, "failure")

    def test_automatic_contract_cannot_be_repartitioned_as_awaiting_approval(self):
        plan = copy.deepcopy(self.plan)
        automatic = next(
            record
            for record in plan["verifications"]
            if record["execution"] == "automatic"
        )["id"]
        plan["expected_verification_ids"].remove(automatic)
        plan["deferred_verifications"].append(automatic)
        plan["verifications"] = [
            record for record in plan["verifications"] if record["id"] != automatic
        ]
        for field, batch in (
            ("component_batches", component_batches),
            ("platform_batches", platform_batches),
            ("e2e_batches", e2e_batches),
        ):
            plan[field] = batch(plan["verifications"])
        plan["expected_dispatch_jobs"] = expected_dispatch_jobs(plan)
        reseal(plan)
        receipts = [receipt for receipt in self.receipts if receipt["id"] != automatic]
        verdict = self.verdict(plan, receipts)
        self.assertEqual(verdict.status, "failure")
        self.assertEqual(
            verdict.errors,
            (f"only approval-gated verifications may be deferred: ['{automatic}']",),
        )

    def test_report_omits_large_plan_payload_and_binds_the_complete_run(self):
        verdict = self.verdict()
        plan = copy.deepcopy(self.plan)
        plan["paths"] = [f"large-change/{index}.md" for index in range(5000)]
        reseal(plan)
        with patch.dict(
            os.environ, {"GITHUB_RUN_ID": "12345", "GITHUB_RUN_ATTEMPT": "2"}
        ):
            report = gate_report(plan, verdict)
        self.assertEqual(
            set(report),
            {
                "schema_version",
                "source_sha",
                "base_sha",
                "plan_sha256",
                "run_id",
                "run_attempt",
                "status",
                "required_verification_ids",
                "deferred_verifications",
                "deferred_images",
                "errors",
            },
        )
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["source_sha"], SHA)
        self.assertEqual(report["base_sha"], BASE)
        self.assertEqual(report["plan_sha256"], plan["plan_sha256"])
        self.assertEqual(report["run_id"], 12345)
        self.assertEqual(report["run_attempt"], 2)
        self.assertEqual(report["status"], "awaiting-approval")
        self.assertEqual(report["required_verification_ids"], list(verdict.required))
        self.assertLess(len(json.dumps(report).encode()), 4096)


class GateApprovalCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.plan, self.receipts, builds = completed(
            PATHS, require_approval=True, base_sha=BASE
        )
        self.plan_path = self.directory / "plan.json"
        self.plan_path.write_text(json.dumps(self.plan))
        self.results = self.directory / "results"
        write_receipts(self.results, self.plan, self.receipts)
        write_builds(self.results, builds)
        self.summary = self.directory / "summary.md"
        self.outputs = self.directory / "outputs.txt"
        self.report = self.directory / "report" / "ci-gate.json"

    def run_gate(self, *arguments, **environment):
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GITHUB_") and key != "EXECUTOR_RESULTS"
        }
        env.update(
            GITHUB_SHA=SHA,
            GITHUB_RUN_ID="12345",
            GITHUB_RUN_ATTEMPT="2",
            GITHUB_STEP_SUMMARY=str(self.summary),
            GITHUB_OUTPUT=str(self.outputs),
        )
        env.update(environment)
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/ci/check_ci_gate.py"),
                "--plan",
                str(self.plan_path),
                "--results",
                str(self.results),
                "--output",
                str(self.report),
                *arguments,
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_default_cli_remains_strict_while_reporting_waiting(self):
        result = self.run_gate()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Waiting for committer approval", self.summary.read_text())
        self.assertIn("status=awaiting-approval\n", self.outputs.read_text())
        self.assertIn("passed=false\n", self.outputs.read_text())
        report = json.loads(self.report.read_text())
        self.assertEqual(report["status"], "awaiting-approval")
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["plan_sha256"], self.plan["plan_sha256"])
        self.assertEqual(report["run_id"], 12345)
        self.assertEqual(report["run_attempt"], 2)

    def test_explicit_allow_deferred_exits_zero_without_reporting_passed(self):
        result = self.run_gate("--allow-deferred")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            json.loads(self.report.read_text())["status"], "awaiting-approval"
        )
        self.assertIn("passed=false\n", self.outputs.read_text())

    def test_complete_cli_reports_success_without_allow_deferred(self):
        self.plan, receipts, builds = completed(
            PATHS,
            require_approval=True,
            base_sha=BASE,
            authorization={"authorized": True, "scopes": ["affected"]},
        )
        self.plan_path.write_text(json.dumps(self.plan))
        write_receipts(self.results, self.plan, receipts)
        write_builds(self.results, builds)
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(self.report.read_text())
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["deferred_verifications"], [])
        self.assertEqual(report["deferred_images"], [])
        self.assertEqual(report["errors"], [])
        self.assertIn("status=success\n", self.outputs.read_text())
        self.assertIn("passed=true\n", self.outputs.read_text())

    def test_allow_deferred_does_not_hide_a_missing_receipt(self):
        removed = next(self.results.rglob(f"{self.receipts[0]['id']}.json"))
        removed.unlink()
        result = self.run_gate("--allow-deferred")
        self.assertEqual(result.returncode, 1)
        report = json.loads(self.report.read_text())
        self.assertEqual(report["status"], "failure")
        self.assertIn(
            f"required verification {self.receipts[0]['id']}: missing", report["errors"]
        )
        self.assertIn("status=failure\n", self.outputs.read_text())

    def test_wrong_workflow_source_remains_failure_with_a_bound_report(self):
        result = self.run_gate("--allow-deferred", GITHUB_SHA="f" * 40)
        self.assertEqual(result.returncode, 1)
        report = json.loads(self.report.read_text())
        self.assertEqual(report["status"], "failure")
        self.assertEqual(report["source_sha"], SHA)
        self.assertEqual(report["base_sha"], BASE)
        self.assertEqual(
            report["required_verification_ids"], self.plan["required_verification_ids"]
        )
        self.assertEqual(
            report["deferred_verifications"], self.plan["deferred_verifications"]
        )
        self.assertEqual(report["deferred_images"], self.plan["deferred_images"])
        self.assertIn(
            "plan source SHA differs from this workflow run", report["errors"]
        )


if __name__ == "__main__":
    unittest.main()
