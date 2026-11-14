"""Suite/rollup evidence must qualify a green publisher diagnostic."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test_pr_ci_status as fixtures
from manual_ci import ManualCIError


class SuiteDiagnosticTests(unittest.TestCase):
    def client(self):
        client = fixtures.DiagnosticClient()
        client.checks = client.checks[:1]
        client.checks[0]["check_suite"]["id"] = 20
        client.required = {80: True}
        client.visible = {80}
        return client

    def diagnose(self, client):
        result = fixtures.DiagnosticTests().diagnose(client)
        self.assertFalse(result["actionable"])
        self.assertEqual(client.writes, [])
        return result

    def test_green_required_gate_in_unsuccessful_suite_is_not_healthy(self):
        for conclusion in ("cancelled", "failure", "skipped", "neutral", "timed_out"):
            with self.subTest(conclusion=conclusion):
                client = self.client()
                client.checks[0]["check_suite"]["id"] = 19
                client.suites[19]["conclusion"] = conclusion
                client.rollup["state"] = "FAILURE"
                result = self.diagnose(client)
                self.assertEqual(result["state"], "suite_mismatch")
                self.assertIn(
                    "required_gate_suite_not_successful", result["reason_codes"]
                )
                self.assertTrue(result["evidence"]["full_qualification"])

    def test_successful_sibling_suite_is_valid_without_matching_run_suite(self):
        client = self.client()
        client.checks[0]["check_suite"]["id"] = 19
        client.suites[19]["conclusion"] = "success"
        self.assertEqual(self.diagnose(client)["state"], "healthy")

    def test_non_successful_rollup_prevents_a_healthy_diagnostic(self):
        for state in ("FAILURE", "ERROR", "PENDING", "EXPECTED"):
            with self.subTest(state=state):
                client = self.client()
                client.rollup["state"] = state
                result = self.diagnose(client)
                self.assertEqual(result["state"], "rollup_blocked")
                self.assertEqual(result["rollup_state"], state)

    def test_current_qualification_suite_is_checked_independently(self):
        client = self.client()
        client.checks[0]["check_suite"]["id"] = 19
        client.suites[19]["conclusion"] = "success"
        client.suites[20]["conclusion"] = "cancelled"
        result = self.diagnose(client)
        self.assertEqual(result["state"], "uncertain")
        self.assertIn("qualification_suite_not_successful", result["reason_codes"])

    def test_hidden_or_non_required_green_gate_is_not_healthy(self):
        for hidden in (False, True):
            with self.subTest(hidden=hidden):
                client = self.client()
                if hidden:
                    client.visible.clear()
                else:
                    client.required[80] = False
                result = self.diagnose(client)
                self.assertEqual(result["state"], "uncertain")
                self.assertIn(
                    "current_required_gate_not_visible", result["reason_codes"]
                )

    def test_missing_requiredness_cannot_establish_health(self):
        client = self.client()
        client.required_complete = False
        self.assertEqual(self.diagnose(client)["state"], "evidence_unavailable")

    def test_invalid_suite_or_rollup_evidence_is_unknown(self):
        mutations = (
            lambda c: c.suites[20].update(id=99),
            lambda c: c.suites[20].update(head_sha="f" * 40),
            lambda c: c.suites[20]["app"].update(id=1),
            lambda c: c.suites[20].update(status="private raw value"),
            lambda c: c.suites[20].update(conclusion=None),
            lambda c: c.suites[20].update(conclusion="private raw value"),
            lambda c: c.run.update(check_suite_id=True),
            lambda c: c.rollup.update(oid="f" * 40),
            lambda c: c.rollup.update(state=None),
            lambda c: c.rollup.update(state="private raw value"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                client = self.client()
                mutate(client)
                result = self.diagnose(client)
                self.assertEqual(result["state"], "evidence_unavailable")
                self.assertNotIn("private raw value", json.dumps(result))

    def test_read_failure_stays_unknown_and_does_not_echo_api_errors(self):
        client = self.client()
        client.gh = lambda *a, **k: (_ for _ in ()).throw(
            ManualCIError("private API response")
        )
        result = self.diagnose(client)
        self.assertEqual(result["state"], "evidence_unavailable")
        self.assertNotIn("private API response", json.dumps(result))

    def fail_health_read(self, client, evidence, error, *, final=False):
        original = client.request if evidence == "suite" else client.gh
        reads = {}

        def read(*args, **kwargs):
            if evidence == "rollup" or "/check-suites/" in args[0]:
                key = args[0] if evidence == "suite" else "rollup"
                reads[key] = reads.get(key, 0) + 1
                if not final or reads[key] > 1:
                    if error is subprocess.TimeoutExpired:
                        raise error(["private API response"], 60)
                    raise error("private API response")
            return original(*args, **kwargs)

        if evidence == "suite":
            client.request = read
        else:
            client.gh = read

    def test_initial_transport_failure_preserves_verified_competition(self):
        for evidence in ("suite", "rollup"):
            for error in (OSError, subprocess.TimeoutExpired):
                with self.subTest(evidence=evidence, error=error):
                    client = fixtures.DiagnosticClient()
                    self.fail_health_read(client, evidence, error)
                    result = self.diagnose(client)
                    self.assertEqual(result["state"], "competing_required")
                    self.assertIn(
                        "check_state_evidence_unavailable", result["reason_codes"]
                    )
                    self.assertTrue(result["evidence"]["current_source"])
                    self.assertTrue(result["evidence"]["snapshot_current"])
                    self.assertNotIn("private API response", json.dumps(result))

    def test_initial_transport_failure_cannot_establish_health(self):
        for evidence in ("suite", "rollup"):
            for error in (OSError, subprocess.TimeoutExpired):
                with self.subTest(evidence=evidence, error=error):
                    client = self.client()
                    self.fail_health_read(client, evidence, error)
                    result = self.diagnose(client)
                    self.assertEqual(result["state"], "evidence_unavailable")
                    self.assertIn(
                        "check_state_evidence_unavailable", result["reason_codes"]
                    )
                    self.assertTrue(result["evidence"]["current_source"])
                    self.assertTrue(result["evidence"]["snapshot_current"])
                    self.assertNotIn("private API response", json.dumps(result))

    def test_final_transport_failure_invalidates_healthy_and_competing_snapshots(self):
        for evidence in ("suite", "rollup"):
            for error in (OSError, subprocess.TimeoutExpired):
                for competing in (False, True):
                    with self.subTest(
                        evidence=evidence, error=error, competing=competing
                    ):
                        client = (
                            fixtures.DiagnosticClient() if competing else self.client()
                        )
                        self.fail_health_read(client, evidence, error, final=True)
                        result = self.diagnose(client)
                        self.assertEqual(result["state"], "evidence_unavailable")
                        self.assertIn(
                            "read_evidence_unavailable", result["reason_codes"]
                        )
                        self.assertFalse(result["evidence"]["snapshot_current"])
                        self.assertNotIn("private API response", json.dumps(result))

    def test_initial_transport_failure_does_not_skip_source_revalidation(self):
        for evidence in ("suite", "rollup"):
            for error in (OSError, subprocess.TimeoutExpired):
                with self.subTest(evidence=evidence, error=error):
                    client = fixtures.DiagnosticClient()
                    self.fail_health_read(client, evidence, error)
                    client.mutate_on_final = lambda c: c.pull["head"].update(
                        sha="f" * 40
                    )
                    result = self.diagnose(client)
                    self.assertEqual(result["state"], "stale")
                    self.assertFalse(result["evidence"]["snapshot_current"])

    def test_suite_or_rollup_change_invalidates_the_collected_snapshot(self):
        for mutate in (
            lambda c: c.suites[20].update(conclusion="cancelled"),
            lambda c: c.rollup.update(state="FAILURE"),
        ):
            with self.subTest(mutate=mutate):
                client = self.client()
                client.mutate_on_final = mutate
                result = self.diagnose(client)
                self.assertEqual(result["state"], "stale")
                self.assertFalse(result["evidence"]["snapshot_current"])

    def test_current_failure_and_deferred_qualification_keep_their_verdicts(self):
        for current_failure in (False, True):
            with self.subTest(current_failure=current_failure):
                client = self.client()
                client.suites[20]["conclusion"] = "cancelled"
                if current_failure:
                    client.run["conclusion"] = "failure"
                else:
                    client.plan.update(
                        expected_verification_ids=["quality"],
                        deferred_verifications=["e2e.dashboard"],
                    )
                    client.refresh_report()
                    client.checks[0].update(status="in_progress", conclusion=None)
                result = self.diagnose(client)
                self.assertEqual(
                    result["state"], "failed" if current_failure else "deferred"
                )

    def test_recorded_suite_topology_with_synthetic_qualification_receipts(self):
        path = Path(__file__).parent / "testdata/pr_gate_suite_snapshots.json"
        for case in json.loads(path.read_text(encoding="utf-8"))["cases"]:
            with self.subTest(pr=case["pr"]):
                client = self.client()
                # Replay public IDs and suite topology; bind the recorded head
                # and external ID to the existing synthetic Plan/run fixture.
                gate = copy.deepcopy(case["gate"])
                gate.update(
                    head_sha=fixtures.HEAD, external_id="semantic-router-ci:3:101:1"
                )
                client.checks = [gate]
                client.visible = {gate["id"]}
                client.required = {gate["id"]: True}
                client.run["check_suite_id"] = case["qualification_run"][
                    "check_suite_id"
                ]
                suites = [case["gate_suite"], case["qualification_suite"]]
                client.suites = {
                    suite["id"]: {**copy.deepcopy(suite), "head_sha": fixtures.HEAD}
                    for suite in suites
                }
                client.rollup["state"] = case["rollup"]["statusCheckRollup"]["state"]
                result = self.diagnose(client)
                self.assertEqual(result["state"], case["expected_state"])
                self.assertEqual(result["run_suite_id"], client.run["check_suite_id"])

    def test_partial_or_malformed_graphql_response_cannot_establish_health(self):
        for payload in (
            {},
            {"data": []},
            {"data": {"repository": []}},
            {"data": {"repository": {"object": None}}},
            {"errors": [{"message": "private raw value"}]},
        ):
            with self.subTest(payload=payload):
                client = self.client()
                client.gh = lambda *a, payload=payload, **k: SimpleNamespace(
                    returncode=0, stdout=json.dumps(payload)
                )
                result = self.diagnose(client)
                self.assertEqual(result["state"], "evidence_unavailable")
                self.assertNotIn("private raw value", json.dumps(result))

    def test_suite_inventory_is_bounded_before_fetching_each_suite(self):
        client = self.client()
        template = client.checks[0]
        client.checks = [
            {**copy.deepcopy(template), "id": identity, "check_suite": {"id": identity}}
            for identity in range(1000, 1101)
        ]
        client.required = {check["id"]: True for check in client.checks}
        client.visible = set(client.required)
        result = self.diagnose(client)
        self.assertEqual(result["state"], "evidence_unavailable")
        self.assertFalse(any("/check-suites/" in path for path in client.reads))


if __name__ == "__main__":
    unittest.main()
