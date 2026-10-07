from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))
from check_ci_gate import evaluate_gate  # noqa: E402
from ci_plan import digest  # noqa: E402
from ci_results import collection_errors, make_receipt  # noqa: E402
from known_issue_waivers import planned_waiver, recipe_waiver  # noqa: E402
from release_guard_waiver import GUARD_WAIVER  # noqa: E402
from test_ci_gate import SHA, completed  # noqa: E402

NAMED = "standalone:balance:request:casual_chat:long_unclassified_fallback"
OTHER = "standalone:balance:request:formal_math_proof:infinite_primes"
ACCEPTANCE = "standalone:balance:acceptance"


def recipe_evidence(failures: list[dict], cases: list[dict]) -> dict:
    return {
        "cases": cases,
        "expected_cases": [case["id"] for case in cases],
        "known_issue_waiver": recipe_waiver(),
        "waived_failures": failures,
    }


class RecipeCpuWaiverTests(unittest.TestCase):
    def test_every_live_recipe_profile_plans_the_same_waiver(self):
        for profile in ("pr", "main", "nightly", "release"):
            with self.subTest(profile=profile):
                self.assertEqual(
                    planned_waiver(profile, "recipe-conformance"), recipe_waiver()
                )
        self.assertIsNone(planned_waiver("pr", "e2e.production-stack"))
        self.assertEqual(
            planned_waiver("release", "e2e.production-stack"), GUARD_WAIVER
        )
        waiver = recipe_waiver()
        self.assertEqual(waiver["issue"], 4706)
        self.assertIn(NAMED, waiver["cases"])
        self.assertNotIn(OTHER, waiver["cases"])
        self.assertIn("#4706", waiver["removal"])

    def test_only_a_named_preview_timeout_is_accepted(self):
        waiver = recipe_waiver()
        passing = [
            {"id": OTHER, "status": "passed"},
            {"id": ACCEPTANCE, "status": "passed"},
        ]
        timeout = [{"case": NAMED, "outcome": "request_timeout"}]
        accepted = recipe_evidence(
            timeout, [{"id": NAMED, "status": "failed"}, *passing]
        )
        self.assertEqual(collection_errors(accepted, "test", waiver=waiver), [])

        rejected = {
            "unnamed failure": recipe_evidence(
                timeout,
                [
                    {"id": NAMED, "status": "failed"},
                    {"id": OTHER, "status": "failed"},
                    {"id": ACCEPTANCE, "status": "passed"},
                ],
            ),
            "failed acceptance": recipe_evidence(
                timeout,
                [
                    {"id": NAMED, "status": "failed"},
                    {"id": OTHER, "status": "passed"},
                    {"id": ACCEPTANCE, "status": "failed"},
                ],
            ),
            "failure without a timeout": recipe_evidence(
                [{"case": NAMED, "outcome": "signals_cut_at_deadline"}],
                [{"id": NAMED, "status": "failed"}, *passing],
            ),
            "other outcome": recipe_evidence(
                [{"case": NAMED, "outcome": "misroute"}],
                [{"id": NAMED, "status": "failed"}, *passing],
            ),
            "unnamed waived case": recipe_evidence(
                [*timeout, {"case": OTHER, "outcome": "request_timeout"}],
                [{"id": NAMED, "status": "failed"}, *passing],
            ),
            "no waived failure": recipe_evidence(
                [], [{"id": NAMED, "status": "failed"}, *passing]
            ),
        }
        for name, evidence in rejected.items():
            with self.subTest(name=name):
                self.assertTrue(collection_errors(evidence, "test", waiver=waiver))
        changed = copy.deepcopy(waiver)
        changed["cases"].append(OTHER)
        self.assertTrue(collection_errors(accepted, "test", waiver=changed))
        self.assertTrue(collection_errors(accepted, "test"))

    def test_gate_accepts_the_planned_recipe_waiver_in_pull_requests(self):
        plan, receipts, builds = completed(["config/recipes/balance/probes.yaml"])
        record = next(
            row for row in plan["verifications"] if row["id"] == "recipe-conformance"
        )
        self.assertEqual(record["known_issue_waiver"], recipe_waiver())
        receipt = next(row for row in receipts if row["id"] == record["id"])
        evidence = copy.deepcopy(receipt["evidence"])
        evidence.update(
            recipe_evidence(
                [{"case": NAMED, "outcome": "request_timeout"}],
                [
                    {"id": NAMED, "status": "failed"},
                    {"id": OTHER, "status": "passed"},
                    {"id": ACCEPTANCE, "status": "passed"},
                ],
            )
        )
        qualified = make_receipt(
            record,
            evidence,
            source_sha=SHA,
            execution_platform="linux/amd64",
            environ={},
        )
        self.assertEqual(qualified["result"], "qualified-with-waiver")
        receipts[receipts.index(receipt)] = qualified
        self.assertTrue(evaluate_gate(plan, receipts, builds=builds).passed)

        tampered = copy.deepcopy(plan)
        changed = next(
            row for row in tampered["verifications"] if row["id"] == record["id"]
        )
        changed["known_issue_waiver"]["cases"].append(OTHER)
        tampered["plan_sha256"] = digest(
            {key: value for key, value in tampered.items() if key != "plan_sha256"}
        )
        self.assertFalse(evaluate_gate(tampered, receipts, builds=builds).passed)


if __name__ == "__main__":
    unittest.main()
