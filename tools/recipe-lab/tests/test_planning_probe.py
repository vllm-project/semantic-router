"""Check acceptance of real and fixture observations at a public recipe boundary."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from recipe_lab.planning_probe import assess_probe


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def observed(name, roles, lane, model):
    return {
        "case": name,
        "roles": roles,
        "public_model": "entry",
        "http": 200,
        "selected": model,
        "returned": model + "-up",
        "decision": lane,
        "request_sha256": sha(name),
    }


def sample(kind):
    two = ["user", "assistant", "tool", "assistant", "tool"]
    rows = [
        observed("count-1", two[:-2], "fallback", "A"),
        observed("count-2", two, "late", "B"),
        {
            "case": "unsupported-image",
            "request_sha256": sha("image"),
            "selected": None,
            "returned": None,
            "http": 503,
            "code": "no_eligible_model",
        },
        {
            "case": "output-exceeds-both",
            "request_sha256": sha("budget"),
            "selected": None,
            "returned": None,
            "http": 400,
            "code": "max_output_tokens_exceeded",
        },
    ]
    if kind == "real":
        rows.append({**observed("tool-call", two, "late", "B"), "tool_ok": True})
        rows.append(
            {
                **observed("tool-result", [*two, "assistant", "tool"], "late", "B"),
                "tool_ok": True,
            }
        )
    else:
        rows.append(observed("long-input", two, "late", "A"))
    return {
        "kind": kind,
        "backend_mode": kind,
        "passed": True,
        "source_sha": sha("candidate"),
        "projected_sha": sha("projected deployment"),
        "public_model": "entry",
        "probes": rows,
    }


class PlanningProbeTest(unittest.TestCase):
    def evaluate(self, receipt, *, require_context_fallback=False, **extras):
        predicate = {
            "feature": "tool_results",
            "lower": 2,
            "upper": None,
            "decision": "late",
            "fallback_decision": "fallback",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            return assess_probe(
                path,
                candidate_sha=sha("candidate"),
                public_model="entry",
                fast="B",
                strong="A",
                backend_aliases={"A": "A-up", "B": "B-up"},
                predicate={**predicate, **extras},
                require_context_fallback=require_context_fallback,
            )

    def test_real_and_fixture_evidence_stay_separate(self):
        real = self.evaluate(sample("real"))
        fixture = self.evaluate(sample("fixture"))
        self.assertEqual(
            (real["status"], fixture["status"]), ("real_pass", "fixture_pass")
        )
        self.assertTrue(real["tool_roundtrip_verified"])
        self.assertFalse(fixture["tool_roundtrip_verified"])
        self.assertEqual(real["tested_boundary_counts"], [1, 2])

    def test_context_fallback_when_model_limits_differ(self):
        real = sample("real")
        report = self.evaluate(real, require_context_fallback=True)
        self.assertEqual(report["status"], "incomplete")
        self.assertFalse(report["context_fallback_observed"])
        fallback = sample("fixture")["probes"][-1]
        real["probes"].append(fallback)
        valid = self.evaluate(real, require_context_fallback=True)
        self.assertTrue(valid["context_fallback_observed"])
        self.assertEqual(valid["status"], "real_pass")
        fallback["selected"] = "B"
        invalid = self.evaluate(real, require_context_fallback=True)
        self.assertEqual(invalid["status"], "failed")
        self.assertFalse(invalid["context_fallback_observed"])

    def test_invalid_identity_mode_or_reported_failure_cannot_pass(self):
        for key, bad in [
            ("kind", "synthetic"),
            ("backend_mode", "fixture"),
            ("source_sha", sha("wrong")),
            ("public_model", "other"),
            ("passed", False),
        ]:
            with self.subTest(field=key):
                receipt = {**sample("real"), key: bad}
                report = self.evaluate(receipt)
                self.assertEqual(report["status"], "failed")
                self.assertIn(key, [i["field"] for i in report["issues"]])

    def test_masked_branch_wrong_return_or_mutated_count_cannot_pass(self):
        for key, bad in [
            ("returned", "other"),
            ("selected", "A"),
            ("decision", "other-to-B"),
            ("roles", ["user", "tool"]),
            ("public_model", "other"),
        ]:
            with self.subTest(field=key):
                receipt = sample("real")
                receipt["probes"][1][key] = bad
                self.assertEqual(self.evaluate(receipt)["status"], "failed")

    def test_both_edges_required_for_finite_window(self):
        report = self.evaluate(sample("fixture"), upper=4)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual({i["case"] for i in report["issues"]}, {"count-4", "count-5"})

    def test_tool_result_is_a_continuation_and_must_succeed(self):
        receipt = sample("real")
        probe = receipt["probes"][-1]
        probe["tool_ok"] = False
        self.assertEqual(self.evaluate(receipt)["status"], "failed")
        probe["tool_ok"] = True
        probe["roles"][0] = "system"
        self.assertEqual(self.evaluate(receipt)["status"], "failed")
        receipt["probes"].pop()
        self.assertEqual(self.evaluate(receipt)["status"], "incomplete")

    def test_refusal_checks_both_error_and_backend(self):
        for key, bad in [("http", 200), ("code", "network_timeout"), ("returned", "B")]:
            with self.subTest(field=key):
                receipt = sample("real")
                receipt["probes"][2][key] = bad
                self.assertEqual(self.evaluate(receipt)["status"], "failed")

    def test_missing_required_output_refusal_and_extra_tool_turn(self):
        receipt = sample("real")
        receipt["probes"] = [
            r for r in receipt["probes"] if r["case"] != "output-exceeds-both"
        ]
        self.assertEqual(self.evaluate(receipt)["status"], "incomplete")
        receipt = sample("real")
        receipt["probes"][-1]["roles"].append("user")
        self.assertEqual(self.evaluate(receipt)["status"], "failed")

    def test_missing_boundary_or_reused_request_is_detected(self):
        receipt = sample("real")
        receipt["probes"][1]["request_sha256"] = receipt["probes"][0]["request_sha256"]
        self.assertEqual(self.evaluate(receipt)["status"], "failed")
        receipt["probes"].pop(1)
        self.assertEqual(self.evaluate(receipt)["status"], "incomplete")

    def test_filtered_message_count_needs_more_than_wire_roles(self):
        receipt = sample("fixture")
        receipt["probes"][0]["roles"] = ["user"]
        receipt["probes"][1]["roles"] = ["user", "user"]
        report = self.evaluate(
            receipt, feature="all_messages", skip_user_messages_matching=True
        )
        self.assertEqual(report["status"], "incomplete")
        self.assertIn(
            "filtered user count requires message content",
            [i["reason"] for i in report["issues"]],
        )
