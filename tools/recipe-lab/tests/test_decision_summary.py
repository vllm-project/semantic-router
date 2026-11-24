"""Fresh development decisions are visible only after both evidence and state agree."""

import unittest
import json
from pathlib import Path
from unittest.mock import patch

import test_schedule
import test_signature
from recipe_lab.decision_summary import summarize
from test_loop import get, put


class DecisionReadbackTest(unittest.TestCase):
    def setUp(self):
        f = test_schedule.ScheduleTest()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        # The reusable schedule fixture mutates its precursor's synthetic packet
        # and then records a new two-task baseline. Only its current records
        # belong in this end-to-end workspace for next.
        rows = [f.ledger.measure(ident) for ident in (f.base, f.measured)]
        f.ledger.path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        self.card = f.card
        self.state = f.ledger.path.parent / "schedules/pilot/state.json"

    def summary(self):
        return summarize(self.f.ledger, self.card)

    def test_pre_enrollment_and_pending_are_read_only(self):
        f = self.f
        self.assertEqual(self.summary()["decision"], "not_enrolled")
        self.assertFalse(self.state.parent.exists())
        slot = f.poll()["next"]
        attempt = f.fresh_run(slot)
        f.poll(attempt)
        snapshot = self.state.read_bytes()
        self.assertNotIn("by_task", self.summary())
        ctx = f.run.cli("next", "--schedule-card", self.card)
        self.assertEqual(ctx["pending_development_studies"], ["pilot"])
        self.assertEqual(ctx["development_decisions"], [])
        self.assertNotIn(
            attempt["measurement"], [m["record_id"] for m in ctx["measurements"]]
        )
        self.assertEqual(self.state.read_bytes(), snapshot)

    def test_completed_decision_compares_only_exact_requests_within_each_pair(self):
        f = self.f
        step = f.poll()
        while step["status"] == "run":
            slot = step["next"]
            rows = test_signature.receipts([4])
            if slot["task"] == "guard":
                rows[0].update(body_sha256_in="1" * 64, body_sha256_sent="1" * 64)
            if slot["role"] == "candidate":
                rows[1].update(selected_model="B", returned_model="B-upstream")
            step = f.poll(f.fresh_run(slot, receipts=rows))
        snapshot = self.state.read_bytes()
        summary = self.summary()
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["decision"], step["status"])
        self.assertTrue(summary["reopen_when"])
        guard = summary["by_task"]["guard"]
        self.assertEqual(guard["paired"], step["paired_by_task"]["guard"])
        self.assertEqual(guard["matched_requests"][0]["different_selected_models"], 1)
        self.assertIsNone(
            summary["by_task"]["task"]["matched_requests"][0][
                "different_selected_models"
            ]
        )
        self.assertEqual(f.run.cli("decision-summary", "--card", self.card), summary)
        self.assertEqual(
            f.run.cli("next", "--schedule-card", self.card)["development_decisions"],
            [summary],
        )
        self.assertEqual(self.state.read_bytes(), snapshot)
        receipt = (
            self.f.root
            / guard["attempts"][0]["assignment_id"]
            / "dsh/runs/guard/fixed/main/requests.jsonl"
        )
        receipt.write_bytes(receipt.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            self.summary()

    def test_failed_output_limit_is_released_with_the_completed_decision(self):
        f = self.f
        symptom = "strict_failure_native_output_limit"
        step = f.poll()
        while step["status"] == "run":
            slot = step["next"]
            rows = test_signature.receipts([4])
            rows[0]["params"] = {"max_completion_tokens": 32768, "private": "omit-me"}
            rows[1]["finish"] = "length"
            rows[1]["usage"] = {
                "completion_tokens": 32768,
                "completion_tokens_details": {"reasoning_tokens": 32768},
            }
            failed = slot["role"] == "baseline"
            step = f.poll(
                f.fresh_run(
                    slot,
                    receipts=rows,
                    strict=not failed,
                    termination=(
                        "native_output_limit" if failed else "natural_completion"
                    ),
                )
            )
            if step["status"] == "run":
                for args in [(), ("--schedule-card", self.card)]:
                    self.assertNotIn(symptom, f.run.cli("next", *args)["symptoms"])
        out = f.run.cli("next", "--schedule-card", self.card)
        self.assertIn(
            "native-output-limit-on-strict-failure",
            [r["id"] for r in out["references"]],
        )
        (summary,) = out["development_decisions"]
        terminal = summary["by_task"]["guard"]["attempts"][0]["terminal"]
        self.assertEqual(
            terminal,
            {
                "termination": "native_output_limit",
                "last_call": 1,
                "finish_reason": "length",
                "explicit_request_limits": {"max_completion_tokens": 32768},
                "completion_tokens": 32768,
                "reasoning_tokens": 32768,
                "upstream_request_limits": None,
            },
        )
        self.assertNotIn("omit-me", json.dumps(summary))

    def test_invalid_cost_has_no_grade_and_holdout_evidence_is_never_opened(self):
        f = self.f
        put(self.card, dict(f.protocol, budget_episodes=1))
        attempt = f.fresh_run(f.poll()["next"], status="runner_error", early=True)
        f.poll(attempt)
        summary = self.summary()
        self.assertEqual(summary["decision"], "defer")
        self.assertEqual(summary["cost"]["invalid_attempt_wall_seconds"], 8)
        row = summary["by_task"]["guard"]["attempts"][0]
        self.assertNotIn("strict", row)
        self.assertNotIn("routing", row)
        state = get(self.state)
        held = f.ledger.append(
            dict(
                f.ledger.measure(f.base),
                split="holdout",
                evidence={str(self.f.root / "sealed"): "f" * 64},
            )
        )
        state["attempts"][0]["evidence"] = {"measurement": held["record_id"]}
        put(self.state, state)
        original = Path.read_bytes

        def guarded(path):
            if path.name == "sealed":
                raise AssertionError("opened holdout evidence")
            return original(path)

        with (
            patch.object(Path, "read_bytes", guarded),
            self.assertRaisesRegex(ValueError, "development evidence"),
        ):
            self.summary()
