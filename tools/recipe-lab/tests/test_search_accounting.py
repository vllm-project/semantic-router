"""Prospective registration, ordered trial starts and source-bound provider usage."""

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

from recipe_lab import search_accounting as accounting
from recipe_lab.loop import Ledger
from recipe_lab.loop_evidence import digest
from tests.test_loop import get, put


def pin(path):
    return {"file": path.name, "sha256": digest(path.read_bytes())}


class SearchAccountingTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.ledger = Ledger(self.root)
        self.methods = {
            m: [m + "-1", m + "-2"] for m in accounting.search_protocol.METHODS
        }
        self.report = {
            "study": "exp",
            "protocol_sha256": "1" * 64,
            "space_sha256": "2" * 64,
            "status": "target_actions_distinguished",
            "target_development": {
                "actions": [
                    {"upper": u, "config_sha256": str(u) * 64} for u in (1, 2, 3)
                ]
            },
            "comparison": {
                "target_selection_headroom": "non_exhaustive",
                "candidate_trials_per_method": 2,
                "split_plans": {},
                "shared_class_representatives": [1, 2, 3],
                "random_class_order": [2, 1, 3],
                "shared_budget_per_method": {
                    "episodes": 8,
                    "proposal_model_tokens": 24,
                },
            },
        }
        self.spec = self.root / "slots.json"
        put(
            self.spec,
            {
                "schema": "recipe-lab-search-allocation-v1",
                "protocol_sha256": "1" * 64,
                "study": "exp",
                "methods": self.methods,
            },
        )
        self.args = SimpleNamespace(allocation=self.spec)
        patcher = patch.object(
            accounting.search_protocol, "preflight", return_value=self.report
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def enroll(self):
        return accounting.enroll(self.ledger, self.args)

    def test_enrollment_is_immutable_and_idempotent(self):
        first = self.enroll()
        self.assertEqual(first, self.enroll())
        self.assertEqual(accounting.load_enrollment(self.ledger, self.report), first)
        other = dict(get(self.spec))
        other["methods"]["random"][0] = "new-name"
        put(self.spec, other)
        with self.assertRaisesRegex(ValueError, "already enrolled"):
            self.enroll()
        with self.assertRaisesRegex(ValueError, "preflight changed"):
            accounting.load_enrollment(
                self.ledger, dict(self.report, space_sha256="3" * 64)
            )

    def test_preexisting_slot_or_shared_method_name_is_rejected(self):
        spec = get(self.spec)
        spec["methods"]["random"][1] = spec["methods"]["guided"][0]
        put(self.spec, spec)
        with self.assertRaisesRegex(ValueError, "unique"):
            self.enroll()
        spec["methods"] = self.methods
        put(self.spec, spec)
        put(self.root / "schedules" / "random-1" / "state.json", {})
        with self.assertRaisesRegex(ValueError, "already started"):
            self.enroll()

    def test_inflight_unregistered_schedule_is_detected_before_enrollment(self):
        name = sorted(slot for slots in self.methods.values() for slot in slots)[0]
        native = self.root / "schedules" / name
        native.mkdir(parents=True)
        writer_lock = accounting.io().writer_lock
        attempted = Event()

        @contextmanager
        def watch(path):
            if path == native / "writer.lock":
                attempted.set()
            with writer_lock(path):
                yield

        with ThreadPoolExecutor(max_workers=1) as pool:
            with writer_lock(native / "writer.lock"):
                self.assertIsNone(
                    accounting.before_trial(self.ledger, name, "1" * 64, "2" * 64)
                )
                with patch.object(accounting.io(), "writer_lock", watch):
                    future = pool.submit(self.enroll)
                    self.assertTrue(attempted.wait(timeout=5))
                    put(native / "state.json", {"card_sha256": "1" * 64})
            with self.assertRaisesRegex(ValueError, "already started"):
                future.result(timeout=5)

    def test_order_and_fresh_native_enrollment_required_even_on_concurrent_start(self):
        ledger = self.ledger
        # An ordinary schedule that checked just before a concurrent search enrollment
        # would have None. Its state cannot later masquerade as an enrolled trial.
        self.assertIsNone(
            accounting.before_trial(ledger, "random-1", "2" * 64, "2" * 64)
        )
        enrolled = self.enroll()
        with self.assertRaisesRegex(ValueError, "previous trial"):
            accounting.before_trial(ledger, "random-2", "1" * 64, "1" * 64)
        valid = accounting.before_trial(ledger, "random-1", "2" * 64, "2" * 64)[
            "enrollment"
        ]
        self.assertEqual(valid, enrolled["enrollment"])
        accounting.after_trial(ledger, "random-1", "2" * 64, "defer", 0)
        path = self.root / "schedules" / "random-1" / "state.json"
        state = {"card_sha256": "2" * 64, "attempts": []}
        put(path, state)
        with self.assertRaisesRegex(ValueError, "not bound"):
            accounting.check_trial_slots(ledger, enrolled, "random", ["random-1"])
        with self.assertRaisesRegex(ValueError, "omitted"):
            accounting.check_trial_slots(ledger, enrolled, "random", [])
        put(path, dict(state, search_enrollment=valid))
        slots = accounting.check_trial_slots(ledger, enrolled, "random", ["random-1"])
        self.assertEqual([s["status"] for s in slots], ["defer", "unused_slot"])
        self.assertEqual(slots[0]["episodes"], 0)
        self.assertEqual(
            accounting.before_trial(ledger, "random-2", "1" * 64, "1" * 64)[
                "enrollment"
            ],
            valid,
        )
        with self.assertRaisesRegex(ValueError, "omitted"):
            accounting.check_trial_slots(ledger, enrolled, "random", ["random-1"])
        put(self.root / "search-studies" / "exp" / "freeze.json", {})
        with self.assertRaisesRegex(ValueError, "already frozen"):
            accounting.before_trial(ledger, "guided-1", "4" * 64, "2" * 64)

    def test_candidates_and_shared_remaining_budget_are_bound_before_dispatch(self):
        self.enroll()

        def opened(study, candidate):
            return accounting.before_trial(self.ledger, study, "f" * 64, candidate * 64)

        with self.assertRaisesRegex(ValueError, "frozen random order"):
            opened("random-1", "1")
        with self.assertRaisesRegex(ValueError, "outside enrolled search space"):
            opened("guided-1", "e")
        self.assertEqual(opened("guided-1", "2")["remaining_method_episodes"], 8)
        accounting.after_trial(self.ledger, "guided-1", "f" * 64, "defer", 7)
        with self.assertRaisesRegex(ValueError, "candidate already used"):
            opened("guided-2", "2")
        opened("guided-2", "1")
        card = {"study": "guided-2", "budget_episodes": 5}
        self.assertEqual(
            accounting.historical_card(self.ledger, card, "f" * 64)["budget_episodes"],
            1,
        )
        self.assertEqual(opened("random-1", "2")["remaining_method_episodes"], 8)

    def test_provider_usage_rechecks_raw_file_and_counts(self):
        enrolled = self.enroll()
        call = self.root / "a.json"
        response = {
            "id": "req-1",
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }
        envelope = {
            "schema": "recipe-lab-proposal-call-v1",
            "enrollment": enrolled["enrollment"],
            "trial_study": "guided-1",
            "method": "guided",
            "response": response,
        }
        put(call, envelope)
        manifest_path = self.root / "raw.json"
        manifest = {
            "schema": "recipe-lab-search-usage-v1",
            "enrollment": enrolled["enrollment"],
            "method": "guided",
            "calls": [pin(call)],
        }
        put(manifest_path, manifest)

        def cost(used):
            return accounting.proposal_usage(
                self.root, pin(manifest_path), used, enrolled, "guided"
            )

        seen = set()
        self.assertEqual(cost(seen), 12)
        with self.assertRaisesRegex(ValueError, "reused"):
            cost(seen)
        response["usage"]["total_tokens"] = 0
        put(call, envelope)
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            cost(set())
        put(manifest_path, dict(manifest, calls=[pin(call)]))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            cost(set())
        put(call, dict(envelope, trial_study="random-1"))
        put(manifest_path, dict(manifest, calls=[pin(call)]))
        with self.assertRaisesRegex(ValueError, "another enrollment or trial"):
            cost(set())
        put(manifest_path, dict(manifest, calls=[]))
        self.assertEqual(cost(set()), 0)

    def test_schedule_allows_inflight_work_and_charges_invalid_retry_to_shared_ceiling(
        self,
    ):
        from test_schedule import ScheduleTest

        trial = ScheduleTest()
        trial.setUp()
        self.addCleanup(trial.doCleanups)
        actual = trial.report["candidate_sha256"]
        self.report["target_development"]["actions"][1]["config_sha256"] = actual
        enrolled = accounting.enroll(trial.ledger, self.args)
        accounting.before_trial(trial.ledger, "guided-1", "f" * 64, "1" * 64)
        accounting.after_trial(trial.ledger, "guided-1", "f" * 64, "defer", 7)
        put(trial.card, dict(trial.protocol, study="guided-2"))
        first = trial.poll()
        self.assertEqual(first["budget_episodes"], 1)
        self.assertEqual(trial.poll()["next"], first["next"])
        attempt = trial.fresh_run(first["next"], status="runner_error", early=True)
        last = trial.poll(attempt)
        self.assertEqual(last["status"], "defer")
        self.assertNotIn("next", last)
        self.assertEqual(last["spent_episodes"], 1)
        self.assertEqual(last["invalid_attempts"], 1)
        self.assertEqual(trial.poll(attempt), last)
        summary = trial.run.cli("decision-summary", "--card", trial.card)
        self.assertEqual(summary["cost"]["budget_episodes"], 1)
        self.assertEqual(summary["decision"], "defer")
        self.assertEqual(enrolled["budget"]["episodes"], 8)


if __name__ == "__main__":
    unittest.main()
