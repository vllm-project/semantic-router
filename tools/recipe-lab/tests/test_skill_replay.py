"""Agent decision replay uses only the enrolled development prefix until scoring."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from recipe_lab.skill_replay import run
from tests import test_schedule
from tests.test_loop import get, put


class SkillReplayTest(unittest.TestCase):
    def setUp(self):
        f = test_schedule.ScheduleTest()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        before, after = f.root / "old.md", f.root / "new.md"
        before.write_text("Continue the scheduled comparison.", encoding="utf-8")
        after.write_text(
            "Stop once the remaining result cannot change selection.", encoding="utf-8"
        )
        self.args = SimpleNamespace(
            point=[(f.card, "1")],
            before_skill=before,
            after_skill=after,
            revision_source=[],
            decisions=None,
        )
        self.state = f.ledger.path.parent / "schedules/pilot/state.json"
        self.source = f.run.cli(
            "compare", "--baseline", f.base, "--candidate", f.measured
        )["record_id"]
        self.step = self.f.poll()
        self.take()

    def take(self, **kwargs):
        slot = self.step["next"]
        result = self.f.fresh_run(slot, **kwargs)
        self.step = self.f.poll(result)
        return result

    def packet(self):
        return run(self.f.ledger, self.args)

    def choices(self, packet, before="continue", after="keep_parent"):
        file = self.f.root / "choices.json"
        pair = {
            k: {"choice": v, "skill_sha256": packet["skills"][k]}
            for k, v in (("before", before), ("after", after))
        }
        put(
            file, {"packet_sha256": packet["packet_sha256"], "choices": {"pilot": pair}}
        )
        self.args.decisions = file

    def test_packet_stable_after_future_grades_and_positive_utility_pinned(self):
        self.args.revision_source = [self.source]
        packet = self.packet()
        self.take(strict=False)
        self.assertEqual(self.step["status"], "drop")
        self.assertEqual(self.packet(), packet)
        self.assertEqual(len(packet["points"]["pilot"]["history"]), 1)
        self.assertNotIn("observed_outcome", json.dumps(packet))
        self.choices(packet)
        score = self.packet()
        self.assertEqual(score["verdict"], "positive_historical_utility")
        self.assertEqual(score["counterfactual_episodes_saved"], 1)
        self.assertEqual(self.packet(), score)
        self.choices(packet, after="continue")
        with self.assertRaisesRegex(ValueError, "already frozen"):
            self.packet()

    def test_quality_regression_overrides_cost_savings(self):
        packet = self.packet()
        while self.step["status"] == "run":
            slot = self.step["next"]
            self.take(strict=(slot["task"], slot["role"]) != ("task", "baseline"))
        self.assertEqual(self.step["status"], "advance")
        self.choices(packet)
        score = self.packet()
        self.assertGreater(score["counterfactual_episodes_saved"], 0)
        self.assertEqual(score["verdict"], "reject_quality_regression")

    def test_pending_development_and_wrong_choice_contract(self):
        packet = self.packet()
        self.choices(packet)
        payload = get(self.args.decisions)
        payload["choices"]["pilot"]["after"]["skill_sha256"] = packet["skills"][
            "before"
        ]
        put(self.args.decisions, payload)
        with self.assertRaisesRegex(ValueError, "skill identity"):
            self.packet()
        self.choices(packet)
        self.assertEqual(self.packet()["verdict"], "incomplete_development_outcomes")
        self.take(strict=False)
        self.assertEqual(self.packet()["verdict"], "positive_needs_revision_source")

    def test_future_and_holdout_sources_fail_before_evidence_open(self):
        later = self.take(strict=False)["measurement"]
        source = self.f.run.cli(
            "compare",
            "--baseline",
            self.f.base,
            "--candidate",
            later,
            "--schedule-card",
            self.f.card,
        )
        self.args.revision_source = [source["record_id"]]
        with self.assertRaisesRegex(ValueError, "future episode"):
            self.packet()
        held = self.f.ledger.append(dict(source, split="holdout"))
        self.args.revision_source = [held["record_id"]]
        with self.assertRaisesRegex(ValueError, "dev comparison"):
            self.packet()

    def test_prefix_does_not_open_future_or_sealed_task_files(self):
        packet = self.packet()
        future = self.take(strict=False)["measurement"]
        forbidden = set(self.f.ledger.measure(future)["evidence"])
        original = Path.read_bytes

        def read(path):
            self.assertNotIn(str(path), forbidden)
            return original(path)

        with patch.object(Path, "read_bytes", read):
            self.assertEqual(self.packet(), packet)
        state = get(self.state)
        held = self.f.ledger.append(
            dict(
                self.f.ledger.measure(future),
                split="holdout",
                evidence={str(self.f.root / "sealed"): "1" * 64},
            )
        )
        state["attempts"][1]["evidence"]["measurement"] = held["record_id"]
        put(self.state, state)
        self.assertEqual(self.packet(), packet)
        self.choices(packet)
        with self.assertRaisesRegex(ValueError, "development evidence"):
            self.packet()

    def test_unfrozen_cutoff_and_duplicate_study(self):
        self.args.point.append((self.f.card, "0"))
        with self.assertRaisesRegex(ValueError, "one checkpoint"):
            self.packet()
        self.args.point = [(self.f.card, "2")]
        with self.assertRaisesRegex(ValueError, "cutoff"):
            self.packet()


if __name__ == "__main__":
    unittest.main()
