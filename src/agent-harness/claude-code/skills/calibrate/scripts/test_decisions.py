import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cases import guard
from decisions import Decision, features, implicit_labels, read_log


class Features(unittest.TestCase):
    def test_escalate_reads_both_signals(self):
        reask_only = Decision(
            "e1",
            None,
            "s",
            "escalate",
            "escalated",
            0.8,
            0.75,
            "again",
            {"feedback": None, "reask": 0.8},
            learnable=True,
        )
        self.assertEqual(
            features(reask_only),
            (0.0, 0.8),
            "the reask score is not taken for feedback",
        )
        both = Decision(
            "e2",
            None,
            "s",
            "escalate",
            "escalated",
            0.9,
            0.8,
            "no",
            {
                "feedback": {"WRONG_ANSWER": 0.9, "WANT_DIFFERENT": 0.05, "SAT": 0.01},
                "reask": None,
            },
            learnable=True,
        )
        self.assertEqual(features(both), (0.9, 0.0))

    def test_a_decision_the_mod_offers_no_answers_for_is_no_threshold_case(self):
        ruled = guard(
            2,
            1.0,
            "held",
            learnable=False,
            rule="deletes a whole top-level directory",
            outcome="declined",
        )
        self.assertIsNone(features(ruled))
        self.assertEqual(
            implicit_labels([ruled]),
            {},
            "a declined hold by a fixed rule says nothing of its score",
        )
        self.assertEqual(features(guard(3, 0.95, "held")), (0.95,))


class LogParsing(unittest.TestCase):
    def test_partial_last_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            full = [
                {
                    "id": "a",
                    "at": "2026-11-06T10:00:00Z",
                    "session": "s",
                    "app": "guard",
                    "verdict": "held",
                    "score": 0.95,
                    "threshold": 0.9,
                    "subject": "git push -f",
                    "detail": {"outcome": "declined"},
                    "learnable": True,
                },
                {
                    "id": "b",
                    "at": 1793700000000,
                    "session": "s",
                    "app": "shield",
                    "verdict": "shielded",
                    "score": 0.99,
                    "threshold": 0.9,
                    "subject": "INSTALL.md",
                    "detail": {"source": "vendor/INSTALL.md"},
                },
            ]
            path.write_text(
                "".join(json.dumps(x) + "\n" for x in full) + '{"id": "c", "app": "gua'
            )
            log = read_log(path)
            self.assertEqual([d.id for d in log.decisions], ["a", "b"])
            self.assertEqual(log.skipped, 1)
            self.assertAlmostEqual(log.decisions[1].at, 1793700000.0)
            self.assertIsNotNone(log.decisions[0].at)
            self.assertEqual(
                [d.learnable for d in log.decisions],
                [True, False],
                "a line without the field was logged before it existed",
            )

    def test_missing_log_is_empty(self):
        self.assertEqual(read_log(Path("/nonexistent/decisions.jsonl")).decisions, [])


class ImplicitLabels(unittest.TestCase):
    def test_declined_hold_is_yes(self):
        found = implicit_labels(
            [
                guard(1, 0.95, "held", outcome="declined"),
                guard(2, 0.95, "held", outcome="approved"),
            ]
        )
        self.assertEqual(found, {"g1": "yes"})

    def test_escalation_soon_after_a_light_route(self):
        r_haiku = Decision(
            "r1",
            1000.0,
            "s",
            "router",
            "routed",
            0.2,
            None,
            "t",
            {"model": "haiku"},
            True,
        )
        r_opus = Decision(
            "r2",
            1000.0,
            "s2",
            "router",
            "routed",
            1.8,
            None,
            "t",
            {"model": "opus"},
            True,
        )
        e1 = Decision(
            "e1", 1200.0, "s", "escalate", "escalated", 0.92, 0.8, "no, wrong", {}
        )
        e2 = Decision(
            "e2", 1200.0, "s2", "escalate", "escalated", 0.92, 0.8, "no, wrong", {}
        )
        late = Decision(
            "r3",
            1000.0,
            "s3",
            "router",
            "routed",
            0.2,
            None,
            "t",
            {"model": "sonnet"},
            True,
        )
        e3 = Decision(
            "e3", 1000.0 + 3600, "s3", "escalate", "escalated", 0.92, 0.8, "no", {}
        )
        self.assertEqual(
            implicit_labels([r_haiku, r_opus, e1, e2, late, e3]), {"r1": "too-low"}
        )


if __name__ == "__main__":
    unittest.main()
