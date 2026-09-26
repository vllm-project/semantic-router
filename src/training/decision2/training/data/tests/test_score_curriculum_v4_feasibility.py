"""Document the prospective Score v4 mechanical gate and its negative dry run."""

from __future__ import annotations

import collections
import random
import unittest

from training.data import build_score_curriculum_v4 as curriculum


class ScoreCurriculumV4FeasibilityTests(unittest.TestCase):
    def test_obligation_raw_event_counters_cannot_label_a_triplet(self) -> None:
        states = curriculum._obligation_states(
            random.Random(7), "library program TEST", "en", 0
        )
        self.assertEqual(
            [curriculum.oracle("obligation_review", state) for state in states],
            [0, 1, 2],
        )
        shallow = []
        for state in states:
            self.assertEqual(len(state["events"]), 9)
            self.assertEqual(state["events"][0]["scope"], "informational")
            self.assertEqual(state["events"][-1]["scope"], "informational")
            shallow.append(
                (
                    collections.Counter(
                        event["assessment"] for event in state["events"]
                    ),
                    {
                        name: collections.Counter(
                            event["assessment"]
                            for event in state["events"]
                            if event["control"] == name
                        )
                        for name in {event["control"] for event in state["events"]}
                    },
                )
            )
        self.assertEqual(shallow[0], shallow[1])
        self.assertEqual(shallow[1], shallow[2])

    def test_streak_uses_same_on_time_and_adjacent_counts(self) -> None:
        states = curriculum._streak_states(
            random.Random(11), "garden project TEST", "en"
        )
        self.assertEqual(
            [curriculum.oracle("timely_streak", state) for state in states], [0, 1, 2]
        )
        for state in states:
            by_day = {event["day"]: event["on_time"] for event in state["days"]}
            self.assertEqual(sum(by_day[day] for day in range(1, 13)), 6)
            self.assertEqual(
                sum(by_day[day] and by_day[day + 1] for day in range(1, 12)), 3
            )
            self.assertFalse(by_day[1])
            self.assertFalse(by_day[12])
        self.assertEqual(
            len({curriculum._streak_shortcut_features(state) for state in states}), 1
        )

    def test_weighted_pairings_never_offer_a_full_single_signal_rank(self) -> None:
        states = curriculum._weighted_states(
            random.Random(13), "science club TEST", "en", collections.Counter()
        )
        self.assertEqual(
            [curriculum.oracle("weighted_points", state) for state in states], [0, 1, 2]
        )
        self.assertTrue(
            all(
                len({state["signals"][position]["mark"] for state in states}) <= 2
                for position in range(5)
            )
        )

    def test_frozen_heldout_gate_blocks_this_version(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"Weighted single-signal held-out classifier exceeds 40%: best=112/243",
        ):
            curriculum.generate()


if __name__ == "__main__":
    unittest.main()
