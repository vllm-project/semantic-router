from __future__ import annotations

import random
import unittest
from collections import Counter

from d25.omni.train.schedule import (
    build_schedule,
    learning_rate_factor,
    padding_efficiency,
    plan_step,
)


class ScheduleTest(unittest.TestCase):
    def test_replay_ratio_and_single_pass_over_main(self) -> None:
        main, replay = list(range(100)), list(range(100, 130))
        schedule = build_schedule(
            main, replay, batch_size=16, replay_ratio=0.5, epochs=1, seed=3
        )
        seen = Counter(i for step in schedule.steps for i in step if i < 100)
        self.assertEqual(seen, Counter(main))
        for step in schedule.steps[:-1]:
            self.assertEqual(sum(1 for i in step if i >= 100), 8)
            self.assertEqual(len(step), 16)
        self.assertGreaterEqual(schedule.replay_passes, 3)
        again = build_schedule(
            main, replay, batch_size=16, replay_ratio=0.5, epochs=1, seed=3
        )
        self.assertEqual(schedule.digest(), again.digest())
        other = build_schedule(
            main, replay, batch_size=16, replay_ratio=0.5, epochs=1, seed=4
        )
        self.assertNotEqual(schedule.digest(), other.digest())

    def test_ratio_70_percent_and_epochs(self) -> None:
        schedule = build_schedule(
            list(range(60)),
            list(range(60, 1000)),
            batch_size=20,
            replay_ratio=0.7,
            epochs=2,
            seed=0,
        )
        self.assertEqual(
            Counter(i for s in schedule.steps for i in s if i < 60),
            Counter({i: 2 for i in range(60)}),
        )
        self.assertEqual(sum(1 for i in schedule.steps[0] if i >= 60), 14)

    def test_extreme_ratios(self) -> None:
        only_main = build_schedule(list(range(10)), [], batch_size=4, replay_ratio=0.0)
        self.assertEqual(sorted(i for s in only_main.steps for i in s), list(range(10)))
        only_replay = build_schedule(
            [], list(range(10, 20)), batch_size=4, replay_ratio=1.0
        )
        self.assertEqual(
            sorted(i for s in only_replay.steps for i in s), list(range(10, 20))
        )
        with self.assertRaises(ValueError):
            build_schedule(list(range(4)), [], batch_size=4, replay_ratio=0.5)

    def test_plan_step_covers_rows_within_budget(self) -> None:
        rng = random.Random(0)
        tokens = [rng.choice([200, 400, 1600, 6000, 7000]) for _ in range(64)]
        group = list(range(64))
        plans = plan_step(group, tokens, world=8, token_budget=8192, max_rows=8)
        flat = [i for rank in plans for batch in rank for i in batch]
        self.assertEqual(sorted(flat), group)
        self.assertEqual(len({len(rank) for rank in plans}), 1)
        for rank in plans:
            sizes = [len(batch) for batch in rank]
            self.assertEqual(sizes, sorted(sizes, key=lambda n: n == 0))
            for batch in rank:
                if len(batch) > 1:
                    self.assertLessEqual(
                        len(batch) * max(tokens[i] for i in batch), 8192
                    )
                self.assertLessEqual(len(batch), 8)
        loads = [sum(tokens[i] for batch in rank for i in batch) for rank in plans]
        self.assertLess(max(loads) - min(loads), 7000 + 1)
        self.assertGreater(padding_efficiency(plans, tokens), 0.5)

    def test_single_rank_and_oversized_row(self) -> None:
        plans = plan_step(
            [0, 1, 2], [10_000, 10, 10], world=1, token_budget=1000, max_rows=4
        )
        self.assertEqual(plans, [[[0], [1, 2]]])
        padded = plan_step([0], [5], world=3, token_budget=100, max_rows=4)
        self.assertEqual(sorted(len(rank) for rank in padded), [1, 1, 1])
        self.assertEqual(sum(1 for rank in padded for batch in rank if not batch), 2)

    def test_learning_rate_factor(self) -> None:
        self.assertAlmostEqual(learning_rate_factor(5, 100, 0.1), 0.5)
        self.assertAlmostEqual(learning_rate_factor(10, 100, 0.1), 1.0)
        self.assertAlmostEqual(learning_rate_factor(100, 100, 0.1), 0.1)
        self.assertGreater(learning_rate_factor(50, 100, 0.1), 0.1)


if __name__ == "__main__":
    unittest.main()
