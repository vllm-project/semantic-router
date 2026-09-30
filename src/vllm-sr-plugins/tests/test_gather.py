from __future__ import annotations

import unittest

from support import PLUGIN_ROOT  # noqa: F401  (import path)

from vllm_sr_plugins.decision2.gather import (
    POSITIONS_KEY,
    Positions,
    parse_positions,
    plan_step,
)


def kwargs(candidates, query):
    return {POSITIONS_KEY: {"candidate_positions": candidates, "query_position": query}}


class ParsePositionsTest(unittest.TestCase):
    def test_accepts_encoder_output(self) -> None:
        self.assertEqual(parse_positions(kwargs([3, 6], 9), 10), Positions((3, 6), 9))

    def test_rejects_malformed_positions(self) -> None:
        for extra, prompt_len in [
            (None, 10),
            ({}, 10),
            (kwargs([3], 9), 10),  # one candidate
            (kwargs([6, 3], 9), 10),  # not increasing
            (kwargs([3, 3], 9), 10),  # duplicate
            (kwargs([3, 9], 9), 10),  # candidate at the query
            (kwargs([-1, 3], 9), 10),
            (kwargs([3, 6], 8), 10),  # query is not the last prompt token
            (kwargs([3, 6.0], 9), 10),
            (kwargs([True, 6], 9), 10),
            (kwargs(list(range(256)), 300), 301),
        ]:
            self.assertIsNone(parse_positions(extra, prompt_len), (extra, prompt_len))


class PlanStepTest(unittest.TestCase):
    def test_whole_prompts_in_one_step(self) -> None:
        plan = plan_step(
            [10, 12],
            [10, 12],
            [10, 12],
            [Positions((3, 6), 9), Positions((2, 5, 8), 11)],
        )
        self.assertEqual(plan.rows, [3, 6, 9, 10 + 2, 10 + 5, 10 + 8, 10 + 11])
        self.assertEqual(plan.counts, [3, 4])
        self.assertEqual(plan.finished, [True, True])

    def test_chunked_prefill_collects_every_row_once(self) -> None:
        positions = Positions((2, 5, 8), 11)
        collected = []
        for scheduled, seq_len in [(4, 4), (4, 8), (4, 12)]:
            plan = plan_step([scheduled], [seq_len], [12], [positions])
            start = seq_len - scheduled
            collected += [start + row for row in plan.rows]
            self.assertEqual(plan.finished, [seq_len == 12])
        self.assertEqual(collected, list(positions.rows))

    def test_mixed_step_offsets_rows_per_request(self) -> None:
        # Request 0 finishes its last 5 tokens; request 1 starts a 7-token chunk.
        plan = plan_step(
            [5, 7],
            [12, 7],
            [12, 20],
            [Positions((2, 8), 11), Positions((1, 6, 15), 19)],
        )
        self.assertEqual(plan.rows, [8 - 7, 11 - 7, 5 + 1, 5 + 6])
        self.assertEqual(plan.counts, [2, 2])
        self.assertEqual(plan.finished, [True, False])

    def test_missing_positions_take_no_rows(self) -> None:
        plan = plan_step([4], [4], [4], [None])
        self.assertEqual((plan.rows, plan.counts, plan.finished), ([], [0], [True]))


if __name__ == "__main__":
    unittest.main()
