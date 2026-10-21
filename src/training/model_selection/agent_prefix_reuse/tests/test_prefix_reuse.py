import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prefix_reuse


def call(t, blocks, tokens_in=0, tokens_out=0, model="m", api_time=0.0):
    return {
        "t": t,
        "hash_ids": blocks,
        "in": tokens_in,
        "out": tokens_out,
        "model": model,
        "api_time": api_time,
    }


def subagent(agent_id, *calls):
    return {
        "type": "subagent",
        "agent_id": agent_id,
        "subagent_type": "Subagent",
        "requests": list(calls),
    }


class PrefixArithmeticTest(unittest.TestCase):
    def test_cached_prefix_stops_at_the_first_unseen_block(self):
        # Block ids are chained, so a block seen after a gap is not a cached prefix.
        self.assertEqual(prefix_reuse.lead([1, 2, 9, 3], {1, 2, 3}), 2)
        self.assertEqual(prefix_reuse.lead([7, 1], {1}), 0)

    def test_main_share_is_token_weighted_and_skips_the_first_turn(self):
        trace = {
            "requests": [
                call(0, [1, 2]),
                call(10, [1, 2, 3, 4]),
                call(20, [1, 2, 3, 4, 5, 6, 7, 8]),
            ]
        }
        report = prefix_reuse.summarise([trace], ttls=())
        self.assertEqual(report["main_turns"], 2)
        self.assertAlmostEqual(report["main_cached_prefix_share"], (2 + 4) / (4 + 8))
        self.assertAlmostEqual(report["main_cached_prefix_median_turn"], 0.5)

    def test_subagent_first_call_is_split_between_parent_and_sibling(self):
        trace = {
            "requests": [
                call(0, [1, 2]),
                subagent("a", call(5, [1, 2, 10, 11])),
                subagent("b", call(6, [1, 2, 10, 12])),
            ]
        }
        report = prefix_reuse.summarise([trace], ttls=())
        self.assertEqual(report["subagent_first_calls"], 2)
        # a shares 1,2 with the parent and nothing with a sibling, b shares 1,2 with the parent and 1,2,10 with a.
        self.assertAlmostEqual(report["subagent_first_shared_with_parent"], (2 + 2) / 8)
        self.assertAlmostEqual(
            report["subagent_first_shared_with_sibling"], (0 + 3) / 8
        )
        self.assertAlmostEqual(report["subagent_first_sibling_ge_90"], 0.0)

    def test_subagent_later_calls_count_only_their_own_stream(self):
        trace = {
            "requests": [
                call(0, [1, 2, 3]),
                subagent("a", call(5, [20, 21]), call(9, [20, 21, 22, 1])),
            ]
        }
        report = prefix_reuse.summarise([trace], ttls=())
        self.assertEqual(report["subagent_later_calls"], 1)
        self.assertAlmostEqual(report["subagent_later_cached_prefix_share"], 2 / 4)

    def test_idle_gap_runs_from_the_end_of_the_same_streams_previous_call(self):
        trace = {
            "requests": [
                call(0, [1], tokens_in=100, api_time=200.0),
                subagent(
                    "a", call(100, [9], tokens_in=10), call(500, [9, 10], tokens_in=10)
                ),
                call(450, [1, 2], tokens_in=200),
            ]
        }
        report = prefix_reuse.summarise([trace], ttls=(300.0,))
        gap = report["idle_gaps"]["300"]
        # The main stream ended at 200, so its call at 450 is 250 s idle and not a gap, although it started
        # 450 s after the previous main call. The subagent's second call is 400 s after its own first call,
        # although the main stream was busy 50 s before, so it is the one gap.
        self.assertEqual(gap["requests"], 1)
        self.assertEqual(gap["median_prompt_tokens"], 2 * prefix_reuse.BLOCK)
        self.assertAlmostEqual(gap["input_token_share"], 10 / 320)

    def test_token_shares_are_by_stream_kind(self):
        trace = {"requests": [call(0, [1], 90, 9), subagent("a", call(1, [5], 10, 1))]}
        report = prefix_reuse.summarise([trace], ttls=())
        self.assertAlmostEqual(report["subagent_input_share"], 0.1)
        self.assertAlmostEqual(report["subagent_output_share"], 0.1)
        self.assertEqual(report["traces_with_subagents"], 1)
        self.assertEqual(report["subagent_groups"], 1)

    def test_report_cli_reads_jsonl_and_verify_rejects_a_wrong_digest(self):
        trace = {"requests": [call(0, [1, 2]), call(10, [1, 2, 3])]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.jsonl"
            path.write_text(json.dumps(trace) + "\nnot json\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(prefix_reuse.main(["report", str(path), "--json"]), 0)
            self.assertEqual(json.loads(out.getvalue())["main_turns"], 1)
            with self.assertRaises(ValueError):
                prefix_reuse.verify(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
