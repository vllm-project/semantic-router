"""Detection, cache loss, and outcome intervals for a restart switch."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.training.model_selection.restart_switch.restart_switch import (
    BLOCK,
    DEFAULT_MARKERS,
    direction_report,
    error_rates,
    iter_traces,
    load_outcomes,
    main,
    recommendation,
    request_has_marker,
    session_rows,
    summarise,
    verify,
    wilson_interval,
)

# Eleven labeled turns: three real restarts (marked compaction, cleared
# conversation, short clear) and eight non-restarts, including a rewind and
# an edit that the next turn continues.
LABELED_KINDS = [
    "first",
    "ordinary",
    "continued_restart",
    "ordinary",
    "continued_restart",
    "ordinary",
    "not_continued",
    "continued_restart",
    "ordinary",
    "short_continued",
    "ordinary",
]
REAL_RESTARTS = [
    False,
    False,
    True,
    False,
    True,
    False,
    False,
    False,
    False,
    True,
    False,
]


def _span(*pairs):
    blocks = []
    for start, stop in pairs:
        blocks.extend(range(start, stop))
    return blocks


def _call(t, blocks, model="m", items=None, compaction=None):
    request = {"t": t, "hash_ids": list(blocks), "model": model}
    if items is not None:
        request["items"] = items
    if compaction is not None:
        request["compaction"] = compaction
    return request


def _labeled_trace():
    compaction = _span((0, 40), (10000, 10040))
    cleared = _span((0, 40), (20000, 20040))
    edited = [*_span((0, 50)), 88888, *_span((51, 260))]
    short_clear = _span((0, 10), (40000, 40010))
    return {
        "requests": [
            _call(0, _span((0, 200))),
            _call(1, _span((0, 260))),
            _call(2, compaction, items=[{"type": "context_compaction"}]),
            _call(3, [*compaction, 10040, 10041]),
            _call(4, cleared),
            _call(5, [*cleared, 20040]),
            _call(6, _span((0, 260))),
            _call(7, edited),
            _call(8, [*edited, 261, 262]),
            _call(9, short_clear),
            _call(10, [*short_clear, 40010]),
        ]
    }


def _outcome(stay_ok, switch_ok, stay_cost=1.0, switch_cost=1.0):
    return {
        "stay_ok": stay_ok,
        "switch_ok": switch_ok,
        "stay_cost": stay_cost,
        "switch_cost": switch_cost,
    }


class MarkerAndFixtureTest(unittest.TestCase):
    def test_marker_is_explicit_and_not_inferred_from_block_ids(self):
        self.assertFalse(request_has_marker({"hash_ids": [1, 2, 3]}, DEFAULT_MARKERS))
        self.assertTrue(
            request_has_marker(
                {"items": [{"type": "context_compaction"}]},
                DEFAULT_MARKERS,
            )
        )
        self.assertTrue(request_has_marker({"compaction": True}, DEFAULT_MARKERS))
        self.assertFalse(request_has_marker({"compaction": False}, DEFAULT_MARKERS))

    def test_labeled_fixture_kinds_and_error_rates(self):
        rows, marker_turns = session_rows(_labeled_trace(), DEFAULT_MARKERS)
        self.assertEqual([row["kind"] for row in rows], LABELED_KINDS)
        self.assertEqual(marker_turns, 1)
        marker = error_rates([row["marker"] for row in rows], REAL_RESTARTS)
        prefix = error_rates([row["prefix_break"] for row in rows], REAL_RESTARTS)
        continued = error_rates(
            [row["continued_restart"] for row in rows],
            REAL_RESTARTS,
        )
        self.assertEqual(
            (marker["tp"], marker["fp"], marker["tn"], marker["fn"]), (1, 0, 8, 2)
        )
        self.assertEqual(marker["false_positive_rate"], 0)
        self.assertAlmostEqual(marker["false_negative_rate"], 2 / 3)
        self.assertEqual(
            (prefix["tp"], prefix["fp"], prefix["tn"], prefix["fn"]), (3, 2, 6, 0)
        )
        self.assertAlmostEqual(prefix["false_positive_rate"], 2 / 8)
        self.assertEqual(prefix["false_negative_rate"], 0)
        self.assertEqual(
            (continued["tp"], continued["fp"], continued["tn"], continued["fn"]),
            (2, 1, 7, 1),
        )
        self.assertAlmostEqual(continued["false_positive_rate"], 1 / 8)
        self.assertAlmostEqual(continued["false_negative_rate"], 1 / 3)

    def test_subagent_marker_and_blocks_stay_off_the_main_prefix(self):
        head = list(range(70))
        extended = head + list(range(1000, 1070))
        restart = list(range(40)) + list(range(2000, 2040))
        trace = {
            "requests": [
                _call(0, head),
                {
                    "type": "subagent",
                    "agent_id": "child",
                    "requests": [
                        _call(1, [9], items=[{"type": "context_compaction"}]),
                    ],
                },
                _call(2, extended),
                _call(3, restart),
                _call(4, [*restart, 2040]),
            ]
        }
        report = summarise([trace])
        # head, the extended turn, the restart, and the turn that continues it.
        # The subagent call between the first two is not a main turn.
        self.assertEqual(report["main_turns"], 4)
        self.assertEqual(report["marker_turns"], 0)
        self.assertEqual(report["ordinary_turns"], 2)
        self.assertEqual(report["continued_restarts"], 1)
        self.assertEqual(report["runs"], 2)
        self.assertEqual(report["restarts_per_session"], 1)
        self.assertEqual(report["restarts_per_run"], 0.5)
        self.assertEqual(report["stay_switch_loss_tokens"], 0)
        self.assertEqual(
            report["ordinary_switch_loss"]["median_cached_tokens"],
            75 * BLOCK,
        )
        self.assertEqual(
            report["restart_switch_loss"]["median_cached_tokens"],
            40 * BLOCK,
        )
        self.assertEqual(
            report["restart_switch_loss"]["total_cached_tokens"],
            40 * BLOCK,
        )
        self.assertEqual(
            report["ordinary_switch_loss"]["total_cached_tokens"],
            150 * BLOCK,
        )
        self.assertEqual(report["half_drop"]["turns"], 0)
        self.assertEqual(report["recommendation"], "no-go")
        self.assertEqual(report["outcomes"]["stronger"]["n"], 0)
        self.assertIsNone(report["outcomes"]["cheaper"]["stay_success"])

    def test_a_different_block_size_is_refused(self):
        with self.assertRaises(ValueError):
            session_rows(
                {"block_size": 32, "requests": [_call(0, [1])]},
                DEFAULT_MARKERS,
            )


class HalfDropCrossCheckTest(unittest.TestCase):
    def test_short_side_call_does_not_reset_conversation_length(self):
        restart = list(range(50)) + list(range(1000, 1070))
        calls = [
            _call(0, [0]),
            _call(1, range(200)),
            _call(2, range(300)),
            _call(3, [5000, 5001]),
            _call(4, range(4000, 4100)),
            _call(5, restart),
            _call(6, restart + list(range(1070, 1080))),
            _call(7, range(2000, 2100), model="x"),
        ]
        report = summarise([{"requests": calls}])
        self.assertEqual(report["half_drop"]["turns"], 1)
        self.assertEqual(report["half_drop"]["sessions"], 1)
        self.assertEqual(report["half_drop"]["median_cached_tokens"], 50 * BLOCK)
        self.assertAlmostEqual(report["half_drop"]["uncached_token_share"], 70 / 581)


class OutcomeTest(unittest.TestCase):
    def test_wilson_interval_for_eight_of_ten(self):
        interval = wilson_interval(8, 10)
        self.assertAlmostEqual(interval["rate"], 0.8)
        self.assertAlmostEqual(interval["low"], 0.49015684672072335)
        self.assertAlmostEqual(interval["high"], 0.9433190520193067)
        self.assertIsNone(wilson_interval(0, 0))

    def test_cost_interval_and_recommendation_gate(self):
        priced = direction_report(
            [
                _outcome(1, 1, 1, 2),
                _outcome(1, 1, 1, 4),
            ]
        )
        self.assertAlmostEqual(priced["cost_effect"]["mean"], 2)
        self.assertAlmostEqual(priced["cost_effect"]["low"], 0.04)
        self.assertAlmostEqual(priced["cost_effect"]["high"], 3.96)
        kept = direction_report([_outcome(1, 1), _outcome(1, 1)])
        dropped = direction_report([_outcome(1, 1), _outcome(1, 0)])
        one = direction_report([_outcome(1, 0, 1, 2)])
        self.assertIsNone(one["success_effect"]["low"])
        self.assertEqual(
            recommendation({"stronger": kept, "cheaper": kept}),
            "go",
        )
        self.assertEqual(
            recommendation({"stronger": dropped, "cheaper": kept}),
            "no-go",
        )
        self.assertEqual(
            recommendation({"stronger": kept, "cheaper": direction_report([])}),
            "no-go",
        )
        self.assertEqual(recommendation({"stronger": one, "cheaper": one}), "no-go")

    def test_outcome_bit_names_the_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(
                '{"stay_ok": 2, "switch_ok": 1, "stay_cost": 1, "switch_cost": 1}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as caught:
                load_outcomes(path)
            self.assertIn("rows.jsonl:1: stay_ok", str(caught.exception))


class CliTest(unittest.TestCase):
    def test_report_cli_is_no_go_without_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.jsonl"
            path.write_text(
                json.dumps({"requests": [_call(0, [1, 2])]}) + "\n", encoding="utf-8"
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["report", str(path), "--json"]), 0)
            report = json.loads(stdout.getvalue())
        self.assertEqual(report["recommendation"], "no-go")
        self.assertEqual(report["main_turns"], 1)

    def test_invalid_trace_json_names_the_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.jsonl"
            path.write_text("{}\nnot json\n", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                list(iter_traces(path))
            self.assertIn("traces.jsonl:2:", str(caught.exception))

    def test_verify_rejects_a_wrong_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
