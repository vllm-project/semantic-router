from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.eval.sealed import candidates as runner
from v2.eval.sealed.schema import MAX_INPUT_CHARS, input_chars, validate
from v2.eval.sealed.sources import narrative_gold as source

try:
    import pyarrow
    import pyarrow.parquet
except ImportError:
    pyarrow = None

CONCRETENESS = "narrative_gold/setting_concreteness"
TEMPORAL = "narrative_gold/setting_temporal_grounding"
CAUSALITY = "narrative_gold/event_causality"
SPAN = "narrative_gold/span_is_event"
NAN = float("nan")


def setting(item, text, concreteness=None, temporal=None):
    return {
        "safe_instance_id": item,
        "sampled_text": text,
        "setting_concreteness_gold": concreteness,
        "setting_temporal_grounding_gold": temporal,
        "setting_concreteness_annotator_2": 1.0,
        "setting_temporal_grounding_annotator_2": 5.0,
    }


def span(text, phrase, start=None):
    start = text.index(phrase) if start is None else start
    return json.dumps([start, start + len(phrase), phrase, "event"])


def event(
    item,
    text,
    first,
    second,
    causality="direct_cause",
    first_event=True,
    second_event=True,
):
    return {
        "safe_instance_id": item,
        "sampled_text": text,
        "assigned_span1": first,
        "assigned_span2": second,
        "span1_is_event_gold": first_event,
        "span2_is_event_gold": second_event,
        "causality_rating_gold": causality,
        "span1_is_event_annotator_2": False,
        "span2_is_event_annotator_2": False,
        "causality_rating_annotator_2": "not_related",
    }


HARBOUR = "A lantern swung above the wet cobbles of the harbour at dawn."
SETTING_ROWS = [
    setting("st_05", "Some say ideas matter more than places.", 1.0, 1.0),
    setting("st_01", HARBOUR, 5, 2),
    setting("st_02", "On Tuesday the ferry left at noon.", 4, None),
    setting("st_03", "It was raining in the valley.", NAN, 3),
    setting("st_04", "The garden wall was painted green.", 3.5, 0),
    setting("st_06", "Nobody knew the time.", True, 6),
    setting("st_07", "Bells rang across the square.", "4", -1),
    setting("dup_1", "The market closed early on Friday.", 2, 2),
    setting("dup_2", "  the MARKET closed   early on friday. ", 5, 5),
    setting("dup_3", "A cat slept on the warm stove.", 3, 3),
    setting("dup_4", "a cat slept on the warm stove.", None, None),
    setting("st_08", "   ", 3, 3),
    setting("st_09", None, 3, 3),
]
FLOOD = "When the dam broke, the valley flooded within hours."
DOOR = "The door creaked open and the dog barked."
EVENT_ROWS = [
    event("ev_01", FLOOD, span(FLOOD, "dam broke"), span(FLOOD, "flooded")),
    event("ev_02", FLOOD, span(FLOOD, "flooded"), span(FLOOD, "dam broke"), "enables"),
    event("ev_03", DOOR, span(DOOR, "creaked"), span(DOOR, " open"), "not_related"),
    event("ev_04", FLOOD, span(FLOOD, "dam broke"), span(FLOOD, "broke, the")),
    event(
        "ev_05", DOOR, span(DOOR, "creaked"), span(DOOR, "barked"), first_event=False
    ),
    event("ev_06", DOOR, span(DOOR, "creaked"), span(DOOR, "barked"), None),
    event("ev_07", DOOR, span(DOOR, "creaked"), span(DOOR, "barked"), NAN),
    event("ev_08", DOOR, span(DOOR, "creaked"), span(DOOR, "barked"), "prevents"),
    event("ev_09", DOOR, span(DOOR, "creaked", start=3), span(DOOR, "barked")),
    event("ev_10", DOOR, "not json", span(DOOR, "barked"), second_event=None),
    event("ev_11", DOOR, json.dumps([9, 9, "x"]), json.dumps([4, 8, ""])),
    event("ev_12", DOOR, [4, 8, "door"], json.dumps([9.0, 16.0, "creaked"])),
    event(
        "ev_13",
        DOOR,
        span(DOOR, "creaked"),
        span(DOOR, "barked"),
        first_event="True",
        second_event=1,
    ),
]
EVENT_SKIPPED = {
    "overlapping_spans": "ev_04",
    "non_event_span": "ev_05",
    "missing_rating": "ev_06",
    "nan_rating": "ev_07",
    "unknown_rating": "ev_08",
    "offset_mismatch": "ev_09",
    "unparsable_span": "ev_10",
    "empty_or_reversed_span": "ev_11",
    "float_offsets": "ev_12",
    "non_bool_event_gold": "ev_13",
}
PARQUET_SETTING = [
    {
        "safe_instance_id": "pq_1",
        "sampled_text": HARBOUR,
        "setting_concreteness_gold": 5.0,
        "setting_temporal_grounding_gold": NAN,
        "setting_concreteness_annotator_1": 1.0,
    },
    {
        "safe_instance_id": "pq_2",
        "sampled_text": "On Tuesday the ferry left at noon.",
        "setting_concreteness_gold": None,
        "setting_temporal_grounding_gold": 3.0,
        "setting_concreteness_annotator_1": 2.0,
    },
]
PARQUET_EVENTS = [
    {
        "safe_instance_id": "pq_3",
        "sampled_text": FLOOD,
        "assigned_span1": span(FLOOD, "dam broke"),
        "assigned_span2": span(FLOOD, "flooded"),
        "span1_is_event_gold": True,
        "span2_is_event_gold": True,
        "causality_rating_gold": "direct_cause",
        "causality_rating_annotator_1": "not_related",
    },
    {
        "safe_instance_id": "pq_4",
        "sampled_text": DOOR,
        "assigned_span1": span(DOOR, "creaked"),
        "assigned_span2": span(DOOR, "barked"),
        "span1_is_event_gold": False,
        "span2_is_event_gold": None,
        "causality_rating_gold": None,
        "causality_rating_annotator_1": "enables",
    },
]


class NarrativeGoldTest(unittest.TestCase):
    def setUp(self):
        self.setting = list(source.setting_candidates(SETTING_ROWS))
        self.events = list(source.event_candidates(EVENT_ROWS))
        self.spans = list(source.span_candidates(EVENT_ROWS))

    def test_all_candidates_valid_and_tasks_declared(self):
        out = self.setting + self.events + self.spans
        for c in out:
            self.assertEqual(validate(c, source.SPEC), [])
            self.assertEqual(c.language, "en")
            self.assertIsNone(c.date)
        self.assertEqual({c.task for c in out}, set(source.SPEC.tasks))

    def test_setting_scores_use_valid_gold_levels_only(self):
        self.assertEqual(
            [(c.source_item_id, c.task, c.gold) for c in self.setting],
            [
                ("st_01", CONCRETENESS, 4),
                ("st_01", TEMPORAL, 1),
                ("st_02", CONCRETENESS, 3),
                ("st_03", TEMPORAL, 2),
                ("st_05", CONCRETENESS, 0),
                ("st_05", TEMPORAL, 0),
            ],
        )
        texts = {row["safe_instance_id"]: row["sampled_text"] for row in SETTING_ROWS}
        questions = {CONCRETENESS: source.CONCRETENESS_Q, TEMPORAL: source.TEMPORAL_Q}
        for c in self.setting:
            self.assertIs(type(c.gold), int)
            self.assertEqual(c.balance_label, str(c.gold))
            self.assertIs(c.question, questions[c.task])
            self.assertEqual(len(c.question["criteria"]), 5)
            self.assertEqual(c.state, {"passage": texts[c.source_item_id]})
            self.assertEqual(c.overlap_texts, [texts[c.source_item_id]])
            self.assertEqual(c.group_id, c.source_item_id)

    def test_duplicate_passages_drop_every_copy(self):
        ids = {c.source_item_id for c in self.setting}
        self.assertFalse({"dup_1", "dup_2", "dup_3", "dup_4"} & ids)
        single = [r for r in SETTING_ROWS if r["safe_instance_id"] != "dup_2"]
        ids = {c.source_item_id for c in source.setting_candidates(single)}
        self.assertIn("dup_1", ids)
        self.assertNotIn("dup_3", ids)

    # Known bug: _level rejects NaN but not +/-inf, so int(value) raises
    # OverflowError and the whole conversion aborts instead of skipping the level.
    @unittest.expectedFailure
    def test_infinite_level_is_skipped_like_other_invalid_levels(self):
        rows = [setting("st_inf", "The pier creaked in the snow.", float("inf"), 2)]
        self.assertEqual(
            [(c.task, c.gold) for c in source.setting_candidates(rows)], [(TEMPORAL, 1)]
        )

    def test_setting_oversize_passages_are_dropped_never_truncated(self):
        sizes = [
            input_chars({"passage": ""}, question)
            for question in (source.CONCRETENESS_Q, source.TEMPORAL_Q)
        ]
        fit = "f" * (MAX_INPUT_CHARS - max(sizes))
        over = "o" * (MAX_INPUT_CHARS - min(sizes) + 1)
        rows = [setting("big_fit", fit, 3, 3), setting("big_over", over, 3, 3)]
        out = list(source.setting_candidates(rows))
        self.assertEqual(
            [(c.source_item_id, c.task) for c in out],
            [("big_fit", CONCRETENESS), ("big_fit", TEMPORAL)],
        )
        for c in out:
            self.assertEqual(c.state, {"passage": fit})
            self.assertLessEqual(input_chars(c.state, c.question), MAX_INPUT_CHARS)

    def test_event_causality_marks_both_events_inline(self):
        self.assertEqual(
            [(c.source_item_id, c.gold, c.state) for c in self.events],
            [
                (
                    "ev_01",
                    "direct_cause",
                    {
                        "passage": "When the <event1>dam broke</event1>, the valley"
                        " <event2>flooded</event2> within hours.",
                        "event_1": "dam broke",
                        "event_2": "flooded",
                    },
                ),
                (
                    "ev_02",
                    "enables",
                    {
                        "passage": "When the <event2>dam broke</event2>, the valley"
                        " <event1>flooded</event1> within hours.",
                        "event_1": "flooded",
                        "event_2": "dam broke",
                    },
                ),
                (
                    "ev_03",
                    "not_related",
                    {
                        "passage": "The door <event1>creaked</event1><event2> open"
                        "</event2> and the dog barked.",
                        "event_1": "creaked",
                        "event_2": " open",
                    },
                ),
            ],
        )
        self.assertEqual(
            list(source.CAUSALITY_Q["criteria"]),
            ["direct_cause", "enables", "not_related"],
        )
        texts = {row["safe_instance_id"]: row["sampled_text"] for row in EVENT_ROWS}
        for c in self.events:
            self.assertIs(c.question, source.CAUSALITY_Q)
            self.assertEqual(c.balance_label, c.gold)
            self.assertEqual(c.group_id, c.source_item_id)
            self.assertEqual(c.overlap_texts, [texts[c.source_item_id]])

    def test_event_rows_need_a_rating_two_gold_events_and_disjoint_valid_spans(self):
        ids = {c.source_item_id for c in self.events}
        for reason, item in EVENT_SKIPPED.items():
            with self.subTest(reason):
                self.assertNotIn(item, ids)

    def test_span_is_event_is_one_noul_per_valid_span_with_bool_gold(self):
        self.assertEqual(
            [(c.source_item_id, c.gold) for c in self.spans],
            [
                ("ev_01:span1", True),
                ("ev_01:span2", True),
                ("ev_02:span1", True),
                ("ev_02:span2", True),
                ("ev_03:span1", True),
                ("ev_03:span2", True),
                ("ev_04:span1", True),
                ("ev_04:span2", True),
                ("ev_05:span1", False),
                ("ev_05:span2", True),
                ("ev_06:span1", True),
                ("ev_06:span2", True),
                ("ev_07:span1", True),
                ("ev_07:span2", True),
                ("ev_08:span1", True),
                ("ev_08:span2", True),
                ("ev_09:span2", True),
                ("ev_12:span1", True),
            ],
        )
        self.assertEqual(set(source.EVENT_Q["criteria"]), {"true", "false"})
        rows = {row["safe_instance_id"]: row for row in EVENT_ROWS}
        for c in self.spans:
            item, index = c.source_item_id.split(":")
            text = rows[item]["sampled_text"]
            raw = rows[item][f"assigned_{index}"]
            start, end, phrase = (json.loads(raw) if isinstance(raw, str) else raw)[:3]
            self.assertEqual(
                c.state,
                {
                    "passage": f"{text[:start]}<span>{phrase}</span>{text[end:]}",
                    "span": phrase,
                },
            )
            self.assertEqual(c.state["passage"].count("<span>"), 1)
            self.assertNotIn("<event", c.state["passage"])
            self.assertIs(c.question, source.EVENT_Q)
            self.assertEqual(c.balance_label, str(c.gold))
            self.assertEqual(c.group_id, item)
            self.assertEqual(c.overlap_texts, [text])
        self.assertIn("ev_05:span1", {c.source_item_id for c in self.spans})
        self.assertNotIn("ev_05", {c.source_item_id for c in self.events})

    def test_event_and_span_oversize_passages_are_dropped_never_truncated(self):
        empty = {"passage": "<span>x</span>", "span": "x"}
        room = MAX_INPUT_CHARS - input_chars(empty, source.EVENT_Q)
        fit, over = "x" + "p" * room, "x" + "p" * (room + 1)
        huge = "Rain fell. Roads flooded." + " " * MAX_INPUT_CHARS
        rows = [
            event("big_fit", fit, json.dumps([0, 1, "x"]), None, second_event=None),
            event("big_over", over, json.dumps([0, 1, "x"]), None, second_event=None),
            event("big_huge", huge, span(huge, "Rain fell"), span(huge, "flooded")),
        ]
        self.assertEqual(list(source.event_candidates(rows)), [])
        (item,) = source.span_candidates(rows)
        self.assertEqual(item.source_item_id, "big_fit:span1")
        self.assertEqual(item.state["passage"], "<span>x</span>" + "p" * room)
        self.assertEqual(input_chars(item.state, item.question), MAX_INPUT_CHARS)

    @unittest.skipUnless(pyarrow is not None, "pyarrow not installed")
    def test_candidates_reads_both_parquet_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, rows in (
                ("setting_annotations", PARQUET_SETTING),
                ("event_relation_annotations", PARQUET_EVENTS),
            ):
                table = pyarrow.Table.from_pylist(rows)
                pyarrow.parquet.write_table(table, root / f"{name}.parquet")
            out = list(source.candidates(root))
            receipt = runner.run("narrative_gold", root, None)
        self.assertEqual(
            [(c.source_item_id, c.task, c.gold) for c in out],
            [
                ("pq_1", CONCRETENESS, 4),
                ("pq_2", TEMPORAL, 2),
                ("pq_3", CAUSALITY, "direct_cause"),
                ("pq_3:span1", SPAN, True),
                ("pq_3:span2", SPAN, True),
                ("pq_4:span1", SPAN, False),
            ],
        )
        expected = [
            *source.setting_candidates(PARQUET_SETTING),
            *source.event_candidates(PARQUET_EVENTS),
            *source.span_candidates(PARQUET_EVENTS),
        ]
        self.assertEqual([c.to_json() for c in out], [c.to_json() for c in expected])
        self.assertEqual(receipt["invalid"], {})
        self.assertEqual(receipt["valid"], len(out))


if __name__ == "__main__":
    unittest.main()
