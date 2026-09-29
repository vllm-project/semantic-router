from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path

from v2.eval.htdev.common import SALT, cut_at_sentence, level
from v2.eval.htdev.tests.fixtures import WRITERS
from v2.eval.sealed.schema import display_order, validate

try:
    import pyarrow  # noqa: F401

    HAVE_ARROW = True
except ImportError:
    HAVE_ARROW = False

PARQUET = {"nycc", "circa", "pubhealth", "brighter", "mhs", "casino"}
LEAKS = (
    "http",
    "wikipedia.org",
    "label",
    "annotator",
    "explanation",
    "WRONG",
    "RIGHT",
    "AUTHOR",
    "judgements",
    "hatespeech",
    "hate_speech_score",
    "satisfaction",
    "points_scored",
    "Snopes",
    "Some Checker",
    "French politics",
)


def convert(key: str, flip: bool):
    module = importlib.import_module(f"v2.eval.htdev.sources.{key}")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / key
        WRITERS[key](root, flip)
        return module, list(module.candidates(root))


class ConverterTest(unittest.TestCase):
    def keys(self):
        for key in sorted(WRITERS):
            if key in PARQUET and not HAVE_ARROW:
                continue
            yield key

    def test_every_candidate_is_valid_and_carries_provenance(self):
        for key in self.keys():
            with self.subTest(key=key):
                module, items = convert(key, False)
                self.assertTrue(items, key)
                for item in items:
                    self.assertEqual(validate(item, module.SPEC), [])
                    self.assertTrue(item.split)
                    self.assertIn("file", item.provenance)
                    self.assertIn("row", item.provenance)
                    self.assertEqual(item.task, module.TASK)

    def test_prompt_is_identical_whatever_the_gold(self):
        for key in self.keys():
            with self.subTest(key=key):
                _, plain = convert(key, False)
                _, flipped = convert(key, True)
                self.assertEqual(
                    [(c.source_item_id, c.state, c.question) for c in plain],
                    [(c.source_item_id, c.state, c.question) for c in flipped],
                )
                self.assertNotEqual(
                    [c.gold for c in plain], [c.gold for c in flipped], key
                )

    def test_template_is_fixed_per_task(self):
        for key in self.keys():
            with self.subTest(key=key):
                _, items = convert(key, False)
                instructions = {c.question["instructions"] for c in items}
                self.assertEqual(len(instructions), 1)
                option_sets = {
                    json.dumps(c.question["criteria"], sort_keys=True) for c in items
                }
                self.assertEqual(len(option_sets), 1)

    def test_states_hold_no_source_fields(self):
        for key in self.keys():
            with self.subTest(key=key):
                _, items = convert(key, False)
                for item in items:
                    text = json.dumps(item.state)
                    for leak in LEAKS:
                        self.assertNotIn(leak, text)

    def test_choice_display_order_is_the_salted_item_hash(self):
        for key in self.keys():
            with self.subTest(key=key):
                _, first = convert(key, False)
                _, second = convert(key, False)
                self.assertEqual(
                    [list(c.question["criteria"]) for c in first],
                    [list(c.question["criteria"]) for c in second],
                )
                for item in first:
                    if item.question["type"] != "choice" or item.option_texts:
                        continue
                    module = importlib.import_module(f"v2.eval.htdev.sources.{key}")
                    canonical = [k for k, _ in module.OPTIONS]
                    order = display_order(item.source_item_id, len(canonical), SALT)
                    self.assertEqual(
                        list(item.question["criteria"]), [canonical[i] for i in order]
                    )


class TemplateDetailTest(unittest.TestCase):
    def test_nycc_shows_entity_titles_and_captions_in_display_order(self):
        _, items = convert("nycc", False) if HAVE_ARROW else (None, [])
        if not items:
            self.skipTest("pyarrow unavailable")
        item = items[0]
        self.assertEqual(item.state["entities"], ["Office chair"])
        self.assertEqual(
            [item.state["caption_a"], item.state["caption_b"]], item.option_texts
        )
        self.assertEqual(len(items), 10)
        self.assertEqual(item.question["criteria"].keys(), {"A", "B"})

    def test_circa_drops_other(self):
        if not HAVE_ARROW:
            self.skipTest("pyarrow unavailable")
        _, items = convert("circa", False)
        self.assertEqual(
            [c.gold for c in items], ["yes", "no", "in_the_middle", "yes_conditional"]
        )

    def test_pubhealth_drops_unproven_and_shows_only_the_claim(self):
        if not HAVE_ARROW:
            self.skipTest("pyarrow unavailable")
        _, items = convert("pubhealth", False)
        self.assertEqual(len(items), 9)
        self.assertEqual({tuple(c.state) for c in items}, {("claim",)})
        self.assertEqual([c.split for c in items[:3]], ["test"] * 3)

    def test_hyperpartisan_body_is_cut_at_a_sentence_end(self):
        _, items = convert("hyperpartisan", False)
        long_item = items[0]
        self.assertLessEqual(len(long_item.state["body"]), 2000)
        self.assertTrue(long_item.state["body"].endswith("."))
        self.assertEqual(long_item.state["title"], "Title 0 & more")
        self.assertEqual(long_item.group_id, "news0.example")
        self.assertEqual(items[0].split, "test")

    def test_diplomacy_context_and_gold(self):
        _, items = convert("diplomacy", False)
        self.assertEqual(len(items), 24)
        self.assertLessEqual(max(len(c.state["previous_messages"]) for c in items), 3)
        self.assertEqual(items[0].gold, True)
        self.assertEqual(items[1].gold, False)
        self.assertEqual(items[0].split, "test")
        self.assertEqual(items[0].group_id, "10:germany-italy")
        self.assertEqual(items[0].cluster_id, "10")

    def test_scruples_story_is_capped(self):
        _, items = convert("scruples", False)
        self.assertEqual(len(items[0].state["story"]), 2000)
        self.assertEqual(set(items[0].state), {"title", "story"})

    def test_score_levels(self):
        self.assertEqual(
            [level(v, (-1.8, -0.6, 0.6, 1.8)) for v in (-2.5, -1.0, 0.0, 1.0, 2.5)],
            [0, 1, 2, 3, 4],
        )
        _, items = convert("pavlick", False)
        self.assertEqual([c.gold for c in items[:5]], [0, 1, 2, 3, 4])
        self.assertEqual(len(items), 10)
        _, items = convert("empathic_reactions", False)
        self.assertEqual([c.gold for c in items], [0, 1, 2, 3, 4])

    def test_mhs_one_item_per_comment_and_card_thresholds(self):
        if not HAVE_ARROW:
            self.skipTest("pyarrow unavailable")
        _, items = convert("mhs", False)
        self.assertEqual([c.gold for c in items], [0, 1, 1, 2])

    def test_majority_rules(self):
        _, items = convert("mfrc", False)
        self.assertEqual([c.gold for c in items], [True, False])
        _, items = convert("hatexplain", False)
        self.assertEqual([c.gold for c in items], ["normal", "hate_speech"])
        _, items = convert("swords", False)
        self.assertEqual([c.gold for c in items], [True, False, True, False])

    def test_moral_stories_splits_follow_the_classification_files(self):
        _, items = convert("moral_stories", False)
        self.assertEqual([c.split for c in items], ["test", "validation", "train"])

    def test_cut_at_sentence(self):
        self.assertEqual(cut_at_sentence("Short."), "Short.")
        text = "One two. " * 400
        cut = cut_at_sentence(text, 100)
        self.assertLessEqual(len(cut), 100)
        self.assertTrue(cut.endswith("."))


if __name__ == "__main__":
    unittest.main()
