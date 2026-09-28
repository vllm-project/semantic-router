from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.eval.sealed import candidates as runner
from v2.eval.sealed.schema import validate
from v2.eval.sealed.sources import implicaturex as source

CANARY = (
    "# The canary GUID for this toy dataset is 00000000-0000-0000-0000-000000000000\n"
)
HEADER = "id,implicature_type,context,utterance,implicature,cancellation\n"
ITEMS = (
    'toy_01,scalar,,"A: some of the cakes were eaten.",'
    '"Not all of the cakes were eaten.","A: actually I think every one went."\n'
    'toy_02,synthetic_conversational,"S1: Are you coming tonight?",'
    '"S2: I have an exam tomorrow.","S2 is not coming tonight.",'
    '"But I could still stop by for an hour."\n'
    "toy_03,discourse,Some context.,An utterance.,An interpretation.,A cancellation.\n"
)
RATINGS = (
    "item_id,contains_cancellation,workerid,likelihood\n"
    "toy_01,False,w1,6\ntoy_01,False,w2,7\n"
    "toy_01,True,w3,1\ntoy_01,True,w4,2\ntoy_01,True,w5,2\n"
    "toy_02,False,w1,4\ntoy_02,False,w2,5\n"
    "toy_02,True,w3,3\ntoy_02,True,w4,x\n"
    "bot_01,False,w1,7\n"
)


class ImplicatureXTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "snap"
        self.root.mkdir()
        (self.root / "implicatureX.csv").write_text(CANARY + HEADER + ITEMS)
        (self.root / "prolific_responses.csv").write_text(CANARY + RATINGS)
        (self.root / "implicatureBot.csv").write_text(
            CANARY + HEADER + "bot_01,scalar,,Bot utterance.,Bot reading.,Bot cancel.\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_each_rated_condition_becomes_a_score_item(self):
        items = list(source.candidates(self.root))
        self.assertEqual(
            [c.source_item_id for c in items],
            [
                "toy_01:without_cancellation",
                "toy_01:with_cancellation",
                "toy_02:without_cancellation",
                "toy_02:with_cancellation",
            ],
        )
        self.assertEqual([c.gold for c in items], [6, 1, 4, 2])
        self.assertEqual([c.balance_label for c in items], ["6", "1", "4", "2"])
        self.assertEqual([c.group_id for c in items], ["toy_01"] * 2 + ["toy_02"] * 2)
        for item in items:
            self.assertEqual(validate(item, source.SPEC), [])
            self.assertEqual(item.question, source.QUESTION)
        self.assertEqual(len(source.QUESTION["criteria"]), 7)

    def test_cancellation_is_appended_to_the_utterance(self):
        plain, cancelled, with_context, continued = source.candidates(self.root)
        self.assertEqual(
            plain.state,
            {
                "utterance": "A: some of the cakes were eaten.",
                "interpretation": "Not all of the cakes were eaten.",
            },
        )
        self.assertEqual(
            cancelled.state["utterance"],
            "A: some of the cakes were eaten.\nA: actually I think every one went.",
        )
        self.assertEqual(with_context.state["context"], "S1: Are you coming tonight?")
        self.assertEqual(
            continued.state["utterance"],
            "S2: I have an exam tomorrow. But I could still stop by for an hour.",
        )
        self.assertNotIn("A: actually I think every one went.", plain.overlap_texts)
        self.assertIn("A: actually I think every one went.", cancelled.overlap_texts)
        for item in (plain, cancelled, with_context, continued):
            text = json.dumps(item.state)
            for leak in ("likelihood", "cancel", "gold", "label", "scalar", "Bot"):
                self.assertNotIn(leak, text)

    def test_mean_is_rounded_half_up(self):
        self.assertEqual(source.rounded_point([4, 5]), 5)
        self.assertEqual(source.rounded_point([1, 2]), 2)
        self.assertEqual(source.rounded_point([2, 2, 3]), 2)
        self.assertEqual(source.rounded_point([3, 3, 4, 4, 4]), 4)
        self.assertEqual(source.rounded_point([7, 7, 7]), 7)

    def test_runner_accepts_the_snapshot(self):
        receipt = runner.run("implicaturex", self.root, Path(self.tmp.name) / "c.jsonl")
        self.assertEqual(receipt["invalid"], {})
        self.assertEqual(receipt["valid"], 4)
        self.assertEqual(receipt["tasks"]["implicaturex/likelihood"]["groups"], 2)


if __name__ == "__main__":
    unittest.main()
