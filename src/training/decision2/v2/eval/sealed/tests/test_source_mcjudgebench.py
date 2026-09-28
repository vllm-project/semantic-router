from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.eval.sealed import candidates as runner
from v2.eval.sealed.schema import validate
from v2.eval.sealed.sources import mcjudgebench as source


def constraint(cid: int, text: str, gold: str) -> dict:
    return {
        "constraint_id": cid,
        "constraint_text": text,
        "constraint_types": ["TagXyz"],
        "gold_label": gold,
    }


ROWS = [
    {
        "instance_id": "mcjb_0001",
        "release_subset": "paper",
        "source_dataset": "infobench",
        "instruction": "Write two lines about snow.",
        "input": "",
        "candidate_response": "Snow falls.\nSnow melts.",
        "constraints": [constraint(1, "Are there two lines?", "yes")],
        "perturbations": [],
    },
    {
        "instance_id": "mcjb_0002",
        "release_subset": "augmentation",
        "source_dataset": "wildifeval",
        "instruction": "Name three fruits as bullets and end with a question.",
        "input": "",
        "candidate_response": "- apple\n- pear\n- plum\n",
        "constraints": [
            constraint(1, "Does the response name three fruits?", "yes"),
            constraint(2, "Does the response use bullets throughout?", "partial"),
            constraint(3, "Does the response end with a question?", "no"),
            constraint(4, "Is the tone friendly?", "unsure"),
        ],
        "perturbations": [
            {
                "perturbation_id": 1,
                "perturbation_type": "local_paraphrase",
                "perturbed_response": "PERTURBED-TEXT",
            }
        ],
    },
    {
        "instance_id": "mcjb_0003",
        "release_subset": "augmentation",
        "source_dataset": "truebench",
        "instruction": "Summarise the note in one sentence.",
        "input": "The meeting moved to Tuesday.",
        "candidate_response": "The meeting is now on Tuesday.",
        "constraints": [constraint(1, "Is the summary one sentence?", "yes")],
        "perturbations": [],
    },
    {
        "instance_id": "mcjb_0004",
        "release_subset": "augmentation",
        "source_dataset": "complexbench",
        "instruction": "Write a long story.",
        "input": "",
        "candidate_response": "word " * 6_000,
        "constraints": [constraint(1, "Is it long?", "yes")],
        "perturbations": [],
    },
]


class McJudgeBenchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "snap"
        (self.root / "data").mkdir(parents=True)
        (self.root / "data" / "test.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_augmentation_constraints_become_score_items(self):
        items = list(source.candidates(self.root))
        self.assertEqual(
            [c.source_item_id for c in items],
            ["mcjb_0002#1", "mcjb_0002#2", "mcjb_0002#3", "mcjb_0003#1"],
        )
        self.assertEqual([c.gold for c in items], [2, 1, 0, 2])
        self.assertEqual([c.balance_label for c in items], ["2", "1", "0", "2"])
        self.assertEqual([c.group_id for c in items], ["mcjb_0002"] * 3 + ["mcjb_0003"])
        for item in items:
            self.assertEqual(validate(item, source.SPEC), [])
            self.assertEqual(item.question, source.QUESTION)
            self.assertIsNone(item.date)
        self.assertEqual(len(source.QUESTION["criteria"]), 3)

    def test_state_holds_only_the_decision_context(self):
        items = list(source.candidates(self.root))
        self.assertEqual(
            list(items[0].state), ["instruction", "response", "requirement"]
        )
        self.assertEqual(
            list(items[3].state), ["instruction", "input", "response", "requirement"]
        )
        self.assertEqual(items[0].state["response"], "- apple\n- pear\n- plum\n")
        for item in items:
            text = json.dumps(item.state)
            for leak in ("gold", "label", "TagXyz", "PERTURBED", "augmentation"):
                self.assertNotIn(leak, text)
            self.assertIn(item.state["requirement"], item.overlap_texts)
            self.assertIn(item.state["response"], item.overlap_texts)

    def test_paper_subset_and_overlong_items_are_dropped(self):
        ids = {c.group_id for c in source.candidates(self.root)}
        self.assertNotIn("mcjb_0001", ids)
        self.assertNotIn("mcjb_0004", ids)

    def test_runner_accepts_the_snapshot_deterministically(self):
        out = Path(self.tmp.name) / "out"
        first = runner.run("mcjudgebench", self.root, out / "a.jsonl")
        second = runner.run("mcjudgebench", self.root, out / "b.jsonl")
        self.assertEqual(first["invalid"], {})
        self.assertEqual(first["valid"], 4)
        task = first["tasks"]["mcjudgebench/constraint"]
        self.assertEqual((task["label:0"], task["label:1"], task["label:2"]), (1, 1, 2))
        self.assertEqual(task["groups"], 2)
        self.assertEqual(first["candidates_sha256"], second["candidates_sha256"])


if __name__ == "__main__":
    unittest.main()
