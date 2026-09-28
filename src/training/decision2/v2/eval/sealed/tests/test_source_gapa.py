from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from v2.eval.sealed import candidates as runner
from v2.eval.sealed.schema import validate
from v2.eval.sealed.sources import gapa as source

COLUMNS = [
    "attention_check",
    "attribute",
    "person_term",
    "prolific_id",
    "question",
    "rating",
    "source",
    "n_attchecks_seen",
    "n_attchecks_passed",
]


def rating(attribute, term, value, rater="P001", passed="5.0", check="False"):
    return {
        "attention_check": check,
        "attribute": attribute,
        "person_term": term,
        "prolific_id": rater,
        "question": f"How likely is it for someone to say that a {term} has {attribute}?",
        "rating": value,
        "source": "human",
        "n_attchecks_seen": "5.0",
        "n_attchecks_passed": passed,
    }


HUMAN = [
    rating("a soft voice", "woman", "6"),
    rating("a soft voice", "woman", "7", rater="P002", passed="4.0"),
    rating("a soft voice", "man", "3"),
    rating("a soft voice", "man", "4", rater="P003"),
    rating("a soft voice", "man", "1", rater="P004", passed="3.0"),
    rating("a soft voice", "nonbinary person", "4", rater="P005"),
    rating("a soft voice", "nonbinary person", "7", rater="P005", check="True"),
    rating("broad shoulders", "man", "7"),
    rating("broad shoulders", "man", "6", rater="P002"),
    rating("broad shoulders", "man", "7", rater="P003"),
    rating("broad shoulders", "person", "5"),
    rating("broad shoulders", "woman", ""),
]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class GapaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "snap"
        write(self.root / "clean" / "human.csv", HUMAN)
        write(self.root / "clean" / "llm.csv", [rating("an llm phrase", "man", "2")])
        write(self.root / "clean" / "novel.csv", [rating("a novel phrase", "man", "2")])
        write(
            self.root / "raw" / "human.csv",
            HUMAN + [rating("a raw phrase", "man", "2")],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_each_attribute_and_person_term_becomes_a_score_item(self):
        items = list(source.candidates(self.root))
        self.assertEqual(
            [c.source_item_id for c in items],
            [
                "a soft voice|man",
                "a soft voice|nonbinary person",
                "a soft voice|woman",
                "broad shoulders|man",
            ],
        )
        self.assertEqual([c.gold for c in items], [3, 3, 6, 6])
        self.assertEqual([c.balance_label for c in items], ["3", "3", "6", "6"])
        self.assertEqual(
            [c.group_id for c in items], ["a soft voice"] * 3 + ["broad shoulders"]
        )
        for item in items:
            self.assertEqual(validate(item, source.SPEC), [])
            self.assertEqual(item.question, source.QUESTION)
            self.assertEqual(item.overlap_texts, [item.group_id])
        self.assertEqual(len(source.QUESTION["criteria"]), 7)

    def test_state_names_only_the_term_and_attribute(self):
        first = next(source.candidates(self.root))
        self.assertEqual(
            first.state, {"person_term": "man", "physical_attribute": "a soft voice"}
        )
        for item in source.candidates(self.root):
            text = json.dumps(item.state)
            for leak in ("rating", "gold", "label", "llm", "novel", "raw"):
                self.assertNotIn(leak, text)

    def test_runner_accepts_the_snapshot_deterministically(self):
        out = Path(self.tmp.name) / "out"
        first = runner.run("gapa", self.root, out / "a.jsonl")
        second = runner.run("gapa", self.root, out / "b.jsonl")
        self.assertEqual(first["invalid"], {})
        self.assertEqual(first["valid"], 4)
        self.assertEqual(first["tasks"]["gapa/association"]["groups"], 2)
        self.assertEqual(first["candidates_sha256"], second["candidates_sha256"])


if __name__ == "__main__":
    unittest.main()
