from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from v2.eval.sealed import candidates as runner
from v2.eval.sealed.schema import (
    Candidate,
    SourceSpec,
    display_order,
    letters,
    validate,
)

SPEC = SourceSpec(
    key="fake",
    dataset_id="org/fake",
    revision="0" * 40,
    licence="mit",
    licence_flag=None,
    first_release="2026-08-01",
    evidence="test",
    label_provenance="human",
    languages=("en",),
    tasks=("fake/stance", "fake/flag", "fake/rating"),
)


def choice(item: str, gold: str = "favor") -> Candidate:
    return Candidate(
        "fake",
        "fake/stance",
        item,
        f"g-{item}",
        gold,
        "en",
        "text",
        {
            "type": "choice",
            "instructions": "Stance?",
            "criteria": {"favor": "For", "against": "Against"},
        },
        gold,
    )


class SchemaTest(unittest.TestCase):
    def test_valid_candidates(self):
        self.assertEqual(validate(choice("a"), SPEC), [])
        noul = Candidate(
            "fake",
            "fake/flag",
            "b",
            "g",
            "True",
            "en",
            {"x": 1},
            {
                "type": "noul",
                "instructions": "?",
                "criteria": {"true": "y", "false": "n"},
            },
            True,
        )
        self.assertEqual(validate(noul, SPEC), [])
        score = Candidate(
            "fake",
            "fake/rating",
            "c",
            "g",
            "2",
            "pt-BR",
            "t",
            {"type": "score", "instructions": "?", "criteria": ["lo", "mid", "hi"]},
            2,
        )
        self.assertEqual(validate(score, SPEC), [])

    def test_invalid_candidates(self):
        self.assertIn(
            "gold not among choice keys", validate(choice("a", "neutral"), SPEC)
        )
        bad = choice("a")
        bad.question["criteria"] = {"Favor Now": "x", "against": "y"}
        self.assertIn(
            "choice keys must be snake_case or single capitals", validate(bad, SPEC)
        )
        early = choice("a")
        early.date = "2026-05-31"
        self.assertIn("row dated before the cutoff", validate(early, SPEC))
        long = choice("a")
        long.state = "x" * 30_000
        self.assertIn("input exceeds MAX_INPUT_CHARS", validate(long, SPEC))
        noul = Candidate(
            "fake",
            "fake/flag",
            "b",
            "g",
            "1",
            "en",
            "t",
            {"type": "noul", "instructions": "?", "criteria": {"yes": "y", "no": "n"}},
            1,
        )
        problems = validate(noul, SPEC)
        self.assertIn("noul criteria must be exactly true/false", problems)
        self.assertIn("noul gold must be bool", problems)

    def test_display_order_is_a_gold_independent_permutation(self):
        order = display_order("item-7", 5)
        self.assertEqual(sorted(order), list(range(5)))
        self.assertEqual(order, display_order("item-7", 5))
        self.assertNotEqual(
            [display_order(f"i{n}", 4)[0] for n in range(40)].count(0), 40
        )
        self.assertEqual(letters(3), ["A", "B", "C"])

    def test_runner_writes_valid_rows_and_counts(self):
        module = types.ModuleType("v2.eval.sealed.sources.fake")
        module.SPEC = SPEC
        module.candidates = lambda root: iter(
            [choice("a"), choice("b", "against"), choice("a"), choice("c", "nope")]
        )
        sys.modules["v2.eval.sealed.sources.fake"] = module
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "snap"
                root.mkdir()
                (root / "data.jsonl").write_text("{}\n")
                receipt = runner.run("fake", root, Path(tmp) / "out" / "c.jsonl")
                rows = [
                    json.loads(x)
                    for x in (Path(tmp) / "out" / "c.jsonl").read_text().splitlines()
                ]
        finally:
            del sys.modules["v2.eval.sealed.sources.fake"]
        self.assertEqual(receipt["valid"], 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            receipt["invalid"],
            {"duplicate task/source_item_id": 1, "gold not among choice keys": 1},
        )
        self.assertEqual(receipt["tasks"]["fake/stance"]["label:against"], 1)
        self.assertIn("data.jsonl", receipt["snapshot"]["files"])


if __name__ == "__main__":
    unittest.main()
