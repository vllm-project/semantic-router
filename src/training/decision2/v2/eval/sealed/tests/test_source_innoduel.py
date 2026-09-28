from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from v2.eval.sealed import candidates as runner
from v2.eval.sealed.schema import validate
from v2.eval.sealed.sources import innoduel

HEADER = (
    "block_id",
    "org_id",
    "respondent",
    "matchup_id",
    "question",
    "chosen_answer",
    "rejected_answer",
    "chosen_answer_en",
    "priority_score",
    "language",
)
QUESTION_1 = "Miten parantaisimme työhyvinvointia?"
QUESTION_2 = "Hur kan vi minska energiförbrukningen?"


def vote(block, org, respondent, matchup, question, chosen, rejected, language):
    return (
        block,
        org,
        respondent,
        matchup,
        question,
        chosen,
        rejected,
        "x",
        "0.9",
        language,
    )


ROWS = [
    vote(
        "1", "Org_01", "R1", "M1", QUESTION_1, "Lisää etäpäiviä", "Uusi kahvikone", "fi"
    ),
    vote(
        "1",
        "Org_01",
        "R1",
        "M2",
        QUESTION_1,
        "Yhteinen lounas",
        "Lisää etäpäiviä",
        "fi",
    ),
    vote(
        "1",
        "Org_01",
        "R2",
        "M2",
        QUESTION_1,
        "Lisää etäpäiviä",
        "Yhteinen lounas",
        "fi",
    ),
    vote(
        "1",
        "Org_01",
        "R2",
        "M3",
        QUESTION_1,
        "Liikuntasetelit",
        "Liikuntasetelit",
        "fi",
    ),
    vote(
        "1", "Org_01", "R3", "M4", QUESTION_1, "Uusi kahvikone", "Yhteinen lounas", "fi"
    ),
    vote(
        "1", "Org_01", "R3", "M5", QUESTION_1, "Liikuntasetelit", "Uusi kahvikone", "fi"
    ),
    vote(
        "2", "Org_01", "R1", "M7", QUESTION_1, "Uusi kahvikone", "Yhteinen lounas", "fi"
    ),
    vote(
        "2", "Org_01", "R1", "M8", QUESTION_1, "Uusi kahvikone", "Liikuntasetelit", "fi"
    ),
    vote(
        "3",
        "Org_02",
        "R1",
        "M1",
        QUESTION_2,
        "Släck lampor",
        "Solpaneler på taket",
        "sv",
    ),
]


def snapshot(tmp: str, rows=ROWS) -> Path:
    root = Path(tmp) / "snap"
    root.mkdir()
    with (root / "sample.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return root


def convert(rows=ROWS):
    with tempfile.TemporaryDirectory() as tmp:
        return list(innoduel.candidates(snapshot(tmp, rows)))


class InnoduelTest(unittest.TestCase):
    def test_merges_votes_and_drops_disagreement(self):
        items = convert()
        self.assertEqual([c.source_item_id for c in items], ["1/M1", "1/M4", "3/M1"])
        merged = items[1]
        self.assertEqual(merged.group_id, "Org_01/1")
        self.assertEqual(items[2].group_id, "Org_02/3")
        self.assertEqual([c.language for c in items], ["fi", "fi", "sv"])
        for item in items:
            self.assertEqual(validate(item, innoduel.SPEC), [])
            self.assertEqual(
                item.overlap_texts,
                [item.state["question"], *item.state["ideas"].values()],
            )
            self.assertEqual(item.balance_label, item.gold)
            self.assertIsNone(item.date)
        self.assertEqual(merged.state["ideas"][merged.gold], "Uusi kahvikone")
        self.assertNotIn("x", merged.overlap_texts)

    def test_state_does_not_depend_on_the_vote(self):
        swapped = [r[:5] + (r[6], r[5]) + r[7:] if r[3] == "M1" else r for r in ROWS]
        before, after = convert()[0], convert(swapped)[0]
        self.assertEqual(before.state, after.state)
        self.assertNotEqual(before.gold, after.gold)

    def test_display_order_balances_positions(self):
        rows = [
            vote(
                "9",
                "Org_09",
                "R1",
                f"M{n}",
                QUESTION_1,
                f"Idea {n}a",
                f"Idea {n}b",
                "en",
            )
            for n in range(40)
        ]
        golds = [c.gold for c in convert(rows)]
        self.assertEqual(len(golds), 40)
        self.assertTrue(10 <= golds.count("A") <= 30)

    def test_runner_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = snapshot(tmp)
            receipt = runner.run("innoduel", root, Path(tmp) / "out.jsonl")
        self.assertEqual(receipt["invalid"], {})
        self.assertEqual(receipt["valid"], 3)
        self.assertEqual(receipt["tasks"]["innoduel/preferred_idea"]["groups"], 2)


if __name__ == "__main__":
    unittest.main()
