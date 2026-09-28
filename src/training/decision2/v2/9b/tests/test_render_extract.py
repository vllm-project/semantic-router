import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))

from clm9b.extract import token_batches  # noqa: E402
from clm9b.render import (  # noqa: E402
    candidate_texts,
    score_level_order,
    state_text,
    to_text,
)


def row(
    kind,
    options,
    state="The invoice was charged twice.",
    instructions="Is this urgent?",
):
    return {
        "id": "r",
        "task_type": kind,
        "state": state,
        "instructions": instructions,
        "options": options,
    }


class RenderTest(unittest.TestCase):
    def test_structured_prose(self):
        text = to_text({"owner": "Lee", "tags": ["a", {"b": 1}], "ok": True})
        self.assertEqual(text, "owner: Lee\n\ntags:\n  - a\n  -\n    b: 1\n\nok: true")

    def test_state_text_puts_question_last(self):
        self.assertEqual(
            state_text(row("noul", [])),
            "The invoice was charged twice.\n\nIs this urgent?",
        )

    def test_candidate_texts_are_state_independent(self):
        options = [
            {"key": "false", "description": "No"},
            {"key": "true", "description": "Yes"},
        ]
        first = candidate_texts(row("noul", options, state="A"))
        second = candidate_texts(row("noul", options, state="B"))
        self.assertEqual(first, ["false: No", "true: Yes"])
        self.assertEqual(first, second)

    def test_empty_descriptions_fall_back(self):
        noul = candidate_texts(
            row(
                "noul",
                [
                    {"key": "true", "description": ""},
                    {"key": "false", "description": None},
                ],
            )
        )
        self.assertEqual(
            noul,
            [
                "true: Yes. This is true: Is this urgent?",
                "false: No. This is false: Is this urgent?",
            ],
        )
        choice = candidate_texts(
            row(
                "choice",
                [
                    {"key": "billing", "description": None},
                    {"key": "tech", "description": "Faults"},
                ],
            )
        )
        self.assertEqual(choice, ["billing", "Faults"])

    def test_score_order_sorts_levels(self):
        options = [
            {"key": "2", "description": "high"},
            {"key": "0", "description": "low"},
            {"key": "1", "description": "mid"},
        ]
        self.assertEqual(score_level_order(row("score", options)), [1, 2, 0])
        with self.assertRaises(ValueError):
            score_level_order(
                row(
                    "score",
                    [
                        {"key": "0", "description": "a"},
                        {"key": "2", "description": "b"},
                    ],
                )
            )


class BatchTest(unittest.TestCase):
    def test_batches_cover_every_row_within_budget(self):
        lengths = [5, 900, 17, 300, 4096, 64, 64, 8]
        batches = token_batches(lengths, budget=1024, max_rows=3)
        flat = sorted(i for batch in batches for i in batch)
        self.assertEqual(flat, list(range(len(lengths))))
        for batch in batches:
            padded = -(-max(lengths[i] for i in batch) // 8) * 8
            self.assertTrue(len(batch) == 1 or padded * len(batch) <= 1024)
            self.assertLessEqual(len(batch), 3)


if __name__ == "__main__":
    unittest.main()
