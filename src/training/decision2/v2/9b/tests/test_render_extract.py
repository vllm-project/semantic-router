import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))

from clm9b.extract import token_batches  # noqa: E402
from clm9b.lux_teacher import example_rows  # noqa: E402

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


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


@unittest.skipUnless(HAS_TORCH, "native segment helpers import torch")
class RenderTest(unittest.TestCase):
    def test_disaggregated_texts_are_the_native_segments(self):
        from clm9b.render import candidate_texts, state_text
        from training.model.decision_model import segments

        sample = row(
            "choice",
            [
                {"key": "billing", "description": {"team": "payments"}},
                {"key": "tech", "description": None},
            ],
            state={"owner": "Lee", "items": [1, 2]},
        )
        prefix, options, _ = segments(sample)
        self.assertEqual(state_text(sample) + "\nOptions:", prefix)
        self.assertEqual(
            [f"\n<option>\n{text}\n</option>" for text in candidate_texts(sample)],
            options,
        )

    def test_candidate_texts_are_state_independent(self):
        from clm9b.render import candidate_texts

        options = [
            {"key": "false", "description": "No"},
            {"key": "true", "description": "Yes"},
        ]
        self.assertEqual(
            candidate_texts(row("noul", options, state="A")),
            candidate_texts(row("noul", options, state="B")),
        )

    def test_score_order_sorts_levels(self):
        from clm9b.render import score_level_order

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

    def test_single_prompt_batches(self):
        self.assertEqual(
            token_batches([30, 10, 20], budget=10**6, max_rows=1), [[1], [2], [0]]
        )


class LuxExampleTest(unittest.TestCase):
    def test_example_rows_use_lux_noul_defaults(self):
        example = {
            "requests": {
                "r": {
                    "state": "s",
                    "questions": {
                        "a": {"type": "noul", "instructions": "Is it?"},
                        "b": {
                            "type": "score",
                            "instructions": "How?",
                            "criteria": ["low", "high"],
                        },
                        "c": {
                            "type": "choice",
                            "instructions": "Which?",
                            "criteria": {"x": "X", "y": "Y"},
                        },
                    },
                }
            }
        }
        rows = {r["id"]: r for r in example_rows(example)}
        self.assertEqual(
            rows["r/a"]["options"],
            [
                {"key": "false", "description": "The answer to the question is no."},
                {"key": "true", "description": "The answer to the question is yes."},
            ],
        )
        self.assertEqual([o["key"] for o in rows["r/b"]["options"]], ["0", "1"])
        self.assertEqual([o["key"] for o in rows["r/c"]["options"]], ["x", "y"])


if __name__ == "__main__":
    unittest.main()
