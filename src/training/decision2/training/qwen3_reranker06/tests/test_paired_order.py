"""Semantic option alignment and frozen control invariants for the paired arm."""

from __future__ import annotations

import unittest

from training.qwen3_reranker06 import pilot, train
from training.qwen3_reranker06.paired_order import (
    aligned_indices,
    reverse_choice,
)
from training.qwen3_reranker06.paired_postsave import categorical


class PairedOrderTests(unittest.TestCase):
    def test_reverse_preserves_non_alphabetic_keys_and_descriptions(self) -> None:
        question = {
            "type": "choice",
            "instructions": "Choose the evidence-supported route",
            "criteria": {
                "zebra": "requires a signed permit",
                "alpha": "requires an unsigned note",
                "m7": "requires both records",
                "a2": "requires neither record",
            },
        }
        before = pilot.option_items(question)
        reversed_question = reverse_choice(question)
        after = pilot.option_items(reversed_question)
        self.assertEqual(after, list(reversed(before)))
        self.assertEqual(question["criteria"]["zebra"], "requires a signed permit")
        self.assertEqual(reversed_question["instructions"], question["instructions"])

        chosen = train.selected_indices(
            "fixed-row-id", [key for key, _ in before], "m7"
        )
        other = aligned_indices(
            [key for key, _ in before], [key for key, _ in after], chosen
        )
        self.assertEqual(
            [before[index][0] for index in chosen],
            [after[index][0] for index in other],
        )
        self.assertIn("m7", [before[index][0] for index in chosen])

    def test_mismatched_keys_and_non_choice_rejected(self) -> None:
        with self.assertRaises(ValueError):
            aligned_indices(["a", "b"], ["a", "c"], [0])
        with self.assertRaises(ValueError):
            aligned_indices(["a", "a"], ["a", "a"], [0])
        with self.assertRaises(ValueError):
            reverse_choice({"type": "score", "criteria": ["low", "high"]})

    def test_postsave_categorical_projection(self) -> None:
        self.assertEqual(
            categorical({"type": "choice", "choice": "m7"}), ("choice", "m7")
        )
        self.assertEqual(categorical({"type": "noul", "noul": 0.51}), ("noul", True))
        self.assertEqual(
            categorical({"type": "score", "native_level": 2}), ("score", 2)
        )
        self.assertEqual(
            categorical({"type": "choice", "error": "context_overflow"}),
            ("choice", ("invalid", "context_overflow")),
        )


if __name__ == "__main__":
    unittest.main()
