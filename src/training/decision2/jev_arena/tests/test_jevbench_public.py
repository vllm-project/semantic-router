"""Tests for JevBench public scoring rules and gold isolation."""

from __future__ import annotations

import math
import unittest

from jev_arena.jevbench_public import _ece_15, _evaluate, _valid_probs, input_digest


class PublicScoreTest(unittest.TestCase):
    def test_exact_label_set_and_renormalization(self) -> None:
        self.assertIsNone(_valid_probs({"a": 0.6}, ["a", "b"]))
        self.assertIsNone(_valid_probs({"a": 0.6, "b": 0.4, "c": 0}, ["a", "b"]))
        self.assertIsNone(_valid_probs({"a": math.nan, "b": 1}, ["a", "b"]))
        self.assertIsNone(_valid_probs({"a": True, "b": 0}, ["a", "b"]))
        self.assertIsNone(_valid_probs({"a": 0.7, "b": 0.34}, ["a", "b"]))
        accepted = _valid_probs({"a": 0.601, "b": 0.403}, ["a", "b"])
        self.assertIsNotNone(accepted)
        probs, strict, renormalized = accepted
        self.assertFalse(strict)
        self.assertTrue(renormalized)
        self.assertAlmostEqual(sum(probs.values()), 1)

    def test_choice_uses_probabilities_and_lexicographic_ties(self) -> None:
        target = {"task_type": "choice", "labels": ["z", "a"], "expected": "a"}
        result = _evaluate(
            {"type": "choice", "choice": "z", "probabilities": {"z": 0.5, "a": 0.5}},
            target,
        )
        self.assertTrue(result["valid"])
        self.assertTrue(result["correct"])
        self.assertEqual(result["predicted"], "a")
        self.assertTrue(result["point_disagrees_with_argmax"])

    def test_noul_probability_and_score_argmax(self) -> None:
        binary = {"task_type": "noul", "labels": ["no", "yes"], "expected": "yes"}
        self.assertTrue(_evaluate({"type": "noul", "noul": 0.8}, binary)["correct"])
        self.assertFalse(_evaluate({"type": "noul", "noul": 1.1}, binary)["valid"])
        ordinal = {"task_type": "score", "labels": ["0", "1", "2"], "expected": 2}
        result = _evaluate(
            {
                "type": "score",
                "score": 1.1,
                "probabilities": {"0": 0.01, "1": 0.48, "2": 0.51},
            },
            ordinal,
        )
        self.assertTrue(result["correct"])

    def test_missing_counts_as_wrong_and_ece(self) -> None:
        target = {"task_type": "choice", "labels": ["a", "b"], "expected": "a"}
        missing = _evaluate(None, target)
        self.assertFalse(missing["valid"])
        self.assertFalse(missing["correct"])
        self.assertIsNone(_ece_15([missing]))
        correct = _evaluate({"probabilities": {"a": 0.8, "b": 0.2}}, target)
        self.assertAlmostEqual(_ece_15([correct]), 0.2)

    def test_input_hash_ignores_gold(self) -> None:
        state, questions = "example", {"decision": {"type": "noul"}}
        digest = input_digest(state, questions)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, input_digest(state, questions))
        self.assertNotEqual(digest, input_digest("other", questions))


if __name__ == "__main__":
    unittest.main()
