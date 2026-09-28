from __future__ import annotations

import unittest

from v2.eval import leak_effects
from v2.eval.leak_audit import Question


def choice(item_id: str, keys, descriptions, gold: int, state=None) -> Question:
    return Question(
        "p", item_id, "g", item_id, "choice", keys, descriptions, gold, "", state
    )


class LeakEffectsTest(unittest.TestCase):
    def test_cue_options(self):
        q = choice("a", ["x", "y", "z"], ["short", "the longest one", "mid-size"], 1)
        self.assertEqual(leak_effects.cue_option(q, "longest-description"), 1)
        self.assertEqual(leak_effects.cue_option(q, "shortest-description"), 0)
        tie = choice("b", ["x", "y"], ["same", "same"], 0)
        self.assertIsNone(leak_effects.cue_option(tie, "longest-description"))
        state = {"current": "S0", "transitions": [{"from": "S0", "to": "S1"}]}
        typed = choice("c", ["S1", "S0", "S2"], ["t"] * 3, 0, state)
        self.assertEqual(leak_effects.cue_option(typed, "state-current"), 1)

    def test_chosen_key(self):
        self.assertEqual(leak_effects.chosen_key({"choice": "a"}), "a")
        self.assertEqual(
            leak_effects.chosen_key({"probabilities": {"a": 0.2, "b": 0.8}}), "b"
        )
        self.assertIsNone(
            leak_effects.chosen_key({"probabilities": {"a": 0.5, "b": 0.5}})
        )
        self.assertIsNone(leak_effects.chosen_key(None))

    def test_follow_model_counts(self):
        questions = [
            (choice("a", ["x", "y"], ["long text", "s"], 0), "q", 0),
            (choice("b", ["x", "y"], ["long text", "s"], 1), "q", 0),
            (choice("c", ["x", "y"], ["long text", "s"], 1), "q", 0),
        ]
        predictions = {
            "a": {"answers": {"q": {"choice": "x"}}},
            "b": {"answers": {"q": {"choice": "x"}}},
            "c": {"answers": {"q": {"choice": "nope"}}},
        }
        result = leak_effects.follow_model(questions, predictions)
        self.assertEqual(result["questions"], 3)
        self.assertEqual(result["valid"], 2)
        self.assertAlmostEqual(result["picks_cue_option"], 2 / 3)
        self.assertEqual(
            result["gold_is_cue_option"], {"n": 1, "correct": 1, "accuracy": 1.0}
        )
        self.assertEqual(result["gold_is_not_cue_option"]["correct"], 0)
        self.assertEqual(result["wrong_answers"], 1)
        self.assertEqual(result["wrong_answers_on_cue_option"], 1.0)


if __name__ == "__main__":
    unittest.main()
