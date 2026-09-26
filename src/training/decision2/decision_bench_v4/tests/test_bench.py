from __future__ import annotations

import json
import unittest

from decision_bench_v4.bench import (
    JUDGMENT_POLICY,
    evaluate_answer,
    input_digest,
    prepare_case,
    summarize,
)


class DecisionBenchAdapterTest(unittest.TestCase):
    def case(self, *, task: str = "ENG-1", category: str = "engineering") -> dict:
        return {
            "id": "example-1",
            "task": task,
            "category": category,
            "modality": "image" if task == "DSN-1" else "text",
            "state": {"record": "Use the evidence in this record."},
            "title": "SECRET GOLD TITLE",
            "provenance": "SECRET PROVENANCE",
            "questions": [
                {
                    "id": "decision",
                    "type": "choice",
                    "gold": "b",
                    "ask": "SECRET ASK",
                    "rationale": "SECRET RATIONALE",
                    "instructions": "Choose the supported label.",
                    "options": {"a": "first", "b": "second"},
                    "option_order": ["b", "a"],
                }
            ],
        }

    def test_native_prompt_is_gold_free_and_preserves_order(self) -> None:
        case = self.case()
        prompt, target = prepare_case(case, {"ENG-1": {"category": "engineering"}})
        visible = json.dumps(prompt)
        for excluded in (
            "SECRET GOLD TITLE",
            "SECRET PROVENANCE",
            "SECRET ASK",
            "SECRET RATIONALE",
            '"gold"',
        ):
            self.assertNotIn(excluded, visible)
        self.assertEqual(prompt["state"], case["state"])
        self.assertEqual(list(prompt["questions"]["decision"]["criteria"]), ["b", "a"])
        self.assertEqual(
            prompt["questions"]["decision"]["instructions"],
            JUDGMENT_POLICY + "Choose the supported label.",
        )
        self.assertEqual(target["gold"], "b")
        self.assertEqual(
            target["source_input_sha256"],
            input_digest(prompt["state"], prompt["questions"]),
        )

    def test_visual_only_icon_is_explicitly_excluded(self) -> None:
        self.assertIsNone(
            prepare_case(
                self.case(task="DSN-1", category="design"),
                {"DSN-1": {"category": "design"}},
            )
        )
        altered = self.case(task="DSN-1", category="design")
        altered["modality"] = "text"
        with self.assertRaises(ValueError):
            prepare_case(altered, {"DSN-1": {"category": "design"}})

    def test_invalid_and_point_probability_disagreement(self) -> None:
        target = {"labels": ["a", "b"], "gold": "b"}
        valid = evaluate_answer(
            {"type": "choice", "choice": "b", "probabilities": {"a": 0.8, "b": 0.2}},
            target,
        )
        self.assertTrue(valid["valid"])
        self.assertTrue(valid["correct"])
        self.assertTrue(valid["point_argmax_disagreement"])
        self.assertAlmostEqual(valid["brier"], 1.28)
        invalid = evaluate_answer({"choice": "b", "probabilities": {"a": 0.8}}, target)
        self.assertFalse(invalid["valid"])
        self.assertFalse(invalid["correct"])
        summary = summarize([valid, invalid])
        self.assertEqual(
            (summary["items"], summary["answered_valid"], summary["correct"]), (2, 1, 1)
        )


if __name__ == "__main__":
    unittest.main()
