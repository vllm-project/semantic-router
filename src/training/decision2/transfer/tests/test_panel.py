"""Small integrity tests; full author-receipt reproduction runs on the host."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from transfer.build import (
    PANEL_VERSION,
    PILOT_TASKS,
    criteria_for,
    normalized_context_sha256,
    parse_prompt,
    sha_bytes,
    sha_value,
)
from transfer.normalize_jev import api_body_sha256, normalize
from transfer.score import ece_15, evaluate, macro_f1, option_sum_diagnostics, score


class TransferPanelTest(unittest.TestCase):
    def test_prompt_parser_and_composite_tropes(self):
        prompt = (
            "Choose a trope.\nA: First trait\nB: Second trait\nConstraint: Pick one."
        )
        self.assertEqual(
            parse_prompt(prompt),
            ("Choose a trope.", {"A": "First trait", "B": "Second trait"}),
        )
        instructions, criteria = criteria_for(
            "tropes",
            prompt,
            {"tropes": {"A": "A", "B": "B", "A, B": "A, B"}},
            {},
        )
        self.assertEqual(instructions, "Choose a trope.")
        self.assertEqual(criteria["A, B"], "A: First trait; B: Second trait")

    def test_metrics_count_invalid_as_misses_and_use_full_brier(self):
        self.assertAlmostEqual(
            macro_f1(["a", "b", "c"], ["a", "a", None], ["a", "b", "c"]), 2 / 9
        )
        gold = {"gold": "b", "labels": ["a", "b"]}
        prediction = {
            "answers": {
                "label": {
                    "type": "choice",
                    "choice": "b",
                    "probabilities": {"a": 0.2, "b": 0.8},
                    "confidence": 0.7,
                }
            }
        }
        result = evaluate(gold, prediction)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["brier_sum"], 0.08)
        other = dict(result, correct=False, pmax=0.6, native_confidence=0.6)
        self.assertAlmostEqual(ece_15([result, other], "pmax"), 0.4)
        self.assertEqual(
            evaluate(gold, {"answers": {"label": {"choice": "b"}}})["reason"],
            "probabilities",
        )

    def test_native_choice_and_accepted_sum_normalization(self):
        gold = {"gold": "a", "labels": ["a", "b"]}
        for raw in ({"a": 0.79, "b": 0.20}, {"a": 0.80, "b": 0.21}):
            result = evaluate(
                gold,
                {
                    "answers": {
                        "label": {
                            "type": "choice",
                            "choice": "a",
                            "probabilities": raw,
                        }
                    }
                },
            )
            q_a = raw["a"] / sum(raw.values())
            self.assertTrue(result["valid"])
            self.assertTrue(result["correct"])
            self.assertAlmostEqual(result["brier_sum"], 2 * (1 - q_a) ** 2)
            self.assertAlmostEqual(result["gold_probability"], q_a)
            self.assertAlmostEqual(result["pmax"], q_a)
            self.assertAlmostEqual(result["option_sum_abs_delta"], 0.01)
        nonmodal = evaluate(
            gold,
            {
                "answers": {
                    "label": {
                        "choice": "a",
                        "probabilities": {"a": 0.49, "b": 0.50},
                    }
                }
            },
        )
        self.assertTrue(nonmodal["valid"])
        self.assertTrue(nonmodal["correct"])
        self.assertGreater(nonmodal["pmax"], 0.50)
        self.assertAlmostEqual(nonmodal["gold_probability"], 0.49 / 0.99)
        self.assertEqual(option_sum_diagnostics([nonmodal])["n"], 1)
        self.assertAlmostEqual(
            -math.log(nonmodal["gold_probability"]), -math.log(0.49 / 0.99)
        )
        normalized_threshold = evaluate(
            gold,
            {
                "answers": {
                    "label": {
                        "choice": "a",
                        "probabilities": {"a": 0.95, "b": 0.06},
                    }
                }
            },
        )
        self.assertAlmostEqual(ece_15([normalized_threshold], "pmax"), 1 - 0.95 / 1.01)

    def test_v2_report_records_original_sum_without_changing_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = PILOT_TASKS[0]
            gold = [
                {
                    "id": str(i),
                    "task": task,
                    "role": "pilot",
                    "panel_version": PANEL_VERSION,
                    "labels": ["a", "b"],
                    "gold": label,
                    "input_sha256": str(i) * 64,
                }
                for i, label in ((1, "a"), (2, "b"))
            ]
            predictions = [
                {
                    "id": "1",
                    "source_input_sha256": "1" * 64,
                    "answers": {
                        "label": {
                            "choice": "a",
                            "probabilities": {
                                "a": 0.95,
                                "b": 0.06,
                            },
                        }
                    },
                }
            ]
            gold_path, prediction_path = root / "gold.jsonl", root / "predictions.jsonl"
            gold_path.write_text("".join(json.dumps(row) + "\n" for row in gold))
            prediction_path.write_text(
                "".join(json.dumps(row) + "\n" for row in predictions)
            )
            report = score(gold_path, prediction_path)
        self.assertEqual(report["score_schema_version"], "css-transfer-score/2")
        self.assertEqual(report["tasks"][task]["n"], 2)
        self.assertEqual(report["tasks"][task]["valid_n"], 1)
        self.assertEqual(report["tasks"][task]["accuracy_all"], 0.5)
        self.assertEqual(
            report["tasks"][task]["option_probability_sum_abs_delta"]["n"], 1
        )
        self.assertAlmostEqual(
            report["roles"]["pilot"]["option_probability_sum_abs_delta"]["max"], 0.01
        )

    def test_normalized_hash(self):
        self.assertEqual(
            normalized_context_sha256("  HELLO\nWorld  "), sha_bytes(b"hello world")
        )

    def test_official_receipt_hash_is_verified_then_rebased(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt = {
                "id": "one",
                "state": "text",
                "questions": {
                    "label": {
                        "type": "choice",
                        "instructions": "Pick",
                        "criteria": {"a": "A", "b": "B"},
                    }
                },
            }
            response = {
                "answers": {
                    "label": {
                        "type": "choice",
                        "choice": "a",
                        "probabilities": {"a": 0.8, "b": 0.2},
                    }
                }
            }
            receipt = {
                "id": "one",
                "requested_model": "jev-1.13.0",
                "returned_model": "jev-1.13.0",
                "http_status": 200,
                "input_sha256": api_body_sha256(
                    prompt["state"], prompt["questions"], "jev-1.13.0"
                ),
                "response": response,
                "latency_seconds": 0.1,
            }
            prompts, receipts, output = (
                root / name
                for name in ("prompts.jsonl", "receipts.jsonl", "output.jsonl")
            )
            prompts.write_text(json.dumps(prompt) + "\n")
            receipts.write_text(json.dumps(receipt) + "\n")
            self.assertEqual(
                normalize(prompts, receipts, output, "jev-1.13.0")["normalized"], 1
            )
            converted = json.loads(output.read_text())
            self.assertEqual(
                converted["source_input_sha256"],
                sha_value(
                    {
                        "state": prompt["state"],
                        "questions": prompt["questions"],
                    }
                ),
            )
            self.assertEqual(converted["api_body_sha256"], receipt["input_sha256"])


if __name__ == "__main__":
    unittest.main()
