from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from benchmark.compare import compare
from benchmark.generate import (
    DEV_FAMILIES,
    FINAL_FAMILIES,
    digest,
    generate,
    prompt_record,
)
from benchmark.score import (
    DataError,
    evaluate_answer,
    score_suite,
    summarize,
    validate_suite,
)


class GeneratorTests(unittest.TestCase):
    def test_family_holdout_and_pair_invariants(self) -> None:
        self.assertFalse(set(DEV_FAMILIES) & set(FINAL_FAMILIES))
        for split in ("dev", "final"):
            for seed_index in range(8):
                seed = f"test-seed-{seed_index}".encode()
                rows = generate(split, seed, groups_per_family=3)
                items = {row["id"]: row for row in rows}
                self.assertEqual(len(items), 48)
                self.assertEqual(validate_suite(items)[0], split)
                self.assertEqual(rows, generate(split, seed, groups_per_family=3))
                for row in rows:
                    prompt = prompt_record(row)
                    self.assertEqual(set(prompt), {"id", "state", "questions"})
                    self.assertNotIn("gold", json.dumps(prompt))
                    self.assertNotIn(seed.decode(), json.dumps(row))

    def test_score_mean_and_invalid_probability_map(self) -> None:
        question = {"type": "score", "criteria": ["zero", "one", "two"]}
        gold = {"type": "score", "value": 1}
        valid = evaluate_answer(
            question,
            gold,
            {
                "type": "score",
                "score": 1.2,
                "probabilities": {"0": 0.0, "1": 0.8, "2": 0.2},
            },
        )
        self.assertEqual(valid["status"], "ok")
        self.assertTrue(valid["correct"])
        invalid = evaluate_answer(
            question,
            gold,
            {
                "score": 1.2,
                "probabilities": {"0": 0.0, "1": 0.2, "2": 0.2},
            },
        )
        self.assertEqual(invalid["status"], "invalid")

    def test_accepted_option_maps_normalize_metrics_without_changing_answer(
        self,
    ) -> None:
        question = {"type": "choice", "criteria": {"a": "A", "b": "B"}}
        gold = {"value": "a", "label_to_semantic": {"a": "A", "b": "B"}}
        for raw in ({"a": 0.99, "b": 0.0}, {"a": 0.80, "b": 0.21}):
            result = evaluate_answer(
                question, gold, {"choice": "a", "probabilities": raw}
            )
            total = sum(raw.values())
            q_a = raw["a"] / total
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["correct"])
            self.assertAlmostEqual(result["probability"]["brier"], (1 - q_a) ** 2)
            self.assertAlmostEqual(result["probability"]["nll"], -math.log(q_a))
            self.assertAlmostEqual(result["probability"]["confidence"], q_a)
            self.assertAlmostEqual(result["probability"]["option_sum_abs_delta"], 0.01)
        nonmodal = evaluate_answer(
            question,
            gold,
            {
                "choice": "a",
                "probabilities": {"a": 0.49, "b": 0.50},
            },
        )
        self.assertEqual(nonmodal["status"], "ok")
        self.assertEqual(nonmodal["point"], "a")
        self.assertTrue(nonmodal["correct"])
        self.assertGreater(nonmodal["probability"]["confidence"], 0.50)
        summary = summarize([nonmodal])
        self.assertEqual(summary["accuracy_all"], 1.0)
        self.assertEqual(summary["option_probability_sum_abs_delta"]["n"], 1)
        self.assertEqual(summary["selective"]["0.5"]["n"], 1)

        normalized_threshold = evaluate_answer(
            question,
            gold,
            {
                "choice": "a",
                "probabilities": {"a": 0.95, "b": 0.06},
            },
        )
        threshold_summary = summarize([normalized_threshold])
        self.assertEqual(threshold_summary["selective"]["0.95"]["n"], 0)
        self.assertAlmostEqual(threshold_summary["ece_10"], 1 - 0.95 / 1.01)

        score_result = evaluate_answer(
            {"type": "score", "criteria": ["zero", "one", "two"]},
            {"value": 1},
            {"score": 1.22, "probabilities": {"0": 0.0, "1": 0.8, "2": 0.21}},
        )
        self.assertEqual(score_result["status"], "ok")
        self.assertAlmostEqual(
            score_result["probability"]["option_sum_abs_delta"], 0.01
        )

    def test_perfect_adapter_predictions_score_all_relations(self) -> None:
        rows = generate("final", b"test-private-entropy-with-at-least-32-bytes", 2)
        predictions = []
        for row in rows:
            answers = {}
            for key, gold in row["gold"].items():
                qtype, value = gold["type"], gold["value"]
                if qtype == "noul":
                    answers[key] = {"type": qtype, "noul": float(value)}
                elif qtype == "choice":
                    labels = row["questions"][key]["criteria"]
                    answers[key] = {
                        "type": qtype,
                        "choice": value,
                        "probabilities": {
                            label: float(label == value) for label in labels
                        },
                    }
                else:
                    count = len(row["questions"][key]["criteria"])
                    answers[key] = {
                        "type": qtype,
                        "score": float(value),
                        "probabilities": {
                            str(i): float(i == value) for i in range(count)
                        },
                    }
            predictions.append(
                {
                    "id": row["id"],
                    "answers": answers,
                    "latency_ms": 10.0,
                    "source_input_sha256": row["provenance"]["payload_sha256"],
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            gold_path = Path(tmp) / "gold.jsonl"
            prediction_path = Path(tmp) / "predictions.jsonl"
            gold_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            prediction_path.write_text(
                "\n".join(json.dumps(row) for row in predictions) + "\n"
            )
            report = score_suite(
                gold_path, prediction_path, "test", "revision", "test-backend"
            )
        self.assertEqual(report["overall"]["accuracy_all"], 1.0)
        self.assertEqual(report["schema_version"], "typed-decision-report/2")
        self.assertGreater(
            report["overall"]["option_probability_sum_abs_delta"]["n"], 0
        )
        self.assertEqual(report["macro_family_accuracy"], 1.0)
        self.assertEqual(report["overall"]["brier"], 0.0)
        for relation in report["pairs"].values():
            self.assertEqual(relation["relation_consistency_all"], 1.0)
            self.assertEqual(relation["joint_accuracy_all"], 1.0)

    def test_paired_group_comparison(self) -> None:
        rows = generate("dev", b"paired-bootstrap-test", 2)
        perfect, wrong = [], []
        for row in rows:
            correct_answers, wrong_answers = {}, {}
            for key, gold in row["gold"].items():
                qtype, value = gold["type"], gold["value"]
                if qtype == "noul":
                    correct_answers[key] = {"noul": float(value)}
                    wrong_answers[key] = {"noul": float(not value)}
                elif qtype == "choice":
                    other = next(
                        label
                        for label in row["questions"][key]["criteria"]
                        if label != value
                    )
                    correct_answers[key] = {"choice": value}
                    wrong_answers[key] = {"choice": other}
                else:
                    other = (value + 1) % len(row["questions"][key]["criteria"])
                    correct_answers[key] = {"score": float(value)}
                    wrong_answers[key] = {"score": float(other)}
            source_hash = row["provenance"]["payload_sha256"]
            perfect.append(
                {
                    "id": row["id"],
                    "answers": correct_answers,
                    "source_input_sha256": source_hash,
                }
            )
            wrong.append(
                {
                    "id": row["id"],
                    "answers": wrong_answers,
                    "source_input_sha256": source_hash,
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, content in (
                ("gold", rows),
                ("perfect", perfect),
                ("wrong", wrong),
            ):
                (root / f"{name}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in content) + "\n"
                )
            result = compare(
                root / "gold.jsonl",
                root / "perfect.jsonl",
                root / "wrong.jsonl",
                left_name="perfect",
                right_name="wrong",
                iterations=100,
            )
        self.assertEqual(result["family_macro"]["delta_left_minus_right"], 1.0)
        self.assertEqual(result["family_macro"]["delta_ci95"], [1.0, 1.0])

    def test_score_rejects_missing_or_stale_prompt_hash(self) -> None:
        rows = generate("dev", b"prediction-source-hash-contract", 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold_path = root / "gold.jsonl"
            predictions_path = root / "predictions.jsonl"
            gold_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            first = rows[0]
            correct_hash = digest(
                {"state": first["state"], "questions": first["questions"]}
            )
            self.assertEqual(correct_hash, first["provenance"]["payload_sha256"])
            prediction = {
                "id": first["id"],
                "answers": {},
                "source_input_sha256": correct_hash,
            }
            predictions_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            score_suite(gold_path, predictions_path, "test", "revision", "backend")
            for bad_hash in (None, "0" * 64):
                prediction["source_input_sha256"] = bad_hash
                predictions_path.write_text(
                    json.dumps(prediction) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(DataError, "source_input_sha256"):
                    score_suite(
                        gold_path, predictions_path, "test", "revision", "backend"
                    )


if __name__ == "__main__":
    unittest.main()
