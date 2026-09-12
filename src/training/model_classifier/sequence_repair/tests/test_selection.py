"""Exact operating points, tied probabilities, and explicit checkpoint selection."""

import copy
import hashlib
import io
import json
import math
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.training.model_classifier.sequence_repair import train
from src.training.model_classifier.sequence_repair.data import classification_metrics
from src.training.model_classifier.sequence_repair.selection import (
    BINARY_FP_SELECTION,
    binary_checkpoint_key,
    fit_binary_operating_point,
    validate_selection_options,
)
from src.training.model_classifier.sequence_repair.train import selection_score


def evidence(gold, scores, positive_id=1):
    labels = {"unsafe": positive_id, "safe": 1 - positive_id}
    rows, predictions = [], []
    for index, (target, score) in enumerate(zip(gold, scores, strict=True)):
        identifier = str(index)
        rows.append({"id": identifier, "label": "unsafe" if target else "safe"})
        probabilities = [score, 1 - score] if positive_id == 0 else [1 - score, score]
        predictions.append({"id": identifier, "probabilities": probabilities})
    return rows, predictions, labels


class BinarySelectionTests(unittest.TestCase):
    def fit(self, gold, scores, budget=0.1, positive_id=1):
        return fit_binary_operating_point(
            *evidence(gold, scores, positive_id), "unsafe", budget
        )

    def test_all_ties_are_included_and_false_positive_budget_is_exact(self):
        result = self.fit([1, 1, 0, 0], [0.9, 0.8, 0.8, 0.2], budget=0)
        self.assertEqual(result["threshold"], 0.9)
        self.assertEqual(result["confusion"], {"tp": 1, "fp": 0, "fn": 1, "tn": 2})
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["support"], {"positive": 2, "negative": 2})
        result = self.fit([1] + [0] * 10, [0.8] * 4 + [0.2] * 7, budget=0.3)
        self.assertEqual(result["maximum_false_positives"], 3)
        self.assertEqual(result["confusion"]["fp"], 3)
        self.assertEqual(result["threshold"], 0.8)

    def test_positive_label_need_not_have_index_one_and_ids_are_joined(self):
        rows, predictions, labels = evidence([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], 0)
        original = copy.deepcopy((rows, predictions, labels))
        first = fit_binary_operating_point(rows, predictions, labels, "unsafe", 0.5)
        second = fit_binary_operating_point(
            rows[::-1], predictions[1:] + predictions[:1], labels, "unsafe", 0.5
        )
        self.assertEqual(first, second)
        self.assertEqual(first["threshold"], 0.7)
        self.assertEqual(first["positive_label_id"], 0)
        self.assertEqual(first["confusion"]["tp"], 2)
        self.assertEqual((rows, predictions, labels), original)

    def test_saturated_safe_probability_is_infeasible_without_hidden_reject_all(self):
        result = self.fit([1, 0], [1.0, 1.0], budget=0)
        self.assertFalse(result["feasible"])
        self.assertIsNone(result["threshold"])
        self.assertIsNone(binary_checkpoint_key(result))
        self.assertIn("probability 1", result["infeasible_reason"])
        json.dumps(result, allow_nan=False)
        result = self.fit([1, 0], [0.1, 0.9], budget=0)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["threshold"], 1.0)
        self.assertEqual(result["recall"], 0.0)

    def test_sweep_matches_independent_exhaustive_confusion_counts(self):
        rng = random.Random(19)
        for _ in range(100):
            gold = [1, 0] + [rng.randrange(2) for _ in range(18)]
            scores = [rng.choice([0, 0.1, 0.4, 0.7, 0.9, 1]) for _ in gold]
            budget = rng.choice([0, 0.1, 0.3, 1])
            result = self.fit(gold, scores, budget)
            negative_support = gold.count(0)
            allowed = math.floor(Fraction(str(budget)) * negative_support)
            candidates = []
            for threshold in {0, 1, *scores}:
                tp = sum(
                    target == 1 and score >= threshold
                    for target, score in zip(gold, scores, strict=True)
                )
                fp = sum(
                    target == 0 and score >= threshold
                    for target, score in zip(gold, scores, strict=True)
                )
                if fp <= allowed:
                    candidates.append((tp / gold.count(1), -fp, threshold))
            self.assertEqual(
                binary_checkpoint_key(result), max(candidates) if candidates else None
            )

    def test_checkpoint_ranking_uses_recall_then_fp_then_threshold(self):
        gold = [1, 1, 0, 0]
        first = self.fit(gold, [0.9, 0.4, 0.45, 0.1], budget=0)
        second = self.fit(gold, [0.9, 0.55, 0.6, 0.1], budget=0)
        third = self.fit(gold, [0.49, 0.45, 0.4, 0.1], budget=0)
        # Argmax F1 prefers the first model (0.733 versus 0.333), while at the
        # fixed zero-FP budget the third model recovers both positive requests.
        first_f1 = classification_metrics(gold, [1, 0, 0, 0], {0: "safe", 1: "unsafe"})
        third_f1 = classification_metrics(gold, [0, 0, 0, 0], {0: "safe", 1: "unsafe"})
        self.assertGreater(first_f1["macro_f1"], third_f1["macro_f1"])
        self.assertEqual(binary_checkpoint_key(first), binary_checkpoint_key(second))
        self.assertGreater(binary_checkpoint_key(third), binary_checkpoint_key(first))
        few_fp = self.fit(gold, [0.9, 0.8, 0.2, 0.1], budget=0.5)
        more_fp = self.fit(gold, [0.9, 0.8, 0.85, 0.1], budget=0.5)
        self.assertGreater(
            binary_checkpoint_key(few_fp), binary_checkpoint_key(more_fp)
        )
        higher = self.fit(gold, [0.95, 0.85, 0.2, 0.1], budget=0.5)
        self.assertGreater(binary_checkpoint_key(higher), binary_checkpoint_key(few_fp))

    def test_invalid_contract_options_support_or_prediction_evidence_rejected(self):
        rows, predictions, labels = evidence([1, 0], [0.9, 0.1])
        for mapping, positive, budget in [
            ({"safe": 0, "unsafe": 1, "other": 2}, "unsafe", 0.1),
            ({"safe": 0, "unsafe": 2}, "unsafe", 0.1),
            (labels, "UNSAFE", 0.1),
            (labels, None, 0.1),
            *(
                (labels, "unsafe", value)
                for value in [None, True, "0.1", -1, 1.1, math.nan, math.inf]
            ),
        ]:
            with self.subTest(
                mapping=mapping, positive=positive, budget=budget
            ), self.assertRaises(ValueError):
                validate_selection_options(
                    BINARY_FP_SELECTION, mapping, positive, budget
                )
        for records, predicted in [
            (rows[:1], predictions[:1]),
            (rows + rows[:1], predictions + predictions[:1]),
            (rows, predictions[:1]),
            (rows, predictions + predictions[:1]),
            (rows, [{"id": "wrong", "probabilities": [0.1, 0.9]}, predictions[1]]),
        ]:
            with self.assertRaises(ValueError):
                fit_binary_operating_point(records, predicted, labels, "unsafe", 0.1)
        for scores in [
            [math.nan, 0.9],
            [math.inf, 0],
            [-0.1, 1.1],
            [0.5],
            [0.6, 0.6],
            [True, 0],
        ]:
            malformed = copy.deepcopy(predictions)
            malformed[0]["probabilities"] = scores
            with self.assertRaises(ValueError):
                fit_binary_operating_point(rows, malformed, labels, "unsafe", 0.1)

    def test_legacy_selection_stays_explicit_and_ignores_no_new_options(self):
        for mode in [
            "macro-f1",
            "source-macro-f1",
            "length-macro-f1",
            "source-present-macro-f1",
        ]:
            validate_selection_options(mode, {"a": 0, "b": 1, "c": 2}, None, None)
            for positive, budget in [("unsafe", None), (None, 0.1)]:
                with self.assertRaises(ValueError):
                    validate_selection_options(
                        mode, {"safe": 0, "unsafe": 1}, positive, budget
                    )
        self.assertEqual(selection_score({"macro_f1": 0.73}, "macro-f1"), 0.73)

    def test_training_entrypoint_records_and_passes_explicit_evaluation_precision(self):
        class EvaluationReachedError(Exception):
            pass

        parameter = SimpleNamespace(requires_grad=True, numel=lambda: 1)
        model = SimpleNamespace(
            config=SimpleNamespace(
                problem_type="single_label_classification",
                classifier_pooling="cls",
                max_position_embeddings=32,
            ),
            parameters=lambda: iter([parameter]),
            gradient_checkpointing_enable=Mock(),
            to=Mock(),
            train=Mock(),
        )
        fake_torch = SimpleNamespace(
            set_num_threads=Mock(),
            manual_seed=Mock(),
            optim=SimpleNamespace(AdamW=Mock()),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
            contract = root / "contract.json"
            contract.write_text(
                json.dumps(
                    {
                        "label2id": {"safe": 0, "unsafe": 1},
                        "classifier_pooling": "mean",
                    }
                )
            )
            for partition in ["train", "dev"]:
                (root / f"{partition}.jsonl").write_text(
                    "".join(
                        json.dumps(
                            {
                                "id": f"{partition}-{label}",
                                "group_id": f"{partition}-{label}",
                                "text": f"{partition} fixture {label}",
                                "label": label,
                            }
                        )
                        + "\n"
                        for label in ["safe", "unsafe"]
                    )
                )
            arguments = [
                "train",
                "--base",
                "fixture-base",
                "--base-revision",
                "fixture-revision",
                "--adapter",
                str(adapter),
                "--contract",
                str(contract),
                "--train",
                str(root / "train.jsonl"),
                "--dev",
                str(root / "dev.jsonl"),
                "--max-length",
                "32",
                "--selection",
                BINARY_FP_SELECTION,
                "--positive-label",
                "unsafe",
                "--selection-false-positive-budget",
                "0.1",
            ]
            for explicit in [None, "bfloat16", "float32"]:
                adapter_config = adapter / "adapter_config.json"
                if explicit is not None:
                    adapter_config.write_text('{"r": 32}\n')
                output = root / str(explicit)
                argv = [*arguments, "--output", str(output)]
                if explicit is not None:
                    argv += ["--evaluation-dtype", explicit]
                with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(
                    sys, "argv", argv
                ), patch.object(
                    train,
                    "load_model",
                    return_value=(
                        model,
                        lambda *args, **kwargs: {"input_ids": [1, 2]},
                        {"safe": 0, "unsafe": 1},
                        {0: "safe", 1: "unsafe"},
                    ),
                ), patch.object(
                    train, "task_head_scope", return_value={"fixture": True}
                ), patch.object(
                    train, "evaluate_records", side_effect=EvaluationReachedError
                ) as evaluate, patch(
                    "builtins.print"
                ), self.assertRaises(
                    EvaluationReachedError
                ):
                    train.main()
                expected = explicit or "bfloat16"
                self.assertEqual(evaluate.call_args.kwargs["dtype"], expected)
                receipt = json.loads((output / "run.json").read_text())
                self.assertEqual(receipt["evaluation_dtype"], expected)
                self.assertIn("BF16 autocast", receipt["precision"])
                self.assertEqual(
                    receipt["contract_sha256"],
                    hashlib.sha256(contract.read_bytes()).hexdigest(),
                )
                # Record the configuration of the constructed model, including
                # its resolved default problem type, rather than copying input.
                self.assertEqual(receipt["classifier_pooling"], "cls")
                self.assertEqual(receipt["problem_type"], "single_label_classification")
                self.assertEqual(
                    receipt["initial_adapter_config_sha256"],
                    (
                        hashlib.sha256(adapter_config.read_bytes()).hexdigest()
                        if explicit is not None
                        else None
                    ),
                )
            with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(
                sys,
                "argv",
                [
                    *arguments,
                    "--output",
                    str(root / "invalid"),
                    "--evaluation-dtype",
                    "float16",
                ],
            ), patch.object(train, "load_model") as load, redirect_stderr(
                io.StringIO()
            ):
                with self.assertRaises(SystemExit) as failure:
                    train.main()
                self.assertEqual(failure.exception.code, 2)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
