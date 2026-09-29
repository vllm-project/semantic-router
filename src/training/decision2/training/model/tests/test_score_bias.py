from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from benchmark.score import evaluate_answer

from training.model.data import file_sha256
from training.model.infer import checkpoint_fingerprint, normalized_answer, run_prompts
from training.model.score_bias import (
    SCORE_BIAS_FORMAT,
    apply,
    load_score_bias,
    validate_offsets,
    validate_score_bias,
)

MODEL = "a" * 64


def report(**changes):
    value = {
        "format": SCORE_BIAS_FORMAT,
        "model_sha256": MODEL,
        "offsets": {"3": [-0.5, 0.0, 0.5], "5": [0.0, 0.1, 0.2, 0.3, -0.6]},
        "fit": {"note": "synthetic"},
    }
    value.update(changes)
    return value


def prompt():
    return {
        "id": "p1",
        "state": {"fact": True},
        "questions": {
            "c": {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {"alpha": "A", "beta": "B", "gamma": "C"},
            },
            "n": {
                "type": "noul",
                "instructions": "Yes?",
                "criteria": {"false": "No", "true": "Yes"},
            },
            "s3": {
                "type": "score",
                "instructions": "Rate",
                "criteria": ["Low", "Mid", "High"],
            },
            "s4": {
                "type": "score",
                "instructions": "Rate",
                "criteria": ["a", "b", "c", "d"],
            },
        },
    }


TABLE = {
    "c": [0.2, 0.0, 0.1],
    "n": [-0.3, 0.4],
    "s3": [0.0, 0.0, 0.8],
    "s4": [0.5, 0.0, 0.0, 0.0],
}


def encode(row, _tokenizer, _max_length):
    return {
        "id": row["id"],
        "ids": [1],
        "keys": [option["key"] for option in row["options"]],
    }


def predict(encoded):
    return [TABLE[item["id"].split("/")[-1]] for item in encoded]


class ScoreBiasFileTest(unittest.TestCase):
    def test_load_binds_the_model_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "score_bias.json"
            path.write_text(json.dumps(report()))
            offsets, raw = load_score_bias(path, MODEL)
            self.assertEqual(offsets[3], [-0.5, 0.0, 0.5])
            self.assertEqual(sorted(offsets), [3, 5])
            self.assertEqual(raw["fit"], {"note": "synthetic"})
            with self.assertRaisesRegex(ValueError, "differs"):
                load_score_bias(path, "b" * 64)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_score_bias(path, "model")

    def test_rejects_malformed_files(self):
        bad = (
            report(format="dev2-score-bias-v0"),
            report(offsets={"3": [0.0, 1.0]}),
            report(offsets={"3": [0.0, float("nan"), 1.0]}),
            report(offsets={"3": [0.0, "1", 1.0]}),
            report(offsets={"1": [0.0]}),
            report(offsets={"256": [0.0] * 256}),
            report(offsets={"03": [0.0, 0.0, 0.0]}),
            report(offsets={}),
            report(fit=None),
            report(model_sha256="A" * 64),
            {**report(), "extra": 1},
        )
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_score_bias(value, MODEL)
        self.assertEqual(validate_offsets({"10": [0] * 10})[10], [0.0] * 10)

    def test_apply_only_touches_present_level_counts(self):
        offsets = validate_offsets({"3": [-1.0, 0.0, 1.0]})
        self.assertEqual(apply(offsets, [1.0, 2.0, 3.0], 3), [0.0, 2.0, 4.0])
        self.assertEqual(apply(offsets, [1.0, 2.0, 3.0, 4.0], 4), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(apply(offsets, [1.0, 2.0], 3), [1.0, 2.0])
        self.assertEqual(apply(offsets, [1.0, None, 2.0], 3), [1.0, None, 2.0])

    def test_bias_file_does_not_change_the_model_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "backbone").mkdir()
            (checkpoint / "decision_config.json").write_text("{}")
            (checkpoint / "decision_head.safetensors").write_bytes(b"head")
            (checkpoint / "backbone" / "model.safetensors").write_bytes(b"weights")
            before = checkpoint_fingerprint(checkpoint)
            (checkpoint / "score_bias.json").write_text(json.dumps(report()))
            self.assertEqual(before, checkpoint_fingerprint(checkpoint))


class ScoreBiasInferTest(unittest.TestCase):
    def run_one(self, **kwargs):
        predictions, counts = run_prompts(
            [prompt()],
            tokenizer=None,
            max_length=64,
            temperature=1.0,
            encode_fn=encode,
            predict_fn=predict,
            model_sha256=MODEL,
            adapter_sha256="adapter",
            **kwargs,
        )
        self.assertEqual(counts["valid_questions"], 4)
        return predictions[0]

    def test_without_bias_the_record_is_unchanged(self):
        plain = self.run_one()
        self.assertNotIn("score_bias_sha256", plain)
        for key, logits in TABLE.items():
            kind = prompt()["questions"][key]["type"]
            keys = [o for o in prompt()["questions"][key]["criteria"]]
            if kind == "score":
                keys = [str(i) for i in range(len(keys))]
            self.assertEqual(
                plain["answers"][key], normalized_answer(kind, keys, logits, 1.0)
            )

    def test_bias_moves_score_only_and_recomputes_the_expected_level(self):
        offsets = validate_offsets({"3": [2.0, 0.0, -2.0], "5": [0.0] * 5})
        plain = self.run_one()
        biased = self.run_one(score_bias=offsets, score_bias_sha256="f" * 64)
        self.assertEqual(biased["score_bias_sha256"], "f" * 64)
        for key in ("c", "n", "s4"):
            self.assertEqual(biased["answers"][key], plain["answers"][key])
        expected = normalized_answer("score", ["0", "1", "2"], [2.0, 0.0, -1.2], 1.0)
        self.assertEqual(biased["answers"]["s3"], expected)
        question = prompt()["questions"]["s3"]
        gold = {"type": "score", "value": 0}
        before = evaluate_answer(question, gold, plain["answers"]["s3"])
        after = evaluate_answer(question, gold, biased["answers"]["s3"])
        self.assertEqual((before["status"], before["point"]), ("ok", 2))
        self.assertEqual((after["status"], after["point"]), ("ok", 0))
        score = biased["answers"]["s3"]
        self.assertTrue(
            math.isclose(
                score["score"],
                sum(int(k) * p for k, p in score["probabilities"].items()),
            )
        )
        for field in ("input_sha256", "model_sha256", "adapter_sha256", "usage"):
            self.assertEqual(biased[field], plain[field])

    def test_bias_needs_its_hash_and_malformed_output_stays_invalid(self):
        offsets = validate_offsets({"3": [0.0, 0.0, 0.0]})
        with self.assertRaises(ValueError):
            self.run_one(score_bias=offsets)
        predictions, counts = run_prompts(
            [prompt()],
            tokenizer=None,
            max_length=64,
            temperature=1.0,
            encode_fn=encode,
            predict_fn=lambda encoded: [[math.nan] * len(e["keys"]) for e in encoded],
            model_sha256=MODEL,
            adapter_sha256="adapter",
            score_bias=offsets,
            score_bias_sha256="f" * 64,
        )
        self.assertEqual(counts["invalid_questions"], 4)
        self.assertEqual(
            predictions[0]["answers"]["s3"]["error"], "invalid_model_output"
        )

    def test_recorded_file_hash_matches_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "score_bias.json"
            path.write_text(json.dumps(report()))
            offsets, _ = load_score_bias(path, MODEL)
            record = self.run_one(
                score_bias=offsets, score_bias_sha256=file_sha256(path)
            )
            self.assertEqual(record["score_bias_sha256"], file_sha256(path))


if __name__ == "__main__":
    unittest.main()
