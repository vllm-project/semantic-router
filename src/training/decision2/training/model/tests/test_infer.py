import json
import math
import tempfile
import unittest
from pathlib import Path

from benchmark.score import evaluate_answer

from training.model.data import file_sha256
from training.model.infer import (
    checkpoint_fingerprint,
    load_prompts,
    normalized_answer,
    product_answer,
    prompt_input_sha256,
    question_to_row,
    run_prompts,
    write_output,
)


def item():
    return {
        "id": "p1",
        "state": {"fact": True},
        "questions": {
            "c": {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {"alpha": "A", "beta": "B"},
            },
            "n": {
                "type": "noul",
                "instructions": "Yes?",
                "criteria": {"false": "No", "true": "Yes"},
            },
            "s": {
                "type": "score",
                "instructions": "Rate",
                "criteria": ["Low", "Mid", "High"],
            },
        },
    }


class InferContractTest(unittest.TestCase):
    def test_native_typed_answers_without_gold(self):
        prompt = item()

        def fake_encode(row, _tokenizer, max_length):
            self.assertEqual(max_length, 64)
            self.assertNotIn("gold", row)
            return {
                "id": row["id"],
                "ids": [1, 2, 3],
                "keys": [option["key"] for option in row["options"]],
            }

        def fake_predict(encoded):
            table = {"c": [3.0, 0.0], "n": [-1.0, 2.0], "s": [-2.0, -1.0, 3.0]}
            return [table[row["id"].split("/")[-1]] for row in encoded]

        predictions, counts = run_prompts(
            [prompt],
            tokenizer=object(),
            max_length=64,
            temperature=1.0,
            encode_fn=fake_encode,
            predict_fn=fake_predict,
            model_sha256="model",
            adapter_sha256="adapter",
        )
        record = predictions[0]
        self.assertEqual(record["id"], "p1")
        self.assertEqual(set(record["answers"]), set(prompt["questions"]))
        self.assertEqual(record["usage"]["input_tokens"], 9)
        self.assertEqual(record["adapter_status"], "ok")
        self.assertEqual(counts["valid_questions"], 3)
        self.assertEqual(counts["truncated_questions"], 0)
        self.assertEqual(record["input_sha256"], prompt_input_sha256(prompt))
        # The keyed v3 benchmark adapter keeps its old response schema; the
        # product-only fields are added later by Decision2.system_one.
        self.assertNotIn("confidence", record["answers"]["c"])
        self.assertNotIn("legend", record["answers"]["s"])
        gold = {
            "c": {
                "type": "choice",
                "value": "alpha",
                "label_to_semantic": {"alpha": "a", "beta": "b"},
            },
            "n": {"type": "noul", "value": True},
            "s": {"type": "score", "value": 2},
        }
        for key in prompt["questions"]:
            scored = evaluate_answer(
                prompt["questions"][key], gold[key], record["answers"][key]
            )
            self.assertEqual(scored["status"], "ok")
            self.assertTrue(scored["correct"])

    def test_invalid_and_over_budget_are_counted_without_truncation(self):
        prompt = item()
        prompt["questions"]["c"]["instructions"] = ""

        def fake_encode(row, _tokenizer, _max_length):
            if row["task_type"] == "score":
                raise ValueError(
                    "p1/s: 100 tokens exceeds max_length=64; no truncation"
                )
            return {"id": row["id"], "ids": [1], "keys": ["false", "true"]}

        predictions, counts = run_prompts(
            [prompt],
            tokenizer=None,
            max_length=64,
            temperature=1,
            encode_fn=fake_encode,
            predict_fn=lambda _: [[float("nan"), 0.0]],
            model_sha256="m",
            adapter_sha256="a",
        )
        record = predictions[0]
        self.assertEqual(set(record["answers"]), {"c", "n", "s"})
        self.assertEqual(record["adapter_status"], "invalid")
        self.assertEqual(counts["invalid_questions"], 3)
        self.assertEqual(counts["over_budget_questions"], 1)
        self.assertEqual(counts["truncated_questions"], 0)
        self.assertEqual(record["adapter_errors"]["s"], "max_length_exceeded")

    def test_type_and_prompt_safety(self):
        prompt = item()
        choice = question_to_row(prompt, "c", prompt["questions"]["c"])
        self.assertEqual(
            [option["key"] for option in choice["options"]], ["alpha", "beta"]
        )
        self.assertEqual(choice["instructions"], "Choose")
        self.assertEqual(
            question_to_row(
                prompt,
                "c",
                {
                    "type": "choice",
                    "instructions": "Choose",
                    "criteria": {"a": "", "b": "B"},
                },
            )["options"][0]["description"],
            "",
        )
        score = question_to_row(prompt, "s", prompt["questions"]["s"])
        self.assertEqual(
            [option["key"] for option in score["options"]], ["0", "1", "2"]
        )
        with self.assertRaisesRegex(ValueError, "noul requires"):
            question_to_row(
                prompt,
                "n",
                {
                    "type": "noul",
                    "instructions": "Yes?",
                    "criteria": {"yes": "Y", "no": "N"},
                },
            )
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            normalized_answer("choice", ["a", "b"], [math.inf, 0], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text(json.dumps(dict(prompt, gold={})) + "\n")
            with self.assertRaisesRegex(ValueError, "never gold"):
                load_prompts(path)
            path.write_text(json.dumps(prompt) + "\n")
            self.assertEqual(len(load_prompts(path)), 1)

    def test_structured_system_one_questions_and_native_limits(self):
        prompt = item()
        prompt["state"] = [{"record": {"priority": 3}}]
        choice = question_to_row(
            prompt,
            "c",
            {
                "type": "choice",
                "instructions": {
                    "question": "Choose from `candidate`",
                    "candidate": [1, 2],
                },
                "criteria": {"a": None, "b": {"rule": ["one", "two"]}},
            },
        )
        self.assertEqual(choice["state"], prompt["state"])
        self.assertIsNone(choice["options"][0]["description"])
        self.assertEqual(choice["options"][1]["description"], {"rule": ["one", "two"]})
        noul = question_to_row(
            prompt,
            "n",
            {
                "type": "noul",
                "instructions": ["Is it true?"],
                "criteria": {"true": {"meaning": "yes"}},
            },
        )
        self.assertEqual(
            [option["key"] for option in noul["options"]], ["false", "true"]
        )
        self.assertEqual(noul["options"][1]["description"], {"meaning": "yes"})
        self.assertEqual(
            [
                option["key"]
                for option in question_to_row(
                    prompt, "n", {"type": "noul", "instructions": "Yes?"}
                )["options"]
            ],
            ["false", "true"],
        )
        score = question_to_row(
            prompt,
            "s",
            {
                "type": "score",
                "instructions": ["Rate", {"scope": "record"}],
                "criteria": [{"level": number} for number in range(10)],
            },
        )
        self.assertEqual(
            [option["key"] for option in score["options"]], [str(i) for i in range(10)]
        )
        self.assertEqual(
            len(
                question_to_row(
                    prompt,
                    "c",
                    {
                        "type": "choice",
                        "instructions": "Choose",
                        "criteria": {str(i): None for i in range(255)},
                    },
                )["options"]
            ),
            255,
        )
        for bad in (
            {
                "type": "choice",
                "instructions": {},
                "criteria": {"a": None, "b": object()},
            },
            {"type": "noul", "instructions": "Yes?", "criteria": {"other": "maybe"}},
            {"type": "score", "instructions": "Rate", "criteria": [None, "high"]},
            {"type": "score", "instructions": "Rate", "criteria": ["x"] * 11},
            {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {str(i): None for i in range(256)},
            },
            {
                "type": "choice",
                "instructions": {"x": float("nan")},
                "criteria": {"a": None, "b": None},
            },
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                question_to_row(prompt, "bad", bad)

    def test_product_fields_preserve_benchmark_probabilities(self):
        cases = (
            ("choice", ["a", "b"], [2.0, 0.0], [None, {"label": "B"}]),
            ("noul", ["false", "true"], [0.0, 2.0], ["No", "Yes"]),
            (
                "score",
                ["0", "1", "2"],
                [0.0, 1.0, 2.0],
                ["Low", {"middle": 1}, ["High"]],
            ),
        )
        for kind, keys, logits, descriptions in cases:
            with self.subTest(kind=kind):
                benchmark = normalized_answer(kind, keys, logits, 1.0)
                product = product_answer(kind, keys, logits, 1.0, descriptions)
                self.assertEqual({name: product[name] for name in benchmark}, benchmark)
                if kind == "noul":
                    self.assertNotIn("confidence", product)
                else:
                    self.assertGreaterEqual(product["confidence"], 0.0)
                    self.assertLessEqual(product["confidence"], 1.0)
        self.assertEqual(
            product["legend"], {"0": "Low", "1": '{"middle":1}', "2": '["High"]'}
        )
        tie = product_answer(
            "choice", ["first", "second"], [0.0, 0.0], 1.0, [None, None]
        )
        self.assertEqual(
            normalized_answer("choice", ["first", "second"], [0.0, 0.0], 1.0)["choice"],
            None,
        )
        self.assertEqual(tie["choice"], "first")
        self.assertEqual(tie["probabilities"], {"first": 0.5, "second": 0.5})
        self.assertEqual(tie["confidence"], 0.0)

    def test_model_and_output_hash_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "checkpoint"
            (checkpoint / "backbone").mkdir(parents=True)
            (checkpoint / "decision_config.json").write_text("{}")
            (checkpoint / "decision_head.safetensors").write_bytes(b"head")
            (checkpoint / "backbone" / "model.safetensors").write_bytes(b"backbone")
            first = checkpoint_fingerprint(checkpoint)
            (checkpoint / "decision_head.safetensors").write_bytes(b"changed")
            second = checkpoint_fingerprint(checkpoint)
            self.assertNotEqual(first["model_sha256"], second["model_sha256"])
            (checkpoint / "calibration.json").write_text('{"temperature_by_type":{}}')
            (checkpoint / "materialization_receipt.json").write_text('{"receipt":true}')
            self.assertEqual(
                second["model_sha256"],
                checkpoint_fingerprint(checkpoint)["model_sha256"],
            )
            output = directory / "predictions.jsonl"
            manifest = {"model_sha256": second["model_sha256"]}
            write_output(output, [{"id": "p1", "answers": {}}], manifest)
            saved = json.loads(
                (directory / "predictions.jsonl.manifest.json").read_text()
            )
            self.assertEqual(saved["predictions_sha256"], file_sha256(output))
            with self.assertRaises(FileExistsError):
                write_output(output, [], manifest)


if __name__ == "__main__":
    unittest.main()
