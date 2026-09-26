"""CPU-only tests for the published This-That adapter projection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from inference import this_that


class FakeTokenizer:
    def encode(self, value, add_special_tokens=False):
        return list(value)


class FakeDecider:
    tokenizer = FakeTokenizer()

    def decide(self, state, questions, **kwargs):
        return [
            SimpleNamespace(options=tuple(q.options), probabilities=(0.2, 0.8), index=1)
            for q in questions
        ]


class FakeQuestion:
    def __init__(self, text, options):
        self.text = text
        self.options = options


class ThisThatAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prompts = self.root / "prompts.jsonl"
        self.output = self.root / "predictions.jsonl"
        self.questions = {
            "route": {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {"a": "Alpha", "b": "Beta"},
            },
            "yes": {"type": "noul", "instructions": "Does this hold?"},
            "rating": {
                "type": "score",
                "instructions": "Rate",
                "criteria": ["low", "high"],
            },
        }
        self.rows = [
            {"id": "one", "state": "short", "questions": self.questions},
            {"id": "two", "state": "x" * 1600, "questions": self.questions},
        ]
        self.prompts.write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows),
            encoding="utf-8",
        )
        self.identity = {
            "backend": "this-that",
            "model_id": this_that.MODEL_ID,
            "model_revision": this_that.MODEL_REVISION,
            "revision_attested": True,
            "source_revision": this_that.SOURCE_REVISION,
            "model_config_sha256": this_that.ARTIFACT_SHA256["config.json"],
            "model_weights_sha256": this_that.ARTIFACT_SHA256["model.safetensors"],
        }

    def test_full_distribution_and_explicit_projection(self):
        _, specs = this_that.native_questions(self.rows[0])
        self.assertEqual(
            [s["options"] for s in specs],
            [["a: Alpha", "b: Beta"], ["no", "yes"], ["0: low", "1: high"]],
        )
        answers = [
            this_that.answer_from_native(
                SimpleNamespace(
                    options=tuple(spec["options"]), probabilities=(0.2, 0.8), index=1
                ),
                spec,
            )
            for spec in specs
        ]
        self.assertEqual(answers[0]["choice"], "b")
        self.assertEqual(answers[1]["noul"], 0.8)
        self.assertEqual(answers[2]["score"], 0.8)
        self.assertEqual(
            answers[2]["projection"], "ordered_declared_choice_expected_index"
        )

    def test_collection_counts_overflow_as_invalid(self):
        fake_build = lambda *_args, **_kwargs: {"ids": [1, 2, 3]}
        runtime = {"runtime_qualification": "unvalidated_rocm"}
        with (
            patch.object(this_that, "verify_release", return_value=self.identity),
            patch.object(
                this_that,
                "load_native",
                return_value=(FakeDecider(), FakeQuestion, fake_build, runtime),
            ),
            patch.object(this_that, "synchronize"),
        ):
            report = this_that.collect(
                model_path=self.root,
                source_path=self.root,
                revision=this_that.MODEL_REVISION,
                prompts=self.prompts,
                output=self.output,
            )
        predictions = [
            json.loads(line) for line in self.output.read_text().splitlines()
        ]
        self.assertEqual(report["context_overflow_now"], 1)
        self.assertEqual(
            predictions[0]["answers"]["route"]["probabilities"], {"a": 0.2, "b": 0.8}
        )
        self.assertEqual(
            predictions[1]["answers"]["rating"]["error"], "context_overflow"
        )
        self.assertEqual(
            predictions[0]["native_output_contract"], "generic_declared_options"
        )

    def test_unsupported_type_fails_before_model_load(self):
        with self.assertRaisesRegex(ValueError, "unsupported question type"):
            this_that.native_questions(
                {
                    "id": "bad",
                    "state": "s",
                    "questions": {"r": {"type": "rank", "instructions": "Rank"}},
                }
            )

    def test_wrong_revision_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact HF revision"):
            this_that.verify_release(self.root, self.root, "moving-main")


if __name__ == "__main__":
    unittest.main()
