"""CPU-only contract tests for the native Kai/Lex collector."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference import kai_lex


class FakeClient:
    def system_one(self, *, state, questions):
        if state == "long":
            raise ValueError(
                "Record systemone:0:0 exceeds 1024 tokens; no implicit truncation"
            )
        return {
            "model": "Decision-1.0-Kai",
            "answers": {
                "route": {
                    "type": "choice",
                    "choice": "a",
                    "probabilities": {"a": 0.75, "b": 0.25},
                    "confidence": 0.75,
                },
            },
            "usage": {"input_tokens": 17, "output_tokens": 0},
        }


class KaiLexAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prompts = self.root / "prompts.jsonl"
        self.output = self.root / "predictions.jsonl"
        self.question = {
            "route": {
                "type": "choice",
                "instructions": "Pick one",
                "criteria": {"a": "Alpha", "b": "Beta"},
            },
        }
        self.rows = [
            {"id": "one", "state": "short", "questions": self.question},
            {"id": "two", "state": "long", "questions": self.question},
        ]
        self.prompts.write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows),
            encoding="utf-8",
        )
        self.identity = {
            "backend": "kai",
            "model_id": kai_lex.MODELS["kai"]["model_id"],
            "model_name": kai_lex.MODELS["kai"]["model_name"],
            "model_revision": kai_lex.MODELS["kai"]["revision"],
            "revision_attested": True,
            "model_config_sha256": kai_lex.MODELS["kai"]["manifest_sha256"],
        }

    def test_native_answers_and_overflow_keep_scoring_denominator(self):
        runtime = {"runtime_matches_validated": True, "runtime_differences": {}}
        with (
            patch.object(kai_lex, "verify_native_bundle", return_value=self.identity),
            patch.object(kai_lex, "runtime_report", return_value=runtime),
            patch.object(
                kai_lex, "load_system_one", return_value=(object(), FakeClient())
            ),
            patch.object(kai_lex, "synchronize"),
        ):
            report = kai_lex.collect(
                backend="kai",
                model_path=self.root,
                revision=self.identity["model_revision"],
                prompts=self.prompts,
                output=self.output,
                device="cuda:0",
            )
        predictions = [
            json.loads(line) for line in self.output.read_text().splitlines()
        ]
        self.assertEqual(report["context_overflow_now"], 1)
        self.assertEqual(
            predictions[0]["answers"]["route"]["probabilities"], {"a": 0.75, "b": 0.25}
        )
        self.assertEqual(
            predictions[1]["answers"]["route"]["error"], "context_overflow"
        )
        self.assertEqual({row["id"] for row in predictions}, {"one", "two"})

    def test_resume_rejects_changed_prompt(self):
        receipt = {
            **self.identity,
            "id": "one",
            "adapter_version": kai_lex.ADAPTER_VERSION,
            "source_input_sha256": kai_lex.digest(
                {
                    "state": self.rows[0]["state"],
                    "questions": self.rows[0]["questions"],
                }
            ),
            "answers": {"route": {"type": "choice", "choice": "a"}},
        }
        self.output.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
        self.assertEqual(
            kai_lex.completed_ids(self.output, self.rows, self.identity), {"one"}
        )
        altered = [dict(self.rows[0], state="different"), self.rows[1]]
        with self.assertRaisesRegex(ValueError, "stale input"):
            kai_lex.completed_ids(self.output, altered, self.identity)

    def test_unsupported_question_type_fails_before_loading(self):
        self.prompts.write_text(
            json.dumps(
                {
                    "id": "one",
                    "state": "hello",
                    "questions": {
                        "route": {
                            "type": "rank",
                            "instructions": "Rank",
                            "criteria": ["a", "b"],
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "unsupported question types"):
            kai_lex.collect(
                backend="kai",
                model_path=self.root,
                revision=self.identity["model_revision"],
                prompts=self.prompts,
                output=self.output,
            )

    def test_wrong_revision_fails_before_reading_files(self):
        with self.assertRaisesRegex(ValueError, "requires published revision"):
            kai_lex.verify_native_bundle(self.root, "kai", "untrusted")


if __name__ == "__main__":
    unittest.main()
