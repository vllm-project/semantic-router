"""CPU-only tests for Laya's native truncation accounting and receipts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.score import evaluate_answer

from inference import laya


class FakeTokenizer:
    mask_token = "[MASK]"


class FakeAgent:
    tok = FakeTokenizer()
    cfg = {"max_len": 1024, "head_max_len": 256}
    device = type("Device", (), {"type": "cuda"})()

    @staticmethod
    def _check_question(_qid, _question):
        return None

    @staticmethod
    def _to_internal(question):
        return {"t": question["type"], "ins": question["instructions"]}

    def system_one(self, *, state, questions):
        return {
            "model": "laya-rl-agent",
            "answers": {
                qid: {
                    "type": "choice",
                    "choice": "a",
                    "probabilities": {"a": 0.7, "b": 0.3},
                    "confidence": 0.2,
                }
                for qid in questions
            },
            "usage": {"input_tokens": 12, "output_tokens": 0},
        }


def fake_encode(_tokenizer, value, **_kwargs):
    return {"input_ids": list(value)}


def fake_render(question):
    return ["a: Alpha", "b: Beta"]


def fake_serialize(state):
    return state


class LayaAdapterTests(unittest.TestCase):
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
        }
        self.rows = [
            {"id": "one", "state": "short", "questions": self.questions},
            {"id": "two", "state": "x" * 1200, "questions": self.questions},
        ]
        self.prompts.write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows),
            encoding="utf-8",
        )
        self.identity = {
            "backend": "laya",
            "model_id": laya.MODEL_ID,
            "model_revision": laya.MODEL_REVISION,
            "revision_attested": True,
            "source_revision": laya.SOURCE_REVISION,
            "model_config_sha256": laya.ARTIFACT_SHA256["rl_agent_config.json"],
            "model_weights_sha256": laya.ARTIFACT_SHA256["model.safetensors"],
        }

    def test_state_truncation_is_detected_without_changing_native_input(self):
        agent = FakeAgent()
        short = laya.truncation_diagnostics(
            agent,
            "short",
            self.questions,
            fake_encode,
            fake_render,
            fake_serialize,
        )
        long = laya.truncation_diagnostics(
            agent,
            "x" * 1200,
            self.questions,
            fake_encode,
            fake_render,
            fake_serialize,
        )
        self.assertFalse(short["any_truncation"])
        self.assertTrue(long["any_truncation"])
        self.assertTrue(long["by_question"]["route"]["state_truncated"])

    def test_native_answer_is_separated_from_invalid_truncated_answer(self):
        runtime = {"runtime_qualification": "unvalidated_rocm"}
        with (
            patch.object(laya, "verify_release", return_value=self.identity),
            patch.object(
                laya,
                "load_native",
                return_value=(
                    FakeAgent(),
                    fake_encode,
                    fake_render,
                    fake_serialize,
                    runtime,
                ),
            ),
            patch.object(laya, "synchronize"),
        ):
            report = laya.collect(
                model_path=self.root,
                source_path=self.root,
                revision=laya.MODEL_REVISION,
                prompts=self.prompts,
                output=self.output,
            )
        predictions = [
            json.loads(line) for line in self.output.read_text().splitlines()
        ]
        self.assertEqual(report["truncated_now"], 1)
        self.assertEqual(predictions[0]["answers"]["route"]["confidence"], 0.2)
        self.assertNotIn("native_answers", predictions[0])
        self.assertTrue(predictions[1]["native_truncation"]["any_truncation"])
        self.assertEqual(
            predictions[1]["answers"]["route"],
            {"type": "choice", "error": "native_input_truncated"},
        )
        self.assertEqual(predictions[1]["native_answers"]["route"]["choice"], "a")
        self.assertEqual(
            evaluate_answer(
                self.questions["route"],
                {"value": "a", "label_to_semantic": {"a": "a", "b": "b"}},
                predictions[1]["answers"]["route"],
            )["status"],
            "invalid",
        )

    def test_only_truncated_question_is_invalidated(self):
        questions = {
            **self.questions,
            "long": {
                "type": "choice",
                "instructions": "x" * 300,
                "criteria": {"a": "Alpha", "b": "Beta"},
            },
        }
        self.prompts.write_text(
            json.dumps({"id": "mixed", "state": "short", "questions": questions})
            + "\n",
            encoding="utf-8",
        )
        with (
            patch.object(laya, "verify_release", return_value=self.identity),
            patch.object(
                laya,
                "load_native",
                return_value=(
                    FakeAgent(),
                    fake_encode,
                    fake_render,
                    fake_serialize,
                    {},
                ),
            ),
            patch.object(laya, "synchronize"),
        ):
            laya.collect(
                model_path=self.root,
                source_path=self.root,
                revision=laya.MODEL_REVISION,
                prompts=self.prompts,
                output=self.output,
            )
        prediction = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(prediction["answers"]["route"]["choice"], "a")
        self.assertEqual(
            prediction["answers"]["long"]["error"], "native_input_truncated"
        )
        self.assertEqual(set(prediction["native_answers"]), {"long"})
        self.assertFalse(
            prediction["native_truncation"]["by_question"]["route"][
                "instructions_truncated"
            ]
        )
        self.assertTrue(
            prediction["native_truncation"]["by_question"]["long"][
                "instructions_truncated"
            ]
        )

    def test_wrong_revision_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact HF revision"):
            laya.verify_release(self.root, self.root, "moving-main")


if __name__ == "__main__":
    unittest.main()
