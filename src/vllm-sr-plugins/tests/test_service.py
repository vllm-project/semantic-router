from __future__ import annotations

import asyncio
import math
import unittest
from pathlib import Path

from support import has

HAVE_TORCH = has("torch") and has("transformers")


class CharTokenizer:
    """One token per character; enough for the segment-wise encoder contract."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [2 + ord(ch) % 997 for ch in text]


def fake_package(*, cap=4096, temperatures=None, score_bias=None):
    from training.model import data, decision_model, infer
    from training.model import score_bias as bias

    from vllm_sr_plugins.decision2.package import Decision2Package, Runtime

    return Decision2Package(
        root=Path("/nonexistent"),
        manifest={"profile": "qwen-full", "files_sha256": {}},
        manifest_sha256="0" * 64,
        model_name="DEV2.0-test",
        max_input_tokens=cap,
        model_sha256="1" * 64,
        temperatures=temperatures or {"choice": 1.0, "noul": 1.0, "score": 1.0},
        score_bias=score_bias,
        tokenizer=CharTokenizer(),
        runtime=Runtime(
            question_to_row=infer.question_to_row,
            encode=decision_model.encode,
            product_answer=infer.product_answer,
            api_json_payload=infer._api_json_payload,
            canonical=data.canonical,
            apply_score_bias=bias.apply,
        ),
    )


class RecordingEngine:
    """Logit k for candidate k, so answers are predictable; records every call."""

    def __init__(self):
        self.calls = []

    async def __call__(self, ids, positions, request_id):
        self.calls.append((ids, positions, request_id))
        count = len(positions["decision2"]["candidate_positions"])
        return [float(k) for k in range(count)]


QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Pick the model",
        "criteria": {"small": "fast", "large": "careful", "none": None},
    },
    "check": {"type": "noul", "instructions": "Is it urgent?"},
    "grade": {
        "type": "score",
        "instructions": "Rate it",
        "criteria": ["bad", "ok", "good"],
    },
}


@unittest.skipUnless(HAVE_TORCH, "needs torch and transformers")
class SystemOneServiceTest(unittest.TestCase):
    def run_request(self, package, questions, state="ticket: printer on fire"):
        from vllm_sr_plugins.decision2.service import SystemOneService

        engine = RecordingEngine()
        service = SystemOneService(package, engine)
        result = asyncio.run(
            service.system_one(state=state, questions=questions, request_id="r")
        )
        return result, engine

    def test_answers_follow_the_runtime_contract(self) -> None:
        from training.model import decision_model, infer

        package = fake_package()
        result, engine = self.run_request(package, QUESTIONS)
        self.assertEqual(result["model"], "DEV2.0-test")
        self.assertEqual(list(result["answers"]), ["route", "check", "grade"])
        item = {"id": "request", "state": "ticket: printer on fire"}
        tokens = 0
        for (qid, question), (ids, positions, request_id) in zip(
            QUESTIONS.items(), engine.calls
        ):
            row = infer.question_to_row(item, qid, question)
            encoded = decision_model.encode(row, CharTokenizer(), 4096)
            tokens += len(encoded["ids"])
            self.assertEqual(ids, encoded["ids"])
            self.assertEqual(
                positions["decision2"],
                {
                    "candidate_positions": encoded["candidate_positions"],
                    "query_position": encoded["query_position"],
                },
            )
            expected = infer.product_answer(
                row["task_type"],
                encoded["keys"],
                [float(k) for k in range(len(encoded["keys"]))],
                1.0,
                [option["description"] for option in row["options"]],
            )
            self.assertEqual(result["answers"][qid], expected)
        self.assertEqual(result["usage"], {"input_tokens": tokens, "output_tokens": 0})
        self.assertEqual(result["answers"]["route"]["choice"], "none")
        self.assertEqual([c[2] for c in engine.calls], ["r-0", "r-1", "r-2"])

    def test_temperatures_and_score_offsets_apply_per_type(self) -> None:
        package = fake_package(
            temperatures={"choice": 2.0, "noul": 1.0, "score": 0.5},
            score_bias={3: [3.0, 0.0, -3.0]},
        )
        result, _ = self.run_request(package, QUESTIONS)
        probabilities = result["answers"]["route"]["probabilities"]
        weights = [math.exp(k / 2.0) for k in range(3)]
        self.assertAlmostEqual(
            probabilities["none"], weights[2] / sum(weights), places=12
        )
        grade = result["answers"]["grade"]["probabilities"]
        # Offsets are added before the temperature: logits become 3, 1, -1.
        weights = [math.exp(v / 0.5) for v in (3.0, 1.0, -1.0)]
        self.assertAlmostEqual(grade["0"], weights[0] / sum(weights), places=12)

    def test_invalid_and_over_budget_questions_are_answered_as_errors(self) -> None:
        questions = {
            "bad": {"type": "rank", "instructions": "x"},
            "long": {"type": "noul", "instructions": "y" * 5000},
            "ok": {"type": "noul", "instructions": "Is it urgent?"},
        }
        result, engine = self.run_request(fake_package(cap=4096), questions)
        self.assertEqual(
            result["answers"]["bad"], {"type": "rank", "error": "invalid_question"}
        )
        self.assertEqual(
            result["answers"]["long"], {"type": "noul", "error": "max_length_exceeded"}
        )
        self.assertIn("noul", result["answers"]["ok"])
        self.assertEqual(len(engine.calls), 1)

    def test_wrong_logit_count_is_invalid_model_output(self) -> None:
        from vllm_sr_plugins.decision2.service import SystemOneService

        async def short(ids, positions, request_id):
            return [0.0]

        service = SystemOneService(fake_package(), short)
        result = asyncio.run(
            service.system_one(
                state="s", questions={"check": QUESTIONS["check"]}, request_id="r"
            )
        )
        self.assertEqual(
            result["answers"]["check"],
            {"type": "noul", "error": "invalid_model_output"},
        )

    def test_nonfinite_logits_are_invalid_model_output(self) -> None:
        from vllm_sr_plugins.decision2.service import SystemOneService

        async def nan(ids, positions, request_id):
            return [float("nan")] * len(positions["decision2"]["candidate_positions"])

        service = SystemOneService(fake_package(), nan)
        result = asyncio.run(
            service.system_one(
                state="s", questions={"check": QUESTIONS["check"]}, request_id="r"
            )
        )
        self.assertEqual(result["answers"]["check"]["error"], "invalid_model_output")

    def test_request_level_validation_raises(self) -> None:
        from vllm_sr_plugins.decision2.service import SystemOneService

        service = SystemOneService(fake_package(), RecordingEngine())
        for state, questions in [
            ("s", {}),
            ("s", {"": QUESTIONS["check"]}),
            ("s", ["not", "a", "mapping"]),
            (7, {"check": QUESTIONS["check"]}),
            ({"x": float("inf")}, {"check": QUESTIONS["check"]}),
        ]:
            with self.assertRaises(ValueError):
                asyncio.run(
                    service.system_one(state=state, questions=questions, request_id="r")
                )


if __name__ == "__main__":
    unittest.main()
