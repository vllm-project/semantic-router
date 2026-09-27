"""CPU-only, gold-free contract checks for the Decision Index native bridge."""

from __future__ import annotations

import math
import sys
import types
import unittest
from unittest.mock import patch

from publication import decision_index_native_engine as bridge

MODEL = "llm-semantic-router/DEV2.0-27B"
DIGEST = "a" * 64


def _questions() -> dict:
    return {
        "choice": {
            "type": "choice",
            "instructions": "Pick one",
            "criteria": {"z": "wait", "a": "approve"},
        },
        "noul": {"type": "noul", "instructions": "Is it approved?"},
    }


def _response() -> dict:
    return {
        "model": MODEL,
        "answers": {
            "choice": {
                "type": "choice",
                "choice": "z",
                "probabilities": {"z": 0.7, "a": 0.3},
            },
            "noul": {"type": "noul", "noul": 0.5},
        },
        "usage": {"input_tokens": 42, "output_tokens": 0},
    }


def _native_row(item: dict, key: str, question: dict) -> dict:
    if not isinstance(question.get("instructions"), str):
        raise ValueError("instructions must be a string")
    criteria = question.get("criteria")
    if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
        raise ValueError("unsupported option count")
    if any(
        not isinstance(k, str) or not k or not isinstance(v, str)
        for k, v in criteria.items()
    ):
        raise ValueError("option descriptions must be strings")
    if question["type"] == "noul" and set(criteria) != {"false", "true"}:
        raise ValueError("invalid Noul criteria")
    return {"id": item["id"] + "/" + key}


class _FakeModel:
    def __init__(self, answer: dict | None = None):
        self.answer = answer or _response()
        self.calls: list[tuple[object, object]] = []
        self.device = types.SimpleNamespace(type="cpu")
        self.torch = types.SimpleNamespace(cuda=types.SimpleNamespace())

    def system_one(self, *, state: object, questions: dict) -> dict:
        self.calls.append((state, questions))
        return self.answer


class NativeIndexEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _FakeModel()
        self.contract = {
            "model_sha256": "b" * 64,
            "calibration_sha256": "c" * 64,
            "base": {"repo_id": "Qwen/Qwen3.8-27B", "revision": "d" * 40},
            "max_length": 4096,
            "parameter_count": 25_688_227_840,
            "temperature_by_type": {"choice": 0.8, "noul": 1.1, "score": 0.6},
        }
        self.load = patch.object(
            bridge,
            "_load_native",
            return_value=(self.model, self.contract, _native_row),
        )
        self.load.start()
        self.addCleanup(self.load.stop)
        upstream = types.ModuleType("decision_index")
        engines = types.ModuleType("decision_index.engines")
        engines.Unsupported = type("Unsupported", (ValueError,), {})
        upstream.engines = engines
        self.modules = patch.dict(
            sys.modules, {"decision_index": upstream, "decision_index.engines": engines}
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.unsupported = engines.Unsupported

    def _engine(self) -> bridge.NativeDecisionIndexEngine:
        return bridge.NativeDecisionIndexEngine(
            model_id=MODEL, package_manifest_sha256=DIGEST, device="cuda:5"
        )

    def test_unmodified_choice_and_noul_use_native_package(self) -> None:
        engine = self._engine()
        state = {"status": "pending"}
        questions = _questions()
        response, raw = engine(state, questions)
        self.assertIs(response, self.model.answer)
        self.assertIsNone(raw)
        self.assertIs(self.model.calls[0][0], state)
        self.assertIs(self.model.calls[0][1], questions)
        self.assertEqual(
            response["answers"]["choice"]["probabilities"], {"z": 0.7, "a": 0.3}
        )
        self.assertEqual(response["answers"]["noul"]["noul"], 0.5)
        self.assertEqual(engine.runtime()["native_max_length"], 4096)
        self.assertEqual(engine.runtime()["native_parameter_count"], 25_688_227_840)
        self.assertNotIn("DECISION2_BASE_DIR", repr(engine.provenance))
        self.assertEqual(engine.provenance["calibration_sha256"], "c" * 64)

    def test_input_capacity_rejects_entire_request_before_inference(self) -> None:
        engine = self._engine()
        examples = []
        for change in (
            "score",
            "structured_instructions",
            "structured_option",
            "too_many",
        ):
            questions = _questions()
            if change == "score":
                questions["choice"]["type"] = "score"
            elif change == "structured_instructions":
                questions["choice"]["instructions"] = {"task": "Choose"}
            elif change == "structured_option":
                questions["choice"]["criteria"]["z"] = {"text": "wait"}
            else:
                questions["choice"]["criteria"] = {str(i): str(i) for i in range(256)}
            examples.append(questions)
        for questions in examples:
            with self.subTest(questions=repr(questions)[:40]):
                with self.assertRaises(self.unsupported):
                    engine("pending", questions)
        with self.assertRaises(self.unsupported):
            engine(["pending"], _questions())
        self.assertEqual(self.model.calls, [])

    def test_native_context_limit_is_unsupported_without_partial_answer(self) -> None:
        response = _response()
        response["answers"]["noul"] = {
            "type": "noul",
            "error": "max_length_exceeded",
        }
        self.model.answer = response
        with self.assertRaisesRegex(self.unsupported, "native_max_length_exceeded"):
            self._engine()("pending", _questions())
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(list(self.model.calls[0][1]), ["choice", "noul"])

    def test_malformed_native_outputs_fail_closed(self) -> None:
        cases = []
        response = _response()
        del response["answers"]["noul"]
        cases.append(response)
        response = _response()
        response["model"] = "another-model"
        cases.append(response)
        response = _response()
        response["answers"]["choice"]["probabilities"] = {"z": 1.0}
        cases.append(response)
        response = _response()
        response["answers"]["choice"]["probabilities"] = {"z": 0.9, "a": 0.9}
        cases.append(response)
        response = _response()
        response["answers"]["choice"]["probabilities"] = {"z": math.nan, "a": 0.3}
        cases.append(response)
        response = _response()
        response["answers"]["choice"]["choice"] = "a"
        cases.append(response)
        response = _response()
        response["answers"]["noul"]["noul"] = -0.1
        cases.append(response)
        response = _response()
        response["usage"]["input_tokens"] = True
        cases.append(response)
        response = _response()
        response["answers"]["choice"]["error"] = "invalid_model_output"
        cases.append(response)
        for response in cases:
            with self.subTest(response=repr(response)):
                self.model.answer = response
                with self.assertRaises(ValueError):
                    self._engine()("pending", _questions())

    def test_configuration_requires_pinned_package_and_explicit_gpu(self) -> None:
        for digest, device in (("unpinned", "cuda:0"), (DIGEST, "cpu")):
            with self.assertRaisesRegex(ValueError, "pinned package digest"):
                bridge.NativeDecisionIndexEngine(
                    model_id=MODEL,
                    package_manifest_sha256=digest,
                    device=device,
                )
        engine = self._engine()
        self.model.answer = {
            "model": MODEL,
            "answers": {
                "warmup": {
                    "type": "choice",
                    "choice": "red",
                    "probabilities": {"red": 0.7, "blue": 0.3},
                }
            },
            "usage": {"input_tokens": 8, "output_tokens": 0},
        }
        engine.warmup()
        self.assertEqual(len(self.model.calls), 1)
        engine.close()
        self.assertIsNone(engine.model)


if __name__ == "__main__":
    unittest.main()
