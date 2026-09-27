"""CPU-only System One product contract for future full-checkpoint bundles."""

from __future__ import annotations

import math
import sys
import types
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from publication.adapter_runtime import Decision2 as AdapterDecision2
from publication.full_bundle import MODEL_SOURCES
from publication.full_runtime_api import Decision2
from publication.runtime_api import Decision2 as LegacyBundleDecision2
from training.model import infer


class _Vector:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return _Vector(self.values[index])

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FullSystemOneContractTest(unittest.TestCase):
    def test_packager_copies_the_same_infer_used_by_product_runtime(self):
        self.assertIn("infer.py", MODEL_SOURCES)

    def test_structured_multiquestion_response_and_no_truncation(self):
        model_sources = types.ModuleType("publication.decision_model")

        def encode(row, _tokenizer, _max_length):
            if row["instructions"] == "over budget":
                raise ValueError("101 tokens exceeds max_length=100; no truncation")
            return {
                "id": row["id"],
                "ids": [1, 2, 3],
                "keys": [option["key"] for option in row["options"]],
            }

        model_sources.encode = encode
        model_sources.collate = lambda encoded, _pad: {"items": encoded}

        def predict(*, items):
            values = {
                "c": [0.0, 0.0],
                "n": [-1.0, 1.0],
                "s": [0.0, 1.0, 2.0],
            }
            return [_Vector(values[row["id"].rsplit("/", 1)[-1]]) for row in items]

        runtime = Decision2(
            model=predict,
            tokenizer=types.SimpleNamespace(pad_token_id=0, eos_token_id=None),
            device=types.SimpleNamespace(type="cpu"),
            manifest={
                "model_id": "test/decision-model",
                "max_length": 100,
                "temperature_by_type": {"choice": 1.0, "noul": 1.0, "score": 1.0},
            },
            torch=types.SimpleNamespace(
                is_tensor=lambda _: False, inference_mode=nullcontext
            ),
        )
        questions = {
            "c": {
                "type": "choice",
                "instructions": {"question": "Route this?", "priority": [1, 2]},
                "criteria": {"first": None, "second": {"rubric": "alternate"}},
            },
            "n": {
                "type": "noul",
                "instructions": ["Is this urgent?"],
                "criteria": {"true": ["urgent"]},
            },
            "s": {
                "type": "score",
                "instructions": {"question": "How urgent?"},
                "criteria": ["Low", {"level": "Medium"}, ["High"]],
            },
            "long": {
                "type": "choice",
                "instructions": "over budget",
                "criteria": {"first": None, "second": None},
            },
        }
        with patch.dict(
            sys.modules,
            {
                "publication.decision_model": model_sources,
                "publication.infer": infer,
            },
        ):
            response = runtime.system_one(
                state={"records": [{"status": "failed"}]}, questions=questions
            )
            array_response = runtime.system_one(
                state=[{"status": "failed"}], questions={"n": questions["n"]}
            )
        self.assertEqual(response["model"], "test/decision-model")
        self.assertEqual(response["usage"], {"input_tokens": 9, "output_tokens": 0})
        self.assertEqual(response["answers"]["c"]["choice"], "first")
        self.assertEqual(
            response["answers"]["c"]["probabilities"], {"first": 0.5, "second": 0.5}
        )
        self.assertEqual(response["answers"]["c"]["confidence"], 0.0)
        self.assertGreater(response["answers"]["n"]["noul"], 0.8)
        self.assertNotIn("confidence", response["answers"]["n"])
        score = response["answers"]["s"]
        self.assertEqual(
            score["legend"], {"0": "Low", "1": '{"level":"Medium"}', "2": '["High"]'}
        )
        self.assertAlmostEqual(
            score["score"],
            sum(int(key) * value for key, value in score["probabilities"].items()),
        )
        self.assertTrue(math.isfinite(score["confidence"]))
        self.assertEqual(
            response["answers"]["long"],
            {"type": "choice", "error": "max_length_exceeded"},
        )
        self.assertEqual(array_response["answers"]["n"]["type"], "noul")

    def test_invalid_state_type_is_rejected_before_model_use(self):
        runtime = Decision2(
            model=None,
            tokenizer=None,
            device=None,
            manifest={"max_length": 8, "temperature_by_type": {}},
            torch=None,
        )
        with patch.dict(
            sys.modules,
            {
                "publication.decision_model": types.ModuleType(
                    "publication.decision_model"
                ),
                "publication.infer": infer,
            },
        ):
            with self.assertRaisesRegex(ValueError, "state must be"):
                runtime.system_one(state=42, questions={"q": {"type": "noul"}})
            with self.assertRaisesRegex(ValueError, "state must be"):
                runtime.system_one(
                    state={1: "non-JSON key"}, questions={"q": {"type": "noul"}}
                )

    def test_all_package_paths_return_the_same_system_one_shape(self):
        model_sources = types.ModuleType("publication.decision_model")
        model_sources.encode = lambda row, _tokenizer, _limit: {
            "id": row["id"],
            "ids": [1, 2],
            "keys": [option["key"] for option in row["options"]],
        }
        model_sources.collate = lambda encoded, _pad: {"items": encoded}

        def predict(*, items):
            values = {
                "choice": [0.0, 0.0],
                "noul": [-1.0, 1.0],
                "score": [0.0, 1.0, 2.0],
            }
            return [_Vector(values[row["id"].rsplit("/", 1)[-1]]) for row in items]

        manifest = {
            "model_id": "test/decision-model",
            "max_length": 100,
            "temperature_by_type": {"choice": 1.0, "noul": 1.0, "score": 1.0},
        }
        runtime_args = {
            "model": predict,
            "tokenizer": types.SimpleNamespace(pad_token_id=0, eos_token_id=None),
            "device": types.SimpleNamespace(type="cpu"),
            "manifest": manifest,
            "torch": types.SimpleNamespace(
                is_tensor=lambda _: False, inference_mode=nullcontext
            ),
        }
        questions = {
            "choice": {
                "type": "choice",
                "instructions": "Where to route?",
                "criteria": {"first": None, "second": "Other"},
            },
            "noul": {"type": "noul", "instructions": "Is this urgent?"},
            "score": {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": ["Low", "Medium", "High"],
            },
        }
        with patch.dict(
            sys.modules,
            {"publication.decision_model": model_sources, "publication.infer": infer},
        ):
            full = Decision2(**runtime_args).system_one(
                state=[{"status": "failed"}], questions=questions
            )
            adapter = AdapterDecision2(**runtime_args).system_one(
                state=[{"status": "failed"}], questions=questions
            )
            legacy = LegacyBundleDecision2(**runtime_args).system_one(
                state=[{"status": "failed"}], questions=questions
            )
            with self.assertRaisesRegex(ValueError, "state must be"):
                AdapterDecision2(**runtime_args).system_one(
                    state=42, questions=questions
                )
        self.assertEqual(adapter, full)
        self.assertEqual(legacy, full)
        self.assertIn("confidence", adapter["answers"]["choice"])
        self.assertIn("legend", adapter["answers"]["score"])


if __name__ == "__main__":
    unittest.main()
