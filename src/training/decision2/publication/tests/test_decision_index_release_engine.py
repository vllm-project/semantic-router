"""CPU-only, gold-free contract checks for the released-package Decision Index engine."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from publication import decision_index_release_engine as engine_module

MODEL_ID = "llm-semantic-router/DEV2.0-4B"
MODEL_NAME = "DEV2.0-4B"
REVISION = "f" * 40

FAKE_RUNTIME = textwrap.dedent(
    """
    import json
    import types
    from pathlib import Path

    CALLS = []

    class _Backend:
        def __init__(self):
            self.device = types.SimpleNamespace(type="cpu")
            self.temperatures = {"choice": 1.0, "noul": 1.0, "score": 1.0}
            self.torch = types.SimpleNamespace(
                __version__="0", version=types.SimpleNamespace(hip=None)
            )

    class Decision2:
        answer = None

        def __init__(self, manifest):
            self.manifest = manifest
            self.backend = _Backend()

        @classmethod
        def from_pretrained(cls, path, *, device=None, base_path=None, threads=None):
            CALLS.append({"path": str(path), "device": device, "base_path": base_path})
            return cls(json.loads((Path(path) / "MODEL_MANIFEST.json").read_text()))

        def system_one(self, *, state, questions):
            CALLS.append({"state": state, "questions": questions})
            return Decision2.answer
    """
)


def _questions() -> dict:
    return {
        "route": {
            "type": "choice",
            "instructions": "Pick one",
            "criteria": {"z": "wait", "a": "approve", "m": None},
        },
        "ok": {"type": "noul", "instructions": "Is it approved?"},
    }


def _response() -> dict:
    return {
        "model": MODEL_NAME,
        "answers": {
            "route": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"z": 0.2, "a": 0.7, "m": 0.1},
                "confidence": 0.3,
            },
            "ok": {"type": "noul", "noul": 0.25},
        },
        "usage": {"input_tokens": 42, "output_tokens": 0},
    }


def _drop_runtime() -> None:
    for name in list(sys.modules):
        if name == "decision2" or name.startswith("decision2."):
            del sys.modules[name]


class CheckResponseTests(unittest.TestCase):
    def test_valid_response_is_returned_unchanged(self) -> None:
        response = _response()
        self.assertIs(
            engine_module.check_response(response, _questions(), MODEL_NAME), response
        )

    def test_exact_ties_resolve_to_the_first_key_in_caller_order(self) -> None:
        response = _response()
        response["answers"]["route"]["probabilities"] = {"z": 0.4, "a": 0.4, "m": 0.2}
        response["answers"]["route"]["choice"] = "z"
        engine_module.check_response(response, _questions(), MODEL_NAME)
        response["answers"]["route"]["choice"] = "a"
        with self.assertRaisesRegex(ValueError, "argmax"):
            engine_module.check_response(response, _questions(), MODEL_NAME)

    def test_package_refusals_are_request_level(self) -> None:
        for reason in engine_module.REFUSALS:
            response = _response()
            response["answers"]["ok"] = {"type": "noul", "error": reason}
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(engine_module.PackageRefusal, reason):
                    engine_module.check_response(response, _questions(), MODEL_NAME)

    def test_malformed_outputs_fail_closed(self) -> None:
        def case(change):
            response = _response()
            change(response)
            return response

        cases = [
            case(lambda r: r["answers"].pop("ok")),
            case(lambda r: r.update(model="DEV2.0-9B")),
            case(lambda r: r["answers"]["route"]["probabilities"].pop("m")),
            case(lambda r: r["answers"]["route"]["probabilities"].update(z=0.3)),
            case(lambda r: r["answers"]["route"]["probabilities"].update(z=math.nan)),
            case(lambda r: r["answers"]["route"].update(choice="z")),
            case(lambda r: r["answers"]["ok"].update(noul=1.5)),
            case(lambda r: r["answers"]["ok"].update(type="choice")),
            case(lambda r: r["usage"].update(input_tokens=True)),
            case(lambda r: r["answers"]["route"].update(error="invalid_model_output")),
        ]
        for response in cases:
            with self.subTest(response=repr(response)[:80]):
                with self.assertRaises(ValueError) as caught:
                    engine_module.check_response(response, _questions(), MODEL_NAME)
                self.assertNotIsInstance(caught.exception, engine_module.PackageRefusal)


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        _drop_runtime()
        self.addCleanup(_drop_runtime)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.package = Path(self.tmp.name) / "DEV2.0-4B"
        (self.package / "decision2").mkdir(parents=True)
        (self.package / "decision2" / "__init__.py").write_text(FAKE_RUNTIME)
        self.manifest = {
            "repo_id": MODEL_ID,
            "model_name": MODEL_NAME,
            "profile": "qwen-full",
            "identity": {"model_sha256": "b" * 64},
            "base": None,
            "calibration": None,
            "max_input_tokens": 32768,
            "parameters": {"loaded": 4_000_000_000},
        }
        self._write_manifest()
        self.env = patch.dict(os.environ, {"DECISION2_PACKAGE_DIR": str(self.package)})
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("DECISION2_BASE_DIR", None)
        upstream = types.ModuleType("decision_index")
        engines = types.ModuleType("decision_index.engines")
        engines.Unsupported = type("Unsupported", (ValueError,), {})
        upstream.engines = engines
        modules = patch.dict(
            sys.modules, {"decision_index": upstream, "decision_index.engines": engines}
        )
        modules.start()
        self.addCleanup(modules.stop)
        self.unsupported = engines.Unsupported

    def _write_manifest(self) -> None:
        data = json.dumps(self.manifest, sort_keys=True).encode()
        (self.package / "MODEL_MANIFEST.json").write_bytes(data)
        self.digest = hashlib.sha256(data).hexdigest()

    def _engine(self, **overrides) -> engine_module.ReleasedDecisionIndexEngine:
        options = {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "package_manifest_sha256": self.digest,
            "device": "cuda:0",
        }
        options.update(overrides)
        return engine_module.ReleasedDecisionIndexEngine(**options)

    def test_requests_reach_the_package_runtime_unchanged(self) -> None:
        engine = self._engine()
        runtime = sys.modules["decision2"]
        self.assertEqual(
            Path(runtime.__file__).resolve(),
            (self.package / "decision2" / "__init__.py").resolve(),
        )
        runtime.Decision2.answer = _response()
        state = {"status": "pending"}
        questions = _questions()
        original = copy.deepcopy(questions)
        response, raw = engine(state, questions)
        self.assertIs(response, runtime.Decision2.answer)
        self.assertIsNone(raw)
        call = runtime.CALLS[-1]
        self.assertIs(call["state"], state)
        self.assertIs(call["questions"], questions)
        self.assertEqual(questions, original)
        self.assertEqual(runtime.CALLS[0]["device"], "cuda:0")
        self.assertIsNone(runtime.CALLS[0]["base_path"])
        self.assertEqual(engine.provenance["revision"], REVISION)
        self.assertNotIn(str(self.package), json.dumps(engine.provenance))
        self.assertEqual(engine.runtime()["native_max_input_tokens"], 32768)

    def test_refusal_is_unsupported_and_defects_are_errors(self) -> None:
        engine = self._engine()
        runtime = sys.modules["decision2"]
        response = _response()
        response["answers"]["ok"] = {"type": "noul", "error": "max_length_exceeded"}
        runtime.Decision2.answer = response
        with self.assertRaisesRegex(self.unsupported, "max_length_exceeded"):
            engine("pending", _questions())
        response = _response()
        response["answers"]["route"]["choice"] = "z"
        runtime.Decision2.answer = response
        with self.assertRaises(ValueError) as caught:
            engine("pending", _questions())
        self.assertNotIsInstance(caught.exception, self.unsupported)

    def test_pins_are_enforced_before_loading(self) -> None:
        for overrides in (
            {"package_manifest_sha256": "0" * 64},
            {"model_id": "llm-semantic-router/DEV2.0-9B"},
        ):
            with self.subTest(overrides=overrides):
                _drop_runtime()
                with self.assertRaises(ValueError):
                    self._engine(**overrides)
        for overrides in (
            {"revision": "main"},
            {"device": "cpu"},
            {"package_manifest_sha256": "unpinned"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "pinned revision"):
                    self._engine(**overrides)

    def test_a_second_runtime_import_is_refused(self) -> None:
        self._engine()
        with self.assertRaisesRegex(RuntimeError, "already imported"):
            engine_module.import_package_runtime(self.package)

    def test_warmup_and_close(self) -> None:
        engine = self._engine()
        runtime = sys.modules["decision2"]
        runtime.Decision2.answer = {
            "model": MODEL_NAME,
            "answers": {
                "warmup": {
                    "type": "choice",
                    "choice": "red",
                    "probabilities": {"red": 0.9, "blue": 0.1},
                }
            },
            "usage": {"input_tokens": 8, "output_tokens": 0},
        }
        engine.warmup()
        self.assertEqual(len(runtime.CALLS), 3)
        engine.close()
        self.assertIsNone(engine.model)


if __name__ == "__main__":
    unittest.main()
