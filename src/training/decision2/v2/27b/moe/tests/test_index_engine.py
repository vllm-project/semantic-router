"""CPU-only, gold-free contract checks for the MoE package's Decision Index engine and parity reference."""

from __future__ import annotations

import copy
import gzip
import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

engine_module = importlib.import_module("v2.27b.moe.index_engine")
index_ref = importlib.import_module("v2.27b.moe.index_ref")
ix1_parity = importlib.import_module("v2.eval.ix1.parity")

MODEL_ID = "llm-semantic-router/DEV2.0-27B-MoE-candidate"
MODEL_NAME = "DEV2.0-27B-MoE-candidate"
ARCH = "gemma4-moe-text-endpoints-global-query-shared-bilinear-mlp"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
TEMPERATURES = {"choice": 1.0, "noul": 1.0, "score": 1.0}


def _questions() -> dict:
    return {
        "route": {
            "type": "choice",
            "instructions": "Pick one",
            "criteria": {"z": "wait", "a": "approve", "m": None},
        },
        "ok": {"type": "noul", "instructions": "Is it approved?"},
    }


def _encode(row, tokenizer, max_length):
    ids = list(range(len(json.dumps(row["state"])) + len(row["options"])))
    if len(ids) > max_length:
        raise ValueError(f"prompt exceeds max_length {max_length}")
    return {"ids": ids, "keys": [o["key"] for o in row["options"]]}


class FakePredict:
    """Logits per question, keyed by the option keys (the row's caller order)."""

    def __init__(self, table: dict[tuple, list[float]]):
        self.table = table
        self.calls: list[list[dict]] = []

    def __call__(self, encoded):
        self.calls.append(encoded)
        return [list(self.table[tuple(item["keys"])]) for item in encoded]


class Package:
    def __init__(self, root: Path):
        self.dir = root / "MOE-Git-soup"
        (self.dir / "checkpoint").mkdir(parents=True)
        self.calibration = {"temperature_by_type": TEMPERATURES}
        (self.dir / "calibration.json").write_text(json.dumps(self.calibration))
        self.config = {
            "architecture": ARCH,
            "experts_implementation": "grouped_mm",
            "prompt_version": "decision2-bos-segmented-options-global-query-v1",
            "lora": {"base_revision": REVISION},
        }
        (self.dir / "checkpoint" / "decision_config.json").write_text(
            json.dumps(self.config)
        )
        self.package = {
            "schema": "decision2-27b-moe-package/1",
            "architecture": ARCH,
            "experts_implementation": "grouped_mm",
            "prompt_version": "decision2-bos-segmented-options-global-query-v1",
            "base": {
                "repo": "google/gemma-4-26B-A4B-it",
                "revision": REVISION,
                "tree_sha256": "c" * 64,
            },
            "lora": {"rank": 64, "alpha": 128, "dropout": 0.05},
            "model_sha256": "a" * 64,
            "checkpoint_sha256": "a" * 64,
            "decision": "T = 1",
            "calibration": {
                "path": "/elsewhere/calibration.json",
                "sha256": self.sha(self.dir / "calibration.json"),
                "temperature_by_type": TEMPERATURES,
            },
            "max_input_tokens": 64,
            "loaded_parameters": 25_310_379_550,
            "active_parameters": 3_899_768_350,
        }
        self.write()

    @staticmethod
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write(self) -> str:
        (self.dir / "PACKAGE.json").write_text(json.dumps(self.package, sort_keys=True))
        self.digest = self.sha(self.dir / "PACKAGE.json")
        return self.digest


class AnswerTests(unittest.TestCase):
    def test_exact_choice_ties_go_to_the_callers_first_key(self) -> None:
        answers = {
            "route": {
                "type": "choice",
                "choice": None,
                "probabilities": {"z": 0.4, "a": 0.4, "m": 0.2},
            },
            "ok": {"type": "noul", "noul": 0.3},
        }
        out = engine_module.system_one_answers(answers, _questions())
        self.assertEqual(out["route"]["choice"], "z")
        self.assertIsNone(answers["route"]["choice"])
        self.assertEqual(out["ok"], answers["ok"])
        refused = {"route": {"type": "choice", "error": "max_length_exceeded"}}
        self.assertEqual(
            engine_module.system_one_answers(refused, _questions()), refused
        )

    def test_requests_follow_the_native_run_prompts_path(self) -> None:
        predict = FakePredict(
            {("z", "a", "m"): [0.0, 2.0, -1.0], ("false", "true"): [0.0, math.log(3.0)]}
        )
        state = {"status": "pending"}
        questions = _questions()
        original = copy.deepcopy(questions)
        response = engine_module.answer_request(
            state,
            questions,
            tokenizer=None,
            max_length=64,
            temperatures=TEMPERATURES,
            encode_fn=_encode,
            predict_fn=predict,
            binding={
                "model_sha256": "a" * 64,
                "adapter_sha256": "b" * 64,
                "calibration_sha256": "c" * 64,
            },
            model_name=MODEL_NAME,
        )
        self.assertEqual(questions, original)
        self.assertEqual(len(predict.calls), 1)
        self.assertEqual(len(predict.calls[0]), 2)
        self.assertEqual(response["model"], MODEL_NAME)
        self.assertEqual(response["answers"]["route"]["choice"], "a")
        self.assertAlmostEqual(response["answers"]["ok"]["noul"], 0.75)
        self.assertEqual(set(response["usage"]), {"input_tokens", "output_tokens"})
        engine_module.check_response(response, questions, MODEL_NAME)


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pkg = Package(Path(self.tmp.name))
        self.base = Path(self.tmp.name) / "base"
        self.base.mkdir()
        env = patch.dict(
            os.environ,
            {
                "DECISION2_MOE_PACKAGE_DIR": str(self.pkg.dir),
                "DECISION2_BASE_DIR": str(self.base),
            },
        )
        env.start()
        self.addCleanup(env.stop)
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
        self.fingerprint = patch.object(
            engine_module.infer,
            "checkpoint_fingerprint",
            return_value={"model_sha256": "a" * 64},
        )
        self.fingerprint.start()
        self.addCleanup(self.fingerprint.stop)
        self.calibration = patch.object(
            engine_module,
            "load_calibration",
            return_value=(dict(TEMPERATURES), {"cal_sha256": "d" * 64}),
        )
        self.calibration.start()
        self.addCleanup(self.calibration.stop)
        self.predict = FakePredict(
            {("z", "a", "m"): [0.0, 2.0, -1.0], ("false", "true"): [0.0, 0.0]}
        )
        predict = self.predict

        def fake_load(engine, checkpoint, base, device):
            engine.loaded = (checkpoint, base, device)
            engine.tokenizer = None
            engine.encode_fn = _encode
            engine.predict_fn = predict
            engine.model = object()

        load = patch.object(engine_module.MoEPackageIndexEngine, "_load", fake_load)
        load.start()
        self.addCleanup(load.stop)

    def _engine(self, **overrides):
        options = {
            "model_id": MODEL_ID,
            "package_sha256": self.pkg.digest,
            "device": "cuda:0",
        }
        options.update(overrides)
        return engine_module.MoEPackageIndexEngine(**options)

    def test_loads_the_staged_package_and_answers(self) -> None:
        engine = self._engine()
        checkpoint, base, device = engine.loaded
        self.assertEqual(checkpoint, self.pkg.dir.resolve() / "checkpoint")
        self.assertEqual(base, self.base.resolve())
        self.assertEqual(device, "cuda:0")
        self.assertEqual(engine.max_length, 64)
        response, raw = engine({"status": "pending"}, _questions())
        self.assertIsNone(raw)
        self.assertEqual(response["answers"]["route"]["choice"], "a")
        self.assertEqual(response["answers"]["ok"]["noul"], 0.5)
        self.assertNotIn(self.tmp.name, json.dumps(engine.provenance))
        self.assertEqual(
            engine.provenance["calibration"]["temperature_by_type"], TEMPERATURES
        )

    def test_refusal_is_unsupported_and_bad_outputs_are_errors(self) -> None:
        engine = self._engine()
        with self.assertRaisesRegex(self.unsupported, "max_length_exceeded"):
            engine("x" * 200, _questions())
        bad = _questions()
        bad["route"]["criteria"] = {"only": "one"}
        with self.assertRaisesRegex(self.unsupported, "invalid_question"):
            engine("pending", bad)
        self.predict.table[("z", "a", "m")] = [0.0, math.nan, 1.0]
        with self.assertRaises(ValueError) as caught:
            engine("pending", _questions())
        self.assertNotIsInstance(caught.exception, self.unsupported)

    def test_pins_are_enforced_before_loading(self) -> None:
        for overrides in (
            {"device": "cpu"},
            {"package_sha256": "unpinned"},
            {"model_id": ""},
        ):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                ValueError, "required"
            ):
                self._engine(**overrides)
        with self.assertRaisesRegex(ValueError, "pinned digest"):
            self._engine(package_sha256="0" * 64)

    def test_package_binding_fails_closed(self) -> None:
        def case(change):
            pkg = Package(Path(tempfile.mkdtemp(dir=self.tmp.name)))
            change(pkg)
            return pkg

        cases = {
            "calibration": case(
                lambda p: (p.dir / "calibration.json").write_text("{}")
            ),
            "experts": case(
                lambda p: (p.package.update(experts_implementation="eager"), p.write())
            ),
            "architecture": case(
                lambda p: (p.package.update(architecture="qwen3.5-moe-x"), p.write())
            ),
            "revision": case(
                lambda p: (p.package["base"].update(revision="0" * 40), p.write())
            ),
        }
        for label, pkg in cases.items():
            with self.subTest(case=label), patch.dict(
                os.environ, {"DECISION2_MOE_PACKAGE_DIR": str(pkg.dir)}
            ):
                with self.assertRaises(ValueError):
                    self._engine(package_sha256=pkg.digest)
        self.fingerprint.stop()
        with patch.object(
            engine_module.infer,
            "checkpoint_fingerprint",
            return_value={"model_sha256": "e" * 64},
        ):
            with self.assertRaisesRegex(ValueError, "model_sha256"):
                self._engine()
        self.fingerprint.start()
        self.calibration.stop()
        with patch.object(
            engine_module,
            "load_calibration",
            return_value=({**TEMPERATURES, "noul": 2.0}, {}),
        ):
            with self.assertRaisesRegex(ValueError, "temperatures"):
                self._engine()
        self.calibration.start()

    def test_warmup_and_close(self) -> None:
        self.predict.table[("red", "blue")] = [1.0, 0.0]
        engine = self._engine()
        engine.warmup()
        self.assertEqual(len(self.predict.calls), 2)
        engine.close()
        self.assertIsNone(engine.model)


class ReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rows = [
            {
                "_evaluation": {"run_id": f"r{i}", "catalog_id": 1},
                "state": f"state {i}",
                "questions": _questions(),
            }
            for i in range(3)
        ]
        self.path = Path(self.tmp.name) / "rows.jsonl.gz"
        with gzip.open(self.path, "wt", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(row) + "\n")

    def test_prompts_are_gold_free_and_keyed_by_run_id(self) -> None:
        prompts = index_ref.prompts(index_ref.read_rows(self.path))
        self.assertEqual([p["id"] for p in prompts], ["r0", "r1", "r2"])
        self.assertEqual(set(prompts[0]), {"id", "state", "questions"})

    def test_convert_matches_the_kit_side_under_the_ix1_parity_gate(self) -> None:
        tie = {
            "type": "choice",
            "choice": None,
            "probabilities": {"z": 0.4, "a": 0.4, "m": 0.2},
        }
        predictions = [
            {
                "id": "r0",
                "answers": {"route": tie, "ok": {"type": "noul", "noul": 0.3}},
                "latency_ms": 5.0,
            },
            {
                "id": "r1",
                "answers": {
                    "route": {"type": "choice", "error": "max_length_exceeded"},
                    "ok": {"type": "noul", "error": "max_length_exceeded"},
                },
                "latency_ms": 1.0,
            },
            {
                "id": "r2",
                "answers": {
                    "route": {"type": "choice", "error": "invalid_model_output"},
                    "ok": {"type": "noul", "noul": 0.5},
                },
                "latency_ms": 2.0,
            },
        ]
        ref = index_ref.convert(self.rows, predictions)
        self.assertEqual([r["status"] for r in ref], ["ok", "unsupported", "error"])
        self.assertEqual(ref[0]["answers"]["route"]["choice"], "z")
        kit = {
            "r0": {
                "run_id": "r0",
                "status": "ok",
                "response": {"answers": ref[0]["answers"]},
            },
            "r1": {"run_id": "r1", "status": "unsupported"},
        }
        report = ix1_parity.compare(kit, {r["run_id"]: r for r in ref[:2]})
        self.assertTrue(report["pass"], report)
        with self.assertRaisesRegex(ValueError, "different run IDs"):
            index_ref.convert(self.rows, predictions[:2])


if __name__ == "__main__":
    unittest.main()
