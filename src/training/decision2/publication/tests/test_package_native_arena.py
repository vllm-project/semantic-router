"""Gold-free, CPU-only contracts for the sealed package JevArena collector."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from jev_arena.jevbench_public import _native_manifest
from publication import package_native_arena as arena

MODEL_ID = "llm-semantic-router/dev-2.0-27b"


def _package_contract() -> dict:
    return {
        "bundle_version": arena.PACKAGE_VERSION,
        "model_id": MODEL_ID,
        "model_sha256": "a" * 64,
        "calibration_sha256": "b" * 64,
        "max_length": 4096,
        "temperature_by_type": {"choice": 0.8, "noul": 1.1, "score": 0.6},
        "loader_files_sha256": {"api.py": "c" * 64, "infer.py": "d" * 64},
        "model_files_sha256": {"decision_config.json": "e" * 64},
        "base": {
            "repo_id": "Qwen/Qwen3.8-27B",
            "revision": "f" * 40,
            "files_sha256": {"config.json": "1" * 64},
        },
        "dependencies": {"torch": "2.12.0", "peft": "0.21.0"},
    }


def _row() -> dict:
    return {
        "id": "one",
        "state": "A review is pending.",
        "questions": {
            "choice": {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {"z": "wait", "a": "approve"},
            },
            "noul": {
                "type": "noul",
                "instructions": "Decide",
                "criteria": {"false": "No", "true": "Yes"},
            },
            "score": {
                "type": "score",
                "instructions": "Rate",
                "criteria": ["Not ready", "Ready"],
            },
        },
    }


class _Model:
    def __init__(self, answers: dict):
        self.answers = answers
        self.received: list[list[str]] = []

    def system_one(self, *, state: object, questions: dict) -> dict:
        self.received.append(list(questions))
        return {
            "model": MODEL_ID,
            "answers": self.answers,
            "usage": {"input_tokens": 25, "output_tokens": 0},
        }


def _validator(item: dict, qid: str, question: object) -> None:
    if not isinstance(question, dict) or "criteria" not in question:
        raise ValueError("Missing explicit criteria")


class PackageNativeArenaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_prompt_preflight_rejects_gold_duplicate_keys_and_nan(self) -> None:
        path = self.root / "prompts.jsonl"
        row = _row()
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.assertEqual(arena.load_gold_free(path), [row])
        self.assertEqual(
            arena.input_digest(row),
            hashlib.sha256(
                json.dumps(
                    {"state": row["state"], "questions": row["questions"]},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        )
        path.write_text(json.dumps({**row, "answer": "z"}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "only id/state/questions"):
            arena.load_gold_free(path)
        path.write_text('{"id":"one","id":"two","state":"s","questions":{}}\n')
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            arena.load_gold_free(path)
        path.write_text('{"id":"one","state":NaN,"questions":{"q":{}}}\n')
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            arena.load_gold_free(path)
        row = _row()
        row["questions"]["choice"]["gold"] = "z"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "question answer field"):
            arena.load_gold_free(path)
        row = _row()
        row["state"] = {"facts": [{"status": "pending"}]}
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.assertEqual(arena.load_gold_free(path), [row])
        row["state"]["facts"][0]["gold_label"] = "approved"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "state answer field"):
            arena.load_gold_free(path)
        row["state"] = ["unsupported"]
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsupported state type"):
            arena.load_gold_free(path)

    def test_sealed_import_uses_package_files_only(self) -> None:
        package = self.root / "bundle"
        code = package / "decision2"
        code.mkdir(parents=True)
        (code / "__init__.py").write_text("from . import api\n", encoding="utf-8")
        (code / "api.py").write_text("VALUE = 3\n", encoding="utf-8")
        (code / "infer.py").write_text("VALUE = 7\n", encoding="utf-8")
        try:
            module, infer = arena._import_sealed_package(package)
            self.assertEqual(module.api.VALUE, 3)
            self.assertEqual(infer.VALUE, 7)
            with self.assertRaisesRegex(RuntimeError, "already imported"):
                arena._import_sealed_package(package)
        finally:
            for name in list(sys.modules):
                if name == "decision2" or name.startswith("decision2."):
                    del sys.modules[name]

    def test_frozen_package_hash_and_dev_revision_are_required(self) -> None:
        package = self.root / "bundle"
        package.mkdir()
        manifest = package / "MODEL_MANIFEST.json"
        manifest.write_text(json.dumps(_package_contract()) + "\n", encoding="utf-8")
        digest = arena._sha_file(manifest)
        contract, actual = arena._package_manifest(
            package, digest, MODEL_ID, f"package-sha256:{digest}"
        )
        self.assertEqual(actual, digest)
        self.assertEqual(contract["max_length"], 4096)
        with self.assertRaisesRegex(ValueError, "frozen SHA"):
            arena._package_manifest(
                package, "0" * 64, MODEL_ID, f"package-sha256:{digest}"
            )
        with self.assertRaisesRegex(ValueError, "Pre-release model revision"):
            arena._package_manifest(package, digest, MODEL_ID, "f" * 40)
        with self.assertRaisesRegex(ValueError, "revision differs"):
            arena._package_manifest(
                package, digest, MODEL_ID, "package-sha256:" + "0" * 64
            )

    def test_collect_preserves_three_native_types_and_standard_receipt(self) -> None:
        row = _row()
        answer = {
            "choice": {
                "type": "choice",
                "choice": "z",
                "probabilities": {"z": 0.7, "a": 0.3},
            },
            "noul": {"type": "noul", "noul": 0.2},
            "score": {
                "type": "score",
                "score": 0.8,
                "probabilities": {"0": 0.2, "1": 0.8},
            },
        }
        model = _Model(answer)
        ticks = iter([1.0, 1.04])
        output, counts = arena.collect(
            [row],
            model=model,
            model_id=MODEL_ID,
            model_revision="f" * 40,
            manifest=_package_contract(),
            package_sha256="2" * 64,
            adapter_sha256="3" * 64,
            question_to_row=_validator,
            clock=lambda: next(ticks),
        )
        self.assertEqual(model.received, [["choice", "noul", "score"]])
        self.assertEqual(counts["valid_questions"], 3)
        self.assertEqual(counts["invalid_questions"], 0)
        self.assertEqual(output[0]["answers"], answer)
        self.assertEqual(output[0]["adapter_status"], "ok")
        self.assertEqual(output[0]["latency_ms"], 40.000000000000036)
        self.assertEqual(output[0]["source_input_sha256"], arena.input_digest(row))
        prompts = self.root / "prompts.jsonl"
        prompts.write_text(json.dumps(row) + "\n", encoding="utf-8")
        receipt = arena.prediction_manifest(
            package=_package_contract(),
            package_sha256="2" * 64,
            model_id=MODEL_ID,
            model_revision="f" * 40,
            input_sha256=arena._sha_file(prompts),
            input_items=1,
            adapter_sha256="3" * 64,
            counts=counts,
        )
        self.assertEqual(receipt["adapter_version"], arena.ADAPTER_VERSION)
        self.assertEqual(receipt["package_manifest_sha256"], "2" * 64)
        self.assertEqual(receipt["calibration_sha256"], "b" * 64)
        self.assertEqual(receipt["model_files_sha256"]["source/config.json"], "1" * 64)
        predictions = self.root / "predictions.jsonl"
        saved = arena.write_predictions(predictions, output, receipt)
        self.assertEqual(saved["predictions_sha256"], arena._sha_file(predictions))
        self.assertEqual(stat.S_IMODE(predictions.stat().st_mode), 0o600)
        self.assertEqual(
            stat.S_IMODE(
                predictions.with_name(predictions.name + ".manifest.json")
                .stat()
                .st_mode
            ),
            0o600,
        )
        self.assertEqual(
            _native_manifest(
                predictions.with_name(predictions.name + ".manifest.json"),
                predictions,
                prompts,
                MODEL_ID,
                "f" * 40,
                1,
            )["adapter_version"],
            arena.ADAPTER_VERSION,
        )
        with self.assertRaises(FileExistsError):
            arena.write_predictions(predictions, output, receipt)

    def test_prediction_output_rejects_nonprivate_directory(self) -> None:
        output = self.root / "visible" / "predictions.jsonl"
        output.parent.mkdir(mode=0o755)
        os.chmod(output.parent, 0o755)
        with self.assertRaisesRegex(ValueError, "mode 0700"):
            arena.write_predictions(output, [], {})
        self.assertFalse(output.exists())

    def test_prediction_files_remain_0600_with_restrictive_umask(self) -> None:
        output = self.root / "private.jsonl"
        previous = os.umask(0o377)
        try:
            arena.write_predictions(output, [], {})
        finally:
            os.umask(previous)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        self.assertEqual(
            stat.S_IMODE(
                output.with_name(output.name + ".manifest.json").stat().st_mode
            ),
            0o600,
        )

    def test_invalid_and_missing_answers_count_as_failure(self) -> None:
        row = _row()
        row["questions"]["missing_criteria"] = {
            "type": "noul",
            "instructions": "Decide",
        }
        model = _Model(
            {
                "choice": {
                    "type": "choice",
                    "choice": None,
                    "probabilities": {"z": 0.5, "a": 0.5},
                },
                "score": {"type": "score", "error": "max_length_exceeded"},
            }
        )
        ticks = iter([1.0, 1.1])
        output, counts = arena.collect(
            [row],
            model=model,
            model_id=MODEL_ID,
            model_revision="f" * 40,
            manifest=_package_contract(),
            package_sha256="2" * 64,
            adapter_sha256="3" * 64,
            question_to_row=_validator,
            clock=lambda: next(ticks),
        )
        self.assertEqual(model.received, [["choice", "noul", "score"]])
        self.assertEqual(counts["questions"], 4)
        self.assertEqual(counts["invalid_questions"], 4)
        self.assertEqual(counts["over_budget_questions"], 1)
        self.assertEqual(output[0]["adapter_status"], "invalid")
        self.assertEqual(
            output[0]["adapter_errors"],
            {
                "choice": "invalid_model_output",
                "noul": "missing_answer",
                "score": "max_length_exceeded",
                "missing_criteria": "invalid_question",
            },
        )
        self.assertEqual(
            output[0]["answers"]["choice"]["error"], "invalid_model_output"
        )

    def test_packaged_api_extra_answers_and_wrong_identity_fail_closed(self) -> None:
        row = _row()
        model = _Model({"unrequested": {"type": "choice"}})
        with self.assertRaisesRegex(ValueError, "extra answers"):
            arena.collect(
                [row],
                model=model,
                model_id=MODEL_ID,
                model_revision="f" * 40,
                manifest=_package_contract(),
                package_sha256="2" * 64,
                adapter_sha256="3" * 64,
                question_to_row=_validator,
            )
        model.system_one = lambda **_: {"model": "wrong", "answers": {}, "usage": {}}
        with self.assertRaisesRegex(ValueError, "another model identity"):
            arena.collect(
                [row],
                model=model,
                model_id=MODEL_ID,
                model_revision="f" * 40,
                manifest=_package_contract(),
                package_sha256="2" * 64,
                adapter_sha256="3" * 64,
                question_to_row=_validator,
            )


if __name__ == "__main__":
    unittest.main()
