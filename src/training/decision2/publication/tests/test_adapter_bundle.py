"""CPU contract tests for the unmerged, external-base PEFT package."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from publication import adapter_bundle, adapter_parity, adapter_runtime
from training.model.data import canonical
from training.model.infer import checkpoint_fingerprint
from training.model.source import source_fingerprint


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )


def _weights(path: Path, shapes: dict[str, list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {}
    cursor = 0
    for name, shape in shapes.items():
        width = 4
        for dim in shape:
            width *= dim
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [cursor, cursor + width],
        }
        cursor += width
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * cursor)


class AdapterBundleTests(unittest.TestCase):
    def test_public_model_id_family(self) -> None:
        for size in ("0.6B", "0.8B", "2B", "4B", "9B", "27B"):
            self.assertIsNotNone(
                adapter_bundle.MODEL_ID.fullmatch(f"llm-semantic-router/DEV2.0-{size}")
            )
        self.assertIsNone(
            adapter_bundle.MODEL_ID.fullmatch("llm-semantic-router/dev-2.0-4b")
        )
        self.assertIsNone(
            adapter_bundle.MODEL_ID.fullmatch("llm-semantic-router/DEV2.0-8B")
        )

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "upstream"
        self.source.mkdir()
        _json(self.source / "config.json", {"model_type": "qwen3_5"})
        _weights(
            self.source / "model.safetensors",
            {
                "model.language_model.embed_tokens.weight": [2, 3],
                "model.visual.patch_embed.weight": [1, 2],
            },
        )
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        target = "layers.0.mlp.down_proj"
        contract = {
            "rank": 1,
            "alpha": 2,
            "dropout": 0.0,
            "target_modules": [target],
            "target_dimensions": {target: [2, 2]},
            "source_kind": "base",
            "source_fingerprint": source_fingerprint(self.source),
            "base_revision": "a" * 40,
        }
        _json(
            self.checkpoint / "decision_config.json",
            {
                "architecture": "qwen3.5-text-endpoints-global-query-shared-bilinear-mlp",
                "prompt_version": "decision2-segmented-options-global-query-v1",
                "checkpoint_format": "peft-lora/1",
                "text_parameter_count": 6,
                "head_dim": 2,
                "lora": contract,
            },
        )
        _json(self.checkpoint / "tokenizer.json", {"version": "1.0"})
        _weights(self.checkpoint / "decision_head.safetensors", {"key.weight": [1, 2]})
        _json(
            self.checkpoint / "adapter/adapter_config.json",
            {
                "peft_type": "LORA",
                "target_modules": [target],
                "r": 1,
                "lora_alpha": 2,
                "lora_dropout": 0.0,
                "bias": "none",
                "base_model_name_or_path": None,
            },
        )
        _weights(
            self.checkpoint / "adapter/adapter_model.safetensors",
            {
                f"base_model.model.{target}.lora_A.weight": [1, 2],
                f"base_model.model.{target}.lora_B.weight": [2, 1],
            },
        )
        (self.checkpoint / "adapter/README.md").write_text(
            "PEFT adapter\n", encoding="utf-8"
        )
        _json(self.checkpoint / "checkpoint.json", {"step": 368})
        (self.checkpoint / "trainer_state.pt").write_bytes(b"private optimizer fixture")
        identity = checkpoint_fingerprint(self.checkpoint, self.source)
        self.calibration = self.root / "cal.json"
        _json(
            self.calibration,
            {
                "calibration_version": "decision2-per-type-temperature/1",
                "model_sha256": identity["model_sha256"],
                "checkpoint_sha256": "b" * 64,
                "cal_sha256": "c" * 64,
                "best_sha256": "d" * 64,
                "complete_sha256": "e" * 64,
                "provenance_sha256": "f" * 64,
                "fit_split": "cal",
                "selection_policy": "completed_run_best_only",
                "temperature_by_type": {"choice": 1.0, "noul": 1.0, "score": 1.0},
                "inference": {"max_length": 512},
            },
        )
        self.scored = self.root / "scored.json"
        runtime = Path(adapter_bundle.__file__).resolve().parents[1] / "training/model"
        source_hashes = {
            name: adapter_runtime._hash(runtime / name)
            for name in adapter_bundle.LOADER_SOURCES
        }
        _json(
            self.scored,
            {
                "checkpoint_format": "peft-lora/1",
                "model_sha256": identity["model_sha256"],
                "model_files_sha256": identity["files_sha256"],
                "predictions_sha256": "1" * 64,
                "adapter_files_sha256": source_hashes,
                "adapter_sha256": hashlib.sha256(
                    canonical(source_hashes).encode()
                ).hexdigest(),
                "calibration": {"file_sha256": adapter_runtime._hash(self.calibration)},
                "max_length": 512,
                "torch_version": "2.9.0+rocm",
                "peft_version": "0.17.0",
            },
        )
        self.lock = self.root / "lock.json"
        _json(
            self.lock,
            {
                "python": ".".join(str(piece) for piece in sys.version_info[:3]),
                "torch": "2.9.0+rocm",
                "transformers": "4.57.1",
                "peft": "0.17.0",
                "safetensors": "0.6.2",
                "huggingface_hub": "0.35.0",
            },
        )

    def assemble(self, output: str = "bundle") -> dict:
        return adapter_bundle.assemble(
            checkpoint=self.checkpoint,
            source=self.source,
            calibration=self.calibration,
            scored_manifest=self.scored,
            dependency_lock=self.lock,
            base_repo_id="Qwen/Qwen3.5-0.8B-Base",
            base_revision="a" * 40,
            model_id="llm-semantic-router/DEV2.0-0.8B",
            output=self.root / output,
        )

    def test_adapter_package_pins_full_base_and_verifies_external_bytes(self) -> None:
        manifest = self.assemble()
        self.assertEqual(
            manifest["parameter_breakdown"],
            {
                "base_text": 6,
                "adapter": 4,
                "head": 2,
                "total": 12,
            },
        )
        self.assertEqual(manifest["publication_status"], "candidate-parity-pending")
        self.assertFalse((self.root / "bundle/model/backbone").exists())
        self.assertFalse((self.root / "bundle/model/adapter/README.md").exists())
        self.assertFalse((self.root / "bundle/model/checkpoint.json").exists())
        self.assertFalse((self.root / "bundle/model/trainer_state.pt").exists())
        adapter_bundle._verify_staged_runtime(self.root / "bundle", self.source)
        (self.source / "config.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Upstream source files differ"):
            adapter_bundle._verify_staged_runtime(self.root / "bundle", self.source)

    def test_own_decision1_source_keeps_unmerged_adapter_and_full_count(self) -> None:
        (self.source / "model.safetensors").unlink()
        _json(self.source / "backbone/config.json", {"model_type": "qwen3_5_text"})
        _weights(self.source / "backbone/model.safetensors", {"embed.weight": [2, 3]})
        _json(
            self.source / "decision_config.json",
            {
                "architecture": "contextual-candidate-endpoint-plus-global-query-shared-bilinear-mlp",
                "prompt_version": "structured-segmented-candidate-endpoints-global-query-v2",
                "head_dim": 2,
            },
        )
        _weights(self.source / "decision_head.safetensors", {"key.weight": [1, 2]})
        metadata = json.loads((self.checkpoint / "decision_config.json").read_text())
        metadata["lora"]["source_kind"] = "decision1"
        metadata["lora"]["base_revision"] = "b" * 40
        metadata["lora"]["source_fingerprint"] = source_fingerprint(self.source)
        _json(self.checkpoint / "decision_config.json", metadata)
        identity = checkpoint_fingerprint(self.checkpoint, self.source)
        scored = json.loads(self.scored.read_text())
        scored["model_sha256"] = identity["model_sha256"]
        scored["model_files_sha256"] = identity["files_sha256"]
        _json(self.scored, scored)
        calibration = json.loads(self.calibration.read_text())
        calibration["model_sha256"] = identity["model_sha256"]
        _json(self.calibration, calibration)
        scored["calibration"]["file_sha256"] = adapter_runtime._hash(self.calibration)
        _json(self.scored, scored)

        arguments = dict(
            checkpoint=self.checkpoint,
            source=self.source,
            calibration=self.calibration,
            scored_manifest=self.scored,
            dependency_lock=self.lock,
            base_revision="a" * 40,
            model_id="llm-semantic-router/DEV2.0-2B",
        )
        with self.assertRaisesRegex(ValueError, "own-family repository"):
            adapter_bundle.assemble(
                **arguments,
                base_repo_id="somebody/decision-model",
                output=self.root / "invalid-origin",
            )
        manifest = adapter_bundle.assemble(
            **arguments,
            base_repo_id="llm-semantic-router/Decision-1.0-Sol-2B",
            output=self.root / "own-source",
        )
        self.assertEqual(manifest["base"]["source_kind"], "decision1")
        self.assertEqual(manifest["parameter_breakdown"]["base_text"], 6)
        self.assertEqual(manifest["parameter_count"], 12)
        self.assertFalse((self.root / "own-source/model/backbone").exists())
        adapter_bundle._verify_staged_runtime(self.root / "own-source", self.source)
        old = json.loads((self.source / "decision_config.json").read_text())
        old["architecture"] = "other"
        _json(self.source / "decision_config.json", old)
        with self.assertRaisesRegex(ValueError, "incompatible with the native loader"):
            adapter_bundle.assemble(
                **arguments,
                base_repo_id="llm-semantic-router/Decision-1.0-Sol-2B",
                output=self.root / "incompatible-origin",
            )

    def test_normal_python_import_keeps_package_verifiable(self) -> None:
        self.assemble()
        script = (
            "import sys; "
            "sys.path.insert(0, sys.argv[1]); "
            "import decision2; "
            "decision2.verify_bundle(sys.argv[2], sys.argv[3]); "
            "assert callable(decision2.api._dependencies); "
            "decision2.verify_bundle(sys.argv[2], sys.argv[3])"
        )
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(self.root / "bundle"),
                str(self.root / "bundle"),
                str(self.source),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue((self.root / "bundle/decision2/__pycache__").is_dir())

    def test_private_address_in_model_metadata_is_rejected(self) -> None:
        metadata = json.loads((self.checkpoint / "decision_config.json").read_text())
        metadata["lora"]["source_fingerprint"]["source_name"] = "192.0.2.1"
        _json(self.checkpoint / "decision_config.json", metadata)
        with self.assertRaisesRegex(ValueError, "private infrastructure"):
            self.assemble()

    def test_tokenizer_merge_slashes_are_data_but_other_absolute_values_fail(
        self,
    ) -> None:
        tokenizer = self.root / "tokenizer.json"
        _json(tokenizer, {"model": {"merges": [["/", "a"], ["b", "//"]]}})
        adapter_bundle._screen_public_file(tokenizer)
        _json(
            tokenizer,
            {
                "model": {"merges": [["/", "a"]]},
                "metadata": {"cache_path": "/etc/secret"},
            },
        )
        with self.assertRaisesRegex(ValueError, "Absolute path value"):
            adapter_bundle._screen_public_file(tokenizer)

    def test_missing_base_and_tampered_loader_are_rejected(self) -> None:
        self.assemble()
        with self.assertRaises(FileNotFoundError):
            adapter_bundle._verify_staged_runtime(
                self.root / "bundle", self.root / "missing"
            )
        (self.root / "bundle/decision2/infer.py").write_text(
            "# modified\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "file inventory"):
            adapter_bundle._verify_staged_runtime(self.root / "bundle", self.source)

    def test_scored_identity_revision_and_dependency_lock_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "commit SHA"):
            adapter_bundle.assemble(
                checkpoint=self.checkpoint,
                source=self.source,
                calibration=self.calibration,
                scored_manifest=self.scored,
                dependency_lock=self.lock,
                base_repo_id="Qwen/example",
                base_revision="main",
                model_id="llm-semantic-router/DEV2.0-0.8B",
                output=self.root / "bad-revision",
            )
        lock = json.loads(self.lock.read_text(encoding="utf-8"))
        lock["peft"] = "different"
        _json(self.lock, lock)
        with self.assertRaisesRegex(ValueError, "differs from native scored"):
            self.assemble()
        self.assertFalse((self.root / "bundle").exists())

    def test_runtime_rejects_missing_or_changed_pinned_dependency(self) -> None:
        lock = json.loads(self.lock.read_text(encoding="utf-8"))
        with patch.object(
            adapter_runtime,
            "version",
            side_effect=adapter_runtime.PackageNotFoundError("torch"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Missing pinned runtime dependency"
            ):
                adapter_runtime._dependencies({"dependencies": lock})
        with patch.object(adapter_runtime, "version", return_value="changed"):
            with self.assertRaisesRegex(RuntimeError, "differs from lock"):
                adapter_runtime._dependencies({"dependencies": lock})

    def test_scored_calibration_loader_hash_cannot_be_omitted(self) -> None:
        scored = json.loads(self.scored.read_text(encoding="utf-8"))
        del scored["adapter_files_sha256"]["calibration.py"]
        _json(self.scored, scored)
        with self.assertRaisesRegex(ValueError, "Native inference loader changed"):
            self.assemble()
        self.assertFalse((self.root / "bundle").exists())

    def test_adapter_and_parameter_tamper_are_rejected(self) -> None:
        self.assemble()
        (self.root / "bundle/model/adapter/adapter_config.json").write_text(
            "{}\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "file inventory"):
            adapter_bundle._verify_staged_runtime(self.root / "bundle", self.source)
        self.assemble("second")
        manifest_path = self.root / "second/MODEL_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["parameter_count"] = 4
        _json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "parameter count"):
            adapter_bundle._verify_staged_runtime(self.root / "second", self.source)


class ParityContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompts = [
            {
                "id": "row",
                "questions": {
                    "a": {"type": "choice", "criteria": {"x": "X", "y": "Y"}},
                    "b": {"type": "noul", "criteria": {"false": "No", "true": "Yes"}},
                    "c": {"type": "score", "criteria": ["Low", "High"]},
                },
            }
        ]
        self.predictions = [
            {
                "id": "row",
                "answers": {
                    "a": {
                        "type": "choice",
                        "choice": "x",
                        "probabilities": {"x": 0.9, "y": 0.1},
                    },
                    "b": {"type": "noul", "noul": 0.8},
                    "c": {
                        "type": "score",
                        "score": 0.7,
                        "probabilities": {"0": 0.3, "1": 0.7},
                    },
                },
            }
        ]

    def test_matching_typed_native_outputs_pass(self) -> None:
        result = adapter_parity.compare_answers(
            self.prompts, self.predictions, self.predictions
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["questions"], 3)

    def test_flip_drift_missing_and_incomplete_roster_fail(self) -> None:
        changed = json.loads(json.dumps(self.predictions))
        changed[0]["answers"]["a"]["choice"] = "y"
        self.assertFalse(
            adapter_parity.compare_answers(self.prompts, self.predictions, changed)[
                "passed"
            ]
        )
        changed = json.loads(json.dumps(self.predictions))
        changed[0]["answers"]["b"]["noul"] = 0.799
        self.assertFalse(
            adapter_parity.compare_answers(self.prompts, self.predictions, changed)[
                "passed"
            ]
        )
        changed = json.loads(json.dumps(self.predictions))
        changed[0]["answers"]["b"]["noul"] = None
        self.assertFalse(
            adapter_parity.compare_answers(self.prompts, self.predictions, changed)[
                "passed"
            ]
        )
        changed = json.loads(json.dumps(self.predictions))
        changed[0]["answers"]["c"] = {"type": "score", "error": "invalid_model_output"}
        self.assertFalse(
            adapter_parity.compare_answers(self.prompts, self.predictions, changed)[
                "passed"
            ]
        )
        with self.assertRaisesRegex(ValueError, "complete question mapping"):
            adapter_parity.compare_answers(
                self.prompts, self.predictions, [{"id": "row", "answers": {}}]
            )
        with self.assertRaisesRegex(ValueError, "must contain native"):
            adapter_parity.compare_answers(
                [{"id": "row", "questions": {"a": {"type": "choice"}}}],
                [{"id": "row", "answers": {"a": self.predictions[0]["answers"]["a"]}}],
                [{"id": "row", "answers": {"a": self.predictions[0]["answers"]["a"]}}],
            )

    def test_tied_choice_and_out_of_domain_probabilities_fail(self) -> None:
        for qid, field, value in (
            ("a", "choice", None),
            ("b", "noul", 1.2),
            ("c", "score", 4.0),
        ):
            changed = json.loads(json.dumps(self.predictions))
            changed[0]["answers"][qid][field] = value
            result = adapter_parity.compare_answers(self.prompts, changed, changed)
            self.assertFalse(result["passed"])
            self.assertEqual(result["invalid_or_missing_n"], 1)
        tied = json.loads(json.dumps(self.predictions))
        tied[0]["answers"]["a"].update(
            {"choice": None, "probabilities": {"x": 0.5, "y": 0.5}}
        )
        self.assertFalse(
            adapter_parity.compare_answers(self.prompts, tied, tied)["passed"]
        )

    def test_near_tie_noul_and_score_follow_scorer_decisions(self) -> None:
        source = json.loads(json.dumps(self.predictions))
        package = json.loads(json.dumps(self.predictions))
        source[0]["answers"]["b"]["noul"] = 0.5
        package[0]["answers"]["b"]["noul"] = 0.500001
        source[0]["answers"]["c"].update(
            {"score": 0.5, "probabilities": {"0": 0.5, "1": 0.5}}
        )
        package[0]["answers"]["c"].update(
            {"score": 0.499999, "probabilities": {"0": 0.500001, "1": 0.499999}}
        )
        result = adapter_parity.compare_answers(self.prompts, source, package)
        self.assertEqual(result["categorical_mismatch_n"], 2)
        self.assertLess(result["max_probability_or_score_drift"], 0.005)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
