"""CPU-only contract tests for portable Decision 2.0 bundle assembly."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from publication.bundle import PUBLIC_MODEL_ID, bundle, sha_file
from publication.rights_gate import OLD_HOLDOUT_COUNTS, OLD_HOLDOUT_SHA
from training.model.data import canonical
from training.model.infer import checkpoint_fingerprint


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fake_safetensors(path: Path) -> None:
    header = b'{"__metadata__":{"format":"pt"}}'
    path.write_bytes(len(header).to_bytes(8, "little") + header)


class BundleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "merged"
        (self.checkpoint / "backbone").mkdir(parents=True)
        self.source_files = {"config.json": "a" * 64}
        self.initialization = {
            "source_name": "Decision-1.0-Nox-4B",
            "files_sha256": self.source_files,
        }
        self.run_dir = self.root / "completed-run"
        selected_dir = self.run_dir / "checkpoint-0000032"
        (selected_dir / "adapter").mkdir(parents=True)
        write_json(
            selected_dir / "decision_config.json",
            {
                "checkpoint_format": "peft-lora/1",
                "lora": {"source_fingerprint": self.initialization},
            },
        )
        write_json(selected_dir / "tokenizer.json", {"version": "1.0"})
        write_json(selected_dir / "adapter" / "adapter_config.json", {"r": 16})
        fake_safetensors(selected_dir / "decision_head.safetensors")
        fake_safetensors(selected_dir / "adapter" / "adapter_model.safetensors")
        selected_files = [
            selected_dir / name
            for name in (
                "decision_config.json",
                "tokenizer.json",
                "decision_head.safetensors",
                "adapter/adapter_config.json",
                "adapter/adapter_model.safetensors",
            )
        ]
        self.checkpoint_files = {
            str(file.relative_to(selected_dir)): sha_file(file)
            for file in selected_files
        }
        self.source_model_files = {
            **{f"source/{name}": digest for name, digest in self.source_files.items()},
            **{
                f"checkpoint/{name}": digest
                for name, digest in self.checkpoint_files.items()
            },
        }
        self.source_sha = hashlib.sha256(
            canonical(self.source_model_files).encode()
        ).hexdigest()
        write_json(
            self.checkpoint / "decision_config.json",
            {
                "architecture": "qwen3.5-text-endpoints-global-query-shared-bilinear-mlp",
                "prompt_version": "decision2-segmented-options-global-query-v1",
                "checkpoint_format": "full",
                "initialization": "merged-peft-lora",
                "lora_origin": {"adapter": {"model_sha256": self.source_sha}},
            },
        )
        write_json(self.checkpoint / "tokenizer.json", {"version": "1.0"})
        write_json(
            self.checkpoint / "backbone" / "config.json", {"model_type": "qwen3_5_text"}
        )
        fake_safetensors(self.checkpoint / "decision_head.safetensors")
        fake_safetensors(self.checkpoint / "backbone" / "model.safetensors")
        self.model_identity = checkpoint_fingerprint(self.checkpoint)
        write_json(
            self.checkpoint / "materialization_receipt.json",
            {
                "materialization_version": "decision2-merged-peft-lora/1",
                "source_model_sha256": self.source_sha,
                "source_model_files_sha256": self.source_model_files,
                "merged_model_sha256": self.model_identity["model_sha256"],
                "merged_model_files_sha256": self.model_identity["files_sha256"],
            },
        )
        write_json(
            self.run_dir / "checkpoint-0000032" / "checkpoint.json",
            {
                "complete": True,
                "step": 32,
            },
        )
        write_json(
            self.run_dir / "BEST.json",
            {
                "checkpoint": "checkpoint-0000032",
                "selection": "select family-macro accuracy descending, then normalized Brier ascending, then earliest step",
            },
        )
        write_json(
            self.run_dir / "COMPLETE.json",
            {
                "status": "complete",
                "best": "checkpoint-0000032",
                "step": 32,
                "planned_updates": 32,
            },
        )
        self.partitions = {"train": "1" * 64, "select": "2" * 64, "cal": "3" * 64}
        write_json(
            self.run_dir / "provenance.json",
            {
                "contract": {
                    "model_source": self.initialization,
                    "data_sha256": self.partitions,
                    "max_length": 1024,
                    "planned_updates": 32,
                    "train_mode": "lora",
                    "init_kind": "decision1",
                    "objective": "ce_brier",
                    "brier_weight": 0.5,
                    "epochs": 1,
                    "microbatch": 1,
                    "accumulation": 16,
                    "seed": 20260926,
                    "head_lr": 1.5e-5,
                    "weight_decay": 0.01,
                    "warmup_ratio": 0.03,
                    "lora": {
                        "rank": 16,
                        "alpha": 32,
                        "dropout": 0.05,
                        "lr": 3e-5,
                        "target_modules": ["q_proj"],
                    },
                },
                "model_source": self.initialization,
                "train_examples": 100,
                "select_examples": 20,
                "cal_examples_audited_only": 30,
                "precision": "FP32 LoRA/head with BF16 backbone compute",
            },
        )
        self.data_manifest = self.root / "data.manifest.json"
        write_json(
            self.data_manifest,
            {
                "schema_version": "decision2-rights-clean-splits/1",
                "outputs": {
                    f"{role}.jsonl": {"sha256": digest, "rows": count}
                    for role, digest, count in (
                        ("train", self.partitions["train"], 100),
                        ("select", self.partitions["select"], 20),
                        ("cal", self.partitions["cal"], 30),
                    )
                },
                "counts": {"source": {"legacy": 100}, "task_type": {"choice": 100}},
                "partition_counts": {
                    "select": {"source": {"internal_select": 20}},
                    "cal": {"source": {"internal_cal": 30}},
                },
                "source_rights": [
                    {
                        "source": "legacy",
                        "rows": 100,
                        "license": "reviewed research source",
                        "evidence": "https://github.com/cardiffnlp/tweeteval",
                    },
                    {
                        "source": "internal SELECT/CAL",
                        "rows": 50,
                        "partition_scope": "SELECT/CAL",
                        "license": "internally generated",
                        "evidence": "https://github.com/cardiffnlp/tweeteval",
                    },
                ],
                "publication_eligible": True,
                "publication_scope": "trained weights/modelcard only; no raw source rows, SELECT/CAL rows, or individual text predictions",
                "publication_conditions": ["Attribute the original authors."],
                "overlap_audits": {"train_vs_select": {"near_context": {"count": 0}}},
                "limitations": ["Synthetic Score cases are scarce."],
            },
        )
        self.training_record = self.root / "training-record.json"
        write_json(
            self.training_record,
            {
                "record_version": "decision2-public-training-record/1",
                "initialization": {
                    "model_id": "llm-semantic-router/Decision-1.0-Nox-4B",
                    "revision": "a" * 40,
                    "source_name": "Decision-1.0-Nox-4B",
                    "license_status": "Apache-2.0 (reviewed release metadata)",
                },
                "sources": [
                    {
                        "source": "legacy",
                        "rows": 100,
                        "url": "https://github.com/cardiffnlp/tweeteval",
                        "license_status": "Research use; source text not redistributed",
                        "attribution": "Original authors",
                    }
                ],
                "known_overlap": [
                    "Original Decision 1.0 training may overlap a transfer task."
                ],
                "evaluation_interpretation": ["Transfer scores are task-specific."],
                "limitations": ["Do not rely on untested long context."],
            },
        )
        self.calibration = self.root / "calibration.json"
        write_json(
            self.calibration,
            {
                "calibration_version": "decision2-per-type-temperature/1",
                "fit_split": "cal",
                "selection_policy": "completed_run_best_only",
                "model_sha256": self.source_sha,
                "selected_checkpoint": "checkpoint-0000032",
                "checkpoint_sha256": hashlib.sha256(
                    canonical(self.checkpoint_files).encode()
                ).hexdigest(),
                "cal_sha256": self.partitions["cal"],
                "best_sha256": sha_file(self.run_dir / "BEST.json"),
                "complete_sha256": sha_file(self.run_dir / "COMPLETE.json"),
                "provenance_sha256": sha_file(self.run_dir / "provenance.json"),
                "initialization_source_sha256": hashlib.sha256(
                    canonical(self.source_files).encode()
                ).hexdigest(),
                "loaded_source_sha256": hashlib.sha256(
                    canonical(self.source_files).encode()
                ).hexdigest(),
                "temperature_by_type": {"choice": 1.2, "noul": 1.1, "score": 1.3},
                "inference": {"max_length": 1024},
            },
        )
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        for name, content in (
            (
                "score-table.md",
                "## Scores\n\n| Model | Overall |\n|---|---|\n| Decision 2.0 | 70% |\n",
            ),
            ("ranking.svg", "<svg></svg>\n"),
            ("matrix.svg", "<svg></svg>\n"),
        ):
            (self.artifacts / name).write_text(content, encoding="utf-8")
        self.prediction_sha = "1" * 64
        write_json(
            self.artifacts / "manifest.json",
            {
                "publication_version": "decision-model-card-artifacts/2",
                "frozen_benchmark": {"gold_sha256": "2" * 64},
                "artifacts_sha256": {
                    name: sha_file(self.artifacts / name)
                    for name in ("score-table.md", "ranking.svg", "matrix.svg")
                },
                "models": [
                    {
                        "key": "d2-4b",
                        "group": "decision2",
                        "size": "4B",
                        "identity": {
                            "id": "decision2-nox",
                            "revision": "best-100",
                            "backend": "local-dynamic-candidate",
                        },
                        "benchmark": {"predictions_sha256": self.prediction_sha},
                    }
                ],
            },
        )
        self.scored_manifest = self.root / "scored.manifest.json"
        write_json(
            self.scored_manifest,
            {
                "model_id": "decision2-nox",
                "model_revision": "best-100",
                "model_sha256": self.source_sha,
                "predictions_sha256": self.prediction_sha,
                "calibration": {"file_sha256": sha_file(self.calibration)},
                "max_length": 1024,
            },
        )
        self.output = self.root / "published"

    def package(self):
        return bundle(
            checkpoint=self.checkpoint,
            calibration=self.calibration,
            card_artifacts=self.artifacts,
            scored_manifest=self.scored_manifest,
            training_record=self.training_record,
            run_dir=self.run_dir,
            training_data_manifest=self.data_manifest,
            score_key="d2-4b",
            model_id="llm-semantic-router/dev-2.0-4b",
            base_model_id="Qwen/Qwen3.5-4B-Base",
            license_id="apache-2.0",
            output=self.output,
        )

    def test_self_contained_bundle_verifies_and_records_premerge_evaluation(self):
        manifest = self.package()
        self.assertEqual(manifest["evaluation_weight_binding"], "premerge_source")
        self.assertEqual(manifest["model_sha256"], self.model_identity["model_sha256"])
        self.assertEqual(
            (self.output / "calibration.json").read_bytes(),
            self.calibration.read_bytes(),
        )
        self.assertTrue(
            (self.output / "model" / "backbone" / "model.safetensors").is_file()
        )
        self.assertTrue((self.output / "decision2" / "api.py").is_file())
        self.assertFalse((self.output / "model" / "adapter").exists())
        card = (self.output / "README.md").read_text()
        self.assertIn("numerical parity", card)
        self.assertIn("## Training and data", card)
        self.assertIn("Research use; source text not redistributed", card)
        self.assertIn("TRAIN 100 rows", card)
        self.assertIn("SELECT 20 rows", card)
        self.assertIn("CAL 30 rows", card)
        self.assertIn("Original Decision 1.0 training may overlap", card)
        public = json.loads((self.output / "training-provenance.json").read_text())
        self.assertEqual(public["verification"]["source_model_sha256"], self.source_sha)
        self.assertEqual(public["verification"]["partition_sha256"], self.partitions)
        self.assertEqual(public["rights"]["mode"], "rights_clean")
        self.assertIn("SELECT/CAL source counts", card)
        self.assertEqual(
            manifest["training_provenance_sha256"],
            sha_file(self.output / "training-provenance.json"),
        )
        self.assertNotIn(
            str(self.root), (self.output / "MODEL_MANIFEST.json").read_text()
        )
        self.assertNotIn(
            str(self.root), (self.output / "training-provenance.json").read_text()
        )
        script = "import sys; sys.path.insert(0, sys.argv[1]); from decision2 import verify_bundle; verify_bundle(sys.argv[1])"
        subprocess.run(
            [sys.executable, "-I", "-c", script, str(self.output)],
            check=True,
            capture_output=True,
            text=True,
            cwd=self.root,
        )
        (self.output / "model" / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        failure = subprocess.run(
            [sys.executable, "-I", "-c", script, str(self.output)],
            capture_output=True,
            text=True,
            cwd=self.root,
        )
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("hash mismatch", failure.stderr)

    def test_verifier_rejects_unlisted_tokenizer_input(self):
        self.package()
        (self.output / "model" / "special_tokens_map.json").write_text(
            "{}\n", encoding="utf-8"
        )
        script = "import sys; sys.path.insert(0, sys.argv[1]); from decision2 import verify_bundle; verify_bundle(sys.argv[1])"
        failure = subprocess.run(
            [sys.executable, "-I", "-c", script, str(self.output)],
            capture_output=True,
            text=True,
            cwd=self.root,
        )
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("Bundle model hash differs", failure.stderr)

    def test_merged_weight_score_binding_is_recorded(self):
        scored = json.loads(self.scored_manifest.read_text())
        scored["model_sha256"] = self.model_identity["model_sha256"]
        write_json(self.scored_manifest, scored)
        manifest = self.package()
        self.assertEqual(manifest["evaluation_weight_binding"], "merged")
        self.assertIn(
            "exact published merged weights", (self.output / "README.md").read_text()
        )

    def test_mismatched_calibration_and_score_are_rejected_before_output(self):
        scored = json.loads(self.scored_manifest.read_text())
        scored["predictions_sha256"] = "0" * 64
        write_json(self.scored_manifest, scored)
        with self.assertRaisesRegex(ValueError, "not bound"):
            self.package()
        self.assertFalse(self.output.exists())
        scored["predictions_sha256"] = self.prediction_sha
        write_json(self.scored_manifest, scored)
        calibration = json.loads(self.calibration.read_text())
        calibration["model_sha256"] = "9" * 64
        write_json(self.calibration, calibration)
        with self.assertRaisesRegex(ValueError, "Calibration model hash"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_private_text_and_unwanted_checkpoint_files_are_rejected(self):
        (self.artifacts / "score-table.md").write_text(
            "Private host /home/person/secret/model and token hf_" + "a" * 24,
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Credential-like"):
            self.package()
        self.assertFalse(self.output.exists())
        (self.artifacts / "score-table.md").write_text(
            "/work/runs/model\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Absolute host path"):
            self.package()
        (self.artifacts / "score-table.md").write_text("safe\n", encoding="utf-8")
        (self.checkpoint / "optimizer.pt").write_bytes(b"private optimizer")
        with self.assertRaisesRegex(ValueError, "non-model file"):
            self.package()

    def test_run_and_selected_weight_mutation_cannot_rebind_card(self):
        best = json.loads((self.run_dir / "BEST.json").read_text())
        best["selection"] = "different rule"
        write_json(self.run_dir / "BEST.json", best)
        with self.assertRaisesRegex(ValueError, "best_sha256"):
            self.package()
        self.assertFalse(self.output.exists())
        best["selection"] = (
            "select family-macro accuracy descending, then normalized Brier ascending, then earliest step"
        )
        write_json(self.run_dir / "BEST.json", best)
        adapter = (
            self.run_dir
            / "checkpoint-0000032"
            / "adapter"
            / "adapter_model.safetensors"
        )
        adapter.write_bytes(adapter.read_bytes() + b"mutation")
        with self.assertRaisesRegex(ValueError, "selected checkpoint weight files"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_mismatched_data_sources_and_private_record_are_rejected(self):
        record = json.loads(self.training_record.read_text())
        record["sources"][0]["rows"] = 99
        write_json(self.training_record, record)
        with self.assertRaisesRegex(ValueError, "declared source counts"):
            self.package()
        record["sources"][0]["rows"] = 100
        record["known_overlap"].append("Private /home/person/research/notes")
        write_json(self.training_record, record)
        with self.assertRaisesRegex(ValueError, "Absolute host path"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_data_manifest_without_rights_is_rejected(self):
        manifest = json.loads(self.data_manifest.read_text())
        manifest["publication_eligible"] = False
        write_json(self.data_manifest, manifest)
        with self.assertRaisesRegex(ValueError, "publication eligibility"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_noncommercial_bundle_copies_exact_research_attestation(self):
        partitions = {"train": self.partitions["train"], **OLD_HOLDOUT_SHA}
        provenance_path = self.run_dir / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["contract"]["data_sha256"] = partitions
        write_json(provenance_path, provenance)
        data = json.loads(self.data_manifest.read_text())
        data["schema_version"] = "decision2-balanced-human-5824/1"
        data["outputs"] = {
            "balanced_human_5824.train.jsonl": {
                "sha256": partitions["train"],
                "rows": 100,
            },
            "select.jsonl": {"sha256": partitions["select"], "rows": 20},
            "cal.jsonl": {"sha256": partitions["cal"], "rows": 30},
        }
        write_json(self.data_manifest, data)
        calibration = json.loads(self.calibration.read_text())
        calibration["cal_sha256"] = partitions["cal"]
        calibration["provenance_sha256"] = sha_file(provenance_path)
        write_json(self.calibration, calibration)
        scored = json.loads(self.scored_manifest.read_text())
        scored["calibration"]["file_sha256"] = sha_file(self.calibration)
        write_json(self.scored_manifest, scored)
        attestation_path = self.root / "research-attestation.json"
        condition = {
            "terms": "Noncommercial research; no raw rows.",
            "evidence": "https://github.com/cardiffnlp/tweeteval",
        }
        write_json(
            attestation_path,
            {
                "schema_version": "decision2-noncommercial-research-attestation/1",
                "noncommercial_use": True,
                "publication_scope": "noncommercial research model weights/modelcard only; no raw rows",
                "no_raw_training_rows": True,
                "data_manifest_sha256": sha_file(self.data_manifest),
                "training_provenance_sha256": sha_file(provenance_path),
                "data_sha256": partitions,
                "source_counts": {"legacy": 100},
                "source_groups": {"legacy": "pilot_source"},
                "holdout_source_counts": OLD_HOLDOUT_COUNTS,
                "holdout_groups": {
                    role: dict.fromkeys(counts, "pilot_source")
                    for role, counts in OLD_HOLDOUT_COUNTS.items()
                },
                "rights_conditions": {"pilot_source": condition},
                "limitations": ["Natural transfer remains uncertain."],
            },
        )
        manifest = bundle(
            checkpoint=self.checkpoint,
            calibration=self.calibration,
            card_artifacts=self.artifacts,
            scored_manifest=self.scored_manifest,
            training_record=self.training_record,
            run_dir=self.run_dir,
            training_data_manifest=self.data_manifest,
            rights_attestation=attestation_path,
            score_key="d2-4b",
            model_id="llm-semantic-router/dev-2.0-4b",
            base_model_id="Qwen/Qwen3.5-4B-Base",
            license_id="other",
            output=self.output,
        )
        self.assertEqual(manifest["rights_mode"], "noncommercial_research")
        self.assertEqual(
            manifest["rights_attestation_sha256"], sha_file(attestation_path)
        )
        self.assertEqual(
            (self.output / "rights-attestation.json").read_bytes(),
            attestation_path.read_bytes(),
        )
        self.assertIn(
            "noncommercial research only", (self.output / "README.md").read_text()
        )
        self.assertIn(
            "license_name: mixed-source-noncommercial-research-terms",
            (self.output / "README.md").read_text(),
        )

    def test_old_codename_and_wrong_size_are_rejected(self):
        self.assertIsNotNone(
            PUBLIC_MODEL_ID.fullmatch("llm-semantic-router/dev-2.0-27b")
        )
        self.assertIsNone(PUBLIC_MODEL_ID.fullmatch("llm-semantic-router/dev-2.0-0.5b"))
        with self.assertRaisesRegex(ValueError, "dev-2.0-xxb"):
            bundle(
                checkpoint=self.checkpoint,
                calibration=self.calibration,
                card_artifacts=self.artifacts,
                scored_manifest=self.scored_manifest,
                training_record=self.training_record,
                run_dir=self.run_dir,
                training_data_manifest=self.data_manifest,
                score_key="d2-4b",
                model_id="llm-semantic-router/Decision-2.0-Nox-4B",
                base_model_id="Qwen/Qwen3.5-4B-Base",
                license_id="apache-2.0",
                output=self.output,
            )
        artifact = json.loads((self.artifacts / "manifest.json").read_text())
        artifact["models"][0]["size"] = "9B"
        write_json(self.artifacts / "manifest.json", artifact)
        with self.assertRaisesRegex(ValueError, "size differs"):
            self.package()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
