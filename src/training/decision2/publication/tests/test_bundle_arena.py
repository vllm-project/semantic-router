"""CPU contracts for the six-axis, architecture-specific release packager."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from publication import bundle_arena
from publication.generate_arena import ARTIFACTS
from publication.tests.test_adapter_bundle import AdapterBundleTests
from training.model.calibration import CALIBRATION_VERSION
from training.model.infer import checkpoint_fingerprint


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def safe_weights(
    path: Path, *, name: str = "weight", shape: tuple[int, ...] = (4,)
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 2
    for dimension in shape:
        size *= dimension
    header = json.dumps(
        {name: {"dtype": "F16", "shape": list(shape), "data_offsets": [0, size]}},
        separators=(",", ":"),
    ).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(size))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def authored_receipt(freeze_sha: str, panel: dict[str, str]) -> dict:
    packet_at = "2026-09-01T00:00:00+00:00"
    review_at = "2026-09-01T01:00:00+00:00"
    key_at = "2026-09-01T02:00:00+00:00"
    adjudicated_at = "2026-09-01T03:00:00+00:00"

    def review(index: int, role: str, answer: str | bool | int) -> dict:
        return {
            "reviewer_identity_sha256": digest(f"reviewer-{role}"),
            "review_sha256": digest(f"sealed-{role}-{index}"),
            "sealed_at_utc": review_at,
            "native_answer": answer,
            "source_a_evidence": "First document, decision rule paragraph",
            "source_b_evidence": "Second document, dated evidence paragraph",
            "both_sources_necessary": True,
            "ambiguity": "none",
            "document_realism": "plausible",
            "shortcut_risk": "none",
            "rights_concern": False,
            "all_paragraphs_checked": True,
            "paragraph_notes": "Checked each paragraph for relevant evidence",
        }

    rows = []
    kinds = ("choice", "noul", "score")
    for index in range(1200):
        kind = kinds[index % 3]
        answer = {"choice": "A", "noul": True, "score": 1}[kind]
        rows.append(
            {
                "original_id_sha256": digest(f"original-{index}"),
                "source_family_sha256": digest(f"source-{index}"),
                "author_identity_sha256": digest("author"),
                "type": kind,
                "domain": f"domain-{index % 12}",
                "operation": f"operation-{index % 20}",
                "form_family": f"form-{index % 9}",
                "template_sha256": digest(f"template-{index % 40}"),
                "length_band": (
                    "short"
                    if index % 20 < 6
                    else "medium" if index % 20 < 13 else "long"
                ),
                "original_review_a": review(index, "original-a", answer),
                "original_review_b": review(index, "original-b", answer),
                "paired_review": review(index, "paired", answer),
                "paired_second_review": (
                    review(index, "paired-second", answer) if index // 3 < 60 else None
                ),
                "adjudication": {
                    "adjudicator_identity_sha256": digest("adjudicator"),
                    "adjudication_sha256": digest(f"adjudicated-{index}"),
                    "completed_at_utc": adjudicated_at,
                    "verdict": "accepted",
                    "original_answer": answer,
                    "paired_answer": answer,
                    "oracle_agreement": True,
                    "semantic_independence_passed": True,
                    "overlap_passed": True,
                    "provenance_passed": True,
                    "rights_passed": True,
                    "unresolved_material_errors": 0,
                },
            }
        )
    return {
        "schema_version": bundle_arena.AUTHORED_EDITORIAL_VERSION,
        "status": "passed",
        "pretest_freeze_sha256": freeze_sha,
        "authored_prompts_sha256": panel["authored_prompts"],
        "authored_targets_sha256": panel["authored_targets"],
        "packet_sealed_at_utc": packet_at,
        "key_opened_at_utc": key_at,
        "completed_at_utc": "2026-09-01T04:00:00+00:00",
        "independent_originals": 1200,
        "type_counts": dict.fromkeys(kinds, 400),
        "rows": rows,
    }


class ArenaBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = self.root / "functional-model"
        self.model_id = "llm-semantic-router/dev-2.0-0.8b"
        self.revision = "checkpoint-000064"
        source_sha = "d" * 64
        write(
            self.model / "model/decision_config.json",
            {
                "checkpoint_format": "full",
                "initialization": "merged-peft-lora",
                "lora_origin": {"adapter": {"model_sha256": source_sha}},
            },
        )
        write(self.model / "model/backbone/config.json", {"model_type": "qwen3_5_text"})
        write(self.model / "model/tokenizer.json", {"version": "1"})
        safe_weights(
            self.model / "model/backbone/model.safetensors", name="backbone.weight"
        )
        safe_weights(self.model / "model/decision_head.safetensors", name="head.weight")
        code_root = Path(__file__).resolve().parents[2]
        for name in bundle_arena.QWEN_RUNTIME:
            path = self.model / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "decision2/api.py":
                source = code_root / "publication/runtime_api.py"
            elif name == "decision2/__init__.py":
                path.write_text(
                    "from .api import Decision2, verify_bundle\n", encoding="utf-8"
                )
                continue
            else:
                source = code_root / "training/model" / Path(name).name
            shutil.copyfile(source, path)
        (self.model / "requirements.txt").write_text("torch\n", encoding="utf-8")
        (self.model / "LICENSE").write_text("Upstream license notice fixture\n")
        fingerprint = checkpoint_fingerprint(self.model / "model")
        self.native_sha = fingerprint["model_sha256"]
        write(
            self.model / "model/materialization_receipt.json",
            {
                "materialization_version": "decision2-merged-peft-lora/1",
                "source_model_sha256": source_sha,
                "merged_model_sha256": self.native_sha,
                "merged_model_files_sha256": fingerprint["files_sha256"],
            },
        )
        write(
            self.model / "calibration.json",
            {
                "calibration_version": CALIBRATION_VERSION,
                "model_sha256": source_sha,
                "checkpoint_sha256": "1" * 64,
                "cal_sha256": "2" * 64,
                "best_sha256": "3" * 64,
                "complete_sha256": "4" * 64,
                "provenance_sha256": "5" * 64,
                "fit_split": "cal",
                "selection_policy": "completed_run_best_only",
                "temperature_by_type": {"choice": 1.0, "noul": 1.0, "score": 1.0},
                "inference": {"max_length": 1024},
            },
        )
        self.files = bundle_arena._inventory(self.model)
        self.files_digest = hashlib.sha256(
            json.dumps(self.files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.native_sha = checkpoint_fingerprint(self.model / "model")["model_sha256"]
        self.calibration_sha = self.files["calibration.json"]
        self.provenance_inputs = {
            name: self.root / f"{name}.json"
            for name in ("data_manifest", "run_provenance", "training_code")
        }
        for name, path in self.provenance_inputs.items():
            write(path, {"identity": name})

        self.record_path = self.root / "package-record.json"
        self.record = {
            "schema_version": bundle_arena.RECORD_VERSION,
            "model_id": self.model_id,
            "model_revision": self.revision,
            "architecture": "qwen3.5-decision-head",
            "parameter_count": 800_000_000,
            "active_weight_files": [
                "model/backbone/model.safetensors",
                "model/decision_head.safetensors",
            ],
            "support_weight_files": [],
            "non_parameter_tensors": [],
            "model_files_sha256": self.files,
            "native_identity": {
                "scheme": "qwen-checkpoint-fingerprint",
                "file": None,
                "sha256": self.native_sha,
            },
            "calibration_file": "calibration.json",
            "calibration_sha256": self.calibration_sha,
            "native_adapter_version": "native-adapter/1",
            "base_model": {
                "id": "Qwen/Qwen3.5-0.8B-Base",
                "revision": "a" * 40,
                "license": "apache-2.0",
            },
            "training": {
                "train_rows": 7455,
                "language_counts": {"en": 6085, "zh": 1370},
                "select_rows": 700,
                "cal_rows": 700,
                "data_manifest_sha256": bundle_arena.sha_file(
                    self.provenance_inputs["data_manifest"]
                ),
                "run_provenance_sha256": bundle_arena.sha_file(
                    self.provenance_inputs["run_provenance"]
                ),
                "training_code_sha256": bundle_arena.sha_file(
                    self.provenance_inputs["training_code"]
                ),
                "selection_policy": "Frozen SELECT only",
            },
            "rights": {
                "status": "passed",
                "scope": "noncommercial_research_weights_card",
                "reviewed_by": "Research release reviewer",
                "no_raw_rows": True,
                "sources": [
                    {
                        "name": "internal synthetic",
                        "license": "research use",
                        "attribution": "Decision 2.0 authors",
                        "use_scope": "noncommercial research",
                        "redistribution": "weights and card only",
                    }
                ],
            },
            "license_id": "other",
            "known_overlap": ["Pilot task exposure is disclosed."],
            "limitations": ["Long context has not been qualified."],
            "evaluation_language_scope": "Mostly English; no multilingual Score conclusion.",
        }
        write(self.record_path, self.record)

        self.score_inputs = {}
        report_hashes = {}
        for family in bundle_arena.SCORE_FAMILIES:
            prediction = self.root / f"{family}.predictions.jsonl"
            prediction.write_text('{"id":"fixture"}\n', encoding="utf-8")
            native_path = self.root / f"{family}.native.json"
            write(
                native_path,
                {
                    "model_id": self.model_id,
                    "model_revision": self.revision,
                    "model_sha256": self.native_sha,
                    "calibration_sha256": self.calibration_sha,
                    "adapter_version": "native-adapter/1",
                    "predictions_sha256": bundle_arena.sha_file(prediction),
                },
            )
            score_path = self.root / f"{family}.score.json"
            report = {"predictions_sha256": bundle_arena.sha_file(prediction)}
            if family in {"public", "dbv4", "authored"}:
                report["prediction_manifest_sha256"] = bundle_arena.sha_file(
                    native_path
                )
            write(score_path, report)
            self.score_inputs[family] = {
                "score": score_path,
                "predictions": prediction,
                "native_manifest": native_path,
            }
            report_hashes[family] = bundle_arena.sha_file(score_path)
        row = {
            "key": "d2-0.8",
            "group": "decision2",
            "model_id": self.model_id,
            "revision": self.revision,
            "size_b": 0.8,
            "rank": 1,
            "score": 50.0,
            "axes": dict.fromkeys(
                (
                    "typed",
                    "transfer",
                    "jevbench_public",
                    "decision_bench_v4",
                    "sealed_authored",
                    "robustness",
                ),
                0.5,
            ),
            "report_sha256": report_hashes,
            "coverage": {
                "synthetic_items": 1600,
                "css_items": 6547,
                "jevbench_public_items": 231,
                "decision_bench_v4_eligible": 1041,
                "decision_bench_v4_ineligible_ne": 30,
                "sealed_authored_items": 1200,
                "effective_text_answers": 10619,
            },
        }
        self.panel_hashes = {
            "synthetic_gold": "4" * 64,
            "css_gold": "5" * 64,
            "jevbench_prompts": "6" * 64,
            "jevbench_targets": "7" * 64,
            "dbv4_prompts": "8" * 64,
            "dbv4_targets": "9" * 64,
            "authored_prompts": "a" * 64,
            "authored_targets": "b" * 64,
        }
        old = {
            **row,
            "key": "d1-0.8",
            "group": "decision1",
            "model_id": "llm-semantic-router/Decision-1.0-Eos-0.8B",
            "revision": "prior-revision",
            "size_b": 0.75,
            "rank": 2,
            "report_sha256": {**report_hashes, "public": "c" * 64},
        }
        self.arena_rank = self.root / "arena-rank.json"
        write(
            self.arena_rank,
            {
                "schema_version": "jevarena-ranking/2",
                "phase": "release",
                "panel_sha256": self.panel_hashes,
                "models": [row, old],
            },
        )
        self.public_rank = self.root / "public-rank.json"
        write(
            self.public_rank,
            {
                "schema_version": "jevarena-jevbench-public-rank/1",
                "items": 231,
                "panel_sha256": {
                    "prompts_sha256": self.panel_hashes["jevbench_prompts"],
                    "targets_sha256": self.panel_hashes["jevbench_targets"],
                },
                "models": [
                    {
                        "key": candidate["key"],
                        "group": candidate["group"],
                        "model_id": candidate["model_id"],
                        "revision": candidate["revision"],
                        "size_b": candidate["size_b"],
                        "report_sha256": candidate["report_sha256"]["public"],
                        "score": 50.0,
                        "accuracy_all": 0.5,
                        "tier_macro_accuracy": 0.5,
                        "rank": candidate["rank"],
                    }
                    for candidate in (row, old)
                ],
            },
        )
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        for name in ARTIFACTS:
            (self.artifacts / name).write_text(f"fixture: {name}\n", encoding="utf-8")
        write(
            self.artifacts / "manifest.json",
            {
                "publication_version": bundle_arena.ARTIFACT_VERSION,
                "phase": "release",
                "artifacts_sha256": {
                    name: bundle_arena.sha_file(self.artifacts / name)
                    for name in ARTIFACTS
                },
                "ranking_sha256": {
                    "arena": bundle_arena.sha_file(self.arena_rank),
                    "jevbench_public": bundle_arena.sha_file(self.public_rank),
                },
                "panel_sha256": self.panel_hashes,
                "coverage": row["coverage"],
                "models": [
                    {
                        "key": candidate["key"],
                        "model_id": candidate["model_id"],
                        "revision": candidate["revision"],
                        "size_b": candidate["size_b"],
                        "arena_rank": candidate["rank"],
                        "jevbench_public_rank": candidate["rank"],
                    }
                    for candidate in (row, old)
                ],
            },
        )
        self.parity_path = self.root / "parity.json"
        self.parity = {
            "schema_version": bundle_arena.PARITY_VERSION,
            "status": "passed",
            "model_id": self.model_id,
            "model_revision": self.revision,
            "native_model_sha256": self.native_sha,
            "model_files_sha256": self.files_digest,
            "calibration_sha256": self.calibration_sha,
            "panels": {
                name: {
                    "items": count,
                    "answers": count,
                    "categorical_mismatch_n": 0,
                    "gate_pass": True,
                    "prompt_sha256": "5" * 64,
                    "report_sha256": "6" * 64,
                    "probability_drift_p99": 0.001,
                    "probability_drift_max": 0.01,
                }
                for name, count in (("dev", 1600), ("css_pilot", 1430))
            },
        }
        write(self.parity_path, self.parity)
        self.freeze_manifest = self.root / "pretest-freeze.json"
        write(
            self.freeze_manifest,
            {
                "status": "frozen",
                "authored_prompts_sha256": self.panel_hashes["authored_prompts"],
                "authored_targets_sha256": self.panel_hashes["authored_targets"],
            },
        )
        self.authored_evidence = authored_receipt(
            bundle_arena.sha_file(self.freeze_manifest), self.panel_hashes
        )
        self.gate_evidence = {
            name: self.root / f"gate-{name}.json" for name in bundle_arena.GATE_CHECKS
        }
        for name, path in self.gate_evidence.items():
            write(
                path,
                (
                    self.authored_evidence
                    if name == "authored_editorial"
                    else {"review": name, "status": "passed"}
                ),
            )
        self.gate_path = self.root / "release-gate.json"
        self.gate = {
            "schema_version": bundle_arena.GATE_VERSION,
            "status": "passed",
            "model_id": self.model_id,
            "model_revision": self.revision,
            "package_record_sha256": bundle_arena.sha_file(self.record_path),
            "parity_receipt_sha256": bundle_arena.sha_file(self.parity_path),
            "artifact_manifest_sha256": bundle_arena.sha_file(
                self.artifacts / "manifest.json"
            ),
            "native_model_sha256": self.native_sha,
            "model_files_sha256": self.files_digest,
            "pretest_freeze_sha256": bundle_arena.sha_file(self.freeze_manifest),
            "checks": {
                name: {
                    "status": "passed",
                    "evidence_sha256": bundle_arena.sha_file(self.gate_evidence[name]),
                }
                for name in bundle_arena.GATE_CHECKS
            },
        }
        write(self.gate_path, self.gate)

    def assemble(self, name: str = "bundle") -> dict:
        # Full-size tensors are mocked only to avoid hundreds of MB in CPU tests.
        with patch.object(bundle_arena, "_parameter_count", return_value=800_000_000):
            return bundle_arena.assemble(
                model_dir=self.model,
                artifacts=self.artifacts,
                arena_rank=self.arena_rank,
                public_rank=self.public_rank,
                package_record=self.record_path,
                parity_receipt=self.parity_path,
                release_gate=self.gate_path,
                provenance_inputs=self.provenance_inputs,
                freeze_manifest=self.freeze_manifest,
                gate_evidence=self.gate_evidence,
                score_inputs=self.score_inputs,
                score_key="d2-0.8",
                output=self.root / name,
            )

    def rewrite_authored_receipt(self) -> None:
        write(self.gate_evidence["authored_editorial"], self.authored_evidence)
        self.gate["checks"]["authored_editorial"]["evidence_sha256"] = (
            bundle_arena.sha_file(self.gate_evidence["authored_editorial"])
        )
        write(self.gate_path, self.gate)

    def test_assembles_exact_same_panel_package_and_detects_tampering(self) -> None:
        result = self.assemble()
        self.assertEqual(result["parameter_count"], 800_000_000)
        self.assertEqual(len(result["score_inputs_sha256"]), 5)
        package = self.root / "bundle"
        card = (package / "README.md").read_text()
        self.assertIn("JevArena rank", card)
        self.assertIn("| en | 6,085 | 81.6% |", card)
        self.assertIn("Evaluation language coverage: Mostly English", card)
        self.assertEqual(bundle_arena.verify(package), result)
        nested_manifest = package / "card-artifacts/PACKAGE_MANIFEST.json"
        nested_manifest.write_text("untracked", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory has changed"):
            bundle_arena.verify(package)
        nested_manifest.unlink()
        (package / "native/calibration.json").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "inventory has changed"):
            bundle_arena.verify(package)

    def test_rejects_blocked_or_missing_review(self) -> None:
        self.gate["checks"]["authored_editorial"]["status"] = "blocked"
        write(self.gate_path, self.gate)
        with self.assertRaisesRegex(ValueError, "review is blocked"):
            self.assemble()
        self.assertFalse((self.root / "bundle").exists())

    def test_rejects_unaccounted_training_languages(self) -> None:
        self.record["training"]["language_counts"]["en"] -= 1
        write(self.record_path, self.record)
        with self.assertRaisesRegex(ValueError, "TRAIN language counts"):
            self.assemble()

    def test_rejects_changed_native_predictions(self) -> None:
        item = self.score_inputs["css"]
        item["predictions"].write_text('{"id":"changed"}\n')
        with self.assertRaisesRegex(ValueError, "predictions differ"):
            self.assemble()

    def test_rejects_changed_artifacts_and_weight_identity(self) -> None:
        (self.artifacts / "score-table.md").write_text("changed")
        with self.assertRaisesRegex(ValueError, "artifact differs"):
            self.assemble()
        (self.artifacts / "score-table.md").write_text("fixture: score-table.md\n")
        self.record["native_identity"]["sha256"] = "0" * 64
        write(self.record_path, self.record)
        with self.assertRaisesRegex(ValueError, "Native model identity differs"):
            self.assemble()

    def test_rejects_parity_failure_and_rights_scope(self) -> None:
        self.parity["panels"]["dev"]["categorical_mismatch_n"] = 1
        write(self.parity_path, self.parity)
        with self.assertRaisesRegex(ValueError, "native parity is missing or failed"):
            self.assemble()
        self.parity["panels"]["dev"]["categorical_mismatch_n"] = 0
        write(self.parity_path, self.parity)
        self.record["license_id"] = "apache-2.0"
        write(self.record_path, self.record)
        with self.assertRaisesRegex(ValueError, "Restricted source terms"):
            self.assemble()

    def test_rejects_changed_review_or_training_provenance_evidence(self) -> None:
        self.gate_evidence["authored_editorial"].write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "External evidence changed"):
            self.assemble()
        write(self.gate_evidence["authored_editorial"], self.authored_evidence)
        self.provenance_inputs["data_manifest"].write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "External evidence changed"):
            self.assemble()

    def test_rejects_bare_or_incomplete_authored_editorial_evidence(self) -> None:
        write(
            self.gate_evidence["authored_editorial"],
            {"review": "authored_editorial", "status": "passed"},
        )
        self.gate["checks"]["authored_editorial"]["evidence_sha256"] = (
            bundle_arena.sha_file(self.gate_evidence["authored_editorial"])
        )
        write(self.gate_path, self.gate)
        with self.assertRaisesRegex(ValueError, "Authored editorial receipt"):
            self.assemble()
        self.assertFalse((self.root / "bundle").exists())

    def test_authored_row_reviews_are_distinct_complete_and_before_key(self) -> None:
        row = self.authored_evidence["rows"][0]
        row["original_review_b"]["reviewer_identity_sha256"] = row["original_review_a"][
            "reviewer_identity_sha256"
        ]
        self.rewrite_authored_receipt()
        with self.assertRaisesRegex(ValueError, "reviewers and adjudicator overlap"):
            self.assemble()
        row["original_review_b"]["reviewer_identity_sha256"] = digest(
            "reviewer-original-b"
        )
        row["original_review_a"]["sealed_at_utc"] = self.authored_evidence[
            "key_opened_at_utc"
        ]
        self.rewrite_authored_receipt()
        with self.assertRaisesRegex(ValueError, "before key access"):
            self.assemble()
        row["original_review_a"]["sealed_at_utc"] = "2026-09-01T01:00:00+00:00"
        row["original_review_a"]["source_a_evidence"] = ""
        self.rewrite_authored_receipt()
        with self.assertRaisesRegex(ValueError, "editorial concerns"):
            self.assemble()

    def test_authored_count_type_and_second_review_cannot_be_faked(self) -> None:
        rows = self.authored_evidence["rows"]
        rows[1]["original_id_sha256"] = rows[0]["original_id_sha256"]
        self.rewrite_authored_receipt()
        with self.assertRaisesRegex(ValueError, "repeats an original"):
            self.assemble()
        rows[1]["original_id_sha256"] = digest("original-1")
        rows[0]["paired_second_review"] = None
        self.rewrite_authored_receipt()
        with self.assertRaisesRegex(ValueError, "second-review coverage"):
            self.assemble()
        rows[0]["paired_second_review"] = authored_receipt(
            self.authored_evidence["pretest_freeze_sha256"], self.panel_hashes
        )["rows"][0]["paired_second_review"]
        self.authored_evidence["type_counts"]["choice"] = 401
        self.rewrite_authored_receipt()
        with self.assertRaisesRegex(ValueError, "allocation"):
            self.assemble()

    def test_rejects_gate_source_mutation_after_validation(self) -> None:
        original = bundle_arena._external_evidence

        def mutate_gate(*args: object, **kwargs: object) -> dict[str, str]:
            result = original(*args, **kwargs)
            changed = json.loads(self.gate_path.read_text())
            changed["status"] = "blocked"
            write(self.gate_path, changed)
            return result

        with patch.object(bundle_arena, "_external_evidence", side_effect=mutate_gate):
            with self.assertRaisesRegex(
                ValueError, "release gate changed while packaging"
            ):
                self.assemble()
        self.assertFalse((self.root / "bundle").exists())

    def test_rejects_freeze_source_mutation_after_validation(self) -> None:
        original = bundle_arena._external_evidence

        def mutate_freeze(*args: object, **kwargs: object) -> dict[str, str]:
            result = original(*args, **kwargs)
            write(self.freeze_manifest, {"status": "reopened"})
            return result

        with patch.object(
            bundle_arena, "_external_evidence", side_effect=mutate_freeze
        ):
            with self.assertRaisesRegex(ValueError, "pretest_freeze"):
                self.assemble()
        self.assertFalse((self.root / "bundle").exists())

    def test_authored_receipt_must_bind_frozen_panel(self) -> None:
        self.authored_evidence["authored_prompts_sha256"] = "f" * 64
        self.rewrite_authored_receipt()
        with self.assertRaisesRegex(ValueError, "not bound to freeze"):
            self.assemble()

    def test_verify_rechecks_internal_gate_after_inventory_rewrite(self) -> None:
        self.assemble()
        package = self.root / "bundle"
        copied_gate = package / "release-gate.json"
        forged = json.loads(copied_gate.read_text())
        forged["package_record_sha256"] = "f" * 64
        write(copied_gate, forged)
        manifest_path = package / "PACKAGE_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["release_gate_sha256"] = bundle_arena.sha_file(copied_gate)
        manifest["files_sha256"]["release-gate.json"] = manifest["release_gate_sha256"]
        write(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "bound to other bytes"):
            bundle_arena.verify(package)

    def test_verify_rechecks_record_rights_after_inventory_rewrite(self) -> None:
        self.assemble()
        package = self.root / "bundle"
        copied_record = package / "release-record.json"
        forged = json.loads(copied_record.read_text())
        forged["license_id"] = "apache-2.0"
        write(copied_record, forged)
        copied_gate = package / "release-gate.json"
        gate = json.loads(copied_gate.read_text())
        gate["package_record_sha256"] = bundle_arena.sha_file(copied_record)
        write(copied_gate, gate)
        manifest_path = package / "PACKAGE_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["package_record_sha256"] = bundle_arena.sha_file(copied_record)
        manifest["release_gate_sha256"] = bundle_arena.sha_file(copied_gate)
        manifest["files_sha256"]["release-record.json"] = manifest[
            "package_record_sha256"
        ]
        manifest["files_sha256"]["release-gate.json"] = manifest["release_gate_sha256"]
        write(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "Restricted source terms"):
            bundle_arena.verify(package)

    def test_rejects_unsafe_or_partial_model_directory(self) -> None:
        (self.model / "raw-data.jsonl").write_text("private row\n")
        with self.assertRaisesRegex(ValueError, "Unsupported model package file"):
            self.assemble()
        (self.model / "raw-data.jsonl").unlink()
        (self.model / "decision2/api.py").unlink()
        with self.assertRaisesRegex(
            ValueError, "differ from the frozen package record"
        ):
            self.assemble()

    def test_safetensors_shape_and_offsets_are_counted(self) -> None:
        self.assertEqual(
            bundle_arena._parameter_count(
                self.model,
                self.record["active_weight_files"],
                [],
                [],
                self.files,
            ),
            8,
        )
        weight = self.model / "model/decision_head.safetensors"
        weight.write_bytes(weight.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "offsets disagree|payload"):
            bundle_arena._tensor_counts(weight)

    def test_encoder_profile_rejects_research_only_checkpoint(self) -> None:
        encoder = self.root / "encoder-model"
        safe_weights(encoder / "model.safetensors", name="encoder.weight")
        safe_weights(encoder / "encoder/model.safetensors", name="encoder.weight")
        write(encoder / "encoder/config.json", {"model_type": "test_encoder"})
        write(encoder / "tokenizer/tokenizer.json", {"version": "1"})
        write(encoder / "rl_agent_config.json", {"agent": "decide"})
        write(encoder / "calibration.json", {"temperature": 1.0})
        (encoder / "runtime.py").write_text("# native encoder adapter\n")
        (encoder / "LICENSE").write_text("Upstream license notice fixture\n")
        write(
            encoder / "CHECKPOINT.json",
            {"research_only": True, "release_qualified": False},
        )
        files = bundle_arena._inventory(encoder)
        with self.assertRaisesRegex(ValueError, "not release-qualified"):
            bundle_arena._profile(encoder, "encoder-decision", files)
        write(
            encoder / "CHECKPOINT.json",
            {"research_only": False, "release_qualified": True},
        )
        files = bundle_arena._inventory(encoder)
        bundle_arena._profile(encoder, "encoder-decision", files)
        self.assertEqual(
            bundle_arena._parameter_count(
                encoder, ["model.safetensors"], ["encoder/model.safetensors"], [], files
            ),
            4,
        )

    def test_native_sum_manifest_must_cover_every_semif_file(self) -> None:
        semif = self.root / "semif-model"
        safe_weights(semif / "model.safetensors")
        for name in (
            "serve.py",
            "decision_config.json",
            "config.json",
            "calib.json",
            "tokenizer.json",
            "decision2_provenance.json",
        ):
            path = semif / name
            if path.suffix == ".json":
                write(path, {"name": name})
            else:
                path.write_text("# native SemIf runtime\n")
        for name in ("LICENSE", "LICENSE-Qwen", "NOTICE"):
            (semif / name).write_text("Upstream license notice fixture\n")
        without_sums = bundle_arena._inventory(semif)
        (semif / "SHA256SUMS").write_text(
            "".join(
                f"{digest}  {name}\n" for name, digest in sorted(without_sums.items())
            )
        )
        files = bundle_arena._inventory(semif)
        bundle_arena._profile(semif, "qwen3.5-semif", files)
        identity = {
            "native_identity": {
                "scheme": "sha256-file",
                "file": "SHA256SUMS",
                "sha256": files["SHA256SUMS"],
            }
        }
        self.assertEqual(
            bundle_arena._native_identity(semif, identity, files), files["SHA256SUMS"]
        )
        (semif / "SHA256SUMS").write_text("0" * 64 + "  model.safetensors\n")
        files = bundle_arena._inventory(semif)
        with self.assertRaisesRegex(
            ValueError, "does not cover exact functional files"
        ):
            bundle_arena._native_identity(semif, identity, files)


class ExternalAdapterArenaBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        release = ArenaBundleTests(
            "test_assembles_exact_same_panel_package_and_detects_tampering"
        )
        release.setUp()
        self.addCleanup(release.doCleanups)
        adapter = AdapterBundleTests(
            "test_adapter_package_pins_full_base_and_verifies_external_bytes"
        )
        adapter.setUp()
        self.addCleanup(adapter.doCleanups)
        inner = adapter.assemble()
        release.model = adapter.root / "bundle"
        release.files = bundle_arena._inventory(
            release.model, allow_adapter_metadata=True
        )
        release.files_digest = hashlib.sha256(
            json.dumps(release.files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        release.native_sha = inner["model_sha256"]
        release.calibration_sha = release.files["calibration.json"]
        combined_files = checkpoint_fingerprint(
            release.model / "model", adapter.source
        )["files_sha256"]
        release.record.update(
            {
                "architecture": bundle_arena.EXTERNAL_ADAPTER,
                "parameter_count": inner["parameter_count"],
                "active_weight_files": [
                    "model/adapter/adapter_model.safetensors",
                    "model/decision_head.safetensors",
                ],
                "support_weight_files": [],
                "non_parameter_tensors": [],
                "model_files_sha256": release.files,
                "native_identity": {
                    "scheme": "external-peft-checkpoint-fingerprint",
                    "file": None,
                    "sha256": inner["model_sha256"],
                },
                "calibration_sha256": release.calibration_sha,
                "native_adapter_version": "decision2-peft-package-native-v1",
            }
        )
        write(release.record_path, release.record)
        report_hashes = {}
        for family, inputs in release.score_inputs.items():
            native_path = inputs["native_manifest"]
            native = json.loads(native_path.read_text(encoding="utf-8"))
            native.update(
                {
                    "model_sha256": inner["model_sha256"],
                    "calibration_sha256": release.calibration_sha,
                    "adapter_version": release.record["native_adapter_version"],
                    "package_manifest_sha256": release.files["MODEL_MANIFEST.json"],
                    "model_files_sha256": combined_files,
                }
            )
            write(native_path, native)
            score_path = inputs["score"]
            score = json.loads(score_path.read_text(encoding="utf-8"))
            if family in {"public", "dbv4", "authored"}:
                score["prediction_manifest_sha256"] = bundle_arena.sha_file(native_path)
            write(score_path, score)
            report_hashes[family] = bundle_arena.sha_file(score_path)
        arena = json.loads(release.arena_rank.read_text(encoding="utf-8"))
        arena["models"][0]["report_sha256"] = report_hashes
        arena["models"][0]["size_b"] = inner["parameter_count"] / 1e9
        write(release.arena_rank, arena)
        public = json.loads(release.public_rank.read_text(encoding="utf-8"))
        public["models"][0]["report_sha256"] = report_hashes["public"]
        public["models"][0]["size_b"] = inner["parameter_count"] / 1e9
        write(release.public_rank, public)
        artifacts = json.loads((release.artifacts / "manifest.json").read_text())
        artifacts["ranking_sha256"] = {
            "arena": bundle_arena.sha_file(release.arena_rank),
            "jevbench_public": bundle_arena.sha_file(release.public_rank),
        }
        artifacts["models"][0]["size_b"] = inner["parameter_count"] / 1e9
        write(release.artifacts / "manifest.json", artifacts)
        release.parity.update(
            {
                "native_model_sha256": inner["model_sha256"],
                "model_files_sha256": release.files_digest,
                "calibration_sha256": release.calibration_sha,
            }
        )
        write(release.parity_path, release.parity)
        release.gate.update(
            {
                "package_record_sha256": bundle_arena.sha_file(release.record_path),
                "parity_receipt_sha256": bundle_arena.sha_file(release.parity_path),
                "artifact_manifest_sha256": bundle_arena.sha_file(
                    release.artifacts / "manifest.json"
                ),
                "native_model_sha256": inner["model_sha256"],
                "model_files_sha256": release.files_digest,
            }
        )
        write(release.gate_path, release.gate)
        self.source_parity = release.root / "adapter-source-parity.json"
        self.source_parity_value = {
            "schema_version": bundle_arena.ADAPTER_PARITY_VERSION,
            "package_manifest_sha256": release.files["MODEL_MANIFEST.json"],
            "scored_prediction_manifest_sha256": inner[
                "scored_prediction_manifest_sha256"
            ],
            "scored_predictions_sha256": inner["scored_predictions_sha256"],
            "model_sha256": inner["model_sha256"],
            "calibration_sha256": inner["calibration_sha256"],
            "gold_free_prompts_sha256": "1" * 64,
            "source_answers_sha256": "2" * 64,
            "package_answers_sha256": "3" * 64,
            "native_runtime": {
                "bf16": True,
                "one_item_batch": True,
                "no_truncation": True,
            },
            "items": 3,
            "questions": 9,
            "types": ["choice", "noul", "score"],
            "invalid_or_missing_n": 0,
            "categorical_mismatch_n": 0,
            "max_probability_or_score_drift": 0,
            "passed": True,
            "predeclared_gate": {
                "invalid_or_missing_n": 0,
                "categorical_mismatch_n": 0,
                "max_probability_or_score_drift": bundle_arena.ADAPTER_MAX_DRIFT,
            },
        }
        write(self.source_parity, self.source_parity_value)
        self.release = release
        self.adapter = adapter
        self.inner = inner
        self.combined_files = combined_files

    def assemble(self) -> dict:
        release = self.release
        with patch.object(bundle_arena, "_size_compatible", return_value=True):
            return bundle_arena.assemble(
                model_dir=release.model,
                artifacts=release.artifacts,
                arena_rank=release.arena_rank,
                public_rank=release.public_rank,
                package_record=release.record_path,
                parity_receipt=release.parity_path,
                release_gate=release.gate_path,
                provenance_inputs=release.provenance_inputs,
                freeze_manifest=release.freeze_manifest,
                gate_evidence=release.gate_evidence,
                score_inputs=release.score_inputs,
                score_key="d2-0.8",
                output=release.root / "external-release",
                base_source=self.adapter.source,
                adapter_source_parity_receipt=self.source_parity,
            )

    def test_external_release_preserves_unmerged_adapter_and_reference(self) -> None:
        result = self.assemble()
        package = self.release.root / "external-release"
        self.assertEqual(result["parameter_count"], 12)
        self.assertEqual(result["external_base"], self.inner["base"])
        self.assertFalse((package / "native/model/backbone").exists())
        self.assertTrue(
            (package / "native/model/adapter/adapter_model.safetensors").exists()
        )
        self.assertIn("generic `AutoModel`", (package / "README.md").read_text())
        with patch.object(bundle_arena, "_size_compatible", return_value=True):
            self.assertEqual(
                bundle_arena.verify(package, base_source=self.adapter.source), result
            )
        imported = subprocess.run(
            [
                sys.executable,
                "-c",
                "import decision2; decision2.verify_bundle(__import__('sys').argv[1])",
                str(package / "native"),
            ],
            cwd=package / "native",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(imported.returncode, 0, imported.stderr)
        with patch.object(bundle_arena, "_size_compatible", return_value=True):
            self.assertEqual(bundle_arena.verify(package), result)
        nested_manifest = package / "card-artifacts/PACKAGE_MANIFEST.json"
        nested_manifest.write_text("untracked", encoding="utf-8")
        with patch.object(bundle_arena, "_size_compatible", return_value=True):
            with self.assertRaisesRegex(ValueError, "inventory has changed"):
                bundle_arena.verify(package)
        nested_manifest.unlink()
        (self.adapter.source / "config.json").write_text("{}\n", encoding="utf-8")
        with patch.object(bundle_arena, "_size_compatible", return_value=True):
            with self.assertRaisesRegex(ValueError, "Upstream source files differ"):
                bundle_arena.verify(package, base_source=self.adapter.source)

    def test_external_release_rejects_unbound_source_parity(self) -> None:
        self.source_parity_value["passed"] = False
        write(self.source_parity, self.source_parity_value)
        with self.assertRaisesRegex(ValueError, "source parity is missing"):
            self.assemble()

    def test_external_release_rejects_wrong_base_and_native_score_package(self) -> None:
        self.release.record["base_model"]["revision"] = "b" * 40
        write(self.release.record_path, self.release.record)
        with self.assertRaisesRegex(ValueError, "base or model ID"):
            self.assemble()
        self.release.record["base_model"]["revision"] = "a" * 40
        write(self.release.record_path, self.release.record)
        native_path = self.release.score_inputs["public"]["native_manifest"]
        native = json.loads(native_path.read_text(encoding="utf-8"))
        native["package_manifest_sha256"] = "f" * 64
        write(native_path, native)
        with self.assertRaisesRegex(ValueError, "another native manifest"):
            self.assemble()
        public_score = self.release.score_inputs["public"]["score"]
        score = json.loads(public_score.read_text(encoding="utf-8"))
        score["prediction_manifest_sha256"] = bundle_arena.sha_file(native_path)
        write(public_score, score)
        row = json.loads(self.release.arena_rank.read_text())["models"][0]
        row["report_sha256"]["public"] = bundle_arena.sha_file(public_score)
        with self.assertRaisesRegex(ValueError, "another external PEFT package"):
            bundle_arena._score_inputs(
                self.release.score_inputs,
                row,
                self.inner["model_sha256"],
                self.release.model_id,
                self.release.revision,
                self.release.calibration_sha,
                self.release.record["native_adapter_version"],
                self.release.files["MODEL_MANIFEST.json"],
                self.combined_files,
            )


if __name__ == "__main__":
    unittest.main()
