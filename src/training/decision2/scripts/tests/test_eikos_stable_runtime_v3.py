"""Gold-free v3 Eikos runtime receipt checks, with synthetic private files."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.eikos_stable_runtime_v3 import (
    FLA_BACKEND,
    FULL_SCHEMA,
    REPEAT_SCHEMA,
    TORCH_BACKEND,
    verified_stable_runtime,
)
from scripts.plan_final_eval import EIKOS_PARITY_PANELS, sha_file


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class EikosStableRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source"
        collector = self.source / "training/eikos/published_infer.py"
        collector.parent.mkdir(parents=True)
        collector.write_text("# pinned collector\n", encoding="utf-8")
        self.collector_sha = sha_file(collector)
        self.repeat_dir = self.root / "repeat"
        self.full_dir = self.root / "full"
        self.repeat_dir.mkdir()
        self.full_dir.mkdir()
        self.lock_path = self.root / "candidate-lock.json"
        self.candidate = {
            "key": "d2-4b",
            "architecture": "eikos_semif",
            "model_id": "llm-semantic-router/DEV2.0-4B",
            "model_sha256": "a" * 64,
            "calibration_sha256": "b" * 64,
            "selected_checkpoint": "checkpoint-0232",
            "parity_reports": {
                "dev": str(self.full_dir / "parity-dev.report.json"),
                "css_pilot": str(self.full_dir / "parity-css.report.json"),
            },
        }
        self.runtime = {
            "torch_deterministic_algorithms": True,
            "gated_delta_backend_before": FLA_BACKEND,
            "gated_delta_backend": TORCH_BACKEND,
        }
        repeat_artifacts = {}
        for run in ("r1", "r2"):
            name = f"{run}.predictions.jsonl.manifest.json"
            write_json(self.repeat_dir / name, self.manifest(1430))
            repeat_artifacts[name] = sha_file(self.repeat_dir / name)
            predictions = self.repeat_dir / f"{run}.predictions.jsonl"
            predictions.write_text('{"id":"synthetic"}\n', encoding="utf-8")
            repeat_artifacts[predictions.name] = sha_file(predictions)
        self.repeat = {
            "schema_version": REPEAT_SCHEMA,
            "model_id": self.candidate["model_id"],
            "package_sha256": self.candidate["model_sha256"],
            "calibration_sha256": self.candidate["calibration_sha256"],
            "collector_sha256": self.collector_sha,
            "prompt_sha256": EIKOS_PARITY_PANELS["css_pilot"][0],
            "predeclared_numeric_repeat_gate_pass": True,
            "original_order_input_digest_and_token_parity": True,
            "comparison": {
                "items": 1430,
                "categorical_mismatch_n": 0,
                "max_option_probability_drift": 0.0,
            },
            "runtime_image_id": "sha256:" + "c" * 64,
            "physical_gpu": {"guid": "synthetic", "index": 0},
            "artifact_sha256": repeat_artifacts,
            **{
                run: {
                    "exit_code": 0,
                    "items": 1430,
                    "valid_questions": 1430,
                    "model_load_warnings": False,
                    "manifest_sha256": repeat_artifacts[
                        f"{run}.predictions.jsonl.manifest.json"
                    ],
                    "predictions_sha256": repeat_artifacts[f"{run}.predictions.jsonl"],
                }
                for run in ("r1", "r2")
            },
        }
        self.parity = {
            "model_id": self.candidate["model_id"],
            "model_sha256": self.candidate["model_sha256"],
            "calibration_sha256": self.candidate["calibration_sha256"],
            "selected_checkpoint": self.candidate["selected_checkpoint"],
            "predeclared_gate_pass": True,
            "total_items": 3030,
            "total_categorical_mismatches": 0,
            "panels": {},
        }
        full_artifacts = {}
        for panel, (prompt_sha, count) in EIKOS_PARITY_PANELS.items():
            name = f"parity-{'css' if panel == 'css_pilot' else panel}.report.json"
            write_json(
                self.full_dir / name,
                {
                    "candidate_manifest_sha256": self.candidate["model_sha256"],
                    "calibration_sha256": self.candidate["calibration_sha256"],
                    "selected_checkpoint": self.candidate["selected_checkpoint"],
                    "prompt_sha256": prompt_sha,
                    "items": count,
                    "predeclared_gate": {"pass": True},
                    "runtime": self.runtime,
                },
            )
            full_artifacts[name] = sha_file(self.full_dir / name)
            self.parity["panels"][panel] = {
                "report_sha256": full_artifacts[name],
                "prompt_sha256": prompt_sha,
                "items": count,
                "gate_pass": True,
            }
        write_json(self.full_dir / "parity-dev-css.receipt.json", self.parity)
        full_artifacts["parity-dev-css.receipt.json"] = sha_file(
            self.full_dir / "parity-dev-css.receipt.json"
        )
        for panel, count in (("dev", 1600), ("css", 1430), ("public", 231)):
            name = f"package-{panel}.predictions.jsonl.manifest.json"
            write_json(self.full_dir / name, self.manifest(count))
            full_artifacts[name] = sha_file(self.full_dir / name)
        self.full = {
            "schema_version": FULL_SCHEMA,
            "model_id": self.candidate["model_id"],
            "all_gates_pass": True,
            "combined_parity_pass": True,
            "runtime_image_id": self.repeat["runtime_image_id"],
            "physical_gpu": self.repeat["physical_gpu"],
            "artifact_sha256": full_artifacts,
            "processes": {
                f"package-{panel}": {"exit_code": 0, "model_load_warnings": False}
                for panel in ("dev", "css", "public")
            },
            "scores": {
                "dev": {"valid": 1600},
                "css_pilot": {"valid": 1430},
                "public231": {"strict_valid": 231, "renormalized": 0},
            },
        }
        self.write_receipts()

    def manifest(self, count: int) -> dict:
        return {
            "model_id": self.candidate["model_id"],
            "model_sha256": self.candidate["model_sha256"],
            "calibration_sha256": self.candidate["calibration_sha256"],
            "model_revision": self.candidate["selected_checkpoint"],
            "collector_source_sha256": self.collector_sha,
            "input_items": count,
            "evaluated_items": count,
            "counts": {"items": count},
            "input_sha256": (
                EIKOS_PARITY_PANELS["dev"][0]
                if count == 1600
                else EIKOS_PARITY_PANELS["css_pilot"][0] if count == 1430 else "c" * 64
            ),
            "runtime": self.runtime,
        }

    def write_receipts(self) -> None:
        parity_path = self.full_dir / "parity-dev-css.receipt.json"
        write_json(parity_path, self.parity)
        self.full["artifact_sha256"][parity_path.name] = sha_file(parity_path)
        write_json(self.repeat_dir / "execution.receipt.json", self.repeat)
        write_json(self.full_dir / "execution.receipt.json", self.full)
        write_json(
            self.lock_path,
            {
                "v3_eikos_stable_runtime": {
                    "d2-4b": {
                        "repeat_execution": self.spec(
                            self.repeat_dir / "execution.receipt.json"
                        ),
                        "full_execution": self.spec(
                            self.full_dir / "execution.receipt.json"
                        ),
                        "full_parity": self.spec(
                            self.full_dir / "parity-dev-css.receipt.json"
                        ),
                    }
                }
            },
        )

    @staticmethod
    def spec(path: Path) -> dict:
        return {"path": str(path), "sha256": sha_file(path)}

    def verify(self) -> dict:
        return verified_stable_runtime(self.lock_path, [self.candidate], self.source)

    def test_complete_fixed_backend_receipts_pass(self) -> None:
        self.assertEqual(set(self.verify()), {"d2-4b"})

    def test_rejects_old_model_id_in_each_receipt_and_manifest(self) -> None:
        old_id = "llm-semantic-router/dev-2.0-4b"
        for receipt in (self.repeat, self.full, self.parity):
            receipt["model_id"] = old_id
            self.write_receipts()
            with self.assertRaisesRegex(ValueError, "identity/gate differs"):
                self.verify()
            receipt["model_id"] = self.candidate["model_id"]
        for directory, receipt, name in (
            (self.repeat_dir, self.repeat, "r1.predictions.jsonl.manifest.json"),
            (self.full_dir, self.full, "package-dev.predictions.jsonl.manifest.json"),
        ):
            path = directory / name
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["model_id"] = old_id
            write_json(path, manifest)
            receipt["artifact_sha256"][name] = sha_file(path)
            if directory == self.repeat_dir:
                receipt["r1"]["manifest_sha256"] = sha_file(path)
            self.write_receipts()
            with self.assertRaisesRegex(ValueError, "backend differs"):
                self.verify()
            manifest["model_id"] = self.candidate["model_id"]
            write_json(path, manifest)
            receipt["artifact_sha256"][name] = sha_file(path)
            if directory == self.repeat_dir:
                receipt["r1"]["manifest_sha256"] = sha_file(path)

    def test_rejects_fla_even_when_all_receipt_hashes_are_refrozen(self) -> None:
        name = "r2.predictions.jsonl.manifest.json"
        path = self.repeat_dir / name
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["runtime"]["gated_delta_backend"] = FLA_BACKEND
        write_json(path, manifest)
        self.repeat["artifact_sha256"][name] = sha_file(path)
        self.repeat["r2"]["manifest_sha256"] = sha_file(path)
        self.write_receipts()
        with self.assertRaisesRegex(ValueError, "backend differs"):
            self.verify()

    def test_rejects_old_parity_path_and_missing_second_run(self) -> None:
        self.candidate["parity_reports"]["dev"] = str(self.root / "old-fla.json")
        with self.assertRaisesRegex(ValueError, "not stable-runtime evidence"):
            self.verify()
        self.candidate["parity_reports"]["dev"] = str(
            self.full_dir / "parity-dev.report.json"
        )
        self.repeat["r2"]["model_load_warnings"] = True
        self.write_receipts()
        with self.assertRaisesRegex(ValueError, "independent r2 receipt failed"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
