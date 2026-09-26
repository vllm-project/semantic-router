"""CPU-only checks for pre-test gates and command-only final planning."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.plan_final_eval import (
    BASELINES,
    CSS_PROMPTS_SHA256,
    EIKOS_PARITY_PANELS,
    FREEZE_VERSION,
    PLAN_VERSION,
    build_plan,
    frozen_candidates,
    frozen_css_prompts,
    protocol_sources,
    sha_file,
)

MODEL_SHA = "a" * 64
CAL_SHA = "b" * 64
BEST_SHA = "c" * 64
COMPLETE_SHA = "d" * 64
PROVENANCE_SHA = "e" * 64


class FinalPlanTests(unittest.TestCase):
    def test_native_catalog_and_command_plan_do_not_execute_work(self) -> None:
        self.assertEqual(len(BASELINES), 15)
        self.assertEqual(sum(model.group == "decision1" for model in BASELINES), 6)
        self.assertEqual(len({model.key for model in BASELINES}), 15)
        candidate = {
            "key": "d2-9b",
            "label": "dev-2.0-9b",
            "size": "9B",
            "model_id": "llm-semantic-router/dev-2.0-9b",
            "selected_checkpoint": "checkpoint-0000025",
            "checkpoint": "/private/run/checkpoint-0000025",
            "source_path": "/private/source",
            "calibration": "/private/calibration.json",
            "max_length": 8192,
            "model_sha256": MODEL_SHA,
            "calibration_sha256": "f" * 64,
            "best_sha256": BEST_SHA,
            "complete_sha256": COMPLETE_SHA,
            "provenance_sha256": PROVENANCE_SHA,
        }
        plan = build_plan(
            candidates=[candidate],
            freeze_sha="f" * 64,
            css_prompts=Path("/private/css-evaluation.prompts.jsonl"),
            css_info={"sha256": CSS_PROMPTS_SHA256, "items": 6547, "task_count": 15},
            protocol_sha={},
            evaluation_root=Path("/private/new-final"),
            source_root=Path("/private/source-code"),
            model_root=Path("/private/models"),
            external_root=Path("/private/external"),
            python="python3",
            kai_lex_python="/private/kai-lex/bin/python",
            fla_path="/private/fla",
        )
        self.assertEqual(plan["plan_version"], PLAN_VERSION)
        self.assertEqual(len(plan["inference"]), 16)
        self.assertEqual(len(plan["scoring"]), 16)
        self.assertEqual(len(plan["paired_ci"]), 15)
        self.assertEqual(
            plan["required_report_versions"]["synthetic_final"],
            "typed-decision-report/2",
        )
        commands = "\n".join(
            command for model in plan["inference"] for command in model["commands"]
        )
        self.assertIn("clients.jev_api", commands)
        self.assertIn("transfer.normalize_jev", commands)
        self.assertIn("training.model.infer", commands)
        self.assertIn("inference.eikos", commands)
        self.assertIn("inference.jevk5", commands)
        self.assertIn("--runtime-path /private/external/jevk5-runtime", commands)
        self.assertIn("--size 9b", commands)
        self.assertIn("--calibration", commands)
        self.assertNotIn("--allow-unvalidated-runtime", commands)
        self.assertIn(
            "/private/new-final/final.gold.jsonl",
            "\n".join(plan["preparation_commands"]),
        )
        for command in (
            plan["preparation_commands"][3],
            plan["scoring"][0]["commands"][0],
            plan["paired_ci"][0]["commands"][0],
            plan["publication_command"],
        ):
            self.assertIn("PYTHONPATH=/private/source-code", command)
        self.assertIn(
            "d2-9b.final.predictions.jsonl.manifest.json",
            plan["raw_prediction_hash_command"],
        )

    def test_freeze_rejects_test_selection_and_calibration_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cal = root / "calibration.json"
            cal.write_text("{}\n", encoding="utf-8")
            lock_path = root / "lock.json"
            entry = {
                "key": "d2-9b",
                "label": "dev-2.0-9b",
                "size": "9B",
                "model_id": "llm-semantic-router/dev-2.0-9b",
                "run_dir": str(root / "run"),
                "cal_data": str(root / "cal.jsonl"),
                "calibration": str(cal),
                "source_path": str(root / "source"),
                "selected_checkpoint": "checkpoint-0000025",
                "model_sha256": MODEL_SHA,
                "calibration_sha256": sha_file(cal),
                "best_sha256": BEST_SHA,
                "complete_sha256": COMPLETE_SHA,
                "provenance_sha256": PROVENANCE_SHA,
            }
            lock = {
                "freeze_version": FREEZE_VERSION,
                "selection_sources": ["select", "cal", "css_evaluation"],
                "candidates": [entry],
            }
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pre-test"):
                frozen_candidates(lock_path)
            lock["selection_sources"] = ["select", "cal", "synthetic_dev", "css_pilot"]
            lock["candidates"][0][
                "model_id"
            ] = "llm-semantic-router/Decision-2.0-Lux-9B"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dev-2.0-xxb"):
                frozen_candidates(lock_path)
            lock["candidates"][0]["model_id"] = "llm-semantic-router/dev-2.0-9b"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            selected = {
                "name": "checkpoint-0000025",
                "checkpoint": root / "run/checkpoint-0000025",
                "best_sha256": BEST_SHA,
                "complete_sha256": COMPLETE_SHA,
                "provenance_sha256": PROVENANCE_SHA,
                "cal_sha256": CAL_SHA,
                "contract": {"max_length": 8192},
            }
            report = {
                "selected_checkpoint": selected["name"],
                "cal_sha256": CAL_SHA,
                "best_sha256": BEST_SHA,
                "complete_sha256": COMPLETE_SHA,
                "provenance_sha256": PROVENANCE_SHA,
                "temperature_by_type": {"choice": 1.0, "noul": 1.0, "score": 1.0},
            }
            with (
                patch("scripts.plan_final_eval.selected_run", return_value=selected),
                patch(
                    "scripts.plan_final_eval.checkpoint_fingerprint",
                    return_value={"model_sha256": MODEL_SHA},
                ),
                patch(
                    "scripts.plan_final_eval.load_calibration",
                    return_value=({}, report),
                ),
            ):
                candidates, lock_sha = frozen_candidates(lock_path)
                self.assertEqual(candidates[0]["max_length"], 8192)
                self.assertEqual(lock_sha, sha_file(lock_path))
                cal.write_text('{"changed":true}\n', encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError, "CAL temperature artifact changed"
                ):
                    frozen_candidates(lock_path)

    def test_pinned_protocol_and_css_prompt_hash_gate(self) -> None:
        source_root = Path(__file__).resolve().parents[2]
        self.assertIn("benchmark/score.py", protocol_sources(source_root))
        with tempfile.TemporaryDirectory() as temporary:
            prompt = Path(temporary) / "css-evaluation.prompts.jsonl"
            prompt.write_text(
                '{"id":"fake","state":"text","questions":{}}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "CSS evaluation gold-free prompt SHA-256"
            ):
                frozen_css_prompts(prompt)

    def test_eikos_freeze_binds_native_selection_package_and_cal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, package, source = root / "run", root / "package", root / "source"
            for folder in (run, package, source):
                folder.mkdir()
            cal_data = root / "cal.jsonl"
            cal_data.write_text("cal\n", encoding="utf-8")
            cal = root / "calib.json"
            cal.write_text("{}\n", encoding="utf-8")
            data_manifest = root / "data-manifest.json"
            data_manifest.write_text("{}\n", encoding="utf-8")
            rights_attestation = root / "noncommercial-attestation.json"
            rights_attestation.write_text('{"scope":"research"}\n', encoding="utf-8")
            provenance = {
                "data_sha256": {
                    "train": "1" * 64,
                    "select": "2" * 64,
                    "cal_audited_only": sha_file(cal_data),
                }
            }
            (run / "provenance.json").write_text(
                json.dumps(provenance), encoding="utf-8"
            )
            (run / "COMPLETE.json").write_text("{}\n", encoding="utf-8")
            (run / "NATIVE_BEST.json").write_text("{}\n", encoding="utf-8")
            report = root / "cal-report.json"
            report.write_text(
                json.dumps(
                    {
                        "checkpoint": "checkpoint-0160",
                        "adapter_weights_sha256": "3" * 64,
                        "calibration_sha256": sha_file(cal),
                        "cal_sha256": sha_file(cal_data),
                    }
                ),
                encoding="utf-8",
            )
            receipt = {
                "training_provenance_sha256": sha_file(run / "provenance.json"),
                "adapter_weights_sha256": "3" * 64,
                "adapter_config_sha256": "4" * 64,
                "training_data_sha256": provenance["data_sha256"],
                "training_data_manifest_sha256": sha_file(data_manifest),
                "calibration_report_sha256": sha_file(report),
                "calibration_data_sha256": sha_file(cal_data),
            }
            (package / "decision2_provenance.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            parity_paths = {}
            for panel, (prompt_sha, count) in EIKOS_PARITY_PANELS.items():
                parity = root / f"{panel}-parity.json"
                parity.write_text(
                    json.dumps(
                        {
                            "inference_variant": "selected_lora_and_merged_same_process",
                            "candidate_manifest_sha256": "5" * 64,
                            "adapter_weights_sha256": "3" * 64,
                            "calibration_sha256": sha_file(cal),
                            "selected_checkpoint": "checkpoint-0160",
                            "prompt_sha256": prompt_sha,
                            "items": count,
                            "answers": count,
                            "choice_mismatch_n": 0,
                            "probability_drift_p99": 0.001,
                            "probability_drift_max": 0.01,
                            "predeclared_gate": {
                                "pass": True,
                                "categorical_mismatches": 0,
                                "probability_drift_p99_lte": 0.005,
                                "probability_drift_max_lte": 0.02,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                parity_paths[panel] = str(parity)
            entry = {
                "key": "d2-4b",
                "label": "dev-2.0-4b",
                "size": "4B",
                "model_id": "llm-semantic-router/dev-2.0-4b",
                "architecture": "eikos_semif",
                "run_dir": str(run),
                "source_path": str(source),
                "package_dir": str(package),
                "cal_data": str(cal_data),
                "calibration": str(cal),
                "calibration_report": str(report),
                "training_data_manifest": str(data_manifest),
                "rights_attestation": str(rights_attestation),
                "rights_attestation_sha256": sha_file(rights_attestation),
                "parity_reports": parity_paths,
                "selected_checkpoint": "checkpoint-0160",
                "model_sha256": "5" * 64,
                "calibration_sha256": sha_file(cal),
                "best_sha256": sha_file(run / "NATIVE_BEST.json"),
                "complete_sha256": sha_file(run / "COMPLETE.json"),
                "provenance_sha256": sha_file(run / "provenance.json"),
            }
            lock = root / "freeze.json"
            lock.write_text(
                json.dumps(
                    {
                        "freeze_version": FREEZE_VERSION,
                        "selection_sources": ["select", "cal"],
                        "candidates": [entry],
                    }
                ),
                encoding="utf-8",
            )
            selected = {
                "name": "checkpoint-0160",
                "adapter_weights_sha256": "3" * 64,
                "adapter_config_sha256": "4" * 64,
            }
            package_id = {
                "model_sha256": "5" * 64,
                "calibration_sha256": sha_file(cal),
                "selected_checkpoint": "checkpoint-0160",
                "rights_mode": "noncommercial_research",
                "rights_attestation_sha256": sha_file(rights_attestation),
                "package_files_checked": 21,
            }
            with (
                patch(
                    "scripts.plan_final_eval.selected_eikos_checkpoint",
                    return_value=selected,
                ),
                patch(
                    "scripts.plan_final_eval.eikos_package_identity",
                    return_value=package_id,
                ),
            ):
                candidates, _ = frozen_candidates(lock)
                self.assertEqual(candidates[0]["architecture"], "eikos_semif")
                self.assertEqual(candidates[0]["max_length"], 16000)
                plan = build_plan(
                    candidates=candidates,
                    freeze_sha="f" * 64,
                    css_prompts=root / "css-evaluation.prompts.jsonl",
                    css_info={
                        "sha256": CSS_PROMPTS_SHA256,
                        "items": 6547,
                        "task_count": 15,
                    },
                    protocol_sha={},
                    evaluation_root=root / "new-final",
                    source_root=root / "source-code",
                    model_root=root / "models",
                    external_root=root / "external",
                    python="python3",
                    kai_lex_python=str(root / "kai/bin/python"),
                    fla_path=str(root / "fla"),
                )
                candidate_plan = next(
                    item for item in plan["inference"] if item["key"] == "d2-4b"
                )
                self.assertEqual(candidate_plan["backend"], "eikos-semif-native")
                self.assertIn(
                    "training.eikos.published_infer", candidate_plan["commands"][0]
                )
                self.assertIn("--rights-attestation", candidate_plan["commands"][0])
                rights_attestation.write_text('{"scope":"changed"}\n', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "rights attestation changed"):
                    frozen_candidates(lock)
                rights_attestation.write_text(
                    '{"scope":"research"}\n', encoding="utf-8"
                )
                receipt["calibration_data_sha256"] = "0" * 64
                (package / "decision2_provenance.json").write_text(
                    json.dumps(receipt), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "lineage differ"):
                    frozen_candidates(lock)
                (package / "decision2_provenance.json").write_text(
                    json.dumps(
                        {**receipt, "calibration_data_sha256": sha_file(cal_data)}
                    ),
                    encoding="utf-8",
                )
                parity = Path(parity_paths["dev"])
                bad_report = json.loads(parity.read_text(encoding="utf-8"))
                bad_report["choice_mismatch_n"] = 1
                parity.write_text(json.dumps(bad_report), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "parity gate"):
                    frozen_candidates(lock)


if __name__ == "__main__":
    unittest.main()
