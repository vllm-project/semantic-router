"""The future read-only audit binds raw bytes to v2 reports without gold."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.audit_final_eval import _ci95, _prediction_rows, audit
from scripts.plan_final_eval import PINNED_PROTOCOL_SHA256, PLAN_VERSION, sha_file


class FinalAuditTests(unittest.TestCase):
    def test_paired_interval_shape(self) -> None:
        self.assertTrue(_ci95([-0.1, 0.2]))
        self.assertFalse(_ci95([0.3, -0.2]))
        self.assertFalse(_ci95([float("nan"), 0.2]))

    def test_eikos_rows_require_frozen_package_and_calibration_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "eikos.predictions.jsonl"
            row = {
                "id": "one",
                "answers": {"q": {"type": "choice", "choice": "a"}},
                "source_input_sha256": "a" * 64,
                "model_id": "llm-semantic-router/dev-2.0-4b",
                "model_revision": "checkpoint-0160",
                "model_sha256": "b" * 64,
                "calibration_sha256": "c" * 64,
                "backend": "eikos-semif-native",
                "adapter_version": "decision2-eikos-semif-native-v1",
            }
            model = {
                "key": "d2-4b",
                "model_id": row["model_id"],
                "revision": row["model_revision"],
                "frozen_architecture": "eikos_semif",
                "frozen_model_sha256": row["model_sha256"],
                "frozen_calibration_sha256": row["calibration_sha256"],
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            _prediction_rows(path, model, 1)
            row["backend"] = "unfrozen-runtime"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Eikos package identity"):
                _prediction_rows(path, model, 1)

    def test_raw_hash_and_v2_score_version_without_gold_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = Path(__file__).resolve().parents[2]
            freeze = root / "freeze.json"
            css_prompts = root / "css.prompts.jsonl"
            freeze.write_text("{}\n", encoding="utf-8")
            css_prompts.write_text("gold-free\n", encoding="utf-8")
            final_pred, css_pred = (
                root / "final.predictions.jsonl",
                root / "css.predictions.jsonl",
            )
            row = {
                "id": "one",
                "answers": {},
                "model": "jev-1.13.0",
                "source_input_sha256": "a" * 64,
            }
            for path in (final_pred, css_pred):
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            raw = root / "RAW_PREDICTIONS.sha256"
            raw.write_text(
                "".join(
                    f"{sha_file(path)}  {path}\n" for path in (final_pred, css_pred)
                )
            )
            final_report, css_report = (
                root / "final.score.json",
                root / "css.score.json",
            )

            def summary(n: int) -> dict[str, int | float]:
                return {
                    "n": n,
                    "correct_n": 0,
                    "valid_n": 0,
                    "invalid_or_missing_n": n,
                    "accuracy_all": 0.0,
                }

            families = (
                "constraint_competition",
                "exception_stack",
                "evidence_join",
                "resource_ledger",
            )
            benchmark = {
                "schema_version": "typed-decision-report/2",
                "split": "final",
                "model": {
                    "id": "TypeSafe/jev-1.13.0",
                    "revision": "jev-1.13.0",
                    "backend": "official-api",
                },
                "predictions_sha256": sha_file(final_pred),
                "gold_sha256": "b" * 64,
                "items": 1600,
                "predicted_items": 1600,
                "overall": summary(1600),
                "macro_family_accuracy": 0.0,
                "by_family": {name: summary(400) for name in families},
                "by_type": {
                    name: summary(n)
                    for name, n in (("choice", 600), ("noul", 500), ("score", 500))
                },
            }
            task_names = (
                "emotion",
                "ibc",
                "media_ideology",
                "indian_english_dialect",
                "raop",
                "talklife",
                "wiki_politeness",
                "tempowic",
                "tropes",
                "flute",
                "mrf",
                "conv_go_awry",
                "persuasion",
                "reddit_humor",
                "wiki_corpus",
            )
            css = {
                "score_schema_version": "css-transfer-score/2",
                "panel_version": "css-transfer/1",
                "predictions_sha256": sha_file(css_pred),
                "gold_sha256": "c" * 64,
                "tasks": {
                    name: {
                        **summary(437 if index < 14 else 429),
                        "role": "evaluation",
                        "macro_f1_all": 0.0,
                    }
                    for index, name in enumerate(task_names)
                },
                "roles": {
                    "evaluation": {
                        "items": 6547,
                        "tasks": 15,
                        "valid_items": 0,
                        "micro_accuracy_all": 0.0,
                        "median_task_macro_f1_all": 0.0,
                        "median_task_accuracy_all": 0.0,
                    }
                },
            }
            final_report.write_text(json.dumps(benchmark), encoding="utf-8")
            css_report.write_text(json.dumps(css), encoding="utf-8")
            plan = {
                "plan_version": PLAN_VERSION,
                "status": "commands_only_not_executed",
                "source_root": str(source_root),
                "planner_source_sha256": sha_file(
                    source_root / "scripts/plan_final_eval.py"
                ),
                "auditor_source_sha256": sha_file(
                    source_root / "scripts/audit_final_eval.py"
                ),
                "protocol_source_sha256": PINNED_PROTOCOL_SHA256,
                "freeze_manifest_path": str(freeze),
                "freeze_manifest_sha256": sha_file(freeze),
                "css_evaluation_prompts": {
                    "path": str(css_prompts),
                    "sha256": sha_file(css_prompts),
                },
                "raw_prediction_hashes_path": str(raw),
                "frozen_candidates": [],
                "inference": [
                    {
                        "key": "jev",
                        "model_id": "TypeSafe/jev-1.13.0",
                        "revision": "jev-1.13.0",
                        "backend": "official-api",
                        "predictions": {
                            "final": str(final_pred),
                            "css-evaluation": str(css_pred),
                        },
                        "receipts": {},
                    }
                ],
                "scoring": [
                    {
                        "key": "jev",
                        "reports": {
                            "final": str(final_report),
                            "css-evaluation": str(css_report),
                        },
                    }
                ],
            }
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            plan_sha = sha_file(plan_path)
            # No synthetic final or CSS gold file exists in this fixture.
            with patch("scripts.audit_final_eval._prediction_rows"):
                self.assertEqual(
                    audit(plan_path, expected_plan_sha256=plan_sha)["v2_reports"], 2
                )
                final_pred.write_text("changed\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed after hash freeze"):
                    audit(plan_path, expected_plan_sha256=plan_sha)
                final_pred.write_text(json.dumps(row) + "\n", encoding="utf-8")
                benchmark["schema_version"] = "typed-decision-report/1"
                final_report.write_text(json.dumps(benchmark), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "mismatched v2 synthetic"):
                    audit(plan_path, expected_plan_sha256=plan_sha)
                plan_path.write_text(
                    json.dumps({**plan, "status": "altered"}), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "pre-test recorded SHA-256"):
                    audit(plan_path, expected_plan_sha256=plan_sha)


if __name__ == "__main__":
    unittest.main()
