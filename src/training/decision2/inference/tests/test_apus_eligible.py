from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transfer.build import EVALUATION_TASKS

from inference.apus_eligible import (
    build_plan,
    check_freeze_binding,
    model_summary,
    panel_info,
    summarize,
    validate_inputs,
)
from inference.run import digest, file_digest

SOURCE_ROOT = Path(__file__).resolve().parents[2]
CHOICE = {"type": "choice", "instructions": "Choose", "criteria": {"a": "A", "b": "B"}}
NOUL = {
    "type": "noul",
    "instructions": "Is it true?",
    "criteria": {"true": "It is true", "false": "It is false"},
}
SCORE = {"type": "score", "instructions": "Rate", "criteria": ["low", "mid", "high"]}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


class APUSFinalAppendixTest(unittest.TestCase):
    def test_freeze_hash_and_selection_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "freeze.json"
            candidate = {
                "key": "d2-9b",
                "model_id": "llm-semantic-router/dev-2.0-9b",
                "selected_checkpoint": "checkpoint-1",
                **dict.fromkeys(
                    (
                        "model_sha256",
                        "calibration_sha256",
                        "best_sha256",
                        "complete_sha256",
                        "provenance_sha256",
                    ),
                    "a" * 64,
                ),
            }
            freeze = {
                "freeze_version": "decision2-pretest-freeze/1",
                "selection_sources": ["train", "select", "cal", "css_pilot"],
                "candidates": [candidate],
            }
            path.write_text(json.dumps(freeze), encoding="utf-8")
            main = {
                "freeze_manifest_path": str(path),
                "freeze_manifest_sha256": file_digest(path),
                "frozen_candidates": [candidate],
            }
            self.assertEqual(check_freeze_binding(main, path), file_digest(path))
            freeze["selection_sources"].append("css_evaluation")
            path.write_text(json.dumps(freeze), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "freeze bytes"):
                check_freeze_binding(main, path)

    def test_gold_free_final_question_denominator_and_css_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "final.prompts.jsonl"
            css = root / "css-evaluation.prompts.jsonl"
            rows = []
            for i in range(1600):
                questions = (
                    {"choice": CHOICE, "noul": NOUL}
                    if i < 400
                    else (
                        {"decision": CHOICE}
                        if i < 800
                        else {"decision": NOUL} if i < 1200 else {"decision": SCORE}
                    )
                )
                rows.append(
                    {"id": f"final-{i}", "state": "state", "questions": questions}
                )
            write_jsonl(final, rows)
            write_jsonl(
                css,
                [
                    {
                        "id": f"css/{EVALUATION_TASKS[i % 15]}/{i}",
                        "state": "state",
                        "questions": {"label": CHOICE},
                    }
                    for i in range(6547)
                ],
            )
            final_info = panel_info(final, final=True)
            css_info = panel_info(css, final=False)
            self.assertEqual(final_info["items"], 1600)
            self.assertEqual(final_info["questions"], 2000)
            self.assertEqual(final_info["eligible_questions"], 1600)
            self.assertEqual(final_info["unsupported_score_questions"], 400)
            self.assertEqual(css_info["task_count"], 15)
            self.assertEqual(css_info["eligible_questions"], 6547)
            rows[0]["gold"] = {"choice": "a"}
            write_jsonl(final, rows)
            with self.assertRaisesRegex(ValueError, "gold-bearing"):
                panel_info(final, final=True)

    def test_main_plan_freeze_required_and_commands_keep_apus_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation_root = root / "main-final"
            evaluation_root.mkdir()
            final = evaluation_root / "final.prompts.jsonl"
            css = root / "css-evaluation.prompts.jsonl"
            write_jsonl(
                final,
                [
                    {
                        "id": f"final-{i}",
                        "state": "s",
                        "questions": {
                            "decision": (
                                CHOICE if i < 800 else NOUL if i < 1200 else SCORE
                            )
                        },
                    }
                    for i in range(1600)
                ],
            )
            write_jsonl(
                css,
                [
                    {
                        "id": f"css/{EVALUATION_TASKS[i % 15]}/{i}",
                        "state": "s",
                        "questions": {"label": CHOICE},
                    }
                    for i in range(6547)
                ],
            )
            freeze_path = root / "freeze.json"
            entry = {
                "key": "d2-9b",
                "model_id": "llm-semantic-router/dev-2.0-9b",
                "selected_checkpoint": "checkpoint-1",
                **dict.fromkeys(
                    (
                        "model_sha256",
                        "calibration_sha256",
                        "best_sha256",
                        "complete_sha256",
                        "provenance_sha256",
                    ),
                    "a" * 64,
                ),
            }
            freeze_path.write_text(
                json.dumps(
                    {
                        "freeze_version": "decision2-pretest-freeze/1",
                        "selection_sources": ["select", "cal"],
                        "candidates": [entry],
                    }
                ),
                encoding="utf-8",
            )
            main_path = root / "main-plan.json"
            protocol = {
                key: file_digest(SOURCE_ROOT / key)
                for key in (
                    "inference/run.py",
                    "benchmark/score.py",
                    "transfer/score.py",
                    "transfer/build.py",
                )
            }
            main = {
                "plan_version": "decision2-final-evaluation-plan/2",
                "status": "commands_only_not_executed",
                "freeze_manifest_path": str(freeze_path),
                "freeze_manifest_sha256": file_digest(freeze_path),
                "frozen_candidates": [entry],
                "raw_prediction_hashes_path": str(
                    evaluation_root / "RAW_PREDICTIONS.sha256"
                ),
                "css_evaluation_prompts": {
                    "path": str(css),
                    "sha256": file_digest(css),
                    "items": 6547,
                },
                "protocol_source_sha256": protocol,
            }
            main_path.write_text(json.dumps(main), encoding="utf-8")
            with patch("inference.apus_eligible.CSS_SHA256", file_digest(css)):
                inputs = validate_inputs(
                    main_plan=main_path,
                    main_plan_sha256=file_digest(main_path),
                    freeze=freeze_path,
                    final_prompts=final,
                    css_prompts=css,
                    source_root=SOURCE_ROOT,
                    output_root=root / "apus-out",
                    plan_path=root / "apus-plan.json",
                )
            self.assertEqual(inputs["final_panel"]["unsupported_score_questions"], 400)
            plan = build_plan(
                inputs=inputs,
                main_plan=main_path,
                freeze=freeze_path,
                source_root=SOURCE_ROOT,
                model_root=root / "models",
                output_root=root / "apus-out",
                plan_path=root / "apus-plan.json",
                python="python3",
            )
            self.assertEqual(len(plan["inference_commands"]), 4)
            self.assertEqual(len(plan["scoring_commands"]), 4)
            self.assertIn("--verify-only", plan["preflight_commands"][0])
            self.assertIn("--effort high", plan["inference_commands"][0])
            self.assertNotIn(
                "rank", " ".join(plan["inference_commands"] + plan["scoring_commands"])
            )
            self.assertIn("final.gold.jsonl", plan["scoring_commands"][0])
            self.assertNotIn("final.gold.jsonl", plan["inference_commands"][0])
            with self.assertRaisesRegex(ValueError, "Main final plan SHA-256"):
                validate_inputs(
                    main_plan=main_path,
                    main_plan_sha256="b" * 64,
                    freeze=freeze_path,
                    final_prompts=final,
                    css_prompts=css,
                    source_root=SOURCE_ROOT,
                    output_root=root / "unused",
                    plan_path=root / "unused-plan.json",
                )

    def test_result_keeps_unsupported_score_in_full_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "final.prompts.jsonl"
            css = root / "css-evaluation.prompts.jsonl"
            final_rows = [
                {
                    "id": "f1",
                    "state": "state",
                    "questions": {"choice": CHOICE, "noul": NOUL, "score": SCORE},
                },
                {"id": "f2", "state": "state", "questions": {"choice": CHOICE}},
            ]
            css_rows = [
                {
                    "id": "css/emotion/1",
                    "state": "state",
                    "questions": {"label": CHOICE},
                }
            ]
            write_jsonl(final, final_rows)
            write_jsonl(css, css_rows)
            model = {
                "model_id": "apus-ailab/APUS-OpenJev-v1-4B",
                "revision": "rev",
                "adapter_version": "adapter",
                "effort": "high",
                "release_manifest_sha256": "a" * 64,
                "model_config_sha256": "b" * 64,
                "predictions": {
                    "final": str(root / "final.predictions.jsonl"),
                    "css-evaluation": str(root / "css.predictions.jsonl"),
                },
                "reports": {
                    "final": str(root / "final.report.json"),
                    "css-evaluation": str(root / "css.report.json"),
                },
            }
            identity = {
                "model_id": model["model_id"],
                "model_revision": model["revision"],
                "adapter_version": model["adapter_version"],
                "effort": "high",
                "release_manifest_sha256": model["release_manifest_sha256"],
                "model_config_sha256": model["model_config_sha256"],
            }

            def receipt(row: dict, answers: dict) -> dict:
                return {
                    "id": row["id"],
                    "answers": answers,
                    "source_input_sha256": digest(
                        {"state": row["state"], "questions": row["questions"]}
                    ),
                    **identity,
                }

            final_receipts = [
                receipt(
                    final_rows[0],
                    {
                        "choice": {"type": "choice", "choice": "a"},
                        "noul": {"type": "noul", "noul": 0.2},
                        "score": {
                            "type": "score",
                            "error": "unsupported_native_ordinal_score",
                        },
                    },
                ),
                receipt(final_rows[1], {"choice": {"type": "choice", "choice": "b"}}),
            ]
            write_jsonl(Path(model["predictions"]["final"]), final_receipts)
            write_jsonl(
                Path(model["predictions"]["css-evaluation"]),
                [receipt(css_rows[0], {"label": {"type": "choice", "choice": "a"}})],
            )
            metrics = lambda n, correct, valid: {
                "n": n,
                "correct_n": correct,
                "valid_n": valid,
                "invalid_or_missing_n": n - valid,
                "accuracy_all": correct / n,
            }
            Path(model["reports"]["final"]).write_text(
                json.dumps(
                    {
                        "schema_version": "typed-decision-report/2",
                        "split": "final",
                        "model": {
                            "id": model["model_id"],
                            "revision": "rev",
                            "backend": "native-openjet-high",
                        },
                        "predictions_sha256": file_digest(
                            Path(model["predictions"]["final"])
                        ),
                        "gold_sha256": "c" * 64,
                        "items": 2,
                        "predicted_items": 2,
                        "overall": metrics(4, 1, 3),
                        "by_type": {
                            "choice": metrics(2, 1, 2),
                            "noul": metrics(1, 0, 1),
                            "score": metrics(1, 0, 0),
                        },
                    }
                ),
                encoding="utf-8",
            )
            Path(model["reports"]["css-evaluation"]).write_text(
                json.dumps(
                    {
                        "score_schema_version": "css-transfer-score/2",
                        "gold_sha256": "d" * 64,
                        "predictions_sha256": file_digest(
                            Path(model["predictions"]["css-evaluation"])
                        ),
                        "roles": {
                            "evaluation": {
                                "items": 1,
                                "tasks": 15,
                                "valid_items": 1,
                                "micro_accuracy_all": 1.0,
                            }
                        },
                        "tasks": {task: {} for task in EVALUATION_TASKS},
                    }
                ),
                encoding="utf-8",
            )
            plan = {
                "final_panel": {
                    "path": str(final),
                    "items": 2,
                    "questions": 4,
                    "eligible_questions": 3,
                    "question_types": {"choice": 2, "noul": 1, "score": 1},
                },
                "css_panel": {"path": str(css), "items": 1},
            }
            result = model_summary(model, plan)
            self.assertEqual(result["final_full"]["accuracy_all"], 0.25)
            self.assertEqual(
                result["final_choice_noul_eligible"]["accuracy_all"], 1 / 3
            )
            final_receipts[0]["answers"]["score"] = {"type": "score", "score": 2}
            write_jsonl(Path(model["predictions"]["final"]), final_receipts)
            with self.assertRaisesRegex(ValueError, "falsely presented"):
                model_summary(model, plan)

    def test_summary_requires_original_prediction_hash_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main_path = root / "main-plan.json"
            main_path.write_text("{}\n", encoding="utf-8")
            freeze_path = root / "freeze.json"
            freeze_path.write_text("{}\n", encoding="utf-8")
            final_panel = {
                "path": str(root / "final.prompts.jsonl"),
                "sha256": "a" * 64,
            }
            css_panel = {"path": str(root / "css.prompts.jsonl"), "sha256": "b" * 64}
            paths = [
                root / f"{size}.{panel}.jsonl"
                for size in ("4b", "9b")
                for panel in ("final", "css-evaluation")
            ]
            for path in paths:
                path.write_text("prediction bytes\n", encoding="utf-8")
            ledger_path = root / "RAW_PREDICTIONS.sha256"
            ledger_path.write_text(
                "".join(f"{file_digest(path)}  {path}\n" for path in paths),
                encoding="utf-8",
            )
            model = lambda size, i: {
                "size": size,
                "predictions": {
                    "final": str(paths[i]),
                    "css-evaluation": str(paths[i + 1]),
                },
            }
            plan_path = root / "apus-plan.json"
            plan = {
                "plan_version": "apus-native-eligible-final-plan/1",
                "plan_path": str(plan_path),
                "main_plan_path": str(main_path),
                "main_plan_sha256": file_digest(main_path),
                "freeze_manifest_path": str(freeze_path),
                "freeze_manifest_sha256": "f" * 64,
                "final_panel": final_panel,
                "css_panel": css_panel,
                "source_root": str(SOURCE_ROOT),
                "source_sha256": {},
                "models": [model("4b", 0), model("9b", 2)],
                "raw_prediction_hashes_path": str(ledger_path),
                "scope": "eligible only",
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            report = {
                "gold_sha256_from_reports": {
                    "final": "c" * 64,
                    "css_evaluation": "d" * 64,
                }
            }
            with (
                patch(
                    "inference.apus_eligible.check_freeze_binding",
                    return_value="f" * 64,
                ),
                patch(
                    "inference.apus_eligible.panel_info",
                    side_effect=[final_panel, css_panel],
                ),
                patch("inference.apus_eligible.source_hashes", return_value={}),
                patch("inference.apus_eligible.model_summary", return_value=report),
            ):
                result = summarize(plan_path, file_digest(plan_path))
            self.assertEqual(len(result["models"]), 2)
            paths[0].write_text("changed prediction bytes\n", encoding="utf-8")
            with (
                patch(
                    "inference.apus_eligible.check_freeze_binding",
                    return_value="f" * 64,
                ),
                patch(
                    "inference.apus_eligible.panel_info",
                    side_effect=[final_panel, css_panel],
                ),
                patch("inference.apus_eligible.source_hashes", return_value={}),
            ):
                with self.assertRaisesRegex(ValueError, "ledger differs"):
                    summarize(plan_path, file_digest(plan_path))


if __name__ == "__main__":
    unittest.main()
