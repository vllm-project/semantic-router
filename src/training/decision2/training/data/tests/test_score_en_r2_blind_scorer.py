"""Only synthetic oracle, prediction and run fixtures; never inspect r2 private data."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.data import score_en_r2_blind_scorer as scorer
from training.model.data import INPUT_FIELDS, digest, file_sha256
from training.model.infer import prompt_input_sha256


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def select_rows() -> list[dict]:
    result = []
    for kind, count in (("choice", 277), ("noul", 271), ("score", 40)):
        for index in range(count):
            if kind == "choice":
                answer = {
                    "type": kind,
                    "choice": "a",
                    "probabilities": {"a": 1.0, "b": 0.0},
                }
                gold = point = "a"
            elif kind == "noul":
                answer = {"type": kind, "noul": 1.0}
                gold = point = "true"
            else:
                answer = {
                    "type": kind,
                    "score": 0.0,
                    "probabilities": {"0": 1.0, "1": 0.0, "2": 0.0},
                }
                gold = point = "0"
            result.append(
                {
                    "id": f"{kind}-{index}",
                    "task_type": kind,
                    "gold_key": gold,
                    "prediction_key": point,
                    "correct": True,
                    "answer": answer,
                    "brier": 0.0,
                    "prompt_sha256": f"input-{kind}-{index}",
                    "token_ids_sha256": f"tokens-{kind}-{index}",
                }
            )
    return result


def synthetic_panel(root: Path) -> tuple[dict[str, str], Path, list[dict]]:
    prepared = root / "prepared"
    prepared.mkdir()
    key_rows = []
    prompts = []
    labels = {}
    options = [
        {"key": str(level), "description": f"Level {level}"} for level in range(3)
    ]
    for operation in scorer.OPS:
        for group in range(20):
            language = "en" if group < 16 else "zh"
            for level in range(3):
                row_id = f"{operation}-{group}-{level}"
                row = {
                    "id": row_id,
                    "group_id": f"{operation}-{group}",
                    "family": f"score_select_{operation}",
                    "language": language,
                    "split": "select",
                    "evaluation_role": "select",
                    "source": "synthetic_unit_test",
                    "render_template": "unit-test/1",
                    "task_type": "score",
                    "state": f"Case {row_id}",
                    "instructions": "Choose the level.",
                    "options": options,
                    "label": level,
                    "audit_metadata": {"operation": operation},
                }
                row["input_sha256"] = digest(
                    {field: row[field] for field in INPUT_FIELDS}
                )
                key_rows.append(row)
                if language == "en":
                    labels[row_id] = level
                    prompts.append(
                        {
                            "id": row_id,
                            "state": row["state"],
                            "questions": {
                                "decision": {
                                    "type": "score",
                                    "instructions": row["instructions"],
                                    "criteria": [
                                        option["description"] for option in options
                                    ],
                                }
                            },
                        }
                    )
    key = root / "synthetic-key.jsonl"
    write_jsonl(key, key_rows)
    write_jsonl(prepared / "r2_en.native-prompts.jsonl", prompts)
    paths = {"prepared_dir": str(prepared), "seal_sha256": "synthetic-seal"}
    for role in ("source", "A", "B"):
        predictions = []
        for prompt in prompts:
            level = labels[prompt["id"]] if role == "A" else 0
            probabilities = {str(i): float(i == level) for i in range(3)}
            predictions.append(
                {
                    "id": prompt["id"],
                    "answers": {
                        "decision": {
                            "type": "score",
                            "score": float(level),
                            "probabilities": probabilities,
                        }
                    },
                    "adapter_status": "ok",
                    "adapter_errors": {},
                    "input_sha256": prompt_input_sha256(prompt),
                    "source_input_sha256": prompt_input_sha256(prompt),
                    "model_sha256": f"model-{role}",
                    "adapter_sha256": "adapter",
                    "truncated_questions": 0,
                    "usage": {"input_tokens": 16},
                }
            )
        prediction_file = root / f"{role}-prediction.jsonl"
        write_jsonl(prediction_file, predictions)
        paths[f"{role.lower()}_prediction"] = str(prediction_file)
    for arm in ("A", "B"):
        run = root / f"run-{arm}"
        for tag in ("select-baseline", "select-step-0000174"):
            write_jsonl(run / f"{tag}-predictions.jsonl", select_rows())
        paths[f"arm_{arm.lower()}_run"] = str(run)
    return paths, key, prompts


def synthetic_completed_run(root: Path, arm: str) -> Path:
    run = root / f"completed-{arm}"
    checkpoint = run / scorer.FINAL_CHECKPOINT
    write_json(
        run / "COMPLETE.json",
        {
            "status": "complete",
            "step": 174,
            "planned_updates": 174,
            "best": scorer.FINAL_CHECKPOINT,
        },
    )
    write_json(
        run / "LATEST.json", {"checkpoint": scorer.FINAL_CHECKPOINT, "step": 174}
    )
    write_json(run / "BEST.json", {"checkpoint": scorer.FINAL_CHECKPOINT})
    write_json(checkpoint / "checkpoint.json", {"step": 174, "complete": True})
    contract = {
        "init_kind": "decision2-lora",
        "train_mode": "lora",
        "direct_lora_arm": arm,
        "initial_model_sha256": scorer.SOURCE_SHA,
        "direct_lora_parity_sha256": scorer.PARITY_SHA,
        "data_sha256": {
            "train": scorer.ARM_SHA[arm],
            "select": scorer.SELECT_SHA,
            "cal": scorer.CAL_SHA,
        },
        "epochs": 1,
        "max_steps": 174,
        "planned_updates": 174,
        "microbatch": 1,
        "accumulation": 16,
        "max_length": 1024,
        "head_dim": 256,
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.05, "lr": 2e-5},
        "head_lr": 1e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "objective": "ce",
        "replay_fraction": 0.0,
        "replay_kl_weight": 0.0,
        "seed": 20260927,
        "gradient_checkpointing": True,
    }
    code_files = dict.fromkeys(
        (
            "data.py",
            "decision_model.py",
            "loss.py",
            "plan.py",
            "source.py",
            "train.py",
            "lora.py",
            "infer.py",
        ),
        "a" * 64,
    )
    write_json(
        run / "provenance.json",
        {
            "contract": contract,
            "code_sha256": code_files,
            "initial_model_identity": {"model_sha256": scorer.SOURCE_SHA},
            "train_examples": 2777,
            "replay_pool_examples": 0,
            "replay_examples_per_epoch": 0,
            "select_examples": 588,
            "train_tokens": scorer.RAW_TOKENS,
            "train_max_tokens": 1024,
        },
    )
    write_jsonl(
        run / "train-metrics.jsonl",
        [
            {
                "event": "train",
                "step": step,
                "tokens": 2720 if step < 174 else scorer.RAW_TOKENS - 173 * 2720,
            }
            for step in range(1, 175)
        ],
    )
    for tag in ("select-baseline", "select-step-0000174"):
        write_jsonl(run / f"{tag}-predictions.jsonl", select_rows())
    return run


class ScoreEnR2BlindScorerTests(unittest.TestCase):
    def test_score_answer_rejects_invalid_tie_and_missing(self) -> None:
        self.assertFalse(scorer.score_answer(None, 0)["correct"])
        base = {
            "adapter_status": "ok",
            "adapter_errors": {},
            "answers": {
                "decision": {
                    "type": "score",
                    "score": 0.5,
                    "probabilities": {"0": 0.5, "1": 0.5, "2": 0.0},
                }
            },
        }
        self.assertFalse(scorer.score_answer(base, 0)["valid"])
        base["answers"]["decision"]["score"] = 0.0
        self.assertFalse(scorer.score_answer(base, 0)["valid"])
        base["answers"]["decision"] = {
            "type": "score",
            "score": 2.0,
            "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0},
        }
        self.assertEqual(scorer.score_answer(base, 2)["brier"], 0.0)
        base["answers"]["decision"]["probabilities"]["0"] = float("nan")
        self.assertFalse(scorer.score_answer(base, 2)["valid"])

    def test_bootstrap_is_stratified_deterministic_group_resampling(self) -> None:
        groups = {
            operation: {
                f"group-{i:02d}": 3 if operation == scorer.OPS[0] else 0
                for i in range(16)
            }
            for operation in scorer.OPS
        }
        self.assertEqual(scorer.paired_bootstrap(groups), (0.25, 0.25))
        self.assertEqual(scorer.paired_bootstrap(groups), (0.25, 0.25))

    def test_synthetic_full_192_scoring_and_no_gain_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, key, prompts = synthetic_panel(Path(directory))
            with (
                patch.object(
                    scorer,
                    "PROMPT_SHA",
                    file_sha256(
                        Path(paths["prepared_dir"]) / "r2_en.native-prompts.jsonl"
                    ),
                ),
                patch.object(scorer, "KEY_SHA", file_sha256(key)),
            ):
                report = scorer.score_sealed(
                    {"paths": paths, "seal_sha256": "synthetic"}, key
                )
                self.assertEqual(report["status"], "ADVANCE_TO_SINGLE_DEV_CSS_PILOT")
                self.assertEqual(report["models"]["A"]["correct"], 192)
                self.assertEqual(report["models"]["B"]["correct"], 64)
                self.assertEqual(report["a_minus_b_correct_count"], 128)
                self.assertEqual(report["models"]["A"]["all_three_correct_groups"], 64)
                self.assertGreater(
                    report["bootstrap"]["a_minus_b_accuracy_interval_95"][0], 0
                )
                self.assertEqual(len(prompts), 192)
                a = Path(paths["a_prediction"])
                b = Path(paths["b_prediction"])
                damaged = scorer._jsonl(a)
                damaged[0]["answers"] = {}
                write_jsonl(a, damaged)
                report = scorer.score_sealed(
                    {"paths": paths, "seal_sha256": "synthetic"}, key
                )
                self.assertEqual(report["models"]["A"]["invalid"], 1)
                self.assertEqual(report["models"]["A"]["correct"], 191)
                self.assertEqual(report["models"]["A"]["all_three_correct_groups"], 63)
                write_jsonl(a, scorer._jsonl(b))
                report = scorer.score_sealed(
                    {"paths": paths, "seal_sha256": "synthetic"}, key
                )
                self.assertEqual(report["status"], "DO_NOT_ADVANCE")
                self.assertEqual(report["a_minus_b_correct_count"], 0)
                self.assertEqual(
                    report["bootstrap"]["a_minus_b_accuracy_interval_95"], (0, 0)
                )

    def test_native_manifest_rejects_missing_or_mutated_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _, prompts = synthetic_panel(Path(directory))
            prediction = Path(paths["source_prediction"])
            rows = scorer._jsonl(prediction)
            manifest_path = prediction.with_name(prediction.name + ".manifest.json")
            identity = {"model_sha256": "model-source", "files_sha256": {}}
            manifest = {
                "predictions_sha256": file_sha256(prediction),
                "input_sha256": file_sha256(
                    Path(paths["prepared_dir"]) / "r2_en.native-prompts.jsonl"
                ),
                "input_items": 192,
                "adapter_version": scorer.ADAPTER_VERSION,
                "adapter_sha256": "adapter",
                "adapter_files_sha256": dict.fromkeys(
                    (
                        "infer.py",
                        "decision_model.py",
                        "data.py",
                        "lora.py",
                        "source.py",
                    ),
                    "a" * 64,
                ),
                "model_sha256": "model-source",
                "model_files_sha256": {},
                "max_length": 1024,
                "temperature": 1.0,
                "counts": {
                    "items": 192,
                    "questions": 192,
                    "valid_questions": 192,
                    "invalid_questions": 0,
                    "truncated_questions": 0,
                    "over_budget_questions": 0,
                },
            }
            write_json(manifest_path, manifest)
            with patch.object(scorer, "PROMPT_SHA", manifest["input_sha256"]):
                self.assertEqual(
                    len(scorer._predictions(prediction, prompts, identity)[0]), 192
                )
                rows[0]["id"] = "tampered"
                write_jsonl(prediction, rows)
                manifest["predictions_sha256"] = file_sha256(prediction)
                write_json(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, "row/input differs"):
                    scorer._predictions(prediction, prompts, identity)

    def test_fixed_run_gate_rejects_incomplete_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = synthetic_completed_run(Path(directory), "A")
            self.assertEqual(
                scorer._run(run, "A")["checkpoint"].name, scorer.FINAL_CHECKPOINT
            )
            complete = json.loads((run / "COMPLETE.json").read_text())
            complete["step"] = 173
            write_json(run / "COMPLETE.json", complete)
            with self.assertRaisesRegex(ValueError, "not complete"):
                scorer._run(run, "A")

    def test_score_command_checks_full_seal_before_key_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seal = Path(directory) / "seal.json"
            result = Path(directory) / "score.json"
            key = Path(directory) / "key-never-opened.jsonl"
            write_json(
                seal,
                {
                    "schema_version": scorer.VERSION,
                    "status": "SEALED_GOLD_FREE_PENDING_SINGLE_UNBLIND",
                    "scorer_sha256": file_sha256(Path(scorer.__file__)),
                    "paths": {},
                },
            )
            arguments = [
                "scorer",
                "score",
                "--seal",
                str(seal),
                "--seal-sha256",
                file_sha256(seal),
                "--key",
                str(key),
                "--output",
                str(result),
            ]
            with (
                patch.object(sys, "argv", arguments),
                patch.object(scorer, "build_seal", side_effect=ValueError("unsealed")),
                patch.object(scorer, "score_sealed") as scoring,
            ):
                with self.assertRaisesRegex(ValueError, "unsealed"):
                    scorer.main()
                scoring.assert_not_called()
            self.assertFalse(seal.with_name(seal.name + ".unblinded").exists())
            self.assertFalse(result.exists())


if __name__ == "__main__":
    unittest.main()
