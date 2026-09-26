"""CAL-only selection, bounded temperature fit, and inference binding tests."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from training.model.calibrate import selected_run
from training.model.calibration import (
    fit_report,
    load_calibration,
    validate_records,
    verified_materialization_origin,
)
from training.model.data import canonical, file_sha256
from training.model.infer import run_prompts


def examples() -> list[dict]:
    rows = []
    for kind, logits, labels in (
        ("choice", [[4.0, 0.0]] * 8, [0] * 5 + [1] * 3),
        ("noul", [[0.15, 0.0]] * 8, [0] * 8),
        ("score", [[1.0, 0.5, 0.0]] * 8, [0] * 6 + [1] * 2),
    ):
        rows.extend(
            {
                "id": f"{kind}-{index}",
                "task_type": kind,
                "logits": values,
                "label": label,
            }
            for index, (values, label) in enumerate(zip(logits, labels))
        )
    return rows


class CalibrationTest(unittest.TestCase):
    def test_per_type_nll_fit_and_complete_before_after_metrics(self):
        result = fit_report(examples())
        self.assertEqual(
            set(result["temperature_by_type"]), {"choice", "noul", "score"}
        )
        self.assertGreater(result["temperature_by_type"]["choice"], 1.0)
        self.assertLess(result["temperature_by_type"]["noul"], 1.0)
        self.assertEqual(result["overall"]["before"]["n"], 24)
        self.assertEqual(result["overall"]["after"]["n"], 24)
        for kind in ("choice", "noul", "score"):
            before, after = (
                result["by_type"][kind]["before"],
                result["by_type"][kind]["after"],
            )
            self.assertLessEqual(after["nll"], before["nll"] + 1e-10)
            self.assertEqual(before["accuracy_all"], after["accuracy_all"])
            for key in ("nll", "brier", "ece_10"):
                self.assertIn(key, before)
                self.assertIn(key, after)
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_records([row for row in examples() if row["task_type"] != "score"])
        bad = examples()
        bad[0] = dict(bad[0], logits=[float("nan"), 0.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_report(bad)

    def test_calibration_loader_rejects_wrong_model_or_invalid_type_map(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            report = {
                "calibration_version": "decision2-per-type-temperature/1",
                "model_sha256": "a" * 64,
                "checkpoint_sha256": "b" * 64,
                "cal_sha256": "c" * 64,
                "best_sha256": "d" * 64,
                "complete_sha256": "e" * 64,
                "provenance_sha256": "f" * 64,
                "fit_split": "cal",
                "selection_policy": "completed_run_best_only",
                "temperature_by_type": {"choice": 2.0, "noul": 0.6, "score": 1.2},
            }
            path.write_text(json.dumps(report))
            temperatures, loaded = load_calibration(path, "a" * 64)
            self.assertEqual(temperatures["choice"], 2.0)
            self.assertEqual(loaded["cal_sha256"], "c" * 64)
            with self.assertRaisesRegex(ValueError, "model hash differs"):
                load_calibration(path, "0" * 64)
            report["temperature_by_type"].pop("noul")
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "exact Choice/Noul/Score"):
                load_calibration(path, "a" * 64)

    def test_materialized_lineage_binds_original_calibration_without_rewriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged = root / "merged"
            merged.mkdir()
            source_sha = "a" * 64
            files = {
                "backbone/model.safetensors": "b" * 64,
                "decision_config.json": "c" * 64,
            }
            merged_sha = hashlib.sha256(canonical(files).encode()).hexdigest()
            (merged / "decision_config.json").write_text(
                json.dumps(
                    {
                        "initialization": "merged-peft-lora",
                        "lora_origin": {"adapter": {"model_sha256": source_sha}},
                    }
                )
            )
            receipt_path = root / "merged.materialization.json"
            receipt = {
                "materialization_version": "decision2-merged-peft-lora/1",
                "source_model_sha256": source_sha,
                "merged_model_sha256": merged_sha,
                "merged_model_files_sha256": files,
            }
            receipt_path.write_text(json.dumps(receipt))
            lineage = verified_materialization_origin(merged, merged_sha)
            self.assertEqual(lineage["source_model_sha256"], source_sha)
            self.assertEqual(lineage["receipt_sha256"], file_sha256(receipt_path))
            portable = merged / "materialization_receipt.json"
            portable.write_bytes(receipt_path.read_bytes())
            receipt_path.unlink()
            self.assertEqual(
                verified_materialization_origin(merged, merged_sha)[
                    "source_model_sha256"
                ],
                source_sha,
            )
            calibration = root / "calibration.json"
            calibration.write_text(
                json.dumps(
                    {
                        "calibration_version": "decision2-per-type-temperature/1",
                        "model_sha256": source_sha,
                        "checkpoint_sha256": "d" * 64,
                        "cal_sha256": "e" * 64,
                        "best_sha256": "f" * 64,
                        "complete_sha256": "1" * 64,
                        "provenance_sha256": "2" * 64,
                        "fit_split": "cal",
                        "selection_policy": "completed_run_best_only",
                        "temperature_by_type": {
                            "choice": 1.2,
                            "noul": 0.9,
                            "score": 1.1,
                        },
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "model hash differs"):
                load_calibration(calibration, merged_sha)
            temperatures, _ = load_calibration(
                calibration,
                merged_sha,
                materialized_source_sha256=lineage["source_model_sha256"],
            )
            self.assertEqual(temperatures["choice"], 1.2)
            receipt["source_model_sha256"] = "3" * 64
            portable.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "does not bind"):
                verified_materialization_origin(merged, merged_sha)

    def test_completed_best_and_original_cal_bytes_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            checkpoint = run / "checkpoint-0000007"
            checkpoint.mkdir(parents=True)
            cal = Path(directory) / "cal.jsonl"
            cal.write_text("audited CAL bytes\n")
            (run / "BEST.json").write_text(json.dumps({"checkpoint": checkpoint.name}))
            (run / "COMPLETE.json").write_text(
                json.dumps({"status": "complete", "best": checkpoint.name})
            )
            (checkpoint / "checkpoint.json").write_text(json.dumps({"complete": True}))
            (run / "provenance.json").write_text(
                json.dumps(
                    {
                        "contract": {
                            "data_sha256": {"cal": file_sha256(cal)},
                            "max_length": 1024,
                        },
                        "cal_examples_audited_only": 3,
                        "model_source": {
                            "files_sha256": {"backbone/model.safetensors": "a" * 64}
                        },
                    }
                )
            )
            selected = selected_run(run, cal)
            self.assertEqual(selected["name"], checkpoint.name)
            self.assertEqual(selected["cal_sha256"], file_sha256(cal))
            cal.write_text("changed CAL bytes\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                selected_run(run, cal)
            cal.write_text("audited CAL bytes\n")
            (run / "COMPLETE.json").write_text(
                json.dumps({"status": "running", "best": checkpoint.name})
            )
            with self.assertRaisesRegex(ValueError, "completed run"):
                selected_run(run, cal)

    def test_uncalibrated_temperature_one_remains_same(self):
        prompt = {
            "id": "p",
            "state": "s",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "choose",
                    "criteria": {"a": "A", "b": "B"},
                },
                "n": {
                    "type": "noul",
                    "instructions": "yes?",
                    "criteria": {"false": "No", "true": "Yes"},
                },
                "s": {
                    "type": "score",
                    "instructions": "rate",
                    "criteria": ["Low", "High"],
                },
            },
        }

        def encode(row, _tokenizer, _length):
            return {
                "id": row["id"],
                "ids": [1, 2],
                "keys": [option["key"] for option in row["options"]],
            }

        def predict(rows):
            return [
                [0.0, 2.0] if row["id"].endswith("/n") else [2.0, 0.0] for row in rows
            ]

        kw = {
            "tokenizer": None,
            "max_length": 16,
            "encode_fn": encode,
            "predict_fn": predict,
            "model_sha256": "model",
            "adapter_sha256": "adapter",
        }
        plain, _ = run_prompts([prompt], temperature=1.0, **kw)
        typed, _ = run_prompts(
            [prompt], temperature={"choice": 1.0, "noul": 1.0, "score": 1.0}, **kw
        )
        self.assertEqual(plain[0]["answers"], typed[0]["answers"])
        self.assertNotIn("calibration_sha256", plain[0])
        applied, _ = run_prompts(
            [prompt],
            temperature={"choice": 2.0, "noul": 0.5, "score": 4.0},
            calibration_sha256="c" * 64,
            **kw,
        )
        self.assertIn("calibration_sha256", applied[0])
        self.assertLess(
            applied[0]["answers"]["q"]["probabilities"]["a"],
            plain[0]["answers"]["q"]["probabilities"]["a"],
        )
        self.assertGreater(
            applied[0]["answers"]["n"]["noul"], plain[0]["answers"]["n"]["noul"]
        )
        self.assertGreater(
            applied[0]["answers"]["s"]["score"], plain[0]["answers"]["s"]["score"]
        )


if __name__ == "__main__":
    unittest.main()
