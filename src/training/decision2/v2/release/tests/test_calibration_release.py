"""Frozen-checkpoint calibration report and temperature parity (stdlib, synthetic logits)."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from training.model.calibration import load_calibration
from training.model.infer import normalized_answer
from v2.release import calibrate_frozen, dev_calibration, temperature_parity

TEMPERATURES = {"choice": 1.37, "noul": 0.82, "score": 2.4}


def synthetic(seed: int = 7):
    rng = random.Random(seed)
    prompts, raw, cal = [], [], []
    for index in range(60):
        kind = ("choice", "noul", "score")[index % 3]
        keys = (
            ["a", "b", "c", "d"][: 2 + index % 3]
            if kind == "choice"
            else (
                ["false", "true"]
                if kind == "noul"
                else [str(i) for i in range(3 + index % 4)]
            )
        )
        logits = [rng.uniform(-9, 9) for _ in keys]
        pid, qid = f"p{index}", "q"
        prompts.append({"id": pid, "questions": {qid: {"type": kind}}})
        for rows, t in ((raw, 1.0), (cal, TEMPERATURES[kind])):
            rows.append(
                {
                    "id": pid,
                    "source_input_sha256": f"{index:064x}",
                    "answers": {qid: normalized_answer(kind, keys, logits, t)},
                }
            )
    return prompts, raw, cal


class TemperatureParityTest(unittest.TestCase):
    def test_temperature_keeps_every_answer_and_offline_recomputation_matches(self):
        prompts, raw, cal = synthetic()
        totals = temperature_parity.compare(
            raw, cal, temperature_parity.question_kinds(prompts), TEMPERATURES
        )
        self.assertEqual(totals["slots"], 60)
        self.assertEqual(totals["category_changes"], 0)
        self.assertEqual(totals["offline_category_changes"], 0)
        self.assertLess(totals["offline_max_abs_drift"], 1e-12)
        self.assertGreater(totals["raw_vs_calibrated_max_abs_change"], 1e-3)

    def test_a_changed_answer_is_counted(self):
        prompts, raw, cal = synthetic()
        row = next(r for r in cal if r["answers"]["q"]["type"] == "choice")
        answer = row["answers"]["q"]
        other = next(k for k in answer["probabilities"] if k != answer["choice"])
        answer["choice"] = other
        totals = temperature_parity.compare(
            raw, cal, temperature_parity.question_kinds(prompts), TEMPERATURES
        )
        self.assertEqual(totals["category_changes"], 1)


class CalibrateFrozenTest(unittest.TestCase):
    def test_report_is_accepted_by_the_inference_contract(self):
        rng = random.Random(3)
        records = []
        for index in range(90):
            kind = ("choice", "noul", "score")[index % 3]
            n = 3 if kind != "noul" else 2
            label = rng.randrange(n)
            logits = [rng.gauss(0, 1) + (2.5 if i == label else 0.0) for i in range(n)]
            records.append(
                {"id": f"c{index}", "task_type": kind, "label": label, "logits": logits}
            )
        identity = {
            "model_sha256": "5" * 64,
            "files_sha256": {"decision_config.json": "a" * 64},
        }
        report = calibrate_frozen.build_report(
            records, identity, "c" * 64, {"max_length": 8192, "batch_size": 1}
        )
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "calibration.json"
            path.write_text(json.dumps(report))
            temperatures, loaded = load_calibration(path, "5" * 64)
            with self.assertRaises(ValueError):
                load_calibration(path, "6" * 64)
        self.assertEqual(set(temperatures), {"choice", "noul", "score"})
        self.assertEqual(loaded["selection_policy"], "frozen_checkpoint")
        self.assertEqual(loaded["inference"]["max_length"], 8192)


class DevCalibrationTest(unittest.TestCase):
    def test_stored_calibrated_rows_return_to_raw_then_to_the_candidate(self):
        prompts, raw, cal = synthetic()
        kinds = temperature_parity.question_kinds(prompts)
        back = dev_calibration.rescale(cal, kinds, TEMPERATURES, invert=True)
        self.assertEqual(dev_calibration.answer_changes(raw, back), 0)
        forward = dev_calibration.rescale(back, kinds, TEMPERATURES)
        self.assertEqual(dev_calibration.answer_changes(cal, forward), 0)
        drift = temperature_parity.compare(
            raw, back, kinds, {k: 1.0 for k in TEMPERATURES}
        )
        self.assertLess(drift["offline_max_abs_drift"], 1e-12)

    def test_rule_rejects_any_worse_criterion(self):
        raw = {
            "typed_dev": {"brier": 0.30, "ece_10": 0.10},
            "css_pilot": {
                "median_task_brier_sum": 0.50,
                "median_task_ece_pmax_15": 0.05,
            },
        }
        better = {
            "typed_dev": {"brier": 0.29, "ece_10": 0.09},
            "css_pilot": {
                "median_task_brier_sum": 0.50,
                "median_task_ece_pmax_15": 0.04,
            },
        }
        self.assertTrue(dev_calibration.decide(raw, better)["adopt"])
        worse = {
            **better,
            "css_pilot": {
                "median_task_brier_sum": 0.51,
                "median_task_ece_pmax_15": 0.04,
            },
        }
        decision = dev_calibration.decide(raw, worse)
        self.assertFalse(decision["adopt"])
        self.assertEqual(decision["worsened"], ["css_pilot_brier"])


if __name__ == "__main__":
    unittest.main()
