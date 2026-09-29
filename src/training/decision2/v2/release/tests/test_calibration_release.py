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


class CalibrateFrozenCliTest(unittest.TestCase):
    """main() wiring with the GPU pieces mocked."""

    IDENTITY = {
        "model_sha256": "7" * 64,
        "files_sha256": {"decision_config.json": "a" * 64},
    }
    KERNELS = {
        "kernel_bindings": {
            "torch_chunk_gated_delta_rule": "fla.ops.chunk_gated_delta_rule"
        },
        "triton_cache_dir": "/cache",
        "triton_cache_autotuning": "1",
    }

    def run_main(self, scratch: Path, *extra: str):
        from unittest import mock

        cal = scratch / "cal.jsonl"
        cal.write_text('{"id": "c0"}\n')
        output = scratch / "calibration.json"
        records = [
            {
                "id": f"c{i}",
                "task_type": kind,
                "label": i % 2,
                "logits": [0.5 * i, -0.2],
            }
            for i, kind in enumerate(("choice", "noul", "score") * 10)
        ]
        argv = [
            "calibrate_frozen",
            "--checkpoint",
            str(scratch / "ckpt"),
            "--cal",
            str(cal),
            "--cal-sha256",
            calibrate_frozen.file_sha256(cal),
            "--output",
            str(output),
            *extra,
        ]
        calls = []
        with mock.patch("sys.argv", argv), mock.patch(
            "training.model.data.load_partition", return_value=[{"id": "c0"}]
        ), mock.patch(
            "training.model.infer.checkpoint_fingerprint",
            side_effect=lambda path, source=None: calls.append(("fingerprint", source))
            or self.IDENTITY,
        ), mock.patch.object(
            calibrate_frozen,
            "cal_logits",
            side_effect=lambda *a: calls.append(("logits", a))
            or (records, {"torch": "t"}),
        ), mock.patch(
            "v2.dec.runtime_check.require_runtime",
            side_effect=lambda: calls.append(("kernels",)) or self.KERNELS,
        ), mock.patch(
            "builtins.print"
        ):
            calibrate_frozen.main()
        return json.loads(output.read_text()), calls

    def test_default_records_no_kernel_identity_and_no_source(self):
        with tempfile.TemporaryDirectory() as scratch:
            report, calls = self.run_main(Path(scratch))
        self.assertEqual([c[0] for c in calls], ["fingerprint", "logits"])
        self.assertIsNone(calls[0][1])
        self.assertEqual(calls[1][1][2:], (8192, 1, None))
        self.assertEqual(
            set(report["inference"]),
            {"max_length", "batch_size", "device", "execution", "runtime"},
        )

    def test_adapter_source_kernels_and_package_limit_are_recorded(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch) / "base"
            report, calls = self.run_main(
                Path(scratch),
                "--source-path",
                str(base),
                "--require-kernels",
                "--max-length",
                "32768",
            )
            path = Path(scratch) / "calibration.json"
            _, loaded = load_calibration(path, self.IDENTITY["model_sha256"])
        self.assertEqual([c[0] for c in calls], ["kernels", "fingerprint", "logits"])
        self.assertEqual(calls[1][1], base)
        self.assertEqual(calls[2][1][2:], (32768, 1, base))
        self.assertEqual(report["inference"]["kernel_runtime"], self.KERNELS)
        self.assertEqual(loaded["inference"]["max_length"], 32768)
        self.assertEqual(report["model_sha256"], self.IDENTITY["model_sha256"])

    def test_failed_kernel_check_stops_before_loading(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as scratch, mock.patch(
            "v2.dec.runtime_check.require_runtime",
            side_effect=RuntimeError("Decoder runtime check failed"),
        ), mock.patch.object(calibrate_frozen, "cal_logits") as logits:
            scratch = Path(scratch)
            (scratch / "cal.jsonl").write_text("{}\n")
            argv = [
                "calibrate_frozen",
                "--checkpoint",
                str(scratch / "ckpt"),
                "--cal",
                str(scratch / "cal.jsonl"),
                "--cal-sha256",
                calibrate_frozen.file_sha256(scratch / "cal.jsonl"),
                "--output",
                str(scratch / "calibration.json"),
                "--require-kernels",
            ]
            with mock.patch("sys.argv", argv), self.assertRaises(RuntimeError):
                calibrate_frozen.main()
            logits.assert_not_called()
            self.assertFalse((scratch / "calibration.json").exists())


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
