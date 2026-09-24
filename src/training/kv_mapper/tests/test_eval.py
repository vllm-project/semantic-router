from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.training.kv_mapper.eval import (
    build_report,
    kv_fit_metrics,
    paired_contrast,
    paired_deltas,
)


class EvalContractTests(unittest.TestCase):
    def test_kv_fit_perfect_map(self) -> None:
        y = np.arange(12, dtype=np.float64).reshape(3, 4)
        got = kv_fit_metrics(y, y)
        self.assertAlmostEqual(got["rel_err"], 0.0, places=6)
        self.assertAlmostEqual(got["cosine"], 1.0, places=6)
        self.assertAlmostEqual(got["r2"], 1.0, places=6)

    def test_kv_fit_rejects_incomplete_or_reshaped_capture(self) -> None:
        for pred, true in (
            ([1, 2], [1, 2, 1000]),
            (np.ones((2, 3)), np.ones((3, 2))),
            ([], []),
        ):
            with self.subTest(pred_shape=np.shape(pred), true_shape=np.shape(true)):
                with self.assertRaisesRegex(ValueError, "matching nonempty shapes"):
                    kv_fit_metrics(pred, true)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            paired_deltas([1.0, 2.0], [1.0])

    def test_paired_ci_recovers_known_offset(self) -> None:
        rng = np.random.default_rng(1)
        cold = rng.normal(3.0, 0.2, size=80)
        mapped = cold - 0.4
        got = paired_contrast(mapped, cold, n_boot=2000, seed=1)
        self.assertEqual(got["n"], 80)
        self.assertAlmostEqual(got["mean"], -0.4, places=6)
        self.assertLess(got["ci_high"], 0.0)
        self.assertGreater(got["ci_low"], -0.5)

    def test_report_contrasts_every_arm(self) -> None:
        def records(scores):
            return [{"id": f"example-{i}", "score": score} for i, score in enumerate(scores)]
        cold = [2.0, 2.1, 1.9, 2.2]
        mapped = [1.6, 1.7, 1.5, 1.8]
        raw = [2.4, 2.5, 2.3, 2.6]
        report = build_report(
            "inject_nll",
            {
                "cold": records(cold),
                "mapped_v1": records(mapped)[::-1],
                "raw_kv": records(raw),
                "random": records([3.0, 3.1, 2.9, 3.2]),
            },
            n_boot=500,
            seed=0,
        )
        self.assertEqual(report["n_items"], 4)
        self.assertEqual(report["reference"], "cold")
        self.assertIn("mapped_v1", report["delta_vs_reference"])
        self.assertIn("raw_kv", report["delta_vs_reference"])
        self.assertNotIn("cold", report["delta_vs_reference"])
        self.assertLess(report["delta_vs_reference"]["mapped_v1"]["mean"], 0.0)

    def test_report_rejects_unpaired_or_duplicate_ids(self) -> None:
        cold = [{"id": "a", "score": 1}, {"id": "b", "score": 2}]
        with self.assertRaisesRegex(ValueError, "example IDs differ"):
            build_report("nll", {"cold": cold, "mapped": [{"id": "a", "score": 1}, {"id": "c", "score": 2}]})
        with self.assertRaisesRegex(ValueError, "duplicate example id"):
            build_report("nll", {"cold": cold, "mapped": [cold[0], cold[0]]})
        with self.assertRaisesRegex(ValueError, "id and score"):
            build_report("nll", {"cold": [1, 2], "mapped": [1, 2]})

    def test_cli_writes_report(self) -> None:
        items = {
            "metric": "hellaswag_acc",
            "reference": "cold",
            "arms": {
                "cold": [{"id": str(i), "score": score} for i, score in enumerate([1, 0, 1, 1])],
                "mapped_v1": [{"id": str(i), "score": score} for i, score in enumerate([1, 1, 1, 1])],
            },
        }
        script = REPO_ROOT / "src/training/kv_mapper/eval_report.py"
        with tempfile.TemporaryDirectory() as directory:
            items_path = Path(directory) / "items.json"
            out_path = Path(directory) / "report.json"
            items_path.write_text(json.dumps(items))
            subprocess.check_call(
                [
                    sys.executable,
                    str(script),
                    "--items",
                    str(items_path),
                    "--output",
                    str(out_path),
                    "--n-boot",
                    "200",
                    "--seed",
                    "0",
                ],
                cwd=str(REPO_ROOT),
            )
            report = json.loads(out_path.read_text())
        self.assertEqual(report["metric"], "hellaswag_acc")
        self.assertEqual(report["n_items"], 4)
        self.assertGreater(report["delta_vs_reference"]["mapped_v1"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
