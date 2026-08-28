#!/usr/bin/env python3
"""Tests for MoM evaluation runner helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.mom_eval.collect_results import build_result_bundle
from bench.mom_eval.compare_regression import compare_regression
from bench.mom_eval.publish_scorecard import build_scorecard_json


class MomEvalRunnerTest(unittest.TestCase):
    def test_build_result_bundle_for_blend(self) -> None:
        manifest_path = REPO_ROOT / "config/recipes/built-in/latest/mom-v1/mom-evaluation.yaml"
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            core_dir = raw_dir / "core"
            core_dir.mkdir(parents=True)
            (core_dir / "gpqa_d.json").write_text(
                json.dumps({"value": 80.0, "num_samples": 20}) + "\n",
                encoding="utf-8",
            )
            baseline_dir = raw_dir / "baselines"
            baseline_dir.mkdir(parents=True)
            (baseline_dir / "balanced_standalone.json").write_text(
                json.dumps({"metrics": {"gpqa_d": {"value": 79.0}}}) + "\n",
                encoding="utf-8",
            )
            bundle = build_result_bundle(
                manifest_path,
                "vllm-sr/mom-v1-blend",
                raw_dir,
                "smoke",
                "test-run",
            )
            self.assertEqual(bundle["schema_version"], "vllm-sr/mom-eval-result/v1")
            self.assertEqual(bundle["identity"]["entrypoint"], "vllm-sr/mom-v1-blend")
            self.assertIn("gpqa_d", bundle["metrics"])

    def test_compare_regression_without_previous_passes(self) -> None:
        current = {
            "identity": {
                "recipe_id": "nonexistent-recipe",
                "entrypoint": "vllm-sr/mom-v1-blend",
                "recipe_version": "9.9.9",
            },
            "metrics": {"gpqa_d": {"value": 80.0}},
            "publication": {"publishable": True, "classification": "launch", "blocking_reasons": []},
        }
        report = compare_regression(current)
        self.assertTrue(report["passed"])

    def test_build_scorecard_json(self) -> None:
        result_path = (
            REPO_ROOT
            / "config/evaluation/scorecards/mom-v1/mom-v1-blend/1.0.0/mom_eval_result.json"
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        scorecard = build_scorecard_json(result)
        self.assertEqual(scorecard["schema_version"], "vllm-sr/mom-scorecard/v1")
        self.assertTrue(scorecard["metrics"])


if __name__ == "__main__":
    unittest.main()
