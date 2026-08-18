"""Dependency-light tests for evaluation I/O and hierarchical orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.training.model_classifier.safety_classifier import evaluate


class EvaluateHelpersTest(unittest.TestCase):
    def test_detect_local_artifact_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").touch()
            merged = root / "merged"
            merged.mkdir()
            (merged / "config.json").touch()

            self.assertEqual(evaluate.detect_local_artifact_type(adapter), "adapter")
            self.assertEqual(evaluate.detect_local_artifact_type(merged), "merged")
            self.assertIsNone(evaluate.detect_local_artifact_type("remote/model"))

    def test_read_prepared_jsonl_and_strict_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prepared.jsonl"
            rows = [
                {
                    "prompt": "one",
                    "label": "safe",
                    "task": "level1",
                    "is_multitarget": False,
                },
                {
                    "text": "two",
                    "label_id": 1,
                    "task": "level2",
                    "is_multitarget": True,
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            selected = evaluate.read_prepared_jsonl(path, task_name="level1")
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["text"], "one")
            self.assertTrue(evaluate._strict_single_target(selected[0]))
            self.assertFalse(evaluate._strict_single_target(rows[1]))

    def test_hierarchical_smoke_routes_only_unsafe_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "prepared.jsonl"
            data_path.write_text(
                json.dumps({"text": "safe prompt", "fingerprint": "a"})
                + "\n"
                + json.dumps({"text": "unsafe prompt", "fingerprint": "b"})
                + "\n",
                encoding="utf-8",
            )
            level1_bundle = evaluate.ModelBundle(None, None, None, "adapter", 2, 512)
            level2_bundle = evaluate.ModelBundle(None, None, None, "merged", 9, 512)

            with (
                mock.patch.object(
                    evaluate,
                    "load_model_bundle",
                    side_effect=[level1_bundle, level2_bundle],
                ),
                mock.patch.object(
                    evaluate,
                    "predict_logits",
                    side_effect=[
                        [[5.0, 0.0], [0.0, 5.0]],
                        [[0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                    ],
                ) as predict,
                mock.patch.object(evaluate, "release_model_bundle"),
            ):
                payload = evaluate.evaluate_hierarchical(
                    level1_model="level1",
                    level2_model="level2",
                    data_path=data_path,
                    output_dir=root / "output",
                )

            self.assertTrue(payload["metrics"]["smoke_passed"])
            self.assertEqual(payload["metrics"]["routed_to_level2"], 1)
            self.assertEqual(predict.call_args_list[1].args[1], ["unsafe prompt"])
            predictions = [
                json.loads(line)
                for line in (root / "output" / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(predictions[0]["final_prediction"], "safe")
            self.assertEqual(predictions[1]["final_prediction"], "S3_sex_crimes")
            self.assertTrue((root / "output" / "metrics.json").is_file())

    def test_hierarchical_smoke_probes_level2_when_everything_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "prepared.jsonl"
            data_path.write_text(json.dumps({"text": "safe"}) + "\n", encoding="utf-8")
            bundles = [
                evaluate.ModelBundle(None, None, None, "merged", 2, 512),
                evaluate.ModelBundle(None, None, None, "merged", 9, 512),
            ]
            with (
                mock.patch.object(evaluate, "load_model_bundle", side_effect=bundles),
                mock.patch.object(
                    evaluate,
                    "predict_logits",
                    side_effect=[
                        [[5.0, 0.0]],
                        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                    ],
                ) as predict,
                mock.patch.object(evaluate, "release_model_bundle"),
            ):
                payload = evaluate.evaluate_hierarchical(
                    level1_model="level1",
                    level2_model="level2",
                    data_path=data_path,
                    output_dir=root / "output",
                )

            self.assertEqual(predict.call_count, 2)
            self.assertEqual(payload["metrics"]["routed_to_level2"], 0)
            self.assertTrue(payload["metrics"]["level2_probe_only"])


if __name__ == "__main__":
    unittest.main()
