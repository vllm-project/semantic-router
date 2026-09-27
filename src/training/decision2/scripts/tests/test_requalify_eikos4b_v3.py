"""Fail-closed checks for the prospective 4B technical requalification."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from inference.run import digest
from scripts import requalify_eikos4b_v3 as requalify
from scripts.eikos_stable_runtime_v3 import FLA_BACKEND, TORCH_BACKEND
from scripts.plan_final_eval import sha_file


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class RequalifyEikos4BTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prompts = self.root / "prompts.jsonl"
        self.predictions = self.root / "predictions.jsonl"
        self.items = [
            {
                "id": f"example-{index}",
                "state": f"state-{index}",
                "questions": {
                    "decision": {
                        "type": "noul",
                        "criteria": {"true": "yes", "false": "no"},
                    }
                },
            }
            for index in range(2)
        ]
        write_jsonl(self.prompts, self.items)
        self.rows = [
            {
                "id": item["id"],
                "answers": {
                    "decision": {"type": "noul", "value": True, "probability": 0.8}
                },
                "source_input_sha256": digest(
                    {"state": item["state"], "questions": item["questions"]}
                ),
                "model_id": requalify.MODEL_ID,
                "model_revision": requalify.CHECKPOINT,
                "model_sha256": requalify.PINNED_SHA256["package/SHA256SUMS"],
                "usage": {"input_tokens": 10},
                "invalid_reason": None,
            }
            for item in self.items
        ]
        write_jsonl(self.predictions, self.rows)

    def test_preflight_rejects_lowercase_id_before_reading_inputs(self) -> None:
        config_path = self.root / "wrong-id.json"
        config_path.write_text(
            json.dumps({"model_id": "llm-semantic-router/dev-2.0-4b"}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Public model ID"):
            requalify.preflight(config_path)

    def test_native_rows_require_exact_order_identity_and_input_digest(self) -> None:
        requalify._verify_rows(
            self.predictions,
            self.prompts,
            2,
            requalify.PINNED_SHA256["package/SHA256SUMS"],
        )
        write_jsonl(self.predictions, list(reversed(self.rows)))
        with self.assertRaisesRegex(ValueError, "Invalid/stale"):
            requalify._verify_rows(
                self.predictions,
                self.prompts,
                2,
                requalify.PINNED_SHA256["package/SHA256SUMS"],
            )
        self.rows[0]["model_id"] = "llm-semantic-router/dev-2.0-4b"
        write_jsonl(self.predictions, self.rows)
        with self.assertRaisesRegex(ValueError, "Invalid/stale"):
            requalify._verify_rows(
                self.predictions,
                self.prompts,
                2,
                requalify.PINNED_SHA256["package/SHA256SUMS"],
            )

    def test_parity_report_must_bind_both_original_order_prediction_files(self) -> None:
        report_path = self.root / "parity-dev.report.json"
        selected = self.root / "parity-dev.selected.jsonl"
        merged = self.root / "parity-dev.merged.jsonl"
        write_jsonl(selected, self.rows)
        write_jsonl(merged, self.rows)
        report = {
            "candidate_manifest_sha256": requalify.PINNED_SHA256["package/SHA256SUMS"],
            "calibration_sha256": requalify.PINNED_SHA256["package/calib.json"],
            "selected_checkpoint": requalify.CHECKPOINT,
            "prompt_sha256": sha_file(self.prompts),
            "selected_predictions_sha256": sha_file(selected),
            "merged_predictions_sha256": sha_file(merged),
            "items": 2,
            "answers": 2,
            "choice_mismatch_n": 0,
            "probability_drift_p99": 0.0,
            "probability_drift_max": 0.0,
            "predeclared_gate": {"pass": True},
            "runtime": {
                "torch_deterministic_algorithms": True,
                "gated_delta_backend_before": FLA_BACKEND,
                "gated_delta_backend": TORCH_BACKEND,
            },
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        requalify._verify_parity(report_path, self.prompts, 2)
        write_jsonl(merged, list(reversed(self.rows)))
        with self.assertRaisesRegex(ValueError, "original inputs"):
            requalify._verify_parity(report_path, self.prompts, 2)

    def test_first_failed_process_preserves_failure_receipt_and_stops(self) -> None:
        config_path = self.root / "config.json"
        source = Path(requalify.__file__).resolve().parents[1]
        config = {
            "source_root": str(source),
            "source_commit": "a" * 40,
            "runtime_image_id": "sha256:" + "b" * 64,
            "physical_gpu": {"guid": "private-test", "index": 0},
            "package": str(self.root),
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")
        calls: list[str] = []

        def fail_first(
            name: str, _module: str, _args: list[str], _directory: Path
        ) -> dict:
            calls.append(name)
            raise RuntimeError("simulated native failure")

        output = self.root / "new-output"
        with mock.patch.object(
            requalify,
            "preflight",
            return_value=(config, {"css_pilot_prompts": self.prompts}),
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated native failure"):
                requalify.execute(config_path, output, fail_first)
        self.assertEqual(calls, ["r1"])
        receipt = json.loads(
            (output / "repeat/execution.receipt.json").read_text(encoding="utf-8")
        )
        self.assertFalse(receipt["predeclared_numeric_repeat_gate_pass"])
        self.assertIn("simulated native failure", receipt["failure"])
        self.assertFalse((output / "full/execution.receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
