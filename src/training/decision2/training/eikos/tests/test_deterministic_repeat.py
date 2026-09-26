"""Full-panel repeat receipts must bind runtime mode and raw predictions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.eikos.deterministic_repeat import audit
from training.eikos.repeatability import sha


class DeterministicRepeatTest(unittest.TestCase):
    def test_identical_panel_passes_and_unattested_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "package"
            package.mkdir()
            (package / "SHA256SUMS").write_text("model\n", encoding="utf-8")
            (package / "calib.json").write_text("{}\n", encoding="utf-8")
            prompts = root / "prompts.jsonl"
            prompts.write_text("frozen prompt bytes\n", encoding="utf-8")
            rows = [
                {
                    "id": f"item-{index}",
                    "source_input_sha256": f"input-{index}",
                    "usage": {"input_tokens": 8},
                    "answers": {
                        "q": {
                            "type": "choice",
                            "choice": "a",
                            "probabilities": {"a": 0.8, "b": 0.2},
                        }
                    },
                }
                for index in range(1430)
            ]
            encoded = "".join(json.dumps(row) + "\n" for row in rows)
            paths = [root / "first.jsonl", root / "second.jsonl"]
            manifests = []
            for path in paths:
                path.write_text(encoded, encoding="utf-8")
                manifest = {
                    "predictions_sha256": sha(path),
                    "input_sha256": sha(prompts),
                    "model_sha256": sha(package / "SHA256SUMS"),
                    "calibration_sha256": sha(package / "calib.json"),
                    "input_items": 1430,
                    "evaluated_items": 1430,
                    "max_items": None,
                    "counts": {
                        "items": 1430,
                        "valid_questions": 1430,
                        "invalid_questions": 0,
                    },
                    "runtime": {
                        "torch_deterministic_algorithms": True,
                        "flash_linear_attention": "0.5.2",
                    },
                    "collector_source_sha256": "c" * 64,
                }
                manifest_path = Path(str(path) + ".manifest.json")
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                manifests.append(manifest_path)
            report = audit(
                predictions_a=paths[0],
                predictions_b=paths[1],
                prompts=prompts,
                package=package,
                output=root / "report.json",
            )
            self.assertTrue(report["predeclared_numeric_repeat_gate_pass"])
            self.assertEqual(report["comparison"]["categorical_mismatch_n"], 0)
            bad = json.loads(manifests[1].read_text(encoding="utf-8"))
            bad["runtime"]["torch_deterministic_algorithms"] = False
            manifests[1].write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "deterministic FLA"):
                audit(
                    predictions_a=paths[0],
                    predictions_b=paths[1],
                    prompts=prompts,
                    package=package,
                    output=root / "rejected.json",
                )


if __name__ == "__main__":
    unittest.main()
