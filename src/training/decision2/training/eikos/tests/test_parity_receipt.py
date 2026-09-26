import json
import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from training.eikos.parity_receipt import combine


class ParityReceiptTest(unittest.TestCase):
    def test_binds_both_same_process_panels(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {}
            for name, count in (("dev", 1600), ("css", 1430)):
                prompt = root / f"{name}.jsonl"
                prompt.write_text(f"{name}\n", encoding="utf-8")
                report = root / f"{name}.json"
                report.write_text(
                    json.dumps(
                        {
                            "role": "gold-free direct selected-LoRA versus merged package parity",
                            "inference_variant": "selected_lora_and_merged_same_process",
                            "items": count,
                            "answers": count,
                            "prompt_sha256": sha256(prompt.read_bytes()).hexdigest(),
                            "candidate_manifest_sha256": "a" * 64,
                            "adapter_weights_sha256": "b" * 64,
                            "calibration_sha256": "c" * 64,
                            "selected_checkpoint": "checkpoint-0160",
                            "choice_mismatch_n": 0,
                            "pmax_abs_drift_max": 0,
                            "probability_drift_max": 0,
                            "predeclared_gate": {"pass": True},
                        }
                    ),
                    encoding="utf-8",
                )
                paths[name] = (report, prompt)
            combined = combine(
                dev_report=paths["dev"][0],
                css_report=paths["css"][0],
                dev_prompts=paths["dev"][1],
                css_prompts=paths["css"][1],
                output=root / "combined.json",
            )
            self.assertTrue(combined["predeclared_gate_pass"])
            self.assertEqual(combined["total_items"], 3030)
            self.assertEqual(combined["model_sha256"], "a" * 64)
            paths["css"][1].write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatched direct parity"):
                combine(
                    dev_report=paths["dev"][0],
                    css_report=paths["css"][0],
                    dev_prompts=paths["dev"][1],
                    css_prompts=paths["css"][1],
                    output=root / "not-created.json",
                )


if __name__ == "__main__":
    unittest.main()
