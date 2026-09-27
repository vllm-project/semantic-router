"""The native package receipt must attest the requested deterministic mode."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from training.eikos import published_infer


class PublishedDeterminismReceiptTest(unittest.TestCase):
    def test_requested_mode_is_enabled_before_load_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompts = root / "prompts.jsonl"
            prompts.write_text("{}\n", encoding="utf-8")
            row = {
                "id": "one",
                "state": {"fact": "A"},
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Choose",
                        "criteria": {"a": "A", "b": "B"},
                    }
                },
            }
            captured = {"events": []}
            torch_state = {"enabled": False}
            fake_torch = SimpleNamespace(
                __version__="test",
                version=SimpleNamespace(hip="test"),
                cuda=SimpleNamespace(
                    get_device_properties=lambda device: SimpleNamespace(
                        gcnArchName="gfx942"
                    )
                ),
                are_deterministic_algorithms_enabled=lambda: torch_state["enabled"],
                use_deterministic_algorithms=lambda enabled: torch_state.update(
                    enabled=enabled
                ),
            )

            def load_native(*args, **kwargs):
                captured["events"].append("load")
                captured["flag_at_load"] = torch_state["enabled"]
                return SimpleNamespace(
                    decide_all=lambda **payload: {
                        "q": (
                            {
                                "type": "choice",
                                "choice": "a",
                                "probabilities": {"a": 1.0, "b": 0.0},
                            },
                            7,
                        )
                    }
                )

            def write_output(output, predictions, manifest):
                captured["predictions"] = predictions
                captured["manifest"] = manifest
                manifest["predictions_sha256"] = "d" * 64

            identity = {
                "model_sha256": "a" * 64,
                "calibration_sha256": "b" * 64,
                "selected_checkpoint": "checkpoint-0145",
                "source_revision": "c" * 40,
                "source_release": {},
                "package_files_checked": 1,
                "rights_mode": "clean",
                "rights_attestation_sha256": None,
            }
            with (
                patch.object(
                    published_infer, "package_identity", return_value=identity
                ),
                patch.object(published_infer, "load_prompts", return_value=[row]),
                patch.object(published_infer, "load_decider", side_effect=load_native),
                patch.object(
                    published_infer,
                    "use_torch_reference_gated_delta",
                    side_effect=lambda: (
                        captured["events"].append("select")
                        or {"gated_delta_backend": "torch-reference"}
                    ),
                ),
                patch.object(published_infer, "synchronize"),
                patch.object(published_infer, "write_output", side_effect=write_output),
                patch.object(published_infer, "version", return_value="test"),
                patch.dict(
                    sys.modules,
                    {"torch": fake_torch, "fla": SimpleNamespace(__version__="0.5.2")},
                ),
            ):
                result = published_infer.collect(
                    model_path=root,
                    prompts=prompts,
                    output=root / "out.jsonl",
                    deterministic_algorithms=True,
                )
                self.assertEqual(captured["events"], ["load"])
                self.assertNotIn("gated_delta_backend", captured["manifest"]["runtime"])
                captured["events"].clear()
                published_infer.collect(
                    model_path=root,
                    prompts=prompts,
                    output=root / "reference.jsonl",
                    deterministic_algorithms=True,
                    torch_reference_gated_delta=True,
                )
                self.assertEqual(captured["events"], ["select", "load"])
                self.assertEqual(
                    captured["manifest"]["runtime"]["gated_delta_backend"],
                    "torch-reference",
                )
                self.assertEqual(
                    captured["manifest"]["model_id"],
                    "llm-semantic-router/DEV2.0-4B",
                )
                self.assertEqual(
                    captured["predictions"][0]["model_id"],
                    "llm-semantic-router/DEV2.0-4B",
                )
                with self.assertRaisesRegex(ValueError, "DEV2.0-4B"):
                    published_infer.collect(
                        model_path=root,
                        prompts=prompts,
                        output=root / "old-id.jsonl",
                        model_id="llm-semantic-router/dev-2.0-4b",
                    )
            self.assertTrue(captured["flag_at_load"])
            self.assertTrue(
                captured["manifest"]["runtime"]["torch_deterministic_algorithms"]
            )
            self.assertEqual(
                captured["manifest"]["runtime"]["flash_linear_attention"], "0.5.2"
            )
            self.assertEqual(result["predictions_sha256"], "d" * 64)
            self.assertEqual(captured["manifest"]["counts"]["items"], 1)


if __name__ == "__main__":
    unittest.main()
