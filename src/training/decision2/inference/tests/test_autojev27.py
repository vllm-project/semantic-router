"""CPU checks for the pinned 27B external native adapter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference import autojev27


class AutoJev27AdmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.model = root / "model"
        self.source = root / "source"
        self.model.mkdir()
        self.source.mkdir()
        self.config = {
            "format_version": 1,
            "base_model": autojev27.BASE_ID,
            "revision": autojev27.BASE_REVISION,
            "codes": [f"code{i}" for i in range(255)],
            "token_ids": list(range(255)),
            "temperature": 1.1,
        }
        self.write_config()

    def write_config(self) -> None:
        (self.model / "decision_config.json").write_text(json.dumps(self.config))

    def verify(self) -> dict:
        files = {
            "decision_config.json": "a" * 64,
            "model-00001-of-00001.safetensors": "b" * 64,
            "readout.safetensors": "c" * 64,
        }
        with (
            patch.object(autojev27, "local_revision", return_value=True),
            patch.object(
                autojev27.subprocess,
                "check_output",
                side_effect=[autojev27.SOURCE_REVISION + "\n", ""],
            ),
            patch.object(
                autojev27,
                "_tree",
                side_effect=[files, {"src/autojev/model.py": "d" * 64}],
            ),
            patch.object(
                autojev27,
                "_weights",
                return_value=(
                    ["model-00001-of-00001.safetensors", "readout.safetensors"],
                    25_000_000_000,
                ),
            ),
        ):
            return autojev27.verify_release(
                self.model, self.source, autojev27.MODEL_REVISION
            )

    def test_pinned_release_checks_three_type_native_config(self) -> None:
        result = self.verify()
        self.assertEqual(result["loaded_parameters"], 25_000_000_000)
        self.assertEqual(result["calibration_temperature"], 1.1)
        self.assertEqual(len(result["native_model_sha256"]), 64)

    def test_wrong_base_revision_is_rejected(self) -> None:
        self.config["revision"] = "0" * 40
        self.write_config()
        with self.assertRaisesRegex(ValueError, "decision config"):
            self.verify()

    def test_only_documented_overflows_are_invalid(self) -> None:
        self.assertEqual(
            autojev27._admission_reason(
                ValueError(
                    "Question branch exceeds the 8192-token limit; no input was truncated."
                )
            ),
            "context_overflow",
        )
        self.assertEqual(
            autojev27._admission_reason(
                ValueError(
                    "Questions must have 1 to 255 options, each with an answer code."
                )
            ),
            "candidate_limit",
        )
        self.assertIsNone(autojev27._admission_reason(ValueError("Tokenizer mismatch")))


if __name__ == "__main__":
    unittest.main()
