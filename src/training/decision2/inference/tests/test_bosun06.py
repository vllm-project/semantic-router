"""Bosun's shared typed-decision contract and pinned package checks."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference.bosun06 import (
    BASE_ID,
    BASE_REVISION,
    MODEL_REVISION,
    VARIANTS,
    candidates_for,
    project,
    verify_packages,
)


class Bosun06Test(unittest.TestCase):
    def test_native_choice_noul_score_mapping(self) -> None:
        choice = {
            "type": "choice",
            "criteria": {"first": "one", "second": "two"},
        }
        candidates, keys = candidates_for(choice)
        self.assertEqual(keys, ["first", "second"])
        self.assertEqual(candidates[1]["description"], "two")
        self.assertEqual(project(choice, keys, [0.2, 0.8])["choice"], "second")

        noul = {
            "type": "noul",
            "criteria": {"true": "Allowed", "false": "Denied"},
        }
        candidates, keys = candidates_for(noul)
        self.assertEqual(keys, ["true", "false"])
        self.assertEqual(candidates[0]["description"], "Allowed")
        self.assertEqual(project(noul, keys, [0.3, 0.7])["noul"], 0.3)

        score = {"type": "score", "criteria": ["Low", "Medium", "High"]}
        candidates, keys = candidates_for(score)
        self.assertEqual(keys, ["0", "1", "2"])
        self.assertEqual(candidates[2]["label"], "High")
        self.assertAlmostEqual(project(score, keys, [0.2, 0.3, 0.5])["score"], 1.3)

    def test_malformed_candidates_or_probabilities_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 255"):
            candidates_for({"type": "score", "criteria": ["x"] * 256})
        with self.assertRaisesRegex(ValueError, "probabilities"):
            project({"type": "choice"}, ["a", "b"], [0.7, 0.7])

    def test_package_verifier_hashes_manifest_and_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, base = root / "model", root / "base"
            model.mkdir()
            base.mkdir()
            config = {
                "base_model_name_or_path": BASE_ID,
                "base_model_revision": BASE_REVISION,
                "prompt_schema": "bosun-decision-prompt-v3-stable-slots",
                "decision_token_count": 256,
                "decision_token_assignment": "presented_slot",
            }
            (model / "config.json").write_text(json.dumps(config))
            config_sha = hashlib.sha256(
                (model / "config.json").read_bytes()
            ).hexdigest()
            (model / "manifest.json").write_text(
                json.dumps({"files": {"config.json": config_sha}})
            )
            (base / "config.json").write_text("{}")
            (base / "model.safetensors").write_bytes(b"base")
            with patch("inference.bosun06.local_revision", side_effect=[True, True]):
                receipt = verify_packages(model, base)
            self.assertEqual(receipt["model_revision"], MODEL_REVISION)
            with patch("inference.bosun06.local_revision", side_effect=[True, True]):
                with self.assertRaisesRegex(ValueError, "contract differs"):
                    verify_packages(model, base, "1.7b")
            model_id, revision, base_id, base_revision, _ = VARIANTS["1.7b"]
            config.update(
                base_model_name_or_path=base_id, base_model_revision=base_revision
            )
            (model / "config.json").write_text(json.dumps(config))
            config_sha = hashlib.sha256(
                (model / "config.json").read_bytes()
            ).hexdigest()
            (model / "manifest.json").write_text(
                json.dumps({"files": {"config.json": config_sha}})
            )
            with patch("inference.bosun06.local_revision", side_effect=[True, True]):
                receipt = verify_packages(model, base, "1.7b")
            self.assertEqual(
                (receipt["model_id"], receipt["model_revision"]), (model_id, revision)
            )
            with self.assertRaisesRegex(ValueError, "Unknown Bosun size"):
                verify_packages(model, base, "4b")
            (model / "config.json").write_text("{}")
            with patch("inference.bosun06.local_revision", side_effect=[True, True]):
                with self.assertRaisesRegex(ValueError, "differs"):
                    verify_packages(model, base)


if __name__ == "__main__":
    unittest.main()
