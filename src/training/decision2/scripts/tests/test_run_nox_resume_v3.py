"""Frozen native continuation command tests."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_nox_resume_v3 import run, sha


class NoxResumeV3Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.predictions = root / "nox.css.predictions.jsonl"
        self.predictions.write_text('{"id":"first"}\n', encoding="utf-8")
        self.recovery = root / "recovery.json"
        self.plan_path = root / "plan.json"
        plan = {
            "inference": [
                {
                    "key": "nox",
                    "group": "decision1",
                    "model_id": "llm-semantic-router/Decision-1.0-Nox-4B",
                    "revision": "fixed",
                    "paths": {
                        "css": str(self.predictions),
                        "public": str(root / "nox.public.predictions.jsonl"),
                    },
                    "commands": ["true", "true", "true"],
                }
            ]
        }
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        self.plan_sha = sha(self.plan_path)
        self.recovery.write_text(
            json.dumps(
                {
                    "schema_version": "decision2-v3-nox-overbudget-recovery/1",
                    "plan_sha256": self.plan_sha,
                    "recovered_predictions_sha256": sha(self.predictions),
                }
            ),
            encoding="utf-8",
        )
        self.log = root / "resume.log"
        self.receipt = root / "resume.receipt.json"

    def test_css_runs_same_command_with_resume_only(self):
        with patch.dict(os.environ, {"GPU_ID": "1"}):
            result = run(
                self.plan_path,
                self.plan_sha,
                "css",
                self.recovery,
                self.log,
                self.receipt,
            )
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["predictions_sha256"], sha(self.predictions))
        self.assertEqual(result["recovery_receipt_sha256"], sha(self.recovery))

    def test_rejects_unbound_partial_predictions(self):
        self.predictions.write_text('{"id":"changed"}\n', encoding="utf-8")
        with patch.dict(os.environ, {"GPU_ID": "1"}):
            with self.assertRaisesRegex(ValueError, "recovery receipt"):
                run(
                    self.plan_path,
                    self.plan_sha,
                    "css",
                    self.recovery,
                    self.log,
                    self.receipt,
                )
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
