"""Fail closed on partial or stale GLiNER baseline prediction files."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inference.gliner25_receipt import audit
from inference.run import digest


class ReceiptAuditTest(unittest.TestCase):
    def test_complete_identity_and_input_binding(self):
        row = {
            "id": "r1",
            "state": "text",
            "questions": {"decision": {"type": "noul", "instructions": "Is it true?"}},
        }
        identity = {"model_id": "pinned", "adapter_version": "v1"}
        item = {
            "id": "r1",
            "answers": {"decision": {"type": "noul", "noul": 0.7}},
            "source_input_sha256": digest(
                {"state": row["state"], "questions": row["questions"]}
            ),
            "invalid_reason": None,
            **identity,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(json.dumps(item) + "\n", encoding="utf-8")
            self.assertEqual(audit([row], path, identity)["valid"], 1)
            item["source_input_sha256"] = "0" * 64
            path.write_text(json.dumps(item) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale"):
                audit([row], path, identity)


if __name__ == "__main__":
    unittest.main()
