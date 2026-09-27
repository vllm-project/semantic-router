"""Small correctness checks for the gold-free release overlap auditor."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.data.audit_release4b_overlap import audit_panel, read_jsonl


class ReleaseOverlapAuditTests(unittest.TestCase):
    def test_same_state_is_counted_in_exact_and_near(self) -> None:
        text = "The inspection memo records a current signed approval."
        train = [{"id": "train/1", "state": text, "source": "source-a"}]
        prompt = [{"id": "eval/1", "state": text, "questions": {"q": {}}}]
        report = audit_panel(train, prompt)
        self.assertEqual(
            report["counts"],
            {"same_row_ids": 0, "exact_raw": 1, "exact_normalized": 1, "near": 1},
        )
        self.assertEqual(report["near"][0][:3], ["train/1", "eval/1", "source-a"])

    def test_prompt_loader_refuses_a_label_field(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "prompts.jsonl"
            path.write_text(
                json.dumps(
                    {"id": "eval/1", "state": "memo", "questions": {}, "gold": "yes"}
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not a gold-free prompt"):
                read_jsonl(path, prompt=True)


if __name__ == "__main__":
    unittest.main()
