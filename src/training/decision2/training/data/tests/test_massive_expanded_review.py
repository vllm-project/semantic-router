"""A gold-blind packet must be tied to one frozen private candidate."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.data import build_massive_expanded_review as expanded
from training.data import build_massive_multilingual as massive


class ExpandedReviewTests(unittest.TestCase):
    def test_complete_packet_hides_labels_and_has_separate_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, output = root / "candidate", root / "review"
            candidate.mkdir()
            rows = []
            for i in range(600):
                for locale in massive.LOCALES:
                    rows.append(
                        {
                            "group_id": f"massive-1.1:{i}",
                            "state": f"request {i}",
                            "instructions": "Choose an action",
                            "options": [
                                {"key": "A", "description": "First"},
                                {"key": "B", "description": "Second"},
                            ],
                            "label": 1,
                            "audit_metadata": {
                                "source_locale": locale,
                                "source_id": str(i),
                                "intent": "alarm_set",
                            },
                        }
                    )
            massive.write_jsonl(candidate / "train.private.jsonl", rows)
            manifest = {
                "training_approved": False,
                "selected_source_groups": {"train": 600},
                "outputs": {
                    "train.private.jsonl": {
                        "sha256": massive.sha(candidate / "train.private.jsonl")
                    }
                },
            }
            (candidate / "manifest.json").write_text(json.dumps(manifest))
            digest = massive.sha(candidate / "manifest.json")
            with self.assertRaisesRegex(ValueError, "digest changed"):
                expanded.build(candidate, "0" * 64, output)
            receipt = expanded.build(candidate, digest, output)
            self.assertEqual(receipt["blind_review_rows"], 600)
            packet = [
                json.loads(line)
                for line in (output / "english-all600.blind.private.jsonl")
                .read_text()
                .splitlines()
            ]
            key = [
                json.loads(line)
                for line in (output / "english-all600.key.private.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                {row["review_id"] for row in packet}, {row["review_id"] for row in key}
            )
            self.assertTrue(
                all(
                    "source_id" not in row
                    and "gold_option_key" not in row
                    and "source_intent" not in row
                    and "label" not in row
                    for row in packet
                )
            )
            self.assertTrue(all(row["gold_option_key"] == "B" for row in key))
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            self.assertEqual(
                (output / "english-all600.key.private.jsonl").stat().st_mode & 0o777,
                0o600,
            )


if __name__ == "__main__":
    unittest.main()
