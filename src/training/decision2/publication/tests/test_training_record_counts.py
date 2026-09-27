"""Structured replay must disclose all merged TRAIN sources, not only additions."""

from __future__ import annotations

import unittest

from publication.training_record import _training_counts


class TrainingCountsTest(unittest.TestCase):
    def test_structured_replay_uses_merged_inventory(self) -> None:
        data = {
            "schema_version": "decision2-human-structured-replay/1",
            "added_counts": {
                "source": {"new": 2},
                "task_type": {"choice": 2},
            },
            "merged_counts": {
                "source": {"old": 3, "new": 2},
                "task_type": {"choice": 4, "score": 1},
            },
            "selected_added_ids": ["new-1", "new-2"],
        }
        self.assertEqual(_training_counts(data)["source"], {"old": 3, "new": 2})
        data["selected_added_ids"] = ["new-1", "new-1"]
        with self.assertRaisesRegex(ValueError, "added IDs"):
            _training_counts(data)
        data["selected_added_ids"] = ["new-1", ["invalid"]]
        with self.assertRaisesRegex(ValueError, "added IDs"):
            _training_counts(data)
        data["selected_added_ids"] = ["new-1", "new-2"]
        data["added_counts"]["source"]["new"] = 3
        with self.assertRaisesRegex(ValueError, "exceed merged"):
            _training_counts(data)

    def test_legacy_manifest_uses_counts(self) -> None:
        data = {
            "schema_version": "decision2-balanced-human-5824/1",
            "counts": {"source": {"old": 3}, "task_type": {"score": 3}},
        }
        self.assertEqual(_training_counts(data), data["counts"])


if __name__ == "__main__":
    unittest.main()
