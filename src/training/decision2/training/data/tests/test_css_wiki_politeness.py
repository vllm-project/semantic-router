from __future__ import annotations

import unittest

from transfer import build as transfer

from training.data import build_css_wiki_politeness as wiki
from training.data import build_pilot as pilot


class WikiPolitenessTests(unittest.TestCase):
    def test_test_label_is_never_read_and_conflicts_are_quarantined(self) -> None:
        test = {
            "id": "test",
            "user": "UNKNOWN_USER",
            "text": "Please move the article discussion.",
            "meta": {},
        }
        good = {
            "id": "good",
            "user": "UNKNOWN_USER",
            "text": "I appreciate your careful work on this page.",
            "meta": {"Binary": 1},
        }
        duplicate = {**good, "id": "good-duplicate"}
        conflict_a = {
            "id": "conflict-a",
            "user": "UNKNOWN_USER",
            "text": "This edit should be reviewed.",
            "meta": {"Binary": 0},
        }
        conflict_b = {**conflict_a, "id": "conflict-b", "meta": {"Binary": -1}}
        test_hash = pilot.sha_bytes(wiki.source_context(test).encode())
        excluded = {
            "raw_context_sha256": {test_hash},
            "normalized_context_sha256": {
                transfer.normalized_context_sha256(wiki.source_context(test))
            },
            "panel_input_sha256": set(),
            "task_source_ids": set(),
        }
        rows, audit = wiki.candidate_pool(
            [test, good, duplicate, conflict_a, conflict_b], excluded, [], {test_hash}
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "css_train/wiki_politeness/good")
        self.assertEqual(rows[0]["options"][rows[0]["label"]]["key"], "1")
        self.assertEqual(audit["excluded_counts"]["css_panel_raw_context"], 1)
        self.assertEqual(audit["excluded_counts"]["duplicate_same_context"], 1)
        self.assertEqual(audit["excluded_counts"]["conflicting_same_context"], 2)

    def test_missing_panel_hash_is_rejected(self) -> None:
        candidate = {
            "id": "a",
            "user": "UNKNOWN_USER",
            "text": "Unique train request.",
            "meta": {"Binary": 0},
        }
        excluded = {
            "raw_context_sha256": set(),
            "normalized_context_sha256": set(),
            "panel_input_sha256": set(),
            "task_source_ids": set(),
        }
        with self.assertRaisesRegex(ValueError, "every CSS politeness test context"):
            wiki.candidate_pool([candidate], excluded, [], {"0" * 64})


if __name__ == "__main__":
    unittest.main()
