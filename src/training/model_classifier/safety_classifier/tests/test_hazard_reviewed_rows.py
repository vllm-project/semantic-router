"""Partial review masks must survive projection without inheriting weak gold."""

import hashlib
import unittest

from src.training.model_classifier.safety_classifier.vela_hazard_reviewed_rows import (
    filter_replay,
    project_reviews,
)


def record(identifier="one", group="group"):
    text = "A complete reviewed training example."
    return {
        "id": identifier,
        "group_id": group,
        "source": "aegis2",
        "source_split": "train",
        "text": text,
        "label": "unsafe",
        "targets": [1] + [0] * 11,
        "label_mask": [1] * 12,
        "reviewed_targets": [0] * 12,
        "reviewed_label_mask": [1] * 12,
        "reviewed_binary_label": "safe",
        "reviewed_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


class HazardReviewedRowsTests(unittest.TestCase):
    def test_partial_negative_is_not_a_fully_safe_example(self):
        row = record()
        row["reviewed_label_mask"][0] = 0
        row["reviewed_binary_label"] = "unknown"
        projected, _, _ = project_reviews([row])
        self.assertEqual(projected[0]["label"], "unknown")
        self.assertEqual(projected[0]["label_mask"], [0] + [1] * 11)
        self.assertEqual(projected[0]["targets"], [0] * 12)
        self.assertEqual(row["targets"][0], 1)
        self.assertEqual(projected[0]["targets_before_exact_review"][0], 1)

    def test_all_unknown_still_supersedes_the_whole_group(self):
        row = record()
        row["reviewed_label_mask"] = [0] * 12
        row["reviewed_binary_label"] = "unknown"
        projected, groups, excluded = project_reviews([row])
        self.assertFalse(projected)
        self.assertEqual(len(excluded), 1)
        replay, excluded = filter_replay(
            [row, record("translation"), record("other", "other")], groups
        )
        self.assertEqual([item["id"] for item in replay], ["other"])
        self.assertEqual(len(excluded), 2)

    def test_masked_positive_and_false_safe_contract_are_rejected(self):
        row = record()
        row["reviewed_targets"][0] = 1
        for binary, observed in [("safe", 1), ("unsafe", 0)]:
            row["reviewed_binary_label"] = binary
            row["reviewed_label_mask"][0] = observed
            with self.assertRaises(ValueError):
                project_reviews([row])

    def test_text_revision_duplicate_and_nontrain_are_rejected(self):
        row = record()
        with self.assertRaises(ValueError):
            project_reviews([row, row])
        row["text"] += " changed"
        with self.assertRaises(ValueError):
            project_reviews([row])
        row = record()
        row["source_split"] = "validation"
        with self.assertRaises(ValueError):
            project_reviews([row])
        with self.assertRaises(ValueError):
            filter_replay([row], set())

    def test_strong_experiment_does_not_launder_weak_wrapped_payload(self):
        wrapped = {**record(), "source": "authored_context"}
        wrapped["source_payload_source"] = "aegis2"
        strong = {**record("strong", "strong"), "source": "reviewed_train"}
        mixed, _ = filter_replay([wrapped, strong], set())
        retained, excluded = filter_replay([wrapped, strong], set(), strong_only=True)
        self.assertEqual(len(mixed), 2)
        self.assertEqual([row["id"] for row in retained], ["strong"])
        self.assertEqual(len(excluded), 1)


if __name__ == "__main__":
    unittest.main()
