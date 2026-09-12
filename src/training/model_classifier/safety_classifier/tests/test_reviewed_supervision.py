"""Unknown risk labels, exact reviews, and conflicting family variants stay distinct."""

import hashlib
import unittest

from src.training.model_classifier.safety_classifier.vela_data import LABELS
from src.training.model_classifier.safety_classifier.vela_reviewed_supervision import (
    repair_weak_rows,
    reviewed_rows,
)


def record(identifier, group, positive=(), *, label="unsafe"):
    return {
        "id": identifier,
        "group_id": group,
        "text": f"Text {identifier}",
        "source": "aegis2",
        "source_split": "train",
        "label": label,
        "targets": [int(name in positive) for name in LABELS],
        "label_mask": [1] * len(LABELS),
    }


class ReviewedSupervisionTests(unittest.TestCase):
    def test_weak_positive_becomes_unknown_without_mutating_input(self):
        row = record("one", "group", ["violence", "misinformation"])
        output, excluded = repair_weak_rows([row], set(), task="hazard")
        index = LABELS.index("misinformation")
        self.assertFalse(excluded)
        self.assertEqual(output[0]["targets"][index], 0)
        self.assertEqual(output[0]["label_mask"][index], 0)
        self.assertEqual(output[0]["targets"][LABELS.index("violence")], 1)
        self.assertEqual(row["targets"][index], 1)
        self.assertEqual(row["label_mask"][index], 1)
        self.assertEqual(output[0]["label"], "unsafe")

    def test_binary_quarantine_never_relabels_unsafe_as_safe(self):
        rows = [
            record("only", "one", ["specialized_advice"]),
            record("other", "two", ["specialized_advice", "violence"]),
            record("safe", "three", label="safe"),
        ]
        output, excluded = repair_weak_rows(rows, set(), task="safety")
        self.assertEqual([row["id"] for row in output], ["other", "safe"])
        self.assertEqual(output[0]["label"], "unsafe")
        self.assertEqual([row["id"] for row in excluded], ["only"])

    def test_reviewed_group_removes_all_variants_without_spreading_gold(self):
        raw = [record("reviewed", "family"), record("translation", "family")]
        item = {
            "id": "reviewed",
            "group_id": "family",
            "text_sha256": hashlib.sha256(raw[0]["text"].encode()).hexdigest(),
            "binary_label": "safe",
            "positive_labels": [],
            "unknown_labels": [],
            "reason": "Reviewed visible task is harmless",
        }
        review = {"items": [item], "protocol": "test", "reviewer": "test"}
        strong, groups = reviewed_rows(raw, review)
        self.assertEqual([row["id"] for row in strong], ["reviewed"])
        weak, excluded = repair_weak_rows(raw, groups, task="hazard")
        self.assertFalse(weak)
        self.assertEqual(len(excluded), 2)
        item["text_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "changed"):
            reviewed_rows(raw, review)

    def test_ambiguous_review_excludes_family_and_nontrain_is_rejected(self):
        row = record("one", "family")
        item = {
            "id": "one",
            "group_id": "family",
            "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
            "binary_label": None,
            "positive_labels": [],
            "unknown_labels": [],
            "reason": "Missing necessary context",
        }
        review = {"items": [item], "protocol": "test", "reviewer": "test"}
        strong, groups = reviewed_rows([row], review)
        self.assertFalse(strong)
        self.assertEqual(groups, {"family"})
        row["source_split"] = "validation"
        with self.assertRaises(ValueError):
            reviewed_rows([row], review)
        with self.assertRaises(ValueError):
            repair_weak_rows([row], set(), task="safety")

    def test_partial_row_with_no_observations_is_excluded(self):
        row = record("one", "family", ["misinformation"])
        row["label_mask"] = list(row["targets"])
        output, excluded = repair_weak_rows([row], set(), task="hazard")
        self.assertFalse(output)
        self.assertEqual(
            excluded[0]["reason"], "no observed labels after weak-positive quarantine"
        )


if __name__ == "__main__":
    unittest.main()
