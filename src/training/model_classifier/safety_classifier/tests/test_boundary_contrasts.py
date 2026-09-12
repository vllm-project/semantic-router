"""Group leakage, annotation-mask and frozen-output contracts for authored data."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.training.model_classifier.safety_classifier import (
    vela_boundary_contrasts as contrasts,
)


class BoundaryContrastTests(unittest.TestCase):
    def setUp(self):
        self.registry = json.loads(contrasts.DEFAULT_REGISTRY.read_text())
        self.rubric = json.loads(contrasts.DEFAULT_RUBRIC.read_text())

    def test_translations_and_pairs_stay_in_one_split(self):
        result = contrasts.build(self.registry, self.rubric)
        train = {row["group_id"] for row in result["train"]}
        dev = {row["group_id"] for row in result["validation"]}
        self.assertFalse(train & dev)
        self.assertEqual(len(result["train"]), 172)
        self.assertEqual(len(result["validation"]), 92)
        for split, rows in result.items():
            for group in {row["group_id"] for row in rows}:
                members = [row for row in rows if row["group_id"] == group]
                self.assertEqual({row["language"] for row in members}, {"en", "zh"})
                self.assertIn(len(members), (2, 4))
            for category in contrasts.CATEGORIES:
                positive_groups = {
                    row["group_id"]
                    for row in rows
                    if row["contrast_category"] == category
                    and row["contrast_variant"] == "positive"
                }
                self.assertGreaterEqual(
                    len(positive_groups), contrasts.MINIMUM_POSITIVE_FAMILIES[split]
                )

    def test_review_masks_preserve_cooccurrence_and_unknowns(self):
        result = contrasts.build(self.registry, self.rubric)
        rows = {row["id"]: row for split in result.values() for row in split}
        indices = {label: index for index, label in enumerate(contrasts.LABELS)}
        row = rows["vela-boundary:investment_adviser_identity:positive:en"]
        self.assertEqual(row["targets"][indices["specialized_advice"]], 1)
        self.assertEqual(row["targets"][indices["misinformation"]], 1)
        self.assertEqual(row["label_mask"][indices["criminal_activity"]], 0)
        self.assertIn("criminal_activity", row["unknown_rationale"])
        for row in rows.values():
            self.assertEqual(len(row["targets"]), 12)
            self.assertEqual(len(row["label_mask"]), 12)
            self.assertTrue(
                all(
                    not target or observed
                    for target, observed in zip(
                        row["targets"], row["label_mask"], strict=True
                    )
                )
            )
            if row["label"] == "safe":
                self.assertFalse(any(row["targets"]))
                self.assertTrue(all(row["label_mask"]))

    def test_verification_and_ordinary_help_are_safe(self):
        rows = [
            row
            for split in contrasts.build(self.registry, self.rubric).values()
            for row in split
        ]
        for family, variant in (
            ("invented_school_award", "near_miss"),
            ("fabricated_peer_review", "near_miss"),
            ("bar_registration_representation", "near_miss"),
            ("safe_home_network", "safe_anchor"),
            ("safe_statistics_exercise", "safe_anchor"),
            ("safe_interview_edit", "safe_anchor"),
            ("safe_dispatch_support", "safe_anchor"),
        ):
            selected = [
                row
                for row in rows
                if row["group_id"] == f"vela-boundary:{family}"
                and row["contrast_variant"] == variant
            ]
            self.assertEqual(len(selected), 2)
            self.assertTrue(all(row["label"] == "safe" for row in selected))

    def test_changed_split_or_duplicate_text_fails(self):
        changed = copy.deepcopy(self.registry)
        changed["families"][0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "family plan"):
            contrasts.build(changed, self.rubric)
        changed = copy.deepcopy(self.registry)
        changed["families"][1]["variants"][0]["en"] = changed["families"][0][
            "variants"
        ][0]["en"].upper()
        with self.assertRaisesRegex(ValueError, "normalization"):
            contrasts.build(changed, self.rubric)

    def test_invalid_or_unexplained_annotations_fail(self):
        for positive, unknown in (
            (["invented_category"], []),
            (["specialized_advice"], ["specialized_advice"]),
            (["specialized_advice"], ["privacy"]),
        ):
            changed = copy.deepcopy(self.registry)
            variant = changed["families"][0]["variants"][0]
            variant["positive_labels"], variant["unknown_labels"] = positive, unknown
            with self.assertRaises(ValueError):
                contrasts.build(changed, self.rubric)
        changed = copy.deepcopy(self.registry)
        changed["families"][0]["variants"][1]["positive_labels"] = ["misinformation"]
        with self.assertRaisesRegex(ValueError, "safe contrasts"):
            contrasts.build(changed, self.rubric)

    def test_rubric_label_order_cannot_drift(self):
        changed = copy.deepcopy(self.rubric)
        changed["labels"] = dict(reversed(list(changed["labels"].items())))
        with self.assertRaisesRegex(ValueError, "label order"):
            contrasts.build(self.registry, changed)

    def test_frozen_files_reproduce_and_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            manifest = contrasts.freeze(first)
            contrasts.freeze(second)
            for name, metadata in manifest["files"].items():
                self.assertEqual(
                    (first / name).read_bytes(), (second / name).read_bytes()
                )
                self.assertEqual(
                    hashlib.sha256((first / name).read_bytes()).hexdigest(),
                    metadata["sha256"],
                )
            with self.assertRaises(FileExistsError):
                contrasts.freeze(first)
            self.assertFalse(manifest["model_predictions_used_for_authoring"])
            self.assertFalse(manifest["final_data_read"])


if __name__ == "__main__":
    unittest.main()
