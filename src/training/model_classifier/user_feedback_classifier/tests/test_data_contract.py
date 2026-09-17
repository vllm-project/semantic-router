"""Regression tests for feedback labels and independent validation."""

import json
import tempfile
import unittest
from pathlib import Path

from src.training.model_classifier.user_feedback_classifier.data_contract import (
    LABEL2ID,
    balanced_class_weights,
    merged_output_directory,
    parse_example,
    validate_splits,
    write_label_mapping,
)


class FeedbackContractTest(unittest.TestCase):
    def rows(self, prefix):
        return [
            parse_example({"text": f"{prefix} {label}", "label_name": label})
            for label in LABEL2ID
        ]

    def test_label_mapping_is_one_source_for_export_and_training(self):
        with tempfile.TemporaryDirectory() as directory:
            write_label_mapping(directory)
            mapping = json.loads((Path(directory) / "label_mapping.json").read_text())
        self.assertEqual(
            mapping["id2label"],
            {
                "0": "SAT",
                "1": "NEED_CLARIFICATION",
                "2": "WRONG_ANSWER",
                "3": "WANT_DIFFERENT",
            },
        )

    def test_unknown_labels_never_default_to_satisfied(self):
        for label in (None, -1, 8, True, "typo", "DISSAT"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_example({"text": "A follow-up", "label": label})

    def test_weights_use_output_ids_for_noncontiguous_classes(self):
        self.assertEqual(balanced_class_weights([1, 1, 3]), [0.0, 0.75, 0.0, 1.5])

    def test_validation_cannot_be_missing_or_reused_training(self):
        train = self.rows("train")
        for validation in (
            [],
            train,
            [{**row, "text": row["text"].upper()} for row in train],
        ):
            with self.assertRaises(ValueError):
                validate_splits(train, validation)

    def test_template_family_must_not_cross_splits(self):
        train, validation = self.rows("train"), self.rows("validation")
        train[0]["group_id"] = validation[0]["group_id"] = "template-family-1"
        with self.assertRaisesRegex(ValueError, "groups"):
            validate_splits(train, validation)

    def test_clean_validation_reports_counts_without_inventing_groups(self):
        manifest = validate_splits(self.rows("train"), self.rows("validation"))
        self.assertEqual(manifest["splits"]["validation"]["rows"], 4)
        self.assertEqual(manifest["splits"]["train"]["rows_with_group"], 0)

    def test_merged_output_never_overwrites_adapter(self):
        for name in ("feedback", "feedback_lora", "feedback-lora"):
            with self.subTest(name=name):
                self.assertNotEqual(merged_output_directory(name), Path(name))
        self.assertEqual(
            merged_output_directory("models/feedback_lora"),
            Path("models/feedback-merged"),
        )
