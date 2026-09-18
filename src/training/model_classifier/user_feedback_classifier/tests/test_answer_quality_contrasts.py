"""Frozen current-turn annotations, bilingual grouping, and reproducible partitions."""

import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from src.training.model_classifier.user_feedback_classifier import (
    vela_answer_quality_contrasts as contrasts,
)
from src.training.model_classifier.user_feedback_classifier.vela_contract import (
    VELA_ID2LABEL,
    VELA_LABEL2ID,
)


class AnswerQualityContrastTests(unittest.TestCase):
    def setUp(self):
        self.registry = {
            "version": "unit-fixture",
            "license": "CC0-1.0",
            "label_rules": dict.fromkeys(VELA_LABEL2ID, "Supplied annotation"),
            "input_protocol": "Complete current user text",
            "authorship": "Synthetic unit fixture, not training data",
            "evidence_scope": "Grouping and annotation transport only",
            "precedence": "Preserve the supplied annotation",
            "split_policy": "Keep each family in its declared partition",
            "final_data_read": False,
            "model_predictions_used": False,
            "families": [
                {
                    "family_id": f"family_{index}",
                    "split": (
                        "train" if index < contrasts.FAMILY_COUNTS["train"] else "dev"
                    ),
                    "intent": f"Fixture {index}",
                    "variants": [
                        {
                            "label": label,
                            "en": f"Fixture {index}, variant {number}, English",
                            "zh": f"样例 {index}, 变体 {number}, 中文",
                            "rationale": f"Reviewed annotation {number}",
                        }
                        for number, label in enumerate(VELA_ID2LABEL.values())
                    ],
                }
                for index in range(30)
            ],
        }
        self.registry["family_plan_sha256"] = contrasts.sha256(
            contrasts.json_bytes(contrasts.family_plan(self.registry))
        )

    def test_each_bilingual_five_way_family_stays_in_one_partition(self):
        records = contrasts.build(self.registry)
        self.assertEqual(len(records["train"]), 200)
        self.assertEqual(len(records["dev"]), 100)
        groups = {}
        for split, rows in records.items():
            groups[split] = defaultdict(list)
            for row in rows:
                groups[split][row["group_id"]].append(row)
                self.assertEqual(row["source_split"], split)
                self.assertEqual(row["source"], contrasts.SOURCE)
            self.assertEqual(len(groups[split]), 20 if split == "train" else 10)
            for family in groups[split].values():
                self.assertEqual(
                    Counter((row["label"], row["language"]) for row in family),
                    Counter(
                        (label, language)
                        for label in VELA_LABEL2ID
                        for language in ["en", "zh"]
                    ),
                )
        self.assertFalse(set(groups["train"]) & set(groups["dev"]))

    def test_supplied_annotations_and_provenance_are_preserved(self):
        variants = self.registry["families"][0]["variants"]
        variants[0]["en"] = "Your answer is accurate and clear."
        variants[-1]["en"] = 'Translate this review: "The answer is accurate."'
        rows = [row for part in contrasts.build(self.registry).values() for row in part]
        lookup = {
            (row["group_id"].split(":")[-1], row["label"], row["language"]): row
            for row in rows
        }
        self.assertEqual(lookup["family_0", "SAT", "en"]["text"], variants[0]["en"])
        self.assertEqual(
            lookup["family_0", "NO_FEEDBACK", "en"]["text"], variants[-1]["en"]
        )
        for row in rows:
            self.assertTrue(row["review_reason"].strip())
            self.assertEqual(
                row["text_sha256"], hashlib.sha256(row["text"].encode()).hexdigest()
            )
            self.assertEqual(row["license"], "CC0-1.0")

    def test_changed_family_assignments_and_duplicate_text_are_rejected(self):
        registry = copy.deepcopy(self.registry)
        registry["families"][0]["split"] = "dev"
        registry["families"][-1]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "frozen plan"):
            contrasts.build(registry)
        registry = copy.deepcopy(self.registry)
        text = registry["families"][0]["variants"][0]["en"]
        registry["families"][-1]["variants"][0]["en"] = "  " + text.upper() + "  "
        with self.assertRaisesRegex(ValueError, "normalization"):
            contrasts.build(registry)

    def test_missing_translation_label_or_rationale_cannot_be_frozen(self):
        for field, value in [
            ("en", ""),
            ("zh", None),
            ("label", "OTHER"),
            ("rationale", ""),
        ]:
            registry = copy.deepcopy(self.registry)
            registry["families"][0]["variants"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                contrasts.build(registry)
        registry = copy.deepcopy(self.registry)
        registry["families"][0]["variants"][1]["label"] = "SAT"
        with self.assertRaises(ValueError):
            contrasts.build(registry)

    def test_protocol_rejects_unknown_labels_and_prediction_authored_data(self):
        registry = copy.deepcopy(self.registry)
        registry["label_rules"]["OTHER"] = "An unsupported label"
        with self.assertRaises(ValueError):
            contrasts.build(registry)
        for field in ["model_predictions_used", "final_data_read"]:
            registry = copy.deepcopy(self.registry)
            registry[field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                contrasts.build(registry)

    def test_frozen_data_and_review_are_deterministic_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            registry_path = Path(directory) / "annotations.json"
            registry_path.write_bytes(contrasts.json_bytes(self.registry))
            manifest = contrasts.freeze(first, registry_path)
            self.assertEqual(manifest, contrasts.freeze(second, registry_path))
            for name, digest in manifest["files"].items():
                self.assertEqual(
                    (first / name).read_bytes(), (second / name).read_bytes()
                )
                self.assertEqual(
                    hashlib.sha256((first / name).read_bytes()).hexdigest(), digest
                )
            with self.assertRaises(FileExistsError):
                contrasts.freeze(first, registry_path)
            contract = json.loads((first / "contract.json").read_text())
            self.assertEqual(contract["label2id"], VELA_LABEL2ID)
            self.assertEqual(
                {int(key): label for key, label in contract["id2label"].items()},
                VELA_ID2LABEL,
            )
            review = json.loads((first / "review.json").read_text())
            self.assertEqual(len(review["rows"]), 300)
            self.assertFalse(manifest["model_predictions_used"])
            self.assertFalse(manifest["final_data_read"])


if __name__ == "__main__":
    unittest.main()
