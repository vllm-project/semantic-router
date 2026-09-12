import copy
import unittest

from src.training.model_classifier.safety_classifier.train_vela_hazard import (
    development_selection_score,
)
from src.training.model_classifier.safety_classifier.vela_data import LABELS
from src.training.model_classifier.safety_classifier.vela_hazard_partial import (
    POSITIVE_SOURCE,
    SAFE_SOURCE,
    original_training_annotations,
    project_training_rows,
)


class PartialHazardTests(unittest.TestCase):
    def test_low_support_does_not_select_checkpoint_when_explicitly_excluded(self):
        metrics = {
            "macro_ap": 0.7,
            "per_label": {
                "supported": {
                    "ap": 0.4,
                    "positive_support": 12,
                    "negative_support": 20,
                },
                "rare": {"ap": 1.0, "positive_support": 2, "negative_support": 30},
            },
        }
        self.assertEqual(development_selection_score(metrics, "macro-ap"), 0.7)
        self.assertEqual(development_selection_score(metrics, "macro-ap", 10), 0.4)
        with self.assertRaises(ValueError):
            development_selection_score(metrics, "macro-ap", 50)

    def row(self, label="unsafe", tag="generic"):
        return {
            "id": "row",
            "group_id": "aegis:group",
            "source": "cultureguard",
            "source_split": "train",
            "source_tag": tag,
            "label": label,
            "targets": [
                int(index in {0, 1} and label == "unsafe")
                for index in range(len(LABELS))
            ],
            "label_mask": [1] * len(LABELS),
        }

    def test_missing_category_is_unknown_and_input_is_unchanged(self):
        row = self.row()
        before = copy.deepcopy(row)
        selected, excluded = project_training_rows([row], {"aegis:group": {"violence"}})
        self.assertFalse(excluded)
        self.assertEqual(selected[0]["targets"], [1] + [0] * 11)
        self.assertEqual(selected[0]["label_mask"], [1] + [0] * 11)
        self.assertEqual(selected[0]["source"], POSITIVE_SOURCE)
        self.assertEqual(selected[0]["raw_targets"], before["targets"])
        self.assertEqual(row, before)

    def test_safe_all_negative_but_generated_unsafe_quarantined(self):
        for tag in ["generic", "adapted"]:
            kept, _ = project_training_rows([self.row("safe", tag)], {})
            self.assertEqual(kept[0]["source"], SAFE_SOURCE)
            self.assertEqual(kept[0]["label_mask"], [1] * 12)
        for tag in ["adapted", "jailbreaking"]:
            kept, excluded = project_training_rows(
                [self.row(tag=tag)], {"aegis:group": {"violence"}}
            )
            self.assertFalse(kept)
            self.assertEqual(len(excluded), 1)
        kept, excluded = project_training_rows([self.row("safe", "jailbreaking")], {})
        self.assertFalse(kept)
        self.assertEqual(len(excluded), 1)

    def test_no_original_or_no_agreement_is_excluded(self):
        for original in [{}, {"aegis:group": {"weapons"}}]:
            kept, excluded = project_training_rows([self.row()], original)
            self.assertFalse(kept)
            self.assertEqual(len(excluded), 1)

    def test_validation_is_never_reprojected(self):
        for split in ["validation", "test", None]:
            row = self.row()
            row["source_split"] = split
            with self.assertRaises(ValueError):
                project_training_rows([row], {})

    def test_original_response_variants_need_consensus(self):
        row = {
            "prompt": "source example",
            "prompt_label": "unsafe",
            "response": "I cannot help",
            "response_label": "safe",
            "violated_categories": "Violence, Illegal Activity",
        }
        other = {**row, "violated_categories": "Violence"}
        mapping = original_training_annotations([row, other])
        self.assertEqual(list(mapping.values()), [{"violence"}])


if __name__ == "__main__":
    unittest.main()
