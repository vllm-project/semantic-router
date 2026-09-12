import copy
import unittest

from src.training.model_classifier.safety_classifier.vela_hazard_supervision import (
    select_training_rows,
)


class HazardSupervisionTests(unittest.TestCase):
    def row(self, label="unsafe", tag="jailbreaking", split="train"):
        return {
            "id": "id",
            "group_id": "group",
            "source": "cultureguard",
            "source_split": split,
            "source_tag": tag,
            "label": label,
            "targets": [1, 0],
            "label_mask": [1, 0],
        }

    def test_exclusion_does_not_relabel_or_change_input(self):
        rows = [self.row(), self.row(tag="generic"), self.row(label="safe")]
        original = copy.deepcopy(rows)
        kept, excluded = select_training_rows(rows)
        self.assertEqual(kept, rows[1:])
        self.assertEqual(len(excluded), 1)
        self.assertEqual(rows, original)

    def test_test_and_development_are_refused(self):
        for split in ["validation", "test", None]:
            with self.subTest(split=split), self.assertRaises(ValueError):
                select_training_rows([self.row(split=split)])

    def test_unknown_provenance_is_refused(self):
        with self.assertRaises(ValueError):
            select_training_rows([self.row(tag="new_unreviewed_generator")])


if __name__ == "__main__":
    unittest.main()
