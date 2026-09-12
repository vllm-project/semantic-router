import copy
import json
import unittest
from pathlib import Path

from src.training.model_classifier.safety_classifier.vela_hard_negatives import build


class HardNegativeTests(unittest.TestCase):
    def registry(self):
        return json.loads(
            (
                Path(__file__).parents[1] / "configs/hazard-hard-negatives-v1.json"
            ).read_text()
        )

    def test_fixed_families_keep_translations_together(self):
        result = build(self.registry())
        train = {row["group_id"] for row in result["train"]}
        dev = {row["group_id"] for row in result["validation"]}
        self.assertFalse(train & dev)
        self.assertEqual(len(result["train"]), 72)
        self.assertEqual(len(result["validation"]), 48)
        for rows in result.values():
            self.assertTrue(
                all(not any(row["targets"]) and all(row["label_mask"]) for row in rows)
            )
            self.assertEqual(len(rows), 2 * len({row["group_id"] for row in rows}))

    def test_duplicate_family_and_text_refused(self):
        registry = self.registry()
        for index in [2, 3]:
            changed = copy.deepcopy(registry)
            changed["families"][1][index] = changed["families"][0][index]
            with self.assertRaises(ValueError):
                build(changed)


if __name__ == "__main__":
    unittest.main()
