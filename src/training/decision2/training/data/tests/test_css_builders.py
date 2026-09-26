import hashlib
import unittest

from transfer.build import normalized_context_sha256

from training.data import build_css_flute as flute
from training.data import build_css_pilot_holdouts as holdouts
from training.data import build_pilot as pilot
from training.model.data import validate_row


class ForbiddenLabels(dict):
    def __getitem__(self, key):
        if key == "test":
            raise AssertionError("Test label was accessed")
        return super().__getitem__(key)


class CSSBuilderTests(unittest.TestCase):
    def test_flute_filters_test_id_and_text_before_label_access(self):
        prompt = "Identify the figure of speech.\nA: Idiom\nB: Metaphor"
        source = {
            "context": {
                "test": "Held-out text",
                "alias": "Held-out text",
                "train": "An original idiom.",
            },
            "labels": ForbiddenLabels(
                {"test": "Idiom", "alias": "Idiom", "train": "Idiom"}
            ),
            "prompts": {"test": prompt, "alias": prompt, "train": prompt},
        }
        raw = hashlib.sha256(b"Held-out text").hexdigest()
        excluded = {
            "task_source_ids": {("flute", "test")},
            "raw_context_sha256": {raw},
            "normalized_context_sha256": {normalized_context_sha256("Held-out text")},
            "panel_input_sha256": set(),
        }
        mappings = ({"flute": {"A": "Idiom", "B": "Metaphor"}}, {})
        rows, receipt = flute.make_candidate_rows(source, mappings, excluded)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "css_train/flute/train")
        self.assertEqual(
            receipt["excluded_counts"],
            {
                "panel_raw_text_sha256": 1,
                "panel_source_id": 1,
                "same_context_duplicate_rows": 0,
            },
        )
        validate_row(rows[0], "train")

    def test_class_stratified_sample_is_nested_and_deterministic(self):
        rows = [pilot.make_composition("css", i) for i in range(8)]
        for index, row in enumerate(rows):
            row["options"][row["label"]]["key"] = "A" if index < 4 else "B"
        first, receipt = flute.sample_class_stratified(rows, 4, "seed")
        second, _ = flute.sample_class_stratified(rows, 4, "seed")
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(receipt["class_quotas"], {"A": 2, "B": 2})

    def test_independent_noul_and_score_oracles(self):
        for task_type in ("noul", "score"):
            rows = [
                holdouts.synthetic_type_row("fixed", task_type, i) for i in range(20)
            ]
            self.assertEqual({row["task_type"] for row in rows}, {task_type})
            self.assertEqual(len({row["group_id"] for row in rows}), 20)
            for row in rows:
                pilot.validate_train_row(row)
                self.assertEqual(
                    row["options"][row["label"]]["key"],
                    str(row["audit_metadata"]["oracle"]).lower(),
                )
            if task_type == "score":
                self.assertEqual(
                    {row["options"][row["label"]]["key"] for row in rows},
                    {"0", "1", "2", "3"},
                )


if __name__ == "__main__":
    unittest.main()
