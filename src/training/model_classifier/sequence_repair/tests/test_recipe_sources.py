import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from src.training.model_classifier.domain_classifier.prepare_data import (
    canonical_groups,
    question_with_choices,
)
from src.training.model_classifier.modality_routing_classifier.extend_contracts import (
    recover_caption,
)
from src.training.model_classifier.modality_routing_classifier.prepare_vela_data import (
    TEMPLATES,
)
from src.training.model_classifier.sequence_repair.data import normalized_text


class SourceContractTests(unittest.TestCase):
    def test_translated_duplicate_groups_are_unioned_transitively(self):
        rows = [
            {"text": "First question", "group_id": "a", "label": "biology"},
            {"text": "FIRST  QUESTION", "group_id": "b", "label": "biology"},
            {"text": "Otra pregunta", "group_id": "b", "label": "biology"},
            {"text": "otra pregunta", "group_id": "c", "label": "biology"},
        ]
        retained, excluded = canonical_groups(rows)
        self.assertEqual(excluded, 0)
        self.assertEqual({row["group_id"] for row in retained}, {"a"})
        retained, excluded = canonical_groups(
            [*rows, {"text": "Otra pregunta", "group_id": "d", "label": "history"}]
        )
        self.assertEqual((retained, excluded), ([], 5))

    def test_domain_question_contains_choices_but_not_answer_key(self):
        row = {
            "question": "Which element?",
            "option_a": "iron",
            "option_b": "wood",
            "option_c": "paper",
            "option_d": "glass",
            "answer": "secret-answer-key",
        }
        text = question_with_choices(row)
        self.assertIn("A. iron", text)
        self.assertNotIn(row["answer"], text)

    def test_caption_recovery_keeps_embedded_colons_and_rejects_changed_source(self):
        caption = "A quiet stage: blue curtains, paper lanterns, soft light"
        row = {
            "language": "en",
            "template_family": "modality-train-1",
            "text": TEMPLATES["en"]["AR"][1].format(caption),
            "caption_sha256": hashlib.sha256(
                normalized_text(caption).encode()
            ).hexdigest(),
        }
        self.assertEqual(recover_caption(row), caption)
        with self.assertRaises(ValueError):
            recover_caption({**row, "text": row["text"] + " changed"})


if __name__ == "__main__":
    unittest.main()
