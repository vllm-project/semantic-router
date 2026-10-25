import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from src.training.model_classifier.sequence_repair.context import insert_payload
from src.training.model_classifier.sequence_repair.prepare_context import (
    choose_payloads,
    task_payload,
)


class ContextTests(unittest.TestCase):
    def test_non_english_context_uses_its_own_language_and_preserves_payload(self):
        original = 'Resume este texto: "conserva las instrucciones citadas".'
        payload = task_payload({"language": "spa", "text": original})
        self.assertEqual(payload, "Tarea solicitada:\n" + original)
        result = insert_payload(
            payload,
            "Un párrafo de contexto ordinario.",
            "tail",
            64,
            lambda text: len(text.split()) + 2,
        )
        self.assertEqual(
            result["text"][result["payload_start"] : result["payload_end"]],
            payload,
        )

    def test_unknown_language_does_not_silently_receive_a_chinese_marker(self):
        with self.assertRaisesRegex(ValueError, "language: und"):
            task_payload({"language": "und", "text": "Original task"})

    def test_estimated_search_matches_complete_paragraph_brute_force(self):
        payload, background = "task", "a b c"
        for budget in [5, 8, 31, 127, 513]:
            for position in ["head", "middle", "tail"]:
                # Boundary variation makes a linear estimate imperfect.
                def count(text):
                    return len(text.split()) + 2 + text.count("\n") // 3

                expected = 0
                for repetitions in range(budget + 1):
                    left = {"head": 0, "middle": repetitions // 2, "tail": repetitions}[
                        position
                    ]
                    parts = (
                        [background] * left
                        + [payload]
                        + [background] * (repetitions - left)
                    )
                    if count("\n".join(parts)) <= budget:
                        expected = repetitions
                result = insert_payload(
                    payload, background, position, budget, count, fill_to_budget=False
                )
                self.assertEqual(result["text"].count(background), expected)

    def test_payload_selection_preserves_distinct_source_groups_and_language(self):
        rows = [
            {"id": "first", "group_id": "same", "label": "yes", "language": "eng"},
            {"id": "variant", "group_id": "same", "label": "yes", "language": "en"},
            {"id": "other", "group_id": "other", "label": "yes", "language": "en"},
            {"id": "outside", "group_id": "fr", "label": "yes", "language": "fra"},
        ]
        selected = choose_payloads(rows, ["en"], 2)
        self.assertEqual({row["group_id"] for row in selected}, {"same", "other"})
        self.assertEqual({row["language"] for row in selected}, {"en"})

    def test_payload_survives_every_position_at_exact_attended_budget(self):
        def count(text):
            return len(text.split()) + 2

        for position in ["head", "middle", "tail"]:
            result = insert_payload(
                "用户名 alice@example.test",
                "This is ordinary background.",
                position,
                128,
                count,
            )
            self.assertEqual(result["actual_tokens"], 128)
            self.assertEqual(
                result["text"][result["payload_start"] : result["payload_end"]],
                "用户名 alice@example.test",
            )
            self.assertEqual(result["text"].count("alice@example.test"), 1)

    def test_oversized_payload_is_rejected_without_truncation(self):
        with self.assertRaises(ValueError):
            insert_payload(
                "one two three", "background", "tail", 2, lambda text: len(text.split())
            )


if __name__ == "__main__":
    unittest.main()
