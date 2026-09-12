import unittest

from src.training.model_classifier.user_feedback_classifier.data_contract import (
    fingerprint,
)
from src.training.model_classifier.user_feedback_classifier.vela_applicability import (
    records,
    summarize,
)
from src.training.model_classifier.user_feedback_classifier.vela_data import (
    project_label,
    split_groups,
    wild_records,
)
from src.training.model_classifier.user_feedback_classifier.vela_feedbackqa import (
    project,
)
from src.training.model_classifier.vela_partition import partition


class VelaFeedbackDataTests(unittest.TestCase):
    def test_external_sat_projection_excludes_qualified_and_knowledge_text(self):
        rows = [
            {
                "question": "One question",
                "feedback": [
                    "This answer is clear and useful to me.",
                    "The answer is clear but missing some details.",
                    "Libraries lend books to their registered members.",
                    "This response is incorrect and unhelpful.",
                ],
                "rating": ["Excellent", "Excellent", "Excellent", "Bad"],
            }
        ]
        result, _ = project(rows, set())
        self.assertEqual([row["text"] for row in result], [rows[0]["feedback"][0]])
        self.assertEqual(result[0]["label"], "SAT")
        self.assertEqual(project(rows, {fingerprint(result[0]["text"])})[0], [])

    def test_non_feedback_queries_have_no_forced_satisfaction_gold(self):
        examples = records()
        self.assertEqual(len(examples), 40)
        self.assertEqual(len({row["group_id"] for row in examples}), 20)
        self.assertTrue(all(row["feedback_label"] is None for row in examples))
        result = summarize(
            [
                {**examples[0], "prediction": "WRONG_ANSWER", "confidence": 0.99},
                {**examples[1], "prediction": "SAT", "confidence": 0.99},
            ]
        )
        self.assertIsNone(result["accuracy"])
        self.assertEqual(
            result["thresholds"]["0.95"]["high_confidence_non_sat_fraction"], 0.5
        )

    def test_budget_partition_preserves_text_and_boundary(self):
        rows = [
            {"id": str(i), "group_id": str(i), "text": "x" * i} for i in [511, 512, 513]
        ]
        within, over = partition(rows, len, 512)
        self.assertEqual([row["actual_tokens"] for row in within], [511, 512])
        self.assertEqual(over[0]["text"], rows[-1]["text"])

    def test_mixed_satisfaction_is_not_arbitrarily_forced(self):
        self.assertIsNone(
            project_label("Thanks but that is wrong", "FEEDBACK", True, True)
        )

    def test_assistant_and_empty_text_are_excluded(self):
        rows = [
            {"UtterranceId": 0, "Role": "User", "Content": None},
            {"UtterranceId": 1, "Role": "Agent", "Content": "Thanks"},
            {
                "UtterranceId": 2,
                "Role": "User",
                "Content": "Thanks, that works",
                "State": "FEEDBACK",
                "Satisfaction": True,
                # The external dataset uses this exact column spelling.
                "Disatisfaction": False,  # codespell:ignore disatisfaction
            },
        ]
        self.assertEqual(len(list(wild_records(rows))), 1)

    def test_flags_do_not_turn_knowledge_text_into_satisfaction(self):
        self.assertIsNone(
            project_label("Paris is the capital of France", "FEEDBACK", True, False)
        )

    def test_conversation_stays_in_one_partition(self):
        rows = [
            {"text": f"Example {i}", "label": "SAT", "group_id": "same-conversation"}
            for i in range(20)
        ]
        result, _ = split_groups(rows)
        self.assertEqual(sum(bool(values) for values in result.values()), 1)

    def test_conflicting_normalized_text_is_excluded(self):
        rows = [
            {"text": "Great", "label": "SAT", "group_id": "a"},
            {"text": " great ", "label": "WRONG_ANSWER", "group_id": "b"},
        ]
        result, conflicts = split_groups(rows)
        self.assertEqual(conflicts, 1)
        self.assertFalse(any(result.values()))


if __name__ == "__main__":
    unittest.main()
