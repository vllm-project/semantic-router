# Preserve Chinese input punctuation.
# ruff: noqa: RUF001

import hashlib
import unittest

from src.training.model_classifier.user_feedback_classifier.vela_data import (
    TRAIN_PERCENT,
)
from src.training.model_classifier.user_feedback_classifier.vela_weak_supervision_v2 import (
    exclusion_reason,
    training_rows,
    visible_user_prose,
)


class WeakFeedbackV2Tests(unittest.TestCase):
    def test_structured_quotes_and_markdown_are_not_feedback_cues(self):
        for text in [
            "请分析“你的答案错误”这句话。",
            "请分析‘你的答案错误’这句话。",
            "请分析「你的答案错误」这句话。",
            "请分析『你的答案错误』这句话。",
            "Analyze this passage:\n> Your answer is incorrect.",
        ]:
            row = {"text": text, "label": "WRONG_ANSWER"}
            with self.subTest(text=text):
                self.assertIsNone(exclusion_reason(row, {"State": "FEEDBACK"}))
                self.assertIsNotNone(
                    exclusion_reason(row, {"State": "FEEDBACK"}, "structured-v3")
                )
                self.assertEqual(row["text"], text)

    def test_structured_quote_policy_keeps_real_prose_outside_the_quote(self):
        self.assertEqual(
            visible_user_prose(
                "Your answer is incorrect. The quoted text is “not correct”.",
                "structured-v3",
            ),
            "Your answer is incorrect. The quoted text is  .",
        )
        self.assertIsNone(
            exclusion_reason(
                {
                    "text": "Your answer is incorrect. The quoted text is “not correct”.",
                    "label": "WRONG_ANSWER",
                },
                {"State": "FEEDBACK"},
                "structured-v3",
            )
        )

    def test_review_exception_requires_the_exact_source_text(self):
        text = "The failure persists: “Your answer is incorrect”."
        group = next(
            f"wildfeedback:conversation:{index}"
            for index in range(100)
            if int(
                hashlib.sha256(
                    f"wildfeedback:conversation:{index}".encode()
                ).hexdigest()[:8],
                16,
            )
            % 100
            < TRAIN_PERCENT
        )
        row = {
            "id": "wildfeedback:0",
            "group_id": group,
            "source": "wildfeedback_weak_projection",
            "text": text,
            "label": "WRONG_ANSWER",
        }
        annotation = {"Role": "User", "Content": text, "State": "FEEDBACK"}
        review = {
            row["id"]: {
                "label": row["label"],
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "action": "retain",
            }
        }
        kept, excluded = training_rows([row], [annotation], "structured-v3", review)
        self.assertEqual(len(kept), 1)
        self.assertEqual(excluded, [])
        review[row["id"]]["text_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            training_rows([row], [annotation], "structured-v3", review)

    def reason(self, text, label, state="FEEDBACK"):
        return exclusion_reason({"text": text, "label": label}, {"State": state})

    def test_words_inside_code_and_requested_content_are_not_feedback(self):
        self.assertIsNotNone(
            self.reason("def transform(data, vertical=False): pass", "WRONG_ANSWER")
        )
        self.assertIsNotNone(
            self.reason('Edit this quote: "Something went wrong".', "WRONG_ANSWER")
        )
        self.assertIsNotNone(
            self.reason(
                "Describe a planet that uses carbon instead of silicon.",
                "WANT_DIFFERENT",
                "CONTINUATION",
            )
        )

    def test_actual_feedback_is_retained(self):
        self.assertIsNone(self.reason("Your calculation is incorrect.", "WRONG_ANSWER"))
        self.assertIsNone(
            self.reason(
                "I don't understand why this happens.",
                "NEED_CLARIFICATION",
                "REFINEMENT",
            )
        )
        self.assertIsNone(
            self.reason(
                "Please rewrite it in bullet points.", "WANT_DIFFERENT", "REFINEMENT"
            )
        )
        self.assertIsNone(
            self.reason("现在这个结果正确了，谢谢。", "SAT", "POSITIVE_Closure")
        )

    def test_mixed_intents_and_new_topics_are_excluded(self):
        self.assertIsNotNone(
            self.reason(
                "Please elaborate as a poem.", "NEED_CLARIFICATION", "REFINEMENT"
            )
        )
        self.assertIsNotNone(
            self.reason(
                "Thanks! Can you write the next chapter?", "SAT", "CONTINUATION"
            )
        )
        self.assertIsNotNone(
            self.reason("Great. Tell me about something else.", "SAT", "NEWTOPIC")
        )

    def test_arbitrary_partition_cannot_be_filtered_as_training(self):
        with self.assertRaises(ValueError):
            training_rows(
                [
                    {
                        "id": "wildfeedback:0",
                        "source": "wildfeedback_weak_projection",
                        "group_id": "test:group",
                    }
                ],
                [],
            )


if __name__ == "__main__":
    unittest.main()
