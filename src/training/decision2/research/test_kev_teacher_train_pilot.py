from __future__ import annotations

import unittest

from research.kev_teacher_train_pilot import aggregate


class KevTeacherTrainPilotTest(unittest.TestCase):
    def test_native_teacher_screen_counts_overflow_and_ordinal_probabilities(
        self,
    ) -> None:
        rows = [
            {
                "task_type": "choice",
                "state": "a",
                "instructions": "pick",
                "options": [
                    {"key": "a", "description": "first"},
                    {"key": "b", "description": "second"},
                ],
                "label": 1,
            },
            {
                "task_type": "noul",
                "state": "b",
                "instructions": "yes?",
                "options": [
                    {"key": "false", "description": "no"},
                    {"key": "true", "description": "yes"},
                ],
                "label": 1,
            },
            {
                "task_type": "score",
                "state": "c",
                "instructions": "rate",
                "options": [
                    {"key": "0", "description": "low"},
                    {"key": "1", "description": "middle"},
                    {"key": "2", "description": "high"},
                ],
                "label": 1,
            },
        ]

        def decide(state: str, questions: object) -> dict[str, object]:
            assert questions
            if state == "a":
                return {
                    "answers": {
                        "decision": {
                            "type": "choice",
                            "probabilities": {"a": 0.2, "b": 0.8},
                        }
                    }
                }
            if state == "b":
                return {"status": "context_overflow", "answers": {"decision": {}}}
            return {
                "answers": {
                    "decision": {
                        "type": "score",
                        "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2},
                    }
                }
            }

        result = aggregate(rows, decide)
        self.assertEqual(result["choice"]["correct"], 1)
        self.assertEqual(result["noul"]["invalid"], 1)
        self.assertEqual(result["noul"]["valid"], 0)
        self.assertEqual(result["score"]["correct"], 1)
        self.assertAlmostEqual(result["score"]["gold_probability_sum"], 0.6)
