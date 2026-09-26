import unittest

from training.eikos.data import native_question


class TestNativeQuestion(unittest.TestCase):
    def test_boolean_native_order(self):
        row = {
            "id": "one",
            "task_type": "noul",
            "instructions": "?",
            "label": 0,
            "options": [
                {"key": "false", "description": "no"},
                {"key": "true", "description": "yes"},
            ],
        }
        question, gold = native_question(row)
        self.assertEqual(list(question["criteria"]), ["true", "false"])
        self.assertEqual(gold, "false")

    def test_choice_shuffle_is_stable(self):
        row = {
            "id": "two",
            "task_type": "choice",
            "instructions": "?",
            "label": 0,
            "options": [{"key": str(i), "description": str(i)} for i in range(8)],
        }
        first, gold = native_question(row, shuffle_seed=123)
        second, _ = native_question(row, shuffle_seed=123)
        self.assertEqual(first, second)
        self.assertEqual(gold, "0")
        self.assertEqual(set(first["criteria"]), set(str(i) for i in range(8)))


if __name__ == "__main__":
    unittest.main()
