import unittest

from multilingual.audit import evidence, script_of, visible_text


class AuditTest(unittest.TestCase):
    def test_visible_training_excludes_label_metadata(self):
        row = {
            "instructions": "Choose the reply",
            "state": {"message": "Bonjour"},
            "options": [{"description": "Oui"}],
            "label": "SECRET GOLD",
            "audit_metadata": {"original": "SECRET SOURCE"},
        }
        full, state = visible_text(row, "training")
        self.assertIn("Bonjour", full)
        self.assertEqual(state, "Bonjour")
        self.assertNotIn("SECRET", full)

    def test_script_mix(self):
        value = evidence("English text 你好你好你好", "")
        self.assertTrue(value["significant_nonlatin"])
        self.assertEqual(value["scripts"]["han"], 6)
        self.assertEqual(script_of("é"), "latin")
        self.assertEqual(script_of("あ"), "kana")
        self.assertEqual(script_of("م"), "arabic")


if __name__ == "__main__":
    unittest.main()
