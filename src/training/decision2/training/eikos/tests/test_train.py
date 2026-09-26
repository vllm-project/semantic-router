import unittest

from training.eikos.data import one_pass_quarantine


class TestQuarantine(unittest.TestCase):
    def test_more_than_native_one_pass_is_explicit(self):
        short = {"id": "short", "options": [{}] * 100}
        long = {
            "id": "long",
            "family": "example",
            "task_type": "choice",
            "options": [{}] * 101,
            "source": "upstream:train",
            "audit_metadata": {"original_source": {"original_source": "license-tag"}},
            "input_sha256": "abc",
        }
        effective, quarantined = one_pass_quarantine([short, long])
        self.assertEqual(effective, [short])
        self.assertEqual(quarantined[0]["id"], "long")
        self.assertEqual(quarantined[0]["rights_source"], "license-tag")
        self.assertEqual(quarantined[0]["option_count"], 101)


if __name__ == "__main__":
    unittest.main()
