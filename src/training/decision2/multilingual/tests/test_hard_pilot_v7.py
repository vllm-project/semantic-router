import unittest

from multilingual.hard_pilot_v7 import LANGUAGES, POLICY, render


class HardPilotV7Test(unittest.TestCase):
    def test_choice_native_options_and_inclusive_policy(self):
        case = {
            "id": "v7-fixture",
            "operation": "choice_latest",
            "option_order": ["A", "HOLD", "B", "C"],
            "facts": {
                "day": 12,
                "scope": "X",
                "records": [
                    {
                        "id": name,
                        "scope": "X",
                        "issued": 9,
                        "expires": 12,
                        "signed": True,
                        "withdrawn": False,
                    }
                    for name in "ABC"
                ],
            },
        }
        for language in LANGUAGES:
            with self.subTest(language=language):
                row = render(case, language)
                self.assertIn(POLICY["choice_latest"][language], row["state"])
                self.assertIn("12", row["state"])
                self.assertEqual(
                    list(row["questions"]["decision"]["criteria"]), list("abcd")
                )
                self.assertEqual(row["questions"]["decision"]["type"], "choice")
                self.assertNotIn("gold", row)

    def test_score_credit_is_additive_in_chinese(self):
        case = {
            "id": "v7-credit-fixture",
            "operation": "score_penalty",
            "facts": {"late": 1, "open": 1, "credit": 2},
        }
        row = render(case, "zh")
        self.assertIn("加分加上", row["state"])
        self.assertIn("+2", row["state"])
        self.assertEqual(row["questions"]["decision"]["type"], "score")
        self.assertEqual(len(row["questions"]["decision"]["criteria"]), 4)

    def test_temporal_policy_covers_revocation_day(self):
        case = {
            "id": "v7-revocation-fixture",
            "operation": "noul_revoke",
            "facts": {"day": 12, "grant": 11, "expires": 12, "revocations": [13]},
        }
        for language in LANGUAGES:
            with self.subTest(language=language):
                row = render(case, language)
                self.assertIn(POLICY["noul_revoke"][language], row["state"])
                self.assertIn("13", row["state"])
                self.assertEqual(row["questions"]["decision"]["type"], "noul")
                self.assertEqual(
                    list(row["questions"]["decision"]["criteria"]),
                    ["false", "true"],
                )


if __name__ == "__main__":
    unittest.main()
