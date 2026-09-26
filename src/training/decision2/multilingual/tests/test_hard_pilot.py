import json
import tempfile
import unittest
from pathlib import Path

from multilingual.hard_pilot import (
    LANGUAGES,
    OPS,
    load_cases,
    oracle,
    render,
    screen_overlap,
)


class HardPilotTest(unittest.TestCase):
    def test_oracle_choice_variants(self):
        latest = {
            "operation": "choice_latest",
            "facts": {
                "day": 12,
                "scope": "X",
                "records": [
                    {
                        "id": "A",
                        "scope": "X",
                        "issued": 8,
                        "expires": 14,
                        "signed": True,
                        "withdrawn": False,
                    },
                    {
                        "id": "B",
                        "scope": "X",
                        "issued": 10,
                        "expires": 14,
                        "signed": False,
                        "withdrawn": False,
                    },
                    {
                        "id": "C",
                        "scope": "Y",
                        "issued": 11,
                        "expires": 14,
                        "signed": True,
                        "withdrawn": False,
                    },
                ],
            },
        }
        self.assertEqual(oracle(latest), "A")
        minimax = {
            "operation": "choice_minimax",
            "facts": {
                "budget": 8,
                "records": [
                    {"id": "A", "spend": 7, "red": 5, "blue": 1},
                    {"id": "B", "spend": 9, "red": 1, "blue": 1},
                    {"id": "C", "spend": 8, "red": 4, "blue": 4},
                ],
            },
        }
        self.assertEqual(oracle(minimax), "C")
        proof = {
            "operation": "choice_two_proofs",
            "facts": {
                "records": [
                    {
                        "id": "A",
                        "version": 2,
                        "owner": True,
                        "safety": True,
                        "withdrawn": False,
                    },
                    {
                        "id": "B",
                        "version": 4,
                        "owner": True,
                        "safety": False,
                        "withdrawn": False,
                    },
                    {
                        "id": "C",
                        "version": 5,
                        "owner": True,
                        "safety": True,
                        "withdrawn": True,
                    },
                ]
            },
        }
        self.assertEqual(oracle(proof), "A")

    def test_oracle_noul_and_score(self):
        self.assertFalse(
            oracle(
                {
                    "operation": "noul_all",
                    "facts": {
                        "people": [
                            {
                                "id": "P",
                                "cleared": True,
                                "exempt": False,
                                "suspended": True,
                            }
                        ]
                    },
                }
            )
        )
        self.assertFalse(
            oracle(
                {
                    "operation": "noul_chain",
                    "facts": {
                        "day": 12,
                        "links": [
                            {
                                "from": "A",
                                "to": "B",
                                "approved": True,
                                "withdrawn": False,
                                "expires": 13,
                            },
                            {
                                "from": "C",
                                "to": "B",
                                "approved": True,
                                "withdrawn": False,
                                "expires": 13,
                            },
                        ],
                    },
                }
            )
        )
        self.assertTrue(
            oracle(
                {
                    "operation": "noul_revoke",
                    "facts": {
                        "day": 12,
                        "grant": 10,
                        "expires": 13,
                        "revocations": [14],
                    },
                }
            )
        )
        self.assertEqual(
            oracle(
                {
                    "operation": "score_weighted",
                    "facts": {"marks": [1, 0, 1], "weights": [3, 2, 1]},
                }
            ),
            2,
        )
        self.assertEqual(
            oracle(
                {
                    "operation": "score_penalty",
                    "facts": {"late": 2, "open": 1, "credit": 0},
                }
            ),
            0,
        )
        self.assertEqual(
            oracle(
                {
                    "operation": "score_order",
                    "facts": {"stages": [True, False, True]},
                }
            ),
            1,
        )

    def test_native_shape_and_locale_scripts(self):
        fixture = {
            "id": "demo",
            "operation": "choice_latest",
            "option_order": ["HOLD", "C", "B", "A"],
            "facts": {
                "day": 12,
                "scope": "X",
                "records": [
                    {
                        "id": name,
                        "scope": "X",
                        "issued": 8 + i,
                        "expires": 14,
                        "signed": True,
                        "withdrawn": False,
                    }
                    for i, name in enumerate(("A", "B", "C"))
                ],
            },
        }
        for language in LANGUAGES:
            row = render(fixture, language)
            question = row["questions"]["decision"]
            self.assertEqual(question["type"], "choice")
            self.assertEqual(list(question["criteria"]), list("abcd"))
            self.assertEqual(len(row["state"]), len(row["state"].strip()))
            self.assertNotIn("gold", row)
            self.assertNotIn("facts", row)

    def test_private_casebook_rejects_gold_and_final(self):
        cases = []
        for operation in OPS:
            for suffix in ("a", "b"):
                cases.append(
                    {
                        "id": f"{operation}-{suffix}",
                        "operation": operation,
                        "facts": (
                            {"marks": [1], "weights": [1]}
                            if operation == "score_weighted"
                            else {}
                        ),
                    }
                )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            path.write_text(json.dumps(cases), encoding="utf-8")
            with self.assertRaises((KeyError, ValueError)):
                load_cases(path)
            cases[0]["gold"] = "A"
            path.write_text(json.dumps(cases), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(path)
            with self.assertRaises(ValueError):
                load_cases(Path(tmp) / "sealed-final.json")

    def test_overlap_detects_same_state(self):
        prompts = [{"id": "x", "state": "A special state for overlap."}]
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / "reference.jsonl"
            reference.write_text(
                json.dumps({"id": "r", "state": "A SPECIAL state for overlap."}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(screen_overlap(prompts, [reference])["exact_matches"], 1)


if __name__ == "__main__":
    unittest.main()
