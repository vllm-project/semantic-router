from __future__ import annotations

import unittest

from v2.eval import gates


def item(i: int, kind: str, value) -> dict:
    if kind == "score":
        question = {"type": "score", "instructions": "?", "criteria": ["a", "b", "c"]}
        truth = {"type": "score", "value": value, "semantic_value": value}
    else:
        question = {
            "type": "noul",
            "instructions": "?",
            "criteria": {"true": "t", "false": "f"},
        }
        truth = {"type": "noul", "value": value, "semantic_value": value}
    return {
        "id": f"i{i}",
        "questions": {"decision": question},
        "gold": {"decision": truth},
    }


class GatesTest(unittest.TestCase):
    def test_constant_score_is_collapsed_and_good_noul_is_ok(self):
        gold = [item(i, "score", i % 3) for i in range(60)] + [
            item(100 + i, "noul", bool(i % 2)) for i in range(60)
        ]
        preds = {}
        for row in gold:
            q = row["questions"]["decision"]
            if q["type"] == "score":
                answer = {"type": "score", "score": 0}
            else:
                answer = {
                    "type": "noul",
                    "noul": 0.9 if row["gold"]["decision"]["value"] else 0.1,
                }
            preds[row["id"]] = {"answers": {"decision": answer}}
        out = gates.type_summary(gold, preds)
        self.assertTrue(out["score"]["verdict"].startswith("COLLAPSED"))
        self.assertEqual(
            out["score"]["recall_by_level"], {"0": 1.0, "1": 0.0, "2": 0.0}
        )
        self.assertEqual(out["noul"]["verdict"], "OK")
        self.assertAlmostEqual(out["noul"]["accuracy"], 1.0)

    def test_wilson(self):
        low, high = gates.wilson(50, 100)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)


if __name__ == "__main__":
    unittest.main()
