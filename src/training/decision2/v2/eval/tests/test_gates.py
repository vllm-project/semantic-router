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

    def test_mcnemar_exact(self):
        self.assertEqual(gates.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(gates.mcnemar_exact(3, 3), 1.0)
        self.assertAlmostEqual(gates.mcnemar_exact(2, 7), 92 / 512)
        self.assertAlmostEqual(gates.mcnemar_exact(7, 2), 92 / 512)
        self.assertLess(gates.mcnemar_exact(6, 27), 0.001)

    def test_public_guard_flags_only_significant_losses(self):
        targets = {
            f"p{i}": {
                "tier": "hard" if i < 40 else "easy",
                "family": "long" if i < 10 else "short",
                "task_type": "choice",
            }
            for i in range(60)
        }

        def outcomes(wrong: set[int]) -> dict:
            return {
                key: {"tier": row["tier"], "correct": int(key[1:]) not in wrong}
                for key, row in targets.items()
            }

        base = outcomes(set())
        big = gates.public_guard(outcomes(set(range(12))), base, targets, 200, 1)
        self.assertEqual(big["delta"], -12)
        self.assertEqual(big["discordant"], {"left_only": 0, "right_only": 12})
        self.assertEqual(big["verdict"], "REGRESSION")
        self.assertEqual(big["tiers"]["hard"], {"items": 40, "left": 28, "right": 40})
        self.assertEqual(big["families"]["long"]["left"], 0)
        small = gates.public_guard(outcomes({0, 1}), base, targets, 200, 1)
        self.assertEqual(small["verdict"], "OK")
        gain = gates.public_guard(base, outcomes(set(range(12))), targets, 200, 1)
        self.assertEqual(gain["verdict"], "OK")
        with self.assertRaises(ValueError):
            gates.public_guard(base, {"p0": base["p0"]}, targets, 200, 1)


if __name__ == "__main__":
    unittest.main()
