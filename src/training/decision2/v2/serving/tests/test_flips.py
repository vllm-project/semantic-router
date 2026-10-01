from __future__ import annotations

import unittest

from v2.serving.flips import margin, panel_flips


def noul(p):
    return {"type": "noul", "noul": p}


def choice(key, probs):
    return {"choice": key, "probabilities": probs}


class FlipsTest(unittest.TestCase):
    def test_margins(self) -> None:
        self.assertAlmostEqual(margin(noul(0.53)), 0.03)
        self.assertAlmostEqual(margin(choice("a", {"a": 0.6, "b": 0.3, "c": 0.1})), 0.3)
        self.assertIsNone(margin({"type": "noul", "error": "invalid_question"}))

    def test_changed_slots_and_shared_flips(self) -> None:
        stored = {
            "p1": {"q": noul(0.51), "r": choice("a", {"a": 0.52, "b": 0.48})},
            "p2": {"q": noul(0.9)},
        }
        one = {
            "p1": {"q": noul(0.49), "r": choice("b", {"a": 0.49, "b": 0.51})},
            "p2": {"q": noul(0.89)},
        }
        two = {
            "p1": {"q": noul(0.48), "r": choice("a", {"a": 0.53, "b": 0.47})},
            "p2": {"q": noul(0.9)},
        }
        result = panel_flips(stored, {"bf16": one, "fp32": two})
        self.assertEqual(result["slots"], 3)
        self.assertEqual(result["runs"]["bf16"]["changed"], 2)
        self.assertEqual(result["runs"]["fp32"]["changed"], 1)
        self.assertEqual(result["changed_in_every_run"], 1)
        self.assertAlmostEqual(result["runs"]["bf16"]["stored_margin_max"], 0.04)
        self.assertEqual(result["slots_at_or_below_max_changed_margin"], 2)


if __name__ == "__main__":
    unittest.main()
