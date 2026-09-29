import sys
import unittest
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from lux9b import m4_rules  # noqa: E402


def arm(choice, noul, score, fams, h3, h=0.55):
    by_family = {f"f{i}": {"correct": c, "n": 400} for i, c in enumerate(fams)}
    t = sum(c / 400 for c in fams) / len(fams)
    return {
        "by_type": {
            "choice": {"correct": choice, "n": 800},
            "noul": {"correct": noul, "n": 400},
            "score": {"correct": score, "n": 400},
        },
        "by_family": by_family,
        "T": t,
        "H_mean": h3,
        "H": h,
        "proxy": 100 * (t * h) ** 0.5,
    }


LUX = arm(799, 272, 331, [400, 272, 331, 399], 0.528)


class AlphaRuleTest(unittest.TestCase):
    def line(self, points, arms):
        return m4_rules.alpha_line(points, {"lux": LUX, **arms}, "lux", "H_mean")

    def test_family_floor_is_exact_at_the_boundary(self):
        on_floor = arm(799, 300, 340, [400, 300, 340, 359], 0.53)
        below = arm(799, 300, 340, [400, 300, 340, 358], 0.53)
        out = self.line(
            [(Fraction(1), "a"), (Fraction(1, 2), "b")], {"a": on_floor, "b": below}
        )
        rows = {r["arm"]: r for r in out["rows"]}
        self.assertTrue(rows["a"]["eligible"])
        self.assertFalse(rows["b"]["eligible"])

    def test_type_and_h_floors(self):
        low_choice = arm(774, 300, 340, [400, 300, 340, 399], 0.53)
        ok_choice = arm(775, 300, 340, [400, 300, 340, 399], 0.53)
        low_h = arm(799, 300, 340, [400, 300, 340, 399], 0.527)
        out = self.line(
            [(Fraction(1, 4), "a"), (Fraction(1, 3), "b"), (Fraction(1, 2), "c")],
            {"a": low_choice, "b": ok_choice, "c": low_h},
        )
        self.assertEqual([r["eligible"] for r in out["rows"]], [False, True, False])

    def test_pick_is_smallest_eligible_alpha_keeping_three_quarters(self):
        small = arm(799, 280, 335, [400, 280, 335, 399], 0.53)  # G = .0075
        mid = arm(799, 300, 345, [400, 300, 345, 399], 0.53)  # G = .02625
        big = arm(799, 305, 348, [400, 305, 348, 399], 0.53)  # G = .03125
        out = self.line(
            [(Fraction(1, 4), "s"), (Fraction(1, 2), "m"), (Fraction(1), "b")],
            {"s": small, "m": mid, "b": big},
        )
        self.assertEqual(out["pick"]["alpha"], "1/2")
        self.assertAlmostEqual(out["G_star"], 0.03125)
        bigger = arm(
            799, 310, 350, [400, 310, 350, 399], 0.53
        )  # G = .035625 > mid / .75
        out = self.line(
            [(Fraction(1, 2), "m"), (Fraction(1), "b")], {"m": mid, "b": bigger}
        )
        self.assertEqual(out["pick"]["alpha"], "1")

    def test_no_pick_when_gain_below_one_point(self):
        tiny = arm(799, 280, 331, [400, 280, 331, 399], 0.53)  # G = .005
        out = self.line([(Fraction(1, 2), "t")], {"t": tiny})
        self.assertIsNone(out["pick"])
        self.assertEqual(out["no_pick_reason"], "G* < 0.01")

    def test_proxy_drop(self):
        out = m4_rules.proxy_drop(
            {
                "K": {"proxy": 76.0},
                "U": {"proxy": 68.0},
                "KN": {"proxy": 68.1},
                "P": None,
            }
        )
        self.assertEqual(out["dropped"], ["U"])
        self.assertEqual(sorted(out["kept"]), ["K", "KN"])


class SeedRuleTest(unittest.TestCase):
    @staticmethod
    def arms(soup, seeds):
        out = {"soup": {"proxy": soup}}
        out.update({f"s{i}": {"proxy": p} for i, p in enumerate(seeds, 1)})
        return out

    def test_soup_at_or_above_seed_mean(self):
        out = m4_rules.seed_rule(
            self.arms(70.0, [69.0, 71.0]), "soup", ["s1", "s2"], "s1"
        )
        self.assertEqual(out["artifact"], "soup")

    def test_three_seeds_fall_back_to_median(self):
        out = m4_rules.seed_rule(
            self.arms(60.0, [70.0, 65.0, 72.0]), "soup", ["s1", "s2", "s3"], "s1"
        )
        self.assertEqual(out["artifact"], "s1")
        out = m4_rules.seed_rule(
            self.arms(60.0, [65.0, 70.0, 72.0]), "soup", ["s1", "s2", "s3"], "s1"
        )
        self.assertEqual(out["artifact"], "s2")

    def test_two_seeds_fall_back_to_primary(self):
        out = m4_rules.seed_rule(
            self.arms(60.0, [65.0, 70.0]), "soup", ["s1", "s2"], "s1"
        )
        self.assertEqual(out["artifact"], "s1")


if __name__ == "__main__":
    unittest.main()
